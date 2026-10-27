//! Explicit, versioned interchange for one scalar group-estimate cohort.
//!
//! This deliberately serializes only the group-estimate profile.  It does not
//! serialize general `FmriFrame` annotations, arbitrary assays, or sources.

use crate::{
    nonblank, Admission, DegreesOfFreedom, Error, Estimand, FitReceipt, IndependentCohort, Result,
    Subject, VarianceOrigin,
};
use fmri_data::{
    Assay, AssayAnnotations, Axis, ExecutionContext, FeatureAxis, FeatureId, FeatureSpace,
    FmriFrame, IndexSpace, MemoryBudget, Metadata, ObservationId, ReadSelection, Selection,
    SourceSpec, Values, VolumeSpace,
};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const FORMAT: &str = "fmri-group-estimate";
const VERSION: u32 = 1;
const MANIFEST: &str = "manifest.json";
const EFFECTS: &str = "effects.f64";
const VARIANCES: &str = "variances.f64";
const DEFAULT_MANIFEST_BYTES: usize = 1 << 20;
const DEFAULT_BLOCK_BYTES: usize = 64 << 20;

#[derive(Clone, Copy, Debug)]
pub struct PortableWriteOptions {
    pub max_block_bytes: usize,
}
impl Default for PortableWriteOptions {
    fn default() -> Self {
        Self {
            max_block_bytes: DEFAULT_BLOCK_BYTES,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct PortableReadOptions {
    pub max_manifest_bytes: usize,
    pub max_block_bytes: usize,
}
impl Default for PortableReadOptions {
    fn default() -> Self {
        Self {
            max_manifest_bytes: DEFAULT_MANIFEST_BYTES,
            max_block_bytes: DEFAULT_BLOCK_BYTES,
        }
    }
}

/// Typed metadata needed to create a streaming bundle without retaining arrays.
#[derive(Clone, Debug)]
pub struct PortableMetadata {
    pub subjects: Vec<Subject>,
    pub estimand: Estimand,
    pub features: FeatureAxis,
    pub variance_origins: Option<Vec<VarianceOrigin>>,
    pub fit_receipts: Vec<FitReceipt>,
    pub admission: Admission,
}
impl PortableMetadata {
    pub fn from_cohort(cohort: &IndependentCohort) -> Self {
        Self {
            subjects: cohort.subjects().to_vec(),
            estimand: cohort.estimand().clone(),
            features: cohort.frame().features().clone(),
            variance_origins: cohort
                .variance_assay()
                .map(|_| cohort.variance_origins().to_vec()),
            fit_receipts: cohort.fit_receipts().to_vec(),
            admission: cohort.admission().clone(),
        }
    }
    fn shape(&self) -> [usize; 2] {
        [self.subjects.len(), self.features.axis().len()]
    }
    fn validate(&self) -> Result<()> {
        let [n, f] = self.shape();
        if n == 0
            || f == 0
            || self.fit_receipts.len() != n
            || self.variance_origins.as_ref().is_some_and(|x| x.len() != n)
            || ![
                &self.estimand.id,
                &self.estimand.definition,
                &self.estimand.units,
                &self.admission.target,
                &self.admission.evidence,
            ]
            .iter()
            .all(|x| nonblank(x))
        {
            return Err(Error::Invalid("portable metadata dimensions"));
        }
        let mut participants = HashSet::new();
        if self.subjects.iter().any(|s| {
            ![&s.dataset, &s.participant, &s.measurement]
                .iter()
                .all(|x| nonblank(x))
                || !participants.insert((&s.dataset, &s.participant))
        }) || self.fit_receipts.iter().any(|r| {
            ![&r.model, &r.noise, &r.selection]
                .iter()
                .all(|x| nonblank(x))
        }) || self
            .variance_origins
            .as_ref()
            .is_some_and(|origins| origins.iter().any(|x| !valid_origin(x, f)))
        {
            return Err(Error::Invalid("portable metadata identity or provenance"));
        }
        Ok(())
    }
}

fn valid_origin(origin: &VarianceOrigin, features: usize) -> bool {
    match origin {
        VarianceOrigin::Known { method } => nonblank(method),
        VarianceOrigin::Unknown { reason } => nonblank(reason),
        VarianceOrigin::Estimated { df, method, .. } => {
            nonblank(method)
                && match df {
                    DegreesOfFreedom::Scalar(x) => x.is_finite() && *x > 0.,
                    DegreesOfFreedom::ByFeature(x) => {
                        x.len() == features && x.iter().all(|x| x.is_finite() && *x > 0.)
                    }
                }
        }
    }
}

/// Creates a new directory and writes feature-contiguous blocks.  `finish` is
/// the only operation that publishes `manifest.json`; dropped/incomplete output
/// therefore cannot be opened as a complete bundle.
pub struct EstimateBundleWriter {
    directory: PathBuf,
    metadata: PortableMetadata,
    effects: BufWriter<File>,
    variances: Option<BufWriter<File>>,
    effect_hash: Sha256,
    variance_hash: Option<Sha256>,
    next_effect: usize,
    next_variance: usize,
    max_block_bytes: usize,
}
impl EstimateBundleWriter {
    pub fn create(
        directory: impl AsRef<Path>,
        metadata: PortableMetadata,
        options: PortableWriteOptions,
    ) -> Result<Self> {
        metadata.validate()?;
        if options.max_block_bytes < 8 || directory.as_ref().exists() {
            return Err(Error::Invalid(
                "portable directory exists or block budget invalid",
            ));
        }
        fs::create_dir(directory.as_ref())?;
        let effects = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.as_ref().join(EFFECTS))?,
        );
        let variances = if metadata.variance_origins.is_some() {
            Some(BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(directory.as_ref().join(VARIANCES))?,
            ))
        } else {
            None
        };
        Ok(Self {
            directory: directory.as_ref().to_path_buf(),
            metadata,
            effects,
            variances,
            effect_hash: Sha256::new(),
            variance_hash: None,
            next_effect: 0,
            next_variance: 0,
            max_block_bytes: options.max_block_bytes,
        })
    }
    pub fn write_effect_block(
        &mut self,
        start_feature: usize,
        values: ArrayView2<'_, f64>,
    ) -> Result<()> {
        Self::write_block(
            &mut self.effects,
            &mut self.effect_hash,
            &mut self.next_effect,
            self.metadata.shape(),
            self.max_block_bytes,
            start_feature,
            values,
        )
    }
    pub fn write_variance_block(
        &mut self,
        start_feature: usize,
        values: ArrayView2<'_, f64>,
    ) -> Result<()> {
        let writer = self
            .variances
            .as_mut()
            .ok_or(Error::Invalid("bundle has no variance payload"))?;
        let hash = self.variance_hash.get_or_insert_with(Sha256::new);
        Self::write_block(
            writer,
            hash,
            &mut self.next_variance,
            self.metadata.shape(),
            self.max_block_bytes,
            start_feature,
            values,
        )
    }
    fn write_block(
        writer: &mut BufWriter<File>,
        hash: &mut Sha256,
        next: &mut usize,
        shape: [usize; 2],
        max: usize,
        start: usize,
        values: ArrayView2<'_, f64>,
    ) -> Result<()> {
        let (n, b) = values.dim();
        let bytes = n
            .checked_mul(b)
            .and_then(|x| x.checked_mul(8))
            .ok_or(Error::Invalid("portable block size"))?;
        if start != *next
            || n != shape[0]
            || b == 0
            || start.checked_add(b).is_none_or(|end| end > shape[1])
            || bytes > max
        {
            return Err(Error::Invalid("portable block order or shape"));
        }
        // Payloads are feature-major: each contiguous column holds all subjects.
        for feature in 0..b {
            for subject in 0..n {
                let bytes = values[(subject, feature)].to_le_bytes();
                writer.write_all(&bytes)?;
                hash.update(bytes);
            }
        }
        *next += b;
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        let features = self.metadata.shape()[1];
        if self.next_effect != features
            || self.variances.is_some() && self.next_variance != features
        {
            return Err(Error::Invalid("portable bundle incomplete"));
        }
        self.effects.flush()?;
        self.effects.get_ref().sync_data()?;
        if let Some(w) = self.variances.as_mut() {
            w.flush()?;
            w.get_ref().sync_data()?;
        }
        let effects = Payload {
            file: EFFECTS.into(),
            sha256: format!("{:x}", self.effect_hash.finalize()),
        };
        let variances = self.variance_hash.take().map(|x| Payload {
            file: VARIANCES.into(),
            sha256: format!("{:x}", x.finalize()),
        });
        let manifest = Manifest {
            format: FORMAT.into(),
            version: VERSION,
            shape: self.metadata.shape(),
            metadata: MetaDto::from_metadata(&self.metadata),
            effects,
            variances,
        };
        let temporary = self.directory.join(".manifest.json.tmp");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer(&mut file, &manifest)?;
        file.flush()?;
        file.sync_data()?;
        // A hard link publishes the fully-written file without replacing a
        // manifest another process may have created in the meantime.
        fs::hard_link(&temporary, self.directory.join(MANIFEST))?;
        fs::remove_file(temporary)?;
        Ok(())
    }
}

/// Streams an existing cohort through bounded feature blocks to a new bundle.
pub fn write_cohort_bundle(
    directory: impl AsRef<Path>,
    cohort: &IndependentCohort,
    block_features: usize,
    options: PortableWriteOptions,
) -> Result<()> {
    if block_features == 0 {
        return Err(Error::Invalid("portable block feature count"));
    }
    let metadata = PortableMetadata::from_cohort(cohort);
    let [n, f] = metadata.shape();
    let mut writer = EstimateBundleWriter::create(directory, metadata, options)?;
    let context = ExecutionContext::new(MemoryBudget::new(options.max_block_bytes));
    let mut er = context.open(cohort.frame().assay(cohort.effect_assay())?.source());
    let mut vr = cohort.variance_assay().map(|name| {
        context.open(
            cohort
                .frame()
                .assay(name)
                .expect("validated cohort")
                .source(),
        )
    });
    for start in (0..f).step_by(block_features) {
        let end = (start + block_features).min(f);
        let select = ReadSelection::new([n, f], Selection::All, Selection::Range(start..end))?;
        let eb = context.read(&mut er, &select)?;
        let ev = match eb.into_values() {
            Values::F64(x) => x,
            _ => return Err(Error::Invalid("portable requires f64 effects")),
        };
        writer.write_effect_block(
            start,
            Array2::from_shape_vec((n, end - start), ev)
                .map_err(|_| Error::Invalid("portable effect shape"))?
                .view(),
        )?;
        if let Some(reader) = vr.as_mut() {
            let vb = context.read(reader, &select)?;
            let vv = match vb.into_values() {
                Values::F64(x) => x,
                _ => return Err(Error::Invalid("portable requires f64 variances")),
            };
            writer.write_variance_block(
                start,
                Array2::from_shape_vec((n, end - start), vv)
                    .map_err(|_| Error::Invalid("portable variance shape"))?
                    .view(),
            )?;
        }
    }
    writer.finish()
}

pub fn read_cohort_bundle(
    directory: impl AsRef<Path>,
    options: PortableReadOptions,
) -> Result<IndependentCohort> {
    if options.max_manifest_bytes == 0 || options.max_block_bytes < 8 {
        return Err(Error::Invalid("portable read limits"));
    }
    let file = File::open(directory.as_ref().join(MANIFEST))?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > options.max_manifest_bytes as u64 {
        return Err(Error::Invalid("portable manifest exceeds limit"));
    }
    let mut bytes = Vec::with_capacity((metadata.len() as usize).min(options.max_manifest_bytes));
    file.take(options.max_manifest_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > options.max_manifest_bytes {
        return Err(Error::Invalid("portable manifest exceeds limit"));
    }
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    if manifest.format != FORMAT || manifest.version != VERSION {
        return Err(Error::Invalid("unknown portable bundle version"));
    }
    let [n, f] = manifest.shape;
    if n == 0
        || f == 0
        || manifest.effects.file != EFFECTS
        || manifest
            .variances
            .as_ref()
            .is_some_and(|x| x.file != VARIANCES)
    {
        return Err(Error::Invalid("portable manifest payload"));
    }
    let metadata = manifest.metadata.into_metadata()?;
    metadata.validate()?;
    if metadata.shape() != [n, f] {
        return Err(Error::Invalid("portable metadata shape"));
    }
    verify_payload(
        &directory.as_ref().join(EFFECTS),
        n,
        f,
        &manifest.effects.sha256,
        options.max_block_bytes,
    )?;
    if let Some(payload) = &manifest.variances {
        verify_payload(
            &directory.as_ref().join(VARIANCES),
            n,
            f,
            &payload.sha256,
            options.max_block_bytes,
        )?;
    }
    if metadata.variance_origins.is_some() != manifest.variances.is_some() {
        return Err(Error::Invalid("portable uncertainty payload mismatch"));
    }
    let ids = metadata
        .subjects
        .iter()
        .map(|x| x.observation.clone())
        .collect::<Vec<_>>();
    let effect = SourceSpec::feature_major_f64_file(directory.as_ref().join(EFFECTS), [n, f])?;
    let mut assays = vec![Assay::new(
        "effect",
        effect,
        ids.clone(),
        metadata.features.axis().ids().to_vec(),
        AssayAnnotations::default(),
    )?];
    if manifest.variances.is_some() {
        assays.push(Assay::new(
            "variance",
            SourceSpec::feature_major_f64_file(directory.as_ref().join(VARIANCES), [n, f])?,
            ids.clone(),
            metadata.features.axis().ids().to_vec(),
            AssayAnnotations::default(),
        )?);
    }
    let frame = FmriFrame::new(
        Axis::new(ids, Vec::new())?,
        metadata.features,
        assays,
        "effect",
        Metadata::new(),
    )?;
    IndependentCohort::new(
        frame,
        metadata.subjects,
        metadata.estimand,
        "effect".into(),
        metadata.variance_origins.map(|x| ("variance".into(), x)),
        metadata.fit_receipts,
        metadata.admission,
    )
}

fn verify_payload(path: &Path, n: usize, f: usize, expected: &str, buffer: usize) -> Result<()> {
    let bytes = n
        .checked_mul(f)
        .and_then(|x| x.checked_mul(8))
        .ok_or(Error::Invalid("portable payload size"))?;
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != bytes as u64 {
        return Err(Error::Invalid("portable payload truncated"));
    }
    let mut hash = Sha256::new();
    let mut chunk = vec![0; buffer.min(64 << 10)];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        hash.update(&chunk[..read]);
    }
    if format!("{:x}", hash.finalize()) != expected {
        return Err(Error::Invalid("portable payload checksum"));
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    version: u32,
    shape: [usize; 2],
    metadata: MetaDto,
    effects: Payload,
    #[serde(default)]
    variances: Option<Payload>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    file: String,
    sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetaDto {
    subjects: Vec<SubjectDto>,
    estimand: EstimandDto,
    feature_space: SpaceDto,
    variance_origins: Option<Vec<OriginDto>>,
    fit_receipts: Vec<FitDto>,
    admission: AdmissionDto,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SubjectDto {
    observation: String,
    dataset: String,
    participant: String,
    measurement: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimandDto {
    id: String,
    definition: String,
    units: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FitDto {
    model: String,
    noise: String,
    selection: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmissionDto {
    target: String,
    evidence: String,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum SpaceDto {
    Index {
        namespace: String,
        ids: Vec<String>,
    },
    Volume {
        dimensions: [usize; 3],
        affine: [[f64; 4]; 4],
        support: Vec<usize>,
        template: Option<String>,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum OriginDto {
    Known {
        method: String,
    },
    Estimated {
        df: DfDto,
        method: String,
        approximate: bool,
    },
    Unknown {
        reason: String,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum DfDto {
    Scalar(f64),
    ByFeature(Vec<f64>),
}

impl MetaDto {
    fn from_metadata(x: &PortableMetadata) -> Self {
        Self {
            subjects: x
                .subjects
                .iter()
                .map(|s| SubjectDto {
                    observation: s.observation.as_str().into(),
                    dataset: s.dataset.clone(),
                    participant: s.participant.clone(),
                    measurement: s.measurement.clone(),
                })
                .collect(),
            estimand: EstimandDto {
                id: x.estimand.id.clone(),
                definition: x.estimand.definition.clone(),
                units: x.estimand.units.clone(),
            },
            feature_space: SpaceDto::from_space(x.features.space()),
            variance_origins: x
                .variance_origins
                .as_ref()
                .map(|v| v.iter().map(OriginDto::from_origin).collect()),
            fit_receipts: x
                .fit_receipts
                .iter()
                .map(|r| FitDto {
                    model: r.model.clone(),
                    noise: r.noise.clone(),
                    selection: r.selection.clone(),
                })
                .collect(),
            admission: AdmissionDto {
                target: x.admission.target.clone(),
                evidence: x.admission.evidence.clone(),
            },
        }
    }
    fn into_metadata(self) -> Result<PortableMetadata> {
        let subjects = self
            .subjects
            .into_iter()
            .map(|s| {
                Ok(Subject {
                    observation: ObservationId::new(s.observation)?,
                    dataset: s.dataset,
                    participant: s.participant,
                    measurement: s.measurement,
                })
            })
            .collect::<std::result::Result<Vec<_>, fmri_data::FmriDatasetError>>()?;
        let features = FeatureAxis::from_space(self.feature_space.into_space()?)?;
        Ok(PortableMetadata {
            subjects,
            estimand: Estimand {
                id: self.estimand.id,
                definition: self.estimand.definition,
                units: self.estimand.units,
            },
            features,
            variance_origins: self
                .variance_origins
                .map(|v| v.into_iter().map(OriginDto::into_origin).collect())
                .transpose()?,
            fit_receipts: self
                .fit_receipts
                .into_iter()
                .map(|r| FitReceipt {
                    model: r.model,
                    noise: r.noise,
                    selection: r.selection,
                })
                .collect(),
            admission: Admission {
                target: self.admission.target,
                evidence: self.admission.evidence,
            },
        })
    }
}
impl SpaceDto {
    fn from_space(x: &FeatureSpace) -> Self {
        match x {
            FeatureSpace::Index(s) => Self::Index {
                namespace: s.namespace().into(),
                ids: x.ids().iter().map(|x| x.as_str().into()).collect(),
            },
            FeatureSpace::Volume(s) => Self::Volume {
                dimensions: s.dimensions(),
                affine: *s.affine(),
                support: s.support().to_vec(),
                template: s.template().map(str::to_owned),
            },
        }
    }
    fn into_space(self) -> Result<FeatureSpace> {
        Ok(match self {
            Self::Index { namespace, ids } => FeatureSpace::Index(IndexSpace::new(
                namespace,
                ids.into_iter()
                    .map(FeatureId::new)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )?),
            Self::Volume {
                dimensions,
                affine,
                support,
                template,
            } => FeatureSpace::Volume(VolumeSpace::new(
                dimensions,
                affine,
                Some(support),
                template,
            )?),
        })
    }
}
impl OriginDto {
    fn from_origin(x: &VarianceOrigin) -> Self {
        match x {
            VarianceOrigin::Known { method } => Self::Known {
                method: method.clone(),
            },
            VarianceOrigin::Unknown { reason } => Self::Unknown {
                reason: reason.clone(),
            },
            VarianceOrigin::Estimated {
                df,
                method,
                approximate,
            } => Self::Estimated {
                df: match df {
                    DegreesOfFreedom::Scalar(x) => DfDto::Scalar(*x),
                    DegreesOfFreedom::ByFeature(x) => DfDto::ByFeature(x.clone()),
                },
                method: method.clone(),
                approximate: *approximate,
            },
        }
    }
    fn into_origin(self) -> Result<VarianceOrigin> {
        Ok(match self {
            Self::Known { method } => VarianceOrigin::Known { method },
            Self::Unknown { reason } => VarianceOrigin::Unknown { reason },
            Self::Estimated {
                df,
                method,
                approximate,
            } => VarianceOrigin::Estimated {
                df: match df {
                    DfDto::Scalar(x) => DegreesOfFreedom::Scalar(x),
                    DfDto::ByFeature(x) => DegreesOfFreedom::ByFeature(x),
                },
                method,
                approximate,
            },
        })
    }
}
