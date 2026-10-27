//! Bounded, typed ingress for 4D NIfTI time series.
//!
//! The reader reopens a streamed decoder for each read. This keeps a subject
//! session from retaining a decompressed 4D image, at the cost of cold-loading
//! compressed files again for independent reads.

use std::path::{Path, PathBuf};

use nifti::{
    header::MAGIC_CODE_NIP1, NiftiHeader, NiftiObject, NiftiType, RandomAccessNiftiVolume,
    ReaderStreamedOptions, Unit,
};

use crate::array::{ArrayReader, DType, MemoryBudget, NumericBlock, ReadCost, Values};
use crate::errors::{invalid, FmriDatasetError, Result};
use crate::{ExecutionContext, ReadSelection, VolumeSpace};

/// Metadata for a 4D NIfTI image, with observations as time frames and
/// features as x-fastest voxel positions in its validated support.
#[derive(Debug, Clone)]
pub struct NiftiSource {
    path: PathBuf,
    shape: [usize; 2],
    space: VolumeSpace,
    datatype: NiftiType,
    signature: HeaderSignature,
}

impl NiftiSource {
    /// Inspect a `.nii` or `.nii.gz` file without loading its 4D values.
    pub fn new(
        path: impl Into<PathBuf>,
        support: Option<Vec<usize>>,
        template: Option<String>,
    ) -> Result<Self> {
        let path = path.into();
        let inspected = inspect(&path, support, template)?;
        Ok(Self {
            path,
            shape: inspected.shape,
            space: inspected.space,
            datatype: inspected.datatype,
            signature: inspected.signature,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn space(&self) -> &VolumeSpace {
        &self.space
    }

    /// Open a fresh streaming reader. The header is revalidated so mutations
    /// after construction cannot silently retain stale geometry metadata.
    pub fn open_reader(&self) -> Result<NiftiReader> {
        let current = inspect(
            &self.path,
            Some(self.space.support().to_vec()),
            self.space.template().map(str::to_owned),
        )?;
        if current.signature != self.signature
            || current.shape != self.shape
            || current.space != self.space
            || current.datatype != self.datatype
        {
            return Err(invalid(
                "NIfTI source",
                "header changed since source construction",
            ));
        }
        Ok(NiftiReader {
            source: self.clone(),
        })
    }
}

impl ExecutionContext {
    /// Open a fresh bounded NIfTI reader under this context's execution policy.
    /// Use [`ExecutionContext::read`] to apply the context budget.
    pub fn open_nifti(&self, source: &NiftiSource) -> Result<NiftiReader> {
        source.open_reader()
    }
}

/// Fresh-reader handle for [`NiftiSource`].
#[derive(Debug)]
pub struct NiftiReader {
    source: NiftiSource,
}

impl ArrayReader for NiftiReader {
    fn shape(&self) -> [usize; 2] {
        self.source.shape
    }

    fn dtype(&self) -> DType {
        DType::F64
    }

    fn estimate(&self, selection: &ReadSelection) -> Result<ReadCost> {
        selection.validate_for(self.shape())?;
        let [rows, columns] = selection.shape();
        let output_bytes = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(FmriDatasetError::SizeOverflow)?;
        if rows == 0 || columns == 0 {
            return Ok(ReadCost {
                output_bytes,
                peak_bytes: output_bytes,
            });
        }
        let frame_bytes = self
            .source
            .space
            .dimensions()
            .into_iter()
            .try_fold(1usize, |n, d| {
                n.checked_mul(d).ok_or(FmriDatasetError::SizeOverflow)
            })?
            .checked_mul(self.source.datatype.size_of())
            .ok_or(FmriDatasetError::SizeOverflow)?;
        // Streamed iteration can temporarily hold the prior raw frame while
        // advancing; count two raw frames rather than assume buffer reuse.
        let peak_bytes = output_bytes
            .checked_add(
                frame_bytes
                    .checked_mul(2)
                    .ok_or(FmriDatasetError::SizeOverflow)?,
            )
            .ok_or(FmriDatasetError::SizeOverflow)?;
        Ok(ReadCost {
            output_bytes,
            peak_bytes,
        })
    }

    fn read(&mut self, selection: &ReadSelection, budget: MemoryBudget) -> Result<NumericBlock> {
        // Validate current header before a budget decision or any voxel value read.
        let checked = self.source.open_reader()?;
        self.source = checked.source;
        let cost = self.estimate(selection)?;
        budget.check(cost.peak_bytes)?;

        let [out_rows, out_columns] = selection.shape();
        let n = out_rows
            .checked_mul(out_columns)
            .ok_or(FmriDatasetError::SizeOverflow)?;
        if n == 0 {
            return NumericBlock::new(selection.shape(), Values::F64(Vec::new()));
        }
        let mut values = crate::array::reserved(n)?;
        values.resize(n, 0.0);

        let object = ReaderStreamedOptions::new()
            .read_file_rank(&self.source.path, 3)
            .map_err(nifti_error)?;
        if HeaderSignature::from_header(object.header()) != self.source.signature {
            return Err(invalid(
                "NIfTI source",
                "header changed while opening streamed reader",
            ));
        }
        let mut volume = object.into_volume();
        let dimensions = self.source.space.dimensions();
        for time in 0..self.source.shape[0] {
            let frame = volume
                .next()
                .ok_or_else(|| invalid("NIfTI data", "missing frame"))?
                .map_err(nifti_error)?;
            for out_row in 0..out_rows {
                if selection.rows().at(out_row) != time {
                    continue;
                }
                for out_col in 0..out_columns {
                    let support_position =
                        self.source.space.support()[selection.columns().at(out_col)];
                    let coordinates = coords(support_position, dimensions)?;
                    values[out_row * out_columns + out_col] =
                        frame.get_f64(&coordinates).map_err(nifti_error)?;
                }
            }
        }
        NumericBlock::new(selection.shape(), Values::F64(values))
    }
}

struct Inspected {
    shape: [usize; 2],
    space: VolumeSpace,
    datatype: NiftiType,
    signature: HeaderSignature,
}

/// Immutable source-relevant header identity checked before streaming and
/// again after the streamed reader opens. Floating fields are retained by bit
/// pattern so a valid NaN in unrelated NIfTI metadata cannot make an unchanged
/// file disagree with itself. This is metadata identity, not a content
/// fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HeaderSignature {
    magic: [u8; 4],
    endianness: nifti::Endianness,
    dimensions: [u16; 5],
    datatype: i16,
    bitpix: i16,
    voxel_offset: u32,
    slope: u32,
    intercept: u32,
    xyzt_units: u8,
    pixdim: [u32; 5],
    intent_code: i16,
    intent_parameters: [u32; 3],
    time_offset: u32,
    slice_duration: u32,
    slice_code: u8,
    qform_code: i16,
    sform_code: i16,
    quaternion: [u32; 6],
    srows: [[u32; 4]; 3],
}

impl HeaderSignature {
    fn from_header(header: &NiftiHeader) -> Self {
        Self {
            magic: header.magic,
            endianness: header.endianness,
            dimensions: [
                header.dim[0],
                header.dim[1],
                header.dim[2],
                header.dim[3],
                header.dim[4],
            ],
            datatype: header.datatype,
            bitpix: header.bitpix,
            voxel_offset: header.vox_offset.to_bits(),
            slope: header.scl_slope.to_bits(),
            intercept: header.scl_inter.to_bits(),
            xyzt_units: header.xyzt_units,
            pixdim: [
                header.pixdim[0].to_bits(),
                header.pixdim[1].to_bits(),
                header.pixdim[2].to_bits(),
                header.pixdim[3].to_bits(),
                header.pixdim[4].to_bits(),
            ],
            intent_code: header.intent_code,
            intent_parameters: [
                header.intent_p1.to_bits(),
                header.intent_p2.to_bits(),
                header.intent_p3.to_bits(),
            ],
            time_offset: header.toffset.to_bits(),
            slice_duration: header.slice_duration.to_bits(),
            slice_code: header.slice_code,
            qform_code: header.qform_code,
            sform_code: header.sform_code,
            quaternion: [
                header.quatern_b.to_bits(),
                header.quatern_c.to_bits(),
                header.quatern_d.to_bits(),
                header.quatern_x.to_bits(),
                header.quatern_y.to_bits(),
                header.quatern_z.to_bits(),
            ],
            srows: [
                header.srow_x.map(f32::to_bits),
                header.srow_y.map(f32::to_bits),
                header.srow_z.map(f32::to_bits),
            ],
        }
    }
}

const MAX_METADATA_GAP: usize = 1024 * 1024;

fn inspect(
    path: &Path,
    support: Option<Vec<usize>>,
    template: Option<String>,
) -> Result<Inspected> {
    let header = NiftiHeader::from_file(path).map_err(nifti_error)?;
    validate_single_file_header(&header)?;
    let datatype = header.data_type().map_err(nifti_error)?;
    if !scalar(datatype) {
        return Err(invalid(
            "NIfTI datatype",
            format!("unsupported non-scalar {datatype:?}"),
        ));
    }
    if header.dimensionality().map_err(nifti_error)? != 4 {
        return Err(invalid(
            "NIfTI dimensions",
            "requires exactly four dimensions (x, y, z, time)",
        ));
    }
    if header.xyzt_to_space().map_err(nifti_error)? != Unit::Mm {
        return Err(invalid(
            "NIfTI spatial units",
            "requires explicit millimeters",
        ));
    }
    let dimensions = [
        usize::from(header.dim[1]),
        usize::from(header.dim[2]),
        usize::from(header.dim[3]),
    ];
    let time = usize::from(header.dim[4]);
    let affine = affine(&header)?;
    let space = VolumeSpace::new(dimensions, affine, support, template)?;
    Ok(Inspected {
        shape: [time, space.support().len()],
        space,
        datatype,
        signature: HeaderSignature::from_header(&header),
    })
}

fn validate_single_file_header(header: &NiftiHeader) -> Result<()> {
    if &header.magic != MAGIC_CODE_NIP1 {
        return Err(invalid(
            "NIfTI storage",
            "requires a single-file n+1 NIfTI image",
        ));
    }
    if !header.vox_offset.is_finite()
        || header.vox_offset < 352.0
        || header.vox_offset.fract() != 0.0
        || header.vox_offset > (352 + MAX_METADATA_GAP) as f32
    {
        return Err(invalid(
            "NIfTI vox_offset",
            "requires a finite integral offset with at most 1 MiB of metadata",
        ));
    }
    if !header.scl_slope.is_finite() || !header.scl_inter.is_finite() {
        return Err(invalid(
            "NIfTI scaling",
            "requires finite slope and intercept",
        ));
    }
    Ok(())
}

fn scalar(datatype: NiftiType) -> bool {
    matches!(
        datatype,
        NiftiType::Uint8
            | NiftiType::Int8
            | NiftiType::Uint16
            | NiftiType::Int16
            | NiftiType::Uint32
            | NiftiType::Int32
            | NiftiType::Uint64
            | NiftiType::Int64
            | NiftiType::Float32
            | NiftiType::Float64
    )
}

fn affine(header: &NiftiHeader) -> Result<[[f64; 4]; 4]> {
    if header.sform_code != 0 {
        header.sform().map_err(nifti_error)?;
        return Ok([
            header.srow_x.map(f64::from),
            header.srow_y.map(f64::from),
            header.srow_z.map(f64::from),
            [0.0, 0.0, 0.0, 1.0],
        ]);
    }
    if header.qform_code == 0 {
        return Err(invalid(
            "NIfTI affine",
            "requires explicit sform or qform code",
        ));
    }
    header.qform().map_err(nifti_error)?;
    qform_affine(header)
}

fn qform_affine(header: &NiftiHeader) -> Result<[[f64; 4]; 4]> {
    // NIfTI reference conversion: nifti_quatern_to_mat44
    // https://raw.githubusercontent.com/NIFTI-Imaging/nifti_clib/master/niftilib/nifti1_io.c
    let [mut b, mut c, mut d] =
        [header.quatern_b, header.quatern_c, header.quatern_d].map(f64::from);
    let sum = b * b + c * c + d * d;
    if !sum.is_finite() {
        return Err(invalid("NIfTI qform", "invalid quaternion"));
    }
    let a2 = 1.0 - sum;
    let a = if a2 < 1e-7 {
        if sum == 0.0 {
            return Err(invalid("NIfTI qform", "invalid zero quaternion"));
        }
        let scale = sum.sqrt().recip();
        b *= scale;
        c *= scale;
        d *= scale;
        0.0
    } else {
        a2.sqrt()
    };
    let [dx, dy, mut dz] = [header.pixdim[1], header.pixdim[2], header.pixdim[3]].map(f64::from);
    if !(dx > 0.0 && dy > 0.0 && dz > 0.0 && [dx, dy, dz].into_iter().all(f64::is_finite)) {
        return Err(invalid(
            "NIfTI qform",
            "requires finite positive voxel spacing",
        ));
    }
    if !header.pixdim[0].is_finite() {
        return Err(invalid("NIfTI qform", "qfac must be finite"));
    }
    if header.pixdim[0] < 0.0 {
        dz = -dz;
    }
    let r = [
        [
            a * a + b * b - c * c - d * d,
            2.0 * (b * c - a * d),
            2.0 * (b * d + a * c),
        ],
        [
            2.0 * (b * c + a * d),
            a * a + c * c - b * b - d * d,
            2.0 * (c * d - a * b),
        ],
        [
            2.0 * (b * d - a * c),
            2.0 * (c * d + a * b),
            a * a + d * d - c * c - b * b,
        ],
    ];
    let translation = [header.quatern_x, header.quatern_y, header.quatern_z].map(f64::from);
    if !translation.into_iter().all(f64::is_finite) {
        return Err(invalid("NIfTI qform", "requires finite translation"));
    }
    Ok([
        [r[0][0] * dx, r[0][1] * dy, r[0][2] * dz, translation[0]],
        [r[1][0] * dx, r[1][1] * dy, r[1][2] * dz, translation[1]],
        [r[2][0] * dx, r[2][1] * dy, r[2][2] * dz, translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn coords(position: usize, dimensions: [usize; 3]) -> Result<[u16; 3]> {
    let x = position % dimensions[0];
    let yz = position / dimensions[0];
    let y = yz % dimensions[1];
    let z = yz / dimensions[1];
    [x, y, z]
        .map(|v| {
            u16::try_from(v).map_err(|_| invalid("NIfTI dimensions", "exceeds u16 coordinates"))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .map(|v| [v[0], v[1], v[2]])
}

fn nifti_error(error: impl std::fmt::Display) -> FmriDatasetError {
    invalid("NIfTI", error.to_string())
}
