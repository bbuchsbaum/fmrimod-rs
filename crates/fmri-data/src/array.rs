use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::errors::{FmriDatasetError, Result};
use crate::selection::ReadSelection;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Owned row-major numerical storage. No implicit precision conversion.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Values {
    #[serde(rename = "f32_bits")]
    F32(#[serde(serialize_with = "serialize_f32")] Vec<f32>),
    #[serde(rename = "f64_bits")]
    F64(#[serde(serialize_with = "serialize_f64")] Vec<f64>),
}

fn serialize_f32<S: serde::Serializer>(
    values: &[f32],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.collect_seq(values.iter().map(|v| v.to_bits()))
}

fn serialize_f64<S: serde::Serializer>(
    values: &[f64],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.collect_seq(values.iter().map(|v| v.to_bits()))
}

impl Values {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::F32(x) => x.len(),
            Self::F64(x) => x.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NumericBlock {
    shape: [usize; 2],
    values: Values,
}

impl NumericBlock {
    pub fn new(shape: [usize; 2], values: Values) -> Result<Self> {
        let expected = shape[0]
            .checked_mul(shape[1])
            .ok_or(FmriDatasetError::SizeOverflow)?;
        if expected != values.len() {
            return Err(FmriDatasetError::LengthMismatch {
                expected,
                found: values.len(),
            });
        }
        Ok(Self { shape, values })
    }
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn dtype(&self) -> DType {
        self.values.dtype()
    }
    pub fn values(&self) -> &Values {
        &self.values
    }
    pub fn into_values(self) -> Values {
        self.values
    }
}

/// Serializable source description. It contains no open reader or callback.
/// The encoding is crate-local; it is not an FDS manifest.
#[derive(Debug, Clone, Serialize)]
pub struct SourceSpec {
    node: Arc<SourceNode>,
}

#[derive(Debug, Serialize)]
enum SourceNode {
    Memory(NumericBlock),
    FeatureMajorF64File(FileSource),
    View {
        source: SourceSpec,
        selection: ReadSelection,
    },
}

/// A regular file of little-endian f64 values, laid out feature-major.
#[derive(Debug, Clone, Serialize)]
struct FileSource {
    path: PathBuf,
    shape: [usize; 2],
}

impl FileSource {
    fn expected_bytes(shape: [usize; 2]) -> Result<usize> {
        shape[0]
            .checked_mul(shape[1])
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(FmriDatasetError::SizeOverflow)
    }

    fn validate_path(path: &Path, shape: [usize; 2]) -> Result<()> {
        let metadata = std::fs::metadata(path)?;
        Self::validate_metadata(&metadata, shape)
    }

    fn validate_metadata(metadata: &Metadata, shape: [usize; 2]) -> Result<()> {
        if !metadata.file_type().is_file() {
            return Err(crate::errors::invalid(
                "feature-major f64 file",
                "path is not a regular file",
            ));
        }
        let expected = Self::expected_bytes(shape)?;
        let found = metadata.len();
        if found != expected as u64 {
            return Err(crate::errors::invalid(
                "feature-major f64 file",
                format!("expected {expected} bytes, found {found}"),
            ));
        }
        Ok(())
    }
}

impl SourceSpec {
    pub fn memory(block: NumericBlock) -> Self {
        Self {
            node: Arc::new(SourceNode::Memory(block)),
        }
    }
    /// Describe a bounded file source without loading any numerical values.
    ///
    /// The file must be a regular file whose exact length is `rows * columns *
    /// 8`. Values are stored little-endian with each feature (column) forming
    /// one contiguous run of rows.
    pub fn feature_major_f64_file(path: impl Into<PathBuf>, shape: [usize; 2]) -> Result<Self> {
        let path = path.into();
        FileSource::validate_path(&path, shape)?;
        Ok(Self {
            node: Arc::new(SourceNode::FeatureMajorF64File(FileSource { path, shape })),
        })
    }
    pub fn shape(&self) -> [usize; 2] {
        match self.node.as_ref() {
            SourceNode::Memory(b) => b.shape(),
            SourceNode::FeatureMajorF64File(source) => source.shape,
            SourceNode::View { selection, .. } => selection.shape(),
        }
    }
    pub fn dtype(&self) -> DType {
        match self.node.as_ref() {
            SourceNode::Memory(b) => b.dtype(),
            SourceNode::FeatureMajorF64File(_) => DType::F64,
            SourceNode::View { source, .. } => source.dtype(),
        }
    }
    pub fn view(&self, selection: ReadSelection) -> Result<Self> {
        selection.validate_for(self.shape())?;
        // Compose against the root so repeated slicing does not grow a reader stack.
        let (source, selection) = match self.node.as_ref() {
            SourceNode::View {
                source,
                selection: parent,
            } => {
                let rows = (0..selection.shape()[0])
                    .map(|i| parent.rows().at(selection.rows().at(i)))
                    .collect();
                let cols = (0..selection.shape()[1])
                    .map(|i| parent.columns().at(selection.columns().at(i)))
                    .collect();
                (
                    source.clone(),
                    ReadSelection::new(
                        source.shape(),
                        crate::Selection::Indices(rows),
                        crate::Selection::Indices(cols),
                    )?,
                )
            }
            _ => (self.clone(), selection),
        };
        Ok(Self {
            node: Arc::new(SourceNode::View { source, selection }),
        })
    }

    fn root_coordinates(&self, row: usize, col: usize) -> (usize, usize) {
        match self.node.as_ref() {
            SourceNode::Memory(_) | SourceNode::FeatureMajorF64File(_) => (row, col),
            SourceNode::View { source, selection } => {
                source.root_coordinates(selection.rows().at(row), selection.columns().at(col))
            }
        }
    }

    fn root_memory(&self) -> Option<&NumericBlock> {
        match self.node.as_ref() {
            SourceNode::Memory(block) => Some(block),
            SourceNode::FeatureMajorF64File(_) => None,
            SourceNode::View { source, .. } => source.root_memory(),
        }
    }

    fn root_file(&self) -> Option<&FileSource> {
        match self.node.as_ref() {
            SourceNode::Memory(_) => None,
            SourceNode::FeatureMajorF64File(source) => Some(source),
            SourceNode::View { source, .. } => source.root_file(),
        }
    }
}

/// Additional live numeric buffer bytes, excluding already-owned frame/source
/// data and metadata. Readers must include all numeric scratch in peak_bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadCost {
    pub output_bytes: usize,
    pub peak_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    bytes: usize,
}

impl MemoryBudget {
    pub fn new(bytes: usize) -> Self {
        Self { bytes }
    }
    pub fn bytes(self) -> usize {
        self.bytes
    }
    pub fn check(self, required: usize) -> Result<()> {
        if required > self.bytes {
            Err(FmriDatasetError::BudgetExceeded {
                required,
                budget: self.bytes,
            })
        } else {
            Ok(())
        }
    }
}

/// Runtime boundary. No Send/Sync promise is imposed on open resources.
pub trait ArrayReader {
    fn shape(&self) -> [usize; 2];
    fn dtype(&self) -> DType;
    fn estimate(&self, selection: &ReadSelection) -> Result<ReadCost>;
    /// Must check budget before allocating or reading numerical values.
    fn read(&mut self, selection: &ReadSelection, budget: MemoryBudget) -> Result<NumericBlock>;
}

pub struct MemoryReader<'a> {
    source: &'a SourceSpec,
}

impl ArrayReader for MemoryReader<'_> {
    fn shape(&self) -> [usize; 2] {
        self.source.shape()
    }
    fn dtype(&self) -> DType {
        self.source.dtype()
    }
    fn estimate(&self, selection: &ReadSelection) -> Result<ReadCost> {
        selection.validate_for(self.shape())?;
        let [r, c] = selection.shape();
        let bytes = r
            .checked_mul(c)
            .and_then(|n| n.checked_mul(self.dtype().byte_width()))
            .ok_or(FmriDatasetError::SizeOverflow)?;
        let scratch_bytes = match self.source.root_file() {
            Some(_) if r != 0 && c != 0 => file_scratch_bytes(self.source, selection)?,
            _ => 0,
        };
        let peak_bytes = bytes
            .checked_add(scratch_bytes)
            .ok_or(FmriDatasetError::SizeOverflow)?;
        Ok(ReadCost {
            output_bytes: bytes,
            peak_bytes,
        })
    }
    fn read(&mut self, selection: &ReadSelection, budget: MemoryBudget) -> Result<NumericBlock> {
        budget.check(self.estimate(selection)?.peak_bytes)?;
        let [rows, cols] = selection.shape();
        let n = rows * cols;
        if n == 0 {
            let values = match self.dtype() {
                DType::F32 => Values::F32(Vec::new()),
                DType::F64 => Values::F64(Vec::new()),
            };
            return NumericBlock::new(selection.shape(), values);
        }
        let values = match self.source.root_file() {
            Some(file_source) => {
                Values::F64(read_file_values(self.source, file_source, selection)?)
            }
            None => match self.dtype() {
                DType::F32 => {
                    let mut out = reserved(n)?;
                    let block = self.source.root_memory().expect("memory root");
                    for r in 0..rows {
                        for c in 0..cols {
                            let (root_row, root_col) = self.source.root_coordinates(
                                selection.rows().at(r),
                                selection.columns().at(c),
                            );
                            if let Values::F32(v) = &block.values {
                                out.push(v[root_row * block.shape()[1] + root_col]);
                            }
                        }
                    }
                    Values::F32(out)
                }
                DType::F64 => {
                    let mut out = reserved(n)?;
                    let block = self.source.root_memory().expect("memory root");
                    for r in 0..rows {
                        for c in 0..cols {
                            let (root_row, root_col) = self.source.root_coordinates(
                                selection.rows().at(r),
                                selection.columns().at(c),
                            );
                            if let Values::F64(v) = &block.values {
                                out.push(v[root_row * block.shape()[1] + root_col]);
                            }
                        }
                    }
                    Values::F64(out)
                }
            },
        };
        NumericBlock::new(selection.shape(), values)
    }
}

const FILE_SCRATCH_BYTES: usize = 64 * 1024;

fn selected_file_rows_are_contiguous(source: &SourceSpec, selection: &ReadSelection) -> bool {
    let rows = selection.shape()[0];
    rows > 0
        && (1..rows).all(|row| {
            let previous = source
                .root_coordinates(selection.rows().at(row - 1), selection.columns().at(0))
                .0;
            let current = source
                .root_coordinates(selection.rows().at(row), selection.columns().at(0))
                .0;
            current == previous.saturating_add(1)
        })
}

fn file_scratch_bytes(source: &SourceSpec, selection: &ReadSelection) -> Result<usize> {
    let rows = selection.shape()[0];
    if selected_file_block_is_contiguous(source, selection) {
        rows.checked_mul(selection.shape()[1])
            .and_then(|n| n.checked_mul(8))
            .map(|bytes| bytes.min(FILE_SCRATCH_BYTES))
            .ok_or(FmriDatasetError::SizeOverflow)
    } else if selected_file_rows_are_contiguous(source, selection) {
        rows.checked_mul(std::mem::size_of::<f64>())
            .map(|bytes| bytes.min(FILE_SCRATCH_BYTES))
            .ok_or(FmriDatasetError::SizeOverflow)
    } else {
        Ok(std::mem::size_of::<f64>())
    }
}

fn selected_file_block_is_contiguous(source: &SourceSpec, selection: &ReadSelection) -> bool {
    let Some(file) = source.root_file() else {
        return false;
    };
    let [rows, columns] = selection.shape();
    if rows == 0
        || columns == 0
        || rows != file.shape[0]
        || !selected_file_rows_are_contiguous(source, selection)
    {
        return false;
    }
    let (first_row, first_column) =
        source.root_coordinates(selection.rows().at(0), selection.columns().at(0));
    first_row == 0
        && (1..columns).all(|c| {
            source
                .root_coordinates(selection.rows().at(0), selection.columns().at(c))
                .1
                == first_column + c
        })
}

fn read_file_values(
    source: &SourceSpec,
    file_source: &FileSource,
    selection: &ReadSelection,
) -> Result<Vec<f64>> {
    // The caller has already applied the same estimate/budget check. Recheck
    // metadata before allocating values so a truncated or replaced file cannot
    // cause an unbounded numerical read.
    let mut file = File::open(&file_source.path)?;
    FileSource::validate_metadata(&file.metadata()?, file_source.shape)?;
    let [rows, columns] = selection.shape();
    let n = rows
        .checked_mul(columns)
        .ok_or(FmriDatasetError::SizeOverflow)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut output = reserved(n)?;
    output.resize(n, 0.0);
    if selected_file_block_is_contiguous(source, selection) {
        let mut scratch = reserved(file_scratch_bytes(source, selection)?)?;
        scratch.resize(file_scratch_bytes(source, selection)?, 0);
        let first_col = source
            .root_coordinates(selection.rows().at(0), selection.columns().at(0))
            .1;
        let offset = first_col
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(8))
            .ok_or(FmriDatasetError::SizeOverflow)?;
        file.seek(SeekFrom::Start(offset as u64))?;
        let mut position = 0;
        while position < n {
            let count = (n - position).min(scratch.len() / 8);
            file.read_exact(&mut scratch[..count * 8])?;
            for (j, bytes) in scratch[..count * 8].chunks_exact(8).enumerate() {
                let index = position + j;
                output[(index % rows) * columns + index / rows] =
                    f64::from_le_bytes(bytes.try_into().expect("eight-byte chunk"));
            }
            position += count;
        }
    } else if selected_file_rows_are_contiguous(source, selection) {
        let scratch_len = file_scratch_bytes(source, selection)?;
        let mut scratch = reserved(scratch_len)?;
        scratch.resize(scratch_len, 0);
        let first_row = source
            .root_coordinates(selection.rows().at(0), selection.columns().at(0))
            .0;
        for output_col in 0..columns {
            let root_col = source
                .root_coordinates(selection.rows().at(0), selection.columns().at(output_col))
                .1;
            let byte_offset = root_col
                .checked_mul(file_source.shape[0])
                .and_then(|offset| offset.checked_add(first_row))
                .and_then(|offset| offset.checked_mul(std::mem::size_of::<f64>()))
                .ok_or(FmriDatasetError::SizeOverflow)?;
            file.seek(SeekFrom::Start(byte_offset as u64))?;
            let mut output_row = 0;
            while output_row < rows {
                let count = ((rows - output_row) * std::mem::size_of::<f64>()).min(scratch.len());
                file.read_exact(&mut scratch[..count])?;
                for (index, bytes) in scratch[..count].chunks_exact(8).enumerate() {
                    output[(output_row + index) * columns + output_col] =
                        f64::from_le_bytes(bytes.try_into().expect("eight-byte chunk"));
                }
                output_row += count / std::mem::size_of::<f64>();
            }
        }
    } else {
        let mut bytes = [0_u8; std::mem::size_of::<f64>()];
        for output_row in 0..rows {
            for output_col in 0..columns {
                let (root_row, root_col) = source.root_coordinates(
                    selection.rows().at(output_row),
                    selection.columns().at(output_col),
                );
                let byte_offset = root_col
                    .checked_mul(file_source.shape[0])
                    .and_then(|offset| offset.checked_add(root_row))
                    .and_then(|offset| offset.checked_mul(std::mem::size_of::<f64>()))
                    .ok_or(FmriDatasetError::SizeOverflow)?;
                file.seek(SeekFrom::Start(byte_offset as u64))?;
                file.read_exact(&mut bytes)?;
                output[output_row * columns + output_col] = f64::from_le_bytes(bytes);
            }
        }
    }
    Ok(output)
}

pub(crate) fn reserved<T>(n: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|e| FmriDatasetError::Allocation(e.to_string()))?;
    Ok(out)
}

/// Owns execution policy; future file resolvers and caches belong here.
pub struct ExecutionContext {
    budget: MemoryBudget,
}

impl ExecutionContext {
    pub fn new(budget: MemoryBudget) -> Self {
        Self { budget }
    }
    pub fn open<'a>(&self, source: &'a SourceSpec) -> MemoryReader<'a> {
        MemoryReader { source }
    }
    pub fn collect(&self, source: &SourceSpec) -> Result<NumericBlock> {
        self.read(&mut self.open(source), &ReadSelection::all(source.shape())?)
    }
    pub fn read(
        &self,
        reader: &mut dyn ArrayReader,
        selection: &ReadSelection,
    ) -> Result<NumericBlock> {
        selection.validate_for(reader.shape())?;
        let dtype = reader.dtype();
        let [rows, columns] = selection.shape();
        let output_bytes = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(dtype.byte_width()))
            .ok_or(FmriDatasetError::SizeOverflow)?;
        let cost = reader.estimate(selection)?;
        if cost.output_bytes != output_bytes || cost.peak_bytes < output_bytes {
            return Err(crate::errors::invalid(
                "reader estimate",
                "output size or peak is inconsistent",
            ));
        }
        self.budget.check(cost.peak_bytes)?;
        let block = reader.read(selection, self.budget)?;
        if block.shape() != selection.shape() {
            return Err(FmriDatasetError::ShapeMismatch {
                expected: selection.shape(),
                found: block.shape(),
            });
        }
        if block.dtype() != dtype {
            return Err(crate::errors::invalid(
                "reader",
                "dtype changed during read",
            ));
        }
        Ok(block)
    }
}
