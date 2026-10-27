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
    View {
        source: SourceSpec,
        selection: ReadSelection,
    },
}

impl SourceSpec {
    pub fn memory(block: NumericBlock) -> Self {
        Self {
            node: Arc::new(SourceNode::Memory(block)),
        }
    }
    pub fn shape(&self) -> [usize; 2] {
        match self.node.as_ref() {
            SourceNode::Memory(b) => b.shape(),
            SourceNode::View { selection, .. } => selection.shape(),
        }
    }
    pub fn dtype(&self) -> DType {
        match self.node.as_ref() {
            SourceNode::Memory(b) => b.dtype(),
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

    fn root_position(&self, row: usize, col: usize) -> (&NumericBlock, usize) {
        match self.node.as_ref() {
            SourceNode::Memory(b) => (b, row * b.shape()[1] + col),
            SourceNode::View { source, selection } => {
                source.root_position(selection.rows().at(row), selection.columns().at(col))
            }
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
        Ok(ReadCost {
            output_bytes: bytes,
            peak_bytes: bytes,
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
        let values = match self.dtype() {
            DType::F32 => {
                let mut out = reserved(n)?;
                for r in 0..rows {
                    for c in 0..cols {
                        let (b, p) = self
                            .source
                            .root_position(selection.rows().at(r), selection.columns().at(c));
                        if let Values::F32(v) = &b.values {
                            out.push(v[p]);
                        }
                    }
                }
                Values::F32(out)
            }
            DType::F64 => {
                let mut out = reserved(n)?;
                for r in 0..rows {
                    for c in 0..cols {
                        let (b, p) = self
                            .source
                            .root_position(selection.rows().at(r), selection.columns().at(c));
                        if let Values::F64(v) = &b.values {
                            out.push(v[p]);
                        }
                    }
                }
                Values::F64(out)
            }
        };
        NumericBlock::new(selection.shape(), values)
    }
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
