//! Explicit feature-block adapter. The caller guarantees that reader rows have
//! the same identities/order as the prepared design. This API does not infer
//! alignment from equal dimensions. It requires f64 and preserves feature order.
use crate::{Fit, FitError, GlsModel, LinearModel, Result};
use fmridataset::{
    ArrayReader, DType, ExecutionContext, MemoryBudget, ReadSelection, Selection, Values,
};
use ndarray::ArrayView2;
use std::ops::Range;

/// Fit consecutive feature blocks and deliver compact results to a sink.
/// `working_bytes` bounds the estimated reader peak plus fitting buffers. It
/// excludes the already-resident model/source, allocator overhead, numerical-backend packing/custom-transform scratch, and data the
/// sink chooses to retain. Reader estimates must honestly bound backend memory.
/// A sink error stops processing; previously delivered blocks are not rolled back.
pub fn fit_blocks<M: BlockModel>(
    model: &M,
    reader: &mut dyn ArrayReader,
    block_features: usize,
    working_bytes: usize,
    mut sink: impl FnMut(Range<usize>, &Fit<'_>) -> Result<()>,
) -> Result<()> {
    let [n, v] = reader.shape();
    let p = model.columns();
    if n != model.observations() || v == 0 || block_features == 0 || reader.dtype() != DType::F64 {
        return Err(FitError::Invalid("reader rows/dtype or block size"));
    }
    for start in (0..v).step_by(block_features) {
        let end = start.saturating_add(block_features).min(v);
        let b = end - start;
        let selection = ReadSelection::new([n, v], Selection::All, Selection::Range(start..end))
            .map_err(|_| FitError::Invalid("reader selection"))?;
        let cost = reader
            .estimate(&selection)
            .map_err(|_| FitError::Invalid("reader estimate"))?;
        let fit_bytes = n
            .checked_mul(model.time_buffers())
            .and_then(|n| n.checked_add(p))
            .and_then(|s| s.checked_add(1))
            .and_then(|s| s.checked_mul(b))
            .and_then(|s| s.checked_mul(8))
            .ok_or(FitError::Invalid("block byte overflow"))?;
        let total = cost
            .peak_bytes
            .checked_add(fit_bytes)
            .ok_or(FitError::Invalid("block byte overflow"))?;
        if total > working_bytes {
            return Err(FitError::Invalid("working memory budget exceeded"));
        }
        let context = ExecutionContext::new(MemoryBudget::new(working_bytes - fit_bytes));
        let block = context
            .read(reader, &selection)
            .map_err(|_| FitError::Invalid("dataset read failed"))?;
        let Values::F64(values) = block.values() else {
            return Err(FitError::Invalid("reader changed dtype"));
        };
        let data = ArrayView2::from_shape((n, b), values.as_slice())
            .map_err(|_| FitError::Invalid("block shape"))?;
        let fit = model.fit_block(data)?;
        sink(start..end, &fit)?;
    }
    Ok(())
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::LinearModel {}
    impl<P: fmri_noise::WhiteningPlan> Sealed for crate::GlsModel<P> {}
}

/// Supported prepared estimators for feature-block execution.
pub trait BlockModel: sealed::Sealed {
    fn observations(&self) -> usize;
    fn columns(&self) -> usize;
    /// Explicit n-by-block buffers in addition to the reader output.
    fn time_buffers(&self) -> usize;
    fn fit_block(&self, data: ArrayView2<'_, f64>) -> Result<Fit<'_>>;
}
impl BlockModel for LinearModel {
    fn observations(&self) -> usize {
        self.design().nrows()
    }
    fn columns(&self) -> usize {
        self.design().ncols()
    }
    fn time_buffers(&self) -> usize {
        1
    }
    fn fit_block(&self, data: ArrayView2<'_, f64>) -> Result<Fit<'_>> {
        self.fit(data)
    }
}
impl<P: fmri_noise::WhiteningPlan> BlockModel for GlsModel<P> {
    fn observations(&self) -> usize {
        self.noise_plan().observations()
    }
    fn columns(&self) -> usize {
        self.model.design().ncols()
    }
    fn time_buffers(&self) -> usize {
        2
    }
    fn fit_block(&self, data: ArrayView2<'_, f64>) -> Result<Fit<'_>> {
        self.fit(data)
    }
}

/// Bind the design's declared observation identities to a prepared model.
/// Equal row counts alone never establish alignment. IDs are validated for
/// uniqueness; assay IDs must match exactly, in order, before opening a reader.
/// This checks identity, not the correctness of externally supplied acquisition metadata.
pub struct BoundModel<'a, M: BlockModel> {
    model: &'a M,
    observations: fmridataset::Axis<fmridataset::ObservationId>,
}
impl<'a, M: BlockModel> BoundModel<'a, M> {
    pub fn new(model: &'a M, ids: Vec<fmridataset::ObservationId>) -> Result<Self> {
        if ids.len() != model.observations() {
            return Err(FitError::Invalid("design observation ID length"));
        }
        let observations = fmridataset::Axis::new(ids, Vec::new())
            .map_err(|_| FitError::Invalid("invalid design observation IDs"))?;
        Ok(Self {
            model,
            observations,
        })
    }
    /// Deliver feature IDs alongside each compact fit. No implicit row reordering.
    pub fn fit_assay(
        &self,
        assay: &fmridataset::Assay,
        block_features: usize,
        working_bytes: usize,
        mut sink: impl FnMut(&[fmridataset::FeatureId], &Fit<'_>) -> Result<()>,
    ) -> Result<()> {
        if self.observations.ids() != assay.observation_ids() {
            return Err(FitError::Invalid(
                "design/assay observation IDs or ordering differ",
            ));
        }
        let context = ExecutionContext::new(MemoryBudget::new(working_bytes));
        let mut reader = context.open(assay.source());
        fit_blocks(
            self.model,
            &mut reader,
            block_features,
            working_bytes,
            |range, fit| sink(&assay.feature_ids()[range], fit),
        )
    }
}
