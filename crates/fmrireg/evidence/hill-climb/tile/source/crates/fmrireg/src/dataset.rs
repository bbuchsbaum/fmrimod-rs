//! Explicit feature-block adapter. The caller guarantees that reader rows have
//! the same identities/order as the prepared design. This API does not infer
//! alignment from equal dimensions. It requires f64 and preserves feature order.
use crate::{Fit, FitError, FitView, FitWorkspace, GlsModel, LinearModel, Result};
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
    #[doc(hidden)]
    fn prepared(&self) -> &LinearModel;
    #[doc(hidden)]
    fn workspace(&self, features: usize) -> Result<FitWorkspace>;
    #[doc(hidden)]
    fn fit_reuse<'m, 'w>(
        &'m self,
        data: ArrayView2<'_, f64>,
        workspace: &'w mut FitWorkspace,
    ) -> Result<FitView<'m, 'w>>;
}
impl BlockModel for LinearModel {
    fn prepared(&self) -> &LinearModel {
        self
    }
    fn workspace(&self, features: usize) -> Result<FitWorkspace> {
        self.workspace(features)
    }
    fn fit_reuse<'m, 'w>(
        &'m self,
        data: ArrayView2<'_, f64>,
        workspace: &'w mut FitWorkspace,
    ) -> Result<FitView<'m, 'w>> {
        self.fit_into(data, workspace)
    }
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
    fn prepared(&self) -> &LinearModel {
        &self.model
    }
    fn workspace(&self, features: usize) -> Result<FitWorkspace> {
        self.workspace(features)
    }
    fn fit_reuse<'m, 'w>(
        &'m self,
        data: ArrayView2<'_, f64>,
        workspace: &'w mut FitWorkspace,
    ) -> Result<FitView<'m, 'w>> {
        self.fit_into(data, workspace)
    }
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

/// Explicit block/worker policy. `working_bytes` bounds numeric reader and fit
/// buffers across ALL workers, excluding resident sources/models, allocator and
/// backend scratch, thread stacks, custom transform scratch, and retained sink data.
/// With an internally threaded BLAS, cap its threads to avoid nested oversubscription.
#[derive(Debug, Clone, Copy)]
pub struct BlockOptions {
    pub block_features: usize,
    pub workers: usize,
    pub working_bytes: usize,
}

/// Reuse one workspace per worker and deliver borrowed results in feature order.
/// Reads and sink calls stay on the caller thread, so readers need not be Send.
/// At most one block per worker is in flight. Errors stop new reads; already
/// running fits finish before return. Previously delivered sink results remain.
pub fn fit_blocks_with_options<'m, M: BlockModel + Sync>(
    model: &'m M,
    reader: &mut dyn ArrayReader,
    options: BlockOptions,
    mut sink: impl FnMut(Range<usize>, &FitView<'m, '_>) -> Result<()>,
) -> Result<()> {
    use std::sync::mpsc::sync_channel;
    let [n, v] = reader.shape();
    let BlockOptions {
        block_features,
        workers,
        working_bytes,
    } = options;
    if n != model.observations()
        || v == 0
        || block_features == 0
        || workers == 0
        || reader.dtype() != DType::F64
    {
        return Err(FitError::Invalid(
            "reader dimensions/dtype, block size or workers",
        ));
    }
    let capacity = block_features.min(v);
    let blocks = v.div_ceil(capacity);
    let workers = workers.min(blocks);
    let fit_bytes =
        FitWorkspace::required_bytes(n, model.columns(), capacity, model.time_buffers() == 2)?;
    // Inspect estimates before allocating a workspace or reading any data.
    // Conservative reservation: each worker gets one maximum reader peak.
    let mut peak = 0;
    for start in (0..v).step_by(capacity) {
        let range = start..start.saturating_add(capacity).min(v);
        let selection = ReadSelection::new([n, v], Selection::All, Selection::Range(range.clone()))
            .map_err(|_| FitError::Invalid("reader selection"))?;
        let cost = reader
            .estimate(&selection)
            .map_err(|_| FitError::Invalid("reader estimate"))?;
        let expected = n
            .checked_mul(range.len())
            .and_then(|s| s.checked_mul(8))
            .ok_or(FitError::Invalid("reader byte overflow"))?;
        if cost.output_bytes != expected || cost.peak_bytes < expected {
            return Err(FitError::Invalid("reader estimate is inconsistent"));
        }
        peak = peak.max(cost.peak_bytes);
    }
    let required = fit_bytes
        .checked_add(peak)
        .and_then(|s| s.checked_mul(workers))
        .ok_or(FitError::Invalid("parallel working byte overflow"))?;
    if required > working_bytes {
        return Err(FitError::Invalid("working memory budget exceeded"));
    }
    let context = ExecutionContext::new(MemoryBudget::new(peak));
    let mut read = |range: Range<usize>| {
        let selection = ReadSelection::new([n, v], Selection::All, Selection::Range(range))
            .map_err(|_| FitError::Invalid("reader selection"))?;
        let block = context
            .read(reader, &selection)
            .map_err(|_| FitError::Invalid("dataset read failed"))?;
        let [rows, cols] = block.shape();
        let Values::F64(values) = block.into_values() else {
            return Err(FitError::Invalid("reader changed dtype"));
        };
        ndarray::Array2::from_shape_vec((rows, cols), values)
            .map_err(|_| FitError::Invalid("block shape"))
    };
    if workers == 1 {
        let mut workspace = model.workspace(capacity)?;
        for start in (0..v).step_by(capacity) {
            let range = start..start.saturating_add(capacity).min(v);
            let data = read(range.clone())?;
            let fit = model.fit_reuse(data.view(), &mut workspace)?;
            sink(range, &fit)?;
        }
        return Ok(());
    }
    type Task = (Range<usize>, ndarray::Array2<f64>, FitWorkspace);
    type Completed = (Range<usize>, FitWorkspace, Result<()>);
    std::thread::scope(|scope| {
        let mut channels = Vec::with_capacity(workers);
        for _ in 0..workers {
            let workspace = model.workspace(capacity)?;
            let (send_task, tasks) = sync_channel::<Task>(1);
            let (send_result, results) = sync_channel::<Completed>(1);
            std::thread::Builder::new()
                .spawn_scoped(scope, move || {
                    while let Ok((range, data, mut workspace)) = tasks.recv() {
                        let result = model.fit_reuse(data.view(), &mut workspace).map(|_| ());
                        drop(data);
                        if send_result.send((range, workspace, result)).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|_| FitError::Invalid("cannot start fitting worker"))?;
            channels.push((send_task, results, Some(workspace)));
        }
        let mut next = 0usize;
        for (send, _, workspace) in &mut channels {
            let range = next..next.saturating_add(capacity).min(v);
            let data = read(range.clone())?;
            send.send((range.clone(), data, workspace.take().unwrap()))
                .map_err(|_| FitError::Invalid("fitting worker stopped"))?;
            next = range.end;
        }
        for index in 0..blocks {
            let (send, receive, _) = &channels[index % workers];
            let (range, workspace, result) = receive
                .recv()
                .map_err(|_| FitError::Invalid("fitting worker stopped"))?;
            result?;
            sink(
                range.clone(),
                &workspace.view(model.prepared(), range.len()),
            )?;
            if next < v {
                let range = next..next.saturating_add(capacity).min(v);
                let data = read(range.clone())?;
                send.send((range.clone(), data, workspace))
                    .map_err(|_| FitError::Invalid("fitting worker stopped"))?;
                next = range.end;
            }
        }
        Ok(())
    })
}

impl<'m, M: BlockModel + Sync> BoundModel<'m, M> {
    /// Parallel/reusable counterpart of `fit_assay`, with identical ID binding.
    pub fn fit_assay_with_options(
        &self,
        assay: &fmridataset::Assay,
        options: BlockOptions,
        mut sink: impl FnMut(&[fmridataset::FeatureId], &FitView<'m, '_>) -> Result<()>,
    ) -> Result<()> {
        if self.observations.ids() != assay.observation_ids() {
            return Err(FitError::Invalid(
                "design/assay observation IDs or ordering differ",
            ));
        }
        let context = ExecutionContext::new(MemoryBudget::new(options.working_bytes));
        let mut reader = context.open(assay.source());
        fit_blocks_with_options(self.model, &mut reader, options, |range, fit| {
            sink(&assay.feature_ids()[range], fit)
        })
    }
}
