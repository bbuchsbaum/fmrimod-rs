//! Bounded, ordered execution of requested outputs through the shared executor.
use crate::{
    dataset::{self, BlockKernel, BlockOptions},
    output::{OutputModel, OutputWorkspace, PreparedOutputs, SummaryView},
    FitError, Result,
};
use fmridataset::{ArrayReader, Axis, ExecutionContext, MemoryBudget, ObservationId};
use ndarray::ArrayView2;
use std::ops::Range;

impl<M: OutputModel + Sync> BlockKernel for PreparedOutputs<'_, M> {
    type Workspace = OutputWorkspace;
    type View<'k, 'w>
        = SummaryView<'k, 'w>
    where
        Self: 'k;
    fn observations(&self) -> usize {
        self.observations()
    }
    fn workspace_bytes(&self, capacity: usize) -> Result<usize> {
        self.workspace_bytes(capacity)
    }
    fn workspace(&self, capacity: usize) -> Result<OutputWorkspace> {
        self.workspace(capacity)
    }
    fn execute(&self, data: ArrayView2<'_, f64>, workspace: &mut OutputWorkspace) -> Result<()> {
        self.execute(data, workspace)
    }
    fn view<'k, 'w>(
        &'k self,
        workspace: &'w OutputWorkspace,
        features: usize,
    ) -> Result<SummaryView<'k, 'w>> {
        self.view(workspace, features)
    }
}

/// Execute a prepared output request. Reader rows must already match the design
/// in identity and order. The reader stays on the calling thread; sinks receive
/// feature blocks in order. Memory accounting and failure semantics are shared
/// with [`dataset::fit_blocks_with_options`]. Retained sink output and immutable
/// model/output geometry are excluded from the working budget.
#[allow(private_bounds)]
pub fn run_blocks_with_options<'p, M: OutputModel + Sync>(
    plan: &'p PreparedOutputs<'_, M>,
    reader: &mut dyn ArrayReader,
    options: BlockOptions,
    sink: impl for<'w> FnMut(Range<usize>, &SummaryView<'p, 'w>) -> Result<()>,
) -> Result<()> {
    dataset::run_blocks(plan, reader, options, sink)
}

/// Bind the design's declared observation identities to an output plan.
/// Matching dimensions alone never establish alignment.
pub struct BoundOutputs<'p, 'm, M> {
    plan: &'p PreparedOutputs<'m, M>,
    observations: Axis<ObservationId>,
}
#[allow(private_bounds)]
impl<'p, 'm, M: OutputModel + Sync> BoundOutputs<'p, 'm, M> {
    pub fn new(plan: &'p PreparedOutputs<'m, M>, ids: Vec<ObservationId>) -> Result<Self> {
        if ids.len() != plan.observations() {
            return Err(FitError::Invalid("design observation ID length"));
        }
        let observations = Axis::new(ids, Vec::new())
            .map_err(|_| FitError::Invalid("invalid design observation IDs"))?;
        Ok(Self { plan, observations })
    }

    /// Check exact observation order before opening the source, and deliver
    /// feature identities alongside the requested estimates and uncertainty.
    pub fn run_assay(
        &self,
        assay: &fmridataset::Assay,
        options: BlockOptions,
        mut sink: impl for<'w> FnMut(&[fmridataset::FeatureId], &SummaryView<'p, 'w>) -> Result<()>,
    ) -> Result<()> {
        if self.observations.ids() != assay.observation_ids() {
            return Err(FitError::Invalid(
                "design/assay observation IDs or ordering differ",
            ));
        }
        let context = ExecutionContext::new(MemoryBudget::new(options.working_bytes));
        let mut reader = context.open(assay.source());
        run_blocks_with_options(self.plan, &mut reader, options, |range, summary| {
            sink(&assay.feature_ids()[range], summary)
        })
    }
}
