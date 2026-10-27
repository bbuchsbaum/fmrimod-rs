#![cfg(feature = "dataset")]
use fmri_data::{ArrayReader, DType, MemoryBudget, NumericBlock, ReadCost, ReadSelection, Values};
use fmri_noise::{ArPlan, ArStart};
use fmri_reg::{
    dataset::BlockOptions,
    output::{EstimandSet, OutputSpec, SummaryView, Uncertainty},
    output_dataset::{run_blocks_with_options, BoundOutputs},
    FitError, GlsModel, LinearModel,
};
use ndarray::{array, Array2};
use std::{cell::Cell, rc::Rc};
struct Reader {
    reads: Rc<Cell<usize>>,
    fail_at: Option<usize>,
    nonfinite: bool,
}
impl ArrayReader for Reader {
    fn shape(&self) -> [usize; 2] {
        [24, 17]
    }
    fn dtype(&self) -> DType {
        DType::F64
    }
    fn estimate(&self, selection: &ReadSelection) -> fmri_data::Result<ReadCost> {
        let b = selection.shape()[1] * 24 * 8;
        Ok(ReadCost {
            output_bytes: b,
            peak_bytes: b + 128,
        })
    }
    fn read(&mut self, s: &ReadSelection, budget: MemoryBudget) -> fmri_data::Result<NumericBlock> {
        budget.check(self.estimate(s)?.peak_bytes)?;
        let read = self.reads.get();
        self.reads.set(read + 1);
        if self.fail_at == Some(read) {
            budget.check(usize::MAX)?;
        }
        let mut values = Vec::new();
        for i in 0..24 {
            for j in 0..s.shape()[1] {
                let global = match s.columns() {
                    fmri_data::Selection::Range(r) => r.start + j,
                    _ => panic!("expected range"),
                };
                values.push(value(i, global));
            }
        }
        if self.nonfinite && read == 1 {
            values[0] = f64::NAN;
        }
        NumericBlock::new(s.shape(), Values::F64(values))
    }
}
fn value(i: usize, j: usize) -> f64 {
    1. + i as f64 * 0.3 + j as f64 + ((i + 3 * j) as f64).sin()
}
fn model() -> GlsModel<ArPlan> {
    let x = Array2::from_shape_fn((24, 2), |(i, j)| if j == 0 { 1. } else { i as f64 / 24. });
    GlsModel::new(
        x.view(),
        ArPlan::new(
            vec![12, 12],
            vec![vec![0.4, -0.1], vec![0.2]],
            ArStart::Stationary,
        )
        .unwrap()
        .with_resets(&[4, 15])
        .unwrap(),
        1e-12,
    )
    .unwrap()
}
fn options(workers: usize) -> BlockOptions {
    BlockOptions {
        block_features: 4,
        workers,
        working_bytes: 100_000,
    }
}

fn spec(level: Uncertainty) -> OutputSpec {
    OutputSpec::new(
        EstimandSet::contrasts(
            vec!["slope".into(), "double".into()],
            array![[0., 1.], [0., 2.]],
        )
        .unwrap(),
    )
    .with_uncertainty(level)
}
#[test]
fn ordered_summary_blocks_match_dense_and_survive_models() {
    let mut retained = Vec::new();
    {
        let model = model();
        let y = Array2::from_shape_fn((24, 17), |(i, j)| value(i, j));
        for level in [Uncertainty::None, Uncertainty::Marginal, Uncertainty::Joint] {
            let plan = model.prepare_outputs(&spec(level)).unwrap();
            let mut ws = plan.workspace(17).unwrap();
            let expected = plan.run_into(y.view(), &mut ws).unwrap();
            for workers in [1, 3, 100] {
                let mut reader = Reader {
                    reads: Rc::new(Cell::new(0)),
                    fail_at: None,
                    nonfinite: false,
                };
                let mut seen = 0;
                run_blocks_with_options(&plan, &mut reader, options(workers), |range, summary| {
                    assert_eq!(range.start, seen);
                    for (j, global) in range.clone().enumerate() {
                        for i in 0..2 {
                            assert!(
                                (summary.estimates()[(i, j)] - expected.estimates()[(i, global)])
                                    .abs()
                                    < 1e-11
                            );
                        }
                        match (summary, &expected) {
                            (SummaryView::Marginal(a), SummaryView::Marginal(b)) => assert!(
                                (a.variance(0, j).unwrap() - b.variance(0, global).unwrap()).abs()
                                    < 1e-11
                            ),
                            (SummaryView::Joint(a), SummaryView::Joint(b)) => assert!(
                                (a.covariance(0, 1, j).unwrap()
                                    - b.covariance(0, 1, global).unwrap())
                                .abs()
                                    < 1e-11
                            ),
                            _ => {}
                        }
                    }
                    seen = range.end;
                    retained.push(summary.to_owned());
                    Ok(())
                })
                .unwrap();
                assert_eq!(seen, 17);
            }
        }
    }
    assert_eq!(retained.len(), 45);
    assert!(retained
        .iter()
        .all(|s| s.estimates().iter().all(|v| v.is_finite())));
}
#[test]
fn budgets_preflight_before_reads_and_errors_stop_sinks() {
    let model = model();
    for level in [Uncertainty::None, Uncertainty::Marginal, Uncertainty::Joint] {
        let plan = model.prepare_outputs(&spec(level)).unwrap();
        let reads = Rc::new(Cell::new(0));
        let mut reader = Reader {
            reads: reads.clone(),
            fail_at: None,
            nonfinite: false,
        };
        let mut opts = options(1);
        opts.working_bytes = plan.workspace_bytes(4).unwrap() + 24 * 4 * 8 + 128;
        run_blocks_with_options(&plan, &mut reader, opts, |_, _| Ok(())).unwrap();
        assert_eq!(reads.get(), 5);
        reads.set(0);
        opts.working_bytes -= 1;
        assert!(
            run_blocks_with_options(&plan, &mut reader, opts, |_, _| panic!("budget")).is_err()
        );
        assert_eq!(reads.get(), 0);
        for workers in [1, 3] {
            let mut calls = 0;
            let mut reader = Reader {
                reads: Rc::new(Cell::new(0)),
                fail_at: None,
                nonfinite: false,
            };
            assert!(
                run_blocks_with_options(&plan, &mut reader, options(workers), |_, _| {
                    calls += 1;
                    Err(FitError::Invalid("sink"))
                })
                .is_err()
            );
            assert_eq!(calls, 1);
            for (fail_at, nonfinite) in [(Some(1), false), (None, true)] {
                let mut reader = Reader {
                    reads: Rc::new(Cell::new(0)),
                    fail_at,
                    nonfinite,
                };
                assert!(
                    run_blocks_with_options(&plan, &mut reader, options(workers), |_, _| Ok(()))
                        .is_err()
                );
            }
        }
    }
}
#[test]
fn summary_assay_checks_identity_and_feature_order() {
    use fmri_data::{Assay, AssayAnnotations, FeatureId, ObservationId, SourceSpec};
    let x = Array2::ones((24, 1));
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let ids: Vec<_> = (0..24)
        .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
        .collect();
    let features: Vec<_> = (0..17)
        .map(|i| FeatureId::new(format!("voxel-{i}")).unwrap())
        .collect();
    let source = SourceSpec::memory(
        NumericBlock::new(
            [24, 17],
            Values::F64(
                (0..24)
                    .flat_map(|i| (0..17).map(move |j| value(i, j)))
                    .collect(),
            ),
        )
        .unwrap(),
    );
    let assay = Assay::new(
        "bold",
        source,
        ids.clone(),
        features.clone(),
        AssayAnnotations::default(),
    )
    .unwrap();
    let request =
        OutputSpec::new(EstimandSet::coefficients(vec!["mean".into()], vec![0], 1).unwrap());
    let plan = model.prepare_outputs(&request).unwrap();
    let bound = BoundOutputs::new(&plan, ids.clone()).unwrap();
    let mut seen = Vec::new();
    bound
        .run_assay(&assay, options(3), |ids, _| {
            seen.extend_from_slice(ids);
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, features);
    let mut wrong = ids;
    wrong.swap(0, 1);
    assert!(BoundOutputs::new(&plan, wrong)
        .unwrap()
        .run_assay(&assay, options(3), |_, _| panic!("ID mismatch"))
        .is_err());
}
