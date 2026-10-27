#![cfg(feature = "dataset")]
use fmri_data::{ArrayReader, DType, MemoryBudget, NumericBlock, ReadCost, ReadSelection, Values};
use fmri_noise::{ArPlan, ArStart};
use fmri_reg::{
    dataset::{fit_blocks_with_options, BlockOptions, BoundModel},
    FitError, GlsModel, LinearModel,
};
use ndarray::Array2;
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
#[test]
fn ordered_parallel_gls_matches_dense_for_every_output_and_worker_count() {
    let model = model();
    let y = Array2::from_shape_fn((24, 17), |(i, j)| value(i, j));
    let expected = model.fit(y.view()).unwrap();
    for workers in [1, 2, 4, 100] {
        let reads = Rc::new(Cell::new(0)); // Reader is deliberately not Send.
        let mut reader = Reader {
            reads: reads.clone(),
            fail_at: None,
            nonfinite: false,
        };
        let mut seen = 0;
        fit_blocks_with_options(&model, &mut reader, options(workers), |range, fit| {
            assert_eq!(range.start, seen);
            for (j, global) in range.clone().enumerate() {
                for i in 0..2 {
                    assert!(
                        (fit.coefficients()[(i, j)] - expected.coefficients()[(i, global)]).abs()
                            < 1e-11
                    );
                }
                assert!(
                    (fit.residual_variance()[j] - expected.residual_variance()[global]).abs()
                        < 1e-12
                );
            }
            seen = range.end;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 17);
        assert_eq!(reads.get(), 5);
    }
}
#[test]
fn aggregate_budget_rejects_before_reading_and_exact_bound_passes() {
    let model = model();
    let reads = Rc::new(Cell::new(0));
    let mut reader = Reader {
        reads: reads.clone(),
        fail_at: None,
        nonfinite: false,
    };
    // Per worker: (2*T+P+1)*B*8 workspace plus T*B*8+128 reader peak.
    let exact = ((48 + 2 + 1) * 4 * 8 + 24 * 4 * 8 + 128) * 3;
    let mut opts = options(3);
    opts.working_bytes = exact - 1;
    assert!(fit_blocks_with_options(&model, &mut reader, opts, |_, _| panic!("no sink")).is_err());
    assert_eq!(reads.get(), 0);
    opts.working_bytes = exact;
    fit_blocks_with_options(&model, &mut reader, opts, |_, _| Ok(())).unwrap();
    assert_eq!(reads.get(), 5);
    opts.workers = 0;
    assert!(fit_blocks_with_options(&model, &mut reader, opts, |_, _| panic!("no sink")).is_err());
}
#[test]
fn errors_join_workers_without_further_sink_delivery() {
    let model = model();
    for workers in [1, 3] {
        let mut calls = 0;
        let mut reader = Reader {
            reads: Rc::new(Cell::new(0)),
            fail_at: None,
            nonfinite: false,
        };
        let result = fit_blocks_with_options(&model, &mut reader, options(workers), |_, _| {
            calls += 1;
            Err(FitError::Invalid("sink stopped"))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
        for (fail_at, nonfinite) in [(Some(1), false), (None, true)] {
            let mut reader = Reader {
                reads: Rc::new(Cell::new(0)),
                fail_at,
                nonfinite,
            };
            assert!(
                fit_blocks_with_options(&model, &mut reader, options(workers), |_, _| Ok(()))
                    .is_err()
            );
        }
    }
}
#[test]
fn parallel_assay_checks_identity_before_reading() {
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
    let bound = BoundModel::new(&model, ids.clone()).unwrap();
    let mut seen = Vec::new();
    bound
        .fit_assay_with_options(&assay, options(3), |ids, _| {
            seen.extend_from_slice(ids);
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, features);
    let mut wrong = ids;
    wrong.swap(0, 1);
    assert!(BoundModel::new(&model, wrong)
        .unwrap()
        .fit_assay_with_options(&assay, options(3), |_, _| panic!("ID mismatch"))
        .is_err());
}

#[test]
fn sinks_can_retain_owned_results_with_the_models_lifetime() {
    let model = model();
    let mut reader = Reader {
        reads: Rc::new(Cell::new(0)),
        fail_at: None,
        nonfinite: false,
    };
    let mut retained = Vec::new();
    fit_blocks_with_options(&model, &mut reader, options(3), |_, fit| {
        retained.push(fit.to_owned());
        Ok(())
    })
    .unwrap();
    assert_eq!(retained.len(), 5);
    let first = retained[0].coefficients()[(0, 0)];
    let mut workspace = model.workspace(4).unwrap();
    model
        .fit_into(Array2::ones((24, 4)).view(), &mut workspace)
        .unwrap();
    assert_eq!(retained[0].coefficients()[(0, 0)], first);
}
