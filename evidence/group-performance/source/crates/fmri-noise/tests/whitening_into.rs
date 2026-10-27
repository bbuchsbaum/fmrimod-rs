use fmri_noise::{Ar1Plan, ArPlan, ArStart, FirstSample, NoiseError, Result, WhiteningPlan};
use ndarray::{array, s, Array2, ArrayView2};

fn close(actual: ArrayView2<'_, f64>, expected: ArrayView2<'_, f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 2e-12 * (1.0 + expected.abs()),
            "{actual} != {expected}"
        );
    }
}

// This dense matrix is constructed directly from the AR(1) definition, rather
// than by using either production whitening method.
fn dense_ar1(
    values: ArrayView2<'_, f64>,
    phi: &[f64],
    run_lengths: &[usize],
    resets: &[usize],
    first: FirstSample,
) -> Array2<f64> {
    let mut w = Array2::<f64>::zeros((values.nrows(), values.nrows()));
    let mut start = 0;
    for (&length, &coefficient) in run_lengths.iter().zip(phi) {
        for t in start..start + length {
            if t == start || resets.contains(&t) {
                w[(t, t)] = match first {
                    FirstSample::Stationary => (1.0 - coefficient * coefficient).sqrt(),
                    FirstSample::ZeroHistory => 1.0,
                };
            } else {
                w[(t, t)] = 1.0;
                w[(t, t - 1)] = -coefficient;
            }
        }
        start += length;
    }
    w.dot(&values)
}

// Construct the conditional AR(p) whitening matrix directly. This does not use
// the production plan's startup/filter implementation.
fn dense_zero_history_arp(
    values: ArrayView2<'_, f64>,
    coefficients: &[Vec<f64>],
    run_lengths: &[usize],
    resets: &[usize],
) -> Array2<f64> {
    let mut w = Array2::<f64>::zeros((values.nrows(), values.nrows()));
    let mut start = 0;
    for (&length, phi) in run_lengths.iter().zip(coefficients) {
        let mut segment = start;
        for t in start..start + length {
            if t == start || resets.contains(&t) {
                segment = t;
            }
            w[(t, t)] = 1.0;
            for k in 0..phi.len().min(t - segment) {
                w[(t, t - k - 1)] = -phi[k];
            }
        }
        start += length;
    }
    w.dot(&values)
}

#[test]
fn ar1_apply_into_reuses_strided_output_and_matches_dense_transform() {
    let source = Array2::from_shape_fn((6, 7), |(i, j)| (10 * i + j) as f64 - 14.0);
    let values = source.slice(s![..;2, 1..7;2]);
    let runs = vec![2, 1];
    let phi = vec![0.4, -0.3];
    let plan = Ar1Plan::new(runs.clone(), phi.clone(), FirstSample::Stationary).unwrap();
    let expected = dense_ar1(values, &phi, &runs, &[], FirstSample::Stationary);

    let mut storage = Array2::from_elem((6, 6), f64::NAN);
    let output = storage.slice_mut(s![..;2, ..;2]);
    plan.apply_into(values, output).unwrap();
    close(storage.slice(s![..;2, ..;2]), expected.view());

    storage.fill(-999.0);
    plan.apply_into(values, storage.slice_mut(s![..;2, ..;2]))
        .unwrap();
    close(storage.slice(s![..;2, ..;2]), expected.view());
}

#[test]
fn arp_apply_into_preserves_startup_run_and_reset_semantics() {
    let source = Array2::from_shape_fn((14, 6), |(i, j)| (i * i + 3 * j) as f64 / 7.0);
    let values = source.slice(s![1..13;2, ..;2]);
    let runs = vec![3, 3];
    let coefficients = vec![vec![0.45, -0.2], vec![-0.25]];
    let resets = vec![2, 4];
    for startup in [ArStart::ZeroHistory, ArStart::Stationary] {
        let plan = ArPlan::new(runs.clone(), coefficients.clone(), startup)
            .unwrap()
            .with_resets(&resets)
            .unwrap();
        let expected = match startup {
            ArStart::ZeroHistory => dense_zero_history_arp(values, &coefficients, &runs, &resets),
            ArStart::Stationary => plan.apply(values).unwrap(),
        };
        let mut storage = Array2::from_elem((12, 6), f64::NAN);
        plan.apply_into(values, storage.slice_mut(s![..;2, ..;2]))
            .unwrap();
        close(storage.slice(s![..;2, ..;2]), expected.view());
    }
}

#[test]
fn apply_into_rejects_invalid_dimensions_nonfinite_values_and_overflow() {
    let ar1 = Ar1Plan::new(vec![2], vec![0.5], FirstSample::ZeroHistory).unwrap();
    let values = array![[1.0], [2.0]];
    assert!(ar1
        .apply_into(values.view(), Array2::zeros((2, 2)).view_mut())
        .is_err());
    assert!(ar1
        .apply_into(
            array![[f64::NAN], [2.0]].view(),
            Array2::zeros((2, 1)).view_mut()
        )
        .is_err());
    assert!(ar1
        .apply_into(
            array![[f64::MAX], [-f64::MAX]].view(),
            Array2::zeros((2, 1)).view_mut(),
        )
        .is_err());

    let arp = ArPlan::new(vec![2], vec![vec![0.5]], ArStart::ZeroHistory).unwrap();
    assert!(arp
        .apply_into(values.view(), Array2::zeros((1, 1)).view_mut())
        .is_err());
    assert!(arp
        .apply_into(
            array![[f64::MAX], [-f64::MAX]].view(),
            Array2::zeros((2, 1)).view_mut(),
        )
        .is_err());
}

struct CopyPlan;
impl WhiteningPlan for CopyPlan {
    fn observations(&self) -> usize {
        2
    }
    fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if values.nrows() != self.observations() {
            return Err(NoiseError::Invalid("unexpected input"));
        }
        Ok(values.to_owned())
    }
}

struct WrongShapePlan;
impl WhiteningPlan for WrongShapePlan {
    fn observations(&self) -> usize {
        2
    }
    fn apply(&self, _: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Ok(Array2::zeros((1, 1)))
    }
}

#[test]
fn trait_default_apply_into_remains_available_to_custom_plans() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let mut output = Array2::zeros((2, 2));
    CopyPlan
        .apply_into(input.view(), output.view_mut())
        .unwrap();
    assert_eq!(output, input);
    assert!(CopyPlan
        .apply_into(input.view(), Array2::zeros((1, 2)).view_mut())
        .is_err());
    assert!(WrongShapePlan
        .apply_into(input.view(), Array2::zeros((2, 2)).view_mut())
        .is_err());
}
