//! Exact known-parameter AR whitening on a subset of the original observations.
use crate::{ArStart, NoiseError, Result, WhiteningPlan};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

/// Validated original run lengths and strictly increasing retained row indices.
#[derive(Debug, Clone)]
pub struct ObservationSelection {
    lengths: Vec<usize>,
    rows: Vec<usize>,
    total: usize,
}
impl ObservationSelection {
    pub fn new(lengths: Vec<usize>, rows: Vec<usize>) -> Result<Self> {
        let total = lengths
            .iter()
            .try_fold(0usize, |n, &r| n.checked_add(r))
            .ok_or(NoiseError::Invalid("run length overflow"))?;
        if lengths.is_empty()
            || lengths.contains(&0)
            || rows.is_empty()
            || rows.iter().any(|&r| r >= total)
            || rows.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(NoiseError::Invalid(
                "positive original runs and nonempty increasing unique retained rows required",
            ));
        }
        Ok(Self {
            lengths,
            rows,
            total,
        })
    }
    pub fn original_run_lengths(&self) -> &[usize] {
        &self.lengths
    }
    pub fn retained_indices(&self) -> &[usize] {
        &self.rows
    }
    pub fn original_observations(&self) -> usize {
        self.total
    }
    /// Select original rows before whitening. Excluded values are never inspected.
    pub fn select(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if values.nrows() != self.total || values.ncols() == 0 {
            return Err(NoiseError::Invalid("original matrix dimensions"));
        }
        let result = values.select(Axis(0), &self.rows);
        if result.iter().any(|v| !v.is_finite()) {
            return Err(NoiseError::Invalid("nonfinite retained values"));
        }
        Ok(result)
    }
}
/// Covariance model for gaps between retained observations within a run.
#[derive(Debug, Clone, Copy)]
pub enum GapPolicy {
    /// Marginalize missing observations; retain the original run's covariance.
    Preserve,
    /// Start an independent segment at the first retained row and after each gap.
    Restart,
}
#[derive(Debug, Clone)]
struct Step {
    transition: DMatrix<f64>,
    gain: DVector<f64>,
    scale: f64,
    reset: bool,
}
/// Prepared innovations transform with fixed coefficients and unit innovation
/// variance. Only retained rows enter the fit. Applies the identical transform to
/// X and Y via GlsModel. No AR coefficient estimation or implicit order reduction.
/// Storage is O(retained_rows * p²); application uses O(p) scratch per feature.
#[derive(Debug, Clone)]
pub struct SelectedArPlan {
    selection: ObservationSelection,
    steps: Vec<Step>,
}
impl SelectedArPlan {
    pub fn new(
        selection: ObservationSelection,
        coefficients: Vec<Vec<f64>>,
        startup: ArStart,
        gaps: GapPolicy,
    ) -> Result<Self> {
        if coefficients.len() != selection.lengths.len() {
            return Err(NoiseError::Invalid(
                "one AR coefficient vector per original run required",
            ));
        }
        let mut steps = Vec::with_capacity(selection.rows.len());
        let (mut base, mut selected) = (0, 0);
        for (&len, phi) in selection.lengths.iter().zip(&coefficients) {
            crate::ar::stationary(phi)?;
            let p = phi.len().max(1);
            let mut transition = DMatrix::zeros(p, p);
            for (j, &v) in phi.iter().enumerate() {
                transition[(0, j)] = v;
            }
            for i in 1..p {
                transition[(i, i - 1)] = 1.;
            }
            let initial = match startup {
                ArStart::Stationary => {
                    let gamma = crate::ar::unit_acvf(phi)?;
                    DMatrix::from_fn(p, p, |i, j| gamma[i.abs_diff(j)])
                }
                ArStart::ZeroHistory => {
                    let mut m = DMatrix::zeros(p, p);
                    m[(0, 0)] = 1.;
                    m
                }
            };
            let mut covariance = initial.clone();
            let mut elapsed = DMatrix::identity(p, p);
            let mut previous = None;
            for local in 0..len {
                let row = base + local;
                if local > 0 {
                    covariance = &transition * &covariance * transition.transpose();
                    covariance[(0, 0)] += 1.;
                    elapsed = &transition * elapsed;
                }
                if selection.rows.get(selected) != Some(&row) {
                    continue;
                }
                let reset = previous.is_none()
                    || (matches!(gaps, GapPolicy::Restart) && previous != Some(local - 1));
                if reset && matches!(gaps, GapPolicy::Restart) {
                    covariance = initial.clone();
                }
                let variance = covariance[(0, 0)];
                if !variance.is_finite() || variance <= 0. {
                    return Err(NoiseError::Invalid("invalid selected innovation variance"));
                }
                let gain = covariance.column(0).into_owned() / variance;
                let scale = 1. / variance.sqrt();
                if covariance
                    .iter()
                    .chain(elapsed.iter())
                    .chain(gain.iter())
                    .any(|v| !v.is_finite())
                    || !scale.is_finite()
                {
                    return Err(NoiseError::Invalid("selected AR preparation overflow"));
                }
                steps.push(Step {
                    transition: elapsed.clone(),
                    gain: gain.clone(),
                    scale,
                    reset,
                });
                covariance -= (&gain * gain.transpose()) * variance;
                // The observed state coordinate is known exactly; remove roundoff.
                covariance.row_mut(0).fill(0.);
                covariance.column_mut(0).fill(0.);
                covariance = (&covariance + covariance.transpose()) * 0.5;
                elapsed = DMatrix::identity(p, p);
                previous = Some(local);
                selected += 1;
            }
            base += len;
        }
        Ok(Self { selection, steps })
    }
    pub fn selection(&self) -> &ObservationSelection {
        &self.selection
    }
}
impl WhiteningPlan for SelectedArPlan {
    fn observations(&self) -> usize {
        self.steps.len()
    }
    fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if values.nrows() != self.observations() || values.ncols() == 0 {
            return Err(NoiseError::Invalid("selected matrix dimensions"));
        }
        let mut out = Array2::zeros(values.dim());
        self.apply_into(values, out.view_mut())?;
        Ok(out)
    }
    fn apply_into(
        &self,
        values: ArrayView2<'_, f64>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<()> {
        if values.nrows() != self.observations()
            || values.ncols() == 0
            || out.dim() != values.dim()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(NoiseError::Invalid(
                "selected matrix dimensions or nonfinite values",
            ));
        }
        let max_p = self.steps.iter().map(|s| s.gain.len()).max().unwrap_or(0);
        let mut state = vec![0.; max_p];
        let mut prediction = vec![0.; max_p];
        for feature in 0..values.ncols() {
            state.fill(0.);
            for (row, step) in self.steps.iter().enumerate() {
                let p = step.gain.len();
                if step.reset {
                    state.fill(0.);
                }
                for i in 0..p {
                    prediction[i] = (0..p).map(|j| step.transition[(i, j)] * state[j]).sum();
                }
                let innovation = values[(row, feature)] - prediction[0];
                out[(row, feature)] = innovation * step.scale;
                for i in 0..p {
                    state[i] = prediction[i] + step.gain[i] * innovation;
                }
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(NoiseError::Invalid("selected AR whitening overflow"));
        }
        Ok(())
    }
}
