//! Continuous sampled features encoded as zero-order-hold bins.
use super::{ConvolutionMethod, EventTable};
use crate::{Hrf, HrfError, Result, SamplingFrame};
use ndarray::{Array2, ArrayView2};

/// Run-local sample times; each sample holds until the next sample.
#[derive(Debug, Clone, Copy)]
pub enum FeatureGrid<'a> {
    /// Uniform bins, including the final bin, with explicit start and interval.
    Uniform {
        /// First sample onset in seconds from this run's start.
        start: f64,
        /// Positive bin width in seconds.
        step: f64,
    },
    /// Strictly increasing onsets; the last bin repeats the last gap.
    Times(&'a [f64]),
}

/// Borrowed input for one run. Features are columns; samples are rows.
#[derive(Debug, Clone, Copy)]
pub struct FeatureRun<'a> {
    /// Finite feature values, including off-mask samples.
    pub values: ArrayView2<'a, f64>,
    /// Run-local timing, independent of the acquisition TR.
    pub grid: FeatureGrid<'a>,
    /// Statistics use true rows only; false rows become zero before convolution.
    pub mask: Option<&'a [bool]>,
}

/// Scaling of feature samples before convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureScale {
    /// Preserve native units.
    None,
    /// Divide by sample SD (denominator n-1), computed on active samples.
    SampleSd,
}

/// Preprocessing applies independently to each run and feature column.
#[derive(Debug, Clone, Copy)]
pub struct FeatureOptions {
    /// Subtract the active-sample mean before convolution.
    pub center: bool,
    /// Scaling choice; no scaling is the default.
    pub scale: FeatureScale,
}
impl Default for FeatureOptions {
    fn default() -> Self {
        Self {
            center: true,
            scale: FeatureScale::None,
        }
    }
}

/// Actual preprocessing constants for one run.
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// Number of samples included by the mask.
    pub active_samples: usize,
    /// Active-sample mean for each feature, even when centering is disabled.
    pub means: Vec<f64>,
    /// Divisor applied to each feature (one when scaling is disabled).
    pub divisors: Vec<f64>,
}

/// Prepared, owned features with reusable preprocessing and run-local timing.
///
/// Zeros remain in the sample grid. No companion presence/intercept column is
/// added. Output columns are basis-major, then feature-major within each basis.
/// Feature means are sample-weighted, not duration-weighted, including irregular grids.
#[derive(Debug, Clone)]
pub struct SampledFeatures {
    onsets: Vec<f64>,
    widths: Vec<f64>,
    run_ids: Vec<usize>,
    amplitudes: Array2<f64>,
    statistics: Vec<FeatureStatistics>,
    minimum_width: f64,
}
fn invalid(message: &str) -> HrfError {
    HrfError::InvalidInput(message.into())
}
impl SampledFeatures {
    /// Prepare one or more nonempty runs with the same feature count.
    /// Empty masks, constant features requested with SD scaling, nonfinite data,
    /// duplicate times, and unrepresentable bin endpoints are errors.
    pub fn new(runs: &[FeatureRun<'_>], options: FeatureOptions) -> Result<Self> {
        let columns = runs
            .first()
            .ok_or_else(|| invalid("at least one feature run required"))?
            .values
            .ncols();
        if columns == 0 {
            return Err(invalid("at least one feature column required"));
        }
        let rows = runs.iter().try_fold(0usize, |n, r| {
            n.checked_add(r.values.nrows())
                .ok_or_else(|| invalid("feature row count overflow"))
        })?;
        rows.checked_mul(columns)
            .ok_or_else(|| invalid("feature matrix size overflow"))?;
        let mut out = Self {
            onsets: Vec::with_capacity(rows),
            widths: Vec::with_capacity(rows),
            run_ids: Vec::with_capacity(rows),
            amplitudes: Array2::zeros((rows, columns)),
            statistics: Vec::with_capacity(runs.len()),
            minimum_width: f64::INFINITY,
        };
        let mut offset = 0;
        for (run_id, run) in runs.iter().enumerate() {
            let n = run.values.nrows();
            if n == 0
                || run.values.ncols() != columns
                || run.values.iter().any(|v| !v.is_finite())
                || run.mask.is_some_and(|m| m.len() != n)
            {
                return Err(invalid("nonempty finite feature matrices with matching columns and mask lengths required"));
            }
            let active: Vec<_> = (0..n).filter(|&i| run.mask.is_none_or(|m| m[i])).collect();
            if active.is_empty() {
                return Err(invalid(
                    "feature mask must retain at least one sample per run",
                ));
            }
            let (times, widths) = match run.grid {
                FeatureGrid::Uniform { start, step } => {
                    if !start.is_finite() || start < 0. || !step.is_finite() || step <= 0. {
                        return Err(invalid("invalid uniform feature start or interval"));
                    }
                    (
                        (0..n).map(|i| start + i as f64 * step).collect::<Vec<_>>(),
                        vec![step; n],
                    )
                }
                FeatureGrid::Times(t) => {
                    if t.len() != n || n < 2 {
                        return Err(invalid("explicit feature times require matching length and at least two samples"));
                    }
                    let mut w: Vec<_> = t.windows(2).map(|w| w[1] - w[0]).collect();
                    w.push(w[w.len() - 1]);
                    (t.to_vec(), w)
                }
            };
            if times.iter().any(|t| !t.is_finite() || *t < 0.)
                || times.windows(2).any(|w| w[1] <= w[0])
                || times
                    .iter()
                    .zip(&widths)
                    .any(|(t, w)| !w.is_finite() || *w <= 0. || !(t + w).is_finite() || t + w <= *t)
            {
                return Err(invalid(
                    "feature times and bin endpoints must be finite, increasing and nonnegative",
                ));
            }
            let mut stats = FeatureStatistics {
                active_samples: active.len(),
                means: Vec::with_capacity(columns),
                divisors: Vec::with_capacity(columns),
            };
            for j in 0..columns {
                // Scale first to avoid overflow in means and sums of squares.
                let magnitude = active
                    .iter()
                    .map(|&i| run.values[(i, j)].abs())
                    .fold(0.0_f64, f64::max)
                    .max(f64::MIN_POSITIVE);
                let mean = active
                    .iter()
                    .map(|&i| run.values[(i, j)] / magnitude)
                    .sum::<f64>()
                    / active.len() as f64;
                let sd = if options.scale == FeatureScale::SampleSd {
                    if active.len() < 2 {
                        return Err(invalid("sample SD needs at least two active samples"));
                    }
                    let variance = active
                        .iter()
                        .map(|&i| (run.values[(i, j)] / magnitude - mean).powi(2))
                        .sum::<f64>()
                        / (active.len() - 1) as f64;
                    let sd = variance.sqrt();
                    if !sd.is_finite()
                        || sd <= 0.
                        || !(sd * magnitude).is_finite()
                        || sd * magnitude <= 0.
                    {
                        return Err(invalid("sample SD must be positive and finite"));
                    }
                    sd
                } else {
                    1.
                };
                stats.means.push(mean * magnitude);
                stats
                    .divisors
                    .push(if options.scale == FeatureScale::SampleSd {
                        sd * magnitude
                    } else {
                        1.
                    });
                for &i in &active {
                    let value =
                        run.values[(i, j)] / magnitude - if options.center { mean } else { 0. };
                    let amplitude = if options.scale == FeatureScale::SampleSd {
                        value / sd
                    } else if !options.center {
                        run.values[(i, j)]
                    } else {
                        value * magnitude
                    };
                    if !amplitude.is_finite() {
                        return Err(invalid("feature preprocessing overflow"));
                    }
                    out.amplitudes[(offset + i, j)] = amplitude;
                }
            }
            out.minimum_width = widths.iter().copied().fold(out.minimum_width, f64::min);
            out.onsets.extend(times);
            out.widths.extend(widths);
            out.run_ids.extend(std::iter::repeat_n(run_id, n));
            out.statistics.push(stats);
            offset += n;
        }
        Ok(out)
    }
    /// Prepared amplitudes, including zero off-mask samples.
    pub fn amplitudes(&self) -> ArrayView2<'_, f64> {
        self.amplitudes.view()
    }
    /// Runwise preprocessing diagnostics.
    pub fn statistics(&self) -> &[FeatureStatistics] {
        &self.statistics
    }
    /// Borrow the equivalent duration-bearing event table without copying samples.
    pub fn as_events(&self) -> EventTable<'_> {
        EventTable {
            onsets: &self.onsets,
            durations: &self.widths,
            amplitudes: self.amplitudes.view(),
            run_ids: &self.run_ids,
        }
    }
    /// Evaluate on an acquisition frame. Runs are matched by position and never
    /// exchange HRF tails. Integration precision must not exceed the shortest bin.
    /// The HRF's own span defines support; there is no hidden 40-second override.
    pub fn evaluate(
        &self,
        frame: &SamplingFrame,
        hrf: &dyn Hrf,
        method: ConvolutionMethod,
    ) -> Result<Array2<f64>> {
        if frame.n_blocks() != self.statistics.len() {
            return Err(invalid("feature and acquisition run counts differ"));
        }
        if frame.precision > self.minimum_width * (1.0 + 32.0 * f64::EPSILON) {
            return Err(invalid(
                "integration precision exceeds the shortest feature bin",
            ));
        }
        self.as_events().evaluate(frame, hrf, method)
    }
}
