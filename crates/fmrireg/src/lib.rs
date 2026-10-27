//! Factor a design once, then fit borrowed time-by-feature blocks.
//!
//! SVD is used once at preparation to determine numerical rank. Block fitting
//! uses matrix products and retains only coefficients and residual variances.
//! Singular designs are allowed; contrasts must lie in the estimable row space.
use nalgebra::DMatrix;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
mod backend;
mod finite;
pub mod output;
#[cfg(feature = "dataset")]
pub mod output_dataset;
mod workspace;
pub use backend::MatrixBackend;
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};
use thiserror::Error;
pub use workspace::FitWorkspace;

#[derive(Debug, Error)]
pub enum FitError {
    #[error("invalid fit input: {0}")]
    Invalid(&'static str),
    #[error("contrast is not estimable at the design's numerical rank")]
    NotEstimable,
    #[error("contrast variance is zero or nonfinite")]
    DegenerateVariance,
    #[error(transparent)]
    Noise(#[from] fmri_noise::NoiseError),
}
pub type Result<T> = std::result::Result<T, FitError>;

/// Prepared least-squares problem. Rank threshold is relative to largest singular value.
#[derive(Debug)]
pub struct LinearModel {
    design: Array2<f64>,
    inverse: Array2<f64>,
    covariance: Array2<f64>,
    row_space: Array2<f64>,
    rank: usize,
    tolerance: f64,
    backend: MatrixBackend,
}

/// Compact results. Variance is RSS / (observations - numerical rank).
#[derive(Debug)]
pub struct Fit<'a> {
    model: &'a LinearModel,
    coefficients: Array2<f64>,
    variance: Array1<f64>,
    df: usize,
}
impl Fit<'_> {
    pub fn coefficients(&self) -> &Array2<f64> {
        &self.coefficients
    }
    pub fn residual_variance(&self) -> &Array1<f64> {
        &self.variance
    }
    pub fn residual_df(&self) -> usize {
        self.df
    }
}

/// Borrowed fit results backed by a reusable workspace.
#[derive(Debug, Clone, Copy)]
pub struct FitView<'m, 'w> {
    model: &'m LinearModel,
    coefficients: ArrayView2<'w, f64>,
    variance: ArrayView1<'w, f64>,
    df: usize,
}
impl<'m, 'w> FitView<'m, 'w> {
    pub fn coefficients(&self) -> ArrayView2<'w, f64> {
        self.coefficients
    }
    pub fn residual_variance(&self) -> ArrayView1<'w, f64> {
        self.variance
    }
    pub fn residual_df(&self) -> usize {
        self.df
    }
    /// Copy compact outputs when the caller needs to retain them past the next fit.
    pub fn to_owned(&self) -> Fit<'m> {
        Fit {
            model: self.model,
            coefficients: self.coefficients.to_owned(),
            variance: self.variance.to_owned(),
            df: self.df,
        }
    }
}

#[derive(Debug)]
pub struct TContrast {
    pub effect: Array1<f64>,
    pub standard_error: Array1<f64>,
    pub statistic: Array1<f64>,
    pub p_value: Array1<f64>,
    pub df: usize,
}
#[derive(Debug)]
pub struct FContrast {
    pub statistic: Array1<f64>,
    pub p_value: Array1<f64>,
    pub numerator_df: usize,
    pub denominator_df: usize,
}

impl LinearModel {
    /// Prepare a design; 1e-12 is a useful relative tolerance for well-scaled f64 designs.
    pub fn new(design: ArrayView2<'_, f64>, relative_tolerance: f64) -> Result<Self> {
        let (n, p) = design.dim();
        if n == 0
            || p == 0
            || !relative_tolerance.is_finite()
            || relative_tolerance <= 0.0
            || relative_tolerance >= 1.0
            || design.iter().any(|x| !x.is_finite())
        {
            return Err(FitError::Invalid(
                "empty/nonfinite design or invalid rank tolerance",
            ));
        }
        let matrix = DMatrix::from_fn(n, p, |i, j| design[(i, j)]);
        let svd = matrix.svd(true, true);
        let u = svd.u.ok_or(FitError::Invalid("SVD did not produce U"))?;
        let vt = svd.v_t.ok_or(FitError::Invalid("SVD did not produce V"))?;
        let cutoff = svd.singular_values[0] * relative_tolerance;
        let retained: Vec<usize> = svd
            .singular_values
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > cutoff)
            .map(|(i, _)| i)
            .collect();
        let rank = retained.len();
        if rank == 0 || rank >= n {
            return Err(FitError::Invalid(
                "no residual degrees of freedom or zero rank",
            ));
        }
        let mut inverse = Array2::zeros((p, n));
        let mut covariance = Array2::zeros((p, p));
        let mut row_space = Array2::zeros((p, p));
        for k in retained {
            let s = svd.singular_values[k];
            for i in 0..p {
                for j in 0..n {
                    inverse[(i, j)] += vt[(k, i)] * u[(j, k)] / s;
                }
                for j in 0..p {
                    covariance[(i, j)] += (vt[(k, i)] / s) * (vt[(k, j)] / s);
                    row_space[(i, j)] += vt[(k, i)] * vt[(k, j)];
                }
            }
        }
        if inverse
            .iter()
            .chain(covariance.iter())
            .any(|x| !f64::is_finite(*x))
        {
            return Err(FitError::Invalid("factorization overflow; rescale design"));
        }
        Ok(Self {
            design: design.to_owned(),
            inverse,
            covariance,
            row_space,
            rank,
            tolerance: relative_tolerance.max(64.0 * f64::EPSILON),
            backend: MatrixBackend::default(),
        })
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn residual_df(&self) -> usize {
        self.design.nrows() - self.rank
    }
    pub fn design(&self) -> &Array2<f64> {
        &self.design
    }
    pub fn unit_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }

    /// Select the fitting backend; preparation and rank policy are unchanged.
    pub fn with_backend(mut self, backend: MatrixBackend) -> Self {
        self.backend = backend;
        self
    }
    pub fn backend(&self) -> MatrixBackend {
        self.backend
    }
    pub fn workspace(&self, max_features: usize) -> Result<FitWorkspace> {
        FitWorkspace::new(
            self.design.nrows(),
            self.design.ncols(),
            max_features,
            false,
        )
    }
    pub fn fit(&self, data: ArrayView2<'_, f64>) -> Result<Fit<'_>> {
        self.validate_data(data)?;
        let mut workspace = self.workspace(data.ncols())?;
        self.fit_buffers_validated(
            data,
            &mut workspace.coefficients,
            &mut workspace.fitted,
            &mut workspace.variance,
        )?;
        Ok(workspace.into_fit(self, data.ncols()))
    }
    /// Fit without allocating numeric outputs. Smaller tail blocks reuse the same
    /// buffers with contiguous layout. Backend packing scratch may still allocate.
    pub fn fit_into<'m, 'w>(
        &'m self,
        data: ArrayView2<'_, f64>,
        workspace: &'w mut FitWorkspace,
    ) -> Result<FitView<'m, 'w>> {
        workspace.check(
            self.design.nrows(),
            self.design.ncols(),
            data.ncols(),
            false,
        )?;
        self.fit_buffers(
            data,
            &mut workspace.coefficients,
            &mut workspace.fitted,
            &mut workspace.variance,
        )?;
        Ok(workspace.view(self, data.ncols()))
    }
    fn validate_shape(&self, data: ArrayView2<'_, f64>) -> Result<()> {
        if data.nrows() != self.design.nrows() || data.ncols() == 0 {
            return Err(FitError::Invalid("data dimensions"));
        }
        Ok(())
    }
    fn validate_data(&self, data: ArrayView2<'_, f64>) -> Result<()> {
        self.validate_shape(data)?;
        if !finite::all_finite(data) {
            return Err(FitError::Invalid("nonfinite data"));
        }
        Ok(())
    }
    fn fit_buffers(
        &self,
        data: ArrayView2<'_, f64>,
        coefficients: &mut [f64],
        fitted: &mut [f64],
        variance: &mut [f64],
    ) -> Result<()> {
        self.validate_data(data)?;
        self.fit_buffers_validated(data, coefficients, fitted, variance)
    }
    fn fit_buffers_validated(
        &self,
        data: ArrayView2<'_, f64>,
        coefficients: &mut [f64],
        fitted: &mut [f64],
        variance: &mut [f64],
    ) -> Result<()> {
        let (n, b) = data.dim();
        let p = self.design.ncols();
        let mut coefficients =
            ArrayViewMut2::from_shape((p, b), &mut coefficients[..p * b]).unwrap();
        let mut fitted = ArrayViewMut2::from_shape((n, b), &mut fitted[..n * b]).unwrap();
        self.backend
            .multiply(self.inverse.view(), data, coefficients.view_mut());
        self.backend
            .multiply(self.design.view(), coefficients.view(), fitted.view_mut());
        let mut variance = ArrayViewMut1::from(&mut variance[..b]);
        variance.fill(0.);
        // Form explicit residuals before squaring, in one contiguous pass. The
        // difference of squared norms would lose precision near perfect fits.
        for (prediction, observed) in fitted.rows().into_iter().zip(data.rows()) {
            ndarray::Zip::from(&mut variance)
                .and(prediction)
                .and(observed)
                .for_each(|sum, &f, &y| {
                    let r = f - y;
                    *sum += r * r;
                });
        }
        variance /= self.residual_df() as f64;
        if coefficients
            .iter()
            .chain(variance.iter())
            .any(|x| !x.is_finite())
        {
            return Err(FitError::Invalid("fit overflow; rescale inputs"));
        }
        Ok(())
    }
    fn check_contrast(&self, c: ArrayView1<'_, f64>) -> Result<()> {
        if c.len() != self.design.ncols() || c.iter().any(|x| !x.is_finite()) {
            return Err(FitError::Invalid(
                "contrast dimensions or nonfinite weights",
            ));
        }
        let scale = c.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if scale == 0.0 {
            return Err(FitError::Invalid("zero contrast"));
        }
        let normalized = &c / scale;
        let projection = self.row_space.dot(&normalized);
        let error = (&projection - &normalized).mapv(|x| x * x).sum().sqrt();
        if error > self.tolerance * 8.0 * normalized.dot(&normalized).sqrt() {
            return Err(FitError::NotEstimable);
        }
        Ok(())
    }
    pub fn t_contrast(
        &self,
        data: ArrayView2<'_, f64>,
        c: ArrayView1<'_, f64>,
    ) -> Result<TContrast> {
        self.fit(data)?.t_contrast(c)
    }
    pub fn f_contrast(
        &self,
        data: ArrayView2<'_, f64>,
        c: ArrayView2<'_, f64>,
    ) -> Result<FContrast> {
        self.fit(data)?.f_contrast(c)
    }
}

impl<'m> Fit<'m> {
    pub fn as_view(&self) -> FitView<'m, '_> {
        FitView {
            model: self.model,
            coefficients: self.coefficients.view(),
            variance: self.variance.view(),
            df: self.df,
        }
    }
    pub fn t_contrast(&self, c: ArrayView1<'_, f64>) -> Result<TContrast> {
        self.as_view().t_contrast(c)
    }
    pub fn f_contrast(&self, c: ArrayView2<'_, f64>) -> Result<FContrast> {
        self.as_view().f_contrast(c)
    }
}
impl FitView<'_, '_> {
    /// Evaluate a contrast without refitting; inference uses the original design factorization.
    pub fn t_contrast(&self, c: ArrayView1<'_, f64>) -> Result<TContrast> {
        self.model.check_contrast(c)?;
        let fit = self;
        let v = c.dot(&self.model.covariance.dot(&c));
        if !v.is_finite() || v <= 0.0 || fit.variance.iter().any(|s| *s <= 0.0) {
            return Err(FitError::DegenerateVariance);
        }
        let effect = c.dot(&fit.coefficients);
        let standard_error = fit.variance.mapv(|s| (s * v).sqrt());
        if standard_error.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(FitError::DegenerateVariance);
        }
        let statistic = &effect / &standard_error;
        if effect
            .iter()
            .chain(statistic.iter())
            .any(|v| !v.is_finite())
        {
            return Err(FitError::Invalid(
                "contrast overflow; rescale weights or inputs",
            ));
        }
        let dist = StudentsT::new(0.0, 1.0, fit.df as f64)
            .map_err(|_| FitError::Invalid("t degrees of freedom"))?;
        let p_value = statistic.mapv(|t| 2.0 * dist.sf(t.abs()));
        Ok(TContrast {
            effect,
            standard_error,
            statistic,
            p_value,
            df: fit.df,
        })
    }
    /// Joint test of independent estimable contrast rows. Redundant rows are rejected.
    pub fn f_contrast(&self, c: ArrayView2<'_, f64>) -> Result<FContrast> {
        if c.nrows() == 0 {
            return Err(FitError::Invalid("empty F contrast"));
        }
        for row in c.rows() {
            self.model.check_contrast(row)?;
        }
        let fit = self;
        if fit.variance.iter().any(|s| *s <= 0.0) {
            return Err(FitError::DegenerateVariance);
        }
        let q = c.nrows();
        let cov = c.dot(&self.model.covariance).dot(&c.t());
        let m = DMatrix::from_fn(q, q, |i, j| cov[(i, j)]);
        let eig = m.symmetric_eigen();
        let max = eig.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        if !max.is_finite()
            || eig
                .eigenvalues
                .iter()
                .any(|s| !s.is_finite() || *s <= max * self.model.tolerance)
        {
            return Err(FitError::Invalid(
                "F contrast rows are dependent or numerically singular",
            ));
        }
        let effects = c.dot(&fit.coefficients);
        let mut statistic = Array1::zeros(self.coefficients.ncols());
        for v in 0..self.coefficients.ncols() {
            let mut sum = 0.0;
            for k in 0..q {
                let z: f64 = (0..q)
                    .map(|i| eig.eigenvectors[(i, k)] * effects[(i, v)])
                    .sum();
                sum += z * z / eig.eigenvalues[k];
            }
            statistic[v] = sum / (q as f64 * fit.variance[v]);
        }
        let dist = FisherSnedecor::new(q as f64, fit.df as f64)
            .map_err(|_| FitError::Invalid("F degrees of freedom"))?;
        if statistic.iter().any(|f| !f.is_finite()) {
            return Err(FitError::Invalid(
                "F statistic overflow; rescale weights or inputs",
            ));
        }
        let p_value = statistic.mapv(|f| dist.sf(f));
        Ok(FContrast {
            statistic,
            p_value,
            numerator_df: q,
            denominator_df: fit.df,
        })
    }
}

/// Known-parameter GLS with one run-aware transform shared by design and data.
pub struct GlsModel<P = fmri_noise::Ar1Plan> {
    noise: P,
    model: LinearModel,
}
impl<P: fmri_noise::WhiteningPlan> GlsModel<P> {
    pub fn new(design: ArrayView2<'_, f64>, noise: P, tolerance: f64) -> Result<Self> {
        let transformed = noise.apply(design)?;
        if transformed.dim() != design.dim() {
            return Err(FitError::Invalid("whitening must preserve matrix shape"));
        }
        Ok(Self {
            model: LinearModel::new(transformed.view(), tolerance)?,
            noise,
        })
    }
    pub fn with_backend(mut self, backend: MatrixBackend) -> Self {
        self.model = self.model.with_backend(backend);
        self
    }
    pub fn backend(&self) -> MatrixBackend {
        self.model.backend()
    }
    pub fn workspace(&self, max_features: usize) -> Result<FitWorkspace> {
        FitWorkspace::new(
            self.model.design.nrows(),
            self.model.design.ncols(),
            max_features,
            true,
        )
    }
    pub fn fit(&self, data: ArrayView2<'_, f64>) -> Result<Fit<'_>> {
        // Preserve the custom-plan contract and reject invalid built-in inputs
        // before allocating fitting buffers.
        let transformed = self.noise.apply(data)?;
        if transformed.dim() != data.dim() {
            return Err(FitError::Invalid("whitening must preserve matrix shape"));
        }
        self.model.fit(transformed.view())
    }
    pub fn fit_into<'m, 'w>(
        &'m self,
        data: ArrayView2<'_, f64>,
        workspace: &'w mut FitWorkspace,
    ) -> Result<FitView<'m, 'w>> {
        workspace.check(
            self.model.design.nrows(),
            self.model.design.ncols(),
            data.ncols(),
            true,
        )?;
        self.model.validate_shape(data)?;
        let (n, b) = data.dim();
        let mut transformed =
            ArrayViewMut2::from_shape((n, b), &mut workspace.transformed[..n * b]).unwrap();
        self.noise.apply_into(data, transformed.view_mut())?;
        self.model.fit_buffers(
            transformed.view(),
            &mut workspace.coefficients,
            &mut workspace.fitted,
            &mut workspace.variance,
        )?;
        Ok(workspace.view(&self.model, b))
    }
    pub fn t_contrast(
        &self,
        data: ArrayView2<'_, f64>,
        c: ArrayView1<'_, f64>,
    ) -> Result<TContrast> {
        self.fit(data)?.t_contrast(c)
    }
}

#[cfg(feature = "dataset")]
pub mod dataset;

/// Explicit choices for design-corrected, run-aware AR(1) noise estimation.
pub struct Ar1Options {
    pub run_lengths: Vec<usize>,
    /// False excludes a row from noise estimation and resets filtering after it.
    /// It does not remove that row from the final regression objective.
    pub noise_valid: Option<Vec<bool>>,
    pub correction_max_lag: usize,
    pub pooling: fmri_noise::Pooling,
}
impl GlsModel {
    /// Prepare one-step feasible GLS from OLS residuals of representative data.
    /// The caller may supply a deterministic feature subset for noise estimation;
    /// subsequent `fit` calls accept any feature blocks with these same time rows.
    pub fn estimate_ar1(
        design: ArrayView2<'_, f64>,
        noise_data: ArrayView2<'_, f64>,
        options: Ar1Options,
        tolerance: f64,
    ) -> Result<Self> {
        let initial = LinearModel::new(design, tolerance)?;
        let fit = initial.fit(noise_data)?;
        let residuals = &noise_data - &design.dot(fit.coefficients());
        let estimator = fmri_noise::Ar1Estimator::new(
            design,
            options.run_lengths,
            options.noise_valid.as_deref(),
            options.correction_max_lag,
            options.pooling,
        )?;
        let plan = estimator.fit(residuals.view())?;
        Self::new(design, plan, tolerance)
    }
}

impl<P: fmri_noise::WhiteningPlan> GlsModel<P> {
    pub fn noise_plan(&self) -> &P {
        &self.noise
    }
    /// Numerical rank of the whitened design, at the model's rank tolerance.
    pub fn rank(&self) -> usize {
        self.model.rank()
    }
    /// Original observation count minus the whitened design's numerical rank.
    pub fn residual_df(&self) -> usize {
        self.model.residual_df()
    }
}

/// Fixed-order AR settings; no automatic order reduction or IID fallback.
pub struct ArOptions {
    pub run_lengths: Vec<usize>,
    pub noise_valid: Option<Vec<bool>>,
    pub order: usize,
    pub correction_max_lag: usize,
    pub pooling: fmri_noise::Pooling,
    pub startup: fmri_noise::ArStart,
}
impl GlsModel<fmri_noise::ArPlan> {
    pub fn estimate_ar(
        design: ArrayView2<'_, f64>,
        noise_data: ArrayView2<'_, f64>,
        options: ArOptions,
        tolerance: f64,
    ) -> Result<Self> {
        let initial = LinearModel::new(design, tolerance)?;
        let fit = initial.fit(noise_data)?;
        let residuals = &noise_data - &design.dot(fit.coefficients());
        let estimator = fmri_noise::ArEstimator::new(
            design,
            options.run_lengths,
            options.noise_valid.as_deref(),
            options.order,
            options.correction_max_lag,
            options.pooling,
            options.startup,
        )?;
        Self::new(design, estimator.fit(residuals.view())?, tolerance)
    }
}
