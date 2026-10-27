use crate::{Fit, FitError, FitView, LinearModel, Result};
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Reusable numeric buffers for up to `capacity()` features. Results borrow
/// these buffers until the next fit. Backend packing/allocator scratch is extra.
#[derive(Debug)]
pub struct FitWorkspace {
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) capacity: usize,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) fitted: Vec<f64>,
    pub(crate) variance: Vec<f64>,
    pub(crate) transformed: Vec<f64>,
}
impl FitWorkspace {
    pub(crate) fn required_bytes(n: usize, p: usize, capacity: usize, gls: bool) -> Result<usize> {
        if n == 0 || p == 0 || capacity == 0 {
            return Err(FitError::Invalid("zero workspace dimension"));
        }
        n.checked_mul(if gls { 2 } else { 1 })
            .and_then(|v| v.checked_add(p))
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_mul(capacity))
            .and_then(|v| v.checked_mul(8))
            .filter(|&v| v <= isize::MAX as usize)
            .ok_or(FitError::Invalid("workspace byte overflow"))
    }
    pub(crate) fn new(n: usize, p: usize, capacity: usize, gls: bool) -> Result<Self> {
        Self::required_bytes(n, p, capacity, gls)?;
        fn zeros(len: usize) -> Result<Vec<f64>> {
            let mut v = Vec::new();
            v.try_reserve_exact(len)
                .map_err(|_| FitError::Invalid("workspace allocation failed"))?;
            v.resize(len, 0.);
            Ok(v)
        }
        Ok(Self {
            n,
            p,
            capacity,
            coefficients: zeros(p * capacity)?,
            fitted: zeros(n * capacity)?,
            variance: zeros(capacity)?,
            transformed: zeros(if gls { n * capacity } else { 0 })?,
        })
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Numeric buffer sizes; excludes backend scratch and allocation overhead.
    pub fn working_bytes(&self) -> usize {
        (self.coefficients.len() + self.fitted.len() + self.variance.len() + self.transformed.len())
            * 8
    }
    pub(crate) fn check(&self, n: usize, p: usize, b: usize, gls: bool) -> Result<()> {
        if n != self.n
            || p != self.p
            || b == 0
            || b > self.capacity
            || (gls && self.transformed.is_empty())
        {
            return Err(FitError::Invalid(
                "workspace dimensions, capacity or GLS buffer",
            ));
        }
        Ok(())
    }
    pub(crate) fn view<'m, 'w>(&'w self, model: &'m LinearModel, b: usize) -> FitView<'m, 'w> {
        FitView {
            model,
            coefficients: ArrayView2::from_shape((self.p, b), &self.coefficients[..self.p * b])
                .unwrap(),
            variance: ArrayView1::from(&self.variance[..b]),
            df: model.residual_df(),
        }
    }
    pub(crate) fn into_fit(mut self, model: &LinearModel, b: usize) -> Fit<'_> {
        self.coefficients.truncate(self.p * b);
        self.variance.truncate(b);
        Fit {
            model,
            coefficients: Array2::from_shape_vec((self.p, b), self.coefficients).unwrap(),
            variance: self.variance.into(),
            df: model.residual_df(),
        }
    }
}
