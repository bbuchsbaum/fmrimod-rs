//! Transformation basis functions.
//!
//! This module provides various transformation functions that can be used
//! as basis functions for parametric modulators in fMRI design matrices.

use crate::types::{Array, BasisFunction, ValidationError, Vector};
use ndarray::{Array1, Array2, ArrayView1};
use std::fmt::Debug;

/// Standard scaling transformation (z-score).
///
/// Centers and scales values to have mean 0 and standard deviation 1.
#[derive(Debug, Clone)]
pub struct Scale {
    /// Whether to center (subtract mean).
    pub center: bool,
    /// Whether to scale (divide by standard deviation).
    pub scale: bool,
    /// Name for the transformation.
    pub name: String,
    /// Computed mean (after evaluation).
    mean: Option<f64>,
    /// Computed standard deviation (after evaluation).
    std: Option<f64>,
}

impl Scale {
    /// Create a new Scale transformation.
    pub fn new(center: bool, scale: bool) -> Self {
        Self {
            center,
            scale,
            name: "scale".to_string(),
            mean: None,
            std: None,
        }
    }

    /// Create with custom name.
    pub fn with_name(center: bool, scale: bool, name: &str) -> Self {
        Self {
            center,
            scale,
            name: name.to_string(),
            mean: None,
            std: None,
        }
    }

    /// Get the computed mean (after evaluation).
    pub fn mean(&self) -> Option<f64> {
        self.mean
    }

    /// Get the computed standard deviation (after evaluation).
    pub fn std(&self) -> Option<f64> {
        self.std
    }

    /// Compatibility alias used by older tests/examples.
    pub fn design_matrix(&self, x: &Array1<f64>) -> Result<Array, ValidationError> {
        self.evaluate(&x.view())
    }
}

impl BasisFunction for Scale {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if x.is_empty() {
            return Err(ValidationError::EmptyArray);
        }

        let n = x.len();
        let mut result = Array2::zeros((n, 1));

        // Copy input values
        for (i, &val) in x.iter().enumerate() {
            result[[i, 0]] = val;
        }

        // Compute statistics
        let mean = x.sum() / n as f64;
        let variance = x.mapv(|v| (v - mean).powi(2)).sum() / n as f64;
        let std = variance.sqrt();

        // Store statistics (requires mutable self in real implementation)
        // For now, we'll compute inline

        // Apply transformations
        if self.center {
            for i in 0..n {
                result[[i, 0]] -= mean;
            }
        }

        if self.scale && std > 0.0 {
            for i in 0..n {
                result[[i, 0]] /= std;
            }
        }

        Ok(result)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        let basis = self.evaluate(x)?;
        if let Some(c) = coef {
            if c.len() != self.nbasis() {
                return Err(ValidationError::LengthMismatch {
                    expected: self.nbasis(),
                    actual: c.len(),
                });
            }
            Ok(basis.dot(c))
        } else {
            Ok(basis.column(0).to_owned())
        }
    }
}

/// Robust scaling transformation.
///
/// Centers at median and scales by MAD (median absolute deviation)
/// or IQR (interquartile range).
#[derive(Debug, Clone)]
pub struct RobustScale {
    /// Whether to center at median.
    pub center: bool,
    /// Scaling method.
    pub scale_method: RobustScaleMethod,
    /// Constant for MAD scaling (1.4826 for normal distribution).
    pub constant: f64,
    /// Name for the transformation.
    pub name: String,
    /// Computed median (after evaluation).
    median: Option<f64>,
    /// Computed scale factor (after evaluation).
    scale_factor: Option<f64>,
}

/// Scaling method for robust scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustScaleMethod {
    /// Median Absolute Deviation.
    Mad,
    /// Interquartile Range.
    Iqr,
}

impl RobustScale {
    /// Create a new RobustScale transformation.
    pub fn new(center: bool, scale_method: RobustScaleMethod) -> Self {
        Self {
            center,
            scale_method,
            constant: 1.4826, // Standard constant for normal distribution
            name: "robustscale".to_string(),
            median: None,
            scale_factor: None,
        }
    }

    /// Create with custom constant.
    pub fn with_constant(center: bool, scale_method: RobustScaleMethod, constant: f64) -> Self {
        Self {
            center,
            scale_method,
            constant,
            name: "robustscale".to_string(),
            median: None,
            scale_factor: None,
        }
    }

    /// Get the computed median (after evaluation).
    pub fn median(&self) -> Option<f64> {
        self.median
    }

    /// Get the computed scale factor (after evaluation).
    pub fn scale_factor(&self) -> Option<f64> {
        self.scale_factor
    }

    /// Compatibility alias used by older tests/examples.
    pub fn design_matrix(&self, x: &Array1<f64>) -> Result<Array, ValidationError> {
        self.evaluate(&x.view())
    }
}

impl BasisFunction for RobustScale {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if x.is_empty() {
            return Err(ValidationError::EmptyArray);
        }

        let n = x.len();
        let mut result = Array2::zeros((n, 1));

        // Copy and sort for median calculation
        let mut sorted: Vec<f64> = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute median
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // Compute scale factor based on method
        let scale_factor = match self.scale_method {
            RobustScaleMethod::Mad => {
                // Median absolute deviation
                let mut deviations: Vec<f64> = x.iter().map(|&v| (v - median).abs()).collect();
                deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mad = if n % 2 == 0 {
                    (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
                } else {
                    deviations[n / 2]
                };

                mad * self.constant
            }
            RobustScaleMethod::Iqr => {
                // Interquartile range
                let q1_idx = n / 4;
                let q3_idx = (3 * n) / 4;

                let q1 = if n % 4 == 0 {
                    (sorted[q1_idx - 1] + sorted[q1_idx]) / 2.0
                } else {
                    sorted[q1_idx]
                };

                let q3 = if (3 * n) % 4 == 0 {
                    (sorted[q3_idx - 1] + sorted[q3_idx]) / 2.0
                } else {
                    sorted[q3_idx]
                };

                q3 - q1
            }
        };

        // Apply transformations
        for (i, &val) in x.iter().enumerate() {
            result[[i, 0]] = val;

            if self.center {
                result[[i, 0]] -= median;
            }

            if scale_factor > 0.0 {
                result[[i, 0]] /= scale_factor;
            }
        }

        Ok(result)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        let basis = self.evaluate(x)?;
        if let Some(c) = coef {
            if c.len() != self.nbasis() {
                return Err(ValidationError::LengthMismatch {
                    expected: self.nbasis(),
                    actual: c.len(),
                });
            }
            Ok(basis.dot(c))
        } else {
            Ok(basis.column(0).to_owned())
        }
    }
}

/// Identity transformation (no change).
///
/// Returns input values unchanged. Useful as a placeholder
/// or for explicit no-transformation in pipelines.
#[derive(Debug, Clone)]
pub struct Ident {
    /// Name for the transformation.
    pub name: String,
}

impl Ident {
    /// Create a new identity transformation.
    pub fn new() -> Self {
        Self {
            name: "ident".to_string(),
        }
    }

    /// Create with custom name.
    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl Default for Ident {
    fn default() -> Self {
        Self::new()
    }
}

impl BasisFunction for Ident {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if x.is_empty() {
            return Err(ValidationError::EmptyArray);
        }

        let n = x.len();
        let mut result = Array2::zeros((n, 1));

        for (i, &val) in x.iter().enumerate() {
            result[[i, 0]] = val;
        }

        Ok(result)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        let basis = self.evaluate(x)?;
        if let Some(c) = coef {
            if c.len() != self.nbasis() {
                return Err(ValidationError::LengthMismatch {
                    expected: self.nbasis(),
                    actual: c.len(),
                });
            }
            Ok(basis.dot(c))
        } else {
            Ok(basis.column(0).to_owned())
        }
    }
}

/// Group-wise scaling transformation.
///
/// Centers and scales values within each group separately.
#[derive(Debug, Clone)]
pub struct ScaleWithin {
    /// Group indicators for each observation.
    pub groups: Vec<String>,
    /// Whether to center within groups.
    pub center: bool,
    /// Whether to scale within groups.
    pub scale: bool,
    /// Name for the transformation.
    pub name: String,
    /// Group statistics (after evaluation).
    group_stats: Option<std::collections::HashMap<String, GroupStats>>,
}

/// Statistics for a group in ScaleWithin transformation.
#[derive(Debug, Clone)]
pub struct GroupStats {
    /// Group mean
    pub mean: f64,
    /// Group standard deviation
    pub std: f64,
    /// Number of observations in group
    pub n: usize,
}

impl ScaleWithin {
    /// Create a new ScaleWithin transformation.
    pub fn new(groups: Vec<String>, center: bool, scale: bool) -> Self {
        Self {
            groups,
            center,
            scale,
            name: "scalewithin".to_string(),
            group_stats: None,
        }
    }

    /// Get group statistics (after evaluation).
    pub fn group_stats(&self) -> Option<&std::collections::HashMap<String, GroupStats>> {
        self.group_stats.as_ref()
    }
}

impl BasisFunction for ScaleWithin {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if x.is_empty() {
            return Err(ValidationError::EmptyArray);
        }

        if x.len() != self.groups.len() {
            return Err(ValidationError::LengthMismatch {
                expected: self.groups.len(),
                actual: x.len(),
            });
        }

        let n = x.len();
        let mut result = Array2::zeros((n, 1));

        // Collect unique groups
        let mut unique_groups: Vec<String> = self.groups.clone();
        unique_groups.sort();
        unique_groups.dedup();

        // Process each group
        let mut group_stats = std::collections::HashMap::new();

        for group in &unique_groups {
            // Find indices for this group
            let indices: Vec<usize> = self
                .groups
                .iter()
                .enumerate()
                .filter(|(_, g)| *g == group)
                .map(|(i, _)| i)
                .collect();

            if indices.is_empty() {
                continue;
            }

            // Collect group data
            let group_data: Vec<f64> = indices.iter().map(|&i| x[i]).collect();

            // Compute group statistics
            let n_group = group_data.len() as f64;
            let mean = group_data.iter().sum::<f64>() / n_group;
            let variance = group_data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_group;
            let std = variance.sqrt();

            group_stats.insert(
                group.clone(),
                GroupStats {
                    mean,
                    std,
                    n: group_data.len(),
                },
            );

            // Transform group data
            for (j, &idx) in indices.iter().enumerate() {
                let mut val = group_data[j];

                if self.center {
                    val -= mean;
                }

                if self.scale && std > 0.0 {
                    val /= std;
                }

                result[[idx, 0]] = val;
            }
        }

        Ok(result)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        let basis = self.evaluate(x)?;
        if let Some(c) = coef {
            if c.len() != self.nbasis() {
                return Err(ValidationError::LengthMismatch {
                    expected: self.nbasis(),
                    actual: c.len(),
                });
            }
            Ok(basis.dot(c))
        } else {
            Ok(basis.column(0).to_owned())
        }
    }
}

/// Standardized transformation with fit/transform pattern.
///
/// Similar to Scale but allows fitting statistics on one dataset
/// and applying to another.
#[derive(Debug, Clone)]
pub struct Standardized {
    /// Whether to center (subtract mean).
    pub center: bool,
    /// Whether to scale (divide by standard deviation).
    pub scale: bool,
    /// Name for the transformation.
    pub name: String,
    /// Fitted mean.
    mean: Option<f64>,
    /// Fitted standard deviation.
    std: Option<f64>,
}

impl Standardized {
    /// Create a new Standardized transformation.
    pub fn new(center: bool, scale: bool) -> Self {
        Self {
            center,
            scale,
            name: "standardized".to_string(),
            mean: None,
            std: None,
        }
    }

    /// Fit the standardization parameters.
    pub fn fit(&mut self, x: &[f64]) -> Result<(), ValidationError> {
        if x.is_empty() {
            return Err(ValidationError::EmptyArray);
        }

        let n = x.len() as f64;

        if self.center {
            self.mean = Some(x.iter().sum::<f64>() / n);
        } else {
            self.mean = Some(0.0);
        }

        if self.scale {
            let mean = self.mean.unwrap();
            let variance = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();

            self.std = Some(if std == 0.0 { 1.0 } else { std });
        } else {
            self.std = Some(1.0);
        }

        Ok(())
    }

    /// Check if the transformer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
    }

    /// Get the fitted mean.
    pub fn fitted_mean(&self) -> Option<f64> {
        self.mean
    }

    /// Get the fitted standard deviation.
    pub fn fitted_std(&self) -> Option<f64> {
        self.std
    }
}

impl BasisFunction for Standardized {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if x.is_empty() {
            return Err(ValidationError::EmptyArray);
        }

        let n = x.len();
        let mut result = Array2::zeros((n, 1));

        // Use fitted parameters or fit on input
        let (mean, std) = if self.is_fitted() {
            (self.mean.unwrap(), self.std.unwrap())
        } else {
            // Auto-fit on input (note: this requires mutable self in practice)
            let n_f = n as f64;
            let mean = if self.center { x.sum() / n_f } else { 0.0 };

            let std = if self.scale {
                let variance = x.mapv(|v| (v - mean).powi(2)).sum() / n_f;
                let s = variance.sqrt();
                if s == 0.0 {
                    1.0
                } else {
                    s
                }
            } else {
                1.0
            };

            (mean, std)
        };

        // Apply transformation
        for (i, &val) in x.iter().enumerate() {
            result[[i, 0]] = (val - mean) / std;
        }

        Ok(result)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        let basis = self.evaluate(x)?;
        if let Some(c) = coef {
            if c.len() != self.nbasis() {
                return Err(ValidationError::LengthMismatch {
                    expected: self.nbasis(),
                    actual: c.len(),
                });
            }
            Ok(basis.dot(c))
        } else {
            Ok(basis.column(0).to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_scale_transform() {
        let transform = Scale::new(true, true);
        let x_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = Array1::from(x_vec);
        let result = transform.evaluate(&x.view()).unwrap();

        // Check shape
        assert_eq!(result.shape(), &[5, 1]);

        // Check mean is approximately 0
        let mean = result.column(0).mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);

        // Check std is approximately 1
        let variance = result.column(0).mapv(|v| v.powi(2)).mean().unwrap();
        assert_relative_eq!(variance.sqrt(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_center_only() {
        let transform = Scale::new(true, false);
        let x_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = Array1::from(x_vec);
        let result = transform.evaluate(&x.view()).unwrap();

        // Check mean is 0
        let mean = result.column(0).mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);

        // Check std is preserved (should be sqrt(2))
        let std = ((result.column(0).mapv(|v| v.powi(2)).sum()) / 5.0).sqrt();
        assert_relative_eq!(std, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_robust_scale_mad() {
        let transform = RobustScale::new(true, RobustScaleMethod::Mad);
        let x_vec = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // 100 is an outlier
        let x = Array1::from(x_vec);
        let result = transform.evaluate(&x.view()).unwrap();

        // The median should be 3.0
        // MAD should be based on deviations from median
        // Result should be less affected by the outlier
        assert_eq!(result.shape(), &[5, 1]);
    }

    #[test]
    fn test_ident_transform() {
        let transform = Ident::new();
        let x_vec = vec![1.0, 2.0, 3.0];
        let x = Array1::from(x_vec);
        let result = transform.evaluate(&x.view()).unwrap();

        assert_eq!(result.shape(), &[3, 1]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 0]], 2.0);
        assert_eq!(result[[2, 0]], 3.0);
    }

    #[test]
    fn test_scale_within_groups() {
        let groups = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let transform = ScaleWithin::new(groups, true, true);
        let x_vec = vec![1.0, 2.0, 10.0, 20.0, 30.0];
        let x = Array1::from(x_vec);
        let result = transform.evaluate(&x.view()).unwrap();

        assert_eq!(result.shape(), &[5, 1]);

        // Group A values should be standardized
        // Mean of 1.5, std of 0.5
        assert_relative_eq!(result[[0, 0]], -1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0]], 1.0, epsilon = 1e-10);

        // Group B values should be standardized
        // Mean of 20, std should make them -1, 0, 1 approximately
    }

    #[test]
    fn test_standardized_fit_transform() {
        let mut transform = Standardized::new(true, true);
        let x_train = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Fit on training data
        transform.fit(&x_train).unwrap();
        assert!(transform.is_fitted());

        // Transform new data
        let x_test_vec = vec![3.0, 6.0];
        let x_test = Array1::from(x_test_vec);
        let result = transform.evaluate(&x_test.view()).unwrap();

        assert_eq!(result.shape(), &[2, 1]);
        // 3.0 should map to 0 (the mean)
        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);
    }
}
