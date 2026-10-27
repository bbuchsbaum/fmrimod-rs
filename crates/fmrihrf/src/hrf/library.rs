//! Pre-defined HRF library.

use crate::error::Result;
use crate::hrf::core::Hrf;
use crate::hrf::derivatives::{SpmG1Hrf, SpmG2Hrf, SpmG3Hrf};
use crate::hrf::functions::*;
use crate::hrf::generators::bind_hrfs;
use crate::hrf::registry::HrfFactory;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Lazy-initialized library of predefined HRFs.
static HRF_LIBRARY: OnceLock<HashMap<String, HrfFactory>> = OnceLock::new();

/// Initialize the HRF library with predefined functions.
fn init_library() -> HashMap<String, HrfFactory> {
    let mut library: HashMap<String, HrfFactory> = HashMap::new();

    // SPM Canonical HRF
    library.insert(
        "spm_canonical".to_string(),
        Box::new(|| spm_canonical(None, None, None, None, None, None)),
    );
    library.insert(
        "spm".to_string(),
        Box::new(|| spm_canonical(None, None, None, None, None, None)),
    );
    library.insert(
        "spm1".to_string(),
        Box::new(|| spm_canonical(None, None, None, None, None, None)),
    );
    library.insert(
        "spmg1".to_string(),
        Box::new(|| spm_canonical(None, None, None, None, None, None)),
    );

    // SPM canonical plus first temporal derivative (two columns).
    library.insert(
        "spm_with_derivative".to_string(),
        Box::new(|| Box::new(SpmG2Hrf::new(SPM_P1, SPM_P2, SPM_A1))),
    );
    library.insert(
        "spm2".to_string(),
        Box::new(|| Box::new(SpmG2Hrf::new(SPM_P1, SPM_P2, SPM_A1))),
    );
    library.insert(
        "spmg2".to_string(),
        Box::new(|| Box::new(SpmG2Hrf::new(SPM_P1, SPM_P2, SPM_A1))),
    );

    // SPM with temporal and dispersion derivatives (same function, all 3 basis)
    library.insert(
        "spm_with_dispersion".to_string(),
        Box::new(|| spm_hrf_with_derivatives()),
    );
    library.insert("spm3".to_string(), Box::new(|| spm_hrf_with_derivatives()));
    library.insert("spmg3".to_string(), Box::new(|| spm_hrf_with_derivatives()));

    // Gamma HRF
    library.insert(
        "gamma".to_string(),
        Box::new(|| gamma_hrf(6.0, 1.0, Some(false))),
    );
    library.insert(
        "gam".to_string(),
        Box::new(|| gamma_hrf(6.0, 1.0, Some(false))),
    );

    // Gaussian HRF
    library.insert(
        "gaussian".to_string(),
        Box::new(|| gaussian_hrf(6.0, 2.0, Some(false))),
    );
    library.insert(
        "gauss".to_string(),
        Box::new(|| gaussian_hrf(6.0, 2.0, Some(false))),
    );

    // B-spline basis HRF
    library.insert(
        "bspline".to_string(),
        Box::new(|| bspline_hrf(5, None, None)),
    );
    library.insert("bs".to_string(), Box::new(|| bspline_hrf(5, None, None)));

    // Fourier basis HRF
    library.insert("fourier".to_string(), Box::new(|| fourier_hrf(5, None)));

    // FIR basis HRF
    library.insert("fir".to_string(), Box::new(|| fir_hrf(10, None)));
    library.insert("tent".to_string(), Box::new(|| tent_hrf(5, Some(24.0))));

    // Mexican Hat HRF
    library.insert("mexhat".to_string(), Box::new(|| mexhat_hrf(2.0, 6.0)));
    library.insert("mexican_hat".to_string(), Box::new(|| mexhat_hrf(2.0, 6.0)));

    // Sine HRF
    library.insert("sine".to_string(), Box::new(|| sine_basis_hrf(5, 24.0)));

    // Boxcar HRF
    library.insert(
        "boxcar".to_string(),
        Box::new(|| boxcar_hrf(1.0, 1.0, false)),
    );

    // Half Cosine HRF
    library.insert(
        "half_cosine".to_string(),
        Box::new(|| half_cosine_hrf_segments(1.0, 5.0, 7.0, 7.0, 0.0, 0.0)),
    );
    library.insert(
        "halfcos".to_string(),
        Box::new(|| half_cosine_hrf_segments(1.0, 5.0, 7.0, 7.0, 0.0, 0.0)),
    );

    // Inverse Logit HRF
    library.insert(
        "inv_logit".to_string(),
        Box::new(|| inv_logit_hrf(6.0, 1.0, 16.0, 1.0, 0.0)),
    );
    library.insert(
        "inverse_logit".to_string(),
        Box::new(|| inv_logit_hrf(6.0, 1.0, 16.0, 1.0, 0.0)),
    );

    // LWU HRF
    library.insert(
        "lwu".to_string(),
        Box::new(|| lwu_hrf(None, None, None, None).unwrap()),
    );

    // LWU Basis HRF
    library.insert(
        "lwu_basis".to_string(),
        Box::new(|| hrf_basis_lwu(&[6.0, 2.5, 0.35], None).unwrap()),
    );

    // Time HRF
    library.insert("time".to_string(), Box::new(|| hrf_time(None)));

    // Daguerre HRF
    library.insert(
        "daguerre".to_string(),
        Box::new(|| {
            crate::hrf::generators::daguerre_generator(Some(3), Some(30.0), Some(4.0)).build()
        }),
    );
    library.insert(
        "dag".to_string(),
        Box::new(|| {
            crate::hrf::generators::daguerre_generator(Some(3), Some(30.0), Some(4.0)).build()
        }),
    );

    // SPM HRFs with analytic derivatives
    library.insert(
        "spmg1_analytic".to_string(),
        Box::new(|| Box::new(SpmG1Hrf::new(SPM_P1, SPM_P2, SPM_A1))),
    );
    library.insert(
        "spmg2_analytic".to_string(),
        Box::new(|| Box::new(SpmG2Hrf::new(SPM_P1, SPM_P2, SPM_A1))),
    );
    library.insert(
        "spmg3_analytic".to_string(),
        Box::new(|| Box::new(SpmG3Hrf::new(SPM_P1, SPM_P2, SPM_A1))),
    );

    library
}

/// Get a predefined HRF by name.
///
/// Available HRFs:
/// - "spm_canonical", "spm", "spm1", "spmg1": SPM canonical HRF
/// - "spm_with_derivative", "spm2", "spmg2": SPM with temporal derivative
/// - "spm_with_dispersion", "spm3", "spmg3": SPM with temporal and dispersion derivatives
/// - "gamma", "gam": Gamma HRF
/// - "gaussian", "gauss": Gaussian HRF
/// - "bspline", "bs": B-spline basis HRF
/// - "fourier": Fourier basis HRF
/// - "fir", "tent": FIR basis HRF
/// - "mexhat", "mexican_hat": Mexican hat HRF
/// - "sine": Sine HRF
/// - "half_cosine", "halfcos": Half cosine HRF
/// - "inv_logit", "inverse_logit": Inverse logit HRF
/// - "lwu": LWU HRF
/// - "lwu_basis": LWU basis HRF
/// - "time": Time HRF
pub fn get_predefined_hrf(name: &str) -> Option<Box<dyn Hrf>> {
    let library = HRF_LIBRARY.get_or_init(init_library);
    library.get(name).map(|factory| factory())
}

/// List all available predefined HRF names.
pub fn list_predefined_hrfs() -> Vec<String> {
    let library = HRF_LIBRARY.get_or_init(init_library);
    let mut names: Vec<String> = library.keys().cloned().collect();
    names.sort();
    names
}

/// Function that generates HRFs with specific parameters.
pub type HrfGeneratorFn = Box<dyn Fn(&HashMap<String, f64>) -> Result<Box<dyn Hrf>>>;

/// Generate an HRF library from a parameter grid.
///
/// # Arguments
/// * `generator_fn` - Function that creates HRFs from parameters
/// * `param_grid` - Grid of parameters to vary
///
/// # Example
/// ```
/// use fmrihrf::hrf::functions::gamma_hrf;
/// use fmrihrf::hrf::library::hrf_library;
/// use std::collections::HashMap;
///
/// // Create library of gamma HRFs with varying shape parameters
/// let mut param_grid = HashMap::new();
/// param_grid.insert("shape".to_string(), vec![4.0, 6.0, 8.0]);
/// param_grid.insert("rate".to_string(), vec![1.0]);
///
/// let generator_fn = Box::new(|params: &HashMap<String, f64>| {
///     let shape = params.get("shape").copied().unwrap_or(6.0);
///     let rate = params.get("rate").copied().unwrap_or(1.0);
///     Ok(gamma_hrf(shape, rate, None))
/// });
///
/// let gamma_lib = hrf_library(generator_fn, &param_grid).unwrap();
/// assert_eq!(gamma_lib.nbasis(), 3);
/// ```
pub fn hrf_library(
    generator_fn: HrfGeneratorFn,
    param_grid: &HashMap<String, Vec<f64>>,
) -> Result<Box<dyn Hrf>> {
    // Get all parameter names and values
    let param_names: Vec<String> = param_grid.keys().cloned().collect();
    let param_values: Vec<&Vec<f64>> = param_grid.values().collect();

    // Generate all combinations
    let mut combinations = vec![vec![]];
    for values in param_values {
        let mut new_combinations = Vec::new();
        for combo in combinations {
            for &value in values {
                let mut new_combo = combo.clone();
                new_combo.push(value);
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
    }

    // Generate HRF for each combination
    let mut hrfs = Vec::new();
    for combo in combinations {
        // Build parameters for this combination
        let mut params = HashMap::new();
        for (i, name) in param_names.iter().enumerate() {
            params.insert(name.clone(), combo[i]);
        }

        // Generate HRF
        let hrf = generator_fn(&params)?;
        hrfs.push(hrf);
    }

    // Combine into single HRF
    if hrfs.len() == 1 {
        Ok(hrfs.into_iter().next().unwrap())
    } else {
        Ok(bind_hrfs(hrfs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_predefined_hrf() {
        // Test various aliases
        let spm1 = get_predefined_hrf("spm").unwrap();
        let spm2 = get_predefined_hrf("spm_canonical").unwrap();
        let spm3 = get_predefined_hrf("spmg1").unwrap();

        assert_eq!(spm1.nbasis(), 1);
        assert_eq!(spm2.nbasis(), 1);
        assert_eq!(spm3.nbasis(), 1);

        // Test HRF with derivatives (always returns 3 basis functions)
        let spm_deriv = get_predefined_hrf("spm2").unwrap();
        assert_eq!(spm_deriv.nbasis(), 2);

        let spm_disp = get_predefined_hrf("spm3").unwrap();
        assert_eq!(spm_disp.nbasis(), 3);

        // Test other HRFs
        assert!(get_predefined_hrf("gamma").is_some());
        assert!(get_predefined_hrf("gaussian").is_some());
        assert!(get_predefined_hrf("bspline").is_some());
        assert!(get_predefined_hrf("lwu").is_some());
        assert!(get_predefined_hrf("time").is_some());

        // Test non-existent HRF
        assert!(get_predefined_hrf("nonexistent").is_none());
    }

    #[test]
    fn test_list_predefined_hrfs() {
        let names = list_predefined_hrfs();

        // Check that we have a reasonable number of HRFs
        assert!(names.len() > 20);

        // Check that some expected names are present
        assert!(names.contains(&"spm".to_string()));
        assert!(names.contains(&"gamma".to_string()));
        assert!(names.contains(&"gaussian".to_string()));
        assert!(names.contains(&"lwu".to_string()));

        // Check that names are sorted
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_hrf_library() {
        // Create parameter grid
        let mut param_grid = HashMap::new();
        param_grid.insert("shape".to_string(), vec![4.0, 6.0, 8.0]);
        param_grid.insert("rate".to_string(), vec![0.9, 1.0]);

        // Create generator function
        let generator_fn = Box::new(|params: &HashMap<String, f64>| {
            let shape = params.get("shape").copied().unwrap_or(6.0);
            let rate = params.get("rate").copied().unwrap_or(1.0);
            Ok(gamma_hrf(shape, rate, Some(true)))
        });

        // Generate library
        let gamma_lib = hrf_library(generator_fn, &param_grid).unwrap();

        // Should have 3 * 2 = 6 basis functions
        assert_eq!(gamma_lib.nbasis(), 6);

        // Test evaluation
        let t = ndarray::Array1::linspace(0.0, 20.0, 100);
        let values = gamma_lib.evaluate(&t).unwrap();
        assert_eq!(values.shape(), &[100, 6]);

        // All values should be non-negative for gamma HRF
        assert!(values.iter().all(|&v| v >= 0.0));
    }
}
