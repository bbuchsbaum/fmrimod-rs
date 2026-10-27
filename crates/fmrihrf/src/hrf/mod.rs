//! Hemodynamic Response Function (HRF) module.

pub mod core;
pub mod recipe;
pub use recipe::HrfRecipe;
pub mod derivatives;
pub mod empirical;
pub mod functions;
pub mod generators;
pub mod library;
pub mod normalization;
pub use normalization::{normalize_hrf_with, Normalization};
pub mod registry;

pub use self::core::{hrf_from_coefficients, Hrf, HrfEvaluator, HrfExt, SimpleHrf, WeightedHrf};
pub use self::derivatives::{
    deriv, derivative_hrf, numeric_derivative, spmg1_derivative, spmg1_dispersion_derivative,
    spmg1_second_derivative, AnalyticDerivative, DerivativeMethod, SpmG1Hrf, SpmG2Hrf, SpmG3Hrf,
};
pub use self::empirical::{
    empirical_hrf, empirical_hrf_with_method, gen_empirical_hrf, InterpolationMethod,
};
pub use self::functions::{
    boxcar_hrf, bspline_hrf, fir_hrf, fourier_hrf, gamma_hrf, gaussian_hrf, half_cosine_hrf,
    half_cosine_hrf_segments, hrf_basis_lwu, hrf_time, inv_logit_hrf, lwu_hrf, mexhat_hrf,
    sine_basis_hrf, spm_canonical, spm_hrf_with_derivatives, tent_hrf, weighted_hrf,
    weighted_hrf_uniform, WeightedMethod,
};
pub use self::generators::{
    bind_hrfs, block_hrf, bspline_generator, daguerre_generator, fir_generator, fourier_generator,
    gamma_generator, gen_hrf, gen_hrf_blocked, gen_hrf_lagged, hrf_from_basis, lag_hrf, make_hrf,
    make_hrf_from_params, normalize_hrf, HrfGenerator,
};
pub use self::library::{get_predefined_hrf, hrf_library, list_predefined_hrfs};
pub use self::registry::{clear_registry, get_hrf, list_available_hrfs, register_hrf, remove_hrf};
