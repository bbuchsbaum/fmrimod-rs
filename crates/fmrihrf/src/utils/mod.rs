//! Utility functions.

pub mod cache;
pub mod misc;

pub use cache::{
    cache_hrf, cached_hrf_eval, clear_hrf_cache, get_cache_stats, reset_cache_stats,
    set_cache_size, CacheStats,
};
pub use misc::{
    acquisition_onsets, apply_binary_op, find_peaks, hrf_toeplitz, is_zero, lerp_arrays,
    normalize_max, recycle_or_error, shift_regressor, single_trial_regressor, time_grid,
};
