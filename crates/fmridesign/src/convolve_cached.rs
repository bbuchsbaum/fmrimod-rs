//! Cached convolution operations for improved performance.

use crate::cache::{cached_design_matrix, CacheKey};
use crate::convolve::{get_hrf_array, ConvolveOptions};
use crate::types::{EventSpec, ValidationError};
use fmrihrf::Hrf;
use ndarray::{Array1, Array2};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Cached convolution of event with HRF.
pub fn convolve_cached(
    event: &dyn EventSpec,
    hrf: &dyn Hrf,
    sampling_points: &Array1<f64>,
    options: &ConvolveOptions,
) -> Result<Array2<f64>, ValidationError> {
    let compute = || {
        crate::convolve::convolve_at_times(
            event,
            hrf,
            sampling_points,
            options.hrf_duration,
            options.precision,
            &options.method,
        )
    };
    // Custom or mutable HRFs cannot safely be identified by their display name.
    let Some(identity) = hrf.cache_identity() else {
        return compute();
    };
    let amplitudes = event.event_amplitudes()?;
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    amplitudes.dim().hash(&mut hasher);
    for &value in event
        .onsets()
        .iter()
        .chain(event.durations())
        .chain(amplitudes.iter())
    {
        value.to_bits().hash(&mut hasher);
    }
    options.hrf_duration.to_bits().hash(&mut hasher);
    options.precision.to_bits().hash(&mut hasher);
    options.method.hash(&mut hasher);
    cached_design_matrix(event.name(), sampling_points, hasher.finish(), compute)
}

/// Cache-aware event model that uses caching for design matrix generation.
pub struct CachedEventModel {
    events: Vec<Box<dyn EventSpec>>,
    sampling_frame: fmrihrf::SamplingFrame,
}

impl CachedEventModel {
    /// Create a new cached event model.
    pub fn new(events: Vec<Box<dyn EventSpec>>, sampling_frame: fmrihrf::SamplingFrame) -> Self {
        Self {
            events,
            sampling_frame,
        }
    }

    /// Generate design matrix with caching.
    pub fn generate_design_matrix(&self) -> Result<Array2<f64>, ValidationError> {
        let sampling_points_vec = self.sampling_frame.samples();
        let n_points = sampling_points_vec.len();
        let sampling_points = Array1::from_vec(sampling_points_vec);

        // Collect all columns
        let mut all_columns = Vec::new();

        for event in &self.events {
            // EventSpec permits custom interior state, so a name alone is not
            // a valid cross-model cache identity.
            let dm = event.design_matrix(&sampling_points)?;

            for col_idx in 0..dm.ncols() {
                all_columns.push(dm.column(col_idx).to_owned());
            }
        }

        // Combine columns into design matrix
        if all_columns.is_empty() {
            Ok(Array2::zeros((n_points, 0)))
        } else {
            let n_cols = all_columns.len();
            let mut design_matrix = Array2::zeros((n_points, n_cols));

            for (j, col) in all_columns.iter().enumerate() {
                design_matrix.column_mut(j).assign(col);
            }

            Ok(design_matrix)
        }
    }

    /// Clear caches related to this model.
    pub fn clear_caches(&self) {
        use crate::cache::GLOBAL_CACHE;

        // In a more sophisticated implementation, we could clear only
        // the cache entries related to this model's events
        GLOBAL_CACHE.design_matrix_cache.clear();
        GLOBAL_CACHE.hrf_cache.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats() -> crate::cache::CacheStats {
        use crate::cache::GLOBAL_CACHE;
        GLOBAL_CACHE.combined_stats()
    }
}

/// Precompute and cache HRF kernels for common configurations.
pub fn precompute_hrf_cache(
    hrf_specs: &[&str],
    sampling_rates: &[f64],
    durations: &[f64],
) -> Result<(), ValidationError> {
    use crate::cache::GLOBAL_CACHE;

    for &hrf_name in hrf_specs {
        let hrf = crate::hrf::get_hrf(hrf_name, &Default::default())?;

        for &sr in sampling_rates {
            for &duration in durations {
                let kernel = get_hrf_array(hrf.as_ref(), sr, duration)?;

                // Create a dummy event to cache the HRF
                let key = CacheKey::Custom(format!(
                    "hrf_kernel_{}_{:x}_{:x}",
                    hrf_name,
                    sr.to_bits(),
                    duration.to_bits()
                ));
                let size = std::mem::size_of::<f64>() * kernel.len();

                GLOBAL_CACHE.hrf_cache.insert(key, kernel, Some(size));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::GLOBAL_CACHE;
    use crate::events::EventVariable;

    #[test]
    fn test_cached_convolution() {
        // Clear caches first
        GLOBAL_CACHE.clear_all();

        // Create a simple event
        let event = EventVariable::new(
            "test_event",
            vec![1.0, 5.0, 10.0],
            vec![1.0, 2.0, 3.0],
            vec![0.5, 0.5, 0.5],
        )
        .unwrap();

        // Create HRF
        let hrf = fmrihrf::hrf::spm_canonical(None, None, None, None, None, Some(true));

        // Create sampling points
        let sampling_points = Array1::linspace(0.0, 20.0, 100);

        // First call should miss cache
        let stats_before = GLOBAL_CACHE.design_matrix_cache.stats();
        let result1 = convolve_cached(
            &event,
            hrf.as_ref(),
            &sampling_points,
            &ConvolveOptions::default(),
        )
        .unwrap();
        let stats_after1 = GLOBAL_CACHE.design_matrix_cache.stats();

        assert_eq!(stats_after1.misses - stats_before.misses, 1);

        // Second call should hit cache
        let result2 = convolve_cached(
            &event,
            hrf.as_ref(),
            &sampling_points,
            &ConvolveOptions::default(),
        )
        .unwrap();
        let stats_after2 = GLOBAL_CACHE.design_matrix_cache.stats();

        assert_eq!(stats_after2.hits - stats_after1.hits, 1);

        // Results should be identical
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_precompute_cache() {
        GLOBAL_CACHE.clear_all();

        let hrfs = vec!["canonical", "gamma"];
        let srs = vec![1.0, 2.0];
        let durations = vec![32.0];

        precompute_hrf_cache(&hrfs, &srs, &durations).unwrap();

        // Check that cache has entries
        assert!(GLOBAL_CACHE.hrf_cache.len() > 0);
    }
}
