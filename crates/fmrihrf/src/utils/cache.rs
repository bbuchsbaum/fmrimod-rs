//! Caching utilities for HRF evaluation.

use crate::error::{HrfError, Result};
use crate::hrf::core::Hrf;
use crate::types::HrfValues;
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cache key for HRF evaluations.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    identity: u64,
    span: u64,
    dt: u64,
}

/// Global HRF evaluation cache.
static HRF_CACHE: std::sync::OnceLock<Arc<Mutex<HrfCache>>> = std::sync::OnceLock::new();

/// Cache statistics for monitoring performance.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of successful cache lookups (cache hit)
    pub hits: u64,
    /// Number of cache misses requiring HRF evaluation
    pub misses: u64,
    /// Number of entries evicted due to cache size limit
    pub evictions: u64,
    /// Total number of cache lookup requests
    pub total_requests: u64,
}

impl CacheStats {
    /// Calculate cache hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.hits as f64 / self.total_requests as f64) * 100.0
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.total_requests = 0;
    }
}

/// HRF evaluation cache with size limits and statistics.
#[derive(Debug)]
pub struct HrfCache {
    cache: HashMap<CacheKey, HrfValues>,
    max_size: usize,
    access_order: Vec<CacheKey>,
    stats: CacheStats,
}

impl HrfCache {
    /// Create a new cache with given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            access_order: Vec::new(),
            stats: CacheStats::default(),
        }
    }

    /// Get cached evaluation if available.
    fn get(&mut self, key: &CacheKey) -> Option<HrfValues> {
        self.stats.total_requests += 1;

        if let Some(values) = self.cache.get(key) {
            self.stats.hits += 1;
            // Move to end of access order (LRU)
            self.access_order.retain(|k| k != key);
            self.access_order.push(key.clone());
            Some(values.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert evaluation into cache.
    fn insert(&mut self, key: CacheKey, values: HrfValues) {
        if self.max_size == 0 {
            return;
        }
        // Check if we need to evict
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&key) {
            // Remove least recently used
            if let Some(lru_key) = self.access_order.first().cloned() {
                self.cache.remove(&lru_key);
                self.access_order.remove(0);
                self.stats.evictions += 1;
            }
        }

        // Insert new entry
        self.cache.insert(key.clone(), values);
        self.access_order.retain(|k| k != &key);
        self.access_order.push(key);
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Reset cache statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }
}

/// Get the global HRF cache instance.
fn get_cache() -> Arc<Mutex<HrfCache>> {
    HRF_CACHE
        .get_or_init(|| Arc::new(Mutex::new(HrfCache::new(128))))
        .clone()
}

/// Evaluate HRF with caching to avoid recomputation.
///
/// # Arguments
/// * `hrf` - HRF object to evaluate
/// * `span` - Temporal span in seconds
/// * `dt` - Time step for evaluation
///
/// # Returns
/// Array of HRF values evaluated from 0 to span with step dt
pub fn cached_hrf_eval(hrf: &dyn Hrf, span: f64, dt: f64) -> Result<HrfValues> {
    if !span.is_finite() || span <= 0.0 || !dt.is_finite() || dt <= 0.0 {
        return Err(HrfError::InvalidInput(
            "span and dt must be positive and finite".into(),
        ));
    }
    let count = crate::regressor::convolution::grid_length(span, dt, 1)?;
    let times = || Array1::from_shape_fn(count, |i| i as f64 * dt);
    let Some(identity) = hrf.cache_identity() else {
        return hrf.evaluate(&times());
    };
    let key = CacheKey {
        identity,
        span: span.to_bits(),
        dt: dt.to_bits(),
    };

    // Get cache
    let cache = get_cache();
    let mut cache_guard = cache.lock().unwrap();

    // Check cache
    if let Some(values) = cache_guard.get(&key) {
        return Ok(values);
    }

    // Drop lock before evaluation to avoid holding it during computation
    drop(cache_guard);

    // Evaluate HRF
    let values = hrf.evaluate(&times())?;
    crate::regressor::convolution::check_shape(&values, count, hrf.nbasis())?;

    // Re-acquire lock and insert
    let mut cache_guard = cache.lock().unwrap();
    cache_guard.insert(key, values.clone());

    Ok(values)
}

/// Clear the HRF evaluation cache.
pub fn clear_hrf_cache() {
    let cache = get_cache();
    let mut cache_guard = cache.lock().unwrap();
    cache_guard.clear();
}

/// Get current cache statistics.
///
/// Returns statistics about cache usage including:
/// - hits: Number of successful cache lookups
/// - misses: Number of cache misses requiring evaluation
/// - evictions: Number of entries evicted due to size limit
/// - total_requests: Total number of cache lookups
///
/// # Example
/// ```
/// use fmrihrf::utils::cache::{cached_hrf_eval, get_cache_stats};
/// use fmrihrf::hrf::functions::gamma_hrf;
///
/// let hrf = gamma_hrf(6.0, 1.0, Some(true));
/// let _ = cached_hrf_eval(&*hrf, 20.0, 0.1);
/// let stats = get_cache_stats();
/// println!("Hit rate: {:.1}%", stats.hit_rate());
/// ```
pub fn get_cache_stats() -> CacheStats {
    let cache = get_cache();
    let cache_guard = cache.lock().unwrap();
    cache_guard.stats()
}

/// Reset cache statistics to zero.
///
/// This does not clear the cache itself, only resets the counters.
pub fn reset_cache_stats() {
    let cache = get_cache();
    let mut cache_guard = cache.lock().unwrap();
    cache_guard.reset_stats();
}

/// Set the maximum cache size.
pub fn set_cache_size(max_size: usize) {
    let cache = get_cache();
    let mut cache_guard = cache.lock().unwrap();
    cache_guard.max_size = max_size;

    // Evict entries if necessary
    while cache_guard.cache.len() > max_size && !cache_guard.access_order.is_empty() {
        if let Some(lru_key) = cache_guard.access_order.first().cloned() {
            cache_guard.cache.remove(&lru_key);
            cache_guard.access_order.remove(0);
            cache_guard.stats.evictions += 1;
        }
    }
}

/// Decorator that adds caching to an HRF.
#[derive(Debug)]
pub struct CachedHrf {
    inner: Box<dyn Hrf>,
}

impl CachedHrf {
    /// Create a new cached HRF wrapper.
    pub fn new(hrf: Box<dyn Hrf>) -> Self {
        Self { inner: hrf }
    }
}

impl Hrf for CachedHrf {
    fn cache_identity(&self) -> Option<u64> {
        self.inner.cache_identity()
    }
    fn evaluate(&self, t: &Array1<f64>) -> Result<HrfValues> {
        // For arbitrary time points, we can't use the grid-based cache
        // Just pass through to the inner HRF
        self.inner.evaluate(t)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn nbasis(&self) -> usize {
        self.inner.nbasis()
    }

    fn span(&self) -> f64 {
        self.inner.span()
    }

    fn params(&self) -> Vec<(&str, f64)> {
        self.inner.params()
    }
}

/// Add caching to an HRF.
pub fn cache_hrf(hrf: Box<dyn Hrf>) -> Box<dyn Hrf> {
    Box::new(CachedHrf::new(hrf))
}

#[cfg(test)]
mod tests {
    static TEST_CACHE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    use super::*;
    use crate::hrf::functions::gamma_hrf;

    #[test]
    fn test_cached_hrf_eval() {
        let _guard = TEST_CACHE.lock().unwrap();
        set_cache_size(128);
        clear_hrf_cache();

        let hrf = gamma_hrf(6.0, 1.0, Some(true));

        // First evaluation
        let result1 = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        // Second evaluation (should be cached)
        let result2 = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        // Results should be identical
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_cache_eviction() {
        let _guard = TEST_CACHE.lock().unwrap();
        set_cache_size(128);
        clear_hrf_cache();
        set_cache_size(2);

        // Create different HRFs
        let hrf1 = gamma_hrf(6.0, 1.0, Some(true));
        let hrf2 = gamma_hrf(7.0, 1.0, Some(true));
        let hrf3 = gamma_hrf(8.0, 1.0, Some(true));

        // Fill cache
        let _ = cached_hrf_eval(&*hrf1, 10.0, 0.1).unwrap();
        let _ = cached_hrf_eval(&*hrf2, 10.0, 0.1).unwrap();

        // This should evict hrf1
        let _ = cached_hrf_eval(&*hrf3, 10.0, 0.1).unwrap();

        // Cache size should still be 2
        let cache = get_cache();
        let cache_guard = cache.lock().unwrap();
        assert!(cache_guard.cache.len() <= 2);
    }

    #[test]
    fn test_clear_cache() {
        let _guard = TEST_CACHE.lock().unwrap();
        set_cache_size(128);
        clear_hrf_cache();

        let hrf = gamma_hrf(6.0, 1.0, Some(true));
        let _ = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        // Cache should have entry
        let cache = get_cache();
        {
            let cache_guard = cache.lock().unwrap();
            assert!(cache_guard.cache.len() > 0);
        }

        // Clear cache
        clear_hrf_cache();

        // Cache should be empty
        {
            let cache_guard = cache.lock().unwrap();
            assert_eq!(cache_guard.cache.len(), 0);
        }
    }

    #[test]
    fn test_cache_statistics() {
        let _guard = TEST_CACHE.lock().unwrap();
        set_cache_size(128);
        clear_hrf_cache();
        reset_cache_stats();

        let hrf = gamma_hrf(6.0, 1.0, Some(true));

        // Get initial stats
        let initial_stats = get_cache_stats();

        // First evaluation (miss)
        let _ = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        let stats1 = get_cache_stats();
        // Check that stats increased (may not be exact due to parallel tests)
        assert!(stats1.misses >= initial_stats.misses + 1);
        assert!(stats1.total_requests >= initial_stats.total_requests + 1);

        // Second evaluation (hit)
        let _ = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        let stats2 = get_cache_stats();
        assert!(stats2.hits >= stats1.hits + 1);
        assert!(stats2.total_requests >= stats1.total_requests + 1);

        // Third evaluation (hit)
        let _ = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        let stats3 = get_cache_stats();
        assert!(stats3.hits >= stats2.hits + 1);
        assert!(stats3.total_requests >= stats2.total_requests + 1);

        // Verify hit rate calculation works
        assert!(stats3.hit_rate() >= 0.0 && stats3.hit_rate() <= 100.0);
    }

    #[test]
    fn test_reset_stats() {
        let _guard = TEST_CACHE.lock().unwrap();
        set_cache_size(128);
        clear_hrf_cache();
        reset_cache_stats();

        let hrf = gamma_hrf(6.0, 1.0, Some(true));

        // Generate some cache activity
        let _ = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();
        let _ = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        let stats = get_cache_stats();
        assert!(stats.total_requests > 0);

        // Reset stats
        reset_cache_stats();

        let stats = get_cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_eviction_statistics() {
        let _guard = TEST_CACHE.lock().unwrap();
        set_cache_size(128);
        clear_hrf_cache();
        reset_cache_stats();
        set_cache_size(2);

        // Get initial stats
        let initial_stats = get_cache_stats();

        // Create HRFs that will fill and exceed cache
        let hrf1 = gamma_hrf(61.1, 1.0, Some(true));
        let hrf2 = gamma_hrf(71.1, 1.0, Some(true));
        let hrf3 = gamma_hrf(81.1, 1.0, Some(true));

        // Fill cache (2 misses)
        let _ = cached_hrf_eval(&*hrf1, 11.11, 0.1).unwrap();
        let _ = cached_hrf_eval(&*hrf2, 11.11, 0.1).unwrap();

        let stats1 = get_cache_stats();
        assert!(stats1.misses >= initial_stats.misses + 2);

        // This should evict hrf1 (1 miss, 1 eviction)
        let _ = cached_hrf_eval(&*hrf3, 11.11, 0.1).unwrap();

        let stats2 = get_cache_stats();
        assert!(stats2.misses >= stats1.misses + 1);
        assert!(stats2.evictions >= stats1.evictions + 1);
    }
}
