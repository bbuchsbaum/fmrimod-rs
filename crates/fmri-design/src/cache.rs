//! Caching utilities for expensive computations.
//!
//! This module provides caching mechanisms to improve performance by storing
//! results of expensive operations like HRF convolution and design matrix generation.

use crate::types::ValidationError;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// A cache key that can represent various types of cached computations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKey {
    /// HRF convolution cache key
    HrfConvolution {
        event_name: String,
        hrf_name: String,
        sampling_points_hash: u64,
        event_hash: u64,
    },
    /// Design matrix cache key
    DesignMatrix {
        event_name: String,
        sampling_points_hash: u64,
        event_params_hash: u64,
    },
    /// Basis evaluation cache key
    BasisEvaluation {
        basis_name: String,
        values_hash: u64,
        n_basis: usize,
    },
    /// Custom cache key
    Custom(String),
}

/// A cached value with metadata.
#[derive(Debug, Clone)]
pub struct CachedValue<T> {
    /// The cached value
    pub value: T,
    /// When the value was cached
    pub timestamp: Instant,
    /// Number of times this value has been accessed
    pub access_count: usize,
    /// Size estimate in bytes (optional)
    pub size_bytes: Option<usize>,
}

impl<T> CachedValue<T> {
    /// Create a new cached value.
    pub fn new(value: T) -> Self {
        Self {
            value,
            timestamp: Instant::now(),
            access_count: 0,
            size_bytes: None,
        }
    }

    /// Create with size estimate.
    pub fn with_size(value: T, size_bytes: usize) -> Self {
        Self {
            value,
            timestamp: Instant::now(),
            access_count: 0,
            size_bytes: Some(size_bytes),
        }
    }

    /// Get age of cached value.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

/// Cache eviction policy.
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Time-based (evict after duration)
    TimeBased(Duration),
}

/// Thread-safe cache for expensive computations.
pub struct ComputationCache<T: Clone> {
    /// The actual cache storage
    cache: Arc<RwLock<HashMap<CacheKey, CachedValue<T>>>>,
    /// Maximum number of entries
    max_entries: Option<usize>,
    /// Maximum total size in bytes
    max_size_bytes: Option<usize>,
    /// Eviction policy
    eviction_policy: EvictionPolicy,
    /// Statistics
    stats: Arc<RwLock<CacheStats>>,
}

/// Cache statistics.
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub total_size_bytes: usize,
}

impl<T: Clone> ComputationCache<T> {
    /// Create a new cache with specified limits.
    pub fn new(max_entries: Option<usize>, max_size_bytes: Option<usize>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
            max_size_bytes,
            eviction_policy: EvictionPolicy::LRU,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Create an unbounded cache.
    pub fn unbounded() -> Self {
        Self::new(None, None)
    }

    /// Set eviction policy.
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Get a value from the cache.
    pub fn get(&self, key: &CacheKey) -> Option<T> {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        if let Some(cached) = cache.get_mut(key) {
            cached.access_count += 1;
            stats.hits += 1;

            // Check time-based eviction
            if let EvictionPolicy::TimeBased(max_age) = self.eviction_policy {
                if cached.age() > max_age {
                    cache.remove(key);
                    stats.evictions += 1;
                    stats.misses += 1;
                    return None;
                }
            }

            Some(cached.value.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Insert a value into the cache.
    pub fn insert(&self, key: CacheKey, value: T, size_bytes: Option<usize>) {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Check if we need to evict
        if let Some(max_entries) = self.max_entries {
            while cache.len() >= max_entries {
                self.evict_one(&mut cache, &mut stats);
            }
        }

        if let Some(max_size) = self.max_size_bytes {
            if let Some(size) = size_bytes {
                while stats.total_size_bytes + size > max_size && !cache.is_empty() {
                    self.evict_one(&mut cache, &mut stats);
                }
            }
        }

        // Insert new value
        let cached = if let Some(size) = size_bytes {
            stats.total_size_bytes += size;
            CachedValue::with_size(value, size)
        } else {
            CachedValue::new(value)
        };

        cache.insert(key, cached);
    }

    /// Clear the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        cache.clear();
        stats.total_size_bytes = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let stats_guard = self.stats.read().unwrap();
        (*stats_guard).clone()
    }

    /// Get cache size.
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.read().unwrap().is_empty()
    }

    /// Evict one entry based on eviction policy.
    fn evict_one(&self, cache: &mut HashMap<CacheKey, CachedValue<T>>, stats: &mut CacheStats) {
        if cache.is_empty() {
            return;
        }

        let key_to_remove = match self.eviction_policy {
            EvictionPolicy::LRU => {
                // Find least recently accessed (oldest timestamp)
                cache
                    .iter()
                    .min_by_key(|(_, v)| v.timestamp)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::LFU => {
                // Find least frequently accessed
                cache
                    .iter()
                    .min_by_key(|(_, v)| v.access_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::FIFO => {
                // Find oldest entry
                cache
                    .iter()
                    .min_by_key(|(_, v)| v.timestamp)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::TimeBased(_) => {
                // Already handled in get()
                None
            }
        };

        if let Some(key) = key_to_remove {
            if let Some(removed) = cache.remove(&key) {
                if let Some(size) = removed.size_bytes {
                    stats.total_size_bytes = stats.total_size_bytes.saturating_sub(size);
                }
                stats.evictions += 1;
            }
        }
    }
}

impl CacheStats {
    /// Get cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Global caches for different types of computations.
pub struct GlobalCaches {
    /// Cache for HRF convolutions
    pub hrf_cache: ComputationCache<Array1<f64>>,
    /// Cache for design matrices
    pub design_matrix_cache: ComputationCache<Array2<f64>>,
    /// Cache for basis evaluations
    pub basis_cache: ComputationCache<Array2<f64>>,
}

impl GlobalCaches {
    /// Create new global caches with default settings.
    pub fn new() -> Self {
        Self {
            hrf_cache: ComputationCache::new(Some(100), Some(100 * 1024 * 1024)) // 100MB
                .with_eviction_policy(EvictionPolicy::LRU),
            design_matrix_cache: ComputationCache::new(Some(50), Some(200 * 1024 * 1024)) // 200MB
                .with_eviction_policy(EvictionPolicy::LRU),
            basis_cache: ComputationCache::new(Some(200), Some(50 * 1024 * 1024)) // 50MB
                .with_eviction_policy(EvictionPolicy::LFU),
        }
    }

    /// Clear all caches.
    pub fn clear_all(&self) {
        self.hrf_cache.clear();
        self.design_matrix_cache.clear();
        self.basis_cache.clear();
    }

    /// Get combined statistics.
    pub fn combined_stats(&self) -> CacheStats {
        let hrf_stats = self.hrf_cache.stats();
        let dm_stats = self.design_matrix_cache.stats();
        let basis_stats = self.basis_cache.stats();

        CacheStats {
            hits: hrf_stats.hits + dm_stats.hits + basis_stats.hits,
            misses: hrf_stats.misses + dm_stats.misses + basis_stats.misses,
            evictions: hrf_stats.evictions + dm_stats.evictions + basis_stats.evictions,
            total_size_bytes: hrf_stats.total_size_bytes
                + dm_stats.total_size_bytes
                + basis_stats.total_size_bytes,
        }
    }
}

impl Default for GlobalCaches {
    fn default() -> Self {
        Self::new()
    }
}

// Lazy static for global cache instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_CACHE: GlobalCaches = GlobalCaches::new();
}

/// Hash helper for floating point arrays.
pub fn hash_array(arr: &ArrayView1<f64>) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    // Hash array shape
    arr.len().hash(&mut hasher);

    // Preserve every bit: sub-millisecond timing and small amplitudes matter.
    for &val in arr.iter() {
        val.to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

/// Estimate size of an array in bytes.
pub fn array_size_bytes<T>(arr: &ArrayBase<impl Data<Elem = T>, impl Dimension>) -> usize {
    std::mem::size_of::<T>() * arr.len()
}

/// Cache-aware HRF convolution.
pub fn cached_hrf_convolution(
    event_name: &str,
    hrf_name: &str,
    event_values: &Array1<f64>,
    sampling_points: &Array1<f64>,
    compute_fn: impl FnOnce() -> Result<Array1<f64>, ValidationError>,
) -> Result<Array1<f64>, ValidationError> {
    let key = CacheKey::HrfConvolution {
        event_name: event_name.to_string(),
        hrf_name: hrf_name.to_string(),
        sampling_points_hash: hash_array(&sampling_points.view()),
        event_hash: hash_array(&event_values.view()),
    };

    // Check cache
    if let Some(cached) = GLOBAL_CACHE.hrf_cache.get(&key) {
        return Ok(cached);
    }

    // Compute and cache
    let result = compute_fn()?;
    let size = array_size_bytes(&result);
    GLOBAL_CACHE
        .hrf_cache
        .insert(key, result.clone(), Some(size));

    Ok(result)
}

/// Cache-aware design matrix computation.
pub fn cached_design_matrix(
    event_name: &str,
    sampling_points: &Array1<f64>,
    params_hash: u64,
    compute_fn: impl FnOnce() -> Result<Array2<f64>, ValidationError>,
) -> Result<Array2<f64>, ValidationError> {
    let key = CacheKey::DesignMatrix {
        event_name: event_name.to_string(),
        sampling_points_hash: hash_array(&sampling_points.view()),
        event_params_hash: params_hash,
    };

    // Check cache
    if let Some(cached) = GLOBAL_CACHE.design_matrix_cache.get(&key) {
        return Ok(cached);
    }

    // Compute and cache
    let result = compute_fn()?;
    let size = array_size_bytes(&result);
    GLOBAL_CACHE
        .design_matrix_cache
        .insert(key, result.clone(), Some(size));

    Ok(result)
}

use ndarray::{ArrayBase, ArrayView1, Data, Dimension};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_basic_cache_operations() {
        let cache: ComputationCache<String> = ComputationCache::new(Some(3), None);

        // Insert some values
        cache.insert(
            CacheKey::Custom("key1".to_string()),
            "value1".to_string(),
            None,
        );
        cache.insert(
            CacheKey::Custom("key2".to_string()),
            "value2".to_string(),
            None,
        );

        // Test get
        assert_eq!(
            cache.get(&CacheKey::Custom("key1".to_string())),
            Some("value1".to_string())
        );
        assert_eq!(cache.get(&CacheKey::Custom("key3".to_string())), None);

        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_eviction() {
        let cache: ComputationCache<i32> = ComputationCache::new(Some(2), None);

        cache.insert(CacheKey::Custom("1".to_string()), 1, None);
        cache.insert(CacheKey::Custom("2".to_string()), 2, None);
        cache.insert(CacheKey::Custom("3".to_string()), 3, None); // Should evict "1"

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&CacheKey::Custom("1".to_string())), None);
        assert_eq!(cache.get(&CacheKey::Custom("2".to_string())), Some(2));
        assert_eq!(cache.get(&CacheKey::Custom("3".to_string())), Some(3));

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_hash_array() {
        let arr1 = array![1.0, 2.0, 3.0];
        let arr2 = array![1.0, 2.0, 3.0];
        let arr3 = array![1.0, 2.0, 3.1];

        assert_eq!(hash_array(&arr1.view()), hash_array(&arr2.view()));
        assert_ne!(hash_array(&arr1.view()), hash_array(&arr3.view()));
    }

    #[test]
    fn test_time_based_eviction() {
        use std::thread::sleep;

        let cache: ComputationCache<i32> = ComputationCache::unbounded()
            .with_eviction_policy(EvictionPolicy::TimeBased(Duration::from_millis(50)));

        cache.insert(CacheKey::Custom("1".to_string()), 1, None);

        // Should still be there
        assert_eq!(cache.get(&CacheKey::Custom("1".to_string())), Some(1));

        // Wait for expiration
        sleep(Duration::from_millis(60));

        // Should be evicted
        assert_eq!(cache.get(&CacheKey::Custom("1".to_string())), None);
    }
}
