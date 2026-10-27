//! Cross-validation utilities for design matrices.
//!
//! This module provides tools for splitting data into training and test sets
//! for cross-validation of fMRI design matrices.

use crate::core::EventModel;
use crate::types::{EventSpec, ValidationError};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashSet;

/// Cross-validation fold specification.
#[derive(Debug, Clone)]
pub struct CVFold {
    /// Fold index
    pub fold: usize,
    /// Training indices
    pub train_indices: Vec<usize>,
    /// Test indices
    pub test_indices: Vec<usize>,
}

/// Cross-validation split strategy.
#[derive(Debug, Clone)]
pub enum CVStrategy {
    /// K-fold cross-validation
    KFold {
        n_folds: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Leave-P-out cross-validation
    LeavePOut { p: usize },
    /// Time series split (for temporal data)
    TimeSeriesSplit { n_splits: usize },
    /// Stratified K-fold (preserves class ratios)
    StratifiedKFold {
        n_folds: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Custom split with explicit indices
    Custom {
        train_indices: Vec<Vec<usize>>,
        test_indices: Vec<Vec<usize>>,
    },
}

/// Cross-validation splitter.
pub struct CVSplitter {
    strategy: CVStrategy,
}

impl CVSplitter {
    /// Create a new cross-validation splitter.
    pub fn new(strategy: CVStrategy) -> Self {
        Self { strategy }
    }

    /// Create K-fold splitter.
    pub fn kfold(n_folds: usize) -> Self {
        Self::new(CVStrategy::KFold {
            n_folds,
            shuffle: false,
            random_state: None,
        })
    }

    /// Create K-fold splitter with shuffling.
    pub fn kfold_shuffled(n_folds: usize, random_state: Option<u64>) -> Self {
        Self::new(CVStrategy::KFold {
            n_folds,
            shuffle: true,
            random_state,
        })
    }

    /// Create leave-one-out splitter.
    pub fn leave_one_out() -> Self {
        Self::new(CVStrategy::LeaveOneOut)
    }

    /// Create time series splitter.
    pub fn time_series(n_splits: usize) -> Self {
        Self::new(CVStrategy::TimeSeriesSplit { n_splits })
    }

    /// Split data into folds.
    pub fn split(&self, n_samples: usize) -> Result<Vec<CVFold>, ValidationError> {
        match &self.strategy {
            CVStrategy::KFold {
                n_folds,
                shuffle,
                random_state,
            } => self.kfold_split(n_samples, *n_folds, *shuffle, *random_state),
            CVStrategy::LeaveOneOut => self.loo_split(n_samples),
            CVStrategy::LeavePOut { p } => self.lpo_split(n_samples, *p),
            CVStrategy::TimeSeriesSplit { n_splits } => {
                self.time_series_split(n_samples, *n_splits)
            }
            CVStrategy::StratifiedKFold { .. } => Err(ValidationError::InvalidParameter(
                "Stratified K-fold requires group labels".to_string(),
            )),
            CVStrategy::Custom {
                train_indices,
                test_indices,
            } => self.custom_split(train_indices, test_indices),
        }
    }

    /// Split data with group labels (for stratified splits).
    pub fn split_stratified(
        &self,
        n_samples: usize,
        groups: &[String],
    ) -> Result<Vec<CVFold>, ValidationError> {
        if groups.len() != n_samples {
            return Err(ValidationError::LengthMismatch {
                expected: n_samples,
                actual: groups.len(),
            });
        }

        match &self.strategy {
            CVStrategy::StratifiedKFold {
                n_folds,
                shuffle,
                random_state,
            } => self.stratified_kfold_split(groups, *n_folds, *shuffle, *random_state),
            _ => self.split(n_samples),
        }
    }

    /// K-fold split implementation.
    fn kfold_split(
        &self,
        n_samples: usize,
        n_folds: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<Vec<CVFold>, ValidationError> {
        if n_folds < 2 {
            return Err(ValidationError::InvalidParameter(
                "Number of folds must be at least 2".to_string(),
            ));
        }

        if n_folds > n_samples {
            return Err(ValidationError::InvalidParameter(format!(
                "Number of folds ({}) cannot exceed number of samples ({})",
                n_folds, n_samples
            )));
        }

        // Create indices
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if shuffle {
            let mut rng = if let Some(seed) = random_state {
                rand::rngs::StdRng::seed_from_u64(seed)
            } else {
                rand::rngs::StdRng::from_entropy()
            };
            indices.shuffle(&mut rng);
        }

        // Create folds
        let mut folds = Vec::new();
        let fold_size = n_samples / n_folds;
        let remainder = n_samples % n_folds;

        let mut start = 0;
        for fold in 0..n_folds {
            let current_fold_size = if fold < remainder {
                fold_size + 1
            } else {
                fold_size
            };

            let end = start + current_fold_size;
            let test_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            folds.push(CVFold {
                fold,
                train_indices,
                test_indices,
            });

            start = end;
        }

        Ok(folds)
    }

    /// Leave-one-out split implementation.
    fn loo_split(&self, n_samples: usize) -> Result<Vec<CVFold>, ValidationError> {
        let mut folds = Vec::new();

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();

            folds.push(CVFold {
                fold: i,
                train_indices,
                test_indices,
            });
        }

        Ok(folds)
    }

    /// Leave-P-out split implementation.
    fn lpo_split(&self, n_samples: usize, p: usize) -> Result<Vec<CVFold>, ValidationError> {
        if p >= n_samples {
            return Err(ValidationError::InvalidParameter(format!(
                "p ({}) must be less than number of samples ({})",
                p, n_samples
            )));
        }

        // This can generate many combinations - limit for practical use
        let max_combinations = 100;
        let n_combinations = num_integer::binomial(n_samples, p);

        if n_combinations > max_combinations {
            return Err(ValidationError::InvalidParameter(format!(
                "Leave-{}-out would generate {} combinations (max: {})",
                p, n_combinations, max_combinations
            )));
        }

        // Generate all combinations of size p
        let indices: Vec<usize> = (0..n_samples).collect();
        let combinations = generate_combinations(&indices, p);

        let mut folds = Vec::new();
        for (i, test_combo) in combinations.iter().enumerate() {
            let test_set: HashSet<usize> = test_combo.iter().copied().collect();
            let train_indices: Vec<usize> = indices
                .iter()
                .filter(|idx| !test_set.contains(idx))
                .copied()
                .collect();

            folds.push(CVFold {
                fold: i,
                train_indices,
                test_indices: test_combo.clone(),
            });
        }

        Ok(folds)
    }

    /// Time series split implementation.
    fn time_series_split(
        &self,
        n_samples: usize,
        n_splits: usize,
    ) -> Result<Vec<CVFold>, ValidationError> {
        if n_splits < 2 {
            return Err(ValidationError::InvalidParameter(
                "Number of splits must be at least 2".to_string(),
            ));
        }

        let min_train_size = n_samples / (n_splits + 1);
        if min_train_size < 1 {
            return Err(ValidationError::InvalidParameter(
                "Not enough samples for requested number of splits".to_string(),
            ));
        }

        let mut folds = Vec::new();
        let test_size = n_samples / (n_splits + 1);

        for i in 0..n_splits {
            let train_end = (i + 1) * test_size;
            let test_end = train_end + test_size;

            let train_indices: Vec<usize> = (0..train_end).collect();
            let test_indices: Vec<usize> = (train_end..test_end.min(n_samples)).collect();

            if !test_indices.is_empty() {
                folds.push(CVFold {
                    fold: i,
                    train_indices,
                    test_indices,
                });
            }
        }

        Ok(folds)
    }

    /// Stratified K-fold split implementation.
    fn stratified_kfold_split(
        &self,
        groups: &[String],
        n_folds: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<Vec<CVFold>, ValidationError> {
        // Group indices by class
        let mut class_indices: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();

        for (i, group) in groups.iter().enumerate() {
            class_indices
                .entry(group.clone())
                .or_insert_with(Vec::new)
                .push(i);
        }

        // Shuffle within each class if requested
        if shuffle {
            let mut rng = if let Some(seed) = random_state {
                rand::rngs::StdRng::seed_from_u64(seed)
            } else {
                rand::rngs::StdRng::from_entropy()
            };

            for indices in class_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Create stratified folds
        let mut folds: Vec<CVFold> = (0..n_folds)
            .map(|i| CVFold {
                fold: i,
                train_indices: Vec::new(),
                test_indices: Vec::new(),
            })
            .collect();

        // Distribute each class across folds
        for (_, indices) in class_indices {
            let class_size = indices.len();
            let fold_size = class_size / n_folds;
            let remainder = class_size % n_folds;

            let mut start = 0;
            for fold in 0..n_folds {
                let current_fold_size = if fold < remainder {
                    fold_size + 1
                } else {
                    fold_size
                };

                let end = start + current_fold_size;

                // Add to test set for this fold
                folds[fold].test_indices.extend(&indices[start..end]);

                // Add to train sets for other folds
                for other_fold in 0..n_folds {
                    if other_fold != fold {
                        folds[other_fold].train_indices.extend(&indices[start..end]);
                    }
                }

                start = end;
            }
        }

        Ok(folds)
    }

    /// Custom split implementation.
    fn custom_split(
        &self,
        train_indices: &[Vec<usize>],
        test_indices: &[Vec<usize>],
    ) -> Result<Vec<CVFold>, ValidationError> {
        if train_indices.len() != test_indices.len() {
            return Err(ValidationError::LengthMismatch {
                expected: train_indices.len(),
                actual: test_indices.len(),
            });
        }

        let folds: Vec<CVFold> = train_indices
            .iter()
            .zip(test_indices.iter())
            .enumerate()
            .map(|(i, (train, test))| CVFold {
                fold: i,
                train_indices: train.clone(),
                test_indices: test.clone(),
            })
            .collect();

        Ok(folds)
    }
}

/// Generate all combinations of size k from items.
fn generate_combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }

    if k > items.len() {
        return vec![];
    }

    if k == items.len() {
        return vec![items.to_vec()];
    }

    let mut result = Vec::new();

    // Include first element
    let with_first = generate_combinations(&items[1..], k - 1);
    for mut combo in with_first {
        combo.insert(0, items[0].clone());
        result.push(combo);
    }

    // Exclude first element
    let without_first = generate_combinations(&items[1..], k);
    result.extend(without_first);

    result
}

/// Split events based on fold indices.
pub fn split_events(
    events: &[Box<dyn EventSpec>],
    fold: &CVFold,
) -> Result<(Vec<Box<dyn EventSpec>>, Vec<Box<dyn EventSpec>>), ValidationError> {
    let mut train_events = Vec::new();
    let mut test_events = Vec::new();

    for event in events {
        // Get event onsets
        let onsets = event.onsets();
        let n_events = onsets.len();

        // Create masks for train and test
        let mut train_mask = vec![false; n_events];
        let mut test_mask = vec![false; n_events];

        for &idx in &fold.train_indices {
            if idx < n_events {
                train_mask[idx] = true;
            }
        }

        for &idx in &fold.test_indices {
            if idx < n_events {
                test_mask[idx] = true;
            }
        }

        // Filter event based on masks
        if train_mask.iter().any(|&x| x) {
            let train_event = event.filter(&train_mask)?;
            train_events.push(train_event);
        }

        if test_mask.iter().any(|&x| x) {
            let test_event = event.filter(&test_mask)?;
            test_events.push(test_event);
        }
    }

    Ok((train_events, test_events))
}

/// Cross-validation iterator for EventModel.
pub struct CVIterator {
    model: EventModel,
    #[allow(dead_code)] // retained for iterator reset/reconfiguration
    splitter: CVSplitter,
    folds: Vec<CVFold>,
    current: usize,
}

impl CVIterator {
    /// Create a new cross-validation iterator.
    pub fn new(model: EventModel, splitter: CVSplitter) -> Result<Self, ValidationError> {
        // Get total number of samples (use first event as reference)
        let n_samples = if let Some(first_event) = model.events.first() {
            first_event.onsets().len()
        } else {
            return Err(ValidationError::InvalidParameter(
                "Model has no events".to_string(),
            ));
        };

        let folds = splitter.split(n_samples)?;

        Ok(Self {
            model,
            splitter,
            folds,
            current: 0,
        })
    }

    /// Get the next fold.
    pub fn next_fold(&mut self) -> Option<Result<(EventModel, EventModel), ValidationError>> {
        if self.current >= self.folds.len() {
            return None;
        }

        let fold = &self.folds[self.current];
        self.current += 1;

        // Split events
        match split_events(&self.model.events, fold) {
            Ok((train_events, test_events)) => {
                // Create train and test models
                let train_model = EventModel::new(train_events, self.model.sampling_frame.clone());

                let test_model = EventModel::new(test_events, self.model.sampling_frame.clone());

                match (train_model, test_model) {
                    (Ok(train), Ok(test)) => Some(Ok((train, test))),
                    (Err(e), _) | (_, Err(e)) => Some(Err(e)),
                }
            }
            Err(e) => Some(Err(e)),
        }
    }

    /// Reset iterator to beginning.
    pub fn reset(&mut self) {
        self.current = 0;
    }

    /// Get number of folds.
    pub fn n_folds(&self) -> usize {
        self.folds.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_split() {
        let splitter = CVSplitter::kfold(5);
        let folds = splitter.split(100).unwrap();

        assert_eq!(folds.len(), 5);

        // Check that each sample appears exactly once in test set
        let mut test_counts = vec![0; 100];
        for fold in &folds {
            for &idx in &fold.test_indices {
                test_counts[idx] += 1;
            }
        }
        assert!(test_counts.iter().all(|&count| count == 1));

        // Check fold sizes
        for fold in &folds {
            assert_eq!(fold.train_indices.len(), 80);
            assert_eq!(fold.test_indices.len(), 20);
        }
    }

    #[test]
    fn test_leave_one_out() {
        let splitter = CVSplitter::leave_one_out();
        let folds = splitter.split(10).unwrap();

        assert_eq!(folds.len(), 10);

        for (i, fold) in folds.iter().enumerate() {
            assert_eq!(fold.test_indices, vec![i]);
            assert_eq!(fold.train_indices.len(), 9);
        }
    }

    #[test]
    fn test_time_series_split() {
        let splitter = CVSplitter::time_series(3);
        let folds = splitter.split(12).unwrap();

        assert_eq!(folds.len(), 3);

        // Check progressive training sets
        assert_eq!(folds[0].train_indices.len(), 3);
        assert_eq!(folds[1].train_indices.len(), 6);
        assert_eq!(folds[2].train_indices.len(), 9);

        // All test sets should be same size
        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 3);
        }
    }

    #[test]
    fn test_stratified_kfold() {
        let groups = vec!["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        let splitter = CVSplitter::new(CVStrategy::StratifiedKFold {
            n_folds: 2,
            shuffle: false,
            random_state: None,
        });

        let folds = splitter.split_stratified(10, &groups).unwrap();
        assert_eq!(folds.len(), 2);

        // Check that classes are balanced in each fold
        for fold in &folds {
            let test_groups: Vec<&String> = fold.test_indices.iter().map(|&i| &groups[i]).collect();

            // Should have 2 A's, 2 B's, and 1 C in each test fold
            let a_count = test_groups.iter().filter(|&&g| g == "A").count();
            let b_count = test_groups.iter().filter(|&&g| g == "B").count();
            let c_count = test_groups.iter().filter(|&&g| g == "C").count();

            assert_eq!(a_count, 2);
            assert_eq!(b_count, 2);
            assert_eq!(c_count, 1);
        }
    }
}
