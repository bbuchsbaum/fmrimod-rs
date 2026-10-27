//! Comprehensive test suite for cross-validation functionality.

use fmri_design::core::{DataFrame, EventModel};
use fmri_design::crossval::{split_events, CVIterator, CVSplitter, CVStrategy};
use fmri_design::events::{EventFactor, EventMatrix, EventVariable};
use fmri_design::types::{EventSpec, ValidationError};
use fmri_hrf::SamplingFrame;
use ndarray::array;
use std::collections::HashMap;

#[test]
fn test_kfold_basic() {
    let splitter = CVSplitter::kfold(5);
    let folds = splitter.split(100).unwrap();

    assert_eq!(folds.len(), 5);

    // Verify each sample appears exactly once in test set
    let mut test_counts = vec![0; 100];
    for fold in &folds {
        for &idx in &fold.test_indices {
            test_counts[idx] += 1;
        }
        assert_eq!(fold.test_indices.len() + fold.train_indices.len(), 100);
    }

    // Each sample should appear in exactly one test fold
    assert!(test_counts.iter().all(|&count| count == 1));
}

#[test]
fn test_kfold_uneven_split() {
    // Test with number of samples not evenly divisible by k
    let splitter = CVSplitter::kfold(7);
    let folds = splitter.split(50).unwrap();

    assert_eq!(folds.len(), 7);

    // First 50 % 7 = 1 fold should have 8 samples, rest should have 7
    let mut fold_sizes: Vec<usize> = folds.iter().map(|f| f.test_indices.len()).collect();
    fold_sizes.sort_unstable();

    assert_eq!(fold_sizes[0], 7); // Smallest fold
    assert_eq!(fold_sizes[6], 8); // Largest fold
}

#[test]
fn test_kfold_shuffled() {
    let splitter1 = CVSplitter::kfold_shuffled(5, Some(42));
    let splitter2 = CVSplitter::kfold_shuffled(5, Some(42));
    let splitter3 = CVSplitter::kfold_shuffled(5, Some(123));

    let folds1 = splitter1.split(20).unwrap();
    let folds2 = splitter2.split(20).unwrap();
    let folds3 = splitter3.split(20).unwrap();

    // Same seed should produce same splits
    assert_eq!(folds1[0].test_indices, folds2[0].test_indices);

    // Different seed should (likely) produce different splits
    assert_ne!(folds1[0].test_indices, folds3[0].test_indices);
}

#[test]
fn test_leave_one_out() {
    let splitter = CVSplitter::leave_one_out();
    let n_samples = 15;
    let folds = splitter.split(n_samples).unwrap();

    assert_eq!(folds.len(), n_samples);

    for (i, fold) in folds.iter().enumerate() {
        // Each fold should have exactly one test sample
        assert_eq!(fold.test_indices.len(), 1);
        assert_eq!(fold.test_indices[0], i);

        // Train set should have all other samples
        assert_eq!(fold.train_indices.len(), n_samples - 1);
        assert!(!fold.train_indices.contains(&i));
    }
}

#[test]
fn test_leave_p_out() {
    let splitter = CVSplitter::new(CVStrategy::LeavePOut { p: 2 });
    let folds = splitter.split(5).unwrap();

    // C(5,2) = 10 combinations
    assert_eq!(folds.len(), 10);

    // Each fold should have exactly 2 test samples
    for fold in &folds {
        assert_eq!(fold.test_indices.len(), 2);
        assert_eq!(fold.train_indices.len(), 3);
    }

    // Collect all test pairs
    let mut test_pairs: Vec<(usize, usize)> = folds
        .iter()
        .map(|f| {
            let mut indices = f.test_indices.clone();
            indices.sort();
            (indices[0], indices[1])
        })
        .collect();
    test_pairs.sort();

    // Should have all unique pairs
    let expected_pairs = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ];
    assert_eq!(test_pairs, expected_pairs);
}

#[test]
fn test_time_series_split() {
    let splitter = CVSplitter::time_series(4);
    let folds = splitter.split(20).unwrap();

    assert_eq!(folds.len(), 4);

    // Each successive fold should have a larger training set
    for i in 1..folds.len() {
        assert!(folds[i].train_indices.len() > folds[i - 1].train_indices.len());
    }

    // Training sets should be contiguous from start
    for fold in &folds {
        let max_train_idx = fold.train_indices.iter().max().unwrap();
        let min_test_idx = fold.test_indices.iter().min().unwrap();
        assert!(max_train_idx < min_test_idx);
    }
}

#[test]
fn test_stratified_kfold() {
    let groups = vec![
        "A", "A", "A", "A", "A", "A", // 6 A's
        "B", "B", "B", "B", // 4 B's
        "C", "C", // 2 C's
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();

    let splitter = CVSplitter::new(CVStrategy::StratifiedKFold {
        n_folds: 3,
        shuffle: false,
        random_state: None,
    });

    let folds = splitter.split_stratified(12, &groups).unwrap();
    assert_eq!(folds.len(), 3);

    // Check class distribution in each fold
    for fold in &folds {
        let test_groups: Vec<&String> = fold.test_indices.iter().map(|&i| &groups[i]).collect();

        let a_count = test_groups.iter().filter(|&&g| g == "A").count();
        let b_count = test_groups.iter().filter(|&&g| g == "B").count();
        let c_count = test_groups.iter().filter(|&&g| g == "C").count();

        // Each fold should have approximately equal class distribution
        assert_eq!(a_count, 2); // 6/3 = 2
        assert!(b_count == 1 || b_count == 2); // 4/3 ≈ 1.33
        assert!(c_count == 0 || c_count == 1); // 2/3 ≈ 0.67
    }
}

#[test]
fn test_custom_split() {
    let train_indices = vec![vec![1, 2, 3, 4], vec![0, 2, 3, 4], vec![0, 1, 3, 4]];

    let test_indices = vec![vec![0], vec![1], vec![2]];

    let splitter = CVSplitter::new(CVStrategy::Custom {
        train_indices: train_indices.clone(),
        test_indices: test_indices.clone(),
    });

    let folds = splitter.split(5).unwrap();
    assert_eq!(folds.len(), 3);

    for i in 0..3 {
        assert_eq!(folds[i].train_indices, train_indices[i]);
        assert_eq!(folds[i].test_indices, test_indices[i]);
    }
}

#[test]
fn test_split_events() {
    // Create sample events
    let event = Box::new(
        EventFactor::new(
            "condition",
            vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            vec!["A", "B", "A", "B", "A", "B"],
            vec![1.0; 6],
        )
        .unwrap(),
    );

    let events: Vec<Box<dyn EventSpec>> = vec![event];

    // Create a simple fold
    let fold = fmri_design::crossval::CVFold {
        fold: 0,
        train_indices: vec![0, 1, 2, 3],
        test_indices: vec![4, 5],
    };

    let (train_events, test_events) = split_events(&events, &fold).unwrap();

    assert_eq!(train_events.len(), 1);
    assert_eq!(test_events.len(), 1);

    // Check that events were split correctly
    assert_eq!(train_events[0].n_events(), 4);
    assert_eq!(test_events[0].n_events(), 2);
}

#[test]
fn test_cv_iterator() {
    // Create a simple event model
    let events: Vec<Box<dyn EventSpec>> = vec![Box::new(
        EventVariable::new(
            "response",
            vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![0.5; 10],
        )
        .unwrap(),
    )];

    let sampling_frame = SamplingFrame::uniform(100, 2.0, 0.1).unwrap();
    let model = EventModel::new(events, sampling_frame).unwrap();

    let splitter = CVSplitter::kfold(5);
    let mut cv_iter = CVIterator::new(model, splitter).unwrap();

    assert_eq!(cv_iter.n_folds(), 5);

    // Iterate through all folds
    let mut fold_count = 0;
    while let Some(result) = cv_iter.next_fold() {
        let (train_model, test_model) = result.unwrap();

        // Each fold should have appropriate number of events
        assert_eq!(train_model.events[0].n_events(), 8);
        assert_eq!(test_model.events[0].n_events(), 2);

        fold_count += 1;
    }

    assert_eq!(fold_count, 5);

    // Test reset
    cv_iter.reset();
    let first_fold_again = cv_iter.next_fold();
    assert!(first_fold_again.is_some());
}

#[test]
fn test_edge_cases() {
    // Test with too many folds
    let splitter = CVSplitter::kfold(10);
    let result = splitter.split(5);
    assert!(result.is_err());

    // Test with single fold
    let splitter = CVSplitter::kfold(1);
    let result = splitter.split(10);
    assert!(result.is_err());

    // Test leave-p-out with p >= n
    let splitter = CVSplitter::new(CVStrategy::LeavePOut { p: 10 });
    let result = splitter.split(5);
    assert!(result.is_err());

    // Test time series with too few samples
    let splitter = CVSplitter::time_series(5);
    let result = splitter.split(3);
    assert!(result.is_err());
}

#[test]
fn test_reproducibility() {
    // Test that stratified splits are reproducible with same seed
    let groups: Vec<String> = (0..100)
        .map(|i| {
            if i % 3 == 0 {
                "A"
            } else if i % 3 == 1 {
                "B"
            } else {
                "C"
            }
        })
        .map(String::from)
        .collect();

    let splitter1 = CVSplitter::new(CVStrategy::StratifiedKFold {
        n_folds: 5,
        shuffle: true,
        random_state: Some(12345),
    });

    let splitter2 = CVSplitter::new(CVStrategy::StratifiedKFold {
        n_folds: 5,
        shuffle: true,
        random_state: Some(12345),
    });

    let folds1 = splitter1.split_stratified(100, &groups).unwrap();
    let folds2 = splitter2.split_stratified(100, &groups).unwrap();

    // Should produce identical splits
    for i in 0..5 {
        assert_eq!(folds1[i].test_indices, folds2[i].test_indices);
        assert_eq!(folds1[i].train_indices, folds2[i].train_indices);
    }
}

#[test]
fn test_no_overlap() {
    // Verify that train and test sets never overlap
    let strategies = vec![
        (
            CVStrategy::KFold {
                n_folds: 5,
                shuffle: false,
                random_state: None,
            },
            20,
        ),
        (CVStrategy::LeaveOneOut, 20),
        (CVStrategy::LeavePOut { p: 3 }, 8),
        (CVStrategy::TimeSeriesSplit { n_splits: 4 }, 20),
    ];

    for (strategy, n_samples) in strategies {
        let splitter = CVSplitter::new(strategy);
        let folds = splitter.split(n_samples).unwrap();

        for fold in &folds {
            let train_set: std::collections::HashSet<_> = fold.train_indices.iter().collect();
            let test_set: std::collections::HashSet<_> = fold.test_indices.iter().collect();

            // No overlap between train and test
            let intersection: Vec<_> = train_set.intersection(&test_set).collect();
            assert!(intersection.is_empty());

            // Union should cover some indices (not necessarily all for time series)
            let union_size = train_set.len() + test_set.len();
            assert!(union_size > 0);
        }
    }
}

#[test]
fn test_design_matrix_consistency() {
    // Test that design matrices from CV splits are consistent
    let events: Vec<Box<dyn EventSpec>> = vec![Box::new(
        EventFactor::new(
            "task",
            vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            vec!["A", "B", "A", "B", "A", "B", "A", "B"],
            vec![2.0; 8],
        )
        .unwrap(),
    )];

    let sampling_frame = SamplingFrame::uniform(200, 2.0, 0.2).unwrap();
    let full_model = EventModel::new(events.clone(), sampling_frame.clone()).unwrap();
    let full_dm = full_model.generate_design_matrix().unwrap();

    // Create 4-fold CV
    let splitter = CVSplitter::kfold(4);
    let mut cv_iter = CVIterator::new(full_model, splitter).unwrap();

    // Check each fold
    while let Some(result) = cv_iter.next_fold() {
        let (train_model, test_model) = result.unwrap();

        // Both should generate valid design matrices
        let train_dm = train_model.generate_design_matrix();
        let test_dm = test_model.generate_design_matrix();

        assert!(train_dm.is_ok());
        assert!(test_dm.is_ok());

        // Check dimensions are reasonable
        let train_dm = train_dm.unwrap();
        let test_dm = test_dm.unwrap();

        assert_eq!(train_dm.ncols(), full_dm.ncols());
        assert_eq!(test_dm.ncols(), full_dm.ncols());
        assert_eq!(train_dm.nrows(), full_dm.nrows());
        assert_eq!(test_dm.nrows(), full_dm.nrows());
    }
}
