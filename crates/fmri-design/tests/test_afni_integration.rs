//! Integration tests for AFNI GLT conversion.

use fmri_design::afni::{format_afni_gltsym, to_glt, write_glt};
use fmri_design::contrast::weights::TermInfo;
use fmri_design::contrast::{
    contrast_weights, oneway_contrast, pair_contrast, unit_contrast, ContrastSpecEnum,
    OnewayContrastSpec, PairContrastSpec, UnitContrastSpec,
};
use fmri_design::events::{EventFactor, EventTerm};
use ndarray::array;
use std::fs;
use tempfile::tempdir;

#[test]
fn test_simple_contrast_to_glt() {
    // Create event
    let event = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec!["A", "B", "C", "A", "B", "C"],
        vec![1.0; 6],
    )
    .unwrap();

    let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

    // Create contrast specification
    let spec = ContrastSpecEnum::Pair(PairContrastSpec {
        name: "A_vs_B".to_string(),
        a: "condition.A".to_string().into(),
        b: "condition.B".to_string().into(),
        where_clause: None,
    });

    // Compute weights
    let contrast = contrast_weights(&spec, &term).unwrap();

    // Convert to GLT
    let glt = to_glt(&contrast).unwrap();

    assert_eq!(glt.glt_str.len(), 1);
    assert_eq!(glt.glt_str[0], "1*condition.A -1*condition.B");
    assert_eq!(glt.name[0], "GLT_A_vs_B");
}

#[test]
fn test_oneway_contrast_to_glt() {
    // Create event with 3 levels
    let event = EventFactor::new(
        "group",
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec!["ctrl", "low", "high", "ctrl", "low", "high"],
        vec![1.0; 6],
    )
    .unwrap();

    let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

    // Create one-way contrast (main effect)
    let spec = ContrastSpecEnum::Oneway(OnewayContrastSpec {
        name: "group_effect".to_string(),
        a: "group".to_string().into(),
        where_clause: None,
    });

    // Compute weights
    let contrast = contrast_weights(&spec, &term).unwrap();

    // Convert to GLT
    let glt = to_glt(&contrast).unwrap();

    // Should have 2 contrasts for 3-level factor
    assert_eq!(glt.glt_str.len(), 2);
    // First contrast: ctrl vs average of others
    assert!(glt.glt_str[0].contains("group.ctrl"));
    // Second contrast: low vs high
    assert!(glt.glt_str[1].contains("group.low"));
    assert!(glt.glt_str[1].contains("group.high"));
}

#[test]
fn test_write_glt_single_file() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("contrast.txt");

    // Create a simple GLT result
    let glt = fmri_design::afni::GltResult {
        glt_str: vec!["1*condA -1*condB".to_string()],
        name: vec!["GLT_A_vs_B".to_string()],
        source: fmri_design::afni::GltSource::Raw {
            weights: array![[1.0, -1.0]],
            condnames: vec!["condA".to_string(), "condB".to_string()],
            name: "A_vs_B".to_string(),
        },
    };

    // Write to file
    write_glt(&glt, &file_path, false).unwrap();

    // Read and verify
    let content = fs::read_to_string(&file_path).unwrap();
    assert!(content.contains("# AFNI GLT contrast specification"));
    assert!(content.contains("# Contrast: GLT_A_vs_B"));
    assert!(content.contains("1*condA -1*condB"));
}

#[test]
fn test_write_glt_multiple_files() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("ftest.txt");

    // Create F-test GLT result
    let glt = fmri_design::afni::GltResult {
        glt_str: vec!["1*A -0.5*B -0.5*C".to_string(), "1*B -1*C".to_string()],
        name: vec!["GLT_main_1".to_string(), "GLT_main_2".to_string()],
        source: fmri_design::afni::GltSource::Raw {
            weights: array![[1.0, -0.5, -0.5], [0.0, 1.0, -1.0]],
            condnames: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            name: "main".to_string(),
        },
    };

    // Write to separate files
    write_glt(&glt, &file_path, true).unwrap();

    // Check that separate files were created
    let file1 = dir.path().join("ftest_1.txt");
    let file2 = dir.path().join("ftest_2.txt");

    assert!(file1.exists());
    assert!(file2.exists());

    let content1 = fs::read_to_string(&file1).unwrap();
    assert!(content1.contains("1*A -0.5*B -0.5*C"));

    let content2 = fs::read_to_string(&file2).unwrap();
    assert!(content2.contains("1*B -1*C"));
}

#[test]
fn test_format_afni_gltsym() {
    let glt = fmri_design::afni::GltResult {
        glt_str: vec!["1*condA -1*condB".to_string()],
        name: vec!["GLT_A_vs_B".to_string()],
        source: fmri_design::afni::GltSource::Raw {
            weights: array![[1.0, -1.0]],
            condnames: vec!["condA".to_string(), "condB".to_string()],
            name: "A_vs_B".to_string(),
        },
    };

    let cmds = format_afni_gltsym(&glt);
    assert_eq!(cmds.len(), 1);
    assert_eq!(
        cmds[0],
        "-gltsym 'SYM: 1*condA -1*condB' -glt_label 1 GLT_A_vs_B"
    );

    // Test single string format
    let cmd_single = fmri_design::afni::format_afni_gltsym_single(&glt);
    assert_eq!(
        cmd_single,
        "-gltsym 'SYM: 1*condA -1*condB' -glt_label 1 GLT_A_vs_B"
    );
}

#[test]
fn test_complex_contrast_workflow() {
    // Create a 2x2 factorial design
    let condition = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec!["A", "B", "A", "B", "A", "B", "A", "B"],
        vec![1.0; 8],
    )
    .unwrap();

    let task = EventFactor::new(
        "task",
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![
            "easy", "easy", "hard", "hard", "easy", "easy", "hard", "hard",
        ],
        vec![1.0; 8],
    )
    .unwrap();

    // Create interaction term
    let term = EventTerm::new(vec![Box::new(condition), Box::new(task)], true).unwrap();

    // Create various contrasts
    let simple_spec = ContrastSpecEnum::Unit(UnitContrastSpec {
        name: "condition_A_task_easy".to_string(),
        where_clause: Some("condition.A_task.easy".to_string().into()),
    });

    let contrast = contrast_weights(&simple_spec, &term).unwrap();
    let glt = to_glt(&contrast).unwrap();

    // Should identify the specific cell
    assert_eq!(glt.glt_str.len(), 1);
    assert_eq!(glt.glt_str[0], "1*condition.A_task.easy");

    // Verify we can write this to a file
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("interaction_contrast.txt");
    write_glt(&glt, &file_path, false).unwrap();
    assert!(file_path.exists());
}
