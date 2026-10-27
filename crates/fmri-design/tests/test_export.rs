//! Tests for export functionality

use fmri_design::core::EventModel;
use fmri_design::design::DesignMatrix;
use fmri_design::events::{EventFactor, EventVariable};
use fmri_design::io::{
    to_csv_string, to_json_string, write_csv, write_json, DataFrameExport, EventModelExport,
    ExportOptions,
};
use fmri_design::sampling::{SamplingFrame, SamplingFrameExt};
use fmri_design::types::EventSpec;
use ndarray::array;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_design_matrix_csv_export() {
    let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let dm = DesignMatrix::new(
        matrix,
        vec!["col1".to_string(), "col2".to_string(), "col3".to_string()],
    );

    let csv = to_csv_string(&dm, None).unwrap();

    // Check headers
    assert!(csv.contains("col1,col2,col3"));

    // Check data
    assert!(csv.contains("1.000000,2.000000,3.000000"));
    assert!(csv.contains("4.000000,5.000000,6.000000"));
}

#[test]
fn test_design_matrix_json_export() {
    let matrix = array![[1.0, 2.0], [3.0, 4.0]];
    let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

    let json = to_json_string(&dm, None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["n_rows"], 2);
    assert_eq!(parsed["n_cols"], 2);
    assert_eq!(parsed["columns"], serde_json::json!(["A", "B"]));
    assert_eq!(parsed["data"], serde_json::json!([[1.0, 2.0], [3.0, 4.0]]));
}

#[test]
fn test_custom_delimiter() {
    let matrix = array![[1.0, 2.0]];
    let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

    let opts = ExportOptions {
        delimiter: '\t',
        ..Default::default()
    };

    let csv = to_csv_string(&dm, Some(opts)).unwrap();
    assert!(csv.contains("A\tB"));
    assert!(csv.contains("1.000000\t2.000000"));
}

#[test]
fn test_custom_precision() {
    let matrix = array![[1.23456789]];
    let dm = DesignMatrix::new(matrix, vec!["value".to_string()]);

    let opts = ExportOptions {
        precision: 2,
        ..Default::default()
    };

    let csv = to_csv_string(&dm, Some(opts)).unwrap();
    assert!(csv.contains("1.23"));
    assert!(!csv.contains("1.234"));
}

#[test]
fn test_no_headers() {
    let matrix = array![[1.0, 2.0]];
    let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

    let opts = ExportOptions {
        include_headers: false,
        ..Default::default()
    };

    let csv = to_csv_string(&dm, Some(opts)).unwrap();
    assert!(!csv.contains("A,B"));
    assert!(csv.contains("1.000000,2.000000"));
}

#[test]
fn test_write_csv_to_file() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.csv");

    let matrix = array![[1.0, 2.0]];
    let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

    write_csv(&file_path, &dm, None).unwrap();

    let contents = fs::read_to_string(&file_path).unwrap();
    assert!(contents.contains("A,B"));
    assert!(contents.contains("1.000000,2.000000"));
}

#[test]
fn test_write_json_to_file() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.json");

    let matrix = array![[1.0, 2.0]];
    let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

    write_json(&file_path, &dm, None).unwrap();

    let contents = fs::read_to_string(&file_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();

    assert_eq!(parsed["n_rows"], 1);
    assert_eq!(parsed["columns"], serde_json::json!(["A", "B"]));
}

#[test]
fn test_event_model_csv_export() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("events.csv");

    let factor = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0],
        vec!["A", "B", "A"],
        vec![1.0; 3],
    )
    .unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = EventModel::new(vec![Box::new(factor)], sf).unwrap();

    model.to_csv(&file_path, None).unwrap();

    let contents = fs::read_to_string(&file_path).unwrap();
    assert!(contents.contains("event_name,event_type,onset,duration,value"));
    assert!(contents.contains("condition,Categorical"));
    assert!(contents.contains("A"));
    assert!(contents.contains("B"));
}

#[test]
fn test_event_model_json_export() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("events.json");

    let factor =
        EventFactor::new("condition", vec![1.0, 2.0], vec!["A", "B"], vec![1.0; 2]).unwrap();

    let variable =
        EventVariable::new("rating", vec![1.0, 2.0], vec![5.0, 7.0], vec![1.0; 2]).unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = EventModel::new(vec![Box::new(factor), Box::new(variable)], sf)
        .unwrap()
        .with_name("test_model");

    model.to_json(&file_path, None).unwrap();

    let contents = fs::read_to_string(&file_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();

    assert_eq!(parsed["name"], "test_model");
    assert_eq!(parsed["events"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["events"][0]["name"], "condition");
    assert_eq!(parsed["events"][0]["event_type"], "Categorical");
    assert_eq!(parsed["events"][1]["name"], "rating");
    assert_eq!(parsed["events"][1]["event_type"], "Continuous");
}

#[test]
fn test_dataframe_csv_export() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("dataframe.csv");

    let factor = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0],
        vec!["A", "B", "A"],
        vec![1.0; 3],
    )
    .unwrap();

    let df = factor.to_dataframe();
    df.to_csv(&file_path, None).unwrap();

    let contents = fs::read_to_string(&file_path).unwrap();
    assert!(contents.contains("onset,duration,condition"));
    assert!(contents.contains("condition"));
    assert!(contents.contains("A"));
    assert!(contents.contains("B"));
}

#[test]
fn test_dataframe_json_export() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("dataframe.json");

    let variable = EventVariable::new("rating", vec![1.0, 2.0], vec![5.0, 7.0], vec![1.0; 2])
        .unwrap()
        .with_center(false);

    let df = variable.to_dataframe();
    df.to_json(&file_path, None).unwrap();

    let contents = fs::read_to_string(&file_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();

    assert!(parsed.is_array());
    let array = parsed.as_array().unwrap();
    assert_eq!(array.len(), 2);
    assert_eq!(array[0]["rating"], 5.0);
    assert_eq!(array[0]["onset"], 1.0);
    assert_eq!(array[0]["duration"], 1.0);
}

#[test]
fn test_pretty_json() {
    let matrix = array![[1.0]];
    let dm = DesignMatrix::new(matrix, vec!["A".to_string()]);

    let opts_pretty = ExportOptions {
        pretty_json: true,
        ..Default::default()
    };

    let opts_compact = ExportOptions {
        pretty_json: false,
        ..Default::default()
    };

    let json_pretty = to_json_string(&dm, Some(opts_pretty)).unwrap();
    let json_compact = to_json_string(&dm, Some(opts_compact)).unwrap();

    // Pretty JSON should have newlines
    assert!(json_pretty.contains('\n'));
    // Compact JSON should not have newlines
    assert!(!json_compact.contains('\n'));
}
