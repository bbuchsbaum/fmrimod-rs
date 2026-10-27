//! Input/output functionality for fmri_design.
//!
//! This module provides functions for importing and exporting data
//! in various formats including CSV and JSON.

pub mod export;
pub mod import;

pub use export::{
    to_csv_string, to_json_string, write_csv, write_json, DataFrameExport, EventModelExport,
    ExportFormat, ExportOptions,
};

pub use import::{
    load_events_csv, load_events_json, read_csv, read_json_dataframe, read_json_events,
    EventModelImport, ImportOptions,
};
