//! AFNI GLT (General Linear Test) conversion functionality.
//!
//! This module provides functions to convert contrasts to AFNI's GLT format
//! for use with 3dDeconvolve and 3dREMLfit.

use crate::contrast::ComputedContrast;
use crate::types::ValidationError;
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// GLT conversion result
#[derive(Debug, Clone)]
pub struct GltResult {
    /// GLT string(s) in AFNI format
    pub glt_str: Vec<String>,
    /// GLT name(s) for AFNI commands
    pub name: Vec<String>,
    /// Original contrast
    pub source: GltSource,
}

/// Source of GLT conversion
#[derive(Debug, Clone)]
pub enum GltSource {
    /// From computed contrast
    ComputedContrast(ComputedContrast),
    /// From raw weights and names
    Raw {
        weights: Array2<f64>,
        condnames: Vec<String>,
        name: String,
    },
}

/// Convert contrast weights to AFNI GLT format.
///
/// Takes non-zero weights and creates terms like "weight*condname".
fn format_glt_string(
    weights: &ArrayView1<f64>,
    condnames: &[String],
) -> Result<String, ValidationError> {
    if weights.len() != condnames.len() {
        return Err(ValidationError::LengthMismatch {
            expected: condnames.len(),
            actual: weights.len(),
        });
    }

    let mut terms = Vec::new();

    for (i, &weight) in weights.iter().enumerate() {
        // Skip near-zero weights
        if weight.abs() > 1e-10 {
            // Format weight with appropriate precision
            let weight_str = if weight == 1.0 {
                "1".to_string()
            } else if weight == -1.0 {
                "-1".to_string()
            } else {
                // Format with 4 significant figures
                format!("{:.4}", weight)
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_string()
            };

            terms.push(format!("{}*{}", weight_str, condnames[i]));
        }
    }

    if terms.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "All contrast weights are zero".to_string(),
        ));
    }

    Ok(terms.join(" "))
}

/// Convert computed contrast to GLT format.
pub fn to_glt(contrast: &ComputedContrast) -> Result<GltResult, ValidationError> {
    let mut glt_strings = Vec::new();
    let mut names = Vec::new();

    // Handle single or multiple contrasts
    if contrast.weights.ndim() == 1
        || (contrast.weights.ndim() == 2 && contrast.weights.ncols() == 1)
    {
        // Single contrast (t-test)
        let weights_view = if contrast.weights.ndim() == 1 {
            // 1D array
            let weights_1d = contrast.weights.as_slice().ok_or_else(|| {
                ValidationError::InvalidParameter("Weights must be contiguous".to_string())
            })?;
            ArrayView1::from(weights_1d)
        } else {
            // 2D array with one column
            contrast.weights.column(0)
        };
        let glt_str = format_glt_string(&weights_view, &contrast.condnames)?;
        glt_strings.push(glt_str);
        names.push(format!("GLT_{}", sanitize_name(&contrast.name)));
    } else {
        // Multiple contrasts (F-test)
        // Check if this is a case where columns represent contrasts (e.g., oneway)
        if contrast.weights.nrows() == contrast.condnames.len() {
            // Iterate over columns (each column is a contrast)
            for i in 0..contrast.weights.ncols() {
                let col = contrast.weights.column(i);
                let glt_str = format_glt_string(&col, &contrast.condnames)?;
                glt_strings.push(glt_str);
                names.push(format!("GLT_{}_{}", sanitize_name(&contrast.name), i + 1));
            }
        } else {
            // Iterate over rows (each row is a contrast)
            for i in 0..contrast.weights.nrows() {
                let row = contrast.weights.row(i);
                let glt_str = format_glt_string(&row, &contrast.condnames)?;
                glt_strings.push(glt_str);
                names.push(format!("GLT_{}_{}", sanitize_name(&contrast.name), i + 1));
            }
        }
    }

    Ok(GltResult {
        glt_str: glt_strings,
        name: names,
        source: GltSource::ComputedContrast(contrast.clone()),
    })
}

/// Convert raw weights to GLT format.
pub fn weights_to_glt(
    weights: &ArrayView2<f64>,
    condnames: &[String],
    name: &str,
) -> Result<GltResult, ValidationError> {
    let mut glt_strings = Vec::new();
    let mut names = Vec::new();

    if weights.nrows() == 1 {
        // Single contrast
        let glt_str = format_glt_string(&weights.row(0), condnames)?;
        glt_strings.push(glt_str);
        names.push(format!("GLT_{}", sanitize_name(name)));
    } else {
        // Multiple contrasts
        for i in 0..weights.nrows() {
            let row = weights.row(i);
            let glt_str = format_glt_string(&row, condnames)?;
            glt_strings.push(glt_str);
            names.push(format!("GLT_{}_{}", sanitize_name(name), i + 1));
        }
    }

    Ok(GltResult {
        glt_str: glt_strings,
        name: names,
        source: GltSource::Raw {
            weights: weights.to_owned(),
            condnames: condnames.to_vec(),
            name: name.to_string(),
        },
    })
}

/// Write GLT to file(s).
///
/// For single contrasts, writes to a single file.
/// For multiple contrasts (F-tests), can write to separate files or a single file with comments.
pub fn write_glt(
    glt: &GltResult,
    path: impl AsRef<Path>,
    separate_files: bool,
) -> Result<(), std::io::Error> {
    let path = path.as_ref();

    if glt.glt_str.len() == 1 || !separate_files {
        // Write to single file
        let mut file = File::create(path)?;

        // Write header comment
        writeln!(file, "# AFNI GLT contrast specification")?;
        writeln!(file, "# Generated by fmri_design-rs")?;
        writeln!(file, "#")?;

        for (i, (glt_str, name)) in glt.glt_str.iter().zip(glt.name.iter()).enumerate() {
            if glt.glt_str.len() > 1 {
                writeln!(
                    file,
                    "# Contrast {} of {}: {}",
                    i + 1,
                    glt.glt_str.len(),
                    name
                )?;
            } else {
                writeln!(file, "# Contrast: {}", name)?;
            }
            writeln!(file, "{}", glt_str)?;
            if i < glt.glt_str.len() - 1 {
                writeln!(file)?; // Empty line between contrasts
            }
        }
    } else {
        // Write to separate files
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("glt");
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("txt");
        let parent = path.parent().unwrap_or(Path::new("."));

        for (i, (glt_str, name)) in glt.glt_str.iter().zip(glt.name.iter()).enumerate() {
            let filename = format!("{}_{}.{}", stem, i + 1, extension);
            let filepath = parent.join(filename);
            let mut file = File::create(filepath)?;

            writeln!(file, "# AFNI GLT contrast specification")?;
            writeln!(file, "# Contrast: {}", name)?;
            writeln!(file, "{}", glt_str)?;
        }
    }

    Ok(())
}

/// Format GLT for AFNI command line.
///
/// Generates the -gltsym and -glt_label options for AFNI commands.
pub fn format_afni_gltsym(glt: &GltResult) -> Vec<String> {
    let mut commands = Vec::new();

    for (i, (glt_str, name)) in glt.glt_str.iter().zip(glt.name.iter()).enumerate() {
        let cmd = format!("-gltsym 'SYM: {}' -glt_label {} {}", glt_str, i + 1, name);
        commands.push(cmd);
    }

    commands
}

/// Format GLT for AFNI command line (as a single string).
pub fn format_afni_gltsym_single(glt: &GltResult) -> String {
    format_afni_gltsym(glt).join(" ")
}

/// Sanitize name for use in AFNI.
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_format_glt_string() {
        let weights = array![1.0, -1.0, 0.0];
        let condnames = vec![
            "condA".to_string(),
            "condB".to_string(),
            "condC".to_string(),
        ];

        let glt = format_glt_string(&weights.view(), &condnames).unwrap();
        assert_eq!(glt, "1*condA -1*condB");
    }

    #[test]
    fn test_format_glt_string_fractional() {
        let weights = array![0.5, -0.3333, 0.0];
        let condnames = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let glt = format_glt_string(&weights.view(), &condnames).unwrap();
        assert_eq!(glt, "0.5*A -0.3333*B");
    }

    #[test]
    fn test_weights_to_glt_single() {
        let weights = array![[1.0, -1.0, 0.0]];
        let condnames = vec![
            "condA".to_string(),
            "condB".to_string(),
            "condC".to_string(),
        ];

        let glt = weights_to_glt(&weights.view(), &condnames, "A_vs_B").unwrap();

        assert_eq!(glt.glt_str.len(), 1);
        assert_eq!(glt.glt_str[0], "1*condA -1*condB");
        assert_eq!(glt.name[0], "GLT_A_vs_B");
    }

    #[test]
    fn test_weights_to_glt_multiple() {
        let weights = array![[1.0, -0.5, -0.5], [0.0, 1.0, -1.0]];
        let condnames = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let glt = weights_to_glt(&weights.view(), &condnames, "main_effect").unwrap();

        assert_eq!(glt.glt_str.len(), 2);
        assert_eq!(glt.glt_str[0], "1*A -0.5*B -0.5*C");
        assert_eq!(glt.glt_str[1], "1*B -1*C");
        assert_eq!(glt.name[0], "GLT_main_effect_1");
        assert_eq!(glt.name[1], "GLT_main_effect_2");
    }

    #[test]
    fn test_format_afni_gltsym() {
        let weights = array![[1.0, -1.0]];
        let condnames = vec!["condA".to_string(), "condB".to_string()];

        let glt = weights_to_glt(&weights.view(), &condnames, "A_vs_B").unwrap();
        let cmds = format_afni_gltsym(&glt);

        assert_eq!(cmds.len(), 1);
        assert_eq!(
            cmds[0],
            "-gltsym 'SYM: 1*condA -1*condB' -glt_label 1 GLT_A_vs_B"
        );
    }

    #[test]
    fn test_zero_weights_error() {
        let weights = array![0.0, 0.0, 0.0];
        let condnames = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = format_glt_string(&weights.view(), &condnames);
        assert!(result.is_err());
    }
}
