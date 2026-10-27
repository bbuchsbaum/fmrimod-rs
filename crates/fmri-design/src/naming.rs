//! Naming utilities for design matrix column names.
//!
//! This module implements the naming scheme from the R fmri_design package,
//! ensuring consistent and R-compatible column names.

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

lazy_static! {
    /// Regex for invalid characters in names
    static ref INVALID_CHARS: Regex = Regex::new(r"[^a-zA-Z0-9._]").unwrap();
    /// Regex for multiple underscores (used when dots are not allowed)
    static ref MULTI_UNDERSCORE: Regex = Regex::new(r"_+").unwrap();
}

/// Zero-pad numbers for consistent width.
///
/// # Arguments
/// * `i` - Number to pad
/// * `n_total` - Total count (determines width)
///
/// # Returns
/// Zero-padded string
pub fn zeropad(i: usize, n_total: usize) -> String {
    let width = if n_total < 1 {
        1
    } else {
        let log_width = n_total.to_string().len();
        if n_total > 1 {
            log_width.max(2)
        } else {
            log_width
        }
    };

    format!("{:0width$}", i, width = width)
}

/// Sanitize strings for use in R/Python names.
///
/// This mirrors the behavior of R's `make.names(x, unique = FALSE)` with an
/// additional `allow_dot` flag that, when `false`, replaces dots with
/// underscores and collapses multiple underscores.
///
/// # Arguments
/// * `x` - String to sanitize
/// * `allow_dot` - If false, replace dots with underscores
///
/// # Returns
/// Sanitized string
pub fn sanitize(x: &str, allow_dot: bool) -> String {
    // Handle empty upfront to match R's make.names behavior of never returning
    // an empty name.
    if x.is_empty() {
        return "X".to_string();
    }

    // First pass: replace invalid characters.
    //
    // When dots are allowed, we map invalid characters to '.' to match
    // `make.names()`; when not allowed we map them to '_' and normalize
    // dots later.
    let replacement = if allow_dot { "." } else { "_" };
    let mut sanitized = INVALID_CHARS.replace_all(x, replacement).to_string();

    // Enforce R's leading character rules:
    // - Name must begin with a letter or '.'
    // - A name starting with '.' cannot be followed by a digit
    //
    // If those rules are violated, prefix with 'X'.
    let mut chars = sanitized.chars();
    if let Some(first) = chars.next() {
        let second = chars.next();
        let invalid_first = !first.is_ascii_alphabetic() && first != '.';
        let dot_followed_by_digit =
            first == '.' && second.map(|c| c.is_ascii_digit()).unwrap_or(false);
        if invalid_first || dot_followed_by_digit {
            sanitized.insert(0, 'X');
        }
    } else {
        // Should not happen because we handled empty above, but be defensive.
        sanitized = "X".to_string();
    }

    // When dots are not allowed, normalize them to underscores and collapse
    // multiple underscores, mirroring the R helper.
    if !allow_dot {
        sanitized = sanitized.replace('.', "_");
        sanitized = MULTI_UNDERSCORE.replace_all(&sanitized, "_").to_string();
        sanitized = sanitized.trim_matches('_').to_string();
        if sanitized.is_empty() {
            sanitized = "X".to_string();
        }
    }

    sanitized
}

/// Sanitize factor level.
///
/// Similar to [`sanitize()`] but removes a leading 'X' that was introduced
/// for levels that originally began with a digit. This mirrors the R helper
/// `sanitize_level()` used for factor levels.
///
/// # Arguments
/// * `level` - Factor level to sanitize
///
/// # Returns
/// Sanitized level
pub fn sanitize_level(level: &str) -> String {
    let sanitized = sanitize(level, true);

    // If original started with digit and sanitized starts with X, remove it
    if let Some(first_char) = level.chars().next() {
        if first_char.is_numeric() && sanitized.starts_with('X') {
            return sanitized[1..].to_string();
        }
    }

    sanitized
}

/// Create basis function suffix.
///
/// # Arguments
/// * `j` - Basis index (1-based)
/// * `nb` - Total number of basis functions
///
/// # Returns
/// Suffix like '_b01', '_b02'
pub fn basis_suffix(j: usize, nb: usize) -> String {
    format!("_b{}", zeropad(j, nb))
}

/// Create feature suffix.
///
/// # Arguments
/// * `j` - Feature index (1-based)
/// * `nf` - Total number of features
///
/// # Returns
/// Suffix like 'f01', 'f02'
pub fn feature_suffix(j: usize, nf: usize) -> String {
    format!("f{}", zeropad(j, nf))
}

/// Make tags unique using # separator.
///
/// # Arguments
/// * `tags` - Tags to make unique
///
/// # Returns
/// Unique tags
pub fn make_unique_tags(tags: &[String]) -> Vec<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut result = Vec::new();

    for tag in tags {
        if let Some(count) = seen.get_mut(tag) {
            *count += 1;
            result.push(format!("{}#{}", tag, *count - 1));
        } else {
            seen.insert(tag.clone(), 1);
            result.push(tag.clone());
        }
    }

    result
}

/// Generate term tag.
///
/// # Arguments
/// * `term_name` - Explicit term name/id
/// * `event_names` - Event variable names
/// * `hrf_name` - HRF name (for suffix)
/// * `basis_type` - Basis type (e.g., 'poly', 'spline')
/// * `existing_tags` - Already used tags for uniqueness
///
/// # Returns
/// Term tag (None for identity terms)
pub fn make_term_tag(
    term_name: Option<&str>,
    event_names: Option<&[String]>,
    _hrf_name: Option<&str>,
    basis_type: Option<&str>,
    existing_tags: Option<&[String]>,
) -> Option<String> {
    let tag_base = if let Some(name) = term_name {
        // Explicit name provided
        sanitize(name, false)
    } else if let Some(names) = event_names {
        if names.is_empty() {
            return None;
        }

        // Build from event names
        if let (Some(basis), true) = (basis_type, names.len() == 1) {
            // Special handling for basis functions
            let var_name = sanitize(&names[0], false);
            let basis_prefix = match basis.to_lowercase().as_str() {
                "poly" | "polynomial" => "poly",
                "bspline" | "spline" => "bs",
                "scale" => "z",
                "standardized" => "std",
                "robustscale" => "robz",
                _ => basis,
            };
            format!("{}_{}", basis_prefix, var_name)
        } else {
            // Join event names
            names
                .iter()
                .map(|name| sanitize(name, false))
                .collect::<Vec<_>>()
                .join("_")
        }
    } else {
        // No information - return None (identity term)
        return None;
    };

    // Ensure not empty
    let tag_base = if tag_base.is_empty() {
        "empty_id_tag".to_string()
    } else {
        tag_base
    };

    // Make unique if needed
    if let Some(existing) = existing_tags {
        let mut all_tags: Vec<String> = existing.to_vec();
        all_tags.push(tag_base);
        let unique_tags = make_unique_tags(&all_tags);
        Some(unique_tags.last().unwrap().clone())
    } else {
        Some(tag_base)
    }
}

/// Create factor level token.
///
/// # Arguments
/// * `var` - Variable name
/// * `level` - Level name
///
/// # Returns
/// Token like 'var.level'
pub fn level_token(var: &str, level: &str) -> String {
    let s_var = sanitize(var, true);
    let s_level = sanitize_level(level);
    format!("{}.{}", s_var, s_level)
}

/// Sanitize continuous column name token.
///
/// # Arguments
/// * `colname` - Proposed column name
///
/// # Returns
/// Sanitized token
pub fn continuous_token(colname: &str) -> String {
    sanitize(colname, true)
}

/// Combine tokens into condition tag.
///
/// # Arguments
/// * `tokens` - Tokens to combine
///
/// # Returns
/// Combined condition tag
pub fn make_cond_tag(tokens: &[String]) -> String {
    tokens.join("_")
}

/// Add basis suffix to condition tags if needed.
///
/// This mirrors the R helper `add_basis()`, which uses `outer()` and then
/// flattens in column-major order. For multiple condition tags this means
/// the first basis index varies slowest.
///
/// # Arguments
/// * `cond_tags` - Base condition tags
/// * `nb` - Number of basis functions
///
/// # Returns
/// Expanded tags with basis suffixes
pub fn add_basis_suffix(cond_tags: &[String], nb: usize) -> Vec<String> {
    if nb <= 1 {
        return cond_tags.to_vec();
    }

    let mut result = Vec::new();
    // Column-major ordering as in R's outer(cond_tags, suffixes, paste0)
    for b in 1..=nb {
        let suffix = basis_suffix(b, nb);
        for tag in cond_tags {
            result.push(format!("{tag}{suffix}"));
        }
    }

    result
}

/// Compose final column names.
///
/// This is the single source of truth for creating design matrix column names.
///
/// # Arguments
/// * `term_tag` - Sanitized, unique term tag (None for identity terms)
/// * `cond_tags` - Base condition tags (without basis suffix)
/// * `nb` - Number of basis functions
///
/// # Returns
/// Final column names
pub fn make_column_names(term_tag: Option<&str>, cond_tags: &[String], nb: usize) -> Vec<String> {
    // Add basis suffix if needed
    let full_cond_tags = add_basis_suffix(cond_tags, nb);

    // Combine term tag and condition tags
    if let Some(tag) = term_tag {
        // Normal case: term_tag_condition_tag[_b##]
        full_cond_tags
            .iter()
            .map(|cond_tag| format!("{}_{}", tag, cond_tag))
            .collect()
    } else {
        // Identity term - use condition tags directly
        full_cond_tags
    }
}

/// Ensure column names are unique.
///
/// Uses R-style suffixing (.1, .2, etc.) for duplicates.
///
/// # Arguments
/// * `colnames` - Column names to make unique
///
/// # Returns
/// Unique column names
pub fn make_unique_colnames(colnames: &[String]) -> Vec<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut result = Vec::new();

    for name in colnames {
        if let Some(count) = seen.get_mut(name) {
            *count += 1;
            result.push(format!("{}.{}", name, count));
        } else {
            seen.insert(name.clone(), 0);
            result.push(name.clone());
        }
    }

    result
}

/// Convert long column names to short names.
///
/// # Arguments
/// * `longnames_list` - Long column names to shorten
/// * `acronym` - Acronym to use for the model (default: first letters)
///
/// # Returns
/// Shortened column names
pub fn shortnames(longnames_list: &[String], acronym: Option<&str>) -> Vec<String> {
    if longnames_list.is_empty() {
        return Vec::new();
    }

    // Extract unique term tags (before underscores)
    let mut term_tags: Vec<String> = longnames_list
        .iter()
        .filter_map(|name| name.split('_').next().map(|s| s.to_string()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    term_tags.sort();

    // Create short version of each term tag
    let mut term_map = HashMap::new();
    if let Some(acr) = acronym {
        // Use provided acronym
        for (i, tag) in term_tags.iter().enumerate() {
            let short = if term_tags.len() > 1 {
                format!("{}{}", acr, i + 1)
            } else {
                acr.to_string()
            };
            term_map.insert(tag.clone(), short);
        }
    } else {
        // Generate from first letters
        for tag in &term_tags {
            // Extract capital letters or first letter of each word part
            let short: String = tag.chars().filter(|c| c.is_uppercase()).collect();

            let short = if short.is_empty() {
                // No capitals, use first letter of each underscore-separated part
                let parts: Vec<&str> = tag.split('_').collect();
                let short: String = parts
                    .iter()
                    .filter_map(|p| p.chars().next())
                    .map(|c| c.to_uppercase().to_string())
                    .collect();

                if short.is_empty() {
                    // Fallback to first 3 chars
                    tag.chars().take(3).collect::<String>().to_uppercase()
                } else {
                    short
                }
            } else {
                short
            };

            term_map.insert(tag.clone(), short);
        }
    }

    // Apply mapping to create short names
    longnames_list
        .iter()
        .map(|name| {
            if let Some((term_tag, rest)) = name.split_once('_') {
                if let Some(short) = term_map.get(term_tag) {
                    format!("{}_{}", short, rest)
                } else {
                    name.clone()
                }
            } else {
                name.clone()
            }
        })
        .collect()
}

/// Alias for make_column_names for consistency.
///
/// # Arguments
/// * `term_tag` - Sanitized, unique term tag (None for identity terms)
/// * `cond_tags` - Base condition tags (without basis suffix)
/// * `nb` - Number of basis functions
///
/// # Returns
/// Long column names
pub fn longnames(term_tag: Option<&str>, cond_tags: &[String], nb: usize) -> Vec<String> {
    make_column_names(term_tag, cond_tags, nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeropad() {
        // These mirror the expectations in the R tests for zeropad()
        assert_eq!(zeropad(3, 9), "03");
        assert_eq!(zeropad(9, 9), "09");
        assert_eq!(zeropad(3, 10), "03");
        assert_eq!(zeropad(12, 99), "12");
        assert_eq!(zeropad(12, 100), "012");
        assert_eq!(zeropad(99, 100), "099");
        assert_eq!(zeropad(1, 10), "01");
        assert_eq!(zeropad(1, 1), "1");
        assert_eq!(zeropad(5, 0), "5");
    }

    #[test]
    fn test_sanitize() {
        // Match R's make.names semantics for allow_dot = TRUE
        assert_eq!(sanitize("a.b c", true), "a.b.c");
        assert_eq!(sanitize("hello world", true), "hello.world");
        assert_eq!(sanitize("123abc", true), "X123abc");
        assert_eq!(sanitize("_test", true), "X_test");
        // When allow_dot = FALSE, dots are normalized to single underscores
        assert_eq!(sanitize("test.name", false), "test_name");
        assert_eq!(sanitize("test__name", false), "test_name");
        assert_eq!(sanitize("", true), "X");
    }

    #[test]
    fn test_sanitize_level() {
        assert_eq!(sanitize_level("123"), "123");
        assert_eq!(sanitize_level("abc"), "abc");
        assert_eq!(sanitize_level("1.5"), "1.5");
    }

    #[test]
    fn test_make_unique_tags() {
        let tags = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let unique = make_unique_tags(&tags);
        assert_eq!(unique, vec!["a", "b", "a#1"]);
    }

    #[test]
    fn test_make_column_names() {
        let cond_tags = vec!["cond1".to_string(), "cond2".to_string()];

        // With term tag
        let names = make_column_names(Some("term"), &cond_tags, 1);
        assert_eq!(names, vec!["term_cond1", "term_cond2"]);

        // Without term tag (identity)
        let names = make_column_names(None, &cond_tags, 1);
        assert_eq!(names, vec!["cond1", "cond2"]);

        // With basis functions
        let names = make_column_names(Some("term"), &cond_tags[..1], 2);
        assert_eq!(names, vec!["term_cond1_b01", "term_cond1_b02"]);
    }

    #[test]
    fn test_make_unique_colnames() {
        let colnames = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let unique = make_unique_colnames(&colnames);
        assert_eq!(unique, vec!["a", "b", "a.1"]);
    }
}
