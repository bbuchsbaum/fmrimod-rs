//! HRF registry for managing available HRF types.

use crate::error::{HrfError, Result};
use crate::hrf::core::Hrf;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Type alias for HRF factory function.
pub type HrfFactory = Box<dyn Fn() -> Box<dyn Hrf> + Send + Sync>;

// Global HRF registry.
lazy_static! {
    static ref HRF_REGISTRY: RwLock<HashMap<String, Arc<HrfFactory>>> = {
        let mut registry = HashMap::new();
        initialize_registry(&mut registry);
        RwLock::new(registry)
    };
}

/// Initialize the registry with predefined HRFs.
fn initialize_registry(registry: &mut HashMap<String, Arc<HrfFactory>>) {
    use crate::hrf::library::{get_predefined_hrf, list_predefined_hrfs};
    for name in list_predefined_hrfs() {
        let key = name.clone();
        let factory: HrfFactory =
            Box::new(move || get_predefined_hrf(&key).expect("predefined HRF names are immutable"));
        registry.insert(name, Arc::new(factory));
    }
    registry.insert(
        "spm_with_derivatives".into(),
        Arc::new(Box::new(crate::hrf::functions::spm_hrf_with_derivatives)),
    );
}

/// Get an HRF from the registry by name.
///
/// # Arguments
/// * `name` - Name of the HRF (case-insensitive)
///
/// # Errors
/// Returns error if HRF name not found in registry.
pub fn get_hrf(name: &str) -> Result<Box<dyn Hrf>> {
    let name_lower = name.to_lowercase();

    let registry = HRF_REGISTRY
        .read()
        .map_err(|_| HrfError::InvalidInput("Failed to acquire registry lock".into()))?;

    if let Some(factory) = registry.get(&name_lower).cloned() {
        // A custom factory may itself use or modify the registry.
        drop(registry);
        Ok(factory())
    } else {
        let mut available: Vec<_> = registry.keys().cloned().collect();
        available.sort();
        Err(HrfError::HrfNotFound(format!(
            "HRF '{}' not found. Available: {}",
            name,
            available.join(", ")
        )))
    }
}

/// List all available HRF names in the registry.
pub fn list_available_hrfs() -> Vec<String> {
    match HRF_REGISTRY.read() {
        Ok(registry) => {
            let mut names: Vec<String> = registry.keys().cloned().collect();
            names.sort();
            names
        }
        Err(_) => vec![],
    }
}

/// Register a custom HRF in the global registry.
///
/// # Arguments
/// * `name` - Name for the HRF
/// * `factory` - Factory function that creates the HRF
/// * `force` - If true, overwrite existing entries
///
/// # Errors
/// Returns error if name already exists and force=false.
pub fn register_hrf<F>(name: &str, factory: F, force: bool) -> Result<()>
where
    F: Fn() -> Box<dyn Hrf> + Send + Sync + 'static,
{
    let name_lower = name.to_lowercase();

    let mut registry = HRF_REGISTRY
        .write()
        .map_err(|_| HrfError::InvalidInput("Failed to acquire registry lock".into()))?;

    if !force && registry.contains_key(&name_lower) {
        return Err(HrfError::InvalidInput(format!(
            "HRF '{}' already exists in registry. Use force=true to overwrite.",
            name
        )));
    }

    registry.insert(name_lower, Arc::new(Box::new(factory)));
    Ok(())
}

/// Remove an HRF from the registry.
///
/// # Arguments
/// * `name` - Name of the HRF to remove
///
/// # Errors
/// Returns error if HRF not found in registry.
pub fn remove_hrf(name: &str) -> Result<()> {
    let name_lower = name.to_lowercase();

    let mut registry = HRF_REGISTRY
        .write()
        .map_err(|_| HrfError::InvalidInput("Failed to acquire registry lock".into()))?;

    match registry.remove(&name_lower) {
        Some(_) => Ok(()),
        None => Err(HrfError::HrfNotFound(format!(
            "HRF '{}' not found in registry",
            name
        ))),
    }
}

/// Clear the HRF registry.
///
/// # Arguments
/// * `keep_predefined` - If true, keep pre-defined HRFs and only clear custom ones
pub fn clear_registry(keep_predefined: bool) -> Result<()> {
    let mut registry = HRF_REGISTRY
        .write()
        .map_err(|_| HrfError::InvalidInput("Failed to acquire registry lock".into()))?;

    registry.clear();

    if keep_predefined {
        initialize_registry(&mut registry);
    }

    Ok(())
}

/// Register a parametric HRF generator.
///
/// This is a convenience function for registering HRFs that take parameters.
pub fn register_parametric_hrf<F>(
    name: &str,
    default_params: HashMap<String, f64>,
    generator: F,
    force: bool,
) -> Result<()>
where
    F: Fn(&HashMap<String, f64>) -> Box<dyn Hrf> + Send + Sync + 'static,
{
    let factory = move || generator(&default_params);
    register_hrf(name, factory, force)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_predefined_hrf() {
        let hrf = get_hrf("spm_canonical").unwrap();
        assert_eq!(hrf.name(), "spm_canonical");
        assert_eq!(hrf.nbasis(), 1);
    }

    #[test]
    fn test_get_hrf_case_insensitive() {
        let hrf1 = get_hrf("SPM_CANONICAL").unwrap();
        let hrf2 = get_hrf("spm_canonical").unwrap();
        assert_eq!(hrf1.name(), hrf2.name());
    }

    #[test]
    fn test_list_available_hrfs() {
        let hrfs = list_available_hrfs();
        assert!(hrfs.contains(&"spm_canonical".to_string()));
        assert!(hrfs.contains(&"gamma".to_string()));
        assert!(hrfs.contains(&"bspline".to_string()));
    }

    #[test]
    fn test_register_custom_hrf() {
        use crate::hrf::functions::gaussian_hrf;

        let factory = || gaussian_hrf(10.0, 3.0, Some(true));
        register_hrf("custom_gaussian", factory, false).unwrap();

        let hrf = get_hrf("custom_gaussian").unwrap();
        assert_eq!(hrf.name(), "gaussian_hrf");

        // Clean up
        remove_hrf("custom_gaussian").unwrap();
    }

    #[test]
    fn test_register_duplicate_fails() {
        use crate::hrf::functions::gaussian_hrf;

        let factory = || gaussian_hrf(10.0, 3.0, Some(true));
        let result = register_hrf("spm", factory, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_hrf() {
        use crate::hrf::functions::gaussian_hrf;

        // Register a custom HRF
        let factory = || gaussian_hrf(10.0, 3.0, Some(true));
        register_hrf("temp_hrf", factory, false).unwrap();

        // Verify it exists
        assert!(get_hrf("temp_hrf").is_ok());

        // Remove it
        remove_hrf("temp_hrf").unwrap();

        // Verify it's gone
        assert!(get_hrf("temp_hrf").is_err());
    }

    #[test]
    fn test_hrf_not_found() {
        let result = get_hrf("nonexistent_hrf");
        assert!(result.is_err());

        if let Err(HrfError::HrfNotFound(msg)) = result {
            assert!(msg.contains("nonexistent_hrf"));
            assert!(msg.contains("Available:"));
        } else {
            panic!("Expected HrfNotFound error");
        }
    }
}
