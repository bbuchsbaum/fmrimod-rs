//! Basis function implementations.
//!
//! This module re-exports the core basis types used throughout the crate.

pub mod dct;
pub mod identity;
pub mod polynomial;
pub mod spline;
pub mod sub_basis;
pub mod transform;

pub use dct::DctBasis;
pub use identity::IdentityBasis;
pub use polynomial::{NaturalPolynomialBasis, PolynomialBasis};
pub use spline::{BSplineBasis, NaturalSplineBasis, SplineBasis};
pub use sub_basis::{sub_basis, sub_basis_slice, SubBasis};
pub use transform::{Ident, RobustScale, RobustScaleMethod, Scale, ScaleWithin, Standardized};

// Convenience aliases
pub type Poly = PolynomialBasis;
pub type Bs = BSplineBasis;
pub type Dct = DctBasis;
