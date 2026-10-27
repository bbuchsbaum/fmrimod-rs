//! Matrix backend selection. Both backends use f64 without changing rank policy.
use ndarray::{linalg::general_mat_mul, ArrayView2, ArrayViewMut2};

/// Backend for the two prepared-fit matrix products.
/// Accelerate is available with the `accelerate` feature on macOS. Strided
/// inputs that cannot be passed directly to CBLAS use the portable kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixBackend {
    #[cfg_attr(not(all(feature = "accelerate", target_os = "macos")), default)]
    Portable,
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    #[default]
    Accelerate,
}
impl MatrixBackend {
    pub(crate) fn multiply(
        self,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        mut c: ArrayViewMut2<'_, f64>,
    ) {
        assert_eq!(a.ncols(), b.nrows());
        assert_eq!(c.dim(), (a.nrows(), b.ncols()));
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        if self == Self::Accelerate
            && a.is_standard_layout()
            && b.is_standard_layout()
            && c.is_standard_layout()
        {
            if let (Ok(m), Ok(n), Ok(k)) = (
                i32::try_from(a.nrows()),
                i32::try_from(b.ncols()),
                i32::try_from(a.ncols()),
            ) {
                // SAFETY: all views are contiguous row-major f64 arrays with
                // checked matching dimensions representable by the LP64 ABI.
                // Rust's mutable view excludes aliasing with either input.
                unsafe {
                    cblas_dgemm(
                        101,
                        111,
                        111,
                        m,
                        n,
                        k,
                        1.,
                        a.as_ptr(),
                        k,
                        b.as_ptr(),
                        n,
                        0.,
                        c.as_mut_ptr(),
                        n,
                    );
                }
                return;
            }
        }
        general_mat_mul(1., &a, &b, 0., &mut c);
    }
}
#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}
