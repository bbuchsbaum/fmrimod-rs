//! Isolated process: allocation instrumentation does not observe parallel tests.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};
struct Counter;
static LARGEST: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, n: usize) -> *mut u8 {
        LARGEST.fetch_max(n, Ordering::Relaxed);
        unsafe { System.realloc(p, layout, n) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
#[test]
fn broadcast_nonfinite_data_is_rejected_before_numeric_allocation() {
    let x = ndarray::Array2::ones((128, 1));
    let model = fmrireg::LinearModel::new(x.view(), 1e-12).unwrap();
    let seed = ndarray::array![[f64::NAN]];
    let large = seed.broadcast((128, 100_000)).unwrap();
    LARGEST.store(0, Ordering::SeqCst);
    assert!(model.fit(large).is_err());
    assert!(
        LARGEST.load(Ordering::SeqCst) < 1024,
        "nonfinite input allocated fitting buffers"
    );
}
