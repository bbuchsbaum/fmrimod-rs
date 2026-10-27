use ndarray::ArrayView2;

/// Scan contiguous values in small vectorizable reductions. Chunk boundaries
/// preserve early rejection without a data-dependent branch for every element.
pub(crate) fn all_finite(values: ArrayView2<'_, f64>) -> bool {
    match values.as_slice_memory_order() {
        Some(values) => values
            .chunks(256)
            .all(|chunk| chunk.iter().fold(true, |finite, &v| finite & v.is_finite())),
        None => values.iter().all(|v| v.is_finite()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array2, ShapeBuilder};
    #[test]
    fn rejects_every_nonfinite_position_and_accepts_finite_extremes() {
        let mut a = Array2::from_elem((7, 137), f64::MAX);
        assert!(all_finite(a.view()));
        for v in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff0000000000001),
        ] {
            for j in 0..a.len() {
                a.as_slice_mut().unwrap()[j] = v;
                assert!(!all_finite(a.view()));
                a.as_slice_mut().unwrap()[j] = f64::from_bits(1);
            }
        }
        assert!(all_finite(a.view()));
        let mut column_major = Array2::zeros((9, 37).f());
        assert!(all_finite(column_major.view()));
        column_major[(3, 20)] = f64::NAN;
        assert!(!all_finite(column_major.view()));
        let mut padded = Array2::zeros((8, 18));
        padded[(3, 6)] = f64::NEG_INFINITY;
        assert!(!all_finite(padded.slice(s![..;-2,..;3])));
        // An excluded invalid element must not contaminate a strided selection.
        assert!(all_finite(padded.slice(s![1..;2,1..;2])));
        let scalar = ndarray::array![[f64::NAN]];
        assert!(!all_finite(scalar.broadcast((300, 1000)).unwrap()));
    }
}
