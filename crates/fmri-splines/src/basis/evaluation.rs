//! Core spline evaluation algorithms ported from C

use crate::errors::{SplineError, SplineResult};
use crate::utils::validation::validate_finite;
use ndarray::Array1;
use num_traits::Float;

/// Difference tables for B-spline evaluation (equivalent to C diff_table)
struct DifferenceTables<T: Float> {
    left_deltas: Vec<T>,  // ldel in C code
    right_deltas: Vec<T>, // rdel in C code
}

impl<T: Float> DifferenceTables<T> {
    fn new(capacity: usize) -> Self {
        DifferenceTables {
            left_deltas: Vec::with_capacity(capacity),
            right_deltas: Vec::with_capacity(capacity),
        }
    }

    /// Compute difference table (exact port of C diff_table function)
    /// ti_start: index in knots array where ti points initially (the knot just > x)
    fn compute(&mut self, knots: &[T], ti_start: usize, x: T, order_minus_1: usize) {
        self.left_deltas.clear();
        self.right_deltas.clear();

        // Port of C code exactly:
        // register double *r = rdel, *l = ldel, *dpt = ti;
        // while (ndiff--) {
        //     *r++ = *dpt++ - x;    // advance dpt forward from ti
        //     *l++ = x - *--ti;     // move ti backward from original position
        // }

        let mut dpt_idx = ti_start; // dpt starts at ti
        let mut ti_idx = ti_start; // ti starts at the same position

        for _ in 0..order_minus_1 {
            // Right delta: *dpt++ - x (advance dpt forward)
            if dpt_idx < knots.len() {
                self.right_deltas.push(knots[dpt_idx] - x);
                dpt_idx += 1;
            } else {
                // Handle boundary case - use a safe default
                self.right_deltas.push(T::one());
            }

            // Left delta: x - *--ti (move ti backward first, then access)
            if ti_idx > 0 {
                ti_idx -= 1;
                self.left_deltas.push(x - knots[ti_idx]);
            } else {
                // Handle boundary case - use a safe default
                self.left_deltas.push(T::one());
            }
        }
    }
}

/// B-spline basis function evaluator
pub struct BSplineBasisEvaluator<T: Float> {
    order: usize,
    diff_tables: DifferenceTables<T>,
    coeff_work: Vec<T>,
}

impl<T: Float> BSplineBasisEvaluator<T> {
    /// Create a new evaluator for the given order
    pub fn new(order: usize) -> Self {
        let order_minus_1 = order - 1;
        BSplineBasisEvaluator {
            order,
            diff_tables: DifferenceTables::new(order_minus_1),
            coeff_work: vec![T::zero(); order],
        }
    }

    /// Compute B-spline basis functions (port of C basis_funcs)
    /// This function expects knots to be the full knot vector and finds the appropriate interval
    pub fn basis_functions(&mut self, knots: &[T], x: T, basis: &mut [T]) -> SplineResult<()> {
        if basis.len() != self.order {
            return Err(SplineError::dimension_mismatch(self.order, basis.len()));
        }

        // Find the knot interval: first knot that is > x
        let mut ti_idx = self.order; // Start after the first `order` knots
        while ti_idx < knots.len() && knots[ti_idx] <= x {
            ti_idx += 1;
        }

        let order_minus_1 = self.order - 1;
        self.diff_tables.compute(knots, ti_idx, x, order_minus_1);

        // Port of C code:
        // b[0] = 1.;
        // for (j = 1; j <= orderm1; j++) {
        //     saved = 0.;
        //     for (r = 0; r < j; r++) {
        //         term = b[r]/(rdel[r] + ldel[j - 1 - r]);
        //         b[r] = saved + rdel[r] * term;
        //         saved = ldel[j - 1 - r] * term;
        //     }
        //     b[j] = saved;
        // }

        basis[0] = T::one();

        for j in 1..=order_minus_1 {
            let mut saved = T::zero();
            for r in 0..j {
                if r < self.diff_tables.right_deltas.len()
                    && (j - 1 - r) < self.diff_tables.left_deltas.len()
                {
                    let denom =
                        self.diff_tables.right_deltas[r] + self.diff_tables.left_deltas[j - 1 - r];

                    if denom != T::zero() {
                        let term = basis[r] / denom;
                        basis[r] = saved + self.diff_tables.right_deltas[r] * term;
                        saved = self.diff_tables.left_deltas[j - 1 - r] * term;
                    } else {
                        // Handle degenerate case
                        basis[r] = saved;
                        saved = T::zero();
                    }
                } else {
                    // Handle index out of bounds
                    basis[r] = saved;
                    saved = T::zero();
                }
            }
            basis[j] = saved;
        }

        Ok(())
    }

    /// Evaluate spline with derivatives (port of C evaluate function)
    pub fn evaluate_with_derivatives(
        &mut self,
        knots: &[T],
        x: T,
        coefficients: &[T],
        n_deriv: usize,
    ) -> SplineResult<T> {
        if coefficients.len() != self.order {
            return Err(SplineError::dimension_mismatch(
                self.order,
                coefficients.len(),
            ));
        }

        // Copy coefficients to working array
        self.coeff_work[..coefficients.len()].copy_from_slice(coefficients);

        let mut outer = self.order - 1;

        // Compute derivatives by differencing coefficients
        // Port of C code:
        // while(nder--) {
        //     for(inner = outer, apt = a, lpt = ti - outer; inner--; apt++, lpt++)
        //         *apt = outer * (*(apt + 1) - *apt)/(*(lpt + outer) - *lpt);
        //     outer--;
        // }
        for _ in 0..n_deriv {
            for inner in 0..outer {
                // Check bounds to avoid index out of bounds
                if inner + outer < knots.len() && inner < knots.len() {
                    let knot_diff = knots[inner + outer] - knots[inner];
                    if knot_diff != T::zero() {
                        self.coeff_work[inner] = T::from(outer).unwrap()
                            * (self.coeff_work[inner + 1] - self.coeff_work[inner])
                            / knot_diff;
                    }
                }
            }
            outer -= 1;
        }

        // Evaluate using de Boor's algorithm
        // Port of C code:
        // diff_table(ti, x, (long) outer);
        // while(outer--)
        //     for(apt = a, lpt = ldel + outer, rpt = rdel, inner = outer + 1;
        //         inner--; lpt--, rpt++, apt++)
        //         *apt = (*(apt + 1) * *lpt + *apt * *rpt)/(*rpt + *lpt);

        // For this case, we need to find the knot interval for x
        let mut ti_idx = 0;
        while ti_idx < knots.len() && knots[ti_idx] <= x {
            ti_idx += 1;
        }
        self.diff_tables.compute(knots, ti_idx, x, outer);

        while outer > 0 {
            for inner in 0..outer {
                let left_del = self.diff_tables.left_deltas[outer - 1 - inner];
                let right_del = self.diff_tables.right_deltas[inner];
                let denom = right_del + left_del;

                self.coeff_work[inner] = (self.coeff_work[inner + 1] * left_del
                    + self.coeff_work[inner] * right_del)
                    / denom;
            }
            outer -= 1;
        }

        Ok(self.coeff_work[0])
    }
}

/// High-level spline value evaluation (port of C spline_value)
pub fn evaluate_spline<T: Float>(
    knots: &Array1<T>,
    coefficients: &Array1<T>,
    order: usize,
    x: &[T],
    deriv: usize,
) -> SplineResult<Array1<T>> {
    validate_finite(x, "x")?;

    if coefficients.len() + order != knots.len() {
        return Err(SplineError::dimension_mismatch(
            knots.len() - order,
            coefficients.len(),
        ));
    }

    let mut evaluator = BSplineBasisEvaluator::new(order);
    let mut results = Vec::with_capacity(x.len());

    let knot_slice = knots.as_slice().unwrap();
    let coeff_slice = coefficients.as_slice().unwrap();

    // Port of C main evaluation loop:
    // knots += *order;  /* First *order knots must be <= all x's */
    // while(n--) {
    //     while(knots <= last && *knots <= *x) {knots++; coeff++;}
    //     Memcpy(a, coeff, *order);
    //     *y++ = evaluate(knots, *x++, a, (longint) *deriv);
    // }

    for &x_val in x {
        // Find the appropriate knot interval
        let mut knot_start = order;
        let last_knot_idx = knots.len() - order;

        while knot_start <= last_knot_idx && knots[knot_start] <= x_val {
            knot_start += 1;
        }

        // Adjust for the interval we found
        let coeff_start = knot_start - order;
        let knot_interval = &knot_slice[knot_start - order..knot_start];
        let local_coeffs = &coeff_slice[coeff_start..coeff_start + order];

        let value =
            evaluator.evaluate_with_derivatives(knot_interval, x_val, local_coeffs, deriv)?;

        results.push(value);
    }

    Ok(Array1::from(results))
}

/// Linear interpolation (port of C lin_interp)
pub fn linear_interpolation<T: Float>(x: &[T], y: &[T], x0: &[T]) -> SplineResult<Array1<T>> {
    if x.len() != y.len() {
        return Err(SplineError::dimension_mismatch(x.len(), y.len()));
    }

    if x.is_empty() {
        return Err(SplineError::insufficient_data(1, 0));
    }

    validate_finite(x, "x")?;
    validate_finite(y, "y")?;
    validate_finite(x0, "x0")?;

    let mut results = Vec::with_capacity(x0.len());

    // Port of C code:
    // while(n--) {
    //     while (*x < *x0) {x++; y++;}
    //     if (x > firstx) {x--; y--;}
    //     if (*x > *x0) *y0++ = *y + (*(y+1) - *y)*(*x0 - *x)/(*(x+1) - *x);
    //     else *y0++ = *y;
    //     x0++;
    // }

    for &x0_val in x0 {
        let mut i = 0;

        // Find the interval containing x0_val
        while i < x.len() && x[i] < x0_val {
            i += 1;
        }

        if i > 0 {
            i -= 1;
        }

        let result = if i + 1 < x.len() && x[i] < x0_val {
            // Interpolate between x[i] and x[i+1]
            let dx = x[i + 1] - x[i];
            let dy = y[i + 1] - y[i];
            let t = (x0_val - x[i]) / dx;
            y[i] + t * dy
        } else {
            // Use the nearest value
            y[i]
        };

        results.push(result);
    }

    Ok(Array1::from(results))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basis_functions() {
        let mut evaluator = BSplineBasisEvaluator::new(4);

        // Create a proper B-spline knot vector: [t0, t1, t2, t3, t4, t5, t6, t7]
        // For cubic B-splines (order 4), we need repeated knots at boundaries
        // Standard example: [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let mut basis = vec![0.0; 4];

        // Evaluate at x = 1.5 (which should be in the interval [1, 2])
        evaluator.basis_functions(&knots, 1.5, &mut basis).unwrap();

        println!("Basis values: {:?}", basis);
        println!("Left deltas: {:?}", evaluator.diff_tables.left_deltas);
        println!("Right deltas: {:?}", evaluator.diff_tables.right_deltas);

        // Check that no values are NaN or infinite
        for &val in &basis {
            assert!(
                val.is_finite(),
                "Basis function value is not finite: {}",
                val
            );
        }

        // The sum of basis functions should be 1
        let sum: f64 = basis.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_interpolation() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let x0 = vec![0.5, 1.5, 2.5];

        let result = linear_interpolation(&x, &y, &x0).unwrap();

        assert_relative_eq!(result[0], 0.5, epsilon = 1e-10); // Linear between 0 and 1
        assert_relative_eq!(result[1], 2.5, epsilon = 1e-10); // Linear between 1 and 4
        assert_relative_eq!(result[2], 6.5, epsilon = 1e-10); // Linear between 4 and 9
    }
}
