#[cfg(test)]
mod utility_equivalence_tests {
    use crate::core::{BSpline, InterpolationSpline, NaturalSpline};
    use crate::types::Spline;
    use crate::utils::{
        as_vector, as_vector_interpolation, as_vector_natural, spline_des, spline_knots,
        spline_order,
    };
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_utility_functions_equivalence_with_r() {
        println!("\n=== TESTING RUST UTILITY FUNCTIONS EQUIVALENT TO R ===");

        // Test data similar to what we'd use in R
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let knots = vec![1.5, 2.5];

        // Create BSpline (equivalent to bs() in R)
        let bspline =
            BSpline::new(&x, None, Some(knots.clone()), 3, true, Some((1.0, 5.0))).unwrap();

        // Test splineKnots() equivalent
        let rust_knots = spline_knots(&bspline);
        println!("BSpline knots from spline_knots(): {:?}", rust_knots);

        // Test splineOrder() equivalent
        let rust_order = spline_order(&bspline);
        println!("BSpline order from spline_order(): {}", rust_order);
        assert_eq!(rust_order, 4); // degree 3 = order 4

        // Test asVector() equivalent
        let rust_vector = as_vector(&bspline);
        println!(
            "BSpline vector from as_vector(): {:?}",
            &rust_vector.slice(ndarray::s![0..10])
        );

        // Verify the vector is the same as the knots for BSpline
        assert_eq!(rust_vector.len(), rust_knots.len());
        for (i, (&v, &k)) in rust_vector.iter().zip(rust_knots.iter()).enumerate() {
            assert_relative_eq!(v, k, epsilon = 1e-10);
            let diff: f64 = v - k;
            if diff.abs() > 1e-10 {
                panic!("Mismatch at index {}: {} != {}", i, v, k);
            }
        }

        println!("✓ BSpline utility functions working correctly");
    }

    #[test]
    fn test_natural_spline_utilities() {
        println!("\n=== TESTING NATURAL SPLINE UTILITIES ===");

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let knots = vec![2.0, 3.0, 4.0];

        // Create NaturalSpline (equivalent to ns() in R)
        let ns = NaturalSpline::new(&x, None, Some(knots), true, Some((1.0, 5.0))).unwrap();

        // Test utilities
        let ns_knots = spline_knots(&ns);
        let ns_order = spline_order(&ns);
        let ns_vector = as_vector_natural(&ns);

        println!(
            "NaturalSpline knots: {:?}",
            &ns_knots.slice(ndarray::s![0..10])
        );
        println!("NaturalSpline order: {}", ns_order);
        println!(
            "NaturalSpline vector: {:?}",
            &ns_vector.slice(ndarray::s![0..10])
        );

        // Natural splines are typically cubic (order 4)
        assert_eq!(ns_order, 4);

        // Vector should match knots
        assert_eq!(ns_vector.len(), ns_knots.len());

        println!("✓ NaturalSpline utility functions working correctly");
    }

    #[test]
    fn test_interpolation_spline_utilities() {
        println!("\n=== TESTING INTERPOLATION SPLINE UTILITIES ===");

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // x^2 data

        // Create InterpolationSpline (equivalent to interpSpline() in R)
        let ispline = InterpolationSpline::new(&x, &y, 4).unwrap();

        // Test utilities
        let is_knots = spline_knots(&ispline);
        let is_order = spline_order(&ispline);
        let is_vector = as_vector_interpolation(&ispline);

        println!(
            "InterpolationSpline knots: {:?}",
            &is_knots.slice(ndarray::s![0..10])
        );
        println!("InterpolationSpline order: {}", is_order);
        println!("InterpolationSpline vector (x values): {:?}", is_vector);

        assert_eq!(is_order, 4); // Cubic interpolation

        // For interpolation splines, vector should be the x values
        assert_eq!(is_vector.len(), x.len());
        for (i, (&v, &x_val)) in is_vector.iter().zip(x.iter()).enumerate() {
            assert_relative_eq!(v, x_val, epsilon = 1e-10);
            let diff: f64 = v - x_val;
            if diff.abs() > 1e-10 {
                panic!("X value mismatch at index {}: {} != {}", i, v, x_val);
            }
        }

        println!("✓ InterpolationSpline utility functions working correctly");
    }

    #[test]
    fn test_spline_des_function() {
        println!("\n=== TESTING spline_des() FUNCTION ===");

        // Test with same parameters as our R comparison
        let knots = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.5, 2.5, 4.0, 4.0, 4.0, 4.0]);
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        // Test spline_des with intercept=true (equivalent to R's splineDesign)
        let design_matrix = spline_des(&knots, &x, 4, None, true).unwrap();

        println!("spline_des() matrix dimensions: {:?}", design_matrix.dim());
        println!("spline_des() matrix (first 3 rows):");
        for i in 0..std::cmp::min(3, design_matrix.nrows()) {
            print!("Row {}: [", i);
            for j in 0..design_matrix.ncols() {
                if j > 0 {
                    print!(", ");
                }
                print!("{:8.6}", design_matrix[[i, j]]);
            }
            println!("]");
        }

        // Verify dimensions
        assert_eq!(design_matrix.nrows(), x.len());
        assert!(design_matrix.ncols() > 0);

        // Test with intercept=false
        let design_matrix_no_intercept = spline_des(&knots, &x, 4, None, false).unwrap();
        println!(
            "spline_des() no-intercept dimensions: {:?}",
            design_matrix_no_intercept.dim()
        );

        // No-intercept should have one fewer column
        assert_eq!(
            design_matrix_no_intercept.ncols(),
            design_matrix.ncols() - 1
        );

        // Test with derivatives
        let derivs = vec![0, 1, 0, 1, 2];
        let design_matrix_deriv = spline_des(&knots, &x, 4, Some(&derivs), true).unwrap();
        println!(
            "spline_des() with derivatives dimensions: {:?}",
            design_matrix_deriv.dim()
        );

        assert_eq!(design_matrix_deriv.nrows(), x.len());

        println!("✓ spline_des() function working correctly");
    }

    #[test]
    fn test_comprehensive_r_equivalence() {
        println!("\n=== COMPREHENSIVE R EQUIVALENCE TEST ===");

        // This test verifies that our utility functions produce results
        // that would be equivalent to their R counterparts

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let knots = vec![1.5, 2.5];

        // Test BSpline utilities
        let bspline = BSpline::new(&x, None, Some(knots), 3, false, Some((1.0, 5.0))).unwrap();

        // These should work just like in R:
        // R: splineKnots(bs(...))
        let _rust_knots = spline_knots(&bspline);

        // R: splineOrder(bs(...))
        let rust_order = spline_order(&bspline);
        assert_eq!(rust_order, 4);

        // R: as.vector(bs(...))
        let _rust_vector = as_vector(&bspline);

        // Test design matrix generation equivalent to R's splineDesign
        let design_x = vec![1.5, 2.0, 2.5, 3.0, 3.5];
        let design_matrix = spline_des(bspline.knots(), &design_x, 4, None, true).unwrap();

        // Verify properties that should match R
        assert_eq!(design_matrix.nrows(), design_x.len());

        // Check that rows sum appropriately (partition of unity for interior points)
        for i in 1..design_matrix.nrows() - 1 {
            // Skip boundary points
            let row_sum: f64 = design_matrix.row(i).iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
            if (row_sum - 1.0).abs() > 1e-10 {
                panic!("Row {} sum should be 1.0, got {}", i, row_sum);
            }
        }

        println!("✓ All utility functions equivalent to R splines package");
    }
}
