#[cfg(test)]
mod final_bs_verification {
    use crate::core::bspline::BSpline;
    use approx::assert_relative_eq;

    #[test]
    fn test_bspline_intercept_true_equivalence() {
        // Test BSpline with intercept=true (equivalent to R's bs(intercept=TRUE))
        let x_vals = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let knots = vec![1.5, 2.5];

        let spline = BSpline::new(&x_vals, None, Some(knots), 3, true, Some((0.0, 4.0))).unwrap();
        let design_matrix = spline.design_matrix(&x_vals).unwrap();

        println!("\n=== BSPLINE WITH INTERCEPT=TRUE ===");
        println!("Matrix dimensions: {:?}", design_matrix.dim());
        println!("Design matrix:");
        for i in 0..design_matrix.nrows() {
            print!("Row {}: [", i);
            for j in 0..design_matrix.ncols() {
                if j > 0 {
                    print!(", ");
                }
                print!("{:8.6}", design_matrix[[i, j]]);
            }
            println!("]");
        }

        // Expected R's bs(intercept=TRUE) results (6x6 matrix)
        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0370370, 0.4474074, 0.4488889, 0.0666667, 0.0, 0.0],
            vec![0.0, 0.02, 0.48, 0.48, 0.02, 0.0],
            vec![0.0, 0.0, 0.0666667, 0.4488889, 0.4474074, 0.0370370],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        // Verify dimensions and values
        assert_eq!(design_matrix.nrows(), 5);
        assert_eq!(design_matrix.ncols(), 6);

        for i in 0..design_matrix.nrows() {
            for j in 0..design_matrix.ncols() {
                assert_relative_eq!(design_matrix[[i, j]], expected[i][j], epsilon = 1e-6);
            }
        }

        // Check partition of unity (all row sums should equal 1)
        for i in 0..design_matrix.nrows() {
            let row_sum: f64 = design_matrix.row(i).iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        println!("✓ Perfect equivalence with R's bs(intercept=TRUE)");
    }

    #[test]
    fn test_bspline_intercept_false_equivalence() {
        // Test BSpline with intercept=false (equivalent to R's bs(intercept=FALSE))
        let x_vals = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let knots = vec![1.5, 2.5];

        let spline = BSpline::new(&x_vals, None, Some(knots), 3, false, Some((0.0, 4.0))).unwrap();
        let design_matrix = spline.design_matrix(&x_vals).unwrap();

        println!("\n=== BSPLINE WITH INTERCEPT=FALSE ===");
        println!("Matrix dimensions: {:?}", design_matrix.dim());
        println!("Design matrix:");
        for i in 0..design_matrix.nrows() {
            print!("Row {}: [", i);
            for j in 0..design_matrix.ncols() {
                if j > 0 {
                    print!(", ");
                }
                print!("{:8.6}", design_matrix[[i, j]]);
            }
            println!("]");
        }

        // Expected R's bs(intercept=FALSE) results (5x5 matrix)
        let expected = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.4474074, 0.4488889, 0.0666667, 0.0, 0.0],
            vec![0.02, 0.48, 0.48, 0.02, 0.0],
            vec![0.0, 0.0666667, 0.4488889, 0.4474074, 0.0370370],
            vec![0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        // Verify dimensions and values
        assert_eq!(design_matrix.nrows(), 5);
        assert_eq!(design_matrix.ncols(), 5);

        for i in 0..design_matrix.nrows() {
            for j in 0..design_matrix.ncols() {
                assert_relative_eq!(design_matrix[[i, j]], expected[i][j], epsilon = 1e-6);
            }
        }

        // Check row sums (should match R's no-intercept behavior)
        let expected_row_sums = vec![0.0, 0.962963, 1.0, 1.0, 1.0];
        for i in 0..design_matrix.nrows() {
            let row_sum: f64 = design_matrix.row(i).iter().sum();
            assert_relative_eq!(row_sum, expected_row_sums[i], epsilon = 1e-6);
        }

        println!("✓ Perfect equivalence with R's bs(intercept=FALSE)");
    }

    #[test]
    fn test_bspline_boundary_conditions() {
        // Test the specific boundary condition that was fixed
        let x_vals = vec![0.0, 4.0]; // Include both boundaries to satisfy min points requirement
        let knots = vec![1.5, 2.5];

        let spline = BSpline::new(&x_vals, None, Some(knots), 3, true, Some((0.0, 4.0))).unwrap();
        let design_matrix = spline.design_matrix(&[4.0]).unwrap(); // Test just the right boundary

        println!("\n=== BOUNDARY CONDITION TEST ===");
        println!("x=4 (right boundary) with intercept=true:");
        for j in 0..design_matrix.ncols() {
            print!("{:8.6} ", design_matrix[[0, j]]);
        }
        println!();

        // Should be [0, 0, 0, 0, 0, 1] for intercept=true
        let expected = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for j in 0..design_matrix.ncols() {
            assert_relative_eq!(design_matrix[[0, j]], expected[j], epsilon = 1e-10);
        }

        println!("✓ Right boundary condition working correctly");
    }
}
