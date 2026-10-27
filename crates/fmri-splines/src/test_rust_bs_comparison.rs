#[cfg(test)]
mod rust_bs_comparison {
    use crate::basis::design_matrix::DesignMatrixBuilder;
    use crate::core::bspline::BSpline;
    use ndarray::Array1;

    #[test]
    fn test_boundary_conditions() {
        // Test boundary condition issue at x=4
        let x_vals = vec![4.0]; // Just the problematic point
        let internal_knots = vec![1.5, 2.5];
        let degree = 3;
        let order = degree + 1;

        let boundary_knots = vec![0.0, 4.0];
        let mut full_knots = Vec::new();

        for _ in 0..=degree {
            full_knots.push(boundary_knots[0]);
        }
        for &knot in &internal_knots {
            full_knots.push(knot);
        }
        for _ in 0..=degree {
            full_knots.push(boundary_knots[1]);
        }

        println!("\n=== BOUNDARY CONDITION TEST ===");
        println!("x value: {:?}", x_vals);
        println!("full knot vector: {:?}", full_knots);

        // Debug the interval finding manually
        let x_val = 4.0;
        let mut knot_ptr = order;
        println!("Starting knot_ptr = {}", knot_ptr);

        while knot_ptr < full_knots.len() && full_knots[knot_ptr] <= x_val {
            println!(
                "knot_ptr = {}, knots[{}] = {}, x_val = {}",
                knot_ptr, knot_ptr, full_knots[knot_ptr], x_val
            );
            knot_ptr += 1;
        }

        println!("Final knot_ptr = {}", knot_ptr);
        let offset = knot_ptr - order;
        println!("offset = {}", offset);
        println!("n_coeffs = {}", full_knots.len() - order);

        if offset >= full_knots.len() - order {
            println!(
                "WARNING: offset {} >= n_coeffs {}",
                offset,
                full_knots.len() - order
            );
        }

        let knots_array = Array1::from(full_knots);
        let builder = DesignMatrixBuilder::new(knots_array, order);
        let design_matrix = builder.build(&x_vals).unwrap();

        println!("Rust matrix for x=4:");
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

        println!("Expected R result: [0, 0, 0, 0, 0, 1]");
    }

    #[test]
    fn test_rust_vs_r_bs_no_intercept() {
        // Test R's default bs(intercept = FALSE) behavior
        let x_vals = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let internal_knots = vec![1.5, 2.5];
        let degree = 3;
        let order = degree + 1;

        let boundary_knots = vec![0.0, 4.0];
        let mut full_knots = Vec::new();

        for _ in 0..=degree {
            full_knots.push(boundary_knots[0]);
        }
        for &knot in &internal_knots {
            full_knots.push(knot);
        }
        for _ in 0..=degree {
            full_knots.push(boundary_knots[1]);
        }

        println!("\n=== RUST BSPLINE TEST (intercept = FALSE) ===");

        let knots_array = Array1::from(full_knots);
        let builder = DesignMatrixBuilder::new_no_intercept(knots_array, order);
        let design_matrix = builder.build(&x_vals).unwrap();

        println!("Rust design matrix dimensions: {:?}", design_matrix.dim());
        println!("Rust design matrix (no intercept):");
        for i in 0..design_matrix.nrows() {
            print!("Row {}: [", i);
            for j in 0..design_matrix.ncols() {
                if j > 0 {
                    print!(", ");
                }
                print!("{:12.6}", design_matrix[[i, j]]);
            }
            println!("]");
        }

        // Check row sums (should not be 1 for no-intercept case)
        println!("\nRust row sums (should match R's no-intercept behavior):");
        for i in 0..design_matrix.nrows() {
            let row_sum: f64 = design_matrix.row(i).iter().sum();
            println!("Row {}: {:.6}", i, row_sum);
        }

        println!("\n=== EXPECTED R bs(intercept = FALSE) RESULTS ===");
        println!("R matrix (5x5):");
        println!("Row 0: [0.0000000, 0.00000000, 0.00000000, 0.0000000, 0.00000000]");
        println!("Row 1: [0.4474074, 0.44888889, 0.06666667, 0.0000000, 0.00000000]");
        println!("Row 2: [0.0200000, 0.48000000, 0.48000000, 0.0200000, 0.00000000]");
        println!("Row 3: [0.0000000, 0.06666667, 0.44888889, 0.4474074, 0.03703704]");
        println!("Row 4: [0.0000000, 0.00000000, 0.00000000, 0.0000000, 1.00000000]");

        println!("\nR row sums:");
        println!("Row 0: 0 (no intercept)");
        println!("Row 1: 0.962963 (no intercept)");
        println!("Row 2: 1.0");
        println!("Row 3: 1.0");
        println!("Row 4: 1.0");
    }

    #[test]
    fn test_rust_vs_r_bs_basic() {
        // Test case 1: Basic cubic B-splines - same as R test
        let x_vals = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let internal_knots = vec![1.5, 2.5];
        let degree = 3;
        let order = degree + 1;

        // R's boundary knots: [0, 4]
        // R's full knot vector: [0, 0, 0, 0, 1.5, 2.5, 4, 4, 4, 4]
        let boundary_knots = vec![0.0, 4.0];
        let mut full_knots = Vec::new();

        // Add repeated boundary knots (degree+1 times)
        for _ in 0..=degree {
            full_knots.push(boundary_knots[0]);
        }

        // Add internal knots
        for &knot in &internal_knots {
            full_knots.push(knot);
        }

        // Add repeated boundary knots (degree+1 times)
        for _ in 0..=degree {
            full_knots.push(boundary_knots[1]);
        }

        println!("\n=== RUST BSPLINE TEST ===");
        println!("x values: {:?}", x_vals);
        println!("degree: {}", degree);
        println!("internal knots: {:?}", internal_knots);
        println!("full knot vector: {:?}", full_knots);

        let knots_array = Array1::from(full_knots);

        // Get design matrix using our implementation directly
        let builder = DesignMatrixBuilder::new(knots_array, order);
        let design_matrix = builder.build(&x_vals).unwrap();

        println!("Rust design matrix dimensions: {:?}", design_matrix.dim());
        println!("Rust design matrix:");
        for i in 0..design_matrix.nrows() {
            print!("Row {}: [", i);
            for j in 0..design_matrix.ncols() {
                if j > 0 {
                    print!(", ");
                }
                print!("{:12.6}", design_matrix[[i, j]]);
            }
            println!("]");
        }

        // Check partition of unity
        println!("\nRust row sums (should be 1 for standard B-splines):");
        for i in 0..design_matrix.nrows() {
            let row_sum: f64 = design_matrix.row(i).iter().sum();
            println!("Row {}: {:.6}", i, row_sum);
        }

        println!("\n=== EXPECTED R bs() RESULTS ===");
        println!("R matrix (from previous test):");
        println!("Row 0: [0.0000000, 0.00000000, 0.00000000, 0.0000000, 0.00000000]");
        println!("Row 1: [0.4474074, 0.44888889, 0.06666667, 0.0000000, 0.00000000]");
        println!("Row 2: [0.0200000, 0.48000000, 0.48000000, 0.0200000, 0.00000000]");
        println!("Row 3: [0.0000000, 0.06666667, 0.44888889, 0.4474074, 0.03703704]");
        println!("Row 4: [0.0000000, 0.00000000, 0.00000000, 0.0000000, 1.00000000]");

        println!("\nR row sums:");
        println!("Row 0: 0 (no intercept)");
        println!("Row 1: 0.962963 (no intercept)");
        println!("Row 2: 1.0");
        println!("Row 3: 1.0");
        println!("Row 4: 1.0");
    }

    #[test]
    fn test_detailed_comparison() {
        // Test case 4: More x values for detailed comparison
        let x_vals: Vec<f64> = (0..=8).map(|i| i as f64 * 0.5).collect(); // [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        let internal_knots = vec![1.5, 2.5];
        let degree = 3;
        let order = degree + 1;

        // Same knot construction as above
        let boundary_knots = vec![0.0, 4.0];
        let mut full_knots = Vec::new();

        for _ in 0..=degree {
            full_knots.push(boundary_knots[0]);
        }
        for &knot in &internal_knots {
            full_knots.push(knot);
        }
        for _ in 0..=degree {
            full_knots.push(boundary_knots[1]);
        }

        let knots_array = Array1::from(full_knots);
        let builder = DesignMatrixBuilder::new(knots_array, order);
        let design_matrix = builder.build(&x_vals).unwrap();

        println!("\n=== DETAILED RUST vs R COMPARISON ===");
        println!("x values: {:?}", x_vals);
        println!("Rust matrix:");
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
    }
}
