#[cfg(test)]
mod debug_matrix_test {
    use crate::basis::design_matrix::DesignMatrixBuilder;
    use ndarray::Array1;

    #[test] 
    fn debug_interpolation_matrix_f64() {
        // Exact same test case as R
        let x_vals = vec![0.0, 1.0, 2.0, 3.0];
        let y_vals = vec![0.0, 1.0, 8.0, 27.0];
        let order = 4;
        
        // R's exact knot construction  
        let knots = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let knots_array = Array1::from(knots);
        
        // R's exact constraint setup
        let constraint_x = vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
        let constraint_derivs = vec![2, 0, 0, 0, 0, 2];
        
        println!("\n=== RUST MATRIX DEBUG (f64) ===");
        println!("Knots: {:?}", knots_array);
        println!("Constraint x: {:?}", constraint_x);
        println!("Constraint derivs: {:?}", constraint_derivs);
        
        // Build the matrix
        let builder = DesignMatrixBuilder::new(knots_array, order);
        let system_matrix = builder.build_with_derivatives(&constraint_x, &constraint_derivs).unwrap();
        
        println!("Matrix dimensions: {:?}", system_matrix.dim());
        println!("Rust matrix A:");
        for i in 0..system_matrix.nrows() {
            print!("Row {}: [", i + 1);
            for j in 0..system_matrix.ncols() {
                if j > 0 { print!(", "); }
                print!("{:12.10}", system_matrix[[i, j]]);
            }
            println!("]");
        }
        
        println!("\nExpected R matrix:");
        println!("Row 1: [ 1.0000000000, -2.0000000000,  1.0000000000,  0.0000000000,  0.0000000000,  0.0000000000]");
        println!("Row 2: [ 0.1666666667,  0.6666666667,  0.1666666667,  0.0000000000,  0.0000000000,  0.0000000000]");
        println!("Row 3: [ 0.0000000000,  0.1666666667,  0.6666666667,  0.1666666667,  0.0000000000,  0.0000000000]");
        println!("Row 4: [ 0.0000000000,  0.0000000000,  0.1666666667,  0.6666666667,  0.1666666667,  0.0000000000]");
        println!("Row 5: [ 0.0000000000,  0.0000000000,  0.0000000000,  0.1666666667,  0.6666666667,  0.1666666667]");
        println!("Row 6: [ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000, -2.0000000000,  1.0000000000]");
        
        // Test RHS vector
        let rhs = vec![0.0, 0.0, 1.0, 8.0, 27.0, 0.0];
        let rhs_array = Array1::from(rhs);
        println!("\nRHS vector: {:?}", rhs_array);
        
        println!("=== END MATRIX DEBUG ===");
        
        // This test is for debugging only - don't assert anything
    }
}