// Debug script to inspect design matrix construction
use ndarray::Array1;
use crate::basis::design_matrix::DesignMatrixBuilder;

fn debug_matrix() {
    // Test data matching R example
    let knots = Array1::from(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let constraint_x = vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
    let constraint_derivs = vec![2, 0, 0, 0, 0, 2];
    let order = 4;
    
    println!("Knots: {:?}", knots);
    println!("Constraint x: {:?}", constraint_x);
    println!("Constraint derivs: {:?}", constraint_derivs);
    
    let builder = DesignMatrixBuilder::new(knots, order);
    match builder.build_with_derivatives(&constraint_x, &constraint_derivs) {
        Ok(matrix) => {
            println!("Design matrix shape: {:?}", matrix.dim());
            println!("Design matrix:");
            for i in 0..matrix.nrows() {
                for j in 0..matrix.ncols() {
                    print!("{:8.4} ", matrix[[i, j]]);
                }
                println!();
            }
        },
        Err(e) => println!("Error: {:?}", e),
    }
}