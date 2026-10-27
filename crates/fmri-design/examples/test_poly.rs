//! Quick test for polynomial basis

use fmri_design::basis::Poly;
use fmri_design::types::BasisFunction;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test polynomial basis directly
    let poly2 = Poly::new(2);
    println!("Poly(2) nbasis: {}", poly2.nbasis());

    let x = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let result = poly2.evaluate(&x.view())?;

    println!("Input shape: {:?}", x.shape());
    println!("Output shape: {:?}", result.shape());
    println!("All rows:");
    for i in 0..x.len() {
        print!("x={:.1}: ", x[i]);
        for j in 0..result.ncols() {
            print!("{:.3} ", result[[i, j]]);
        }
        println!();
    }

    Ok(())
}
