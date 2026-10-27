use fmri_reg::LinearModel;
use ndarray::Array2;
use std::time::Instant;
fn main() {
    let mut cases = Vec::new();
    for (n, p, v) in [(300, 8, 1000), (300, 8, 10000), (1200, 32, 1000)] {
        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            if j == 0 {
                1.
            } else {
                ((i + 1) as f64 * (j + 1) as f64 * 0.013).sin()
            }
        });
        let y = Array2::from_shape_fn((n, v), |(i, j)| {
            ((i + 1) as f64 * 0.071 + (j + 1) as f64 * 0.03).sin()
                + 0.1 * ((i + 1) as f64 * 1.3).cos()
        });
        let start = Instant::now();
        let model = LinearModel::new(x.view(), 1e-12).unwrap();
        let setup = start.elapsed().as_secs_f64();
        let warm = model.fit(y.view()).unwrap();
        let beta_sum = warm.coefficients().sum();
        let variance_sum = warm.residual_variance().sum();
        drop(warm);
        let mut times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            std::hint::black_box(model.fit(y.view()).unwrap());
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        cases.push(serde_json::json!({"n":n,"p":p,"features":v,"prepare_seconds":setup,"median_fit_seconds":times[2],"beta_sum":beta_sum,"variance_sum":variance_sum,"repetitions":5}));
    }
    println!("{}", serde_json::to_string_pretty(&cases).unwrap());
}
