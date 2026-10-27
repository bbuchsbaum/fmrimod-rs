use fmri_noise::{ArPlan, ArStart};
use ndarray::{s, Array2};

fn coefficients(order: usize, run: usize) -> Vec<f64> {
    (0..order)
        .map(|lag| {
            let sign = if (lag + run).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            sign * 0.04 / (lag + 1) as f64
        })
        .collect()
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    for column in 0..b.len() {
        let pivot = (column..b.len())
            .max_by(|&left, &right| a[left][column].abs().total_cmp(&a[right][column].abs()))
            .unwrap();
        a.swap(column, pivot);
        b.swap(column, pivot);
        let diagonal = a[column][column];
        for value in &mut a[column][column..] {
            *value /= diagonal;
        }
        b[column] /= diagonal;
        let pivot_row = a[column].clone();
        for row in 0..b.len() {
            if row != column {
                let scale = a[row][column];
                for (value, basis) in a[row][column..].iter_mut().zip(&pivot_row[column..]) {
                    *value -= scale * basis;
                }
                b[row] -= scale * b[column];
            }
        }
    }
    b
}

fn startup_inverse(phi: &[f64]) -> Vec<Vec<f64>> {
    let p = phi.len();
    if p == 0 {
        return Vec::new();
    }
    let mut equations = vec![vec![0.0; p + 1]; p + 1];
    for h in 0..=p {
        equations[h][h] = 1.0;
        for (lag, &value) in phi.iter().enumerate() {
            equations[h][h.abs_diff(lag + 1)] -= value;
        }
    }
    let mut rhs = vec![0.0; p + 1];
    rhs[0] = 1.0;
    let gamma = solve(equations, rhs);
    let mut lower = vec![vec![0.0; p]; p];
    for row in 0..p {
        for column in 0..=row {
            let mut value = gamma[row.abs_diff(column)];
            for (left, right) in lower[row][..column].iter().zip(&lower[column][..column]) {
                value -= left * right;
            }
            lower[row][column] = if row == column {
                value.sqrt()
            } else {
                value / lower[column][column]
            };
        }
    }
    let mut columns = vec![vec![0.0; p]; p];
    for (column, solution) in columns.iter_mut().enumerate() {
        for row in 0..p {
            let rhs = if row == column { 1.0 } else { 0.0 };
            let prior: f64 = lower[row][..row]
                .iter()
                .zip(&solution[..row])
                .map(|(a, b)| a * b)
                .sum();
            solution[row] = (rhs - prior) / lower[row][row];
        }
    }
    (0..p)
        .map(|row| columns.iter().map(|column| column[row]).collect())
        .collect()
}

fn reference(
    input: &Array2<f64>,
    lengths: &[usize],
    coefficients: &[Vec<f64>],
    resets: &[usize],
    startup: ArStart,
) -> Array2<f64> {
    let (rows, columns) = input.dim();
    let mut output = Array2::zeros((rows, columns));
    let mut reset = vec![false; rows];
    let mut run_start = 0;
    for &length in lengths {
        reset[run_start] = true;
        run_start += length;
    }
    for &index in resets {
        reset[index] = true;
    }
    let inverses: Vec<_> = coefficients
        .iter()
        .map(|phi| startup_inverse(phi))
        .collect();
    run_start = 0;
    for (run, &length) in lengths.iter().enumerate() {
        let phi = &coefficients[run];
        let mut segment = run_start;
        for time in run_start..run_start + length {
            if reset[time] {
                segment = time;
            }
            let offset = time - segment;
            for feature in 0..columns {
                output[(time, feature)] =
                    if matches!(startup, ArStart::Stationary) && offset < phi.len() {
                        (0..=offset)
                            .map(|k| inverses[run][offset][k] * input[(segment + k, feature)])
                            .sum()
                    } else {
                        let mut sum = 0.0;
                        for (lag, &coefficient) in phi.iter().take(offset).enumerate() {
                            sum += coefficient * input[(time - lag - 1, feature)];
                        }
                        input[(time, feature)] - sum
                    };
            }
        }
        run_start += length;
    }
    output
}

fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 3e-11 * (1.0 + expected.abs()));
    }
}

#[test]
fn ar_apply_matches_independent_scalar_reference_across_orders_and_strides() {
    let lengths = [9, 15];
    let resets = [2, 14];
    let input = Array2::from_shape_fn((24, 4), |(row, column)| {
        ((row * 7 + column * 3) as f64 - 13.0) / 11.0
    });
    for order in 0..=8 {
        let phi = vec![coefficients(order, 0), coefficients(order, 1)];
        for startup in [ArStart::ZeroHistory, ArStart::Stationary] {
            let plan = ArPlan::new(lengths.to_vec(), phi.clone(), startup)
                .unwrap()
                .with_resets(&resets)
                .unwrap();
            let view = input.slice(s![..;-1, ..;-1]);
            let expected = reference(&view.to_owned(), &lengths, &phi, &resets, startup);
            assert_close(&plan.apply(view).unwrap(), &expected);

            let mut backing = Array2::zeros((24, 4));
            let output = backing.slice_mut(s![..;-1, ..;-1]);
            plan.apply_into(input.slice(s![..;-1, ..;-1]), output)
                .unwrap();
            assert_close(&backing.slice(s![..;-1, ..;-1]).to_owned(), &expected);
        }
    }
}

#[test]
fn zero_history_fused_path_preserves_scalar_rounding_on_mixed_scales() {
    let values = [1e16, -1e16, 1e-300, -1e-300, -0.0, 0.0, 1.0, -1.0];
    let input = Array2::from_shape_fn((31, 8), |(i, j)| values[(i + j) % values.len()]);
    for order in 0..=4 {
        let lengths = [9, 22];
        let phi = vec![coefficients(order, 0), coefficients(order, 1)];
        let resets = [2, 17];
        let plan = ArPlan::new(lengths.to_vec(), phi.clone(), ArStart::ZeroHistory)
            .unwrap()
            .with_resets(&resets)
            .unwrap();
        let expected = reference(&input, &lengths, &phi, &resets, ArStart::ZeroHistory);
        let actual = plan.apply(input.view()).unwrap();
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
