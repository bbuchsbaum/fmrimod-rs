//! Reproducible synthetic streamed GLS benchmark; see scripts/benchmark_gls.py.
use fmri_noise::{ArEstimator, ArStart, Pooling};
use fmrireg::LinearModel;
use ndarray::{Array1, Array2, ArrayViewMut2};
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};
fn data(n: usize, start: usize, b: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, b), |(i, j)| {
        let t = (i + 1) as f64;
        let f = (start + j + 1) as f64;
        (t * 0.071 + f * 0.03).sin()
            + 0.3 * (t * 0.317 + f * 0.13).cos()
            + 0.1 * (t * 1.3 + f * 0.017).sin()
    })
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let a: Vec<usize> = args[1..8].iter().map(|s| s.parse().unwrap()).collect();
    let (n, p, v, order, block, workers, reps) = (a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    let reuse = args.get(9).is_some_and(|s| s == "reuse");
    let backend = if std::env::var("FMRIREG_BACKEND").as_deref() == Ok("portable") {
        fmrireg::MatrixBackend::Portable
    } else {
        fmrireg::MatrixBackend::default()
    };
    assert!(n % 2 == 0 && block > 0 && workers > 0 && reps > 0);
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i + 1) as f64 * (j + 1) as f64 * 0.013).sin()
        }
    });
    let valid: Vec<_> = (0..n)
        .map(|i| ![n / 4 - 1, n / 4, 3 * n / 4].contains(&i))
        .collect();
    let begin = Instant::now();
    let ols = LinearModel::new(x.view(), 1e-12)
        .unwrap()
        .with_backend(backend);
    let initial_prepare = begin.elapsed().as_secs_f64();
    let noise = data(n, 0, 32);
    let begin = Instant::now();
    let fit = ols.fit(noise.view()).unwrap();
    let residuals = &noise - &x.dot(fit.coefficients());
    let residual_seconds = begin.elapsed().as_secs_f64();
    let begin = Instant::now();
    let estimator = ArEstimator::new(
        x.view(),
        vec![n / 2, n / 2],
        Some(&valid),
        order,
        10,
        Pooling::Run,
        ArStart::ZeroHistory,
    )
    .unwrap();
    let correction_prepare = begin.elapsed().as_secs_f64();
    let begin = Instant::now();
    let plan = estimator.fit(residuals.view()).unwrap();
    let noise_fit = begin.elapsed().as_secs_f64();
    let begin = Instant::now();
    let xw = plan.apply(x.view()).unwrap();
    let design_whiten = begin.elapsed().as_secs_f64();
    let begin = Instant::now();
    let model = LinearModel::new(xw.view(), 1e-12)
        .unwrap()
        .with_backend(backend);
    let gls_prepare = begin.elapsed().as_secs_f64();
    let mut contrast = Array1::zeros(p);
    contrast[1] = 1.0;
    let run = || {
        let begin = Instant::now();
        let sums = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    let plan = &plan;
                    let model = &model;
                    let contrast = &contrast;
                    scope.spawn(move || {
                        let mut totals = [0.0; 4];
                        let mut workspace = reuse.then(|| model.workspace(block).unwrap());
                        let mut whitened = if reuse {
                            vec![0.; n * block]
                        } else {
                            Vec::new()
                        };
                        for start in (worker * block..v).step_by(workers * block) {
                            let tick = Instant::now();
                            let y = data(n, start, block.min(v - start));
                            totals[0] += tick.elapsed().as_secs_f64();
                            if let Some(workspace) = &mut workspace {
                                let tick = Instant::now();
                                let mut yw = ArrayViewMut2::from_shape(
                                    (n, y.ncols()),
                                    &mut whitened[..n * y.ncols()],
                                )
                                .unwrap();
                                plan.apply_into(y.view(), yw.view_mut()).unwrap();
                                totals[1] += tick.elapsed().as_secs_f64();
                                let tick = Instant::now();
                                let fit = model.fit_into(yw.view(), workspace).unwrap();
                                totals[2] += tick.elapsed().as_secs_f64();
                                let tick = Instant::now();
                                std::hint::black_box(fit.t_contrast(contrast.view()).unwrap());
                                totals[3] += tick.elapsed().as_secs_f64();
                            } else {
                                let tick = Instant::now();
                                let yw = plan.apply(y.view()).unwrap();
                                totals[1] += tick.elapsed().as_secs_f64();
                                let tick = Instant::now();
                                let fit = model.fit(yw.view()).unwrap();
                                totals[2] += tick.elapsed().as_secs_f64();
                                let tick = Instant::now();
                                std::hint::black_box(fit.t_contrast(contrast.view()).unwrap());
                                totals[3] += tick.elapsed().as_secs_f64();
                            }
                        }
                        totals
                    })
                })
                .collect();
            let mut sums = [0.0; 4];
            for h in handles {
                for (s, t) in sums.iter_mut().zip(h.join().unwrap()) {
                    *s += t;
                }
            }
            sums
        });
        serde_json::json!({"wall":begin.elapsed().as_secs_f64(),"generate":sums[0],"whiten":sums[1],"fit":sums[2],"contrast":sums[3]})
    };
    let cold = run();
    let times: Vec<_> = (0..reps).map(|_| run()).collect();
    // Untimed validation pass, one feature record at a time; no full-volume output retention.
    if args[8] != "-" {
        let mut out = BufWriter::new(File::create(&args[8]).unwrap());
        let mut workspace = model.workspace(block).unwrap();
        for start in (0..v).step_by(block) {
            let y = data(n, start, block.min(v - start));
            let yw = plan.apply(y.view()).unwrap();
            let fit = model.fit_into(yw.view(), &mut workspace).unwrap();
            let c = fit.t_contrast(contrast.view()).unwrap();
            for j in 0..y.ncols() {
                for i in 0..p {
                    out.write_all(&fit.coefficients()[(i, j)].to_le_bytes())
                        .unwrap();
                }
                for value in [
                    fit.residual_variance()[j],
                    c.effect[j],
                    c.standard_error[j],
                    c.statistic[j],
                    c.p_value[j],
                ] {
                    out.write_all(&value.to_le_bytes()).unwrap();
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"reuse":reuse,"backend":format!("{backend:?}"),"n":n,"p":p,"features":v,"order":order,"block":block,"workers":workers,"rank":model.rank(),"df":model.residual_df(),"phi":plan.coefficients(),"setup":{"initial_prepare":initial_prepare,"residual":residual_seconds,"correction_prepare":correction_prepare,"noise_fit":noise_fit,"design_whiten":design_whiten,"gls_prepare":gls_prepare},"cold":cold,"warm":times})
    );
}
