//! Warm, equivalent-output benchmark and an unshipped direct-projection probe.
#[path = "../src/backend.rs"]
mod backend;
use fmri_noise::{ArPlan, ArStart, WhiteningPlan};
use fmrireg::{
    output::{EstimandSet, OutputSpec, SummaryView, Uncertainty},
    GlsModel, LinearModel, MatrixBackend,
};
use nalgebra::DMatrix;
use ndarray::{Array2, ArrayView2};
use std::{hint::black_box, time::Instant};

fn inverse(x: &Array2<f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    let svd = DMatrix::from_fn(n, p, |i, j| x[(i, j)]).svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut out = Array2::zeros((p, n));
    for k in 0..svd.singular_values.len() {
        if svd.singular_values[k] <= svd.singular_values[0] * 1e-12 {
            continue;
        }
        for i in 0..p {
            for j in 0..n {
                out[(i, j)] += vt[(k, i)] * u[(j, k)] / svd.singular_values[k];
            }
        }
    }
    out
}
fn finite(y: ArrayView2<'_, f64>) -> bool {
    y.as_slice()
        .unwrap()
        .chunks(256)
        .all(|s| s.iter().fold(true, |a, x| a & x.is_finite()))
}
fn sample(mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let now = Instant::now();
    f();
    let reps = (0.008 / now.elapsed().as_secs_f64())
        .ceil()
        .clamp(1., 2000.) as usize;
    let mut samples = Vec::new();
    for _ in 0..5 {
        let now = Instant::now();
        for _ in 0..reps {
            f();
        }
        samples.push(now.elapsed().as_secs_f64() * 1000. / reps as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[2]
}
fn error(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
fn local_backend(b: MatrixBackend) -> backend::MatrixBackend {
    match b {
        MatrixBackend::Portable => backend::MatrixBackend::Portable,
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        MatrixBackend::Accelerate => backend::MatrixBackend::Accelerate,
    }
}
macro_rules! measure {
    ($model:expr,$data:expr,$weights:expr,$noise:expr,$backend:expr,$name:expr) => {{
        let model=&$model; let y=&$data; let l=&$weights; let backend=local_backend($backend);
        let (n,b)=y.dim(); let p=l.ncols(); let q=l.nrows();
        let mut full_ws=model.workspace(b).unwrap();
        let mut effects=Array2::zeros((q,b));
        let base=sample(||{let fit=model.fit_into(y.view(),&mut full_ws).unwrap();backend.multiply(l.view(),fit.coefficients().view(),effects.view_mut());black_box(&effects);black_box(fit.residual_variance());});
        for level in [Uncertainty::None,Uncertainty::Marginal] {
            let spec=OutputSpec::new(EstimandSet::contrasts((0..q).map(|j|format!("c{j}")).collect(),l.clone()).unwrap()).with_uncertainty(level);
            let plan=model.prepare_outputs(&spec).unwrap();let mut ws=plan.workspace(b).unwrap();
            let actual=plan.run_into(y.view(),&mut ws).unwrap();
            let max_error=error(actual.estimates(),effects.view()); assert!(max_error<1e-10);
            if let SummaryView::Marginal(v)=&actual {
                let reference=model.fit_into(y.view(),&mut full_ws).unwrap();
                for (a,b) in v.residual_scales().iter().zip(reference.residual_variance()) { assert!((a-b).abs()<1e-12); }
                black_box(v.standard_error(0,0).unwrap());
            }
            let time=sample(||{let out=plan.run_into(y.view(),&mut ws).unwrap();black_box(out.estimates());});
            // Reverse order controls for drift; reported times average both medians.
            let time2=sample(||{let out=plan.run_into(y.view(),&mut ws).unwrap();black_box(out.estimates());});
            let base2=sample(||{let fit=model.fit_into(y.view(),&mut full_ws).unwrap();backend.multiply(l.view(),fit.coefficients().view(),effects.view_mut());black_box(&effects);black_box(fit.residual_variance());});
            println!("fit,{:?},{},{n},{p},{q},{b},{level:?},{:.6},{:.6},{:.4},{},{},{max_error:.3e}",$backend,$name,(base+base2)/2.,(time+time2)/2.,(base+base2)/(time+time2),full_ws.working_bytes(),ws.working_bytes());
        }
        // Candidate is confined to this example. Explicit W is retained.
        let transformed_x=$noise.apply(model_design($name,n,p).view()).unwrap();
        let d=inverse(&transformed_x); let k=l.dot(&d);
        let mut wy=Array2::zeros((n,b));let mut projected=Array2::zeros((q,b));
        let candidate=sample(||{$noise.apply_into(y.view(),wy.view_mut()).unwrap();assert!(finite(wy.view()));backend.multiply(k.view(),wy.view(),projected.view_mut());assert!(finite(projected.view()));black_box(&projected);});
        let max_error=error(projected.view(),effects.view()); assert!(max_error < 1e-10);
        println!("candidate,{:?},{},{n},{p},{q},{b},None,{base:.6},{candidate:.6},{:.4},0,{}, {max_error:.3e}",$backend,$name,base/candidate,8*(n+q)*b);
    }};
}
// This deterministic design is shared by preparation and the candidate probe.
fn model_design(_name: &str, n: usize, p: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |(i, j)| {
        if j == 0 {
            1.
        } else {
            ((i + 1) as f64 * (j + 1) as f64 * 0.017).sin()
        }
    })
}
#[derive(Clone)]
struct Identity(usize);
impl WhiteningPlan for Identity {
    fn observations(&self) -> usize {
        self.0
    }
    fn apply(&self, y: ArrayView2<'_, f64>) -> fmri_noise::Result<Array2<f64>> {
        Ok(y.to_owned())
    }
    fn apply_into(
        &self,
        y: ArrayView2<'_, f64>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> fmri_noise::Result<()> {
        out.assign(&y);
        Ok(())
    }
}
fn adversarial(backend: MatrixBackend) {
    let local = local_backend(backend);
    for epsilon in [1e-3, 1e-6, 1e-9] {
        let x = Array2::from_shape_fn((128, 3), |(i, j)| match j {
            0 => 1.,
            1 => (i as f64 * 0.3).sin(),
            _ => (i as f64 * 0.3).sin() + epsilon * (i as f64 * 0.7).cos(),
        });
        let l = ndarray::array![[0., 1., 1.], [0., 1., -1.]];
        let d = inverse(&x);
        let k = l.dot(&d);
        let model = LinearModel::new(x.view(), 1e-12)
            .unwrap()
            .with_backend(backend);
        let spec = OutputSpec::new(
            EstimandSet::contrasts(vec!["sum".into(), "difference".into()], l).unwrap(),
        );
        let plan = model.prepare_outputs(&spec).unwrap();
        for offset in [1., 1e6, 1e12] {
            let y = Array2::from_shape_fn((128, 8), |(i, j)| {
                offset * (j + 1) as f64 + 0.25 * (i as f64 * 0.1).sin()
            });
            let mut ws = plan.workspace(8).unwrap();
            let reference = plan.run_into(y.view(), &mut ws).unwrap();
            let mut result = Array2::zeros((2, 8));
            local.multiply(k.view(), y.view(), result.view_mut());
            let abs = error(result.view(), reference.estimates());
            let normalized = result
                .iter()
                .zip(reference.estimates())
                .map(|(a, b)| (a - b).abs() / (1. + b.abs()))
                .fold(0., f64::max);
            println!("adverse,{backend:?},epsilon={epsilon:.1e},offset={offset:.1e},max_abs={abs:.5e},max_scaled={normalized:.5e}");
        }
    }
}
fn main() {
    #[allow(unused_mut)]
    let mut backends = vec![MatrixBackend::Portable];
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    backends.push(MatrixBackend::Accelerate);
    println!("kind,backend,model,n,p,q,b,uncertainty,full_ms,requested_ms,speedup,full_bytes,requested_bytes,max_abs_error");
    for backend in backends {
        for n in [300, 1200] {
            for p in [8, 32] {
                for q in [1, 4, 16] {
                    let x = model_design("", n, p);
                    let y = Array2::from_shape_fn((n, 2048), |(i, j)| {
                        3. + ((i + 1) as f64 * 0.03 + (j + 1) as f64 * 0.02).sin()
                    });
                    let l = Array2::from_shape_fn((q, p), |(i, j)| {
                        if j == (i + 1) % p {
                            1.
                        } else if j == (i + 2) % p {
                            -0.5
                        } else {
                            0.
                        }
                    });
                    let ols = LinearModel::new(x.view(), 1e-12)
                        .unwrap()
                        .with_backend(backend);
                    measure!(ols, y, l, Identity(n), backend, "OLS");
                    let noise = ArPlan::new(
                        vec![n / 4; 4],
                        vec![vec![0.4, -0.1, 0.08, -0.03]; 4],
                        ArStart::ZeroHistory,
                    )
                    .unwrap();
                    let gls = GlsModel::new(x.view(), noise.clone(), 1e-12)
                        .unwrap()
                        .with_backend(backend);
                    measure!(gls, y, l, noise, backend, "AR4");
                }
            }
        }
        adversarial(backend);
    }
}
