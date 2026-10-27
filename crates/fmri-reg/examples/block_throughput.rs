//! Production ordered executor benchmark. Run with `--features dataset,accelerate`.
#[cfg(feature = "dataset")]
fn main() {
    use fmri_data::{
        ArrayReader, DType, MemoryBudget, NumericBlock, ReadCost, ReadSelection, Selection, Values,
    };
    use fmri_noise::{ArPlan, ArStart};
    use fmri_reg::{
        dataset::{fit_blocks, fit_blocks_with_options, BlockOptions},
        GlsModel,
    };
    use ndarray::Array2;
    use std::{hint::black_box, io::Write, time::Instant};
    struct Generated {
        n: usize,
        v: usize,
    }
    impl ArrayReader for Generated {
        fn shape(&self) -> [usize; 2] {
            [self.n, self.v]
        }
        fn dtype(&self) -> DType {
            DType::F64
        }
        fn estimate(&self, s: &ReadSelection) -> fmri_data::Result<ReadCost> {
            let bytes = s.shape()[0] * s.shape()[1] * 8;
            Ok(ReadCost {
                output_bytes: bytes,
                peak_bytes: bytes,
            })
        }
        fn read(
            &mut self,
            s: &ReadSelection,
            budget: MemoryBudget,
        ) -> fmri_data::Result<NumericBlock> {
            budget.check(self.estimate(s)?.peak_bytes)?;
            let Selection::Range(r) = s.columns() else {
                panic!("range expected")
            };
            // Reserve exactly the declared numeric output, without growth
            // reallocations that would exceed this reader's peak estimate.
            let mut values = Vec::with_capacity(self.n * r.len());
            for i in 0..self.n {
                for j in r.clone() {
                    let z = (i as u64 + 1)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add((j as u64 + 1).wrapping_mul(1442695040888963407));
                    values.push((z >> 11) as f64 / (1_u64 << 53) as f64);
                }
            }
            NumericBlock::new(s.shape(), Values::F64(values))
        }
    }
    let args: Vec<_> = std::env::args().collect();
    let (n, p, v, b, w) = (
        args[1].parse::<usize>().unwrap(),
        args[2].parse::<usize>().unwrap(),
        args[3].parse::<usize>().unwrap(),
        args[4].parse::<usize>().unwrap(),
        args[5].parse::<usize>().unwrap(),
    );
    let legacy = args.get(7).is_some_and(|s| s == "legacy");
    let backend = if std::env::var("FMRIREG_BACKEND").as_deref() == Ok("portable") {
        fmri_reg::MatrixBackend::Portable
    } else {
        fmri_reg::MatrixBackend::default()
    };
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        if j == 0 {
            1.
        } else {
            ((i + 1) as f64 * (j + 1) as f64 * 0.013).sin()
        }
    });
    let plan = ArPlan::new(
        vec![n / 2, n - n / 2],
        vec![vec![0.5, -0.2, 0.1, -0.05]; 2],
        ArStart::ZeroHistory,
    )
    .unwrap();
    let model = GlsModel::new(x.view(), plan, 1e-12)
        .unwrap()
        .with_backend(backend);
    let mut reader = Generated { n, v };
    let options = BlockOptions {
        block_features: b,
        workers: w,
        working_bytes: 1024 * 1024 * 1024,
    };
    let mut run = || {
        let start = Instant::now();
        if legacy {
            fit_blocks(&model, &mut reader, b, options.working_bytes, |_, f| {
                black_box(f.coefficients());
                black_box(f.residual_variance());
                Ok(())
            })
            .unwrap();
        } else {
            fit_blocks_with_options(&model, &mut reader, options, |_, f| {
                black_box(f.coefficients());
                black_box(f.residual_variance());
                Ok(())
            })
            .unwrap();
        }
        start.elapsed().as_secs_f64()
    };
    let cold = run();
    let warm: Vec<_> = (0..5).map(|_| run()).collect();
    if args[6] != "-" {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&args[6]).unwrap());
        fit_blocks_with_options(&model, &mut reader, options, |_, f| {
            for j in 0..f.coefficients().ncols() {
                for i in 0..p {
                    out.write_all(&f.coefficients()[(i, j)].to_le_bytes())
                        .unwrap();
                }
                out.write_all(&f.residual_variance()[j].to_le_bytes())
                    .unwrap();
            }
            Ok(())
        })
        .unwrap();
    }
    println!(
        "{}",
        serde_json::json!({"n":n,"p":p,"features":v,"block":b,"workers":w,"legacy":legacy,"backend":format!("{backend:?}"),"cold":cold,"warm":warm,"scope":"known AR4 GLS coefficients and variance; serial synthetic reader and ordered sink; thread launch included"})
    );
}
#[cfg(not(feature = "dataset"))]
fn main() {
    eprintln!("enable the dataset feature");
}
