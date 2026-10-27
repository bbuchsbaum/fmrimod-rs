use fmri_group::*;
use fmridataset::*;
use ndarray::Array2;
pub fn features(n: usize) -> FeatureAxis {
    FeatureAxis::from_space(FeatureSpace::Index(
        IndexSpace::new(
            "test",
            (0..n)
                .map(|i| FeatureId::new(format!("f{i}")).unwrap())
                .collect(),
        )
        .unwrap(),
    ))
    .unwrap()
}
pub fn subjects(n: usize) -> Vec<Subject> {
    (0..n)
        .map(|i| Subject {
            observation: ObservationId::new(format!("o{i}")).unwrap(),
            dataset: "d".into(),
            participant: format!("s{i}"),
            measurement: "run-1".into(),
        })
        .collect()
}
pub fn estimand() -> Estimand {
    Estimand {
        id: "task-control".into(),
        definition: "task minus control, same scale".into(),
        units: "percent signal".into(),
    }
}
pub fn admission() -> Admission {
    Admission {
        target: "test support".into(),
        evidence: "synthetic identical feature definitions".into(),
    }
}
pub fn fit_receipt() -> FitReceipt {
    FitReceipt {
        model: "synthetic".into(),
        noise: "iid".into(),
        selection: "all".into(),
    }
}
pub fn cohort(y: Array2<f64>, v: Option<Array2<f64>>) -> IndependentCohort {
    let (n, f) = y.dim();
    let rows = subjects(n);
    let ids = rows
        .iter()
        .map(|s| s.observation.clone())
        .collect::<Vec<_>>();
    let fs = features(f);
    let mut assays = vec![Assay::new(
        "effect",
        SourceSpec::memory(
            NumericBlock::new([n, f], Values::F64(y.iter().copied().collect())).unwrap(),
        ),
        ids.clone(),
        fs.axis().ids().to_vec(),
        AssayAnnotations::default(),
    )
    .unwrap()];
    let unc = v.map(|v| {
        assays.push(
            Assay::new(
                "variance",
                SourceSpec::memory(
                    NumericBlock::new([n, f], Values::F64(v.iter().copied().collect())).unwrap(),
                ),
                ids.clone(),
                fs.axis().ids().to_vec(),
                AssayAnnotations::default(),
            )
            .unwrap(),
        );
        (
            "variance".into(),
            vec![
                VarianceOrigin::Known {
                    method: "fixture".into()
                };
                n
            ],
        )
    });
    let frame = FmriFrame::new(
        Axis::new(ids, vec![]).unwrap(),
        fs,
        assays,
        "effect",
        Metadata::new(),
    )
    .unwrap();
    IndependentCohort::new(
        frame,
        rows,
        estimand(),
        "effect".into(),
        unc,
        vec![fit_receipt(); n],
        admission(),
    )
    .unwrap()
}
