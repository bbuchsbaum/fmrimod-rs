use fmrimod::outputs::*;
use fmrimod::regression::output::Uncertainty;
use fmrimod::regression::LinearModel;
use ndarray::array;

fn design() -> DesignColumns {
    DesignColumns::new(vec![
        DesignColumn::new("intercept", "signal", "constant").unwrap(),
        DesignColumn::new("face", "signal", "face-minus-baseline").unwrap(),
        DesignColumn::new("scene", "signal", "scene-minus-baseline").unwrap(),
    ])
    .unwrap()
}

fn semantics(id: &str) -> ScientificSemantics {
    ScientificSemantics::new(
        id,
        "Face effect",
        "percent signal change",
        "face-baseline",
        "none",
        "per-run",
        "v1",
    )
    .unwrap()
}

fn estimand(id: &str, weights: Vec<(&str, f64)>) -> NamedLinearEstimand {
    NamedLinearEstimand::new(
        semantics(id),
        weights
            .into_iter()
            .map(|(name, value)| NamedWeight::new(name, value).unwrap())
            .collect(),
    )
    .unwrap()
}

fn requirement(
    id: &str,
    estimands: Vec<NamedLinearEstimand>,
    uncertainty: Uncertainty,
) -> ConsumerRequirement {
    ConsumerRequirement::new(id, "sub-01", "design-and-rank-v3", estimands, uncertainty).unwrap()
}

#[test]
fn compiles_union_preserves_consumer_order_and_maximum_uncertainty() {
    let compiled = compile_requirements(
        &design(),
        &[
            requirement(
                "group-a",
                vec![
                    estimand("face", vec![("face", 1.0)]),
                    estimand("scene", vec![("scene", 1.0)]),
                ],
                Uncertainty::Marginal,
            ),
            requirement(
                "group-b",
                vec![
                    estimand("scene", vec![("scene", 1.0)]),
                    estimand("face", vec![("face", 1.0)]),
                ],
                Uncertainty::Joint,
            ),
        ],
    )
    .unwrap();
    assert_eq!(compiled.resolved_estimands().len(), 2);
    assert_eq!(compiled.consumer_maps()[0].output_indices(), &[0, 1]);
    assert_eq!(compiled.consumer_maps()[1].output_indices(), &[1, 0]);
    assert_eq!(compiled.output().uncertainty(), Uncertainty::Joint);
}

#[test]
fn same_id_requires_same_scientific_definition_and_resolved_weights() {
    let different_unit = ScientificSemantics::new(
        "face",
        "Face effect",
        "beta",
        "face-baseline",
        "none",
        "per-run",
        "v1",
    )
    .unwrap();
    let incompatible =
        NamedLinearEstimand::new(different_unit, vec![NamedWeight::new("face", 1.0).unwrap()])
            .unwrap();
    let err = compile_requirements(
        &design(),
        &[
            requirement(
                "a",
                vec![estimand("face", vec![("face", 1.0)])],
                Uncertainty::None,
            ),
            requirement("b", vec![incompatible], Uncertainty::None),
        ],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OutputRequirementError::ConflictingEstimand("face".into())
    );

    let per_subject = ScientificSemantics::new(
        "face",
        "Face effect",
        "percent signal change",
        "face-baseline",
        "none",
        "per-subject",
        "v1",
    )
    .unwrap();
    let incompatible =
        NamedLinearEstimand::new(per_subject, vec![NamedWeight::new("face", 1.0).unwrap()])
            .unwrap();
    let err = compile_requirements(
        &design(),
        &[
            requirement(
                "a",
                vec![estimand("face", vec![("face", 1.0)])],
                Uncertainty::None,
            ),
            requirement("b", vec![incompatible], Uncertainty::None),
        ],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OutputRequirementError::ConflictingEstimand("face".into())
    );
}

#[test]
fn rejects_bad_design_names_and_weights_before_data_read() {
    assert_eq!(
        DesignColumns::new(vec![
            DesignColumn::new("x", "u", "c").unwrap(),
            DesignColumn::new("x", "u", "c").unwrap()
        ])
        .unwrap_err(),
        OutputRequirementError::DuplicateDesignColumn("x".into())
    );
    assert_eq!(
        NamedWeight::new("face", f64::NAN).unwrap_err(),
        OutputRequirementError::NonFiniteWeight("face".into())
    );
    let unknown = estimand("face", vec![("not-a-column", 1.0)]);
    assert_eq!(
        compile_requirements(
            &design(),
            &[requirement("a", vec![unknown], Uncertainty::None)]
        )
        .unwrap_err(),
        OutputRequirementError::UnknownDesignColumn("not-a-column".into())
    );
}

#[test]
fn rejects_duplicate_consumers_and_mismatched_subject_or_model() {
    assert_eq!(
        compile_requirements(
            &design(),
            &[
                requirement(
                    "a",
                    vec![estimand("face", vec![("face", 1.0)])],
                    Uncertainty::None
                ),
                requirement(
                    "a",
                    vec![estimand("scene", vec![("scene", 1.0)])],
                    Uncertainty::None
                ),
            ]
        )
        .unwrap_err(),
        OutputRequirementError::DuplicateConsumer("a".into())
    );

    let other_subject = ConsumerRequirement::new(
        "b",
        "sub-02",
        "design-and-rank-v3",
        vec![estimand("scene", vec![("scene", 1.0)])],
        Uncertainty::None,
    )
    .unwrap();
    assert!(matches!(
        compile_requirements(
            &design(),
            &[requirement("a", vec![], Uncertainty::None), other_subject]
        ),
        Err(OutputRequirementError::SubjectMismatch { .. })
    ));
}

fn binding() -> OutputBinding {
    OutputBinding::new(vec![ResolvedEstimand::for_reuse(
        semantics("face"),
        vec![0.0, 1.0, 0.0],
    )
    .unwrap()])
    .unwrap()
}

fn identity(whitening: WhiteningIdentity, data_version: &str) -> ReuseIdentity {
    ReuseIdentity::new(
        "source-v1",
        data_version,
        "rows-original",
        "rows-after-censor",
        "mni-2mm-v1",
        "design-v3-rank-svd",
        whitening,
        "backend-f64-v2",
        binding(),
    )
    .unwrap()
}

#[test]
fn capability_requires_exact_identity_and_does_not_promote_marginal_to_joint() {
    let base_identity = identity(
        WhiteningIdentity::BuiltIn {
            version: "ar1-v2".into(),
        },
        "data-v1",
    );
    let marginal = OutputCapability::new(base_identity.clone(), Uncertainty::Marginal);
    assert!(marginal.can_satisfy(&base_identity, Uncertainty::None));
    assert!(marginal.can_satisfy(&base_identity, Uncertainty::Marginal));
    assert!(!marginal.can_satisfy(&base_identity, Uncertainty::Joint));
    assert!(!marginal.can_satisfy(
        &identity(
            WhiteningIdentity::BuiltIn {
                version: "ar1-v2".into()
            },
            "data-v2"
        ),
        Uncertainty::Marginal
    ));
    assert!(!marginal.can_satisfy(
        &identity(
            WhiteningIdentity::BuiltIn {
                version: "ar1-v3".into()
            },
            "data-v1"
        ),
        Uncertainty::Marginal
    ));
}

#[test]
fn custom_whitening_must_be_explicitly_versioned_for_durable_reuse() {
    let built_in = identity(
        WhiteningIdentity::BuiltIn {
            version: "ar1-v2".into(),
        },
        "data-v1",
    );
    assert_eq!(built_in.cache_eligibility(), CacheEligibility::BuiltIn);
    let custom = identity(
        WhiteningIdentity::CustomVersioned {
            deterministic_version: "vendor-filter-9".into(),
        },
        "data-v1",
    );
    assert_eq!(
        custom.cache_eligibility(),
        CacheEligibility::CustomVersioned
    );
    let session = identity(WhiteningIdentity::SessionOnly, "data-v1");
    assert_eq!(session.cache_eligibility(), CacheEligibility::SessionOnly);
    assert!(!OutputCapability::new(session.clone(), Uncertainty::Joint)
        .can_satisfy(&session, Uncertainty::None));
    assert!(ReuseIdentity::new(
        "source",
        "data",
        "original",
        "selected",
        "space",
        "rank",
        WhiteningIdentity::CustomVersioned {
            deterministic_version: " ".into()
        },
        "backend",
        binding()
    )
    .is_err());
}

#[test]
fn compiled_requirement_prepares_and_runs_on_ols() {
    let compiled = compile_requirements(
        &design(),
        &[requirement(
            "consumer",
            vec![estimand("face", vec![("face", 1.0)])],
            Uncertainty::Marginal,
        )],
    )
    .unwrap();
    let model_design = array![
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let model = LinearModel::new(model_design.view(), 1e-12).unwrap();
    let prepared = model.prepare_outputs(compiled.output()).unwrap();
    let mut workspace = prepared.workspace(1).unwrap();
    let data = array![[1.0], [2.0], [3.0], [4.0]];
    let summary = prepared.run_into(data.view(), &mut workspace).unwrap();
    assert_eq!(summary.estimates().dim(), (1, 1));
    assert_eq!(summary.uncertainty(), Uncertainty::Marginal);
}
