use fmri_group::{
    FitOptions, GroupContrast, GroupDesign, MetaOutputSpec, MetaRegressionOptions, MetaUncertainty,
};
use fmrimod::dataset::ObservationId;
use fmrimod::group_requirements::{GroupAnalysisSpec, PortableBundleRequirement};
use fmrimod::outputs::{
    compile_requirements, ConsumerRequirement, DesignColumn, DesignColumns, NamedLinearEstimand,
    NamedWeight, ScientificSemantics,
};
use fmrimod::regression::output::Uncertainty;
use ndarray::array;

fn design() -> DesignColumns {
    DesignColumns::new(vec![
        DesignColumn::new("intercept", "signal", "constant").unwrap(),
        DesignColumn::new("face", "signal", "face-minus-baseline").unwrap(),
    ])
    .unwrap()
}

fn face_estimand() -> NamedLinearEstimand {
    NamedLinearEstimand::new(
        ScientificSemantics::new(
            "face",
            "Face effect",
            "percent signal change",
            "face-minus-baseline",
            "none",
            "per-run",
            "v1",
        )
        .unwrap(),
        vec![NamedWeight::new("face", 1.0).unwrap()],
    )
    .unwrap()
}

fn meta(outputs: MetaOutputSpec) -> GroupAnalysisSpec {
    GroupAnalysisSpec::MetaRegression {
        options: MetaRegressionOptions::default(),
        outputs,
    }
}

fn meta_outputs(uncertainty: MetaUncertainty) -> MetaOutputSpec {
    let design = GroupDesign::new(
        vec![
            ObservationId::new("sub-01").unwrap(),
            ObservationId::new("sub-02").unwrap(),
        ],
        vec!["intercept".into()],
        array![[1.0], [1.0]],
    )
    .unwrap();
    MetaOutputSpec {
        contrasts: vec![GroupContrast::named("mean", &design, &[("intercept", 1.0)]).unwrap()],
        uncertainty,
        p_values: false,
        confidence_level: None,
    }
}

#[test]
fn ols_requires_effects_only_regardless_of_its_execution_options() {
    let spec = GroupAnalysisSpec::Ols {
        options: FitOptions::default(),
    };
    let requirement = spec
        .first_level_requirement("group", "sub-01", "model-v1", face_estimand())
        .unwrap();

    assert_eq!(requirement.uncertainty(), Uncertainty::None);
    assert!(!spec.requires_first_level_variance());
    assert_eq!(
        spec.portable_bundle_requirement(),
        PortableBundleRequirement::EffectsOnly
    );
}

#[test]
fn meta_regression_requires_input_variance_even_for_effects_only_group_output() {
    let spec = meta(meta_outputs(MetaUncertainty::None));
    let requirement = spec
        .first_level_requirement("group", "sub-01", "model-v1", face_estimand())
        .unwrap();

    assert_eq!(requirement.consumer_id(), "group");
    assert_eq!(requirement.subject_id(), "sub-01");
    assert_eq!(requirement.model_id(), "model-v1");
    assert_eq!(requirement.estimands(), &[face_estimand()]);
    assert_eq!(requirement.uncertainty(), Uncertainty::Marginal);
    assert!(spec.requires_first_level_variance());
    assert_eq!(
        spec.portable_bundle_requirement(),
        PortableBundleRequirement::EffectsAndMarginalVariance
    );
}

#[test]
fn group_and_viewer_union_preserves_scalar_binding_and_requires_variance() {
    let group = meta(meta_outputs(MetaUncertainty::Joint))
        .first_level_requirement("group", "sub-01", "model-v1", face_estimand())
        .unwrap();
    let viewer = ConsumerRequirement::new(
        "viewer",
        "sub-01",
        "model-v1",
        vec![face_estimand()],
        Uncertainty::None,
    )
    .unwrap();

    let compiled = compile_requirements(&design(), &[viewer, group]).unwrap();
    assert_eq!(compiled.output().uncertainty(), Uncertainty::Marginal);
    assert_eq!(compiled.resolved_estimands().len(), 1);
    assert_eq!(compiled.resolved_estimands()[0].semantics().id(), "face");
    assert_eq!(compiled.resolved_estimands()[0].weights(), &[0.0, 1.0]);
    assert_eq!(compiled.consumer_maps()[0].consumer_id(), "viewer");
    assert_eq!(compiled.consumer_maps()[0].output_indices(), &[0]);
    assert_eq!(compiled.consumer_maps()[1].consumer_id(), "group");
    assert_eq!(compiled.consumer_maps()[1].output_indices(), &[0]);
}
