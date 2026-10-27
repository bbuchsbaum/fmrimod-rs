use fmrimod::{
    dataset::FeatureId,
    group::{FdrMethod, TestFamily},
};

#[test]
fn group_facade_preserves_dataset_identity() {
    let family = TestFamily::new(
        "whole-map",
        "mean",
        vec![
            (FeatureId::new("parcel-a").unwrap(), Some(0.01)),
            (FeatureId::new("parcel-b").unwrap(), Some(0.2)),
        ],
    )
    .unwrap();
    let adjusted = family.adjust(FdrMethod::BenjaminiHochberg);
    assert_eq!(adjusted.tested, 2);
    assert_eq!(adjusted.values[0].0.as_str(), "parcel-a");
    assert_eq!(adjusted.values[0].1, Some(0.02));
}
