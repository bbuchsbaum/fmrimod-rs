use crate::{Error, Result};
use fmridataset::FeatureId;
use std::collections::HashSet;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdrMethod {
    BenjaminiHochberg,
    BenjaminiYekutieli,
}
/// A fully enumerated test family for one group contrast. Missing p-values are
/// excluded from the denominator, retained as None, and recorded by count.
/// Callers must supply the entire scientific family, not each execution block.
#[derive(Clone, Debug)]
pub struct TestFamily {
    id: String,
    contrast: String,
    values: Vec<(FeatureId, Option<f64>)>,
}
#[derive(Clone, Debug)]
pub struct AdjustedFamily {
    pub id: String,
    pub contrast: String,
    pub method: FdrMethod,
    pub tested: usize,
    pub values: Vec<(FeatureId, Option<f64>)>,
}
impl TestFamily {
    pub fn new(
        id: impl Into<String>,
        contrast: impl Into<String>,
        values: Vec<(FeatureId, Option<f64>)>,
    ) -> Result<Self> {
        let id = id.into();
        let contrast = contrast.into();
        if !crate::nonblank(&id)
            || !crate::nonblank(&contrast)
            || values.is_empty()
            || values.iter().map(|x| &x.0).collect::<HashSet<_>>().len() != values.len()
            || values
                .iter()
                .filter_map(|x| x.1)
                .any(|p| !p.is_finite() || !(0. ..=1.).contains(&p))
        {
            return Err(Error::Invalid("test family identity or p-values"));
        }
        Ok(Self {
            id,
            contrast,
            values,
        })
    }
    pub fn adjust(&self, method: FdrMethod) -> AdjustedFamily {
        let mut valid = self
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, x)| x.1.map(|p| (i, p)))
            .collect::<Vec<_>>();
        valid.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let m = valid.len();
        let factor = if method == FdrMethod::BenjaminiYekutieli {
            (1..=m).map(|i| 1. / i as f64).sum()
        } else {
            1.
        };
        let mut values = self
            .values
            .iter()
            .map(|x| (x.0.clone(), None))
            .collect::<Vec<_>>();
        let mut upper = 1_f64;
        for (rank, &(i, p)) in valid.iter().enumerate().rev() {
            upper = upper.min(p * (m as f64 / (rank + 1) as f64) * factor);
            values[i].1 = Some(upper);
        }
        AdjustedFamily {
            id: self.id.clone(),
            contrast: self.contrast.clone(),
            method,
            tested: m,
            values,
        }
    }
}
