//! Formula-based event model construction.

use crate::core::{DataFrame, EventModel};
use crate::covariate::{create_covariate_events, is_covariate_term as is_explicit_covariate_term};
use crate::events::{EventFactor, EventTerm, EventVariable, HrfEvent};
use crate::formula::{parse_formula, HrfSpec, Term, TermParam};
use crate::hrf::HrfSpec as HrfEnumSpec;
use crate::sampling::SamplingFrame;
use crate::types::{EventSpec, ValidationError};

/// Build an event model from a formula string and data.
///
/// # Arguments
/// * `formula` - R-style formula string (e.g., "onset ~ hrf(condition) + covariate(motion_x)")
/// * `data` - DataFrame containing event and covariate data
/// * `sampling_frame` - Sampling frame for the design matrix
/// * `tr` - Repetition time in seconds
///
/// # Example
/// ```ignore
/// use fmri_design::formula::event_model;
/// use fmri_design::core::DataFrame;
/// use fmri_design::sampling::SamplingFrame;
///
/// let mut data = DataFrame::new();
/// data.insert("onset", vec![1.0, 5.0, 10.0]);
/// data.insert("condition", vec!["A", "B", "A"]);
/// data.insert("duration", vec![1.0, 1.0, 1.0]);
/// data.insert("motion_x", vec![0.1, -0.2, 0.3]);
///
/// let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);
/// let model = event_model(
///     "onset ~ hrf(condition) + covariate(motion_x)",
///     &data,
///     sampling_frame,
///     2.0,
/// ).unwrap();
/// ```
pub fn event_model(
    formula: &str,
    data: &DataFrame,
    sampling_frame: SamplingFrame,
    tr: f64,
) -> Result<EventModel, ValidationError> {
    // Parse the formula
    let terms = parse_formula(formula)?;

    // Separate different types of terms
    let mut event_terms = Vec::new();
    let mut covariate_terms = Vec::new();
    let mut trialwise_terms = Vec::new();

    for term in terms {
        if term.events.contains(&"trialwise".to_string()) {
            trialwise_terms.push(term);
        } else if term.events.contains(&"_all_covariates_".to_string()) {
            // Special marker for covariate() without arguments
            covariate_terms.push(term);
        } else if term.events.contains(&"_covariate_placeholder_".to_string()) {
            // Special marker from covariate() function
            covariate_terms.push(term);
        } else if is_explicit_covariate_term(&term) {
            covariate_terms.push(term);
        } else {
            event_terms.push(term);
        }
    }

    // Build event specifications
    let mut all_events: Vec<Box<dyn EventSpec>> = Vec::new();

    // Process regular event terms
    for term in event_terms {
        let event_spec = build_event_spec(&term, data, tr, &sampling_frame)?;
        all_events.push(event_spec);
    }

    // Process covariates
    if !covariate_terms.is_empty() {
        let covariate_specs = build_covariate_spec(&covariate_terms, data, &sampling_frame)?;
        all_events.extend(covariate_specs);
    }

    // Process trialwise terms
    for term in trialwise_terms {
        let trialwise_spec = build_trialwise_spec(&term, data)?;
        all_events.push(trialwise_spec);
    }

    // Create EventTerms from specifications and box them
    let event_terms: Vec<Box<dyn EventSpec>> = all_events
        .into_iter()
        .map(|event| EventTerm::new(vec![event], false))
        .map(|term| term.map(|t| Box::new(t) as Box<dyn EventSpec>))
        .collect::<Result<Vec<_>, _>>()?;

    // Create the event model
    EventModel::new(event_terms, sampling_frame)
}

/// Build event specification from a term.
fn build_event_spec(
    term: &Term,
    data: &DataFrame,
    tr: f64,
    frame: &SamplingFrame,
) -> Result<Box<dyn EventSpec>, ValidationError> {
    let mut event: Box<dyn EventSpec> = if term.is_interaction() {
        let events = term
            .events
            .iter()
            .map(|name| build_single_event(name, data, tr))
            .collect::<Result<_, _>>()?;
        Box::new(EventTerm::new(events, true)?)
    } else {
        let name = term
            .events
            .first()
            .ok_or_else(|| ValidationError::InvalidParameter("empty event term".into()))?;
        build_single_event(name, data, tr)?
    };
    if let Some(hrf) = &term.hrf {
        event = apply_hrf_to_event(event, hrf, data, frame)?;
    }
    Ok(event)
}

/// Build a single event from data.
fn build_single_event(
    event_name: &str,
    data: &DataFrame,
    tr: f64,
) -> Result<Box<dyn EventSpec>, ValidationError> {
    // Get onset times
    let onsets_str = data
        .get("onset")
        .or_else(|| data.get("onsets"))
        .ok_or_else(|| ValidationError::MissingColumn("onset".to_string()))?;

    let onsets: Vec<f64> = onsets_str
        .iter()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ValidationError::InvalidType {
            expected: "numeric".to_string(),
            actual: "string".to_string(),
        })?;

    // Get durations (default to TR if not specified)
    let durations: Vec<f64> =
        if let Some(dur_str) = data.get("duration").or_else(|| data.get("durations")) {
            dur_str
                .iter()
                .map(|s| s.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ValidationError::InvalidType {
                    expected: "numeric".to_string(),
                    actual: "string".to_string(),
                })?
        } else {
            vec![tr; onsets.len()]
        };

    // Check if event is categorical or continuous
    if let Some(values) = data.get(event_name) {
        // Check if it's categorical (string values)
        if values.iter().all(|v| v.parse::<f64>().is_err()) {
            // Categorical event
            let levels: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
            Ok(Box::new(EventFactor::new(
                event_name, onsets, levels, durations,
            )?))
        } else {
            // Continuous event
            let values: Vec<f64> = values
                .iter()
                .map(|v| v.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ValidationError::InvalidType {
                    expected: "numeric".to_string(),
                    actual: "mixed".to_string(),
                })?;

            Ok(Box::new(EventVariable::new(
                event_name, onsets, values, durations,
            )?))
        }
    } else {
        // Event name not found in data - create a simple event
        let levels: Vec<&str> = vec![event_name; onsets.len()];
        Ok(Box::new(EventFactor::new(
            event_name, onsets, levels, durations,
        )?))
    }
}

/// Apply HRF to an event.
fn apply_hrf_to_event(
    event: Box<dyn EventSpec>,
    hrf_spec: &HrfSpec,
    data: &DataFrame,
    frame: &SamplingFrame,
) -> Result<Box<dyn EventSpec>, ValidationError> {
    // Convert formula HrfSpec to hrf module HrfSpec. We currently support a
    // small set of named HRFs that map onto the implementations in
    // `crate::hrf`.
    let hrf_enum = match hrf_spec {
        HrfSpec::Named(name) => {
            let normalized_name = match name.as_str() {
                "spm_canonical" => "canonical",
                other => other,
            };

            if normalized_name == "double_gamma" {
                let mut params = std::collections::HashMap::new();
                params.insert("peak_shape".to_string(), 6.0);
                params.insert("peak_scale".to_string(), 1.0);
                params.insert("undershoot_shape".to_string(), 16.0);
                params.insert("undershoot_scale".to_string(), 1.0);
                params.insert("peak_weight".to_string(), 1.0);
                params.insert("undershoot_weight".to_string(), 0.35);
                HrfEnumSpec::Named {
                    name: normalized_name.to_string(),
                    params,
                }
            } else {
                HrfEnumSpec::Named {
                    name: normalized_name.to_string(),
                    params: std::collections::HashMap::new(),
                }
            }
        }
        HrfSpec::Custom(params) => HrfEnumSpec::Named {
            name: "gamma".into(),
            params: params.clone(),
        },
    };

    // Wrap the event in an HRF-aware EventSpec. The actual HRF convolution is
    // performed on demand in `HrfEvent::design_matrix`, using the sampling
    // points provided by the EventModel.
    let event = HrfEvent::new(event, hrf_enum).with_precision(frame.precision)?;
    let blocks = data.get("run").or_else(|| {
        if frame.n_blocks() > 1 {
            data.get("block")
        } else {
            None
        }
    });
    let run_ids = if let Some(ids) = blocks {
        ids.iter()
            .map(|id| {
                id.parse::<f64>()
                    .ok()
                    .filter(|i| {
                        i.is_finite() && *i >= 1.0 && i.fract() == 0.0 && *i < usize::MAX as f64
                    })
                    .map(|i| i as usize - 1)
                    .ok_or_else(|| {
                        ValidationError::InvalidParameter(
                            "block/run values must be one-based positive integers".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if frame.n_blocks() == 1 {
        vec![0; event.onsets().len()]
    } else {
        return Err(ValidationError::MissingColumn(
            "block (one-based run for each event)".into(),
        ));
    };
    Ok(Box::new(event.with_runs(frame.clone(), run_ids)?))
}

/// Build covariate specification from terms.
fn build_covariate_spec(
    terms: &[Term],
    data: &DataFrame,
    sampling_frame: &SamplingFrame,
) -> Result<Vec<Box<dyn EventSpec>>, ValidationError> {
    let mut all_covariates = Vec::new();

    for term in terms {
        if term.events.contains(&"_all_covariates_".to_string()) {
            // Use all numeric columns
            for (name, values) in data.iter() {
                if name != "onset" && name != "duration" {
                    if values.iter().all(|v| v.parse::<f64>().is_ok()) {
                        all_covariates.push(name.clone());
                    }
                }
            }
        } else if term.events.contains(&"_covariate_placeholder_".to_string()) {
            // Extract covariate names from the parameter
            if let Some(TermParam::String(names_str)) = term.params.get("_covariate_names") {
                let names: Vec<String> = names_str.split(',').map(|s| s.to_string()).collect();
                all_covariates.extend(names);
            }
        } else {
            // Use specified covariates
            for event in &term.events {
                if event != "_covariate_placeholder_" {
                    all_covariates.push(event.clone());
                }
            }
        }
    }

    // Remove duplicates
    all_covariates.sort();
    all_covariates.dedup();

    if all_covariates.is_empty() {
        return Ok(Vec::new());
    }

    // Expected number of timepoints is determined by the sampling frame,
    // matching the R implementation which validates nrow(covariate) ==
    // sum(blocklens(sframe)).
    let expected_n_timepoints = sampling_frame.n_scans();

    // Create covariate events
    let covariate_events = create_covariate_events(
        data,
        &all_covariates,
        expected_n_timepoints,
        None, // No prefix for now
    )?;

    // Convert to EventSpec trait objects
    Ok(covariate_events
        .into_iter()
        .map(|event| Box::new(event) as Box<dyn EventSpec>)
        .collect())
}

/// Build trialwise specification from a term.
///
/// Creates one regressor per trial by generating an EventFactor with a
/// unique level for each event row. Equivalent to R's `trialwise()`.
fn build_trialwise_spec(
    term: &Term,
    data: &DataFrame,
) -> Result<Box<dyn EventSpec>, ValidationError> {
    use crate::trialwise::{create_trial_factor, get_trialwise_config};

    let config = get_trialwise_config(term).unwrap_or_default();

    // Get onsets from data
    let onsets_str = data
        .get("onset")
        .or_else(|| data.get("onsets"))
        .ok_or_else(|| ValidationError::MissingColumn("onset".to_string()))?;

    let onsets: Vec<f64> = onsets_str
        .iter()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ValidationError::InvalidType {
            expected: "numeric".to_string(),
            actual: "string".to_string(),
        })?;

    let n_trials = onsets.len();

    // Get durations
    let durations: Vec<f64> =
        if let Some(dur_str) = data.get("duration").or_else(|| data.get("durations")) {
            dur_str
                .iter()
                .map(|s| s.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|_| vec![0.0; n_trials])
        } else {
            vec![0.0; n_trials]
        };

    let onset_arr = ndarray::Array1::from_vec(onsets);
    let dur_arr = ndarray::Array1::from_vec(durations);

    let trial_factor =
        create_trial_factor(n_trials, Some(&onset_arr), Some(&dur_arr), &config.label)?;

    Ok(Box::new(trial_factor))
}

/// Builder pattern for creating event models with formulas.
pub struct EventModelBuilder {
    terms: Vec<Term>,
    data: Option<DataFrame>,
    sampling_frame: Option<SamplingFrame>,
    tr: f64,
}

impl EventModelBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            terms: Vec::new(),
            data: None,
            sampling_frame: None,
            tr: 2.0, // Default TR
        }
    }

    /// Set TR.
    pub fn tr(mut self, tr: f64) -> Self {
        self.tr = tr;
        self
    }

    /// Set data.
    pub fn data(mut self, data: DataFrame) -> Self {
        self.data = Some(data);
        self
    }

    /// Set sampling frame.
    pub fn sampling_frame(mut self, frame: SamplingFrame) -> Self {
        self.sampling_frame = Some(frame);
        self
    }

    /// Add a term.
    pub fn add_term(mut self, term: Term) -> Self {
        self.terms.push(term);
        self
    }

    /// Add terms from a formula string.
    pub fn formula(mut self, formula: &str) -> Result<Self, ValidationError> {
        let terms = parse_formula(formula)?;
        self.terms.extend(terms);
        Ok(self)
    }

    /// Build the event model.
    pub fn build(self) -> Result<EventModel, ValidationError> {
        let data = self
            .data
            .ok_or_else(|| ValidationError::InvalidParameter("Data not provided".to_string()))?;
        let sampling_frame = self.sampling_frame.ok_or_else(|| {
            ValidationError::InvalidParameter("Sampling frame not provided".to_string())
        })?;

        // Process terms and create event model
        event_model_from_terms(self.terms, &data, sampling_frame, self.tr)
    }
}

/// Create event model from pre-parsed terms.
fn event_model_from_terms(
    terms: Vec<Term>,
    data: &DataFrame,
    sampling_frame: SamplingFrame,
    tr: f64,
) -> Result<EventModel, ValidationError> {
    // Same logic as event_model but with pre-parsed terms
    let mut event_terms = Vec::new();
    let mut covariate_terms = Vec::new();
    let mut trialwise_terms = Vec::new();

    for term in terms {
        if term.events.contains(&"trialwise".to_string()) {
            trialwise_terms.push(term);
        } else if term.events.contains(&"_all_covariates_".to_string()) {
            covariate_terms.push(term);
        } else if term.events.contains(&"_covariate_placeholder_".to_string()) {
            covariate_terms.push(term);
        } else if is_explicit_covariate_term(&term) {
            covariate_terms.push(term);
        } else {
            event_terms.push(term);
        }
    }

    let mut all_events: Vec<Box<dyn EventSpec>> = Vec::new();

    for term in event_terms {
        let event_spec = build_event_spec(&term, data, tr, &sampling_frame)?;
        all_events.push(event_spec);
    }

    if !covariate_terms.is_empty() {
        let covariate_specs = build_covariate_spec(&covariate_terms, data, &sampling_frame)?;
        all_events.extend(covariate_specs);
    }

    for term in trialwise_terms {
        let trialwise_spec = build_trialwise_spec(&term, data)?;
        all_events.push(trialwise_spec);
    }

    let event_terms: Vec<Box<dyn EventSpec>> = all_events
        .into_iter()
        .map(|event| EventTerm::new(vec![event], false))
        .map(|term| term.map(|t| Box::new(t) as Box<dyn EventSpec>))
        .collect::<Result<Vec<_>, _>>()?;

    EventModel::new(event_terms, sampling_frame)
}

impl Default for EventModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingFrameExt;
    use crate::types::EventType;

    #[test]
    fn test_event_model_simple() {
        let mut data = DataFrame::new();
        data.insert("onset", vec!["1.0", "5.0", "10.0"]);
        data.insert("condition", vec!["A", "B", "A"]);
        data.insert("duration", vec!["1.0", "1.0", "1.0"]);

        let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

        // Use hrf() wrapper matching R's onset ~ hrf(condition) pattern
        let model = event_model("hrf(condition)", &data, sampling_frame, 2.0).unwrap();

        assert_eq!(model.n_events(), 1);
    }

    #[test]
    fn test_event_model_with_hrf() {
        let mut data = DataFrame::new();
        data.insert("onset", vec!["1.0", "5.0", "10.0"]);
        data.insert("condition", vec!["A", "B", "A"]);
        data.insert("duration", vec!["1.0", "1.0", "1.0"]);

        let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

        let model = event_model("hrf(condition)", &data, sampling_frame, 2.0).unwrap();

        assert_eq!(model.n_events(), 1);
    }

    #[test]
    fn test_event_model_builder() -> Result<(), ValidationError> {
        let mut data = DataFrame::new();
        data.insert("onset", vec!["1.0", "5.0"]);
        data.insert("condition", vec!["A", "B"]);
        data.insert("rating", vec!["3.5", "4.2"]);

        let model = EventModelBuilder::new()
            .tr(2.0)
            .data(data)
            .sampling_frame(SamplingFrame::from_duration(20.0, 2.0))
            .formula("hrf(condition)")?
            .build()?;

        assert_eq!(model.n_events(), 1);
        Ok(())
    }

    #[test]
    fn test_bare_terms_route_to_event_specs() -> Result<(), ValidationError> {
        let mut data = DataFrame::new();
        data.insert("onset", vec!["1.0", "5.0", "10.0"]);
        data.insert("duration", vec!["1.0", "1.0", "1.0"]);
        data.insert("condition", vec!["A", "B", "A"]);
        data.insert("rt", vec!["0.4", "0.6", "0.5"]);

        let model = event_model(
            "condition + rt",
            &data,
            SamplingFrame::from_duration(20.0, 2.0),
            2.0,
        )?;

        let condition_term = model.events()[0]
            .as_any()
            .downcast_ref::<EventTerm>()
            .unwrap();
        let rt_term = model.events()[1]
            .as_any()
            .downcast_ref::<EventTerm>()
            .unwrap();

        assert_eq!(
            condition_term.events()[0].event_type(),
            EventType::Categorical
        );
        assert_eq!(rt_term.events()[0].event_type(), EventType::Continuous);

        Ok(())
    }
}
