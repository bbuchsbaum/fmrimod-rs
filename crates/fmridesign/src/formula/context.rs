//! Formula context and data binding.
//!
//! This module provides context for evaluating formulas against data sources.

use super::parser::{FormulaExpr, FormulaParser};
use super::term::Term;
use crate::core::DataFrame;
use crate::events::{EventFactor, EventVariable};
use crate::types::{EventSpec, ValidationError};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Context for evaluating formulas with data binding.
#[derive(Debug, Clone)]
pub struct FormulaContext {
    /// Data sources mapped by name
    data_sources: HashMap<String, DataSource>,
    /// Variables available in the context
    variables: HashMap<String, Variable>,
    /// Default data source name
    default_source: Option<String>,
}

/// A data source that can provide values for formula evaluation.
#[derive(Debug)]
pub enum DataSource {
    /// DataFrame source
    DataFrame(DataFrame),
    /// Event data source
    Events(Vec<Box<dyn EventSpec>>),
    /// Matrix data source
    Matrix {
        data: Array2<f64>,
        row_names: Option<Vec<String>>,
        col_names: Option<Vec<String>>,
    },
}

impl Clone for DataSource {
    fn clone(&self) -> Self {
        match self {
            Self::DataFrame(df) => Self::DataFrame(df.clone()),
            Self::Events(events) => {
                // Clone events by using their clone_box method
                let cloned_events: Vec<Box<dyn EventSpec>> =
                    events.iter().map(|e| e.clone_box()).collect();
                Self::Events(cloned_events)
            }
            Self::Matrix {
                data,
                row_names,
                col_names,
            } => Self::Matrix {
                data: data.clone(),
                row_names: row_names.clone(),
                col_names: col_names.clone(),
            },
        }
    }
}

/// A variable in the formula context.
#[derive(Debug)]
pub enum Variable {
    /// Numeric values
    Numeric(Array1<f64>),
    /// Categorical values
    Categorical(Vec<String>),
    /// Boolean values
    Boolean(Vec<bool>),
    /// Event specification
    Event(Box<dyn EventSpec>),
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        match self {
            Self::Numeric(arr) => Self::Numeric(arr.clone()),
            Self::Categorical(vec) => Self::Categorical(vec.clone()),
            Self::Boolean(vec) => Self::Boolean(vec.clone()),
            Self::Event(event) => Self::Event(event.clone_box()),
        }
    }
}

impl FormulaContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            data_sources: HashMap::new(),
            variables: HashMap::new(),
            default_source: None,
        }
    }

    /// Create a context with a default DataFrame.
    pub fn with_dataframe(df: DataFrame) -> Self {
        let mut ctx = Self::new();
        ctx.add_dataframe("data", df);
        ctx.set_default_source("data");
        ctx
    }

    /// Create a context with event data.
    pub fn with_events(events: Vec<Box<dyn EventSpec>>) -> Self {
        let mut ctx = Self::new();
        ctx.add_events("events", events);
        ctx.set_default_source("events");
        ctx
    }

    /// Add a DataFrame data source.
    pub fn add_dataframe(&mut self, name: &str, df: DataFrame) -> &mut Self {
        self.data_sources
            .insert(name.to_string(), DataSource::DataFrame(df));
        self
    }

    /// Add an events data source.
    pub fn add_events(&mut self, name: &str, events: Vec<Box<dyn EventSpec>>) -> &mut Self {
        self.data_sources
            .insert(name.to_string(), DataSource::Events(events));
        self
    }

    /// Add a matrix data source.
    pub fn add_matrix(
        &mut self,
        name: &str,
        data: Array2<f64>,
        row_names: Option<Vec<String>>,
        col_names: Option<Vec<String>>,
    ) -> &mut Self {
        self.data_sources.insert(
            name.to_string(),
            DataSource::Matrix {
                data,
                row_names,
                col_names,
            },
        );
        self
    }

    /// Set the default data source.
    pub fn set_default_source(&mut self, name: &str) -> &mut Self {
        self.default_source = Some(name.to_string());
        self
    }

    /// Add a variable to the context.
    pub fn add_variable(&mut self, name: &str, var: Variable) -> &mut Self {
        self.variables.insert(name.to_string(), var);
        self
    }

    /// Resolve a variable name to its value.
    pub fn resolve_variable(&self, name: &str) -> Result<Variable, ValidationError> {
        // First check explicit variables
        if let Some(var) = self.variables.get(name) {
            return Ok(var.clone());
        }

        // Then check data sources
        if let Some(source_name) = &self.default_source {
            if let Some(source) = self.data_sources.get(source_name) {
                return self.resolve_from_source(source, name);
            }
        }

        // Check all data sources
        for (_, source) in &self.data_sources {
            if let Ok(var) = self.resolve_from_source(source, name) {
                return Ok(var);
            }
        }

        Err(ValidationError::InvalidParameter(format!(
            "Variable '{}' not found in context",
            name
        )))
    }

    /// Resolve a variable from a specific data source.
    fn resolve_from_source(
        &self,
        source: &DataSource,
        name: &str,
    ) -> Result<Variable, ValidationError> {
        match source {
            DataSource::DataFrame(df) => {
                // Try to get numeric column
                if let Some(values) = df.get_column(name) {
                    return Ok(Variable::Numeric(Array1::from_vec(values)));
                }

                // Try to get string column
                if let Some(values) = df.get_str_column(name) {
                    return Ok(Variable::Categorical(values));
                }

                Err(ValidationError::InvalidParameter(format!(
                    "Column '{}' not found in DataFrame",
                    name
                )))
            }
            DataSource::Events(events) => {
                // Find event by name
                for event in events {
                    if event.name() == name {
                        return Ok(Variable::Event(event.clone_box()));
                    }
                }

                Err(ValidationError::InvalidParameter(format!(
                    "Event '{}' not found",
                    name
                )))
            }
            DataSource::Matrix {
                data, col_names, ..
            } => {
                // If column names are provided, try to find column by name
                if let Some(names) = col_names {
                    if let Some(idx) = names.iter().position(|n| n == name) {
                        if idx < data.ncols() {
                            let col = data.column(idx).to_owned();
                            return Ok(Variable::Numeric(col));
                        }
                    }
                }

                Err(ValidationError::InvalidParameter(format!(
                    "Variable '{}' not found in matrix",
                    name
                )))
            }
        }
    }

    /// Evaluate a formula expression in this context.
    pub fn evaluate_expression(&self, expr: &FormulaExpr) -> Result<Variable, ValidationError> {
        match expr {
            FormulaExpr::Variable(name) => self.resolve_variable(name),

            FormulaExpr::Number(n) => Ok(Variable::Numeric(Array1::from_elem(1, *n))),

            FormulaExpr::Boolean(b) => Ok(Variable::Boolean(vec![*b])),

            FormulaExpr::StringLiteral(s) => Ok(Variable::Categorical(vec![s.clone()])),

            FormulaExpr::FunctionCall { name, args, kwargs } => {
                self.evaluate_function(name, args, kwargs)
            }

            FormulaExpr::Addition(left, right) => {
                // For addition, we return both variables (used in formula terms)
                let _left_var = self.evaluate_expression(left)?;
                let _right_var = self.evaluate_expression(right)?;

                // This is mainly used for term construction, not actual addition
                Err(ValidationError::InvalidParameter(
                    "Addition expressions should be handled at the term level".to_string(),
                ))
            }

            FormulaExpr::Interaction(left, right) => {
                // Similar to addition, interactions are handled at term level
                let _left_var = self.evaluate_expression(left)?;
                let _right_var = self.evaluate_expression(right)?;

                Err(ValidationError::InvalidParameter(
                    "Interaction expressions should be handled at the term level".to_string(),
                ))
            }

            FormulaExpr::BasisExpansion(var_expr, _basis_expr) => {
                // Return the variable, basis expansion is handled at term level
                self.evaluate_expression(var_expr)
            }

            FormulaExpr::Pipe(left, _right) => {
                // Pipe operations are handled at the term level
                self.evaluate_expression(left)
            }
        }
    }

    /// Evaluate a function call.
    fn evaluate_function(
        &self,
        name: &str,
        args: &[FormulaExpr],
        _kwargs: &[(String, FormulaExpr)],
    ) -> Result<Variable, ValidationError> {
        match name {
            "scale" => {
                if args.is_empty() {
                    return Err(ValidationError::InvalidParameter(
                        "scale() requires at least one argument".to_string(),
                    ));
                }

                let var = self.evaluate_expression(&args[0])?;

                match var {
                    Variable::Numeric(values) => {
                        let mean = values.mean().unwrap_or(0.0);
                        let std = values.std(0.0);

                        if std > 0.0 {
                            let scaled = (values - mean) / std;
                            Ok(Variable::Numeric(scaled))
                        } else {
                            Ok(Variable::Numeric(values))
                        }
                    }
                    _ => Err(ValidationError::InvalidParameter(
                        "scale() requires numeric variable".to_string(),
                    )),
                }
            }

            "center" => {
                if args.is_empty() {
                    return Err(ValidationError::InvalidParameter(
                        "center() requires at least one argument".to_string(),
                    ));
                }

                let var = self.evaluate_expression(&args[0])?;

                match var {
                    Variable::Numeric(values) => {
                        let mean = values.mean().unwrap_or(0.0);
                        Ok(Variable::Numeric(values - mean))
                    }
                    _ => Err(ValidationError::InvalidParameter(
                        "center() requires numeric variable".to_string(),
                    )),
                }
            }

            "log" => {
                if args.is_empty() {
                    return Err(ValidationError::InvalidParameter(
                        "log() requires at least one argument".to_string(),
                    ));
                }

                let var = self.evaluate_expression(&args[0])?;

                match var {
                    Variable::Numeric(values) => {
                        let logged = values.mapv(|x| x.ln());
                        Ok(Variable::Numeric(logged))
                    }
                    _ => Err(ValidationError::InvalidParameter(
                        "log() requires numeric variable".to_string(),
                    )),
                }
            }

            _ => Err(ValidationError::InvalidParameter(format!(
                "Unknown function: {}",
                name
            ))),
        }
    }

    /// Evaluate a where clause expression to create a boolean mask.
    pub fn evaluate_where_clause(
        &self,
        expr: &str,
        conditions: &[String],
    ) -> Result<Vec<bool>, ValidationError> {
        let n = conditions.len();
        let mut mask = vec![true; n];

        // Simple implementation for common patterns
        // TODO: Full expression evaluation

        // Pattern: variable == 'value'
        if let Some(caps) = regex::Regex::new(r#"(\w+)\s*==\s*['"]([^'"]+)['"]"#)
            .unwrap()
            .captures(expr)
        {
            let _var_name = &caps[1];
            let target_value = &caps[2];

            // Check if it's a condition-based filter
            for (i, cond) in conditions.iter().enumerate() {
                if !cond.contains(target_value) {
                    mask[i] = false;
                }
            }

            return Ok(mask);
        }

        // Pattern: variable != 'value'
        if let Some(caps) = regex::Regex::new(r#"(\w+)\s*!=\s*['"]([^'"]+)['"]"#)
            .unwrap()
            .captures(expr)
        {
            let _var_name = &caps[1];
            let target_value = &caps[2];

            for (i, cond) in conditions.iter().enumerate() {
                if cond.contains(target_value) {
                    mask[i] = false;
                }
            }

            return Ok(mask);
        }

        // Pattern: variable > value (numeric)
        if let Some(caps) = regex::Regex::new(r"(\w+)\s*>\s*(\d+\.?\d*)")
            .unwrap()
            .captures(expr)
        {
            let var_name = &caps[1];
            if let Ok(threshold) = caps[2].parse::<f64>() {
                // Try to get numeric variable
                if let Ok(Variable::Numeric(values)) = self.resolve_variable(var_name) {
                    if values.len() == n {
                        for (i, &val) in values.iter().enumerate() {
                            mask[i] = val > threshold;
                        }
                        return Ok(mask);
                    }
                }
            }
        }

        // Pattern: variable < value (numeric)
        if let Some(caps) = regex::Regex::new(r"(\w+)\s*<\s*(\d+\.?\d*)")
            .unwrap()
            .captures(expr)
        {
            let var_name = &caps[1];
            if let Ok(threshold) = caps[2].parse::<f64>() {
                if let Ok(Variable::Numeric(values)) = self.resolve_variable(var_name) {
                    if values.len() == n {
                        for (i, &val) in values.iter().enumerate() {
                            mask[i] = val < threshold;
                        }
                        return Ok(mask);
                    }
                }
            }
        }

        // If we can't parse the expression, return all true
        Ok(mask)
    }

    /// Create events from variables in context.
    pub fn create_event(
        &self,
        name: &str,
        var: &Variable,
    ) -> Result<Box<dyn EventSpec>, ValidationError> {
        match var {
            Variable::Event(event) => Ok(event.clone_box()),

            Variable::Numeric(values) => {
                // Create continuous event with uniform onsets
                let n = values.len();
                let onsets: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let durations = vec![1.0; n];

                Ok(Box::new(EventVariable::new(
                    name.to_string(),
                    onsets,
                    values.to_vec(),
                    durations,
                )?))
            }

            Variable::Categorical(values) => {
                // Create categorical event
                let n = values.len();
                let onsets: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let durations = vec![1.0; n];
                let str_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();

                Ok(Box::new(EventFactor::new(
                    name.to_string(),
                    onsets,
                    str_refs,
                    durations,
                )?))
            }

            Variable::Boolean(values) => {
                // Convert to categorical
                let str_values: Vec<String> = values
                    .iter()
                    .map(|&b| if b { "TRUE" } else { "FALSE" }.to_string())
                    .collect();
                let n = values.len();
                let onsets: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let durations = vec![1.0; n];
                let str_refs: Vec<&str> = str_values.iter().map(|s| s.as_str()).collect();

                Ok(Box::new(EventFactor::new(
                    name.to_string(),
                    onsets,
                    str_refs,
                    durations,
                )?))
            }
        }
    }
}

/// Extension trait for Term to support data binding.
pub trait TermDataBinding {
    /// Bind a term to data in a context.
    fn bind_to_context(
        &self,
        context: &FormulaContext,
    ) -> Result<Vec<Box<dyn EventSpec>>, ValidationError>;
}

impl TermDataBinding for Term {
    fn bind_to_context(
        &self,
        context: &FormulaContext,
    ) -> Result<Vec<Box<dyn EventSpec>>, ValidationError> {
        let mut events = Vec::new();

        for event_name in &self.events {
            let var = context.resolve_variable(event_name)?;
            let event = context.create_event(event_name, &var)?;
            events.push(event);
        }

        Ok(events)
    }
}

/// Parse a formula with data binding.
pub fn parse_formula_with_context(
    formula: &str,
    context: &FormulaContext,
) -> Result<Vec<Box<dyn EventSpec>>, ValidationError> {
    let parser = FormulaParser::new();
    let parsed = parser.parse(formula)?;
    let terms = parser.to_terms(&parsed)?;

    let mut all_events = Vec::new();

    for term in terms {
        let events = term.bind_to_context(context)?;
        all_events.extend(events);
    }

    Ok(all_events)
}

/// Builder for formula context.
pub struct FormulaContextBuilder {
    context: FormulaContext,
}

impl FormulaContextBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            context: FormulaContext::new(),
        }
    }

    /// Add a DataFrame.
    pub fn with_dataframe(mut self, name: &str, df: DataFrame) -> Self {
        self.context.add_dataframe(name, df);
        self
    }

    /// Add events.
    pub fn with_events(mut self, name: &str, events: Vec<Box<dyn EventSpec>>) -> Self {
        self.context.add_events(name, events);
        self
    }

    /// Add a matrix.
    pub fn with_matrix(
        mut self,
        name: &str,
        data: Array2<f64>,
        row_names: Option<Vec<String>>,
        col_names: Option<Vec<String>>,
    ) -> Self {
        self.context.add_matrix(name, data, row_names, col_names);
        self
    }

    /// Set default source.
    pub fn default_source(mut self, name: &str) -> Self {
        self.context.set_default_source(name);
        self
    }

    /// Build the context.
    pub fn build(self) -> FormulaContext {
        self.context
    }
}

impl Default for FormulaContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dataframe_polars::DataFrameBuilder;

    #[test]
    fn test_context_with_dataframe() {
        let df = DataFrameBuilder::new()
            .with_column("x", vec![1.0, 2.0, 3.0])
            .unwrap()
            .with_column("y", vec![4.0, 5.0, 6.0])
            .unwrap()
            .build();

        let context = FormulaContext::with_dataframe(df);

        let var_x = context.resolve_variable("x").unwrap();
        match var_x {
            Variable::Numeric(values) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], 1.0);
            }
            _ => panic!("Expected numeric variable"),
        }
    }

    #[test]
    fn test_scale_function() {
        let df = DataFrameBuilder::new()
            .with_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap()
            .build();

        let context = FormulaContext::with_dataframe(df);

        let scaled = context
            .evaluate_expression(&FormulaExpr::FunctionCall {
                name: "scale".to_string(),
                args: vec![FormulaExpr::Variable("x".to_string())],
                kwargs: vec![],
            })
            .unwrap();

        match scaled {
            Variable::Numeric(values) => {
                assert_eq!(values.len(), 5);
                // Scaled values should have mean 0 and std 1
                assert!((values.mean().unwrap()).abs() < 1e-10);
                assert!((values.std(0.0) - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected numeric variable"),
        }
    }
}
