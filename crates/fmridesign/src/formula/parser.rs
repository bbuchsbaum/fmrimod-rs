//! Formula parser implementation for R-style formulas.

use super::term::Term;
use crate::types::ValidationError;
use nom::{
    branch::alt,
    bytes::complete::take_while1,
    character::complete::{char, multispace0},
    combinator::{map, opt, recognize},
    multi::{many0, separated_list0},
    sequence::{delimited, preceded, separated_pair, tuple},
    IResult,
};

/// Formula expression AST.
#[derive(Debug, Clone, PartialEq)]
pub enum FormulaExpr {
    /// Variable reference
    Variable(String),
    /// Function call (e.g., hrf(condition))
    FunctionCall {
        name: String,
        args: Vec<FormulaExpr>,
        kwargs: Vec<(String, FormulaExpr)>,
    },
    /// Addition of terms
    Addition(Box<FormulaExpr>, Box<FormulaExpr>),
    /// Interaction between terms
    Interaction(Box<FormulaExpr>, Box<FormulaExpr>),
    /// Basis expansion (@ operator)
    BasisExpansion(Box<FormulaExpr>, Box<FormulaExpr>),
    /// Pipe operation (|> operator)
    Pipe(Box<FormulaExpr>, Box<FormulaExpr>),
    /// String literal
    StringLiteral(String),
    /// Numeric literal
    Number(f64),
    /// Boolean literal
    Boolean(bool),
}

/// Parsed formula representation.
#[derive(Debug, Clone)]
pub struct ParsedFormula {
    /// Left-hand side (optional)
    pub lhs: Option<String>,
    /// Right-hand side expression
    pub rhs: FormulaExpr,
}

/// Formula parser.
pub struct FormulaParser;

impl FormulaParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self
    }

    /// Parse a complete formula string.
    pub fn parse(&self, input: &str) -> Result<ParsedFormula, ValidationError> {
        let input = input.trim();
        match parse_formula_internal(input) {
            Ok((remaining, formula)) => {
                let remaining = remaining.trim();
                if !remaining.is_empty() {
                    return Err(ValidationError::InvalidParameter(format!(
                        "Unexpected input after formula: '{}'",
                        remaining
                    )));
                }
                Ok(formula)
            }
            Err(e) => Err(ValidationError::InvalidParameter(format!(
                "Failed to parse formula: {:?}",
                e
            ))),
        }
    }

    /// Convert parsed formula to terms.
    pub fn to_terms(&self, formula: &ParsedFormula) -> Result<Vec<Term>, ValidationError> {
        expr_to_terms(&formula.rhs)
    }
}

// Parser combinators

/// Parse identifier (variable or function name).
fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        take_while1(|c: char| c.is_alphabetic() || c == '_'),
        opt(take_while1(|c: char| {
            c.is_alphanumeric() || c == '_' || c == '.'
        })),
    )))(input)
}

/// Parse a number.
fn number(input: &str) -> IResult<&str, f64> {
    map(
        recognize(tuple((
            opt(char('-')),
            take_while1(|c: char| c.is_numeric()),
            opt(tuple((char('.'), take_while1(|c: char| c.is_numeric())))),
        ))),
        |s: &str| s.parse::<f64>().unwrap(),
    )(input)
}

/// Parse a string literal.
fn string_literal(input: &str) -> IResult<&str, String> {
    alt((
        delimited(
            char('"'),
            map(take_while1(|c| c != '"'), String::from),
            char('"'),
        ),
        delimited(
            char('\''),
            map(take_while1(|c| c != '\''), String::from),
            char('\''),
        ),
    ))(input)
}

/// Parse a boolean literal.
fn boolean(input: &str) -> IResult<&str, bool> {
    alt((
        map(nom::bytes::complete::tag("TRUE"), |_| true),
        map(nom::bytes::complete::tag("True"), |_| true),
        map(nom::bytes::complete::tag("true"), |_| true),
        map(nom::bytes::complete::tag("FALSE"), |_| false),
        map(nom::bytes::complete::tag("False"), |_| false),
        map(nom::bytes::complete::tag("false"), |_| false),
    ))(input)
}

/// Parse keyword argument (name=value).
fn kwarg(input: &str) -> IResult<&str, (String, FormulaExpr)> {
    map(
        separated_pair(
            identifier,
            delimited(multispace0, char('='), multispace0),
            alt((
                map(number, FormulaExpr::Number),
                map(boolean, FormulaExpr::Boolean),
                map(string_literal, FormulaExpr::StringLiteral),
                map(identifier, |s| FormulaExpr::Variable(s.to_string())),
            )),
        ),
        |(name, value)| (name.to_string(), value),
    )(input)
}

/// Parse function arguments (positional and keyword).
fn function_args(input: &str) -> IResult<&str, (Vec<FormulaExpr>, Vec<(String, FormulaExpr)>)> {
    let (input, args_and_kwargs) = separated_list0(
        delimited(multispace0, char(','), multispace0),
        alt((
            // Try to parse as kwarg first
            map(kwarg, |kw| (None, Some(kw))),
            // Otherwise parse as positional arg
            map(expression, |expr| (Some(expr), None)),
        )),
    )(input)?;

    // Separate positional and keyword arguments
    let mut args = Vec::new();
    let mut kwargs = Vec::new();
    let mut seen_kwarg = false;

    for (arg, kwarg) in args_and_kwargs {
        if let Some(kw) = kwarg {
            seen_kwarg = true;
            kwargs.push(kw);
        } else if let Some(a) = arg {
            if seen_kwarg {
                // Positional arg after keyword arg is an error
                return Err(nom::Err::Error(nom::error::Error::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )));
            }
            args.push(a);
        }
    }

    Ok((input, (args, kwargs)))
}

/// Parse a primary expression (variable, literal, or function call).
fn primary_expr(input: &str) -> IResult<&str, FormulaExpr> {
    delimited(
        multispace0,
        alt((
            // Parenthesized expression
            delimited(char('('), expression, char(')')),
            // Number literal
            map(number, FormulaExpr::Number),
            // Boolean literal
            map(boolean, FormulaExpr::Boolean),
            // String literal
            map(string_literal, FormulaExpr::StringLiteral),
            // Function call or variable
            map(
                tuple((
                    identifier,
                    opt(delimited(char('('), function_args, char(')'))),
                )),
                |(name, args_kwargs)| {
                    if let Some((args, kwargs)) = args_kwargs {
                        FormulaExpr::FunctionCall {
                            name: name.to_string(),
                            args,
                            kwargs,
                        }
                    } else {
                        FormulaExpr::Variable(name.to_string())
                    }
                },
            ),
        )),
        multispace0,
    )(input)
}

/// Parse basis expansion (using @ operator).
fn basis_expansion_expr(input: &str) -> IResult<&str, FormulaExpr> {
    let (input, first) = primary_expr(input)?;

    // Parse additional basis expansion terms if any
    let (input, rest) = many0(preceded(
        delimited(multispace0, char('@'), multispace0),
        primary_expr,
    ))(input)?;

    // Fold all terms into a left-associative tree
    Ok((
        input,
        rest.into_iter().fold(first, |acc, expr| {
            FormulaExpr::BasisExpansion(Box::new(acc), Box::new(expr))
        }),
    ))
}

/// Parse interaction term (using : operator).
fn interaction_expr(input: &str) -> IResult<&str, FormulaExpr> {
    let (input, first) = basis_expansion_expr(input)?;

    // Parse additional interaction terms if any
    let (input, rest) = many0(preceded(
        delimited(multispace0, char(':'), multispace0),
        basis_expansion_expr,
    ))(input)?;

    // Fold all terms into a left-associative tree
    Ok((
        input,
        rest.into_iter().fold(first, |acc, expr| {
            FormulaExpr::Interaction(Box::new(acc), Box::new(expr))
        }),
    ))
}

/// Parse addition expression.
fn addition_expr(input: &str) -> IResult<&str, FormulaExpr> {
    let (input, first) = interaction_expr(input)?;

    // Parse additional terms if any
    let (input, rest) = many0(preceded(
        delimited(multispace0, char('+'), multispace0),
        interaction_expr,
    ))(input)?;

    // Fold all terms into a left-associative tree
    Ok((
        input,
        rest.into_iter().fold(first, |acc, expr| {
            FormulaExpr::Addition(Box::new(acc), Box::new(expr))
        }),
    ))
}

/// Parse pipe expression (|> operator).
fn pipe_expr(input: &str) -> IResult<&str, FormulaExpr> {
    let (input, first) = addition_expr(input)?;

    // Parse additional pipe operations if any
    let (input, rest) = many0(preceded(
        delimited(multispace0, nom::bytes::complete::tag("|>"), multispace0),
        addition_expr,
    ))(input)?;

    // Fold all operations into a left-associative tree
    Ok((
        input,
        rest.into_iter().fold(first, |acc, expr| {
            FormulaExpr::Pipe(Box::new(acc), Box::new(expr))
        }),
    ))
}

/// Parse a complete expression.
fn expression(input: &str) -> IResult<&str, FormulaExpr> {
    pipe_expr(input)
}

/// Parse a complete formula (with optional LHS).
fn parse_formula_internal(input: &str) -> IResult<&str, ParsedFormula> {
    delimited(
        multispace0,
        alt((
            // Formula with LHS and RHS
            map(
                separated_pair(
                    identifier,
                    delimited(multispace0, char('~'), multispace0),
                    expression,
                ),
                |(lhs, rhs)| ParsedFormula {
                    lhs: Some(lhs.to_string()),
                    rhs,
                },
            ),
            // Just RHS (no ~)
            map(expression, |rhs| ParsedFormula { lhs: None, rhs }),
        )),
        multispace0,
    )(input)
}

/// Convert expression to terms.
fn expr_to_terms(expr: &FormulaExpr) -> Result<Vec<Term>, ValidationError> {
    match expr {
        FormulaExpr::Addition(left, right) => {
            let mut terms = expr_to_terms(left)?;
            terms.extend(expr_to_terms(right)?);
            Ok(terms)
        }
        FormulaExpr::Pipe(left, right) => {
            // Pipe operator applies the right side to the left side
            apply_pipe_operation(left, right)
        }
        FormulaExpr::Variable(name) => Ok(vec![Term::new(name)]),
        FormulaExpr::Interaction(left, right) => {
            let left_terms = expr_to_terms(left)?;
            let right_terms = expr_to_terms(right)?;

            if left_terms.len() != 1 || right_terms.len() != 1 {
                return Err(ValidationError::InvalidParameter(
                    "Interaction operands must each resolve to exactly one term".to_string(),
                ));
            }

            let left_term = &left_terms[0];
            let right_term = &right_terms[0];

            let mut all_vars = left_term.events.clone();
            all_vars.extend(right_term.events.clone());

            let mut term = Term::interaction(all_vars)?;

            if let Some(hrf) = left_term.hrf.clone().or_else(|| right_term.hrf.clone()) {
                term = term.with_hrf(hrf);
            }

            for (key, value) in &left_term.params {
                term.params.insert(key.clone(), value.clone());
            }
            for (key, value) in &right_term.params {
                term.params.insert(key.clone(), value.clone());
            }

            Ok(vec![term])
        }
        FormulaExpr::FunctionCall { name, args, kwargs } => {
            match name.as_str() {
                "hrf" => {
                    if args.is_empty() {
                        return Err(ValidationError::InvalidParameter(
                            "hrf() requires at least one argument".to_string(),
                        ));
                    }

                    let inner_terms = expr_to_terms(&args[0])?;
                    if inner_terms.len() != 1 {
                        return Err(ValidationError::InvalidParameter(
                            "hrf() argument must resolve to exactly one term".to_string(),
                        ));
                    }
                    let mut term = inner_terms.into_iter().next().unwrap();

                    // Check for HRF type in second positional argument
                    if args.len() > 1 {
                        if let FormulaExpr::StringLiteral(hrf_name) = &args[1] {
                            term = term.with_hrf(hrf_name.as_str());
                        }
                    } else {
                        // Default to canonical HRF
                        term = term.with_hrf("canonical");
                    }

                    // Process keyword arguments
                    for (key, value) in kwargs {
                        match key.as_str() {
                            "hrf" | "basis" => {
                                if let FormulaExpr::StringLiteral(s) = value {
                                    term = term.with_hrf(s.as_str());
                                }
                            }
                            _ => {
                                // Store other kwargs as parameters
                                match value {
                                    FormulaExpr::StringLiteral(s) => {
                                        term = term.set_param(key, s.as_str());
                                    }
                                    FormulaExpr::Number(n) => {
                                        term = term.set_param(key, *n);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    Ok(vec![term])
                }
                "trialwise" => {
                    // Trialwise single-trial regressors
                    use crate::trialwise::TrialwiseConfig;

                    let mut config = TrialwiseConfig::default();

                    // Process keyword arguments
                    for (key, value) in kwargs {
                        match key.as_str() {
                            "basis" | "hrf" => {
                                if let FormulaExpr::StringLiteral(s) = value {
                                    config.basis = s.clone();
                                }
                            }
                            "lag" => {
                                if let FormulaExpr::Number(n) = value {
                                    config.lag = *n;
                                }
                            }
                            "nbasis" => {
                                if let FormulaExpr::Number(n) = value {
                                    config.nbasis = *n as usize;
                                }
                            }
                            "add_sum" => {
                                if let FormulaExpr::Boolean(b) = value {
                                    config.add_sum = *b;
                                }
                            }
                            "label" => {
                                if let FormulaExpr::StringLiteral(s) = value {
                                    config.label = s.clone();
                                }
                            }
                            _ => {}
                        }
                    }

                    let term = crate::trialwise::trialwise_with_config(config);
                    Ok(vec![term])
                }
                "poly" => {
                    // Polynomial basis function
                    if args.is_empty() {
                        return Err(ValidationError::InvalidParameter(
                            "poly() requires at least one argument".to_string(),
                        ));
                    }

                    let event_vars = extract_variables(&args[0])?;
                    let mut term = if event_vars.len() == 1 {
                        Term::new(&event_vars[0])
                    } else {
                        Term::interaction(event_vars)?
                    };

                    // Get degree from second argument or kwargs
                    let degree = if args.len() > 1 {
                        if let FormulaExpr::Number(n) = &args[1] {
                            *n as i32
                        } else {
                            2
                        }
                    } else {
                        // Check kwargs for degree
                        kwargs
                            .iter()
                            .find(|(k, _)| k == "degree")
                            .and_then(|(_, v)| {
                                if let FormulaExpr::Number(n) = v {
                                    Some(*n as i32)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(2)
                    };

                    // TODO: Create polynomial basis and attach to term
                    term = term.set_param("poly_degree", degree as f64);

                    Ok(vec![term])
                }
                "bs" => {
                    // B-spline basis function
                    if args.is_empty() {
                        return Err(ValidationError::InvalidParameter(
                            "bs() requires at least one argument".to_string(),
                        ));
                    }

                    let event_vars = extract_variables(&args[0])?;
                    let mut term = if event_vars.len() == 1 {
                        Term::new(&event_vars[0])
                    } else {
                        Term::interaction(event_vars)?
                    };

                    // Get df (degrees of freedom) from second argument or kwargs
                    let df = if args.len() > 1 {
                        if let FormulaExpr::Number(n) = &args[1] {
                            *n as i32
                        } else {
                            4 // Default df
                        }
                    } else {
                        // Check kwargs for df
                        kwargs
                            .iter()
                            .find(|(k, _)| k == "df")
                            .and_then(|(_, v)| {
                                if let FormulaExpr::Number(n) = v {
                                    Some(*n as i32)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(4)
                    };

                    term = term.set_param("bs_df", df as f64);

                    Ok(vec![term])
                }
                "scale" => {
                    // Scale/standardize a variable
                    if args.is_empty() {
                        return Err(ValidationError::InvalidParameter(
                            "scale() requires at least one argument".to_string(),
                        ));
                    }

                    let event_vars = extract_variables(&args[0])?;
                    let mut term = if event_vars.len() == 1 {
                        Term::new(&event_vars[0])
                    } else {
                        Term::interaction(event_vars)?
                    };

                    // Get centering option from kwargs
                    let center = kwargs
                        .iter()
                        .find(|(k, _)| k == "center")
                        .and_then(|(_, v)| match v {
                            FormulaExpr::Boolean(b) => Some(*b),
                            _ => None,
                        })
                        .unwrap_or(true);

                    // Get scaling option from kwargs
                    let scale_bool = kwargs
                        .iter()
                        .find(|(k, _)| k == "scale")
                        .and_then(|(_, v)| match v {
                            FormulaExpr::Boolean(b) => Some(*b),
                            _ => None,
                        })
                        .unwrap_or(true);

                    term = term.set_param("scale_center", if center { 1.0 } else { 0.0 });
                    term = term.set_param("scale_scale", if scale_bool { 1.0 } else { 0.0 });

                    Ok(vec![term])
                }
                "covariate" => {
                    // Covariate function - no HRF convolution
                    use crate::covariate::CovariateConfig;
                    let mut config = CovariateConfig::default();

                    // Extract variable names from args
                    let mut covariate_names = Vec::new();
                    for arg in args {
                        if let Ok(vars) = extract_variables(arg) {
                            covariate_names.extend(vars);
                        }
                    }

                    // If no arguments, we'll use all numeric columns from the data later
                    // For now, use a special marker
                    if covariate_names.is_empty() {
                        covariate_names.push("_all_covariates_".to_string());
                    }

                    config.covariates = covariate_names;

                    // Process kwargs
                    for (key, value) in kwargs {
                        match (key.as_str(), value) {
                            ("prefix", FormulaExpr::StringLiteral(s)) => {
                                config.prefix = Some(s.clone());
                            }
                            _ => {}
                        }
                    }

                    let term = crate::covariate::covariate_with_config(config);
                    Ok(vec![term])
                }
                _ => Err(ValidationError::InvalidParameter(format!(
                    "Unknown function: {}",
                    name
                ))),
            }
        }
        FormulaExpr::BasisExpansion(left, right) => {
            // Handle basis expansion using @ operator
            // Left side is the variable(s), right side is the basis specification
            let event_vars = extract_variables(left)?;
            let mut term = if event_vars.len() == 1 {
                Term::new(&event_vars[0])
            } else {
                Term::interaction(event_vars)?
            };

            // Handle right side - could be a function call like poly(2) or bs(df=4)
            match right.as_ref() {
                FormulaExpr::FunctionCall { name, args, kwargs } => {
                    match name.as_str() {
                        "poly" => {
                            // Extract degree
                            let degree = if !args.is_empty() {
                                if let FormulaExpr::Number(n) = &args[0] {
                                    *n as i32
                                } else {
                                    2
                                }
                            } else {
                                kwargs
                                    .iter()
                                    .find(|(k, _)| k == "degree")
                                    .and_then(|(_, v)| {
                                        if let FormulaExpr::Number(n) = v {
                                            Some(*n as i32)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(2)
                            };
                            term = term.set_param("basis_type", "poly");
                            term = term.set_param("poly_degree", degree as f64);
                        }
                        "bs" => {
                            // Extract df
                            let df = if !args.is_empty() {
                                if let FormulaExpr::Number(n) = &args[0] {
                                    *n as i32
                                } else {
                                    4
                                }
                            } else {
                                kwargs
                                    .iter()
                                    .find(|(k, _)| k == "df")
                                    .and_then(|(_, v)| {
                                        if let FormulaExpr::Number(n) = v {
                                            Some(*n as i32)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(4)
                            };
                            term = term.set_param("basis_type", "bs");
                            term = term.set_param("bs_df", df as f64);
                        }
                        _ => {
                            return Err(ValidationError::InvalidParameter(format!(
                                "Unknown basis function: {}",
                                name
                            )));
                        }
                    }
                }
                FormulaExpr::Variable(basis_name) => {
                    // Simple basis name like @linear or @quadratic
                    term = term.set_param("basis_type", basis_name.as_str());
                }
                _ => {
                    return Err(ValidationError::InvalidParameter(
                        "Invalid basis specification after @ operator".to_string(),
                    ));
                }
            }

            Ok(vec![term])
        }
        _ => Err(ValidationError::InvalidParameter(
            "Invalid expression type in formula".to_string(),
        )),
    }
}

/// Apply pipe operation - right side is applied to left side.
fn apply_pipe_operation(
    left: &FormulaExpr,
    right: &FormulaExpr,
) -> Result<Vec<Term>, ValidationError> {
    // Get the terms from the left side
    let left_terms = expr_to_terms(left)?;

    // Apply the right side operation to each term
    let mut result_terms = Vec::new();

    for term in left_terms {
        // Right side should be a function call that modifies the term
        match right {
            FormulaExpr::FunctionCall { name, args, kwargs } => {
                let modified_term = apply_function_to_term(term, name, args, kwargs)?;
                result_terms.push(modified_term);
            }
            FormulaExpr::Variable(func_name) => {
                // Simple function name without args
                let modified_term =
                    apply_function_to_term(term, func_name, &Vec::new(), &Vec::new())?;
                result_terms.push(modified_term);
            }
            _ => {
                return Err(ValidationError::InvalidParameter(
                    "Pipe operator expects a function on the right side".to_string(),
                ));
            }
        }
    }

    Ok(result_terms)
}

/// Apply a function to a term.
fn apply_function_to_term(
    mut term: Term,
    func_name: &str,
    args: &[FormulaExpr],
    kwargs: &[(String, FormulaExpr)],
) -> Result<Term, ValidationError> {
    match func_name {
        "hrf" => {
            // Apply HRF to the term
            let hrf_name = if !args.is_empty() {
                if let FormulaExpr::StringLiteral(s) = &args[0] {
                    s.as_str()
                } else {
                    "canonical"
                }
            } else {
                // Check kwargs for hrf specification
                kwargs
                    .iter()
                    .find(|(k, _)| k == "hrf" || k == "basis")
                    .and_then(|(_, v)| {
                        if let FormulaExpr::StringLiteral(s) = v {
                            Some(s.as_str())
                        } else {
                            None
                        }
                    })
                    .unwrap_or("canonical")
            };

            term = term.with_hrf(hrf_name);

            // Apply any other parameters
            for (key, value) in kwargs {
                match value {
                    FormulaExpr::StringLiteral(s) => {
                        term = term.set_param(key, s.as_str());
                    }
                    FormulaExpr::Number(n) => {
                        term = term.set_param(key, *n);
                    }
                    _ => {}
                }
            }

            Ok(term)
        }
        "scale" => {
            // Apply scaling parameters
            let center = kwargs
                .iter()
                .find(|(k, _)| k == "center")
                .and_then(|(_, v)| match v {
                    FormulaExpr::Boolean(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(true);

            let scale_bool = kwargs
                .iter()
                .find(|(k, _)| k == "scale")
                .and_then(|(_, v)| match v {
                    FormulaExpr::Boolean(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(true);

            term = term.set_param("scale_center", if center { 1.0 } else { 0.0 });
            term = term.set_param("scale_scale", if scale_bool { 1.0 } else { 0.0 });

            Ok(term)
        }
        "poly" => {
            // Apply polynomial basis
            let degree = if !args.is_empty() {
                if let FormulaExpr::Number(n) = &args[0] {
                    *n as i32
                } else {
                    2
                }
            } else {
                kwargs
                    .iter()
                    .find(|(k, _)| k == "degree")
                    .and_then(|(_, v)| {
                        if let FormulaExpr::Number(n) = v {
                            Some(*n as i32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(2)
            };

            term = term.set_param("basis_type", "poly");
            term = term.set_param("poly_degree", degree as f64);

            Ok(term)
        }
        "bs" => {
            // Apply B-spline basis
            let df = if !args.is_empty() {
                if let FormulaExpr::Number(n) = &args[0] {
                    *n as i32
                } else {
                    4
                }
            } else {
                kwargs
                    .iter()
                    .find(|(k, _)| k == "df")
                    .and_then(|(_, v)| {
                        if let FormulaExpr::Number(n) = v {
                            Some(*n as i32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(4)
            };

            term = term.set_param("basis_type", "bs");
            term = term.set_param("bs_df", df as f64);

            Ok(term)
        }
        _ => Err(ValidationError::InvalidParameter(format!(
            "Unknown function in pipe: {}",
            func_name
        ))),
    }
}

/// Extract variable names from an expression.
fn extract_variables(expr: &FormulaExpr) -> Result<Vec<String>, ValidationError> {
    match expr {
        FormulaExpr::Variable(name) => Ok(vec![name.clone()]),
        FormulaExpr::Interaction(left, right) => {
            let mut vars = extract_variables(left)?;
            vars.extend(extract_variables(right)?);
            Ok(vars)
        }
        FormulaExpr::FunctionCall { name, args, .. } => {
            // Special case for trialwise() which doesn't need arguments
            if name == "trialwise" {
                Ok(vec!["trialwise".to_string()])
            } else if args.is_empty() {
                // For function calls like poly(rt, 2), extract the variable from the first argument
                Err(ValidationError::InvalidParameter(format!(
                    "Function {} requires arguments",
                    name
                )))
            } else {
                extract_variables(&args[0])
            }
        }
        FormulaExpr::BasisExpansion(left, _) => {
            // For basis expansion, extract variables from the left side only
            extract_variables(left)
        }
        _ => Err(ValidationError::InvalidParameter(
            "Expected variable or interaction in expression".to_string(),
        )),
    }
}

/// Parse a formula string and convert to terms.
pub fn parse_formula(input: &str) -> Result<Vec<Term>, ValidationError> {
    let parser = FormulaParser::new();
    let parsed = parser.parse(input)?;
    parser.to_terms(&parsed)
}

#[cfg(test)]
mod tests {
    use super::super::term::{HrfSpec, TermParam};
    use super::*;

    #[test]
    fn test_parse_simple_variable() {
        let parser = FormulaParser::new();
        let result = parser.parse("condition").unwrap();
        assert!(result.lhs.is_none());
        assert_eq!(result.rhs, FormulaExpr::Variable("condition".to_string()));
    }

    #[test]
    fn test_parse_addition() {
        let parser = FormulaParser::new();
        let result = parser.parse("condition + rating").unwrap();
        assert!(result.lhs.is_none());

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[0].events, vec!["condition"]);
        assert_eq!(terms[1].events, vec!["rating"]);
    }

    #[test]
    fn test_parse_interaction() {
        let parser = FormulaParser::new();
        let result = parser.parse("condition:block").unwrap();

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].events, vec!["condition", "block"]);
        assert!(terms[0].is_interaction());
    }

    #[test]
    fn test_parse_hrf_function() {
        let parser = FormulaParser::new();
        let result = parser.parse("hrf(condition)").unwrap();

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].events, vec!["condition"]);
    }

    #[test]
    fn test_parse_formula_with_lhs() {
        let parser = FormulaParser::new();
        let result = parser.parse("onset ~ condition + rating").unwrap();
        assert_eq!(result.lhs, Some("onset".to_string()));

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 2);
    }

    #[test]
    fn test_parse_complex_formula() {
        let parser = FormulaParser::new();
        let result = parser
            .parse("onset ~ hrf(condition) + hrf(rating) + condition:block")
            .unwrap();

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 3);
        assert_eq!(terms[2].events, vec!["condition", "block"]);
    }

    #[test]
    fn test_parse_hrf_with_kwargs() {
        let parser = FormulaParser::new();
        let result = parser.parse("hrf(condition, hrf=\"gamma\")").unwrap();

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].events, vec!["condition"]);
        match &terms[0].hrf {
            Some(HrfSpec::Named(name)) => assert_eq!(name, "gamma"),
            _ => panic!("Expected named HRF"),
        }
    }

    #[test]
    fn test_parse_multiple_kwargs() {
        let parser = FormulaParser::new();
        let result = parser
            .parse("hrf(condition, hrf=\"spm_canonical\", duration=2.0)")
            .unwrap();

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);

        // Check HRF
        match &terms[0].hrf {
            Some(HrfSpec::Named(name)) => assert_eq!(name, "spm_canonical"),
            _ => panic!("Expected named HRF"),
        }

        // Check duration parameter
        match terms[0].params.get("duration") {
            Some(TermParam::Number(n)) => assert_eq!(*n, 2.0),
            _ => panic!("Expected duration parameter"),
        }
    }

    #[test]
    fn test_parse_poly_function() {
        let parser = FormulaParser::new();
        let result = parser.parse("poly(rating, 3)").unwrap();

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].events, vec!["rating"]);

        // Check polynomial degree parameter
        match terms[0].params.get("poly_degree") {
            Some(TermParam::Number(n)) => assert_eq!(*n, 3.0),
            _ => panic!("Expected poly_degree parameter"),
        }
    }

    #[test]
    fn test_parse_trialwise_function() {
        let parser = FormulaParser::new();
        let result = parser.parse("trialwise(basis=\"spmg1\", lag=0.5)").unwrap();

        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].events, vec!["trialwise"]);

        // Check parameters
        match terms[0].params.get("basis") {
            Some(TermParam::String(s)) => assert_eq!(s, "spmg1"),
            _ => panic!("Expected basis parameter"),
        }

        match terms[0].params.get("lag") {
            Some(TermParam::Number(n)) => assert_eq!(*n, 0.5),
            _ => panic!("Expected lag parameter"),
        }
    }

    #[test]
    fn test_parse_pipe_operator() {
        let parser = FormulaParser::new();

        // Simple pipe
        let result = parser.parse("condition |> hrf()").unwrap();
        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].events, vec!["condition"]);
        assert!(terms[0].hrf.is_some());

        // Pipe with HRF specification
        let result2 = parser.parse("condition |> hrf(\"gamma\")").unwrap();
        let terms2 = parser.to_terms(&result2).unwrap();
        assert_eq!(terms2.len(), 1);
        match &terms2[0].hrf {
            Some(HrfSpec::Named(name)) => assert_eq!(name, "gamma"),
            _ => panic!("Expected gamma HRF"),
        }
    }

    #[test]
    fn test_parse_chained_pipes() {
        let parser = FormulaParser::new();

        // Chain multiple operations
        let result = parser.parse("rating |> scale() |> hrf()").unwrap();
        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].events, vec!["rating"]);

        // Check that both scale and hrf were applied
        assert!(terms[0].hrf.is_some());
        assert!(terms[0].params.contains_key("scale_center"));
        assert!(terms[0].params.contains_key("scale_scale"));
    }

    #[test]
    fn test_parse_pipe_with_addition() {
        let parser = FormulaParser::new();

        // Pipe applies to entire addition expression
        let result = parser.parse("condition + rating |> hrf()").unwrap();
        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 2);

        // Both terms should have HRF applied
        assert!(terms[0].hrf.is_some());
        assert!(terms[1].hrf.is_some());
    }

    #[test]
    fn test_parse_pipe_with_basis() {
        let parser = FormulaParser::new();

        // Apply polynomial basis via pipe
        let result = parser.parse("rating |> poly(3)").unwrap();
        let terms = parser.to_terms(&result).unwrap();
        assert_eq!(terms.len(), 1);

        // Check polynomial parameters
        match terms[0].params.get("basis_type") {
            Some(TermParam::String(s)) => assert_eq!(s, "poly"),
            _ => panic!("Expected basis_type parameter"),
        }
        match terms[0].params.get("poly_degree") {
            Some(TermParam::Number(n)) => assert_eq!(*n, 3.0),
            _ => panic!("Expected poly_degree parameter"),
        }
    }
}
