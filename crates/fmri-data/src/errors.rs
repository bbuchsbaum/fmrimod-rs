use thiserror::Error;

#[derive(Debug, Error)]
pub enum FmriDatasetError {
    #[error("invalid {field}: {reason}")]
    Invalid { field: &'static str, reason: String },
    #[error("shape mismatch: expected {expected:?}, got {found:?}")]
    ShapeMismatch {
        expected: [usize; 2],
        found: [usize; 2],
    },
    #[error("unknown {kind}: {id}")]
    UnknownId { kind: &'static str, id: String },
    #[error("feature spaces differ")]
    SpaceMismatch,
    #[error("size arithmetic overflow")]
    SizeOverflow,
    #[error("numeric buffer budget exceeded: need {required} bytes, have {budget}")]
    BudgetExceeded { required: usize, budget: usize },
    #[error("allocation failed: {0}")]
    Allocation(String),
    #[error("length mismatch: expected {expected}, got {found}")]
    LengthMismatch { expected: usize, found: usize },
    #[error("invalid selection index: {0}")]
    InvalidIndex(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub(crate) fn invalid(field: &'static str, reason: impl Into<String>) -> FmriDatasetError {
    FmriDatasetError::Invalid {
        field,
        reason: reason.into(),
    }
}

pub type Result<T> = std::result::Result<T, FmriDatasetError>;
