//! Event specifications for fMRI design matrices.

pub mod cells;
pub mod dataframe_conversion;
pub mod event_basis;
pub mod event_factor;
pub mod event_hrf;
pub mod event_matrix;
pub mod event_term;
pub mod event_variable;

// Re-export main event types
pub use cells::Cells;
pub use dataframe_conversion::{events_from_dataframe, FromDataFrame};
pub use event_basis::EventBasis;
pub use event_factor::EventFactor;
pub use event_hrf::HrfEvent;
pub use event_matrix::EventMatrix;
pub use event_term::EventTerm;
pub use event_variable::EventVariable;
