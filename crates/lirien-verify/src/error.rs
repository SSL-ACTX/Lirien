//! Error representation for the formal verification engine.

use lirien_ir::ir::SourceLocation;
use std::fmt;

/// Errors encountered during JIT formal logic verification.
#[derive(Debug)]
pub enum VerifierError {
    /// A logical contradiction was detected (e.g. branch preconditions conflict).
    Contradiction(String),
    /// An error returned directly from the Z3 or SMT solver backend.
    BackendError(String),
    /// The verifier encountered an IR instruction not yet supported for verification.
    Unimplemented(String, Option<SourceLocation>),
    /// A type mismatch occurred in a verification assertion.
    TypeMismatch(String, String, Option<SourceLocation>),
    /// A user-supplied refinement type predicate assertion proved falsifiable.
    AssertionFailed(String, Option<SourceLocation>),
    /// A general verification verification error.
    General(String, Option<SourceLocation>),
}

impl VerifierError {
    /// Attaches or updates the source code location associated with the error.
    pub fn with_location(self, loc: SourceLocation) -> Self {

        match self {
            VerifierError::Unimplemented(s, _) => VerifierError::Unimplemented(s, Some(loc)),
            VerifierError::TypeMismatch(expected, found, _) => {
                VerifierError::TypeMismatch(expected, found, Some(loc))
            }
            VerifierError::AssertionFailed(message, _) => {
                VerifierError::AssertionFailed(message, Some(loc))
            }
            VerifierError::General(s, _) => VerifierError::General(s, Some(loc)),
            other => other,
        }
    }
}

impl fmt::Display for VerifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loc_str = |loc: &Option<SourceLocation>| {
            if let Some(l) = loc {
                format!(" at source offset {}", l.offset)
            } else {
                "".to_string()
            }
        };

        match self {
            VerifierError::Contradiction(s) => write!(f, "Logical contradiction: {}", s),
            VerifierError::BackendError(s) => write!(f, "Solver backend error: {}", s),
            VerifierError::Unimplemented(s, loc) => {
                write!(f, "Verification not implemented for: {}{}", s, loc_str(loc))
            }
            VerifierError::TypeMismatch(expected, found, location) => {
                write!(
                    f,
                    "Type mismatch in verification: expected {}, found {}{}",
                    expected,
                    found,
                    loc_str(location)
                )
            }
            VerifierError::AssertionFailed(message, location) => {
                write!(f, "Assertion failed: {}{}", message, loc_str(location))
            }
            VerifierError::General(s, loc) => write!(f, "Verification error: {}{}", s, loc_str(loc)),
        }
    }
}

impl std::error::Error for VerifierError {}

impl From<String> for VerifierError {
    fn from(s: String) -> Self {
        VerifierError::General(s, None)
    }
}

/// Specialized Result type for the verification engine, yielding a [`VerifierError`] upon failure.
pub type VerifierResult<T> = Result<T, VerifierError>;

