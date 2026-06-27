//! Error representation for the JIT compiler builder.
//!
//! This module defines the [`BuilderError`] enum and the [`BuilderResult`] type alias
//! used to track compilation errors when transforming Python AST into Lirien IR.

use crate::ir::SourceLocation;
use std::fmt;

/// Errors encountered when compiling Python AST into Lirien Intermediate Representation.
#[derive(Debug)]
pub enum BuilderError {
    /// Attempted to read a variable before it was defined.
    UnboundVariable(String, Option<SourceLocation>),
    /// A type mismatch occurred (e.g. assigning a float to an integer field).
    TypeMismatch {
        /// Description of the expected type.
        expected: String,
        /// Description of the actual type found.
        found: String,
        /// Source location where the mismatch occurred.
        location: Option<SourceLocation>,
    },
    /// Field or attribute access failed on a compound object.
    AttributeNotFound {
        /// Target object type or identifier.
        target: String,
        /// Attribute name that was not found.
        attr: String,
        /// Source location where the access occurred.
        location: Option<SourceLocation>,
    },
    /// Encountered a Python expression that is unsupported by Lirien.
    UnsupportedExpression(String, Option<SourceLocation>),
    /// Encountered a Python statement that is unsupported by Lirien.
    UnsupportedStatement(String, Option<SourceLocation>),
    /// A generic or user-defined compilation error.
    General(String, Option<SourceLocation>),
    /// An unexpected or compiler-internal assertion failure.
    Internal(String, Option<SourceLocation>),
}

impl BuilderError {
    /// Attaches or updates the source location on the compiler builder error.
    pub fn with_location(self, loc: SourceLocation) -> Self {
        match self {
            BuilderError::UnboundVariable(s, _) => BuilderError::UnboundVariable(s, Some(loc)),
            BuilderError::TypeMismatch {
                expected, found, ..
            } => BuilderError::TypeMismatch {
                expected,
                found,
                location: Some(loc),
            },
            BuilderError::AttributeNotFound { target, attr, .. } => {
                BuilderError::AttributeNotFound {
                    target,
                    attr,
                    location: Some(loc),
                }
            }
            BuilderError::UnsupportedExpression(s, _) => {
                BuilderError::UnsupportedExpression(s, Some(loc))
            }
            BuilderError::UnsupportedStatement(s, _) => {
                BuilderError::UnsupportedStatement(s, Some(loc))
            }
            BuilderError::General(s, _) => BuilderError::General(s, Some(loc)),
            BuilderError::Internal(s, _) => BuilderError::Internal(s, Some(loc)),
        }
    }
}

impl fmt::Display for BuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loc_str = |loc: &Option<SourceLocation>| {
            if let Some(l) = loc {
                format!(" at source offset {}", l.offset)
            } else {
                "".to_string()
            }
        };

        match self {
            BuilderError::UnboundVariable(s, loc) => {
                write!(f, "Unbound variable: {}{}", s, loc_str(loc))
            }
            BuilderError::TypeMismatch {
                expected,
                found,
                location,
            } => {
                write!(
                    f,
                    "Type mismatch: expected {}, found {}{}",
                    expected,
                    found,
                    loc_str(location)
                )
            }
            BuilderError::AttributeNotFound {
                target,
                attr,
                location,
            } => {
                write!(
                    f,
                    "Attribute '{}' not found on {}{}",
                    attr,
                    target,
                    loc_str(location)
                )
            }
            BuilderError::UnsupportedExpression(s, loc) => {
                write!(f, "Unsupported expression: {}{}", s, loc_str(loc))
            }
            BuilderError::UnsupportedStatement(s, loc) => {
                write!(f, "Unsupported statement: {}{}", s, loc_str(loc))
            }
            BuilderError::General(s, loc) => write!(f, "Builder error: {}{}", s, loc_str(loc)),
            BuilderError::Internal(s, loc) => {
                write!(f, "Internal builder error: {}{}", s, loc_str(loc))
            }
        }
    }
}

impl std::error::Error for BuilderError {}

impl From<String> for BuilderError {
    fn from(s: String) -> Self {
        BuilderError::General(s, None)
    }
}

/// Specialized Result type for the builder, yielding a [`BuilderError`] upon failure.
pub type BuilderResult<T> = Result<T, BuilderError>;
