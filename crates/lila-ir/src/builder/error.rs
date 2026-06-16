use crate::ir::SourceLocation;
use std::fmt;

#[derive(Debug)]
pub enum BuilderError {
    UnboundVariable(String, Option<SourceLocation>),
    TypeMismatch {
        expected: String,
        found: String,
        location: Option<SourceLocation>,
    },
    AttributeNotFound {
        target: String,
        attr: String,
        location: Option<SourceLocation>,
    },
    UnsupportedExpression(String, Option<SourceLocation>),
    UnsupportedStatement(String, Option<SourceLocation>),
    General(String, Option<SourceLocation>),
    Internal(String, Option<SourceLocation>),
}

impl BuilderError {
    pub fn with_location(self, loc: SourceLocation) -> Self {
        match self {
            BuilderError::UnboundVariable(s, _) => BuilderError::UnboundVariable(s, Some(loc)),
            BuilderError::TypeMismatch {
                expected,
                found, ..
            } => BuilderError::TypeMismatch {
                expected,
                found,
                location: Some(loc),
            },
            BuilderError::AttributeNotFound {
                target,
                attr, ..
            } => BuilderError::AttributeNotFound {
                target,
                attr,
                location: Some(loc),
            },
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

pub type BuilderResult<T> = Result<T, BuilderError>;
