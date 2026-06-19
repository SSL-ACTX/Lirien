use cranelift_module::ModuleError;
use lirien_ir::ir::SourceLocation;
use std::fmt;

#[derive(Debug)]
pub enum LoweringError {
    ModuleError(Box<ModuleError>, Option<SourceLocation>),
    General(String, Option<SourceLocation>),
    InstructionNotSupported(String, Option<SourceLocation>),
    TypeMismatch {
        expected: String,
        found: String,
        location: Option<SourceLocation>,
    },
    SymbolNotFound(String, Option<SourceLocation>),
    MissingMetadata(String, Option<SourceLocation>),
}

impl LoweringError {
    pub fn with_location(self, loc: SourceLocation) -> Self {
        match self {
            LoweringError::ModuleError(e, _) => LoweringError::ModuleError(e, Some(loc)),
            LoweringError::General(s, _) => LoweringError::General(s, Some(loc)),
            LoweringError::InstructionNotSupported(s, _) => {
                LoweringError::InstructionNotSupported(s, Some(loc))
            }
            LoweringError::TypeMismatch {
                expected,
                found, ..
            } => LoweringError::TypeMismatch {
                expected,
                found,
                location: Some(loc),
            },
            LoweringError::SymbolNotFound(s, _) => LoweringError::SymbolNotFound(s, Some(loc)),
            LoweringError::MissingMetadata(s, _) => LoweringError::MissingMetadata(s, Some(loc)),
        }
    }
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loc_str = |loc: &Option<SourceLocation>| {
            if let Some(l) = loc {
                format!(" at source offset {}", l.offset)
            } else {
                "".to_string()
            }
        };

        match self {
            LoweringError::ModuleError(e, loc) => {
                write!(f, "Cranelift Module Error: {}{}", e, loc_str(loc))
            }
            LoweringError::General(s, loc) => write!(f, "Lowering Error: {}{}", s, loc_str(loc)),
            LoweringError::InstructionNotSupported(s, loc) => {
                write!(f, "Instruction not supported: {}{}", s, loc_str(loc))
            }
            LoweringError::TypeMismatch {
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
            LoweringError::SymbolNotFound(s, loc) => {
                write!(f, "Symbol not found: {}{}", s, loc_str(loc))
            }
            LoweringError::MissingMetadata(s, loc) => {
                write!(f, "Missing metadata: {}{}", s, loc_str(loc))
            }
        }
    }
}

impl std::error::Error for LoweringError {}

impl From<ModuleError> for LoweringError {
    fn from(e: ModuleError) -> Self {
        LoweringError::ModuleError(Box::new(e), None)
    }
}

impl From<String> for LoweringError {
    fn from(s: String) -> Self {
        LoweringError::General(s, None)
    }
}
