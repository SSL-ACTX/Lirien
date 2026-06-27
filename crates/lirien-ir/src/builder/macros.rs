//! Macros for JIT builder code generation and validation.
//!
//! This module provides utility macros to simplify instruction creation (`push_inst`),
//! type checking (`ensure_type`), and error formatting (`builder_error`).

/// Appends a new instruction to the builder and automatically attaches the current source location.
///
/// # Examples
/// ```ignore
/// push_inst!(builder, InstructionKind::Add(dest, lhs, rhs));
/// ```
#[macro_export]
macro_rules! push_inst {
    ($builder:expr, $kind:expr) => {{
        let loc = $builder.current_location;
        let kind = $kind;
        let inst = $builder.add_instruction(kind);
        if let Some(l) = loc {
            inst.location = Some(l);
        }
        inst
    }};
    ($builder:expr, $kind:expr, $loc:expr) => {{
        let kind = $kind;
        let inst = $builder.add_instruction(kind);
        inst.location = Some($loc);
        inst
    }};
}

/// Asserts that an SSA value matches an expected type, returning a `TypeMismatch` error if it doesn't.
#[macro_export]
macro_rules! ensure_type {
    ($builder:expr, $val:expr, $expected:expr, $loc:expr) => {{
        let found = $builder.func.get_type($val);
        if found != $expected {
            return Err($crate::builder::error::BuilderError::TypeMismatch {
                expected: format!("{:?}", $expected),
                found: format!("{:?}", found),
                location: Some($loc),
            });
        }
    }};
}

/// Helper macro to construct a [`BuilderError`](crate::builder::error::BuilderError).
///
/// Can construct specific error variants or generic/internal error messages with or without locations.
#[macro_export]
macro_rules! builder_error {
    (TypeMismatch, $expected:expr, $found:expr) => {
        $crate::builder::error::BuilderError::TypeMismatch {
            expected: $expected,
            found: $found,
            location: None,
        }
    };
    (AttributeNotFound, $target:expr, $attr:expr) => {
        $crate::builder::error::BuilderError::AttributeNotFound {
            target: $target,
            attr: $attr,
            location: None,
        }
    };
    ($variant:ident, $loc:expr; $($arg:tt)*) => {
        $crate::builder::error::BuilderError::$variant(format!($($arg)*), Some($loc))
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::builder::error::BuilderError::$variant(format!($($arg)*), None)
    };
}
