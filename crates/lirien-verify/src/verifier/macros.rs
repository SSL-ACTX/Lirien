//! Utility macros for verifier SMT assertion generation and error handling.
//!
//! Provides shortcuts to perform assertions, generate fresh unique SMT constants,
//! and instantiate verification errors.

/// Asserts a condition or implication directly into the active SMT solver context.
///
/// # Examples
/// ```ignore
/// solver_assert!(t_ctx, condition);
/// solver_assert!(t_ctx, premise => conclusion);
/// ```
#[macro_export]
macro_rules! solver_assert {
    ($t_ctx:expr, $cond:expr) => {
        let cond = $cond;
        $t_ctx.backend.assert(&cond);
    };
    ($t_ctx:expr, $premise:expr => $conclusion:expr) => {
        let premise = &$premise;
        let conclusion = &$conclusion;
        $t_ctx.backend.assert_implies(premise, conclusion);
    };
}

/// Generates a fresh SMT integer constant, suffixed with a unique verifier session ID.
#[macro_export]
macro_rules! fresh_int {
    ($t_ctx:expr, $name:expr) => {
        $t_ctx.backend.int_const(&format!("{}_{}", $name, $t_ctx.uid))
    };
}

/// Generates a fresh SMT boolean constant, suffixed with a unique verifier session ID.
#[macro_export]
macro_rules! fresh_bool {
    ($t_ctx:expr, $name:expr) => {
        $t_ctx.backend.bool_const(&format!("{}_{}", $name, $t_ctx.uid))
    };
}

/// Helper macro to instantiate a [`VerifierError`](crate::error::VerifierError) with or without source location.
#[macro_export]
macro_rules! verifier_error {
    ($variant:ident, $loc:expr, $($arg:tt)*) => {
        $crate::error::VerifierError::$variant(format!($($arg)*), Some($loc))
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::VerifierError::$variant(format!($($arg)*), None)
    };
}

