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

#[macro_export]
macro_rules! fresh_int {
    ($t_ctx:expr, $name:expr) => {
        $t_ctx.backend.int_const(&format!("{}_{}", $name, $t_ctx.uid))
    };
}

#[macro_export]
macro_rules! fresh_bool {
    ($t_ctx:expr, $name:expr) => {
        $t_ctx.backend.bool_const(&format!("{}_{}", $name, $t_ctx.uid))
    };
}

#[macro_export]
macro_rules! verifier_error {
    ($variant:ident, $loc:expr, $($arg:tt)*) => {
        $crate::error::VerifierError::$variant(format!($($arg)*), Some($loc))
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::VerifierError::$variant(format!($($arg)*), None)
    };
}
