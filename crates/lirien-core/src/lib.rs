//! # Lirien Core
//!
//! Core primitives, diagnostics, and logging infrastructure for the Lirien JIT compiler.
//! Currently, this crate manages logging configuration using `tracing` and `tracing-subscriber`.

pub mod logging;

#[doc(inline)]
pub use logging::configure_tracing;
#[doc(inline)]
pub use logging::init as init_diagnostics;
#[doc(inline)]
pub use logging::set_log_level;

