//! Logging and diagnostics management for Lirien.
//!
//! This module provides functions to initialize and dynamically reconfigure
//! the logging subsystem using the `tracing` ecosystem.
//! It supports setting log levels via environment variables (`LILA_LOG`)
//! or at runtime.

use std::collections::HashMap;
use std::sync::OnceLock;
use tracing_subscriber::{fmt, prelude::*, reload, EnvFilter};

/// Type alias for the reload function stored in the global reload handle.
type ReloadFn = Box<dyn Fn(EnvFilter) -> Result<(), String> + Send + Sync>;

/// Global once-initialized reload handle to dynamically adjust tracing levels.
static RELOAD_HANDLE: OnceLock<ReloadFn> = OnceLock::new();

/// Initializes the global tracing subscriber.
///
/// By default, it reads the `LILA_LOG` environment variable for filter directives,
/// defaulting to the `info` level.
/// The output format can be controlled via `LILA_LOG_FORMAT`:
/// - `"compact"`: Uses a compact, target-annotated layout.
/// - Any other value (defaulting to `"full"`): Uses the full layout.
///
/// A reload handle is stored internally to allow runtime log level adjustments
/// via [`set_log_level`] and [`configure_tracing`].
pub fn init() {
    let filter = EnvFilter::try_from_env("LILA_LOG")
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    let (filter, handle) = reload::Layer::new(filter);

    let format = std::env::var("LILA_LOG_FORMAT").unwrap_or_else(|_| "full".to_string());

    let registry = tracing_subscriber::registry().with(filter);

    if format == "compact" {
        let _ = registry
            .with(fmt::layer().compact().with_target(true))
            .try_init();
    } else {
        let _ = registry.with(fmt::layer().with_target(true)).try_init();
    }

    let _ = RELOAD_HANDLE.set(Box::new(move |new_filter| {
        handle.reload(new_filter).map_err(|e| e.to_string())
    }));

    tracing::debug!("Lirien diagnostics initialized.");
}

/// Dynamically updates the global log level using a filter directive string.
///
/// The filter string follows the `tracing-subscriber::EnvFilter` syntax
/// (e.g., `"info,lirien_ir=debug"`).
///
/// # Errors
/// Returns an error string if the directive is invalid or if the logging system
/// has not been initialized.
pub fn set_log_level(level: &str) -> Result<(), String> {
    if let Some(reload_fn) = RELOAD_HANDLE.get() {
        let new_filter = EnvFilter::try_new(level).map_err(|e| e.to_string())?;
        (reload_fn)(new_filter)?;
    }
    Ok(())
}

/// Configures tracing using a map of component names to their target log levels.
///
/// If a component key is `"all"`, it specifies the global log level directive.
/// Otherwise, it maps the component to `lirien::<component>=<level>`.
///
/// # Errors
/// Returns an error if any of the generated directives fail to parse.
pub fn configure_tracing(config: HashMap<String, String>) -> Result<(), String> {
    let mut directives = Vec::new();
    for (component, level) in config {
        if component == "all" {
            directives.push(level);
        } else {
            directives.push(format!("lirien::{}={}", component, level));
        }
    }
    set_log_level(&directives.join(","))
}

