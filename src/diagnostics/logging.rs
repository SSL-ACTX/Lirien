use std::collections::HashMap;
use std::sync::OnceLock;
use tracing_subscriber::{fmt, prelude::*, reload, EnvFilter};

type ReloadFn = Box<dyn Fn(EnvFilter) -> Result<(), String> + Send + Sync>;

static RELOAD_HANDLE: OnceLock<ReloadFn> = OnceLock::new();

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

    tracing::debug!("Lila diagnostics initialized.");
}

pub fn set_log_level(level: &str) -> Result<(), String> {
    if let Some(reload_fn) = RELOAD_HANDLE.get() {
        let new_filter = EnvFilter::try_new(level).map_err(|e| e.to_string())?;
        (reload_fn)(new_filter)?;
    }
    Ok(())
}

pub fn configure_tracing(config: HashMap<String, String>) -> Result<(), String> {
    let mut directives = Vec::new();
    for (component, level) in config {
        if component == "all" {
            directives.push(level);
        } else {
            directives.push(format!("lila::{}={}", component, level));
        }
    }
    set_log_level(&directives.join(","))
}
