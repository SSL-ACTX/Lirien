use std::sync::OnceLock;
use tracing_subscriber::{fmt, prelude::*, reload, EnvFilter};

type ReloadFn = Box<dyn Fn(EnvFilter) -> Result<(), String> + Send + Sync>;

static RELOAD_HANDLE: OnceLock<ReloadFn> = OnceLock::new();

pub fn init() {
    let filter = EnvFilter::try_from_env("LILA_LOG")
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    let (filter, handle) = reload::Layer::new(filter);

    let _ = tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .try_init();

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
