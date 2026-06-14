use lila_ir::ir::Function;
use seahash::SeaHasher;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use tracing::{debug, info};

fn get_cache_dir() -> PathBuf {
    let mut dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    dir.push(".lila_cache");
    dir
}

pub fn compute_hash(
    source: &str,
    func_name: &str,
    struct_layouts: &HashMap<String, Vec<(String, String)>>,
    enum_layouts: &HashMap<String, Vec<(String, String)>>,
    type_aliases: &HashMap<String, String>,
) -> u64 {
    let mut hasher = SeaHasher::new();

    // Incorporate the compiler version and build hash to automatically invalidate
    // caches on Rust updates or development rebuilds.
    let pkg_version = env!("CARGO_PKG_VERSION");
    pkg_version.hash(&mut hasher);

    if let Ok(build_hash) = env::var("LILA_BUILD_HASH") {
        build_hash.hash(&mut hasher);
    } else {
        // Fallback for cases where build.rs env is not available at runtime
        // but was available at compile time via option_env!
        if let Some(hash) = option_env!("LILA_BUILD_HASH") {
            hash.hash(&mut hasher);
        }
    }

    source.hash(&mut hasher);
    func_name.hash(&mut hasher);

    // Hash maps by sorting keys to ensure deterministic hashing
    let mut s_keys: Vec<_> = struct_layouts.keys().collect();
    s_keys.sort();
    for k in s_keys {
        k.hash(&mut hasher);
        for (f_name, f_type) in &struct_layouts[k] {
            f_name.hash(&mut hasher);
            f_type.hash(&mut hasher);
        }
    }

    let mut e_keys: Vec<_> = enum_layouts.keys().collect();
    e_keys.sort();
    for k in e_keys {
        k.hash(&mut hasher);
        for (v_name, v_type) in &enum_layouts[k] {
            v_name.hash(&mut hasher);
            v_type.hash(&mut hasher);
        }
    }

    let mut t_keys: Vec<_> = type_aliases.keys().collect();
    t_keys.sort();
    for k in t_keys {
        k.hash(&mut hasher);
        type_aliases[k].hash(&mut hasher);
    }

    hasher.finish()
}

pub fn compute_hash_full(
    source: &str,
    func_name: &str,
    struct_layouts: &HashMap<String, Vec<(String, String)>>,
    enum_layouts: &HashMap<String, Vec<(String, String)>>,
    type_aliases: &HashMap<String, String>,
    named_tuple_layouts: &HashMap<String, Vec<(String, String)>>,
) -> u64 {
    let mut hasher = SeaHasher::new();

    let pkg_version = env!("CARGO_PKG_VERSION");
    pkg_version.hash(&mut hasher);

    if let Ok(build_hash) = env::var("LILA_BUILD_HASH") {
        build_hash.hash(&mut hasher);
    } else {
        if let Some(hash) = option_env!("LILA_BUILD_HASH") {
            hash.hash(&mut hasher);
        }
    }

    source.hash(&mut hasher);
    func_name.hash(&mut hasher);

    let mut s_keys: Vec<_> = struct_layouts.keys().collect();
    s_keys.sort();
    for k in s_keys {
        k.hash(&mut hasher);
        for (f_name, f_type) in &struct_layouts[k] {
            f_name.hash(&mut hasher);
            f_type.hash(&mut hasher);
        }
    }

    let mut e_keys: Vec<_> = enum_layouts.keys().collect();
    e_keys.sort();
    for k in e_keys {
        k.hash(&mut hasher);
        for (v_name, v_type) in &enum_layouts[k] {
            v_name.hash(&mut hasher);
            v_type.hash(&mut hasher);
        }
    }

    let mut t_keys: Vec<_> = type_aliases.keys().collect();
    t_keys.sort();
    for k in t_keys {
        k.hash(&mut hasher);
        type_aliases[k].hash(&mut hasher);
    }

    let mut nt_keys: Vec<_> = named_tuple_layouts.keys().collect();
    nt_keys.sort();
    for k in nt_keys {
        k.hash(&mut hasher);
        for (f_name, f_type) in &named_tuple_layouts[k] {
            f_name.hash(&mut hasher);
            f_type.hash(&mut hasher);
        }
    }

    hasher.finish()
}

pub fn invalidate(hash: u64) {
    let cache_dir = get_cache_dir();
    let file_path = cache_dir.join(format!("{:016x}.lir", hash));
    if file_path.exists() {
        let _ = fs::remove_file(file_path);
        info!(target: "lila::cache", "Invalidated stale cache: {:016x}", hash);
    }
}

pub fn load_ir(hash: u64) -> Option<Vec<Function>> {
    let cache_dir = get_cache_dir();
    let file_path = cache_dir.join(format!("{:016x}.lir", hash));

    if file_path.exists() {
        debug!(target: "lila::cache", "Found cached IR at {:?}", file_path);
        match fs::read(&file_path) {
            Ok(bytes) => {
                match bincode::deserialize::<Vec<Function>>(&bytes) {
                    Ok(funcs) => {
                        info!(target: "lila::cache", "Successfully loaded IR from cache.");
                        return Some(funcs);
                    }
                    Err(e) => {
                        debug!(target: "lila::cache", "Failed to deserialize cached IR: {}", e);
                        // Corrupt cache, might as well delete it
                        let _ = fs::remove_file(file_path);
                    }
                }
            }
            Err(e) => {
                debug!(target: "lila::cache", "Failed to read cache file: {}", e);
            }
        }
    }

    None
}

pub fn save_ir(hash: u64, funcs: &Vec<Function>) {
    let cache_dir = get_cache_dir();
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        debug!(target: "lila::cache", "Failed to create cache directory: {}", e);
        return;
    }

    let file_path = cache_dir.join(format!("{:016x}.lir", hash));

    match bincode::serialize(funcs) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&file_path, bytes) {
                debug!(target: "lila::cache", "Failed to write cache file: {}", e);
            } else {
                info!(target: "lila::cache", "Successfully saved IR to cache: {:?}", file_path);
            }
        }
        Err(e) => {
            debug!(target: "lila::cache", "Failed to serialize IR: {}", e);
        }
    }
}
