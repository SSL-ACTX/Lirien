use lirien_ir::ir::Function;
use lirien_ir::registry::SerializedSignature;
use seahash::SeaHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use tracing::{debug, info};

#[derive(Serialize, Deserialize)]
pub struct CachedPayload {
    pub functions: Vec<Function>,
    pub dependencies: HashMap<String, SerializedSignature>,
}


fn get_cache_dir() -> PathBuf {
    let mut dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    dir.push(".lirien_cache");
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
    typed_dict_layouts: &HashMap<String, Vec<(String, String)>>,
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

    let mut td_keys: Vec<_> = typed_dict_layouts.keys().collect();
    td_keys.sort();
    for k in td_keys {
        k.hash(&mut hasher);
        for (f_name, f_type) in &typed_dict_layouts[k] {
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
        info!(target: "lirien::cache", "Invalidated stale cache: {:016x}", hash);
    }
}

pub fn load_ir(hash: u64) -> Option<Vec<Function>> {
    let cache_dir = get_cache_dir();
    let file_path = cache_dir.join(format!("{:016x}.lir", hash));

    if file_path.exists() {
        debug!(target: "lirien::cache", "Found cached IR at {:?}", file_path);
        match fs::read(&file_path) {
            Ok(bytes) => {
                match bincode::deserialize::<CachedPayload>(&bytes) {
                    Ok(payload) => {
                        let registry = lirien_ir::registry::GLOBAL_REGISTRY.lock().unwrap();
                        let mut all_valid = true;

                        for (dep_name, cached_sig) in &payload.dependencies {
                            if let Some(current_sig) = registry.get(dep_name) {
                                let current_serialized = SerializedSignature::from(current_sig);
                                if &current_serialized != cached_sig {
                                    info!(
                                        target: "lirien::cache",
                                        "Cache dependency mismatch for '{}' calling '{}'. Invalidating cache.",
                                        payload.functions.first().map(|f| &f.name[..]).unwrap_or("unknown"),
                                        dep_name
                                    );
                                    all_valid = false;
                                    break;
                                }
                            } else {
                                info!(
                                    target: "lirien::cache",
                                    "Cache dependency '{}' not found in registry. Invalidating cache.",
                                    dep_name
                                );
                                all_valid = false;
                                break;
                            }
                        }

                        if all_valid {
                            info!(target: "lirien::cache", "Successfully loaded IR from cache.");
                            return Some(payload.functions);
                        } else {
                            let _ = fs::remove_file(&file_path);
                        }
                    }
                    Err(e) => {
                        debug!(target: "lirien::cache", "Failed to deserialize cached IR: {}", e);
                        // Corrupt cache, might as well delete it
                        let _ = fs::remove_file(file_path);
                    }
                }
            }
            Err(e) => {
                debug!(target: "lirien::cache", "Failed to read cache file: {}", e);
            }
        }
    }

    None
}

pub fn save_ir(hash: u64, funcs: &Vec<Function>) {
    let cache_dir = get_cache_dir();
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        debug!(target: "lirien::cache", "Failed to create cache directory: {}", e);
        return;
    }

    let file_path = cache_dir.join(format!("{:016x}.lir", hash));

    let mut dependencies = HashMap::new();
    {
        let registry = lirien_ir::registry::GLOBAL_REGISTRY.lock().unwrap();
        for func in funcs {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if let lirien_ir::ir::InstructionKind::Call(_, called_func, _) = &inst.kind {
                        if let Some(sig) = registry.get(called_func) {
                            dependencies.insert(called_func.clone(), SerializedSignature::from(sig));
                        }
                    }
                }
            }
        }
    }

    let payload = CachedPayload {
        functions: funcs.clone(),
        dependencies,
    };

    match bincode::serialize(&payload) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&file_path, bytes) {
                debug!(target: "lirien::cache", "Failed to write cache file: {}", e);
            } else {
                info!(target: "lirien::cache", "Successfully saved IR to cache: {:?}", file_path);
            }
        }
        Err(e) => {
            debug!(target: "lirien::cache", "Failed to serialize IR: {}", e);
        }
    }
}
