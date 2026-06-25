//! In-process and disk cache for compiled JIT functions.
//!
//! Exposes APIs to lookup and store verification bounds, IR layouts, and compiled pointers
//! to avoid re-verifying and re-compiling functions that haven't changed.

use lirien_ir::ir::{Function, Type};
use lirien_ir::registry::SerializedSignature;
use seahash::SeaHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tracing::{debug, info};

/// Structure representing serialized JIT cache payload stored on disk.
#[derive(Serialize, Deserialize)]
pub struct CachedPayload {
    /// List of IR functions within the cached compilation group.
    pub functions: Vec<Function>,
    /// Signatures of external dependencies checked during cache verification.
    pub dependencies: HashMap<String, SerializedSignature>,
}


/// An in-process cached entry for a fully compiled function.
/// Keyed by the IR hash (same hash used for the disk cache).
#[derive(Clone, Debug)]
pub struct NativeCacheEntry {
    pub name: String,
    pub pointer: usize,
    pub arg_types: Vec<Type>,
    pub arg_refinements: HashMap<usize, String>,
    pub return_type: Type,
    pub return_refinement: Option<String>,
}

#[derive(Clone, Debug)]
pub struct NativeCachePayload {
    pub entries: Vec<NativeCacheEntry>,
    pub dependencies: HashMap<String, SerializedSignature>,
}

static NATIVE_CODE_CACHE: OnceLock<Mutex<HashMap<u64, NativeCachePayload>>> = OnceLock::new();

fn get_native_cache() -> &'static Mutex<HashMap<u64, NativeCachePayload>> {
    NATIVE_CODE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Check the L1 in-process cache for a previously compiled function group.
pub fn native_cache_lookup(hash: u64) -> Option<Vec<NativeCacheEntry>> {
    let cache = get_native_cache().lock().unwrap();
    if let Some(payload) = cache.get(&hash) {
        let registry = lirien_ir::registry::GLOBAL_REGISTRY.lock().unwrap();
        for (dep_name, cached_sig) in &payload.dependencies {
            if let Some(current_sig) = registry.get(dep_name) {
                let current_serialized = SerializedSignature::from(current_sig);
                if &current_serialized != cached_sig {
                    debug!(
                        target: "lirien::cache",
                        "L1 cache dependency mismatch for '{}'. Invalidating L1 cache.",
                        dep_name
                    );
                    return None;
                }
            } else {
                debug!(
                    target: "lirien::cache",
                    "L1 cache dependency '{}' not found in registry. Invalidating L1 cache.",
                    dep_name
                );
                return None;
            }
        }
        Some(payload.entries.clone())
    } else {
        None
    }
}

/// Store a compiled function group in the L1 in-process cache.
pub fn native_cache_store(
    hash: u64,
    entries: Vec<NativeCacheEntry>,
    dependencies: HashMap<String, SerializedSignature>,
) {
    let mut cache = get_native_cache().lock().unwrap();
    cache.insert(
        hash,
        NativeCachePayload {
            entries,
            dependencies,
        },
    );
}

/// Invalidate an entry from the native code cache.
pub fn native_cache_invalidate(hash: u64) {
    let mut cache = get_native_cache().lock().unwrap();
    cache.remove(&hash);
}



fn get_cache_dir() -> PathBuf {
    if let Ok(val) = env::var("LIRIEN_CACHE_DIR") {
        PathBuf::from(val)
    } else {
        let mut dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        dir.push(".lirien_cache");
        dir
    }
}

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

#[cfg(unix)]
fn lock_shared(file: &fs::File) -> std::io::Result<()> {
    let fd = file.as_raw_fd();
    let res = unsafe { libc::flock(fd, libc::LOCK_SH) };
    if res == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn lock_exclusive(file: &fs::File) -> std::io::Result<()> {
    let fd = file.as_raw_fd();
    let res = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if res == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn unlock(file: &fs::File) -> std::io::Result<()> {
    let fd = file.as_raw_fd();
    let res = unsafe { libc::flock(fd, libc::LOCK_UN) };
    if res == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn lock_shared(_file: &fs::File) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn lock_exclusive(_file: &fs::File) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock(_file: &fs::File) -> std::io::Result<()> {
    Ok(())
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
        let mut file = match fs::File::open(&file_path) {
            Ok(f) => f,
            Err(e) => {
                debug!(target: "lirien::cache", "Failed to open cache file: {}", e);
                return None;
            }
        };

        if let Err(e) = lock_shared(&file) {
            debug!(target: "lirien::cache", "Failed to acquire shared lock: {}", e);
            return None;
        }

        use std::io::Read;
        let mut bytes = Vec::new();
        let read_res = file.read_to_end(&mut bytes);
        let _ = unlock(&file);

        if read_res.is_err() {
            debug!(target: "lirien::cache", "Failed to read cache file: {:?}", read_res.err());
            return None;
        }

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

    None
}

pub fn collect_dependencies(funcs: &[Function]) -> HashMap<String, SerializedSignature> {
    let mut dependencies = HashMap::new();
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
    dependencies
}

pub fn save_ir(hash: u64, funcs: &[Function]) {
    let cache_dir = get_cache_dir();
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        debug!(target: "lirien::cache", "Failed to create cache directory: {}", e);
        return;
    }

    let file_path = cache_dir.join(format!("{:016x}.lir", hash));

    let dependencies = collect_dependencies(funcs);

    let payload = CachedPayload {
        functions: funcs.to_owned(),
        dependencies,
    };

    match bincode::serialize(&payload) {
        Ok(bytes) => {
            let open_res = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&file_path);

            match open_res {
                Ok(mut file) => {
                    if let Err(e) = lock_exclusive(&file) {
                        debug!(target: "lirien::cache", "Failed to acquire exclusive lock: {}", e);
                        return;
                    }

                    use std::io::Write;
                    let write_res = file.write_all(&bytes);
                    let _ = unlock(&file);

                    if let Err(e) = write_res {
                        debug!(target: "lirien::cache", "Failed to write cache file: {}", e);
                    } else {
                        info!(target: "lirien::cache", "Successfully saved IR to cache: {:?}", file_path);
                        // Run eviction after a successful save
                        evict_lru();
                    }
                }
                Err(e) => {
                    debug!(target: "lirien::cache", "Failed to open/create cache file: {}", e);
                }
            }
        }
        Err(e) => {
            debug!(target: "lirien::cache", "Failed to serialize IR: {}", e);
        }
    }
}

const DEFAULT_MAX_CACHE_BYTES: u64 = 50 * 1024 * 1024; // 50 MB

fn get_max_cache_bytes() -> u64 {
    env::var("LIRIEN_CACHE_MAX_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(DEFAULT_MAX_CACHE_BYTES)
}

pub fn evict_lru() {
    let cache_dir = get_cache_dir();
    if !cache_dir.exists() {
        return;
    }

    let max_bytes = get_max_cache_bytes();

    let entries: Vec<_> = match fs::read_dir(&cache_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "lir")
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => return,
    };

    let mut file_infos: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total_size: u64 = 0;

    for entry in &entries {
        if let Ok(meta) = entry.metadata() {
            let size = meta.len();
            let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            total_size += size;
            file_infos.push((entry.path(), size, mtime));
        }
    }

    if total_size <= max_bytes {
        return;
    }

    // Sort by modification time ascending (oldest first)
    file_infos.sort_by_key(|(_, _, mtime)| *mtime);

    let mut evicted = 0u64;
    let overshoot = total_size - max_bytes;

    for (path, size, _) in &file_infos {
        if evicted >= overshoot {
            break;
        }
        if fs::remove_file(path).is_ok() {
            debug!(target: "lirien::cache", "Evicted cache file: {:?} ({} bytes)", path, size);
            evicted += size;
        }
    }

    info!(
        target: "lirien::cache",
        "Cache eviction complete: freed {} bytes (limit: {} bytes)",
        evicted, max_bytes
    );
}
