use std::process::Command;

fn main() {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok());

    let git_hash = output
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=LILA_BUILD_HASH={}", git_hash);

    // Also re-run if any of the source files change
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../lila-ir/src");
    println!("cargo:rerun-if-changed=../lila-verify/src");
    println!("cargo:rerun-if-changed=../lila-backend/src");
}
