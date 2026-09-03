//! Derives the hash of this crate's sources, which the backend shader caches
//! fold into every artifact key: a compile is a function of the assembled
//! source text and of the flags this crate builds, so a change here must miss
//! rather than replay stale bytes.
//!
//! Self-contained rather than calling `concinnity_toolchain::hash_sources`,
//! because that crate depends on this one and a build dependency back would
//! cycle. Consumers fold in the constant instead of reaching across the
//! workspace for these files, which a registry checkout would not have.
//!
//! Also locates the workspace root, under whose `vendor/` directory `vendor.rs`
//! looks for the pinned Slang release.

use std::path::{Path, PathBuf};

const FNV_OFFSET: u32 = 0x811c_9dc5;
const FNV_PRIME: u32 = 0x0100_0193;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    emit_workspace_root(&manifest, &out_dir);

    let src = manifest.join("src");
    println!("cargo:rerun-if-changed={}", src.display());

    let mut files = Vec::new();
    collect(&src, &mut files);
    files.sort();

    let mut hash = FNV_OFFSET;
    for file in &files {
        let name = file
            .strip_prefix(&src)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        hash = fnv(hash, name.as_bytes());
        let bytes = std::fs::read(file).unwrap_or_default();
        // Carriage returns are stripped so CRLF checkouts hash like LF ones.
        let normalized: Vec<u8> = bytes.into_iter().filter(|&b| b != b'\r').collect();
        hash = fnv(hash, &normalized);
    }

    let out = out_dir.join("source_hash.rs");
    std::fs::write(
        &out,
        format!(
            "/// Hash of this crate's sources, folded into every shader-artifact\n\
             /// cache key so a compiler change misses instead of replaying bytes\n\
             /// an older invocation produced.\n\
             pub const SOURCE_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write source_hash.rs");
}

// The workspace root, two levels above this manifest, under which `vendor/`
// holds the pinned Slang release. A registry copy has neither that layout nor a
// manifest there, so it emits `None` and resolution falls through to PATH.
//
// `vendor/` is watched only once it exists, because an absent rerun path reruns
// this script on every build. A `vendor/` created later is picked up when
// something touches this file, which is whatever put the release there.
fn emit_workspace_root(manifest: &Path, out_dir: &Path) {
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .filter(|root| root.join("Cargo.toml").is_file());

    let value = match root {
        Some(path) => {
            let vendor = path.join("vendor");
            if vendor.is_dir() {
                println!("cargo:rerun-if-changed={}", vendor.display());
            }
            format!("Some({:?})", path.to_string_lossy())
        }
        None => "None".to_string(),
    };
    std::fs::write(
        out_dir.join("workspace_root.rs"),
        format!("const WORKSPACE_ROOT: Option<&str> = {value};\n"),
    )
    .expect("write workspace_root.rs");
}

fn fnv(mut hash: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
