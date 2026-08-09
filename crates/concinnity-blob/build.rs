// Derives the schema hash baked into every blob header: an FNV-1a hash over
// the postcard-visible schema sources -- the asset schema crate, the divergent
// runtime asset structs, the component registry (list order is the tag), and
// the blob record shapes. Any change to those sources changes the hash, so a
// stale blob fails the load check instead of mis-decoding, with no manually
// maintained version. Over-sensitivity (a comment edit invalidates blobs) is
// deliberate: it can only force a rebuild, never a mis-decode.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest
        .parent()
        .and_then(Path::parent)
        .expect("crate lives at <workspace>/crates/concinnity-blob")
        .to_path_buf();
    let roots = [
        manifest.join("../concinnity-asset/src"),
        manifest.join("../concinnity-core/src/assets"),
        manifest.join("../concinnity-core/src/ecs/registry.rs"),
        manifest.join("src/schema.rs"),
    ];

    let mut files = Vec::new();
    for root in &roots {
        // Directory-level rerun directives catch added and removed files.
        println!("cargo:rerun-if-changed={}", root.display());
        collect(root, &mut files);
    }
    // Workspace-relative names with normalized separators participate in the
    // hash, so a rename or move changes it and platforms agree on it. Sorted
    // so directory walk order never matters.
    let mut named: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|file| {
            let canon = file.canonicalize().unwrap_or_else(|_| file.clone());
            let name = canon
                .strip_prefix(
                    workspace
                        .canonicalize()
                        .unwrap_or_else(|_| workspace.clone()),
                )
                .unwrap_or(&canon)
                .to_string_lossy()
                .replace('\\', "/");
            (name, file)
        })
        .collect();
    named.sort();

    let mut hash: u32 = FNV_OFFSET;
    for (name, file) in &named {
        hash = fnv(hash, name.as_bytes());
        let contents = std::fs::read(file).unwrap_or_default();
        // Carriage returns are stripped so CRLF checkouts hash like LF ones.
        let normalized: Vec<u8> = contents.into_iter().filter(|&b| b != b'\r').collect();
        hash = fnv(hash, &normalized);
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("schema_hash.rs");
    std::fs::write(
        &out,
        format!(
            "/// Hash of the postcard-visible schema sources this build was compiled\n\
             /// against, derived by build.rs and stamped into every blob header. A blob\n\
             /// whose stored hash differs was written by a different engine schema and\n\
             /// fails the load check instead of mis-decoding.\n\
             pub const SCHEMA_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write schema_hash.rs");
}

const FNV_OFFSET: u32 = 0x811c9dc5;
const FNV_PRIME: u32 = 0x0100_0193;

fn fnv(mut hash: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// Every .rs file under `root` (or `root` itself when it is a file).
fn collect(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_file() {
        out.push(root.to_path_buf());
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        panic!("schema source root missing: {}", root.display());
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
