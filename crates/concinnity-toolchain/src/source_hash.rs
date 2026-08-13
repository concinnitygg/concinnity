// src/source_hash.rs
//
// FNV-1a over a set of Rust sources, for a cache or container whose stored
// bytes are a function of the code that produced them. Folding the hash into a
// key means a change to any of those sources changes the key, so a stale entry
// misses instead of being replayed, with no manually maintained version number.
// Over-sensitivity (a comment edit invalidates entries) is deliberate: it can
// only force a recompile, never a stale replay.
//
// Pure walk + hash; the environment reading and the `cargo::` directives live
// in lib.rs.

use std::path::{Path, PathBuf};

const FNV_OFFSET: u32 = 0x811c9dc5;
const FNV_PRIME: u32 = 0x0100_0193;

pub(crate) fn fnv(mut hash: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// Every .rs file under `root` (or `root` itself when it is a file). Panics on a
// missing root: a typo would otherwise hash nothing and leave the key blind to
// the sources it is meant to track.
pub(crate) fn collect(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_file() {
        out.push(root.to_path_buf());
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        panic!("source hash root missing: {}", root.display());
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

// Workspace-relative name with normalized separators, so a rename or a move
// changes the hash and platforms agree on it. `workspace` must already be
// canonicalized.
pub(crate) fn relative_name(workspace: &Path, file: &Path) -> String {
    let canon = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    canon
        .strip_prefix(workspace)
        .unwrap_or(&canon)
        .to_string_lossy()
        .replace('\\', "/")
}

// Hash (name, path) pairs, names participating so a rename registers. Sorted
// first so directory walk order never matters. An unreadable file hashes as
// empty rather than failing the build.
pub(crate) fn hash_named(named: &mut [(String, PathBuf)]) -> u32 {
    named.sort();
    let mut hash = FNV_OFFSET;
    for (name, file) in named.iter() {
        hash = fnv(hash, name.as_bytes());
        let contents = std::fs::read(file).unwrap_or_default();
        // Carriage returns are stripped so CRLF checkouts hash like LF ones.
        let normalized: Vec<u8> = contents.into_iter().filter(|&b| b != b'\r').collect();
        hash = fnv(hash, &normalized);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(dir: &Path, files: &[&str]) -> Vec<(String, PathBuf)> {
        files.iter().map(|f| (f.to_string(), dir.join(f))).collect()
    }

    #[test]
    fn an_edit_changes_the_hash() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), b"fn a() {}").unwrap();
        let before = hash_named(&mut named(dir.path(), &["a.rs"]));
        std::fs::write(dir.path().join("a.rs"), b"fn a() { b() }").unwrap();
        assert_ne!(before, hash_named(&mut named(dir.path(), &["a.rs"])));
    }

    // Names participate, so moving a file to a new path invalidates entries
    // even when its bytes are untouched.
    #[test]
    fn a_rename_changes_the_hash() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), b"fn a() {}").unwrap();
        std::fs::write(dir.path().join("b.rs"), b"fn a() {}").unwrap();
        assert_ne!(
            hash_named(&mut named(dir.path(), &["a.rs"])),
            hash_named(&mut named(dir.path(), &["b.rs"])),
        );
    }

    #[test]
    fn walk_order_does_not_matter() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), b"one").unwrap();
        std::fs::write(dir.path().join("b.rs"), b"two").unwrap();
        assert_eq!(
            hash_named(&mut named(dir.path(), &["a.rs", "b.rs"])),
            hash_named(&mut named(dir.path(), &["b.rs", "a.rs"])),
        );
    }

    // A CRLF checkout must produce the same hash as an LF one, or a Windows
    // developer's cache misses on every entry a macOS build wrote.
    #[test]
    fn line_endings_do_not_affect_the_hash() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lf.rs"), b"fn a() {}\nfn b() {}\n").unwrap();
        std::fs::write(dir.path().join("lf.rs.crlf"), b"fn a() {}\r\nfn b() {}\r\n").unwrap();
        let lf = hash_named(&mut [("x".to_string(), dir.path().join("lf.rs"))]);
        let crlf = hash_named(&mut [("x".to_string(), dir.path().join("lf.rs.crlf"))]);
        assert_eq!(lf, crlf);
    }

    #[test]
    fn a_missing_file_hashes_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.rs"), b"").unwrap();
        assert_eq!(
            hash_named(&mut [("x".to_string(), dir.path().join("empty.rs"))]),
            hash_named(&mut [("x".to_string(), dir.path().join("gone.rs"))]),
        );
    }

    #[test]
    fn collect_walks_nested_directories_for_rs_files_only() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("inner");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join("top.rs"), b"").unwrap();
        std::fs::write(dir.path().join("notes.md"), b"").unwrap();
        std::fs::write(nested.join("deep.rs"), b"").unwrap();

        let mut out = Vec::new();
        collect(dir.path(), &mut out);
        let mut names: Vec<String> = out
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["deep.rs", "top.rs"]);
    }

    #[test]
    fn collect_accepts_a_single_file_root() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("one.rs");
        std::fs::write(&file, b"").unwrap();
        let mut out = Vec::new();
        collect(&file, &mut out);
        assert_eq!(out, [file]);
    }

    #[test]
    #[should_panic(expected = "source hash root missing")]
    fn collect_panics_on_a_missing_root() {
        let mut out = Vec::new();
        collect(Path::new("/no/such/source/root"), &mut out);
    }

    #[test]
    fn names_are_workspace_relative_with_forward_slashes() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().canonicalize().unwrap();
        let nested = workspace.join("crates").join("thing").join("src");
        std::fs::create_dir_all(&nested).unwrap();
        let file = nested.join("lib.rs");
        std::fs::write(&file, b"").unwrap();
        assert_eq!(
            relative_name(&workspace, &file),
            "crates/thing/src/lib.rs".to_string()
        );
    }
}
