// Which kinds of final binary a package builds, discovered from the package
// rather than declared by its build script.
//
// Cargo scopes a `rustc-link-arg-*` by target kind and rejects a key naming a
// kind the package has none of, so a build script that hardcodes its kinds
// breaks the moment the package is built without one. That is exactly what a
// published crate is: `include` leaves `examples/` out, so the packaged
// manifest carries no example target and `rustc-link-arg-examples` becomes a
// hard error for anyone running `cargo install`.
//
// Nothing here reads the process environment; the manifest text and the
// package directory arrive as arguments so the rules are testable.

use std::path::{Path, PathBuf};

use crate::BinaryTargets;

/// Every kind of final binary the package rooted at `dir` builds, where
/// `manifest` is that package's `Cargo.toml`. Never empty: a package with
/// neither kind still needs the unscoped setup `BinaryTargets::None` stands
/// for.
///
/// Pair this with [`watched_inputs`], or a package that gains a target kind
/// keeps the directives of the kinds it had.
pub(crate) fn binary_targets(manifest: &str, dir: &Path) -> Vec<BinaryTargets> {
    let mut kinds = Vec::new();
    if has_kind(manifest, dir, Kind::Bin) {
        kinds.push(BinaryTargets::Bins);
    }
    if has_kind(manifest, dir, Kind::Example) {
        kinds.push(BinaryTargets::Examples);
    }
    if kinds.is_empty() {
        kinds.push(BinaryTargets::None);
    }
    kinds
}

/// The inputs [`binary_targets`] read, for the caller to emit
/// `rerun-if-changed` over: a build script that declares any `rerun-if-*` opts
/// out of Cargo's rerun-on-any-change, and a manifest edit alone does not
/// re-run it (confirmed against Cargo 1.97).
///
/// Only paths that exist are returned, because Cargo treats a
/// `rerun-if-changed` naming a missing path as always-changed and would re-run
/// the script every build. The manifest always exists, so an added or removed
/// `[[bin]]`/`[[example]]` table is always caught; watching a conventional
/// directory catches a source added to or removed from one Cargo already
/// auto-discovers. What is left uncaught is a package whose *first*
/// `examples/` or `src/bin/` appears with no manifest edit beside it, which
/// takes a `cargo clean` to pick up.
pub(crate) fn watched_inputs(dir: &Path) -> Vec<PathBuf> {
    [
        dir.join("Cargo.toml"),
        dir.join("src").join("bin"),
        dir.join("examples"),
    ]
    .into_iter()
    .filter(|path| path.exists())
    .collect()
}

#[derive(Clone, Copy)]
enum Kind {
    Bin,
    Example,
}

impl Kind {
    // The `[[<table>]]` an explicit target of this kind is declared under.
    fn table(self) -> &'static str {
        match self {
            Kind::Bin => "[[bin]]",
            Kind::Example => "[[example]]",
        }
    }

    // The `[package]` key that turns this kind's auto-discovery off.
    fn auto_key(self) -> &'static str {
        match self {
            Kind::Bin => "autobins",
            Kind::Example => "autoexamples",
        }
    }
}

// Cargo's own rule: an explicitly declared target always counts, and
// auto-discovery adds the conventional paths unless the package opted out.
fn has_kind(manifest: &str, dir: &Path, kind: Kind) -> bool {
    if manifest.lines().any(|line| strip(line) == kind.table()) {
        return true;
    }
    auto_discovery(manifest, kind.auto_key()) && has_conventional_targets(dir, kind)
}

// The paths Cargo auto-discovers a target of this kind from.
fn has_conventional_targets(dir: &Path, kind: Kind) -> bool {
    match kind {
        Kind::Bin => {
            dir.join("src").join("main.rs").is_file() || holds_source(&dir.join("src").join("bin"))
        }
        Kind::Example => holds_source(&dir.join("examples")),
    }
}

// A target directory holds a target when it holds a `.rs` file or a
// subdirectory with a `main.rs`, which is how Cargo reads `src/bin/` and
// `examples/`.
fn holds_source(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        if path.is_dir() {
            return path.join("main.rs").is_file();
        }
        path.extension().is_some_and(|ext| ext == "rs")
    })
}

// A `[package]` boolean, defaulting to true when absent, which is how Cargo
// reads `autobins` and `autoexamples`.
fn auto_discovery(manifest: &str, key: &str) -> bool {
    let mut in_package = false;
    for line in manifest.lines() {
        let line = strip(line);
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(rest) = line.strip_prefix(key)
            && let Some(value) = rest.trim_start().strip_prefix('=')
        {
            return value.trim() != "false";
        }
    }
    true
}

// A manifest line with its trailing comment and surrounding whitespace gone.
fn strip(line: &str) -> &str {
    line.split('#').next().unwrap_or("").trim()
}

#[cfg(test)]
mod tests;
