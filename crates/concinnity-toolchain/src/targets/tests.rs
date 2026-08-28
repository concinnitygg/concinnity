use std::fs;
use std::path::Path;

use tempfile::TempDir;

use super::*;

// Write `contents` to `path`, creating its parent directories.
fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

// The minimum manifest: a package that declares no target of either kind.
const BARE: &str = "[package]\nname = \"pkg\"\n";

#[test]
fn a_lib_only_package_builds_no_final_binary() {
    let dir = TempDir::new().unwrap();
    write(&dir.path().join("src").join("lib.rs"), "");
    assert_eq!(
        binary_targets(BARE, dir.path()),
        vec![BinaryTargets::None],
        "a package with no bin and no example still needs the unscoped setup"
    );
}

#[test]
fn conventional_paths_are_discovered() {
    let dir = TempDir::new().unwrap();
    write(&dir.path().join("src").join("main.rs"), "fn main() {}");
    write(&dir.path().join("examples").join("cube.rs"), "fn main() {}");
    assert_eq!(
        binary_targets(BARE, dir.path()),
        vec![BinaryTargets::Bins, BinaryTargets::Examples]
    );
}

#[test]
fn subdirectories_with_a_main_are_targets_but_bare_ones_are_not() {
    let dir = TempDir::new().unwrap();
    write(
        &dir.path()
            .join("src")
            .join("bin")
            .join("cli")
            .join("main.rs"),
        "fn main() {}",
    );
    write(
        &dir.path().join("examples").join("assets").join("cube.obj"),
        "",
    );
    assert_eq!(binary_targets(BARE, dir.path()), vec![BinaryTargets::Bins]);
}

#[test]
fn a_non_rust_file_is_not_a_target() {
    let dir = TempDir::new().unwrap();
    write(&dir.path().join("examples").join("README.md"), "");
    assert_eq!(binary_targets(BARE, dir.path()), vec![BinaryTargets::None]);
}

#[test]
fn auto_discovery_opt_out_hides_undeclared_sources() {
    let dir = TempDir::new().unwrap();
    write(&dir.path().join("src").join("main.rs"), "fn main() {}");
    write(&dir.path().join("examples").join("cube.rs"), "fn main() {}");
    let manifest = "[package]\nname = \"pkg\"\nautobins = false\nautoexamples = false\n";
    assert_eq!(
        binary_targets(manifest, dir.path()),
        vec![BinaryTargets::None],
        "sources Cargo will not auto-discover are not targets"
    );
}

#[test]
fn an_explicit_table_counts_even_with_auto_discovery_off() {
    let dir = TempDir::new().unwrap();
    let manifest = "[package]\nname = \"pkg\"\nautobins = false\nautoexamples = false\n\n\
                    [[bin]]\nname = \"cli\"\npath = \"src/bin/cli.rs\"\n\n\
                    [[example]]\nname = \"cube\"\npath = \"examples/cube.rs\"\n";
    assert_eq!(
        binary_targets(manifest, dir.path()),
        vec![BinaryTargets::Bins, BinaryTargets::Examples],
        "a declared target counts without its source being on disk"
    );
}

// The published crate is the case that broke `cargo install`: the packaged
// manifest keeps the bin tables Cargo normalized but drops the example table,
// because `include` left `examples/` out of the .crate.
#[test]
fn a_packaged_crate_without_examples_builds_only_bins() {
    let dir = TempDir::new().unwrap();
    write(
        &dir.path().join("src").join("bin").join("cli.rs"),
        "fn main() {}",
    );
    let manifest = "[package]\nname = \"pkg\"\nautobins = false\nautoexamples = false\n\n\
                    [[bin]]\nname = \"cli\"\npath = \"src/bin/cli.rs\"\n";
    assert_eq!(
        binary_targets(manifest, dir.path()),
        vec![BinaryTargets::Bins]
    );
}

#[test]
fn auto_keys_are_read_only_from_the_package_table() {
    let dir = TempDir::new().unwrap();
    write(&dir.path().join("examples").join("cube.rs"), "fn main() {}");
    let manifest = "[workspace.package]\nautoexamples = false\n\n[package]\nname = \"pkg\"\n";
    assert_eq!(
        binary_targets(manifest, dir.path()),
        vec![BinaryTargets::Examples],
        "a key in another table does not opt this package out"
    );
}

#[test]
fn comments_and_spacing_do_not_hide_a_declaration() {
    let dir = TempDir::new().unwrap();
    let manifest = "[package]\nname = \"pkg\"\n  autoexamples   =   false  # off\n\n\
                    \t[[example]]  # the one example\nname = \"cube\"\n";
    assert_eq!(
        binary_targets(manifest, dir.path()),
        vec![BinaryTargets::Examples]
    );
}

#[test]
fn a_key_sharing_a_prefix_is_not_the_auto_key() {
    let dir = TempDir::new().unwrap();
    write(&dir.path().join("src").join("main.rs"), "fn main() {}");
    let manifest = "[package]\nname = \"pkg\"\nautobinsomething = false\n";
    assert_eq!(
        binary_targets(manifest, dir.path()),
        vec![BinaryTargets::Bins]
    );
}

#[test]
fn this_workspace_root_builds_both_kinds() {
    // The real manifest two directories up from this crate: the package that
    // owns the CLI, the player, and the `cube` example.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("read manifest");
    assert_eq!(
        binary_targets(&manifest, root),
        vec![BinaryTargets::Bins, BinaryTargets::Examples]
    );
}

#[test]
fn watched_inputs_skip_paths_that_do_not_exist() {
    let dir = TempDir::new().unwrap();
    write(&dir.path().join("Cargo.toml"), BARE);
    // No `src/bin` and no `examples`: naming either would make Cargo treat the
    // script as always-changed and re-run it every build.
    assert_eq!(
        watched_inputs(dir.path()),
        vec![dir.path().join("Cargo.toml")]
    );

    write(&dir.path().join("examples").join("cube.rs"), "fn main() {}");
    assert_eq!(
        watched_inputs(dir.path()),
        vec![dir.path().join("Cargo.toml"), dir.path().join("examples")]
    );
}

#[test]
fn the_manifest_is_always_watched_so_a_declared_target_is_caught() {
    // A package that declares its targets explicitly (this workspace's root
    // does, with auto-discovery off) is only knowable from the manifest, so
    // the manifest has to be in the watch set even when nothing else is.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    assert!(watched_inputs(root).contains(&root.join("Cargo.toml")));
}
