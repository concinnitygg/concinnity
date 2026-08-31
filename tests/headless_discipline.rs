//! No test in this workspace may stand up a window.
//!
//! Under a backend feature -- and `native` is on by default, so an ordinary
//! `cargo test` has one -- a world started on the windowed loop takes a GPU and
//! blocks on an event loop the harness cannot end. That is a hang with no
//! failing assertion to read, on whichever host happens to run it.
//!
//! Two things stop that. `concinnity_testing::forbid_windows` arms a runtime
//! tripwire, so a test that opts in gets a panic naming the backend. This scan
//! is the half that needs no opt-in: it reads the workspace's own test code and
//! fails on the shapes that reach a window, before one is ever built.
//!
//! The rule is deliberately narrow. It names the two calls that stand a window
//! up and the marker that proves the caller chose the other loop, so a test
//! phrased any other way is not a false positive.

use std::path::{Path, PathBuf};

use concinnity_testing::source;

// This file states the forbidden shape as test data, so scanning it would
// report itself.
const SELF: &str = "headless_discipline.rs";

// This package is the workspace root, so its manifest directory is the tree
// every crate lives under.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn rust_sources() -> Vec<PathBuf> {
    let root = workspace_root();
    source::rust_sources(
        &[root.join("src"), root.join("tests"), root.join("crates")],
        SELF,
    )
}

// Starting a world against the full engine table builds the render stack when
// the world declares a `GraphicsConfig`, and building it opens a window. The
// manifest reader answers the same questions without one.
const WINDOWED_START: &[&str] = &[".start(SYSTEMS)", ".start(crate::ecs::SYSTEMS)"];

// What proves the test chose the loop with no renderer behind it.
const HEADLESS: &[&str] = &[
    "into_headless",
    "HEADLESS_SYSTEMS",
    "assert_starts_headless",
    "system_manifest",
    "forbid_windows",
    "without_windows",
];

#[test]
fn no_test_starts_a_graphics_world_on_the_windowed_loop() {
    let root = workspace_root();
    let mut offenders = Vec::new();

    for path in rust_sources() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        for body in source::test_bodies(&text) {
            let declares_graphics = body.text.contains("GraphicsConfig");
            let starts_windowed = WINDOWED_START.iter().any(|call| body.text.contains(call));
            let chose_headless = HEADLESS.iter().any(|marker| body.text.contains(marker));

            if declares_graphics && starts_windowed && !chose_headless {
                let rel = path.strip_prefix(&root).unwrap_or(&path);
                offenders.push(format!("{}:{}", rel.display(), body.line));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these tests start a GraphicsConfig world on the windowed loop, which \
         opens a window and never returns. Read the manifest with \
         `system_manifest`, or run the world with `into_headless`:\n  {}",
        offenders.join("\n  ")
    );
}

// The scan is only worth as much as its ability to see the shape it forbids.
#[test]
fn the_scan_recognises_the_shape_it_forbids() {
    let offending = r#"
#[test]
fn opens_a_window() {
    let mut world = World::new();
    world.add_component(GraphicsConfig::default());
    world.start(SYSTEMS).unwrap();
}
"#;
    let bodies = source::test_bodies(offending);
    assert_eq!(bodies.len(), 1, "one test found");
    let body = &bodies[0].text;
    assert!(body.contains("GraphicsConfig"));
    assert!(WINDOWED_START.iter().any(|c| body.contains(c)));
    assert!(!HEADLESS.iter().any(|m| body.contains(m)));

    let allowed = offending.replace("world.start(SYSTEMS)", "world.system_manifest(SYSTEMS)");
    let body = &source::test_bodies(&allowed)[0].text;
    assert!(
        HEADLESS.iter().any(|m| body.contains(m)),
        "the manifest form reads as headless"
    );
}

// A commented-out call is not a call.
#[test]
fn the_scan_ignores_commented_lines() {
    let commented = r#"
#[test]
fn only_talks_about_it() {
    // world.add_component(GraphicsConfig::default());
    // world.start(SYSTEMS).unwrap();
    assert!(true);
}
"#;
    let body = &source::test_bodies(commented)[0].text;
    assert!(!body.contains("GraphicsConfig"), "comments are stripped");
}

// The walk has to actually reach the crates, or the guard passes by looking at
// nothing.
#[test]
fn the_scan_reads_the_whole_workspace() {
    let files = rust_sources();
    assert!(
        files.len() > 300,
        "only {} sources found; the walk is not reaching the crates",
        files.len()
    );
    assert!(
        files.iter().any(|p| p.ends_with("src/driver.rs")),
        "the facade's own driver is in the scan"
    );
    assert!(
        !files.iter().any(|p| p.ends_with(SELF)),
        "the guard does not scan itself"
    );
    assert!(
        files
            .iter()
            .any(|p| p.to_string_lossy().contains("concinnity-engine")),
        "the engine crate is in the scan"
    );
}
