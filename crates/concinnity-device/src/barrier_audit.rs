// src/barrier_audit.rs
//
// Ownership guard for the explicit backends' resource barriers. The render graph
// derives every inter-pass frame-path transition and the per-backend graph
// executor is the one place that emits it; a barrier written anywhere else is
// either one of the categories below or a hazard the graph no longer reasons
// about.
//
// Only one backend compiles per build, so the call sites are counted as plain
// text rather than through any backend module: a macOS build still catches a
// stray `ResourceBarrier` added to the DirectX path, which is otherwise verified
// only on Windows.
//
// The table is exact in both directions. Adding a barrier fails until it is
// classified here; removing one fails until the count is corrected, so the table
// cannot drift out of date while still passing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const VULKAN_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/vulkan");
const DIRECTX_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/directx");

// Why a barrier is allowed to live outside the graph executor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reason {
    // The executor's own emit path: graph-derived transitions and the aliasing
    // barriers that pooled transients need at a slot's reuse boundary. The only
    // place a frame-path barrier belongs.
    GraphDriven,
    // Finer than the graph's one-state-per-resource granularity, so no derived
    // transition can express it: the Hi-Z reduction's per-mip chain, where one
    // mip is written while the previous is sampled.
    IntraPass,
    // Resource creation, staging upload, and resize transitions. Outside any
    // frame's command stream.
    Upload,
    // GPU work that is not a graph pass: reflection-probe bake, screenshot
    // readback, acceleration-structure builds.
    OutOfFrame,
    // The swapchain image's render-target/present pair, owned by the presentation
    // path rather than by a graph resource.
    Present,
    // A frame-path transition the executor's barrier registry does not drive yet,
    // so its encoder still owns it inline. Each entry disappears as its resource
    // joins the registry; a table with no `Inline` rows means the frame path is
    // fully graph-derived.
    Inline,
}

// One backend's barrier surface: the call that emits a barrier, and the exact
// per-file count with its justification. Paths are relative to the backend root.
struct BackendAudit {
    backend: &'static str,
    root: &'static str,
    call: &'static str,
    sites: &'static [(&'static str, usize, Reason)],
}

const AUDITS: &[BackendAudit] = &[
    BackendAudit {
        backend: "vulkan",
        root: VULKAN_ROOT,
        call: "cmd_pipeline_barrier",
        sites: &[
            ("graph_exec.rs", 3, Reason::GraphDriven),
            ("hiz.rs", 3, Reason::IntraPass),
            ("texture.rs", 1, Reason::Upload),
            ("probe.rs", 3, Reason::OutOfFrame),
            ("raytrace.rs", 6, Reason::OutOfFrame),
            ("screenshot.rs", 2, Reason::OutOfFrame),
            ("auto_exposure.rs", 4, Reason::Inline),
            ("cull.rs", 1, Reason::Inline),
            ("decal.rs", 2, Reason::Inline),
            ("fog.rs", 4, Reason::Inline),
            ("glass.rs", 3, Reason::Inline),
            ("line.rs", 2, Reason::Inline),
            ("main.rs", 1, Reason::Inline),
            ("particle.rs", 2, Reason::Inline),
            ("planar.rs", 1, Reason::Inline),
            ("post/upscale/mod.rs", 2, Reason::Inline),
            ("raymarch.rs", 2, Reason::Inline),
        ],
    },
    BackendAudit {
        backend: "directx",
        root: DIRECTX_ROOT,
        call: ".ResourceBarrier(",
        sites: &[
            ("graph_exec.rs", 4, Reason::GraphDriven),
            ("hiz.rs", 4, Reason::IntraPass),
            ("allocator.rs", 1, Reason::Upload),
            ("resources.rs", 5, Reason::Upload),
            ("texture.rs", 8, Reason::Upload),
            ("transient_pool.rs", 2, Reason::Upload),
            ("geometry_rebuild.rs", 6, Reason::OutOfFrame),
            ("probe.rs", 5, Reason::OutOfFrame),
            ("raytrace.rs", 11, Reason::OutOfFrame),
            ("screenshot.rs", 2, Reason::OutOfFrame),
            ("draw/mod.rs", 1, Reason::Present),
            ("draw/composite.rs", 2, Reason::Present),
            ("auto_exposure.rs", 6, Reason::Inline),
            ("cull.rs", 6, Reason::Inline),
            ("draw/main.rs", 3, Reason::Inline),
            ("fog.rs", 8, Reason::Inline),
            ("glass.rs", 5, Reason::Inline),
            ("particle.rs", 10, Reason::Inline),
            ("planar.rs", 4, Reason::Inline),
            ("post/bloom.rs", 2, Reason::Inline),
            ("post/fullscreen.rs", 2, Reason::Inline),
            ("post/gbuffer.rs", 2, Reason::Inline),
            ("post/rt_reflections.rs", 2, Reason::Inline),
            ("post/ssao.rs", 2, Reason::Inline),
            ("post/upscale/fsr.rs", 2, Reason::Inline),
            ("raymarch.rs", 6, Reason::Inline),
        ],
    },
];

// Every `.rs` file under `dir`, recursively, as a path relative to `dir`.
fn rust_sources(dir: &Path, prefix: &str, out: &mut Vec<(String, PathBuf)>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf-8 file name")
            .to_string();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if path.is_dir() {
            rust_sources(&path, &rel, out);
        } else if name.ends_with(".rs") {
            out.push((rel, path));
        }
    }
}

// Occurrences of `call` in `source`, ignoring line comments so a barrier named in
// prose does not read as a call site.
fn call_sites(source: &str, call: &str) -> usize {
    source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .map(|code| code.matches(call).count())
        .sum()
}

// The counted call sites per file for one backend, omitting files with none.
fn scan(audit: &BackendAudit) -> BTreeMap<String, usize> {
    let mut files = Vec::new();
    rust_sources(Path::new(audit.root), "", &mut files);
    files
        .into_iter()
        .filter_map(|(rel, path)| {
            let source =
                std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
            let n = call_sites(&source, audit.call);
            (n > 0).then_some((rel, n))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_barrier_call_site_is_classified() {
        let mut failures = Vec::new();
        for audit in AUDITS {
            let found = scan(audit);
            let expected: BTreeMap<&str, (usize, Reason)> = audit
                .sites
                .iter()
                .map(|&(path, n, reason)| (path, (n, reason)))
                .collect();

            for (path, &n) in &found {
                match expected.get(path.as_str()) {
                    None => failures.push(format!(
                        "{}/{path}: {n} `{}` call(s) with no entry in the audit table; \
                         classify them (or move them into the graph executor)",
                        audit.backend, audit.call
                    )),
                    Some(&(want, _)) if want != n => failures.push(format!(
                        "{}/{path}: audit table says {want} `{}` call(s), found {n}; \
                         update the table",
                        audit.backend, audit.call
                    )),
                    Some(_) => {}
                }
            }
            for (path, (want, reason)) in &expected {
                if !found.contains_key(*path) {
                    failures.push(format!(
                        "{}/{path}: audit table says {want} `{}` call(s) ({reason:?}), found none; \
                         drop the entry",
                        audit.backend, audit.call
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    #[test]
    fn the_scan_ignores_barriers_named_in_comments() {
        // Negative control for the counter: prose mentioning the call must not read
        // as a call site, or every doc comment would inflate the table.
        assert_eq!(
            call_sites("// feeds cmd_pipeline_barrier\n", "cmd_pipeline_barrier"),
            0
        );
        assert_eq!(
            call_sites("    device.cmd_pipeline_barrier(\n", "cmd_pipeline_barrier"),
            1
        );
        // A trailing comment does not hide a real call on the same line.
        assert_eq!(
            call_sites(
                "    cmd.ResourceBarrier(&[b]); // transition\n",
                ".ResourceBarrier("
            ),
            1
        );
    }

    #[test]
    fn the_scan_reaches_nested_backend_modules() {
        // The table classifies files under `post/` and `draw/`, so a non-recursive
        // walk would silently exempt them.
        let vulkan = scan(&AUDITS[0]);
        assert!(
            vulkan.keys().any(|p| p.contains('/')),
            "expected nested paths in the vulkan scan, got {:?}",
            vulkan.keys().collect::<Vec<_>>()
        );
    }
}
