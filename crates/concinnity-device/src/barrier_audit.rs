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
    // mip is written while the previous is sampled; the MSAA resolve step, which
    // needs its source and destination in the resolve states for the length of
    // one call; and the refraction snapshots, where a pass copies the attachment
    // it is blending into because a fragment cannot sample it. Each opens and
    // closes within one node, so no net state crosses the node boundary.
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
    // A frame-path transition on a resource the graph does not model at all: the
    // per-cascade and per-plane cull's own indirect buffers, the auto-exposure
    // histogram + readback ring, the per-frame-variable planar reflection targets.
    // Deriving one needs the resource declared first, which is a change to the
    // graph rather than to the executor.
    Ungraphed,
    // A frame-path transition on a resource the graph *does* model, whose encoder
    // still owns it because the executor's barrier registry does not resolve it
    // yet. Each entry disappears as its resource joins the registry; a table with
    // no `Inline` rows means every modelled resource is graph-derived.
    Inline,
    // A Vulkan render pass's attachment layout declaration. These are frame-path
    // transitions the driver performs at pass boundaries rather than calls the
    // encoder makes, which is why they need counting at all: the scene targets'
    // whole round trip (SHADER_READ_ONLY -> COLOR_ATTACHMENT and back) is
    // expressed this way and would otherwise be invisible to this audit. Making
    // them graph-derived means the graph choosing each pass's initial and final
    // layouts, not replacing them with barriers -- a render pass folding its own
    // transitions is the efficient form and the one a tiler wants.
    AttachmentLayout,
}

// One backend's barrier surface: every call that emits a barrier, and the exact
// per-(file, call) count with its justification. Paths are relative to the
// backend root.
//
// `calls` is a list because a backend emits frame-path transitions through more
// than one API, and counting only the obvious one exempts the rest from the
// audit: Vulkan has `cmd_pipeline_barrier` and the attachment layout
// declarations a render pass performs at its own boundaries. The table is keyed
// by (file, call) rather than by file, because one file legitimately holds
// several kinds for different reasons -- `glass.rs` has two inline transfer
// barriers and one attachment round trip.
struct BackendAudit {
    backend: &'static str,
    root: &'static str,
    calls: &'static [&'static str],
    sites: &'static [(&'static str, &'static str, usize, Reason)],
}

const AUDITS: &[BackendAudit] = &[
    BackendAudit {
        backend: "vulkan",
        root: VULKAN_ROOT,
        calls: &["cmd_pipeline_barrier", ".final_layout("],
        sites: &[
            (
                "graph_exec.rs",
                "cmd_pipeline_barrier",
                3,
                Reason::GraphDriven,
            ),
            ("hiz.rs", "cmd_pipeline_barrier", 1, Reason::IntraPass),
            // The sim is bundled inside the ParticlesDraw node, so its
            // transfer -> compute -> vertex chain never crosses a node
            // boundary. The third orders the spawn-counter reset against the
            // previous frame's reset and dispatch: one buffer per emitter
            // rather than one per frame in flight, so the same node's previous
            // instance is what it waits on, and the graph models neither the
            // buffer nor a cross-frame edge on it.
            ("particle.rs", "cmd_pipeline_barrier", 3, Reason::IntraPass),
            ("texture.rs", "cmd_pipeline_barrier", 1, Reason::Upload),
            ("probe.rs", "cmd_pipeline_barrier", 3, Reason::OutOfFrame),
            ("raytrace.rs", "cmd_pipeline_barrier", 6, Reason::OutOfFrame),
            (
                "screenshot.rs",
                "cmd_pipeline_barrier",
                2,
                Reason::OutOfFrame,
            ),
            (
                "auto_exposure.rs",
                "cmd_pipeline_barrier",
                4,
                Reason::Ungraphed,
            ),
            ("cull.rs", "cmd_pipeline_barrier", 1, Reason::Ungraphed),
            ("planar.rs", "cmd_pipeline_barrier", 1, Reason::Ungraphed),
            // The refraction snapshot: both passes copy the scene image into a
            // private snapshot and sample that, because a fragment cannot read
            // the attachment it is blending into. The pair opens the copy and
            // closes it, restoring the scene image to the layout the render
            // pass's colour LOAD declares -- so no net state crosses the node
            // boundary and there is no graph edge to derive it from. The
            // snapshot itself is not a graph resource.
            ("glass.rs", "cmd_pipeline_barrier", 2, Reason::IntraPass),
            ("raymarch.rs", "cmd_pipeline_barrier", 2, Reason::IntraPass),
            ("main.rs", "cmd_pipeline_barrier", 1, Reason::Inline),
            (
                "post/upscale/mod.rs",
                "cmd_pipeline_barrier",
                2,
                Reason::Inline,
            ),
            (
                "render_pass.rs",
                ".final_layout(",
                9,
                Reason::AttachmentLayout,
            ),
            ("decal.rs", ".final_layout(", 1, Reason::AttachmentLayout),
            ("fog.rs", ".final_layout(", 1, Reason::AttachmentLayout),
            ("glass.rs", ".final_layout(", 1, Reason::AttachmentLayout),
            ("line.rs", ".final_layout(", 1, Reason::AttachmentLayout),
            ("particle.rs", ".final_layout(", 1, Reason::AttachmentLayout),
            ("raymarch.rs", ".final_layout(", 2, Reason::AttachmentLayout),
            (
                "post/gbuffer.rs",
                ".final_layout(",
                4,
                Reason::AttachmentLayout,
            ),
            (
                "post/reflection_composite.rs",
                ".final_layout(",
                1,
                Reason::AttachmentLayout,
            ),
            (
                "post/rt_reflections.rs",
                ".final_layout(",
                1,
                Reason::AttachmentLayout,
            ),
            (
                "post/ssao.rs",
                ".final_layout(",
                2,
                Reason::AttachmentLayout,
            ),
            (
                "post/ssgi.rs",
                ".final_layout(",
                2,
                Reason::AttachmentLayout,
            ),
            ("post/ssr.rs", ".final_layout(", 1, Reason::AttachmentLayout),
            ("post/taa.rs", ".final_layout(", 1, Reason::AttachmentLayout),
        ],
    },
    BackendAudit {
        backend: "directx",
        root: DIRECTX_ROOT,
        calls: &[".ResourceBarrier("],
        sites: &[
            ("graph_exec.rs", ".ResourceBarrier(", 5, Reason::GraphDriven),
            ("hiz.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            ("particle.rs", ".ResourceBarrier(", 6, Reason::IntraPass),
            // The MSAA resolve step: the graph rests both HDR targets in
            // RENDER_TARGET, and `ResolveSubresource` needs them in the resolve
            // states for the length of one call.
            ("draw/main.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            ("raymarch.rs", ".ResourceBarrier(", 4, Reason::IntraPass),
            // The refraction snapshot: a fragment cannot sample the attachment
            // it is blending into, so the pass copies the scene into a private
            // target and restores the scene to the state it was handed.
            ("glass.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            // The SSGI gather samples the scene the composite then blends into,
            // so this node reads and writes one resource; the graph models that
            // as a single write and the gather borrows the read state.
            ("post/ssgi.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            // The same shape in two more bundled nodes, on targets the graph does
            // not model because they never cross a node boundary: SSAO's raw
            // occlusion, which its own blur consumes, and the RT reflection
            // radiance the roughness composite consumes.
            ("post/ssao.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            (
                "post/rt_reflections.rs",
                ".ResourceBarrier(",
                2,
                Reason::IntraPass,
            ),
            // The bloom octave chain, which writes mip N while sampling mip N+1.
            // The graph drives mip 0 (`bloom_top`) across the node boundary;
            // these order the steps within it.
            ("post/bloom.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            // The shared fullscreen bracket, now used only by targets that live
            // inside one node: the SSR / RT resolve output, the reflection blur,
            // and the SSGI gather. Callers whose target the graph drives take
            // `bind_fullscreen_rt` instead.
            (
                "post/fullscreen.rs",
                ".ResourceBarrier(",
                2,
                Reason::IntraPass,
            ),
            ("allocator.rs", ".ResourceBarrier(", 1, Reason::Upload),
            ("resources.rs", ".ResourceBarrier(", 5, Reason::Upload),
            ("texture.rs", ".ResourceBarrier(", 6, Reason::Upload),
            ("transient_pool.rs", ".ResourceBarrier(", 2, Reason::Upload),
            (
                "geometry_rebuild.rs",
                ".ResourceBarrier(",
                6,
                Reason::OutOfFrame,
            ),
            ("probe.rs", ".ResourceBarrier(", 5, Reason::OutOfFrame),
            ("raytrace.rs", ".ResourceBarrier(", 11, Reason::OutOfFrame),
            ("screenshot.rs", ".ResourceBarrier(", 2, Reason::OutOfFrame),
            ("draw/composite.rs", ".ResourceBarrier(", 2, Reason::Present),
            (
                "auto_exposure.rs",
                ".ResourceBarrier(",
                6,
                Reason::Ungraphed,
            ),
            ("cull.rs", ".ResourceBarrier(", 6, Reason::Ungraphed),
            ("planar.rs", ".ResourceBarrier(", 4, Reason::Ungraphed),
            // The last `Inline` row on this backend. One of its two barriers is
            // really intra-pass (the G-buffer depth, which the graph does not
            // model, borrowed for FSR's read); the other transitions the
            // upscaler's output, which *is* the graph's `scene_color` under
            // temporal upscaling. Driving it needs a resting state the graph
            // cannot express: the output alternates UNORDERED_ACCESS and
            // PIXEL_SHADER_RESOURCE depending on whether a previous frame
            // dispatched.
            (
                "post/upscale/fsr.rs",
                ".ResourceBarrier(",
                2,
                Reason::Inline,
            ),
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

// The counted sites per (file, call) for one backend, omitting empty pairs.
fn scan(audit: &BackendAudit) -> BTreeMap<(String, &'static str), usize> {
    let mut files = Vec::new();
    rust_sources(Path::new(audit.root), "", &mut files);
    let mut found = BTreeMap::new();
    for (rel, path) in files {
        let source = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        for &call in audit.calls {
            let n = call_sites(&source, call);
            if n > 0 {
                found.insert((rel.clone(), call), n);
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_barrier_call_site_is_classified() {
        let mut failures = Vec::new();
        for audit in AUDITS {
            let found = scan(audit);
            let expected: BTreeMap<(String, &str), (usize, Reason)> = audit
                .sites
                .iter()
                .map(|&(path, call, n, reason)| ((path.to_string(), call), (n, reason)))
                .collect();

            for ((path, call), &n) in &found {
                match expected.get(&(path.clone(), *call)) {
                    None => failures.push(format!(
                        "{}/{path}: {n} `{call}` site(s) with no entry in the audit table; \
                         classify them (or move them into the graph executor)",
                        audit.backend
                    )),
                    Some(&(want, _)) if want != n => failures.push(format!(
                        "{}/{path}: audit table says {want} `{call}` site(s), found {n}; \
                         update the table",
                        audit.backend
                    )),
                    Some(_) => {}
                }
            }
            for ((path, call), (want, reason)) in &expected {
                if !found.contains_key(&(path.clone(), *call)) {
                    failures.push(format!(
                        "{}/{path}: audit table says {want} `{call}` site(s) ({reason:?}), \
                         found none; drop the entry",
                        audit.backend
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
            vulkan.keys().any(|(p, _)| p.contains('/')),
            "expected nested paths in the vulkan scan, got {:?}",
            vulkan.keys().collect::<Vec<_>>()
        );
    }
}
