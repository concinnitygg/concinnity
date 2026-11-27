//! Ownership guard for the explicit backends' resource barriers. The render graph
//! derives every inter-pass frame-path transition and the per-backend graph
//! executor is the one place that emits it; a barrier written anywhere else is
//! either one of the categories below or a hazard the graph no longer reasons
//! about.
//!
//! Only one backend compiles per build, so the call sites are counted as plain
//! text rather than through any backend module: a macOS build still catches a
//! stray `ResourceBarrier` added to the DirectX path, which is otherwise verified
//! only on Windows.
//!
//! The table is exact in both directions. Adding a barrier fails until it is
//! classified here; removing one fails until the count is corrected, so the table
//! cannot drift out of date while still passing.

use concinnity_testing::source::relative_rust_sources;
use std::collections::BTreeMap;
use std::path::Path;

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
    // one call; the refraction snapshots, where a pass copies the attachment
    // it is blending into because a fragment cannot sample it; and the cull's
    // packing, whose scan and scatter read what the dispatch before them wrote.
    // Each opens and closes within one node, so no net state crosses the node
    // boundary.
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
    // no `Inline` rows means every modeled resource is graph-derived.
    Inline,
    // The recording surface's own `cmd_pipeline_barrier` (`record.rs`). Not a
    // barrier decision: it is the one place the call is spelled for every pass
    // that records through `Recorder`, and each of those call sites is counted
    // under `rec.pipeline_barrier(` with its own reason.
    RecordingSurface,
    // A Vulkan rendering scope's moves of its own attachments in and out: the
    // scene targets' round trip (SHADER_READ_ONLY -> COLOR_ATTACHMENT and back)
    // around each pass that draws into them, through the shared scope helpers.
    // Making them graph-derived means the graph choosing each pass's attachment
    // layouts, which is a change to the graph rather than to the executor.
    ScopeLayout,
}

// One backend's barrier surface: every call that emits a barrier, and the exact
// per-(file, call) count with its justification. Paths are relative to the
// backend root.
//
// `calls` is a list because a backend emits frame-path transitions through more
// than one API, and counting only the obvious one exempts the rest from the
// audit: Vulkan has `cmd_pipeline_barrier` and the synchronization2 moves a
// rendering scope makes around its attachments. The table is keyed by (file,
// call) rather than by file, because one file legitimately holds several kinds
// for different reasons -- `record.rs` holds the recording surface's
// `cmd_pipeline_barrier` and the shared scope helper's moves.
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
        calls: &[
            "cmd_pipeline_barrier",
            "rec.pipeline_barrier(",
            ".transition(&",
        ],
        sites: &[
            (
                "graph_exec.rs",
                "rec.pipeline_barrier(",
                4,
                Reason::GraphDriven,
            ),
            ("hiz.rs", "rec.pipeline_barrier(", 1, Reason::IntraPass),
            // One each for the legacy and the synchronization2 barrier.
            (
                "record.rs",
                "cmd_pipeline_barrier",
                2,
                Reason::RecordingSurface,
            ),
            // The shared scope helpers: the main pass's, the G-buffer's, the
            // post seam's, the reactive writers', and the recorder's own
            // draw-over-the-scene scope.
            ("main_pass.rs", ".transition(&", 3, Reason::ScopeLayout),
            (
                "post/gbuffer_scope.rs",
                ".transition(&",
                2,
                Reason::ScopeLayout,
            ),
            (
                "post/post_device.rs",
                ".transition(&",
                2,
                Reason::ScopeLayout,
            ),
            ("reactive_mask.rs", ".transition(&", 4, Reason::ScopeLayout),
            ("record.rs", ".transition(&", 2, Reason::ScopeLayout),
            // The swapchain image, discarded into the composite and handed to
            // the presentation engine after it.
            ("draw/composite.rs", ".transition(&", 2, Reason::Present),
            // The refraction snapshot's copy: the scene opened as the source and
            // the snapshot closed for sampling, inside the Raymarch node.
            ("raymarch/mod.rs", ".transition(&", 2, Reason::IntraPass),
            // The refraction snapshot's copy, the scene opened as its source and
            // the snapshot closed for sampling, and the glass reflection layers,
            // opened for their pre-pass and closed for the scene pass to read.
            // The snapshot and layers are private to the transparent node, and
            // the scene leaves it as it came, through the writer scope.
            ("transparent.rs", ".transition(&", 4, Reason::IntraPass),
            // The reactive mask cleared outside a rendering scope, for a writer that
            // draws nothing on the frame it was to clear or an upscaler that
            // must have a mask on a frame with no writer: from its resting
            // layout to a transfer and back, inside the node.
            (
                "reactive_mask.rs",
                "rec.pipeline_barrier(",
                2,
                Reason::IntraPass,
            ),
            ("texture.rs", "rec.pipeline_barrier(", 1, Reason::Upload),
            // The staged geometry copies' submit: a write-after-read barrier
            // ahead of the copies and a visibility barrier after them.
            (
                "geometry_upload.rs",
                "rec.pipeline_barrier(",
                2,
                Reason::Upload,
            ),
            // The probe cube array's one transition into the layout it keeps,
            // on a one-shot submit when the array is allocated.
            ("probe_set.rs", "rec.pipeline_barrier(", 1, Reason::Upload),
            ("probe.rs", "rec.pipeline_barrier(", 3, Reason::OutOfFrame),
            (
                "probe_prefilter.rs",
                "rec.pipeline_barrier(",
                2,
                Reason::OutOfFrame,
            ),
            // One shared `build_barrier` for every acceleration-structure build
            // sequence, plus the two around the skin dispatch.
            (
                "raytrace.rs",
                "rec.pipeline_barrier(",
                3,
                Reason::OutOfFrame,
            ),
            (
                "screenshot.rs",
                "rec.pipeline_barrier(",
                2,
                Reason::OutOfFrame,
            ),
            (
                "auto_exposure.rs",
                "rec.pipeline_barrier(",
                4,
                Reason::Ungraphed,
            ),
            ("cull.rs", "rec.pipeline_barrier(", 1, Reason::Ungraphed),
            ("compact.rs", "rec.pipeline_barrier(", 3, Reason::IntraPass),
            // A compute pass's output through the copy into this frame's
            // readback and on to the host (the shadow receivers, the view
            // samples), and the receiver reduction's partials between its two
            // dispatches.
            (
                "readback_ring.rs",
                "rec.pipeline_barrier(",
                1,
                Reason::IntraPass,
            ),
            (
                "compact_tests.rs",
                "cmd_pipeline_barrier",
                1,
                Reason::OutOfFrame,
            ),
            // The model-history ring: the snapshot dispatch writes the slot a
            // pre-pass `frames_in_flight - 1` frames ago read, so the write is
            // fenced behind those vertex reads and released to the next frame's.
            // The graph models neither the ring nor the dispatch, which is
            // bundled inside the G-buffer pre-pass node.
            (
                "post/gbuffer.rs",
                "rec.pipeline_barrier(",
                1,
                Reason::Ungraphed,
            ),
            // Two for each plane's cluster lists, binned over its reflected
            // view: one orders the binning after the plane's previous mirror
            // render read them, the other before this render reads them. The
            // per-plane lists are not graph resources.
            ("planar.rs", "rec.pipeline_barrier(", 2, Reason::Ungraphed),
            // The prop probe's faces: each face's cull and packing after the
            // previous face's packing and indirect draw used the shared slot,
            // which is not a graph resource.
            (
                "prop_probe.rs",
                "rec.pipeline_barrier(",
                1,
                Reason::Ungraphed,
            ),
            (
                "post/upscale/mod.rs",
                "rec.pipeline_barrier(",
                1,
                Reason::Inline,
            ),
            // The atmosphere's kernels: each samples what the one before it
            // wrote, the irradiance cube opens for its kernel and closes for the
            // passes that sample it, and last frame's readers finish before the
            // tables are rewritten. A sky redraw opens the sky cube and its
            // capture together, closes the capture once reduced, and closes the
            // cube with the irradiance cube. All inside the one node; the tables
            // rest in GENERAL and the cubes sampled on either side of it.
            (
                "atmosphere.rs",
                "rec.pipeline_barrier(",
                6,
                Reason::IntraPass,
            ),
            // The ocean simulation's kernels: the last frame's water draw
            // before the arrays are rewritten, each kernel's writes before the
            // next one's reads, and the water draw's reads after the last. All
            // inside the one node; the arrays rest in GENERAL on either side.
            (
                "water_sim.rs",
                "rec.pipeline_barrier(",
                3,
                Reason::IntraPass,
            ),
            // The clouds: the noise volume's first build before its readers,
            // the shadow map opened for its kernel and closed for the shaded
            // passes inside the atmosphere's node, and around the march, last
            // frame's composite before the targets are rewritten, the march
            // before the reconstruction reads it, and this frame's composite
            // after. The shadow map rests sampled, the rest GENERAL.
            ("clouds.rs", "rec.pipeline_barrier(", 6, Reason::IntraPass),
            // The RT reflection kernels: last frame's trace before the ray
            // arguments reset, the reset and SSGI's pyramid before the
            // classify, the classify's list before the indirect trace, the
            // trace's sky misses before the cloud march reads them, and the
            // trace before the composite samples its output (or, with no BVH,
            // the output's clear between the composites). The output rests
            // sampled on either side of the node.
            (
                "post/rt_reflections.rs",
                "rec.pipeline_barrier(",
                7,
                Reason::IntraPass,
            ),
        ],
    },
    BackendAudit {
        backend: "directx",
        root: DIRECTX_ROOT,
        calls: &[".ResourceBarrier("],
        sites: &[
            // One site: every graph-derived barrier is staged into `BarrierBatch`
            // and issued from its single flush, so a pass costs one call.
            ("graph_exec.rs", ".ResourceBarrier(", 1, Reason::GraphDriven),
            // The single-pass downsampler takes one barrier, between the
            // dispatch that writes mip 5 and the tail that reduces it.
            ("hiz.rs", ".ResourceBarrier(", 1, Reason::IntraPass),
            // The atmosphere's tables and cubes rest readable by every shader
            // stage; each kernel opens its output for unordered access and
            // closes it again inside the node. A sky redraw opens the capture
            // and the cube together, closes the capture once reduced, and the
            // cube after its GGX mips.
            ("atmosphere.rs", ".ResourceBarrier(", 9, Reason::IntraPass),
            // The ocean's arrays, opened for unordered access at the head of
            // the simulation node, each mip handed back to the resting state
            // once written so the next reduction reads it.
            ("water_sim.rs", ".ResourceBarrier(", 3, Reason::IntraPass),
            // The clouds' noise volume, maps, march and targets rest readable
            // by every shader stage; each kernel opens its output for unordered
            // access and closes it again inside its node (the atmosphere's for
            // the volume and the maps, the march's for the march and this
            // frame's target).
            ("clouds.rs", ".ResourceBarrier(", 7, Reason::IntraPass),
            // The wake's field, opened for unordered access after the ocean's
            // arrays in the same node, each mip handed back the same way.
            ("water_wake.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            // The one-shot zero-init of a new emitter's pool, into COPY_DEST and
            // back to its UNORDERED_ACCESS resting state. The pool's
            // UAV -> NON_PIXEL_SHADER_RESOURCE pair crosses the sim -> draw node
            // boundary, so the graph derives it and its end-of-frame restore.
            ("particle.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            // The MSAA resolve step: the graph rests both HDR targets in
            // RENDER_TARGET, and `ResolveSubresource` needs them in the resolve
            // states for the length of one call.
            ("draw/main.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            ("raymarch/mod.rs", ".ResourceBarrier(", 4, Reason::IntraPass),
            // The refraction snapshot: a fragment cannot sample the attachment
            // it is blending into, so the pass copies the scene into a private
            // target and restores the scene to the state it was handed. The
            // glass reflection layers, private to the node, open for their
            // pre-pass and close for the scene pass to read.
            ("transparent.rs", ".ResourceBarrier(", 4, Reason::IntraPass),
            // The reactive mask on a frame the graph does not carry it, cleared
            // for an upscaler that must have one and returned to its resting
            // state once the dispatch has read it, all inside the Upscale node.
            (
                "reactive_mask.rs",
                ".ResourceBarrier(",
                3,
                Reason::IntraPass,
            ),
            // The shared post seam's bracket for a target private to its pass
            // (`PostTargetState::Pass`): the SSR reflection target, the SSGI
            // gather target, SSAO's raw occlusion, the reflection blur and the
            // bloom octaves below the top, which rest readable and never cross a
            // node boundary.
            (
                "post/post_device.rs",
                ".ResourceBarrier(",
                2,
                Reason::IntraPass,
            ),
            // The model-history ring's snapshot dispatch: a UAV barrier over a
            // buffer, which lives in COMMON and is promoted implicitly at each
            // use, so it claims no state and crosses no node boundary.
            ("post/gbuffer.rs", ".ResourceBarrier(", 1, Reason::IntraPass),
            // The RT reflection kernels' node, on resources the graph does not
            // model because they never cross a node boundary: the output, the
            // ray list and arguments, the sky miss list and its arguments, and
            // SSGI's pyramid opened for the classify, each moved in before the
            // classify, between it and the indirect trace, around the cloud
            // march that reads the misses, and back to rest after.
            (
                "post/rt_reflections.rs",
                ".ResourceBarrier(",
                5,
                Reason::IntraPass,
            ),
            // The bloom chain's top octave. The graph drives `bloom_top` across
            // the node boundary in RENDER_TARGET; the downsample chain samples it
            // between the prefilter and the last upsample, so the node borrows it.
            ("post/bloom.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            ("allocator.rs", ".ResourceBarrier(", 1, Reason::Upload),
            // A new visible-blade buffer, moved to its UNORDERED_ACCESS resting
            // state on a one-shot list ahead of the first frame that binds it.
            ("grass.rs", ".ResourceBarrier(", 1, Reason::Upload),
            // The staged geometry copies' submit brackets the shared buffers in
            // COPY_DEST; its test reads a destination back through the same pair.
            ("geometry_upload.rs", ".ResourceBarrier(", 4, Reason::Upload),
            (
                "resources/geometry.rs",
                ".ResourceBarrier(",
                2,
                Reason::Upload,
            ),
            (
                "resources/skinning.rs",
                ".ResourceBarrier(",
                1,
                Reason::Upload,
            ),
            (
                "resources/streaming.rs",
                ".ResourceBarrier(",
                2,
                Reason::Upload,
            ),
            ("texture.rs", ".ResourceBarrier(", 6, Reason::Upload),
            ("transient_pool.rs", ".ResourceBarrier(", 2, Reason::Upload),
            (
                "resources/geometry_rebuild.rs",
                ".ResourceBarrier(",
                3,
                Reason::OutOfFrame,
            ),
            ("probe.rs", ".ResourceBarrier(", 6, Reason::OutOfFrame),
            (
                "probe_prefilter.rs",
                ".ResourceBarrier(",
                3,
                Reason::OutOfFrame,
            ),
            ("raytrace.rs", ".ResourceBarrier(", 11, Reason::OutOfFrame),
            ("screenshot.rs", ".ResourceBarrier(", 2, Reason::OutOfFrame),
            ("draw/composite.rs", ".ResourceBarrier(", 2, Reason::Present),
            (
                "auto_exposure.rs",
                ".ResourceBarrier(",
                3,
                Reason::Ungraphed,
            ),
            ("cull.rs", ".ResourceBarrier(", 4, Reason::Ungraphed),
            ("compact.rs", ".ResourceBarrier(", 2, Reason::IntraPass),
            // The receiver reduction's partials between its two dispatches.
            (
                "shadow_receivers.rs",
                ".ResourceBarrier(",
                1,
                Reason::IntraPass,
            ),
            // A compute pass's output into and out of the copy to this frame's
            // readback: the shadow receivers' bounds and the view samples.
            (
                "readback_ring.rs",
                ".ResourceBarrier(",
                2,
                Reason::IntraPass,
            ),
            // Four for the mirror targets and their cull's indirect buffer, two
            // flipping the per-plane cluster lists (not graph resources) into
            // and out of `UNORDERED_ACCESS` around their binning.
            ("planar.rs", ".ResourceBarrier(", 6, Reason::Ungraphed),
            // The prop probe's capture into its face targets' state, the shared
            // multisampled color around each face's resolve, and its cube of the
            // probe array back after the convolution. None is a graph resource.
            ("prop_probe.rs", ".ResourceBarrier(", 4, Reason::Ungraphed),
            // The last `Inline` row on this backend. One of its two barriers is
            // really intra-pass (the G-buffer depth, which the graph does not
            // model, borrowed for the upscaler's read); the other transitions the
            // upscaler's output, which *is* the graph's `scene_color` under
            // temporal upscaling. Driving it needs a resting state the graph
            // cannot express: the output alternates UNORDERED_ACCESS and
            // PIXEL_SHADER_RESOURCE depending on whether a previous frame
            // dispatched.
            (
                "post/upscale/mod.rs",
                ".ResourceBarrier(",
                2,
                Reason::Inline,
            ),
        ],
    },
];

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
    let files = relative_rust_sources(Path::new(audit.root));
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
