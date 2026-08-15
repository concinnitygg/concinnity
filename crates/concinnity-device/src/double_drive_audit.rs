// src/double_drive_audit.rs
//
// Guard against a resource being transitioned twice: once by the graph executor's
// barrier registry and again by an encoder that kept its inline half.
//
// This is the one failure the barrier audit cannot see. That table proves every
// barrier is *classified*; it says nothing about whether a classified barrier is
// still *needed*. When a resource joins the registry, deleting its inline
// transitions is a separate manual step, and a table left describing barriers
// that are now redundant is perfectly self-consistent. What that leaves is a
// frame announcing a before-state the resource has already left, once per
// barrier per frame: a debug-layer error on every frame, and only on the backend
// whose debug layer runs.
//
// The check is textual, for the same reason as `barrier_audit`: only one backend
// compiles per build, so a macOS run has to audit the DirectX source as source if
// it is to audit it at all. It works by naming, per backend, the context field
// that backs each registry-resolved resource, then finding every barrier whose
// *target expression* mentions one of those fields outside the executor. Anything
// it finds is either a bug of the shape above or belongs in `ALLOWED` with a
// reason.
//
// Its reach ends where the target stops being named at the barrier: a pass that
// picks between two resources into a local and transitions that local (the
// translucent pass, whose target is the reflection composite's output or the HDR
// spine depending on whether a resolve ran) is invisible here. `barrier_audit`
// still counts and classifies those files, which is what keeps a new barrier in
// one of them from passing unnoticed.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const VULKAN_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/vulkan");
const DIRECTX_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/directx");

// The executor file itself: the one place a registry-resolved resource is
// *supposed* to be transitioned.
const EXECUTOR: &str = "graph_exec.rs";

// One backend's registry surface. `fields` are the context field paths the
// registry resolver maps graph labels onto; `targets` are the regex-free markers
// that introduce a barrier's target expression, paired with how far past the
// marker the target text runs.
struct BackendRegistry {
    backend: &'static str,
    root: &'static str,
    // (graph label, the text a barrier on that resource contains). The token is
    // how an *encoder* names the target, which is not always how the resolver
    // does: an encoder usually binds a local first (`hiz.pyramid`) where the
    // resolver walks from `self` (`self.cull.hiz`). It has to be distinctive
    // enough not to match unrelated code -- a bare field name like `resource`
    // matches half the backend.
    fields: &'static [(&'static str, &'static str)],
    // Text that introduces a barrier target. Everything from the marker to the
    // end of the statement is searched for a registry field.
    markers: &'static [&'static str],
}

const REGISTRIES: &[BackendRegistry] = &[
    BackendRegistry {
        backend: "vulkan",
        root: VULKAN_ROOT,
        fields: &[
            ("draw_args", "cull.indirect_buffers"),
            ("draw_args2", "cull.indirect_buffers2"),
            ("cull_status", "cull.cull_status_buffers"),
            ("cluster_light_list", "light_cull.cluster_buffer"),
            ("ao_output", "transient_pool.image_for"),
            ("shadow_map", "shadow.map.image"),
            ("spot_shadow_map", "spot_shadow.map.image"),
            ("fog_froxel_volume", "volume.image"),
            ("hdr_depth", "depth_images"),
            ("hiz_pyramid", "hiz.pyramid"),
        ],
        // A Vulkan barrier names its target inside the builder chain, or through
        // one of the file-local helpers that take the handle as their first
        // argument. Matching the builder rather than any `.image(` / `.buffer(`
        // matters: descriptor writes use the same setter names.
        markers: &[
            "vk::ImageMemoryBarrier::default()",
            "vk::BufferMemoryBarrier::default()",
            "hiz_image_barrier(",
            "depth_barrier(",
            "color_barrier(",
        ],
    },
    BackendRegistry {
        backend: "directx",
        root: DIRECTX_ROOT,
        fields: &[
            ("draw_args", "cull.indirect_cmd_buffers"),
            ("draw_args2", "cull.indirect_cmd_buffers_2"),
            ("cull_status", "cull.cull_status_buffers"),
            ("cluster_light_list", "light_cull.cluster_buffer"),
            ("ao_output", "transient_pool.resource_for"),
            ("shadow_map", "shadow.resource"),
            ("spot_shadow_map", "spot_shadow.resource"),
            ("fog_froxel_volume", "volume_resource"),
            ("hdr_depth", "depth_resource"),
            ("hiz_pyramid", "hiz.texture"),
            ("hdr_color", "hdr.color"),
            ("hdr_resolve", "hdr_scene_target()"),
            ("scene_pre_taa", "rc.output"),
            // `post_scene_target` resolves to the reflection composite's output
            // or, with no resolve, the spine -- so it covers both scene labels.
            // It exists so the passes that pick between them still name the
            // resource at the barrier, which a local binding would hide.
            ("scene_pre_taa", "post_scene_target()"),
            ("scene_color", "taa.history"),
            ("bloom_top", "bloom.mips"),
            ("gbuffer_normal_depth", "gb.normal_depth"),
            ("gbuffer_roughness", "gb.roughness"),
            ("gbuffer_velocity", "gb.velocity"),
        ],
        markers: &["transition_barrier(", "uav_barrier(", "aliasing_barrier("],
    },
];

// Barriers that legitimately target a registry-resolved resource outside the
// executor. Each is finer than the graph's one-state-per-resource granularity, so
// no derived transition can replace it.
const ALLOWED: &[(&str, &str, &str)] = &[
    // The Hi-Z reduction writes mip N while sampling mip N-1. The graph drives the
    // pyramid's open and close around the whole chain; these order the steps
    // within it.
    ("vulkan", "hiz.rs", "hiz.pyramid"),
    ("directx", "hiz.rs", "hiz.texture"),
    // The MSAA resolve step. The graph rests both HDR targets in RENDER_TARGET
    // and `ResolveSubresource` needs RESOLVE_SOURCE / RESOLVE_DEST for the
    // length of one call, so each pass borrows them and hands them back.
    ("directx", "draw/main.rs", "hdr.color"),
    ("directx", "draw/main.rs", "hdr_scene_target()"),
    ("directx", "raymarch.rs", "hdr.color"),
    // Raymarch also snapshots the scene for refractive user shaders: a fragment
    // cannot sample the attachment it is writing.
    ("directx", "raymarch.rs", "hdr_scene_target()"),
    // The SSGI gather samples the scene its composite blends into, so the node
    // reads and writes one resource where the graph models a single write.
    ("directx", "post/ssgi.rs", "hdr_scene_target()"),
    // The translucent pass's refraction snapshot, on whichever scene the graph
    // handed it.
    ("directx", "glass.rs", "post_scene_target()"),
    // The bloom chain: the graph drives `bloom_top` (mip 0) across the node, and
    // these order the octaves inside it -- including mip 0's own borrow, which
    // the downsample samples between the prefilter and the final upsample.
    ("directx", "post/bloom.rs", "bloom.mips"),
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

// Source with line comments removed, so a field named in prose is not a hit.
fn code_only(source: &str) -> String {
    source
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

// Whitespace removed, so a target token matches regardless of how rustfmt broke
// the expression. `self.shadow.resource` is one token to a reader and three lines
// to the formatter, and a reflow must not be able to hide a double-drive.
fn squeeze(code: &str) -> String {
    code.chars().filter(|c| !c.is_whitespace()).collect()
}

// The text of each barrier target expression in `code`: from each marker to the
// end of its statement. A builder chain spans several lines, so the statement
// terminator is the bound rather than the line.
fn barrier_targets<'a>(code: &'a str, markers: &[&str]) -> Vec<&'a str> {
    let mut out = Vec::new();
    for marker in markers {
        let mut from = 0;
        while let Some(rel) = code[from..].find(marker) {
            let start = from + rel;
            let end = code[start..]
                .find(';')
                .map(|e| start + e)
                .unwrap_or(code.len());
            out.push(&code[start..end]);
            from = start + marker.len();
        }
    }
    out
}

// Every (file, field) pair where a barrier outside the executor targets a
// registry-resolved resource.
fn double_driven(registry: &BackendRegistry) -> BTreeSet<(String, &'static str)> {
    let mut files = Vec::new();
    rust_sources(Path::new(registry.root), "", &mut files);
    let mut found = BTreeSet::new();
    for (rel, path) in files {
        if rel == EXECUTOR {
            continue;
        }
        let code = squeeze(&code_only(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel}: {e}")),
        ));
        for target in barrier_targets(&code, registry.markers) {
            for (_, token) in registry.fields {
                if target.contains(token) {
                    found.insert((rel.clone(), *token));
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_encoder_transitions_a_graph_driven_resource() {
        let mut failures = Vec::new();
        for registry in REGISTRIES {
            let found = double_driven(registry);
            let allowed: BTreeSet<(String, &str)> = ALLOWED
                .iter()
                .filter(|(b, ..)| *b == registry.backend)
                .map(|(_, file, field)| (file.to_string(), *field))
                .collect();

            for (file, field) in &found {
                if allowed.contains(&(file.clone(), *field)) {
                    continue;
                }
                let label = registry
                    .fields
                    .iter()
                    .find(|(_, f)| f == field)
                    .map(|(l, _)| *l)
                    .unwrap_or("?");
                failures.push(format!(
                    "{}/{file}: emits a barrier targeting `{field}`, which the graph executor \
                     already drives as `{label}`. Both will run every frame and the second will \
                     name a state the resource has already left. Remove the inline transition, \
                     or add it to ALLOWED with the reason it is finer than the graph can express.",
                    registry.backend
                ));
            }
            for (file, field) in &allowed {
                if !found.contains(&(file.clone(), *field)) {
                    failures.push(format!(
                        "{}/{file}: ALLOWED lists `{field}` but no barrier there targets it; \
                         drop the entry",
                        registry.backend
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    #[test]
    fn the_registry_field_table_matches_the_resolver() {
        // The table is hand-written, so it can drift from the code it describes: a
        // resource dropped from the registry, or a field renamed, would silently
        // stop being checked and this guard would pass by looking at nothing.
        // Every label must still be resolved by the executor, and every target
        // token must still name something in the backend.
        for registry in REGISTRIES {
            let mut files = Vec::new();
            rust_sources(Path::new(registry.root), "", &mut files);
            let all: String = squeeze(
                &files
                    .iter()
                    .map(|(rel, path)| {
                        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {rel}: {e}"))
                    })
                    .collect::<String>(),
            );
            let resolver = std::fs::read_to_string(Path::new(registry.root).join(EXECUTOR))
                .unwrap_or_else(|e| panic!("read {EXECUTOR}: {e}"));
            for (label, token) in registry.fields {
                assert!(
                    resolver.contains(&format!("\"{label}\"")),
                    "{}: the executor no longer resolves {label}; drop it from this table \
                     (its encoder owns its barriers again)",
                    registry.backend
                );
                assert!(
                    all.contains(token),
                    "{}: no code names `{token}` (backing {label}); the field was renamed and \
                     this guard stopped checking it",
                    registry.backend
                );
            }
        }
    }

    #[test]
    fn the_scan_finds_a_planted_double_drive() {
        // Negative control. Without this the check could pass by finding nothing
        // at all -- a marker that stopped matching would read as "clean".
        let code = squeeze(&code_only(
            "let b = vk::ImageMemoryBarrier::default()\n    .image(self.depth_images[i].image);\n",
        ));
        let targets = barrier_targets(&code, &["vk::ImageMemoryBarrier::default()"]);
        assert_eq!(targets.len(), 1);
        assert!(targets[0].contains("depth_images"));

        // And a barrier on something the graph does not drive stays silent.
        let code = squeeze(&code_only(
            "let b = vk::ImageMemoryBarrier::default().image(planar.target);\n",
        ));
        let targets = barrier_targets(&code, &["vk::ImageMemoryBarrier::default()"]);
        assert!(!targets[0].contains("depth_images"));
    }
}
