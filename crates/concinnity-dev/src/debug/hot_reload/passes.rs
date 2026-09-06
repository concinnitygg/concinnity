// src/debug/hot_reload/passes.rs
//
// The world.jsonl / ProceduralMesh / VolumetricFog / world-loaded Shader stage
// reload passes: re-read the on-disk source, diff against the captured state,
// and apply changes through the backend. Each returns a small tally the drive
// logs.

use crate::gfx::graphics_system::hot_reload_sources::*;

// Per-reload tally for the volumetric-fog path. Counts are surfaced separately
// so a single info! line per asset class keeps the log readable.
#[derive(Default, Debug)]
pub(crate) struct FogReloadResult {
    // True when the resolved `Option<FogSettings>` differed from what was
    // last pushed; the trait call fired and the dedupe state advanced.
    pub updated: bool,
}

// Re-read `world.jsonl` and push the first declared `VolumetricFog` through
// [`RenderBackend::update_fog_settings`]. Disabled / missing assets push
// `None`. Each authored asset runs through the same clamp chain as init:
// [`crate::components::VolumetricFog::from_args`] (asset-side floors) then
// [`crate::gfx::volumetric_fog::FogSettings::resolve`] (gfx-side ceilings),
// so a reload cannot land out-of-range values.
//
// `last_pushed` is the dedupe state owned by the caller; the function
// compares the freshly-resolved value against it and only fires the trait
// call (and reports `updated = true`) on a real change. Backend-agnostic:
// rides on the `update_fog_settings` trait, which has a default no-op on
// backends that have not yet implemented runtime fog mutation.
pub(super) fn reload_volumetric_fog(
    path: &str,
    last_pushed: &mut Option<crate::gfx::volumetric_fog::FogSettings>,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> FogReloadResult {
    let mut result = FogReloadResult::default();

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                "VolumetricFog hot-reload: failed to read '{}': {} (no update)",
                path,
                e,
            );
            return result;
        }
    };
    let entries = match concinnity_cook::build_only::expand_world_from_str(
        &content,
        crate::project::assets_dir().as_deref(),
    ) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                "VolumetricFog hot-reload: failed to parse '{}': {} (no update)",
                path,
                e,
            );
            return result;
        }
    };

    // First declared VolumetricFog wins; mirrors the init drain in
    // `run_init`. A missing entry or one with `enabled = false` resolves to
    // `None`, which disables the pass.
    let mut resolved: Option<crate::gfx::volumetric_fog::FogSettings> = None;
    for entry in &entries {
        let asset_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if asset_type != "VolumetricFog" {
            continue;
        }
        let args = entry
            .get("args")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let parsed: crate::components::VolumetricFog = match serde_json::from_value(args) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "VolumetricFog hot-reload: failed to parse args: {} (kept previous state)",
                    e,
                );
                return result;
            }
        };
        // Apply the build-side validator, exactly as a rebuild would at bake.
        let clamped = concinnity_cook::authoring::validate::volumetric_fog(parsed);
        if clamped.enabled {
            resolved = Some(crate::gfx::volumetric_fog::FogSettings::resolve(
                clamped.color,
                clamped.density,
                clamped.height_falloff,
                clamped.height_reference,
                clamped.max_distance,
                clamped.phase_g,
                clamped.ambient,
            ));
        }
        break;
    }

    if resolved != *last_pushed {
        backend.update_fog_settings(resolved);
        *last_pushed = resolved;
        result.updated = true;
    }
    result
}

// Per-reload tally for the procedural-mesh path. Counts are surfaced separately
// so a single info! line per asset class keeps the log readable.
#[derive(Default, Debug)]
pub(crate) struct ProceduralMeshReloadResult {
    // Procedural meshes whose args differed from the captured snapshot and
    // whose regenerated geometry was pushed to the backend (in-place
    // `update_mesh_geometry` or batched `rebuild_static_geometry`).
    pub regenerated: usize,
    // Procedural meshes whose args matched the snapshot: nothing pushed.
    pub unchanged: usize,
    // Procedural meshes whose generator returned an error (left at the
    // pre-reload geometry).
    pub failed: usize,
}

// Re-read `world.jsonl` and, for every captured `ProceduralMesh`, diff its
// current generator args against the init-time snapshot. When the args
// differ, re-run `compile_mesh_payload` + `deserialise_with_lods` to get
// fresh vertices / indices / LOD alternates, then dispatch to
// `update_mesh_geometry` (same vertex/index/LOD counts) or batch into a
// single `rebuild_static_geometry` (size changed); mirrors the file-backed
// `Mesh` reload's in-place-vs-rebuild dispatch but driven by JSONL args
// instead of a `.glb` file. Updates the captured args on success so a
// subsequent reload diffs against the post-edit state.
//
// Backend-agnostic: rides on the existing `update_mesh_geometry` /
// `rebuild_static_geometry` trait surface, implemented on all three backends.
pub(super) fn reload_procedural_meshes(
    path: &str,
    procedural_meshes: &mut ProceduralMeshSourceMap,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> ProceduralMeshReloadResult {
    let mut result = ProceduralMeshReloadResult::default();
    if procedural_meshes.is_empty() {
        return result;
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                "ProceduralMesh hot-reload: failed to read '{}': {} (no regen)",
                path,
                e
            );
            return result;
        }
    };
    // Expand prefabs / etc. so auto-injected ProceduralMeshes match the
    // init-time captured set; otherwise an unchanged entry shows up as
    // "missing from JSONL" every reload.
    let entries = match concinnity_cook::build_only::expand_world_from_str(
        &content,
        crate::project::assets_dir().as_deref(),
    ) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                "ProceduralMesh hot-reload: failed to parse '{}': {} (no regen)",
                path,
                e
            );
            return result;
        }
    };

    // Build a name -> entry map from the new JSONL for O(N) per-entry lookup:
    // the parsed component (typed equality against the captured init state,
    // so both sides see identical filled-in defaults) plus the raw args JSON
    // the regen compile consumes.
    let mut new_args_by_name: std::collections::HashMap<
        String,
        (crate::components::ProceduralMesh, serde_json::Value),
    > = std::collections::HashMap::new();
    for entry in &entries {
        let asset_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if asset_type != "ProceduralMesh" {
            continue;
        }
        let Some(name) = entry.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let raw_args = entry
            .get("args")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let parsed: crate::components::ProceduralMesh =
            match serde_json::from_value(raw_args.clone()) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(
                        "ProceduralMesh hot-reload: failed to parse '{}' args: {} \
                         (kept old geometry)",
                        name,
                        e
                    );
                    continue;
                }
            };
        new_args_by_name.insert(name.to_string(), (parsed, raw_args));
    }

    // Collect rebuild changes batched into a single rebuild call (mirrors the
    // file-backed Mesh path). Each entry either commits in place, queues a
    // rebuild change, or is unchanged.
    let mut rebuild_changes: Vec<crate::gfx::backend::DrawGeometryUpdate> = Vec::new();
    let mut rebuild_entry_indices: Vec<usize> = Vec::new();
    // Args to write back into the source map after a successful update,
    // staged here so a failed regen / rebuild doesn't clobber the captured
    // value (the diff next reload would then miss the still-pending edit).
    let mut staged_args: Vec<(usize, crate::components::ProceduralMesh)> = Vec::new();

    for (entry_idx, entry) in procedural_meshes.entries.iter().enumerate() {
        let Some((new_args, raw_args)) = new_args_by_name.get(&entry.name) else {
            // No matching entry in the new JSONL: treat as removed-from-jsonl.
            // We deliberately do not destroy the existing draws here: a Prop
            // referencing this mesh is still rendering through its draw slot.
            result.unchanged += 1;
            continue;
        };
        if new_args == &entry.args {
            result.unchanged += 1;
            continue;
        }

        // Regenerate from the new args. Routed through the build wrapper so a
        // live-edited `heightfield` ProceduralMesh still decodes its source
        // image (core's compile_mesh_payload links no image decoders).
        let payload = match concinnity_cook::compile::mesh_compile::compile_mesh_payload(
            raw_args,
            crate::project::assets_dir().as_deref(),
        ) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "ProceduralMesh hot-reload: regen failed for '{}': {} (kept old geometry)",
                    entry.name,
                    e
                );
                result.failed += 1;
                continue;
            }
        };
        let (vertices, indices, lod_alternates) =
            match crate::gfx::mesh_payload::deserialise_with_lods(&payload) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        "ProceduralMesh hot-reload: deserialise failed for '{}': {} \
                         (kept old geometry)",
                        entry.name,
                        e
                    );
                    result.failed += 1;
                    continue;
                }
            };

        // Match the file-backed Mesh dispatch: size-changing regens go through
        // `rebuild_static_geometry`, size-matched go through
        // `update_mesh_geometry`. The size check probes any one of the draw
        // slots: they were built from the same source so they all carry the
        // same vertex / index / LOD counts.
        let size_changed = entry.draw_indices.iter().any(|&draw_idx| {
            let base_changed = match backend.draw_geometry_size(draw_idx) {
                Some((v, i)) => v != vertices.len() || i != indices.len(),
                None => false,
            };
            let lod_changed = match backend.draw_lod_index_counts(draw_idx) {
                Some(counts) => {
                    counts.len() != lod_alternates.len()
                        || counts
                            .iter()
                            .zip(lod_alternates.iter())
                            .any(|(c, (_, idx))| *c != idx.len())
                }
                None => false,
            };
            base_changed || lod_changed
        });

        if size_changed {
            for &draw_idx in &entry.draw_indices {
                rebuild_changes.push(crate::gfx::backend::DrawGeometryUpdate {
                    draw_idx,
                    vertices: vertices.clone(),
                    indices: indices.clone(),
                    lod_alternates: lod_alternates.clone(),
                });
            }
            rebuild_entry_indices.push(entry_idx);
            staged_args.push((entry_idx, new_args.clone()));
            continue;
        }

        let mut slot_failures = 0usize;
        for &draw_idx in &entry.draw_indices {
            if let Err(e) =
                backend.update_mesh_geometry(draw_idx, &vertices, &indices, &lod_alternates)
            {
                tracing::warn!(
                    "ProceduralMesh hot-reload: backend rejected update for '{}' draw {}: \
                     {}",
                    entry.name,
                    draw_idx,
                    e
                );
                slot_failures += 1;
            }
        }
        if slot_failures == 0 {
            result.regenerated += 1;
            staged_args.push((entry_idx, new_args.clone()));
        } else {
            result.failed += 1;
        }
    }

    if !rebuild_changes.is_empty() {
        match backend.rebuild_static_geometry(rebuild_changes) {
            Ok(()) => {
                result.regenerated += rebuild_entry_indices.len();
            }
            Err(e) => {
                tracing::error!(
                    "ProceduralMesh hot-reload: rebuild_static_geometry failed: {} \
                     ({} entry/entries kept old geometry)",
                    e,
                    rebuild_entry_indices.len()
                );
                result.failed += rebuild_entry_indices.len();
                // Drop the staged-args writes for the failed rebuild so the
                // captured args still reflect what the backend is actually
                // rendering.
                let failed_idxs: std::collections::HashSet<usize> =
                    rebuild_entry_indices.iter().copied().collect();
                staged_args.retain(|(idx, _)| !failed_idxs.contains(idx));
            }
        }
    }

    for (entry_idx, new_args) in staged_args {
        if let Some(e) = procedural_meshes.entries.get_mut(entry_idx) {
            e.args = new_args;
        }
    }

    result
}

// Re-expand the world's stories from disk and return every compiled `Story`
// whose graph changed since the last pass. The expansion is the same one the
// build runs (`expand_world_from_str` includes the StoryImport pass), so the
// returned graphs match what a fresh `cn debug` would compile; `cn debug`
// only, where the interner is warm and re-interning the generated names
// yields the ids the running world already carries. `snapshots` is the
// caller-owned name -> compiled-args map used to skip unchanged stories (a
// save of one story's `.md` must not re-render the others); it starts empty,
// so the first pass after a `.md` event hands over every story once.
//
// Limits, by design: the pass re-reads the graph only. Stage furniture is
// fixed at init, so a story whose widest menu grew renders (and accepts
// clicks for) only the buttons the world was built with, and a new image or
// audio file needs a restart (the sprite atlas pool and clip cache are
// init-time; a swapped-in id that was never pooled falls back with a warn).
pub(super) fn reload_stories(
    path: &str,
    snapshots: &mut std::collections::HashMap<String, serde_json::Value>,
) -> Vec<crate::components::Story> {
    // Each expanded Story deserializes its scaffold name-references, so the
    // name resolver must be installed. In a running editor the initial build
    // already set it up; installing it here too is a cheap no-op and keeps a
    // standalone reload correct.
    crate::ecs::asset_id::ensure_name_resolver();
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                "story hot-reload: failed to read '{}': {} (no update)",
                path,
                e
            );
            return Vec::new();
        }
    };
    let entries = match concinnity_cook::build_only::expand_world_from_str(
        &content,
        crate::project::assets_dir().as_deref(),
    ) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                "story hot-reload: failed to expand '{}': {} (kept the running story)",
                path,
                e,
            );
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for entry in &entries {
        if entry.get("type").and_then(|t| t.as_str()) != Some("Story") {
            continue;
        }
        let Some(name) = entry.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let args = entry
            .get("args")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        if snapshots.get(name) == Some(&args) {
            continue;
        }
        match serde_json::from_value::<crate::components::Story>(args.clone()) {
            Ok(story) => {
                snapshots.insert(name.to_string(), args);
                out.push(story);
            }
            Err(e) => {
                tracing::error!(
                    "story hot-reload: failed to parse compiled story '{}': {} \
                     (kept the running story)",
                    name,
                    e,
                );
            }
        }
    }
    out
}

// Per-reload tally for the world-loaded shader-stage path. Counts are
// surfaced separately from the asset-payload / world / procedural-mesh
// tallies so a single info line per asset class keeps the log readable.
#[derive(Default, Debug)]
pub(crate) struct ShaderStageReloadResult {
    // Shader stage sources that re-compiled successfully and contributed
    // to the backend pipeline rebuild. Counts every kind that the caller
    // re-read off disk, including stages whose source bytes were
    // byte-for-byte unchanged (a no-op shader save still re-compiles,
    // detecting a no-op would mean caching every shader's byte payload).
    pub recompiled: usize,
    // Shader stage sources that failed to compile (logged per-stage in
    // the helper). The backend rebuild is skipped entirely on any failure:
    // the live pipelines stay bound and the next save can recover.
    pub failed: usize,
    // `true` when the backend pipeline rebuild ran and swapped fresh
    // pipelines into the live context. `false` when one or more stages
    // failed to compile or the backend rejected the rebuild: the next
    // save retries from scratch.
    pub pipelines_rebuilt: bool,
}

// Recompile the world default Shader from its files and swap the live
// pipelines. Fired by the asset watcher when one of the Shader's files
// changes, or by the debug-WS `reload-assets` command.
//
// The files are read and compiled through the cook's own
// [`concinnity_cook::compile::shader::compile_world_shader`], so a save
// recompiles exactly the path `cn build` took. Any failure logs and aborts the
// pass without touching the backend, so a typo never desyncs a live pipeline.
// When the compile succeeds the programs go through
// [`crate::gfx::backend::RenderBackend::update_world_shader_pipelines`], which
// runs the rebuild-then-swap dance on every backend.
//
// Does not detect "unchanged source": every fire recompiles the whole Shader.
// A save is rare enough, and the compile cheap enough, that per-file mtime
// tracking would not be worth the state.
pub(super) fn reload_shader_stages(
    shader_stages: &ShaderStageSourceMap,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> ShaderStageReloadResult {
    use crate::components::ShaderStage;
    use concinnity_core::render::slang_programs::surface::Sources;

    let mut result = ShaderStageReloadResult::default();
    if shader_stages.is_empty() {
        return result;
    }

    let mut texts: std::collections::HashMap<ShaderStage, String> =
        std::collections::HashMap::new();
    for entry in &shader_stages.entries {
        match concinnity_cook::compile::shader::read_shader_source(&entry.resolved_path) {
            Ok(text) => {
                texts.insert(entry.stage, text);
            }
            Err(e) => {
                tracing::error!(
                    "Shader hot-reload: failed to read '{}' ({:?}): {} \
                     (live pipelines kept their previous source)",
                    entry.resolved_path,
                    entry.stage,
                    e
                );
                result.failed += 1;
            }
        }
    }
    if result.failed > 0 {
        return result;
    }
    let Some(fragment) = texts.get(&ShaderStage::Fragment) else {
        tracing::error!("Shader hot-reload: no fragment file was captured for the world Shader");
        result.failed += 1;
        return result;
    };
    let sources = Sources {
        vertex: texts.get(&ShaderStage::Vertex).map(String::as_str),
        fragment,
    };
    let programs = match concinnity_cook::compile::shader::compile_world_shader(
        "world default",
        &sources,
        crate::cook_platform(),
    ) {
        Ok(programs) => programs,
        Err(e) => {
            tracing::error!(
                "Shader hot-reload: compile failed: {} (live pipelines kept their previous source)",
                e
            );
            result.failed += 1;
            return result;
        }
    };
    result.recompiled = texts.len();

    match backend.update_world_shader_pipelines(&programs) {
        Ok(()) => {
            result.pipelines_rebuilt = true;
        }
        Err(e) => {
            tracing::error!(
                "Shader hot-reload: backend pipeline rebuild rejected: {} \
                 (live pipelines kept their previous source)",
                e
            );
        }
    }
    result
}
