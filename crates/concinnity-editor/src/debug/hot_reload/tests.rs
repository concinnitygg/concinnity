// src/debug/hot_reload/tests.rs
//
// Unit tests for the hot-reload machinery (moved here from the single-file
// module). Pull each submodule's items in explicitly.

use super::decode::*;
use super::passes::*;
use super::state::*;
use super::watcher::*;
use crate::gfx::graphics_system::hot_reload_sources::*;
use notify::{Event, EventKind};
use std::path::PathBuf;

#[test]
fn empty_map_round_trips() {
    let m = TextureSourceMap::new();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn pushes_and_collects_unique_parent_dirs() {
    let mut m = TextureSourceMap::new();
    m.push_texture("assets/a.png".to_string(), 0, 0);
    m.push_texture("assets/b.png".to_string(), 0, 1);
    m.push_texture("textures/nm.png".to_string(), 0, 1);
    assert_eq!(m.entries.len(), 3);
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 2);
    assert!(dirs.iter().any(|p| p.ends_with("assets")));
    assert!(dirs.iter().any(|p| p.ends_with("textures")));
}

#[test]
fn bare_filenames_skip_watch_dir() {
    // A source with no parent directory has nowhere to watch; the watcher
    // would otherwise try to subscribe to "" which notify rejects.
    let mut m = TextureSourceMap::new();
    m.push_texture("standalone.png".to_string(), 0, 0);
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn glb_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/model.glb"));
    assert!(is_asset_event(&evt));
}

#[test]
fn cube_extension_is_an_asset_event() {
    // ColorLut sources travel through the same watcher; `.cube` must pass.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/grade.cube"));
    assert!(is_asset_event(&evt));
}

#[test]
fn hdr_extension_is_an_asset_event() {
    // EnvironmentMap sources travel through the same watcher; `.hdr` must pass.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/studio.hdr"));
    assert!(is_asset_event(&evt));
}

#[test]
fn hdr_extension_matches_case_insensitively() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/STUDIO.HDR"));
    assert!(is_asset_event(&evt));
}

#[test]
fn unrelated_extension_is_not_an_asset_event() {
    // `.jsonl` (the world file) is now a recognised asset event; pick an
    // extension nothing in the engine cares about as the negative case.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/world.txt"));
    assert!(!is_asset_event(&evt));
}

#[test]
fn state_with_only_environment_map_still_spawns_a_watcher() {
    // With no textures and no LUT but an EnvironmentMap, the watcher must
    // still be set up so `.hdr` saves trigger reloads. The state stores the
    // EnvironmentMapSource verbatim for the reload helper to consult.
    let env_map = EnvironmentMapSource {
        resolved_path: format!(
            "{}/asset_hot_reload_envmap_only_{}.hdr",
            std::env::temp_dir().display(),
            std::process::id()
        ),
        prefilter_face_size: 64,
        irradiance_face_size: 16,
        prefilter_samples: 64,
        prefilter_clamp: 12.0,
    };
    // Parent dir (temp dir) exists, so the watcher should subscribe.
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        environment_map: Some(env_map.clone()),
        ..Default::default()
    });
    assert!(state.environment_map.is_some());
    let captured = state.environment_map.as_ref().unwrap();
    assert_eq!(captured.prefilter_face_size, env_map.prefilter_face_size);
    assert_eq!(captured.irradiance_face_size, env_map.irradiance_face_size);
    assert_eq!(captured.prefilter_samples, env_map.prefilter_samples);
}

#[test]
fn fully_empty_state_skips_watcher_creation() {
    // No textures, no LUT, no EnvironmentMap, no meshes, no skinned, no
    // world path → nothing to watch.
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    assert!(state.environment_map.is_none());
    assert!(state.color_lut.is_none());
    assert!(state.map.is_empty());
    assert!(state.meshes.is_empty());
    assert!(state.skinned_meshes.is_empty());
}

#[test]
fn mesh_source_map_collects_unique_parent_dirs() {
    let mut m = MeshSourceMap::new();
    m.entries.push(MeshSourceEntry {
        source: "assets/models/a.glb".to_string(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0, 1],
    });
    m.entries.push(MeshSourceEntry {
        source: "assets/models/a.glb".to_string(),
        primitive_index: 1,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![2],
    });
    m.entries.push(MeshSourceEntry {
        source: "assets/hdri/b.glb".to_string(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![3],
    });
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 2);
    assert!(dirs.iter().any(|p| p.ends_with("assets/models")));
    assert!(dirs.iter().any(|p| p.ends_with("assets/hdri")));
}

#[test]
fn mesh_source_map_skips_bare_filenames_in_watch_dirs() {
    // A bare filename has no parent directory; the watcher would otherwise
    // try to subscribe to "" which notify rejects. The debug-WS
    // `reload-assets` command path still works for these.
    let mut m = MeshSourceMap::new();
    m.entries.push(MeshSourceEntry {
        source: "standalone.glb".to_string(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0],
    });
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn state_with_only_meshes_still_spawns_a_watcher() {
    let mut meshes = MeshSourceMap::new();
    meshes.entries.push(MeshSourceEntry {
        source: format!("{}/dummy.glb", std::env::temp_dir().display()),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0],
    });
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        meshes,
        ..Default::default()
    });
    assert_eq!(state.meshes.len(), 1);
    assert_eq!(state.meshes.entries[0].draw_indices, vec![0]);
}

#[test]
fn skinned_mesh_source_map_collects_unique_parent_dirs() {
    let mut m = SkinnedMeshSourceMap::new();
    m.entries.push(SkinnedMeshSourceEntry {
        source: "assets/models/a.glb".to_string(),
        skin_index: 0,
        skinned_index: 0,
        vertex_base: 0,
        vertex_count: 100,
        index_count: 300,
        joint_count: 24,
    });
    m.entries.push(SkinnedMeshSourceEntry {
        source: "assets/models/b.glb".to_string(),
        skin_index: 0,
        skinned_index: 1,
        vertex_base: 100,
        vertex_count: 50,
        index_count: 150,
        joint_count: 5,
    });
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 1);
    assert!(dirs[0].ends_with("assets/models"));
}

#[test]
fn state_with_only_skinned_still_spawns_a_watcher() {
    let mut skinned = SkinnedMeshSourceMap::new();
    skinned.entries.push(SkinnedMeshSourceEntry {
        source: format!("{}/skinned.glb", std::env::temp_dir().display()),
        skin_index: 0,
        skinned_index: 0,
        vertex_base: 0,
        vertex_count: 8,
        index_count: 24,
        joint_count: 2,
    });
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        skinned_meshes: skinned,
        ..Default::default()
    });
    assert_eq!(state.skinned_meshes.len(), 1);
    assert_eq!(state.skinned_meshes.entries[0].joint_count, 2);
}

#[test]
fn state_with_only_world_jsonl_still_spawns_a_watcher() {
    // World-only worlds (no Texture/LUT/EnvMap/Mesh/Skinned) still want
    // their world.jsonl watched so Prop transform edits propagate.
    let world_path = format!(
        "{}/world_only_{}.jsonl",
        std::env::temp_dir().display(),
        std::process::id()
    );
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        world_jsonl_path: Some(world_path.clone()),
        ..Default::default()
    });
    assert_eq!(state.world_jsonl_path.as_deref(), Some(world_path.as_str()));
}

#[test]
fn jsonl_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/world.jsonl"));
    assert!(is_asset_event(&evt));
}

#[test]
fn md_extension_is_an_asset_event() {
    // StoryImport sources travel through the same watcher; the closure
    // routes `.md` saves to the story re-expansion flag.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/story.md"));
    assert!(is_asset_event(&evt));
}

#[test]
fn reload_stories_re_expands_and_dedupes_by_snapshot() {
    let dir = std::env::temp_dir().join(format!("story_reload_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let md = dir.join("tale.md");
    std::fs::write(
        &md,
        "---\ntitle: Tale\ncharacters:\n  a: Ana\n---\n\n# start\n\nHello there.\n",
    )
    .unwrap();
    let world = dir.join("world.jsonl");
    std::fs::write(
        &world,
        format!(
            "{}\n",
            serde_json::json!({
                "name": "tale", "type": "StoryImport",
                "args": {"source": md.to_str().unwrap()}
            })
        ),
    )
    .unwrap();

    let mut snapshots = std::collections::HashMap::new();
    let stories = reload_stories(world.to_str().unwrap(), &mut snapshots);
    assert_eq!(stories.len(), 1);
    assert_eq!(stories[0].title, "Tale");
    assert_eq!(stories[0].nodes[0].pages[0].text, "Hello there.");
    assert!(stories[0].scaffold.screen.is_some());

    // Unchanged source: the snapshot filters it out.
    let unchanged = reload_stories(world.to_str().unwrap(), &mut snapshots);
    assert!(unchanged.is_empty());

    // Edited dialogue comes back exactly once.
    std::fs::write(
        &md,
        "---\ntitle: Tale\ncharacters:\n  a: Ana\n---\n\n# start\n\nHello again.\n",
    )
    .unwrap();
    let edited = reload_stories(world.to_str().unwrap(), &mut snapshots);
    assert_eq!(edited.len(), 1);
    assert_eq!(edited[0].nodes[0].pages[0].text, "Hello again.");

    // A broken edit keeps the running story (and the snapshot state).
    std::fs::write(&md, "---\ntitle: Broken\n").unwrap();
    let broken = reload_stories(world.to_str().unwrap(), &mut snapshots);
    assert!(broken.is_empty());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn fresh_state_has_no_envmap_in_flight() {
    // The off-thread envmap convolution slot must start empty; a
    // non-`None` value at construction would skip the very first reload
    // request a `reload_assets` pass made.
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let slot = state.env_map_inflight.lock().expect("lock");
    assert!(slot.is_none());
}

#[test]
fn fresh_state_has_no_asset_batch_in_flight() {
    // Same invariant as the envmap slot: a non-`None` value at
    // construction would make the very first `reload_assets` think a
    // worker was already running and skip the spawn.
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let slot = state.asset_batch_inflight.lock().expect("lock");
    assert!(slot.is_none());
}

#[test]
fn decode_asset_batch_with_empty_inputs_returns_empty_batch() {
    // The worker body must handle the no-sources case cleanly so an
    // accidentally-spawned worker on a world without any file-backed
    // assets exits quickly with nothing to apply.
    let batch = decode_asset_batch(Vec::new(), None, Vec::new(), Vec::new());
    assert!(batch.textures.is_empty());
    assert!(batch.color_lut.is_none());
    assert!(batch.meshes.is_empty());
    assert!(batch.skinned_meshes.is_empty());
    assert_eq!(batch.decode_failures, 0);
}

#[test]
fn apply_skinned_layouts_refreshes_every_matching_entry() {
    // After a size-changing skinned rebuild, every source-map entry
    // whose `skinned_index` appears in the returned layouts should pick
    // up the new vertex_base / vertex_count / index_count so subsequent
    // in-place reloads write to the correct shared-buffer regions.
    let mut entries = vec![
        SkinnedMeshSourceEntry {
            source: "a.glb".to_string(),
            skin_index: 0,
            skinned_index: 0,
            vertex_base: 0,
            vertex_count: 10,
            index_count: 30,
            joint_count: 4,
        },
        SkinnedMeshSourceEntry {
            source: "b.glb".to_string(),
            skin_index: 0,
            skinned_index: 1,
            vertex_base: 10,
            vertex_count: 20,
            index_count: 60,
            joint_count: 6,
        },
    ];
    let layouts = vec![
        crate::gfx::backend::SkinnedSlotLayout {
            skinned_index: 0,
            vertex_base: 0,
            vertex_count: 15,
            index_count: 45,
        },
        crate::gfx::backend::SkinnedSlotLayout {
            skinned_index: 1,
            vertex_base: 15,
            vertex_count: 20,
            index_count: 60,
        },
    ];
    apply_skinned_layouts_to_entries(&mut entries, &layouts);
    assert_eq!(entries[0].vertex_base, 0);
    assert_eq!(entries[0].vertex_count, 15);
    assert_eq!(entries[0].index_count, 45);
    // Unchanged slot 1 was still re-packed (vertex_base shifted from 10
    // → 15 because slot 0's vertex_count grew from 10 → 15).
    assert_eq!(entries[1].vertex_base, 15);
    assert_eq!(entries[1].vertex_count, 20);
    assert_eq!(entries[1].index_count, 60);
    // joint_count is untouched: that lives outside the rebuild's scope.
    assert_eq!(entries[0].joint_count, 4);
    assert_eq!(entries[1].joint_count, 6);
}

#[test]
fn drain_pending_skeleton_updates_clears_the_queue() {
    // The render thread polls + drains in one step; a second drain on
    // the same frame must return nothing so a successful apply does not
    // double-write the SkeletonPose components.
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    state.pending_skeleton_updates.push(PendingSkeletonUpdate {
        skinned_index: 0,
        new_skeleton: crate::gfx::skinning::Skeleton::new(Vec::new()),
    });
    state.pending_skeleton_updates.push(PendingSkeletonUpdate {
        skinned_index: 3,
        new_skeleton: crate::gfx::skinning::Skeleton::new(Vec::new()),
    });
    let drained = state.drain_pending_skeleton_updates();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].skinned_index, 0);
    assert_eq!(drained[1].skinned_index, 3);
    // Second drain returns empty.
    assert!(state.drain_pending_skeleton_updates().is_empty());
}

#[test]
fn procedural_mesh_source_map_round_trips_empty() {
    let m = ProceduralMeshSourceMap::new();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

#[test]
fn procedural_mesh_args_normalise_via_round_trip() {
    // The init pipeline captures args as `serde_json::to_value(component)`;
    // the reload pipeline normalises new on-disk args through
    // `ProceduralMesh::deserialize → serialize`. Same input must produce
    // the same JSON value on both sides, otherwise an unchanged JSONL
    // would still trigger a spurious regen.
    let user_args = serde_json::json!({
        "generator": "box",
        "half_extents": [0.5, 0.5, 0.5],
    });

    // Init-side: parse + re-serialize (mirroring what `serde_json::to_value`
    // on the deserialised component yields).
    let init_component: crate::assets::ProceduralMesh =
        serde_json::from_value(user_args.clone()).unwrap();
    let init_norm = serde_json::to_value(&init_component).unwrap();

    // Reload-side: parse user args → component → re-serialize.
    let reload_component: crate::assets::ProceduralMesh =
        serde_json::from_value(user_args.clone()).unwrap();
    let reload_norm = serde_json::to_value(&reload_component).unwrap();

    assert_eq!(init_norm, reload_norm);
}

#[test]
fn procedural_mesh_args_diff_detects_real_changes() {
    // A meaningful arg change must produce a distinct normalised value so
    // the diff fires.
    let v1: crate::assets::ProceduralMesh = serde_json::from_value(serde_json::json!({
        "generator": "box",
        "half_extents": [0.5, 0.5, 0.5],
    }))
    .unwrap();
    let v2: crate::assets::ProceduralMesh = serde_json::from_value(serde_json::json!({
        "generator": "box",
        "half_extents": [1.0, 1.0, 1.0],
    }))
    .unwrap();
    let n1 = serde_json::to_value(&v1).unwrap();
    let n2 = serde_json::to_value(&v2).unwrap();
    assert_ne!(n1, n2);
}

#[test]
fn procedural_mesh_reload_result_default_is_all_zero() {
    let r = ProceduralMeshReloadResult::default();
    assert_eq!(r.regenerated, 0);
    assert_eq!(r.unchanged, 0);
    assert_eq!(r.failed, 0);
}

#[test]
fn state_with_only_procedural_meshes_still_spawns_a_watcher() {
    // ProceduralMesh entries on their own (no textures, no LUTs, no meshes)
    // need to keep the watcher alive: their trigger is the `PENDING_WORLD`
    // flag flipped from the world.jsonl watcher. Without a world_jsonl_path
    // there is nothing to subscribe to, so we declare one here.
    let world_path = format!(
        "{}/asset_hot_reload_proc_only_world_{}.jsonl",
        std::env::temp_dir().display(),
        std::process::id()
    );
    let mut proc = ProceduralMeshSourceMap::new();
    proc.entries.push(ProceduralMeshSourceEntry {
        name: "box_mesh".to_string(),
        args: serde_json::from_value(serde_json::json!({"generator": "box"})).unwrap(),
        draw_indices: vec![0],
    });
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        procedural_meshes: proc,
        world_jsonl_path: Some(world_path),
        ..Default::default()
    });
    assert_eq!(state.procedural_meshes.len(), 1);
    assert_eq!(state.procedural_meshes.entries[0].name, "box_mesh");
}

#[test]
fn shader_stage_source_map_round_trips_empty() {
    let m = ShaderStageSourceMap::new();
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn shader_stage_source_map_collects_unique_parent_dirs() {
    use crate::assets::shader::ShaderKind;
    let mut m = ShaderStageSourceMap::new();
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Vertex,
        resolved_path: "assets/shaders/default.metal".to_string(),
    });
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Fragment,
        resolved_path: "assets/shaders/default.metal".to_string(),
    });
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::VertexInstanced,
        resolved_path: "assets/other/custom.metal".to_string(),
    });
    let dirs = m.watch_dirs();
    assert_eq!(dirs.len(), 2);
    assert!(dirs.iter().any(|p| p.ends_with("assets/shaders")));
    assert!(dirs.iter().any(|p| p.ends_with("assets/other")));
}

#[test]
fn shader_stage_source_map_skips_bare_filenames_in_watch_dirs() {
    // A bare filename has no parent directory; the watcher would try to
    // subscribe to "" which notify rejects. The debug-WS `reload-assets`
    // command still works for these.
    use crate::assets::shader::ShaderKind;
    let mut m = ShaderStageSourceMap::new();
    m.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Vertex,
        resolved_path: "standalone.metal".to_string(),
    });
    assert!(m.watch_dirs().is_empty());
}

#[test]
fn metal_extension_is_an_asset_event() {
    // World-loaded Shader stage sources travel through the same watcher
    // as the texture / mesh paths; the closure routes them to the
    // shader-stage flag rather than the texture-decode batch.
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/scene.metal"));
    assert!(is_asset_event(&evt));
}

#[test]
fn hlsl_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/scene_vert.hlsl"));
    assert!(is_asset_event(&evt));
}

#[test]
fn glsl_extension_is_an_asset_event() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/scene.glsl"));
    assert!(is_asset_event(&evt));
}

#[test]
fn shader_extension_matches_case_insensitively() {
    assert!(is_shader_extension("metal"));
    assert!(is_shader_extension("Metal"));
    assert!(is_shader_extension("METAL"));
    assert!(is_shader_extension("hlsl"));
    assert!(is_shader_extension("glsl"));
    assert!(!is_shader_extension("png"));
    assert!(!is_shader_extension("glb"));
}

fn modified(path: &str) -> Event {
    Event::new(EventKind::Modify(notify::event::ModifyKind::Any)).add_path(PathBuf::from(path))
}

// Each source kind routes to the narrowest pass that can serve it, so a
// dialogue save never pays for a texture decode and a shader save never
// re-reads the world.
#[test]
fn each_extension_routes_to_its_reload_pass() {
    for (path, expected) in [
        ("/tmp/lit.metal", ReloadKind::ShaderStages),
        ("/tmp/lit.hlsl", ReloadKind::ShaderStages),
        ("/tmp/lit.glsl", ReloadKind::ShaderStages),
        ("/tmp/LIT.METAL", ReloadKind::ShaderStages),
        ("/tmp/world.jsonl", ReloadKind::World),
        ("/tmp/WORLD.JSONL", ReloadKind::World),
        ("/tmp/intro.md", ReloadKind::Stories),
        ("/tmp/INTRO.MD", ReloadKind::Stories),
        ("/tmp/model.glb", ReloadKind::Assets),
        ("/tmp/albedo.png", ReloadKind::Assets),
        ("/tmp/studio.hdr", ReloadKind::Assets),
        ("/tmp/grade.cube", ReloadKind::Assets),
        ("/tmp/buffer.bin", ReloadKind::Assets),
    ] {
        assert_eq!(
            classify_event(&modified(path)),
            Some(expected),
            "{path} routes to {expected:?}"
        );
    }
}

// A change nothing is sourced from kicks no pass at all -- neither does an
// access event on a file that would otherwise be relevant.
#[test]
fn an_irrelevant_change_routes_nowhere() {
    assert_eq!(classify_event(&modified("/tmp/notes.txt")), None);
    let accessed = Event::new(EventKind::Access(notify::event::AccessKind::Read))
        .add_path(PathBuf::from("/tmp/model.glb"));
    assert_eq!(classify_event(&accessed), None);
}

// notify reports renames and temp sidecars as multi-path events; the shader
// pass wins over the broader asset pass whichever slot the shader lands in, so
// an editor's atomic save still recompiles rather than re-decoding textures.
#[test]
fn a_shader_among_several_paths_still_routes_to_the_shader_pass() {
    let leading = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/lit.metal"))
        .add_path(PathBuf::from("/tmp/albedo.png"));
    let trailing = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/albedo.png"))
        .add_path(PathBuf::from("/tmp/lit.metal"));
    assert_eq!(classify_event(&leading), Some(ReloadKind::ShaderStages));
    assert_eq!(classify_event(&trailing), Some(ReloadKind::ShaderStages));
}

// A create or a remove is as much a reload trigger as a modify: an asset added
// or deleted beside its siblings changes what the world resolves to.
#[test]
fn creates_and_removes_route_like_modifies() {
    for kind in [
        EventKind::Create(notify::event::CreateKind::File),
        EventKind::Remove(notify::event::RemoveKind::File),
    ] {
        let evt = Event::new(kind).add_path(PathBuf::from("/tmp/model.glb"));
        assert_eq!(classify_event(&evt), Some(ReloadKind::Assets), "{kind:?}");
    }
}

#[test]
fn state_with_only_shader_stages_still_spawns_a_watcher() {
    // World loaded only via shader-stage edits (no textures, no
    // meshes, no LUTs, no IBL, no world.jsonl) still want the watcher
    // alive so `.metal` saves trigger the recompile pass.
    use crate::assets::shader::ShaderKind;
    let mut stages = ShaderStageSourceMap::new();
    stages.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Vertex,
        resolved_path: format!(
            "{}/asset_hot_reload_shader_only_{}.metal",
            std::env::temp_dir().display(),
            std::process::id()
        ),
    });
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        shader_stages: stages,
        ..Default::default()
    });
    assert_eq!(state.shader_stages.len(), 1);
    assert_eq!(state.shader_stages.entries[0].kind, ShaderKind::Vertex);
}

#[test]
fn shader_stage_reload_result_default_is_all_zero() {
    let r = ShaderStageReloadResult::default();
    assert_eq!(r.recompiled, 0);
    assert_eq!(r.failed, 0);
    assert!(!r.pipelines_rebuilt);
}

#[test]
fn reload_shader_stages_on_empty_map_is_a_no_op() {
    // The helper must short-circuit before touching the backend on a
    // world with no captured Shader sources (e.g. the GLSL-only
    // path on Vulkan, or a world that pre-dated the capture). The
    // default backend trait impl errors on
    // `update_world_shader_pipelines`; an empty map must not hit it.
    struct DummyBackend;
    impl crate::gfx::scene_flow::SceneControl for DummyBackend {
        fn update_visibility(&mut self, _: usize, _: bool) {}
        fn set_fade(&mut self, _: f32) {}
    }
    impl crate::gfx::backend::RenderBackend for DummyBackend {
        fn window_closed(&mut self) -> bool {
            false
        }
        fn capture_cursor(&mut self) {}
        fn take_input(&mut self) -> crate::gfx::input::RenderInput {
            crate::gfx::input::RenderInput::default()
        }
        fn wait_idle(&self) {}
        fn draw_frame(&mut self, _: crate::gfx::backend::FrameParams<'_>) -> Result<(), String> {
            Ok(())
        }
        fn update_view(&mut self, _: [[f32; 4]; 4]) {}
        fn update_model(&mut self, _: usize, _: [[f32; 4]; 4]) {}
        fn retire_draw_object(&mut self, _: usize) {}
        fn upload_skinned(
            &mut self,
            _: &[crate::gfx::mesh_payload::SkinnedVertex],
            _: &[u16],
            _: Vec<crate::gfx::render_types::SkinnedDrawObject>,
            _: &[u8],
            _: &[u8],
            _: &[u8],
        ) -> Result<(), String> {
            Ok(())
        }
        fn update_skinned_pose(&mut self, _: usize, _: &[[[f32; 4]; 4]]) {}
        fn evict_texture_slot(&mut self, _: usize) -> Result<(), String> {
            Ok(())
        }
        fn update_texture_slot(
            &mut self,
            _: usize,
            _: &concinnity_core::build::texture::TextureImage,
        ) -> Result<(), String> {
            Ok(())
        }
        fn evict_mesh(&mut self, _: usize, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn upload_mesh(
            &mut self,
            _: usize,
            _: &[crate::gfx::mesh_payload::Vertex],
            _: &[u16],
            _: u64,
        ) -> Result<(), String> {
            Ok(())
        }
        fn setup_chunk_streaming(
            &mut self,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
        ) -> Result<(), String> {
            Ok(())
        }
        fn add_chunk_mesh(
            &mut self,
            _: crate::gfx::backend::ChunkMesh<'_>,
        ) -> Result<usize, String> {
            Ok(0)
        }
        fn remove_chunk_mesh(&mut self, _: usize, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn set_chunk_model(&mut self, _: usize, _: [[f32; 4]; 4]) -> Result<(), String> {
            Ok(())
        }
    }

    let map = ShaderStageSourceMap::new();
    let mut backend = DummyBackend;
    let r = reload_shader_stages(&map, &mut backend);
    assert_eq!(r.recompiled, 0);
    assert_eq!(r.failed, 0);
    assert!(!r.pipelines_rebuilt);
}

#[test]
fn apply_skinned_layouts_leaves_entries_without_a_matching_layout_alone() {
    // A backend that returned layouts for only a subset of slots (e.g.
    // a future partial-rebuild path) should still leave the other
    // entries' captured state untouched. The Metal `rebuild_skinned_geometry`
    // returns a layout for every slot, but this guard keeps the
    // contract robust to backend variation.
    let mut entries = vec![SkinnedMeshSourceEntry {
        source: "a.glb".to_string(),
        skin_index: 0,
        skinned_index: 7,
        vertex_base: 42,
        vertex_count: 12,
        index_count: 36,
        joint_count: 2,
    }];
    let layouts = vec![crate::gfx::backend::SkinnedSlotLayout {
        skinned_index: 0,
        vertex_base: 0,
        vertex_count: 99,
        index_count: 99,
    }];
    apply_skinned_layouts_to_entries(&mut entries, &layouts);
    assert_eq!(entries[0].vertex_base, 42);
    assert_eq!(entries[0].vertex_count, 12);
    assert_eq!(entries[0].index_count, 36);
}

// Shared fixtures for the decode / poll / pass tests below.

// A minimal RenderBackend that records every hot-reload dispatch so tests can
// assert which trait calls a pass made. Failure toggles let the error branches
// fire without a real GPU.
#[derive(Default)]
struct RecordingBackend {
    texture_updates: Vec<(usize, u32, u32)>,
    lut_updates: Vec<u32>,
    mesh_updates: Vec<usize>,
    static_rebuild_change_counts: Vec<usize>,
    skinned_updates: Vec<usize>,
    skinned_rebuild_change_counts: Vec<usize>,
    skeleton_updates: Vec<(usize, usize)>,
    env_updates: usize,
    fog_updates: usize,
    // Overrides for `draw_geometry_size`; absent draws report `None` like the
    // trait default.
    geometry_sizes: std::collections::HashMap<usize, (usize, usize)>,
    // Overrides for `draw_lod_index_counts`; absent draws report `None` like
    // the trait default.
    lod_counts: std::collections::HashMap<usize, Vec<usize>>,
    // Layouts `rebuild_skinned_geometry` hands back on success, stored as
    // (skinned_index, vertex_base, vertex_count, index_count).
    skinned_layouts: Vec<(usize, u16, usize, usize)>,
    fail_texture_updates: bool,
    fail_lut_updates: bool,
    fail_mesh_updates: bool,
    fail_static_rebuild: bool,
    fail_skinned_updates: bool,
    fail_skinned_rebuild: bool,
    fail_skeleton_update: bool,
    fail_env_updates: bool,
}

impl crate::gfx::scene_flow::SceneControl for RecordingBackend {
    fn update_visibility(&mut self, _: usize, _: bool) {}
    fn set_fade(&mut self, _: f32) {}
}

impl crate::gfx::backend::RenderBackend for RecordingBackend {
    fn window_closed(&mut self) -> bool {
        false
    }
    fn capture_cursor(&mut self) {}
    fn take_input(&mut self) -> crate::gfx::input::RenderInput {
        crate::gfx::input::RenderInput::default()
    }
    fn wait_idle(&self) {}
    fn draw_frame(&mut self, _: crate::gfx::backend::FrameParams<'_>) -> Result<(), String> {
        Ok(())
    }
    fn update_view(&mut self, _: [[f32; 4]; 4]) {}
    fn update_model(&mut self, _: usize, _: [[f32; 4]; 4]) {}
    fn retire_draw_object(&mut self, _: usize) {}
    fn upload_skinned(
        &mut self,
        _: &[crate::gfx::mesh_payload::SkinnedVertex],
        _: &[u16],
        _: Vec<crate::gfx::render_types::SkinnedDrawObject>,
        _: &[u8],
        _: &[u8],
        _: &[u8],
    ) -> Result<(), String> {
        Ok(())
    }
    fn update_skinned_pose(&mut self, _: usize, _: &[[[f32; 4]; 4]]) {}
    fn evict_texture_slot(&mut self, _: usize) -> Result<(), String> {
        Ok(())
    }
    fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &concinnity_core::build::texture::TextureImage,
    ) -> Result<(), String> {
        self.texture_updates
            .push((slot, image.width(), image.height()));
        if self.fail_texture_updates {
            return Err("texture update rejected".to_string());
        }
        Ok(())
    }
    fn evict_mesh(&mut self, _: usize, _: u64) -> Result<(), String> {
        Ok(())
    }
    fn upload_mesh(
        &mut self,
        _: usize,
        _: &[crate::gfx::mesh_payload::Vertex],
        _: &[u16],
        _: u64,
    ) -> Result<(), String> {
        Ok(())
    }
    fn setup_chunk_streaming(
        &mut self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<(), String> {
        Ok(())
    }
    fn add_chunk_mesh(&mut self, _: crate::gfx::backend::ChunkMesh<'_>) -> Result<usize, String> {
        Ok(0)
    }
    fn remove_chunk_mesh(&mut self, _: usize, _: u64) -> Result<(), String> {
        Ok(())
    }
    fn set_chunk_model(&mut self, _: usize, _: [[f32; 4]; 4]) -> Result<(), String> {
        Ok(())
    }

    fn update_color_lut(&mut self, size: u32, _: &[u8]) -> Result<(), String> {
        self.lut_updates.push(size);
        if self.fail_lut_updates {
            return Err("lut update rejected".to_string());
        }
        Ok(())
    }
    fn draw_geometry_size(&self, draw_idx: usize) -> Option<(usize, usize)> {
        self.geometry_sizes.get(&draw_idx).copied()
    }
    fn draw_lod_index_counts(&self, draw_idx: usize) -> Option<Vec<usize>> {
        self.lod_counts.get(&draw_idx).cloned()
    }
    fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        _: &[crate::gfx::mesh_payload::Vertex],
        _: &[u16],
        _: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        self.mesh_updates.push(draw_idx);
        if self.fail_mesh_updates {
            return Err("mesh update rejected".to_string());
        }
        Ok(())
    }
    fn rebuild_static_geometry(
        &mut self,
        changes: Vec<crate::gfx::backend::DrawGeometryUpdate>,
    ) -> Result<(), String> {
        self.static_rebuild_change_counts.push(changes.len());
        if self.fail_static_rebuild {
            return Err("static rebuild rejected".to_string());
        }
        Ok(())
    }
    fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        _: u16,
        _: &[crate::gfx::mesh_payload::SkinnedVertex],
        _: &[u16],
    ) -> Result<(), String> {
        self.skinned_updates.push(skinned_index);
        if self.fail_skinned_updates {
            return Err("skinned update rejected".to_string());
        }
        Ok(())
    }
    fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<crate::gfx::backend::SkinnedDrawGeometryUpdate>,
    ) -> Result<Vec<crate::gfx::backend::SkinnedSlotLayout>, String> {
        self.skinned_rebuild_change_counts.push(changes.len());
        if self.fail_skinned_rebuild {
            return Err("skinned rebuild rejected".to_string());
        }
        Ok(self
            .skinned_layouts
            .iter()
            .map(|&(skinned_index, vertex_base, vertex_count, index_count)| {
                crate::gfx::backend::SkinnedSlotLayout {
                    skinned_index,
                    vertex_base,
                    vertex_count,
                    index_count,
                }
            })
            .collect())
    }
    fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        self.skeleton_updates.push((skinned_index, new_joint_count));
        if self.fail_skeleton_update {
            return Err("skeleton update rejected".to_string());
        }
        Ok(())
    }
    fn update_environment_map(&mut self, _: &[u8]) -> Result<(), String> {
        self.env_updates += 1;
        if self.fail_env_updates {
            return Err("environment map update rejected".to_string());
        }
        Ok(())
    }
    fn update_fog_settings(&mut self, _: Option<crate::gfx::volumetric_fog::FogSettings>) {
        self.fog_updates += 1;
    }
}

// CRC-32 (IEEE) for the hand-rolled PNG chunks below. Bitwise so the test
// fixture needs no image or checksum dependency.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = kind.to_vec();
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    out
}

// Write a valid 1x1 RGBA8 PNG using a stored (uncompressed) deflate block so
// the texture-decode path has a real file to chew on without an encoder dep.
fn write_tiny_png(path: &std::path::Path) {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&1u32.to_be_bytes());
    ihdr.extend_from_slice(&1u32.to_be_bytes());
    // 8-bit RGBA, deflate, adaptive filtering, no interlace.
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    // One scanline: filter byte 0 + a single RGBA pixel.
    let raw = [0u8, 10, 20, 30, 255];
    let mut idat = vec![0x78, 0x01, 0x01];
    idat.extend_from_slice(&(raw.len() as u16).to_le_bytes());
    idat.extend_from_slice(&(!(raw.len() as u16)).to_le_bytes());
    idat.extend_from_slice(&raw);
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    idat.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    png.extend(png_chunk(b"IHDR", &ihdr));
    png.extend(png_chunk(b"IDAT", &idat));
    png.extend(png_chunk(b"IEND", &[]));
    std::fs::write(path, png).unwrap();
}

// Write a 2x2x2 identity-ish Adobe Cube LUT (8 data lines).
fn write_tiny_cube(path: &std::path::Path) {
    let text = "LUT_3D_SIZE 2\n\
                0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
    std::fs::write(path, text).unwrap();
}

fn zero_vertex() -> crate::gfx::mesh_payload::Vertex {
    crate::gfx::mesh_payload::Vertex {
        pos: [0.0; 3],
        normal: [0.0; 3],
        tangent: [0.0; 3],
        color: [0.0; 3],
        uv: [0.0; 2],
    }
}

fn zero_skinned_vertex() -> crate::gfx::mesh_payload::SkinnedVertex {
    crate::gfx::mesh_payload::SkinnedVertex {
        pos: [0.0; 3],
        normal: [0.0; 3],
        tangent: [0.0; 3],
        color: [0.0; 3],
        uv: [0.0; 2],
        joints: [0; 4],
        weights: [0.0; 4],
    }
}

fn joint_def(name: &str) -> crate::assets::JointDef {
    crate::assets::JointDef {
        name: name.to_string(),
        parent: -1,
        translation: [0.0; 3],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    }
}

// Push a ready-made decode batch into the state's in-flight slot as if a
// worker had just finished, so `poll_pending_assets` has something to drain.
fn inject_batch(state: &AssetHotReloadState, batch: DecodedAssetBatch) {
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(batch).unwrap();
    *state.asset_batch_inflight.lock().unwrap() = Some(rx);
}

// decode_asset_batch

#[test]
fn decode_asset_batch_decodes_a_png_texture() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("albedo.png");
    write_tiny_png(&png);
    let entry = TextureSourceEntry {
        source: png.to_string_lossy().into_owned(),
        image_index: 0,
        slot: 3,
    };
    let batch = decode_asset_batch(vec![entry], None, Vec::new(), Vec::new());
    assert_eq!(batch.decode_failures, 0);
    assert_eq!(batch.textures.len(), 1);
    let tex = &batch.textures[0];
    assert_eq!(tex.slot, 3);
    assert_eq!((tex.width, tex.height), (1, 1));
    assert_eq!(tex.pixels, vec![10, 20, 30, 255]);
}

#[test]
fn decode_asset_batch_counts_a_missing_texture_as_a_failure() {
    let dir = tempfile::tempdir().unwrap();
    let entry = TextureSourceEntry {
        source: dir
            .path()
            .join("never_written.png")
            .to_string_lossy()
            .into_owned(),
        image_index: 0,
        slot: 0,
    };
    let batch = decode_asset_batch(vec![entry], None, Vec::new(), Vec::new());
    assert!(batch.textures.is_empty());
    assert_eq!(batch.decode_failures, 1);
}

#[test]
fn decode_asset_batch_mixes_successes_and_failures() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("ok.png");
    write_tiny_png(&png);
    let good = TextureSourceEntry {
        source: png.to_string_lossy().into_owned(),
        image_index: 0,
        slot: 1,
    };
    let bad = TextureSourceEntry {
        source: dir.path().join("gone.png").to_string_lossy().into_owned(),
        image_index: 0,
        slot: 2,
    };
    let batch = decode_asset_batch(vec![good, bad], None, Vec::new(), Vec::new());
    assert_eq!(batch.textures.len(), 1);
    assert_eq!(batch.decode_failures, 1);
}

#[test]
fn decode_asset_batch_counts_an_unparseable_glb_texture_as_a_failure() {
    // A `.glb`-suffixed source routes through the shared parse cache; garbage
    // bytes must fail the parse and count, not panic.
    let dir = tempfile::tempdir().unwrap();
    let glb = dir.path().join("broken.glb");
    std::fs::write(&glb, b"not a glb at all").unwrap();
    let entry = TextureSourceEntry {
        source: glb.to_string_lossy().into_owned(),
        image_index: 0,
        slot: 0,
    };
    let batch = decode_asset_batch(vec![entry], None, Vec::new(), Vec::new());
    assert!(batch.textures.is_empty());
    assert_eq!(batch.decode_failures, 1);
}

#[test]
fn decode_asset_batch_decodes_a_cube_lut() {
    let dir = tempfile::tempdir().unwrap();
    let cube = dir.path().join("grade.cube");
    write_tiny_cube(&cube);
    let lut = ColorLutSource {
        resolved_path: cube.to_string_lossy().into_owned(),
    };
    let batch = decode_asset_batch(Vec::new(), Some(lut), Vec::new(), Vec::new());
    assert_eq!(batch.decode_failures, 0);
    let lut = batch.color_lut.expect("decoded lut");
    assert_eq!(lut.size, 2);
    assert_eq!(lut.data.len(), 2 * 2 * 2 * 4);
}

#[test]
fn decode_asset_batch_counts_a_missing_lut_as_a_failure() {
    let dir = tempfile::tempdir().unwrap();
    let lut = ColorLutSource {
        resolved_path: dir.path().join("gone.cube").to_string_lossy().into_owned(),
    };
    let batch = decode_asset_batch(Vec::new(), Some(lut), Vec::new(), Vec::new());
    assert!(batch.color_lut.is_none());
    assert_eq!(batch.decode_failures, 1);
}

#[test]
fn decode_asset_batch_counts_a_missing_mesh_source_as_a_failure() {
    let dir = tempfile::tempdir().unwrap();
    let entry = MeshSourceEntry {
        source: dir.path().join("gone.glb").to_string_lossy().into_owned(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0],
    };
    let batch = decode_asset_batch(Vec::new(), None, vec![entry], Vec::new());
    assert!(batch.meshes.is_empty());
    assert_eq!(batch.decode_failures, 1);
}

#[test]
fn decode_asset_batch_counts_a_missing_skinned_source_as_a_failure() {
    let dir = tempfile::tempdir().unwrap();
    let entry = SkinnedMeshSourceEntry {
        source: dir.path().join("gone.glb").to_string_lossy().into_owned(),
        skin_index: 0,
        skinned_index: 0,
        vertex_base: 0,
        vertex_count: 8,
        index_count: 24,
        joint_count: 2,
    };
    let batch = decode_asset_batch(Vec::new(), None, Vec::new(), vec![entry]);
    assert!(batch.skinned_meshes.is_empty());
    assert_eq!(batch.decode_failures, 1);
}

// reload_assets (worker spawn plumbing)

#[test]
fn reload_assets_with_no_sources_spawns_nothing() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    reload_assets(&state);
    assert!(state.asset_batch_inflight.lock().unwrap().is_none());
    assert!(state.env_map_inflight.lock().unwrap().is_none());
}

#[test]
fn reload_assets_skips_the_spawn_while_a_batch_is_in_flight() {
    let mut map = TextureSourceMap::new();
    map.push_texture("standalone.png".to_string(), 0, 0);
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        map,
        ..Default::default()
    });
    // Simulate a still-running worker: a receiver with a live sender and no
    // payload yet.
    let (_tx, rx) = std::sync::mpsc::channel::<DecodedAssetBatch>();
    *state.asset_batch_inflight.lock().unwrap() = Some(rx);
    reload_assets(&state);
    // Our receiver is still in the slot (a fresh spawn would have replaced it
    // with one whose worker sends a decode-failure batch).
    let slot = state.asset_batch_inflight.lock().unwrap();
    assert!(matches!(
        slot.as_ref().unwrap().try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
}

#[test]
fn reload_assets_spawns_a_decode_worker_for_a_lut_source() {
    let dir = tempfile::tempdir().unwrap();
    let cube = dir.path().join("grade.cube");
    write_tiny_cube(&cube);
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        color_lut: Some(ColorLutSource {
            resolved_path: cube.to_string_lossy().into_owned(),
        }),
        ..Default::default()
    });
    reload_assets(&state);
    let rx = state
        .asset_batch_inflight
        .lock()
        .unwrap()
        .take()
        .expect("worker scheduled");
    // The worker terminates after decoding the single small LUT, so a
    // blocking receive completes without wall-clock timing assumptions.
    let batch = rx.recv().expect("worker sent a batch");
    assert_eq!(batch.color_lut.expect("decoded lut").size, 2);
    // No EnvironmentMap declared: the envmap slot stays untouched.
    assert!(state.env_map_inflight.lock().unwrap().is_none());
}

// poll_pending_assets

#[test]
fn poll_pending_assets_with_nothing_in_flight_returns_false() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let mut backend = RecordingBackend::default();
    assert!(!poll_pending_assets(&mut state, &mut backend));
}

#[test]
fn poll_pending_assets_keeps_waiting_while_the_worker_runs() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (_tx, rx) = std::sync::mpsc::channel::<DecodedAssetBatch>();
    *state.asset_batch_inflight.lock().unwrap() = Some(rx);
    let mut backend = RecordingBackend::default();
    assert!(!poll_pending_assets(&mut state, &mut backend));
    // Still parked for the next frame.
    assert!(state.asset_batch_inflight.lock().unwrap().is_some());
}

#[test]
fn poll_pending_assets_clears_the_slot_when_the_worker_disconnects() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (tx, rx) = std::sync::mpsc::channel::<DecodedAssetBatch>();
    drop(tx);
    *state.asset_batch_inflight.lock().unwrap() = Some(rx);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert!(state.asset_batch_inflight.lock().unwrap().is_none());
}

#[test]
fn poll_pending_assets_reloads_every_texture_through_the_shared_pool() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    // Albedo and normal maps share one pool, so both reload through
    // `update_texture_slot` at their own pool slot (no separate normal path).
    let batch = DecodedAssetBatch {
        textures: vec![
            DecodedTexture {
                slot: 2,
                width: 4,
                height: 4,
                pixels: vec![0; 64],
                source: "a.png".to_string(),
            },
            DecodedTexture {
                slot: 5,
                width: 8,
                height: 8,
                pixels: vec![0; 256],
                source: "n.png".to_string(),
            },
        ],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.texture_updates, vec![(2, 4, 4), (5, 8, 8)]);
    assert!(state.asset_batch_inflight.lock().unwrap().is_none());
}

#[test]
fn poll_pending_assets_survives_a_backend_texture_rejection() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let batch = DecodedAssetBatch {
        textures: vec![DecodedTexture {
            slot: 0,
            width: 1,
            height: 1,
            pixels: vec![0; 4],
            source: "a.png".to_string(),
        }],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend {
        fail_texture_updates: true,
        ..Default::default()
    };
    // The failure is tallied and logged; the poll still reports consumption.
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.texture_updates.len(), 1);
}

#[test]
fn poll_pending_assets_applies_a_color_lut() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let batch = DecodedAssetBatch {
        color_lut: Some(DecodedColorLut {
            size: 2,
            data: vec![0; 32],
            source: "grade.cube".to_string(),
        }),
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.lut_updates, vec![2]);
}

fn one_mesh_state(draw_indices: Vec<usize>) -> AssetHotReloadState {
    let mut meshes = MeshSourceMap::new();
    meshes.entries.push(MeshSourceEntry {
        source: "model.glb".to_string(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices,
    });
    AssetHotReloadState::from_sources(HotReloadSources {
        meshes,
        ..Default::default()
    })
}

fn decoded_mesh(entry_idx: usize, vertex_count: usize) -> DecodedMesh {
    DecodedMesh {
        entry_idx,
        vertices: vec![zero_vertex(); vertex_count],
        indices: vec![0; vertex_count * 3],
        lod_alternates: Vec::new(),
    }
}

#[test]
fn poll_pending_assets_updates_meshes_in_place_per_draw_slot() {
    let mut state = one_mesh_state(vec![2, 5]);
    let batch = DecodedAssetBatch {
        meshes: vec![decoded_mesh(0, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    // Default draw_geometry_size (None) means no size change is detectable:
    // the in-place path fires for every draw slot of the entry.
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.mesh_updates, vec![2, 5]);
    assert!(backend.static_rebuild_change_counts.is_empty());
}

#[test]
fn poll_pending_assets_rebuilds_a_size_changed_mesh() {
    let mut state = one_mesh_state(vec![2, 5]);
    let batch = DecodedAssetBatch {
        meshes: vec![decoded_mesh(0, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    // The backend reports different init-time counts for draw 2, so the
    // whole entry (both slots) is queued into one rebuild call.
    backend.geometry_sizes.insert(2, (999, 999));
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert!(backend.mesh_updates.is_empty());
    assert_eq!(backend.static_rebuild_change_counts, vec![2]);
}

#[test]
fn poll_pending_assets_skips_an_out_of_range_mesh_entry() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let batch = DecodedAssetBatch {
        meshes: vec![decoded_mesh(7, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert!(backend.mesh_updates.is_empty());
    assert!(backend.static_rebuild_change_counts.is_empty());
}

fn one_skinned_state(entry: SkinnedMeshSourceEntry) -> AssetHotReloadState {
    let mut skinned = SkinnedMeshSourceMap::new();
    skinned.entries.push(entry);
    AssetHotReloadState::from_sources(HotReloadSources {
        skinned_meshes: skinned,
        ..Default::default()
    })
}

fn skinned_entry() -> SkinnedMeshSourceEntry {
    SkinnedMeshSourceEntry {
        source: "rig.glb".to_string(),
        skin_index: 0,
        skinned_index: 4,
        vertex_base: 0,
        vertex_count: 2,
        index_count: 3,
        joint_count: 2,
    }
}

fn decoded_skinned(entry_idx: usize, vertex_count: usize, joints: usize) -> DecodedSkinnedMesh {
    DecodedSkinnedMesh {
        entry_idx,
        vertices: vec![zero_skinned_vertex(); vertex_count],
        indices: vec![0; 3],
        skeleton: (0..joints).map(|i| joint_def(&format!("j{i}"))).collect(),
    }
}

#[test]
fn poll_pending_assets_updates_skinned_meshes_in_place() {
    let mut state = one_skinned_state(skinned_entry());
    let batch = DecodedAssetBatch {
        skinned_meshes: vec![decoded_skinned(0, 2, 2)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.skinned_updates, vec![4]);
    assert!(backend.skinned_rebuild_change_counts.is_empty());
    assert!(state.pending_skeleton_updates.is_empty());
}

#[test]
fn poll_pending_assets_rebuilds_size_changed_skinned_and_refreshes_the_layout() {
    let mut state = one_skinned_state(skinned_entry());
    let batch = DecodedAssetBatch {
        skinned_meshes: vec![decoded_skinned(0, 5, 2)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    backend.skinned_layouts.push((4, 7, 5, 3));
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert!(backend.skinned_updates.is_empty());
    assert_eq!(backend.skinned_rebuild_change_counts, vec![1]);
    let entry = &state.skinned_meshes.entries[0];
    assert_eq!(entry.vertex_base, 7);
    assert_eq!(entry.vertex_count, 5);
    assert_eq!(entry.index_count, 3);
}

#[test]
fn poll_pending_assets_queues_a_skeleton_update_on_joint_count_change() {
    let mut state = one_skinned_state(skinned_entry());
    let batch = DecodedAssetBatch {
        skinned_meshes: vec![decoded_skinned(0, 2, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.skeleton_updates, vec![(4, 3)]);
    assert_eq!(state.skinned_meshes.entries[0].joint_count, 3);
    let drained = state.drain_pending_skeleton_updates();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].skinned_index, 4);
}

#[test]
fn poll_pending_assets_failed_skinned_rebuild_drops_queued_skeleton_updates() {
    let mut state = one_skinned_state(skinned_entry());
    // Both the joint count and the vertex count changed, so the rebuild path
    // fires with a skeleton update queued behind it.
    let batch = DecodedAssetBatch {
        skinned_meshes: vec![decoded_skinned(0, 5, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend {
        fail_skinned_rebuild: true,
        ..Default::default()
    };
    assert!(poll_pending_assets(&mut state, &mut backend));
    // The geometry kept its old shape, so the SkeletonPose refresh and the
    // joint-count write must both be discarded.
    assert!(state.pending_skeleton_updates.is_empty());
    assert!(backend.skeleton_updates.is_empty());
    assert_eq!(state.skinned_meshes.entries[0].joint_count, 2);
}

#[test]
fn poll_pending_assets_failed_joint_resize_keeps_the_old_count() {
    let mut state = one_skinned_state(skinned_entry());
    let batch = DecodedAssetBatch {
        skinned_meshes: vec![decoded_skinned(0, 2, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend {
        fail_skeleton_update: true,
        ..Default::default()
    };
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(state.skinned_meshes.entries[0].joint_count, 2);
    // The queued SkeletonPose update is pruned so it cannot desync from the
    // unchanged backend joint buffers.
    assert!(state.pending_skeleton_updates.is_empty());
}

// poll_pending_envmap

#[test]
fn poll_pending_envmap_with_nothing_in_flight_returns_false() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let mut backend = RecordingBackend::default();
    assert!(!poll_pending_envmap(&state, &mut backend));
    assert_eq!(backend.env_updates, 0);
}

#[test]
fn poll_pending_envmap_keeps_waiting_while_the_worker_runs() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (_tx, rx) = std::sync::mpsc::channel::<Result<Vec<u8>, String>>();
    *state.env_map_inflight.lock().unwrap() = Some(rx);
    let mut backend = RecordingBackend::default();
    assert!(!poll_pending_envmap(&state, &mut backend));
    assert!(state.env_map_inflight.lock().unwrap().is_some());
}

#[test]
fn poll_pending_envmap_applies_a_successful_payload() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(Ok(vec![1u8, 2, 3])).unwrap();
    *state.env_map_inflight.lock().unwrap() = Some(rx);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_envmap(&state, &mut backend));
    assert_eq!(backend.env_updates, 1);
    assert!(state.env_map_inflight.lock().unwrap().is_none());
}

#[test]
fn poll_pending_envmap_consumes_a_failed_convolution_without_a_backend_call() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(Err("bad hdr".to_string())).unwrap();
    *state.env_map_inflight.lock().unwrap() = Some(rx);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_envmap(&state, &mut backend));
    assert_eq!(backend.env_updates, 0);
    assert!(state.env_map_inflight.lock().unwrap().is_none());
}

#[test]
fn poll_pending_envmap_clears_the_slot_when_the_worker_disconnects() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (tx, rx) = std::sync::mpsc::channel::<Result<Vec<u8>, String>>();
    drop(tx);
    *state.env_map_inflight.lock().unwrap() = Some(rx);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_envmap(&state, &mut backend));
    assert_eq!(backend.env_updates, 0);
    assert!(state.env_map_inflight.lock().unwrap().is_none());
}

#[test]
fn poll_pending_envmap_survives_a_backend_rejection() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(Ok(vec![9u8; 4])).unwrap();
    *state.env_map_inflight.lock().unwrap() = Some(rx);
    let mut backend = RecordingBackend {
        fail_env_updates: true,
        ..Default::default()
    };
    // The backend rejected the payload; the poll still reports consumption and
    // clears the slot rather than retrying the same failed convolution.
    assert!(poll_pending_envmap(&state, &mut backend));
    assert_eq!(backend.env_updates, 1);
    assert!(state.env_map_inflight.lock().unwrap().is_none());
}

#[test]
fn reload_assets_spawns_an_envmap_worker() {
    let dir = tempfile::tempdir().unwrap();
    // A path that does not resolve to a valid HDR: the worker still spawns and
    // parks a receiver, then reports a decode failure on its own thread.
    let hdr = dir.path().join("sky.hdr");
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        environment_map: Some(EnvironmentMapSource {
            resolved_path: hdr.to_string_lossy().into_owned(),
            prefilter_face_size: 8,
            irradiance_face_size: 8,
            prefilter_samples: 4,
            prefilter_clamp: 4.0,
        }),
        ..Default::default()
    });
    reload_assets(&state);
    // The convolution worker was scheduled onto its own slot. Drain the parked
    // receiver so the worker thread completes before the test ends.
    let rx = state
        .env_map_inflight
        .lock()
        .unwrap()
        .take()
        .expect("envmap worker scheduled");
    let _ = rx.recv();
}

// poll_pending_assets: remaining apply branches

#[test]
fn poll_pending_assets_survives_a_color_lut_rejection() {
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let batch = DecodedAssetBatch {
        color_lut: Some(DecodedColorLut {
            size: 2,
            data: vec![0; 32],
            source: "grade.cube".to_string(),
        }),
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend {
        fail_lut_updates: true,
        ..Default::default()
    };
    // The failure is tallied and logged; the poll still reports consumption.
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.lut_updates, vec![2]);
}

#[test]
fn poll_pending_assets_rebuilds_on_a_changed_lod_breakdown() {
    let mut state = one_mesh_state(vec![0]);
    let batch = DecodedAssetBatch {
        meshes: vec![DecodedMesh {
            entry_idx: 0,
            vertices: vec![zero_vertex(); 3],
            indices: vec![0; 9],
            lod_alternates: vec![(10.0, vec![0u16; 6])],
        }],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    // The slot's live LOD1 carries 3 indices; the reload's carries 6, so the
    // entry is queued for a rebuild rather than an in-place update even though
    // the base geometry size is unchanged.
    backend.lod_counts.insert(0, vec![3]);
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert!(backend.mesh_updates.is_empty());
    assert_eq!(backend.static_rebuild_change_counts, vec![1]);
}

#[test]
fn poll_pending_assets_skips_an_out_of_range_skinned_entry() {
    let mut state = one_skinned_state(skinned_entry());
    let batch = DecodedAssetBatch {
        skinned_meshes: vec![decoded_skinned(9, 2, 2)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend::default();
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert!(backend.skinned_updates.is_empty());
    assert!(backend.skinned_rebuild_change_counts.is_empty());
}

#[test]
fn poll_pending_assets_survives_a_skinned_in_place_rejection() {
    let mut state = one_skinned_state(skinned_entry());
    // Same vertex / index / joint counts as the entry: the in-place update path
    // fires, and the backend rejects it.
    let batch = DecodedAssetBatch {
        skinned_meshes: vec![decoded_skinned(0, 2, 2)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend {
        fail_skinned_updates: true,
        ..Default::default()
    };
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert_eq!(backend.skinned_updates, vec![4]);
    assert!(backend.skinned_rebuild_change_counts.is_empty());
}

// reload_volumetric_fog

fn write_world_line(dir: &std::path::Path, line: &str) -> String {
    let path = dir.join("world.jsonl");
    std::fs::write(&path, format!("{line}\n")).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn reload_volumetric_fog_missing_file_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gone.jsonl");
    let mut last = None;
    let mut backend = RecordingBackend::default();
    let r = reload_volumetric_fog(path.to_str().unwrap(), &mut last, &mut backend);
    assert!(!r.updated);
    assert!(last.is_none());
    assert_eq!(backend.fog_updates, 0);
}

#[test]
fn reload_volumetric_fog_invalid_jsonl_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(dir.path(), "{ this is not json");
    let mut last = None;
    let mut backend = RecordingBackend::default();
    let r = reload_volumetric_fog(&path, &mut last, &mut backend);
    assert!(!r.updated);
    assert_eq!(backend.fog_updates, 0);
}

#[test]
fn reload_volumetric_fog_pushes_an_enabled_fog_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(
        dir.path(),
        r#"{"name":"fog","type":"VolumetricFog","args":{"enabled":true,"density":0.5}}"#,
    );
    let mut last = None;
    let mut backend = RecordingBackend::default();
    let r = reload_volumetric_fog(&path, &mut last, &mut backend);
    assert!(r.updated);
    assert!(last.is_some());
    assert_eq!(backend.fog_updates, 1);

    // Unchanged file: the dedupe swallows the second pass.
    let r = reload_volumetric_fog(&path, &mut last, &mut backend);
    assert!(!r.updated);
    assert_eq!(backend.fog_updates, 1);
}

#[test]
fn reload_volumetric_fog_disabling_pushes_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(
        dir.path(),
        r#"{"name":"fog","type":"VolumetricFog","args":{"enabled":true}}"#,
    );
    let mut last = None;
    let mut backend = RecordingBackend::default();
    assert!(reload_volumetric_fog(&path, &mut last, &mut backend).updated);
    assert!(last.is_some());

    let path = write_world_line(
        dir.path(),
        r#"{"name":"fog","type":"VolumetricFog","args":{"enabled":false}}"#,
    );
    assert!(reload_volumetric_fog(&path, &mut last, &mut backend).updated);
    assert!(last.is_none());
    assert_eq!(backend.fog_updates, 2);
}

#[test]
fn reload_volumetric_fog_removed_asset_pushes_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(
        dir.path(),
        r#"{"name":"fog","type":"VolumetricFog","args":{"enabled":true}}"#,
    );
    let mut last = None;
    let mut backend = RecordingBackend::default();
    assert!(reload_volumetric_fog(&path, &mut last, &mut backend).updated);

    // The fog line disappears from the world entirely.
    let path = write_world_line(
        dir.path(),
        r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
    );
    assert!(reload_volumetric_fog(&path, &mut last, &mut backend).updated);
    assert!(last.is_none());
}

#[test]
fn reload_volumetric_fog_bad_args_keep_the_previous_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(
        dir.path(),
        r#"{"name":"fog","type":"VolumetricFog","args":{"density":"oops"}}"#,
    );
    let mut last = None;
    let mut backend = RecordingBackend::default();
    let r = reload_volumetric_fog(&path, &mut last, &mut backend);
    assert!(!r.updated);
    assert!(last.is_none());
    assert_eq!(backend.fog_updates, 0);
}

// reload_procedural_meshes

fn normalised_box_args(half: f32) -> crate::assets::ProceduralMesh {
    serde_json::from_value(serde_json::json!({
        "generator": "box",
        "half_extents": [half, half, half],
    }))
    .unwrap()
}

fn one_proc_mesh_map(name: &str, args: crate::assets::ProceduralMesh) -> ProceduralMeshSourceMap {
    let mut map = ProceduralMeshSourceMap::new();
    map.entries.push(ProceduralMeshSourceEntry {
        name: name.to_string(),
        args,
        draw_indices: vec![0, 1],
    });
    map
}

fn box_world_line(name: &str, half: f32) -> String {
    format!(
        r#"{{"name":"{name}","type":"ProceduralMesh","args":{{"generator":"box","half_extents":[{half},{half},{half}]}}}}"#,
    )
}

#[test]
fn reload_procedural_meshes_with_an_empty_map_short_circuits() {
    let mut map = ProceduralMeshSourceMap::new();
    let mut backend = RecordingBackend::default();
    let r = reload_procedural_meshes("does_not_matter.jsonl", &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (0, 0, 0));
}

#[test]
fn reload_procedural_meshes_missing_file_regenerates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend::default();
    let r = reload_procedural_meshes(
        dir.path().join("gone.jsonl").to_str().unwrap(),
        &mut map,
        &mut backend,
    );
    assert_eq!((r.regenerated, r.unchanged, r.failed), (0, 0, 0));
    assert!(backend.mesh_updates.is_empty());
}

#[test]
fn reload_procedural_meshes_skips_unchanged_args() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(dir.path(), &box_world_line("box_mesh", 0.5));
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend::default();
    let r = reload_procedural_meshes(&path, &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (0, 1, 0));
    assert!(backend.mesh_updates.is_empty());
}

#[test]
fn reload_procedural_meshes_treats_a_missing_jsonl_entry_as_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(
        dir.path(),
        r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
    );
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend::default();
    let r = reload_procedural_meshes(&path, &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (0, 1, 0));
}

#[test]
fn reload_procedural_meshes_treats_unparseable_args_as_unchanged() {
    // A live-edited entry with args the generator cannot parse never reaches
    // the regen; the entry falls out of the new-args map and reads as
    // missing-from-jsonl.
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(
        dir.path(),
        r#"{"name":"box_mesh","type":"ProceduralMesh","args":{"generator":42}}"#,
    );
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend::default();
    let r = reload_procedural_meshes(&path, &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (0, 1, 0));
}

#[test]
fn reload_procedural_meshes_regenerates_changed_args_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(dir.path(), &box_world_line("box_mesh", 1.0));
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend::default();
    let r = reload_procedural_meshes(&path, &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (1, 0, 0));
    // In-place path (the default backend reports no size data): one update
    // per draw slot, and the captured args advance to the new snapshot.
    assert_eq!(backend.mesh_updates, vec![0, 1]);
    assert!(backend.static_rebuild_change_counts.is_empty());
    assert_eq!(map.entries[0].args, normalised_box_args(1.0));
}

#[test]
fn reload_procedural_meshes_keeps_args_when_the_backend_rejects_the_update() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(dir.path(), &box_world_line("box_mesh", 1.0));
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend {
        fail_mesh_updates: true,
        ..Default::default()
    };
    let r = reload_procedural_meshes(&path, &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (0, 0, 1));
    // The captured args stay at the pre-edit snapshot so the next reload
    // still sees the pending change.
    assert_eq!(map.entries[0].args, normalised_box_args(0.5));
}

#[test]
fn reload_procedural_meshes_rebuilds_on_size_change() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(dir.path(), &box_world_line("box_mesh", 1.0));
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend::default();
    backend.geometry_sizes.insert(0, (1, 1));
    let r = reload_procedural_meshes(&path, &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (1, 0, 0));
    assert!(backend.mesh_updates.is_empty());
    assert_eq!(backend.static_rebuild_change_counts, vec![2]);
    assert_eq!(map.entries[0].args, normalised_box_args(1.0));
}

#[test]
fn reload_procedural_meshes_failed_rebuild_keeps_captured_args() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(dir.path(), &box_world_line("box_mesh", 1.0));
    let mut map = one_proc_mesh_map("box_mesh", normalised_box_args(0.5));
    let mut backend = RecordingBackend {
        fail_static_rebuild: true,
        ..Default::default()
    };
    backend.geometry_sizes.insert(0, (1, 1));
    let r = reload_procedural_meshes(&path, &mut map, &mut backend);
    assert_eq!((r.regenerated, r.unchanged, r.failed), (0, 0, 1));
    assert_eq!(map.entries[0].args, normalised_box_args(0.5));
}

// reload_stories (edge cases beyond the round-trip test above)

#[test]
fn reload_stories_missing_world_file_returns_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut snapshots = std::collections::HashMap::new();
    let stories = reload_stories(
        dir.path().join("gone.jsonl").to_str().unwrap(),
        &mut snapshots,
    );
    assert!(stories.is_empty());
    assert!(snapshots.is_empty());
}

#[test]
fn reload_stories_ignores_worlds_without_stories() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_world_line(
        dir.path(),
        r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
    );
    let mut snapshots = std::collections::HashMap::new();
    assert!(reload_stories(&path, &mut snapshots).is_empty());
    assert!(snapshots.is_empty());
}

// reload_shader_stages

#[test]
fn reload_shader_stages_missing_source_counts_as_failed_without_a_rebuild() {
    use crate::assets::shader::ShaderKind;
    let dir = tempfile::tempdir().unwrap();
    let mut map = ShaderStageSourceMap::new();
    map.entries.push(ShaderStageSourceEntry {
        kind: ShaderKind::Vertex,
        resolved_path: dir
            .path()
            .join("zz_never_written_stage.metal")
            .to_string_lossy()
            .into_owned(),
    });
    // The RecordingBackend would record a pipeline rebuild; a compile failure
    // must abort the pass before the backend is touched.
    let mut backend = RecordingBackend::default();
    let r = reload_shader_stages(&map, &mut backend);
    assert_eq!(r.recompiled, 0);
    assert_eq!(r.failed, 1);
    assert!(!r.pipelines_rebuilt);
}

// AssetHotReloadState

#[test]
fn state_reload_flag_round_trips() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    assert!(!state.reload_requested());
    state
        .pending
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(state.reload_requested());
    state.clear_flag();
    assert!(!state.reload_requested());
}

#[test]
fn state_debug_format_summarises_the_catalogue() {
    let mut map = TextureSourceMap::new();
    map.push_texture("standalone.png".to_string(), 0, 0);
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        map,
        ..Default::default()
    });
    let dump = format!("{state:?}");
    assert!(dump.contains("AssetHotReloadState"));
    assert!(dump.contains("entries: 1"));
    assert!(dump.contains("pending: false"));
    assert!(dump.contains("env_map_inflight: false"));
    assert!(dump.contains("asset_batch_inflight: false"));
}

#[test]
fn fresh_state_starts_with_empty_story_snapshots() {
    let state = AssetHotReloadState::from_sources(HotReloadSources::default());
    assert!(state.story_snapshots.is_empty());
    assert!(state.world_jsonl_path.is_none());
}

// watcher helpers

#[test]
fn access_events_are_not_asset_events() {
    let evt = Event::new(EventKind::Access(notify::event::AccessKind::Any))
        .add_path(PathBuf::from("/tmp/a.png"));
    assert!(!is_asset_event(&evt));
}

#[test]
fn events_without_paths_are_not_asset_events() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any));
    assert!(!is_asset_event(&evt));
}

#[test]
fn extensionless_paths_are_not_asset_events() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/Makefile"));
    assert!(!is_asset_event(&evt));
}

#[test]
fn create_and_remove_events_count_as_asset_events() {
    let create = Event::new(EventKind::Create(notify::event::CreateKind::Any))
        .add_path(PathBuf::from("/tmp/a.png"));
    assert!(is_asset_event(&create));
    let remove = Event::new(EventKind::Remove(notify::event::RemoveKind::Any))
        .add_path(PathBuf::from("/tmp/a.png"));
    assert!(is_asset_event(&remove));
}

#[test]
fn uppercase_texture_extension_still_matches() {
    let evt = Event::new(EventKind::Modify(notify::event::ModifyKind::Any))
        .add_path(PathBuf::from("/tmp/ALBEDO.PNG"));
    assert!(is_asset_event(&evt));
}

#[test]
fn spawn_watcher_returns_none_when_no_directory_is_watchable() {
    let dir = tempfile::tempdir().unwrap();
    let mut meshes = MeshSourceMap::new();
    meshes.entries.push(MeshSourceEntry {
        source: dir
            .path()
            .join("no_such_subdir")
            .join("model.glb")
            .to_string_lossy()
            .into_owned(),
        primitive_index: 0,
        lod_levels: 1,
        lod_distances: Vec::new(),
        draw_indices: vec![0],
    });
    let sources = HotReloadSources {
        meshes,
        ..Default::default()
    };
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    assert!(spawn_watcher(&sources, flag).is_none());
}

#[test]
fn spawn_watcher_subscribes_to_an_existing_source_directory() {
    let dir = tempfile::tempdir().unwrap();
    let mut map = TextureSourceMap::new();
    map.push_texture(
        dir.path().join("albedo.png").to_string_lossy().into_owned(),
        0,
        0,
    );
    let sources = HotReloadSources {
        map,
        ..Default::default()
    };
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // No file is ever written into the watched directory, so the closure
    // never fires and the process-global pending flags stay untouched.
    assert!(spawn_watcher(&sources, flag).is_some());
}

#[test]
fn story_source_dirs_collects_unique_parents_from_story_imports() {
    let dir = tempfile::tempdir().unwrap();
    let world = dir.path().join("world.jsonl");
    let lines = [
        r#"{"name":"s1","type":"StoryImport","args":{"source":"stories/tale.md"}}"#,
        r#"{"name":"s2","type":"StoryImport","args":{"source":"bare.md"}}"#,
        r#"{"name":"s3","type":"StoryImport","args":{"source":"stories/other.md"}}"#,
        r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
        r#"{"name":"s4","type":"StoryImport","args":{}}"#,
        "not json at all",
    ];
    std::fs::write(&world, lines.join("\n")).unwrap();
    let dirs = story_source_dirs(world.to_str().unwrap());
    assert_eq!(dirs.len(), 2);
    assert!(dirs.contains(&PathBuf::from(".")));
    assert!(dirs.contains(&PathBuf::from("stories")));
}

#[test]
fn story_source_dirs_of_a_missing_world_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let dirs = story_source_dirs(dir.path().join("gone.jsonl").to_str().unwrap());
    assert!(dirs.is_empty());
}

// decode backend-rejection sub-branches (poll_pending_assets)
//
// The static-mesh in-place and rebuild error paths are driven purely by the
// RecordingBackend's failure toggles, so no real model file is needed. The
// ColorLut and in-place skinned rejection paths have no matching toggle on the
// shared RecordingBackend and are left uncovered here.

#[test]
fn poll_pending_assets_survives_an_in_place_mesh_rejection() {
    // Default draw_geometry_size (None) keeps the in-place path; the backend
    // rejects every update but the poll must tally the failure and still report
    // the batch consumed rather than panic.
    let mut state = one_mesh_state(vec![2, 5]);
    let batch = DecodedAssetBatch {
        meshes: vec![decoded_mesh(0, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend {
        fail_mesh_updates: true,
        ..Default::default()
    };
    assert!(poll_pending_assets(&mut state, &mut backend));
    // Both draw slots were attempted before the rejection was recorded.
    assert_eq!(backend.mesh_updates, vec![2, 5]);
    assert!(backend.static_rebuild_change_counts.is_empty());
    assert!(state.asset_batch_inflight.lock().unwrap().is_none());
}

#[test]
fn poll_pending_assets_survives_a_failed_static_rebuild() {
    // A reported size change routes the whole entry into rebuild_static_geometry,
    // which the backend rejects. The poll tallies the failure and clears the
    // in-flight slot without touching the in-place path.
    let mut state = one_mesh_state(vec![2, 5]);
    let batch = DecodedAssetBatch {
        meshes: vec![decoded_mesh(0, 3)],
        ..Default::default()
    };
    inject_batch(&state, batch);
    let mut backend = RecordingBackend {
        fail_static_rebuild: true,
        ..Default::default()
    };
    backend.geometry_sizes.insert(2, (999, 999));
    assert!(poll_pending_assets(&mut state, &mut backend));
    assert!(backend.mesh_updates.is_empty());
    assert_eq!(backend.static_rebuild_change_counts, vec![2]);
    assert!(state.asset_batch_inflight.lock().unwrap().is_none());
}

// spawn_envmap_worker in-flight skip (via reload_assets)

#[test]
fn reload_assets_skips_the_envmap_spawn_while_a_convolution_is_in_flight() {
    // With only an EnvironmentMap declared, reload_assets spawns no asset-decode
    // worker (no textures / LUT / meshes) and must leave an already-running
    // envmap convolution untouched so the user re-triggers after it lands.
    let dir = tempfile::tempdir().unwrap();
    let env_map = EnvironmentMapSource {
        resolved_path: dir.path().join("studio.hdr").to_string_lossy().into_owned(),
        prefilter_face_size: 64,
        irradiance_face_size: 16,
        prefilter_samples: 64,
        prefilter_clamp: 12.0,
    };
    let state = AssetHotReloadState::from_sources(HotReloadSources {
        environment_map: Some(env_map),
        ..Default::default()
    });
    // Simulate a still-running convolution: a receiver whose sender is alive and
    // has sent nothing yet.
    let (_tx, rx) = std::sync::mpsc::channel::<Result<Vec<u8>, String>>();
    *state.env_map_inflight.lock().unwrap() = Some(rx);
    reload_assets(&state);
    // Our receiver is still parked (a fresh spawn would have replaced it), and
    // no asset-decode batch was scheduled.
    let slot = state.env_map_inflight.lock().unwrap();
    assert!(matches!(
        slot.as_ref().unwrap().try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    assert!(state.asset_batch_inflight.lock().unwrap().is_none());
}

// run_frame (the per-frame reload entry, driven over a RecordingBackend)

// Reset the process-global reload flags so a run_frame test starts from a known
// state regardless of what ran before it. Callers hold `test_support::lock()`
// for the whole test so this is race-free.
fn clear_pending_flags() {
    super::pending::take_pending_world();
    super::pending::take_pending_shader_stages();
    super::pending::take_pending_stories();
}

// Drive one `run_frame` over a fresh RecordingBackend with an empty
// WorldReloadState and fog bookkeeping, returning the effects, the backend (so
// callers can assert which dispatches fired), and the fog bookkeeping the
// world.jsonl pass may have updated.
fn drive_run_frame(
    state: &mut AssetHotReloadState,
) -> (
    FrameHotReloadEffects,
    RecordingBackend,
    Option<crate::gfx::volumetric_fog::FogSettings>,
) {
    let mut backend = RecordingBackend::default();
    let world_reload: Option<crate::gfx::graphics_system::WorldReloadState> = None;
    let mut last_fog: Option<crate::gfx::volumetric_fog::FogSettings> = None;
    let effects = {
        let mut apply = crate::gfx::graphics_system::HotReloadApplyParts {
            backend: &mut backend,
            world_reload: &world_reload,
            last_fog_settings: &mut last_fog,
        };
        run_frame(state, &mut apply, None)
    };
    (effects, backend, last_fog)
}

#[test]
fn run_frame_with_no_pending_flags_returns_empty_effects() {
    let _guard = crate::test_support::lock();
    clear_pending_flags();
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    let (effects, backend, last_fog) = drive_run_frame(&mut state);
    assert!(effects.skeleton_updates.is_empty());
    assert!(effects.story_updates.is_empty());
    assert!(last_fog.is_none());
    assert_eq!(backend.fog_updates, 0);
    assert!(backend.mesh_updates.is_empty());
    assert!(backend.texture_updates.is_empty());
}

#[test]
fn run_frame_consumes_the_state_reload_flag_without_spawning_on_an_empty_catalogue() {
    let _guard = crate::test_support::lock();
    clear_pending_flags();
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    state
        .pending
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let (effects, _backend, _last_fog) = drive_run_frame(&mut state);
    // The flag was consumed and, with no file-backed sources, no worker spawned.
    assert!(!state.reload_requested());
    assert!(state.asset_batch_inflight.lock().unwrap().is_none());
    assert!(state.env_map_inflight.lock().unwrap().is_none());
    assert!(effects.story_updates.is_empty());
}

#[test]
fn run_frame_consumes_the_shader_stage_flag() {
    let _guard = crate::test_support::lock();
    clear_pending_flags();
    let mut state = AssetHotReloadState::from_sources(HotReloadSources::default());
    super::pending::set_pending_shader_stages();
    let _ = drive_run_frame(&mut state);
    // run_frame swallowed the flag; the empty Shader map made the pass a
    // no-op, but the flag consumption is the observable that it fired.
    assert!(!super::pending::take_pending_shader_stages());
}

#[test]
fn run_frame_reloads_stories_when_the_story_flag_is_set() {
    let _guard = crate::test_support::lock();
    clear_pending_flags();
    let dir = tempfile::tempdir().unwrap();
    let md = dir.path().join("tale.md");
    std::fs::write(
        &md,
        "---\ntitle: Tale\ncharacters:\n  a: Ana\n---\n\n# start\n\nHello there.\n",
    )
    .unwrap();
    let world = dir.path().join("world.jsonl");
    std::fs::write(
        &world,
        format!(
            "{}\n",
            serde_json::json!({
                "name": "tale", "type": "StoryImport",
                "args": {"source": md.to_str().unwrap()}
            })
        ),
    )
    .unwrap();
    let mut state = AssetHotReloadState::from_sources(HotReloadSources {
        world_jsonl_path: Some(world.to_string_lossy().into_owned()),
        ..Default::default()
    });
    super::pending::set_pending_stories();
    let (effects, _backend, _last_fog) = drive_run_frame(&mut state);
    assert_eq!(effects.story_updates.len(), 1);
    assert_eq!(effects.story_updates[0].title, "Tale");
    // The flag was consumed by the pass.
    assert!(!super::pending::take_pending_stories());
}

#[test]
fn run_frame_reloads_world_assets_when_the_world_flag_is_set() {
    let _guard = crate::test_support::lock();
    clear_pending_flags();
    let dir = tempfile::tempdir().unwrap();
    let world = write_world_line(
        dir.path(),
        r#"{"name":"fog","type":"VolumetricFog","args":{"enabled":true,"density":0.5}}"#,
    );
    let mut state = AssetHotReloadState::from_sources(HotReloadSources {
        world_jsonl_path: Some(world),
        ..Default::default()
    });
    super::pending::set_pending_world();
    let (_effects, backend, last_fog) = drive_run_frame(&mut state);
    // The world.jsonl pass applied the enabled fog exactly once and recorded it
    // into the caller's dedupe bookkeeping.
    assert_eq!(backend.fog_updates, 1);
    assert!(last_fog.is_some());
    assert!(!super::pending::take_pending_world());
}

// -- driver -------------------------------------------------------------

use super::driver::{HotReloadDriver, apply_effects};

#[test]
fn driver_on_a_world_without_graphics_stays_unarmed() {
    let mut driver = HotReloadDriver::new();
    let mut world = crate::ecs::World::new_empty();
    driver.drive(&mut world);
    assert!(driver.pending().is_none());
}

#[test]
fn driver_rearm_swaps_the_pending_flag() {
    // A world rebuild re-captures sources; arming again must hand out a fresh
    // flag (the server refreshes its shared copy every tick for this reason).
    let mut driver = HotReloadDriver::new();
    driver.arm(HotReloadSources::default());
    let first = driver.pending().expect("armed driver exposes a flag");
    driver.arm(HotReloadSources::default());
    let second = driver.pending().expect("re-armed driver exposes a flag");
    assert!(!std::sync::Arc::ptr_eq(&first, &second));
}

#[test]
fn armed_driver_survives_a_drive_over_an_empty_world() {
    // A backendless world (headless test, or a frame before init finishes)
    // must not panic or drop the armed state.
    let mut driver = HotReloadDriver::new();
    driver.arm(HotReloadSources::default());
    let mut world = crate::ecs::World::new_empty();
    driver.drive(&mut world);
    assert!(driver.pending().is_some());
}

#[test]
fn apply_effects_splices_the_matching_skeleton_pose_only() {
    use crate::assets::SkeletonPose;
    use crate::gfx::skinning::{Joint, JointPose, Skeleton};

    let mut world = crate::ecs::World::new_empty();
    world.add_component(SkeletonPose::new(
        Default::default(),
        0,
        Skeleton::new(Vec::new()),
    ));
    world.add_component(SkeletonPose::new(
        Default::default(),
        1,
        Skeleton::new(Vec::new()),
    ));

    let new_skeleton = Skeleton::new(vec![Joint {
        name: String::new(),
        parent: None,
        bind: JointPose::default(),
    }]);
    apply_effects(
        &mut world,
        FrameHotReloadEffects {
            skeleton_updates: vec![PendingSkeletonUpdate {
                skinned_index: 1,
                new_skeleton,
            }],
            story_updates: Vec::new(),
        },
    );

    let joints_of = |idx: usize| {
        world
            .query::<SkeletonPose>()
            .find(|p| p.skinned_index == idx)
            .map(|p| (p.skeleton.len(), p.joint_matrices.len()))
            .unwrap()
    };
    // The targeted pose carries the new 1-joint hierarchy with matching
    // matrices; the other pose keeps its empty skeleton (its seed matrices,
    // padded to a 1-identity minimum, are untouched too).
    assert_eq!(joints_of(1), (1, 1));
    assert_eq!(joints_of(0).0, 0);
}

#[test]
fn apply_effects_sends_a_story_reload_event() {
    let mut world = crate::ecs::World::new_empty();
    let story = crate::assets::Story {
        asset_id: Default::default(),
        title: "Tale".to_string(),
        nodes: Vec::new(),
        text_speed: 0.0,
        scaffold: Default::default(),
        save_key: String::new(),
    };
    apply_effects(
        &mut world,
        FrameHotReloadEffects {
            skeleton_updates: Vec::new(),
            story_updates: vec![story],
        },
    );

    let mut cursor = crate::ecs::EventCursor::default();
    let events = world
        .events::<crate::assets::StoryReload>()
        .expect("a StoryReload event queue exists after apply");
    let received: Vec<_> = events.read(&mut cursor).collect();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].story.title, "Tale");
}
