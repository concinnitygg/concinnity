// src/build.rs: shared in-memory build orchestration

#[allow(unused_imports)]
pub use concinnity_cook::{
    PipelineResult, build_compiled, build_pipeline_from_str, validate_asset, validate_world_jsonl,
};

use crate::ecs::{ComponentAsset, World};
use concinnity_cook::world::LoadedWorld;

// Load, validate, and (when server credentials are present) fetch the missing
// source files for a world. The returned LoadedWorld has passed the full
// validation front half and is ready for concinnity_cook::build_compiled.
//
// This is the shared front half of every in-memory build: `build_world_from_path`
// (the CLI interpreted `run` and the FFI preview) funnels through here so
// validation and asset fetching behave identically. The `cn build` blob path
// prepares through concinnity_cook directly and does not use this.
pub(crate) fn prepare(content: &str) -> std::io::Result<LoadedWorld> {
    // Install this build's shader toolchain (and the backend's shader-layout
    // validator, where it ships one) before any shader compiles: the cook has no
    // compiler of its own, and a user shader that mis-declares an engine buffer
    // struct should fail the build with a clear message instead of faulting the
    // GPU at run time. Idempotent, so covering `run` and the FFI entry points
    // here costs nothing when the CLI already installed at startup.
    concinnity_shader::install();

    let loaded = concinnity_cook::prepare_world(content)
        .map_err(|errs| concinnity_cook::check::report_validation_errors(&errs))?;

    Ok(loaded)
}

// Normalized asset-type match (lowercase, underscores stripped), matching the
// convention used across the cook world passes.
fn type_is(asset: &concinnity_cook::world::WorldJsonlAsset, norm_type: &str) -> bool {
    asset.asset_type.to_lowercase().replace('_', "") == norm_type
}

// The first declared ColorLut's authored `source` path (non-empty), or `None`.
// Dev-only; feeds the hot-reload watcher.
fn scan_color_lut_source(assets: &[concinnity_cook::world::WorldJsonlAsset]) -> Option<String> {
    assets
        .iter()
        .find(|a| type_is(a, "colorlut"))
        .and_then(|a| a.args.get("source").and_then(|v| v.as_str()))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

// The first declared file-backed EnvironmentMap's re-bake inputs, or `None` (a
// procedural `generator` has no file to watch). The face-size / sample defaults
// mirror the EnvironmentMap schema defaults in `concinnity-asset/src/environment_map.rs`.
fn scan_environment_map_source(
    assets: &[concinnity_cook::world::WorldJsonlAsset],
) -> Option<crate::resource::EnvironmentMapSourceInfo> {
    let a = assets.iter().find(|a| type_is(a, "environmentmap"))?;
    let generator = a
        .args
        .get("generator")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let source = a.args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    if !generator.is_empty() || source.is_empty() {
        return None;
    }
    let u32_arg = |key: &str, default: u32| {
        a.args
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(default)
    };
    Some(crate::resource::EnvironmentMapSourceInfo {
        source: source.to_string(),
        prefilter_face_size: u32_arg("prefilter_face_size", 512),
        irradiance_face_size: u32_arg("irradiance_face_size", 8),
        prefilter_samples: u32_arg("prefilter_samples", 1024),
        prefilter_clamp: a
            .args
            .get("prefilter_clamp")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(12.0),
    })
}

// Compile a prepared world and assemble it into an in-memory World, ready to
// run without touching any blob files on disk.
pub fn world_from_loaded(loaded: LoadedWorld) -> std::io::Result<World> {
    // Capture the dev-only hot-reload source info for the singleton ColorLut and
    // EnvironmentMap resources BEFORE `build_compiled` consumes the asset list.
    // These kinds are authored (never injected by an expansion pass), so the raw
    // world list carries every one; the renderer's `capture_sources` path reads
    // these to seed the file-reload watcher now that the drained `source` field is
    // gone. Only the first of each is used (the runtime uses handle 0).
    let color_lut_source = scan_color_lut_source(&loaded.assets);
    let environment_map_source = scan_environment_map_source(&loaded.assets);

    let mut result = build_compiled(loaded.assets, None)?;

    let payload_sections: Vec<Option<Vec<u8>>> = result.payloads.into_iter().map(Some).collect();
    let mut world = World::from_blob(crate::blob::BlobData::new(payload_sections));
    // Index every named component's entity as it is minted, matching the
    // shipped runtime's `load_blob`, so name references resolve for any type.
    let mut by_name = std::collections::BTreeMap::new();
    for def in &result.defs {
        let mut component = ComponentAsset::from_baked(def).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset construction failed: {:?}", e),
            )
        })?;
        if let Some(locator) = &def.payload {
            component.inject_locator(locator.clone());
        }
        let entity = world.add(component);
        if let Some(id) = def.name {
            by_name.insert(id, entity);
        }
    }
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    // Load the compiled resource stream into its per-kind tables, exactly as the
    // shipped runtime's `load_blob` does, so the in-memory `cn debug` world reads
    // audio clips and textures by handle too.
    crate::resource::install_resource_tables(&mut world, &mut result.resources);
    world.insert_resource(crate::ecs::BlobSceneGroups(result.scene_groups));
    world.insert_resource(crate::ecs::BlobMeshBounds(result.mesh_bounds));
    // Dev-only source catalogues for the hot-reload watcher (see the scan above).
    world.insert_resource(crate::resource::ColorLutSources(color_lut_source));
    world.insert_resource(crate::resource::EnvironmentMapSources(
        environment_map_source,
    ));
    // Dev-only: the texture source catalogue, so the renderer's hot-reload
    // capture and the runtime spawn-by-name path can map a texture handle back to
    // its file / name. Not present in the shipped `load_blob` path.
    world.insert_resource(crate::resource::TextureSources(
        result
            .texture_sources
            .iter()
            .map(|t| crate::resource::TextureSource {
                name_id: t.name_id,
                source: t.source.clone(),
                image_index: t.image_index,
            })
            .collect(),
    ));
    // Dev-only: the mesh source catalogue, so the renderer's hot-reload capture
    // can map a mesh handle back to the `.glb`/`.fbx` that backs it.
    world.insert_resource(crate::resource::MeshSources(
        result
            .mesh_sources
            .iter()
            .map(|m| crate::resource::MeshSource {
                source: m.source.clone(),
                primitive_index: m.primitive_index,
                lod_levels: m.lod_levels,
                lod_distances: m.lod_distances.clone(),
            })
            .collect(),
    ));
    Ok(world)
}

// Run the full in-memory pipeline on a world.jsonl string, returning a
// ready-to-run World without touching any blob files on disk. The editor uses
// this to boot an empty (or otherwise non-renderable) world from a seeded
// GraphicsConfig so a window still opens.
pub fn build_world_from_str(content: &str) -> std::io::Result<World> {
    let loaded = prepare(content)?;
    world_from_loaded(loaded)
}

// Read a world.jsonl file from disk and run the full in-memory pipeline on it,
// returning a ready-to-run World. The interpreted `run` (in the CLI crate)
// loads its world through here; it is the file-backed counterpart of `prepare`
// + `world_from_loaded`.
pub fn build_world_from_path(world_path: &str) -> std::io::Result<World> {
    let content = std::fs::read_to_string(world_path)?;
    build_world_from_str(&content)
}

// Compile a world.jsonl file and write the compiled blobs + world-lock.json to
// the active `.concinnity/data/` state dir, exactly as `cn build` does. This is
// `cn build` as a library call: the editor's SAVE goes through here to persist
// edits, reusing the validated compile + blob-write tail rather than patching
// blobs directly. Same-process recompiles are fast because the payload / expand
// caches are warm.
pub fn build_world_to_disk(world_path: &str) -> std::io::Result<()> {
    let content = std::fs::read_to_string(world_path)?;
    build_world_str_to_disk(&content)
}

// The string-backed tail of `build_world_to_disk`: compile world content and
// write the blobs + lock without reading (or writing) a world.jsonl. The
// editor console's build command goes through here so it compiles the
// in-memory entries as they stand, saved or not.
pub(crate) fn build_world_str_to_disk(content: &str) -> std::io::Result<()> {
    build_world_str_to_disk_with_progress(content, None)
}

// `build_world_str_to_disk` with a compile-progress callback (the editor's
// cook workers feed their operation card through it).
pub(crate) fn build_world_str_to_disk_with_progress(
    content: &str,
    progress: Option<&(dyn Fn(concinnity_cook::BuildProgress) + Sync)>,
) -> std::io::Result<()> {
    let loaded = prepare(content)?;
    let result = concinnity_cook::build_compiled_with_progress(loaded.assets, None, progress)?;
    if let Some(p) = progress {
        p(concinnity_cook::BuildProgress {
            stage: "write",
            done: 0,
            total: 0,
        });
    }
    concinnity_cook::write_build_outputs(&result, &loaded.injected, &loaded.shadowed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_accepts_a_valid_world() {
        let loaded =
            prepare("{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}\n").unwrap();
        assert!(loaded.assets.iter().any(|a| a.name == "phys"));
        assert!(loaded.authored.contains(&"phys".to_string()));
    }

    #[test]
    fn prepare_rejects_an_invalid_world() {
        assert!(prepare("{\"name\":\"odd\",\"type\":\"NotARealAssetType\"}\n").is_err());
        assert!(prepare("{ not json\n").is_err());
    }

    fn asset(json: serde_json::Value) -> concinnity_cook::world::WorldJsonlAsset {
        concinnity_cook::world::WorldJsonlAsset::from_value(&json)
    }

    // The type match is the cook's normalized one, so `color_lut`, `ColorLut`,
    // and `colorlut` all name the same kind.
    #[test]
    fn the_lut_scan_takes_the_first_source_however_the_type_is_spelled() {
        for ty in ["ColorLut", "color_lut", "colorlut", "COLOR_LUT"] {
            let assets = [asset(
                serde_json::json!({"name":"grade","type":ty,"args":{"source":"luts/warm.cube"}}),
            )];
            assert_eq!(
                scan_color_lut_source(&assets),
                Some("luts/warm.cube".to_string()),
                "type {ty}"
            );
        }

        // Only the first is used: the runtime binds handle 0.
        let assets = [
            asset(serde_json::json!({"name":"a","type":"ColorLut","args":{"source":"first.cube"}})),
            asset(
                serde_json::json!({"name":"b","type":"ColorLut","args":{"source":"second.cube"}}),
            ),
        ];
        assert_eq!(
            scan_color_lut_source(&assets),
            Some("first.cube".to_string())
        );
    }

    // Nothing to watch: no LUT at all, or one with no authored source path.
    #[test]
    fn the_lut_scan_yields_nothing_without_a_source() {
        assert_eq!(scan_color_lut_source(&[]), None);
        let no_source = [asset(
            serde_json::json!({"name":"grade","type":"ColorLut","args":{}}),
        )];
        assert_eq!(scan_color_lut_source(&no_source), None);
        let empty = [asset(
            serde_json::json!({"name":"grade","type":"ColorLut","args":{"source":""}}),
        )];
        assert_eq!(scan_color_lut_source(&empty), None);
        let other_kind = [asset(
            serde_json::json!({"name":"sky","type":"EnvironmentMap","args":{"source":"x.hdr"}}),
        )];
        assert_eq!(scan_color_lut_source(&other_kind), None);
    }

    // A file-backed environment map carries its re-bake inputs so the watcher
    // can reproduce the original bake; unset ones fall back to the schema
    // defaults rather than zero.
    #[test]
    fn the_environment_map_scan_defaults_the_unset_bake_inputs() {
        let assets = [asset(
            serde_json::json!({"name":"sky","type":"EnvironmentMap","args":{"source":"studio.hdr"}}),
        )];
        let info = scan_environment_map_source(&assets).expect("a file-backed map");
        assert_eq!(info.source, "studio.hdr");
        assert_eq!(info.prefilter_face_size, 512);
        assert_eq!(info.irradiance_face_size, 8);
        assert_eq!(info.prefilter_samples, 1024);
        assert_eq!(info.prefilter_clamp, 12.0);
    }

    #[test]
    fn the_environment_map_scan_carries_the_authored_bake_inputs() {
        let assets = [asset(serde_json::json!({
            "name":"sky","type":"environment_map","args":{
                "source":"studio.hdr",
                "prefilter_face_size": 256,
                "irradiance_face_size": 16,
                "prefilter_samples": 64,
                "prefilter_clamp": 4.5
            }
        }))];
        let info = scan_environment_map_source(&assets).expect("a file-backed map");
        assert_eq!(info.prefilter_face_size, 256);
        assert_eq!(info.irradiance_face_size, 16);
        assert_eq!(info.prefilter_samples, 64);
        assert_eq!(info.prefilter_clamp, 4.5);
    }

    // A procedural map has no file behind it, so there is nothing to watch --
    // even when a stale `source` is still authored alongside the generator.
    #[test]
    fn the_environment_map_scan_skips_a_procedural_map() {
        let generated = [asset(serde_json::json!({
            "name":"sky","type":"EnvironmentMap","args":{"generator":"sky"}
        }))];
        assert!(scan_environment_map_source(&generated).is_none());

        let both = [asset(serde_json::json!({
            "name":"sky","type":"EnvironmentMap","args":{"generator":"sky","source":"studio.hdr"}
        }))];
        assert!(scan_environment_map_source(&both).is_none());

        assert!(scan_environment_map_source(&[]).is_none());
        let no_source = [asset(
            serde_json::json!({"name":"sky","type":"EnvironmentMap","args":{}}),
        )];
        assert!(scan_environment_map_source(&no_source).is_none());
    }

    // The assembled world publishes both dev-only source catalogues, which is
    // what seeds the hot-reload watcher.
    #[test]
    fn the_assembled_world_publishes_the_watcher_source_catalogues() {
        let loaded =
            prepare("{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}\n").unwrap();
        let world = world_from_loaded(loaded).unwrap();
        assert!(
            world
                .resource::<crate::resource::ColorLutSources>()
                .is_some_and(|s| s.0.is_none()),
            "a world with no LUT publishes an empty catalogue, not none at all"
        );
        assert!(
            world
                .resource::<crate::resource::EnvironmentMapSources>()
                .is_some()
        );
    }

    #[test]
    fn world_from_loaded_assembles_an_in_memory_world() {
        let loaded =
            prepare("{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}\n").unwrap();
        let expanded = loaded.assets.len();
        let world = world_from_loaded(loaded).unwrap();
        // Every expanded asset landed as a component; nothing was dropped on
        // the way through compile + assembly.
        assert_eq!(world.component_count(), expanded);
        // Every named component's entity is indexed, so name references
        // resolve for any type, not just decomposed Props.
        let index = world
            .resource::<concinnity_core::ecs::EntityByName>()
            .expect("assembly publishes the name -> entity index");
        assert_eq!(index.0.len(), expanded);
    }

    #[test]
    fn build_world_from_str_assembles_an_in_memory_world() {
        // The string path is what the editor uses to seed an empty world; it
        // must produce the same assembled world as the file-backed path.
        let world =
            build_world_from_str("{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}\n")
                .unwrap();
        assert!(world.component_count() >= 1);
    }

    #[test]
    fn build_world_from_missing_path_is_not_found() {
        let err = build_world_from_path("/no/such/concinnity-world-xyz.jsonl")
            .expect_err("a missing world path must error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    // Restores the previous working directory on drop, so a chdir-in-test does
    // not leak into other tests (they run in parallel threads of one process).
    struct CwdGuard(std::path::PathBuf);
    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }

    // build_world_to_disk compiles a world.jsonl and writes the blobs + lock to
    // the cwd-relative state tree, exactly as `cn build` does. Runs under the
    // process cwd lock in an isolated temp dir so it neither races other tests
    // nor pollutes the repo. Uses a payload-free world (PhysicsConfig) so it
    // needs no source files or shader compilation.
    #[test]
    fn build_world_to_disk_writes_blobs_and_lock() {
        let _guard = crate::test_support::lock();
        let dir = tempfile::tempdir().unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();
        let _cwd = CwdGuard(prev);

        std::fs::write(
            "world.jsonl",
            "{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}\n",
        )
        .unwrap();

        build_world_to_disk("world.jsonl").expect("compile + write should succeed");

        // The primary blob (data/0) and the provenance lock are both written.
        assert!(
            concinnity_store::paths::data_dir().join("0").exists(),
            "data/0 blob written"
        );
        assert!(
            dir.path().join("world-lock.json").exists(),
            "world-lock.json written"
        );
    }
}
