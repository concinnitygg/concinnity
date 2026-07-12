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
    // Install the render backend's shader-layout validator before any shader
    // compiles, so a user shader that mis-declares an engine buffer struct fails
    // the build with a clear message instead of faulting the GPU at run time.
    // One call here covers the CLI build, `run`, and the FFI entry points.
    ensure_shader_layout_validator();

    let loaded = concinnity_cook::prepare_world(content)
        .map_err(|errs| concinnity_cook::check::report_validation_errors(&errs))?;

    Ok(loaded)
}

// Register the backend's shader-layout validator with the core build pipeline.
// Only the Metal backend ships one today; other backends leave the hook
// unregistered and build exactly as before.
#[cfg(backend_metal)]
fn ensure_shader_layout_validator() {
    crate::shader_reflect::register_shader_layout_validator();
}

#[cfg(not(backend_metal))]
fn ensure_shader_layout_validator() {}

// Compile a prepared world and assemble it into an in-memory World, ready to
// run without touching any blob files on disk.
pub fn world_from_loaded(loaded: LoadedWorld) -> std::io::Result<World> {
    let result = build_compiled(loaded.assets, None)?;

    let payload_sections: Vec<Option<Vec<u8>>> = result.payloads.into_iter().map(Some).collect();
    let mut world = World::new(crate::blob::BlobData::new(payload_sections));
    for def in &result.defs {
        let mut component = ComponentAsset::from_def(def).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset construction failed: {:?}", e),
            )
        })?;
        if let Some(locator) = &def.payload {
            component.inject_locator(locator.clone());
        }
        world.add(component);
    }
    // Load the compiled resource stream into its per-kind tables, exactly as the
    // shipped runtime's `load_blob` does, so the in-memory `cn debug` world reads
    // audio clips by handle too.
    world.insert_resource(crate::resource::AudioClipTable::from_records(
        &result.resources,
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
    let loaded = prepare(&content)?;
    let result = concinnity_cook::build_compiled(loaded.assets, None)?;
    concinnity_cook::write_build_outputs(&result, &loaded.injected)?;
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

    #[test]
    fn world_from_loaded_assembles_an_in_memory_world() {
        let loaded =
            prepare("{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}\n").unwrap();
        let expanded = loaded.assets.len();
        let world = world_from_loaded(loaded).unwrap();
        // Every expanded asset landed as a component; nothing was dropped on
        // the way through compile + assembly.
        assert_eq!(world.component_count(), expanded);
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
            concinnity_core::paths::data_dir().join("0").exists(),
            "data/0 blob written"
        );
        assert!(
            dir.path().join("world-lock.json").exists(),
            "world-lock.json written"
        );
    }
}
