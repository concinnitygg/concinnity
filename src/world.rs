//! Compile authored worlds into runnable [`World`]s, entirely in memory.
//!
//! The runtime plays compiled blobs, never `world.jsonl` source declarations.
//! This module owns the in-process compile step: validate and expand the
//! declarations, compile each asset's payload, then assemble the components
//! into a [`World`] backed by the compiled blob. Source-backed assets are
//! cached under `.concinnity/cache/` (relative to the current directory), so
//! a second run with unchanged sources skips the expensive decode.
//!
//! # Adding assets to a world
//!
//! Typed builders in [`asset`] produce one [`AssetSpec`] each; a batch of
//! specs compiles into a world:
//!
//! ```no_run
//! use concinnity::App;
//! use concinnity::world::{asset, compile_assets};
//!
//! fn main() -> std::io::Result<()> {
//!     let specs = vec![
//!         asset::camera("camera", [0.0, 2.4, 9.0], 0.0, -0.12),
//!         asset::directional_light("sun", [1.0, 0.96, 0.86], [-0.35, 0.85, 0.35], 2.2),
//!         asset::room("room", [16.0, 20.0, 5.0]),
//!     ];
//!     let mut app = App::new();
//!     app.load_world(compile_assets(&specs)?);
//!     app.run()
//! }
//! ```
//!
//! An [`AssetSpec`] can also be built field by field for any registered asset
//! type, without a dedicated builder:
//!
//! ```
//! use concinnity::world::{asset, AssetSpec};
//!
//! let lamp = AssetSpec::new("lamp", "PointLight").set("intensity", 8.0f32);
//! assert_eq!(lamp.asset_type, "PointLight");
//!
//! let sun = asset::directional_light("sun", [1.0, 0.96, 0.86], [-0.35, 0.85, 0.35], 2.2);
//! assert_eq!(sun.asset_type, "DirectionalLight");
//! ```
//!
//! # Compiling a world.jsonl file
//!
//! A world authored as `world.jsonl` (the `cn` CLI's medium) compiles the
//! same way:
//!
//! ```no_run
//! use concinnity::App;
//! use concinnity::world::compile_world;
//!
//! fn main() -> std::io::Result<()> {
//!     let content = std::fs::read_to_string("world.jsonl")?;
//!     let mut app = App::new();
//!     app.load_world(compile_world(&content)?);
//!     app.run()
//! }
//! ```

use concinnity_cook::world::LoadedWorld;
use concinnity_cook::{build_compiled, check::report_validation_errors, prepare_world};
use concinnity_engine::blob::BlobData;
use concinnity_engine::ecs::{ComponentAsset, World};
use concinnity_world::spec::spec_to_value;
use concinnity_world::world::write_world_jsonl;

pub use concinnity_world::spec::AssetSpec;
pub use concinnity_world::spec::asset;

/// Compile world.jsonl content into a runnable [`World`] entirely in memory:
/// validate and expand the declarations, compile each asset's payload, then
/// assemble the components into a world backed by the compiled blob.
pub fn compile_world(content: &str) -> std::io::Result<World> {
    // The cook ships no shader compilers of its own; hand it this build's
    // before any ShaderStage is compiled.
    concinnity_shader::install();

    let loaded: LoadedWorld =
        prepare_world(content).map_err(|errs| report_validation_errors(&errs))?;

    let mut result = build_compiled(loaded.assets, None)?;

    let payload_sections: Vec<Option<Vec<u8>>> = result.payloads.into_iter().map(Some).collect();
    let mut world = World::from_blob(BlobData::new(payload_sections));

    for def in &result.defs {
        let mut component = ComponentAsset::from_baked(def).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("asset construction failed: {e:?}"),
            )
        })?;
        if let Some(locator) = &def.payload {
            component.inject_locator(locator.clone());
        }
        world.add(component);
    }

    // Load the compiled blob's resource stream into the per-kind tables the
    // systems read by handle. Kinds that have left the component registry
    // (textures, audio clips, fonts, colour LUTs, environment maps) live here,
    // not in `defs`, so without this the renderer sees an empty texture pool
    // and every material's albedo handle resolves out of range. Same call the
    // shipped runtime's `App::load_blob` makes.
    concinnity_engine::resource::install_resource_tables(&mut world, &mut result.resources);

    Ok(world)
}

/// Compile typed [`AssetSpec`]s into a runnable [`World`], the struct-first
/// equivalent of [`compile_world`].
pub fn compile_assets(specs: &[AssetSpec]) -> std::io::Result<World> {
    let values: Vec<_> = specs.iter().map(spec_to_value).collect();
    let content = write_world_jsonl(&values)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    compile_world(&content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::assets::{Camera3D, DirectionalLight};

    #[test]
    fn compile_assets_builds_a_world_from_specs() {
        let specs = vec![
            asset::camera("camera", [0.0, 2.4, 9.0], 0.0, -0.12),
            asset::directional_light("sun", [1.0, 0.96, 0.86], [-0.35, 0.85, 0.35], 2.2),
            asset::room("room", [16.0, 20.0, 5.0]),
        ];
        let world = compile_assets(&specs).expect("source-free specs compile in memory");
        assert!(world.query::<Camera3D>().next().is_some());
        let sun = world
            .query::<DirectionalLight>()
            .next()
            .expect("the sun compiled into a component");
        assert_eq!(sun.intensity, 2.2);
    }

    #[test]
    fn compile_world_reports_validation_errors() {
        let err = compile_world(r#"{"name": "x", "type": "NotARegisteredType", "args": {}}"#)
            .expect_err("an unknown asset type fails validation");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn compile_assets_of_empty_specs_yields_the_injected_defaults() {
        // An empty authored world still compiles: the pipeline injects the
        // engine defaults (DebugHud et al), so the world is valid but carries
        // no authored scene.
        let world = compile_assets(&[]).expect("an empty world compiles");
        assert!(world.query::<Camera3D>().next().is_none());
    }
}
