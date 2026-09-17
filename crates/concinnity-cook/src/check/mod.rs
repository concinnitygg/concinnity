//! Semantic validation of an expanded world: per-asset arg checks, cross-asset
//! reference checks, and world-shape rules (crate::check::shape). Structural
//! validation (name/type present, known type, unique names) happens earlier in
//! crate::authoring::world::load_world.
//!
//! Most checks here are pure JSON-shape validation. A few asset types validate
//! by running their compiler (mesh generators, texture generators,
//! cubemap/environment-map sources); those live in the four modules below and
//! run in the same collection pass as the pure ones.

pub(crate) mod animation_graph;
pub(crate) mod asset_refs;
pub(crate) mod audio;
pub mod behavior;
pub(crate) mod cross_reference;
pub(crate) mod cubemap_texture;
pub(crate) mod environment_map;
pub mod fault;
pub(crate) mod instanced_prop;
pub(crate) mod mesh;
pub(crate) mod physics;
/// `Prop` argument checks.
pub(crate) mod prop;
/// `SdfVolume` argument checks.
pub(crate) mod sdf_volume;
/// `Shader` argument checks.
pub(crate) mod shader;
pub(crate) mod shape;
pub(crate) mod texture;
pub(crate) mod voxel_chunk;
pub(crate) mod voxel_world;

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// The pure per-asset checks: JSON-shape validation that runs no compiler.
fn check_authored_asset(
    asset_type: RegisteredType,
    name: &str,
    args: &serde_json::Value,
) -> Result<(), String> {
    match asset_type {
        RegisteredType::AnimationGraph => animation_graph::check(name, args),
        RegisteredType::Behavior => behavior::check(name, args),
        RegisteredType::Variables => behavior::check_variables(name, args),
        RegisteredType::Shader => shader::check(name, args),
        RegisteredType::Prop => prop::check(name, args),
        RegisteredType::SdfVolume => sdf_volume::check(name, args),
        RegisteredType::VoxelChunk => voxel_chunk::check(name, args),
        RegisteredType::VoxelWorld => voxel_world::check(name, args),
        RegisteredType::InstancedProp => instanced_prop::check(name, args),
        RegisteredType::TriggerVolume => physics::check(name, args),
        RegisteredType::AudioEmitter => audio::check_emitter(name, args),
        RegisteredType::AudioCue => audio::check_cue(name, args),
        RegisteredType::PropBody => audio::check_prop_body(name, args),
        _ => Ok(()),
    }
}

// The per-asset checks that validate by running the asset's compiler.
fn check_compiled_asset(
    asset_type: RegisteredType,
    name: &str,
    args: &serde_json::Value,
) -> Result<(), String> {
    match asset_type {
        RegisteredType::Texture => texture::check(name, args),
        RegisteredType::CubemapTexture => cubemap_texture::check(name, args),
        RegisteredType::EnvironmentMap => environment_map::check(name, args),
        RegisteredType::Mesh | RegisteredType::ProceduralMesh => mesh::check(name, args),
        _ => Ok(()),
    }
}

// The full per-asset check: the pure checks plus the compile-backed ones.
pub(crate) fn check_asset(
    asset_type: RegisteredType,
    name: &str,
    args: &serde_json::Value,
) -> Result<(), String> {
    check_authored_asset(asset_type, name, args)?;
    check_compiled_asset(asset_type, name, args)
}

/// Run all semantic validation on a fully expanded world. Collects every
/// problem found (per-asset arg errors, unresolved cross-references, and
/// graphics-rule violations) so the caller can report them in a single pass.
pub(crate) fn check_world(assets: &[WorldJsonlAsset]) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    // Names must still be unique after expansion and injection: a duplicate
    // here means a generated or injected asset silently aliased another (the
    // authored world's uniqueness was already checked before expansion).
    let mut seen_names: std::collections::HashSet<&str> = Default::default();
    for asset in assets {
        if !seen_names.insert(asset.name.as_str()) {
            errors.push(format!(
                "duplicate name '{}' after build-time expansion: a generated or \
                 injected asset collides with another; rename one of them",
                asset.name
            ));
        }
    }

    // The world's declared variable table, if it declares one. Behaviors are
    // checked against it rather than in isolation, so a `set` resolves to the
    // variable's declared type and a misspelled name is caught here.
    let declared_vars = assets
        .iter()
        .find(|a| a.asset_type == RegisteredType::Variables)
        .map(|a| behavior::DeclaredVars::from_args(&a.args))
        .unwrap_or_default();

    for asset in assets {
        let checked = if asset.asset_type == RegisteredType::Behavior {
            behavior::check_with_vars(&asset.name, &asset.args, &declared_vars)
        } else {
            check_authored_asset(asset.asset_type, &asset.name, &asset.args)
        };
        if let Err(e) = checked {
            errors.push(e);
        }
        if let Err(e) = check_compiled_asset(asset.asset_type, &asset.name, &asset.args) {
            errors.push(e);
        }
    }

    if let Err(ref_errors) = cross_reference::validate_cross_references(assets) {
        errors.extend(ref_errors);
    }

    shape::check_shape(assets, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type,
            args,
        }
    }

    #[test]
    fn graphics_config_with_full_render_stack_passes_graphics_rules() {
        let assets = vec![
            asset("gfx", RegisteredType::GraphicsConfig, serde_json::json!({})),
            asset("win", RegisteredType::Window, serde_json::json!({})),
            asset(
                "scene_shader",
                RegisteredType::Shader,
                serde_json::json!({"fragment": "x.slang"}),
            ),
        ];
        assert!(check_world(&assets).is_ok());
    }

    #[test]
    fn per_asset_and_cross_reference_errors_both_collected() {
        // Prop with no mesh/model/prefab (per-asset error) plus a Material
        // with a missing albedo texture (cross-reference error).
        let assets = vec![
            asset("bad_prop", RegisteredType::Prop, serde_json::json!({})),
            asset(
                "bad_mat",
                RegisteredType::Material,
                serde_json::json!({"albedo":"ghost"}),
            ),
        ];
        let errs = check_world(&assets).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad_prop")));
        assert!(errs.iter().any(|e| e.contains("ghost")));
    }

    // The composed pass surfaces a compile-backed error (unknown texture
    // generator) alongside a pure one (Prop with no source) -- both check sets
    // run in one collection.
    #[test]
    fn composed_checks_collect_pure_and_compile_backed_errors() {
        let assets = vec![
            asset("bad_prop", RegisteredType::Prop, serde_json::json!({})),
            asset(
                "bad_tex",
                RegisteredType::Texture,
                serde_json::json!({"generator": "not_a_generator"}),
            ),
        ];
        let errs = check_world(&assets).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad_prop")));
        assert!(errs.iter().any(|e| e.contains("not_a_generator")));
    }

    // Each compile-backed asset type reaches its own check.
    #[test]
    fn check_asset_routes_compile_backed_types() {
        let args = serde_json::json!({"source": "studio.png"});
        let err = check_asset(RegisteredType::CubemapTexture, "c", &args).unwrap_err();
        assert!(err.contains("Radiance .hdr"), "{err}");

        let args = serde_json::json!({"generator": "aurora"});
        let err = check_asset(RegisteredType::EnvironmentMap, "e", &args).unwrap_err();
        assert!(err.contains("unknown EnvironmentMap generator"), "{err}");
    }

    #[test]
    fn check_asset_runs_both_check_sets() {
        // A pure check arm.
        assert!(check_asset(RegisteredType::Prop, "p", &serde_json::json!({})).is_err());
        // A compile-backed arm.
        assert!(
            check_asset(
                RegisteredType::Texture,
                "t",
                &serde_json::json!({"generator": "not_a_generator"})
            )
            .is_err()
        );
        // A type neither set knows is fine.
        assert!(check_asset(RegisteredType::Window, "w", &serde_json::json!({})).is_ok());
    }
}
