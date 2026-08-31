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

use concinnity_core::platform::Platform;

use crate::authoring::world::WorldJsonlAsset;

/// Print each validation error in CLI form and collapse them into a single
/// io::Error. Shared by the `cn test` command and the build orchestrator so a
/// failed world surfaces every problem in one pass.
pub fn report_validation_errors(errors: &[String]) -> std::io::Error {
    for e in errors {
        eprintln!("error:   {}", e);
    }
    eprintln!("\nvalidation failed ({} error(s))", errors.len());
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("validation failed with {} error(s)", errors.len()),
    )
}

// The pure per-asset checks: JSON-shape validation that runs no compiler.
fn check_authored_asset(
    type_norm: &str,
    name: &str,
    args: &serde_json::Value,
    platform: Platform,
) -> Result<(), String> {
    match type_norm {
        "animationgraph" => animation_graph::check(name, args),
        "behavior" => behavior::check(name, args),
        "variables" => behavior::check_variables(name, args),
        "shader" => shader::check(name, args, platform),
        "prop" => prop::check(name, args),
        "sdfvolume" | "sdf" => sdf_volume::check(name, args, platform),
        "voxelchunk" | "chunk" => voxel_chunk::check(name, args),
        "voxelworld" => voxel_world::check(name, args),
        "instancedprop" | "instanced" => instanced_prop::check(name, args),
        "triggervolume" => physics::check(name, args),
        "audioemitter" => audio::check_emitter(name, args),
        "audiocue" => audio::check_cue(name, args),
        "propbody" => audio::check_prop_body(name, args),
        _ => Ok(()),
    }
}

// The per-asset checks that validate by running the asset's compiler.
fn check_compiled_asset(
    type_norm: &str,
    name: &str,
    args: &serde_json::Value,
) -> Result<(), String> {
    match type_norm {
        "texture" => texture::check(name, args),
        "cubemaptexture" | "cubemap" => cubemap_texture::check(name, args),
        "environmentmap" | "envmap" | "ibl" => environment_map::check(name, args),
        "mesh" | "proceduralmesh" => mesh::check(name, args),
        _ => Ok(()),
    }
}

// The full per-asset check: the pure checks plus the compile-backed ones.
pub(crate) fn check_asset(
    type_norm: &str,
    name: &str,
    args: &serde_json::Value,
    platform: Platform,
) -> Result<(), String> {
    check_authored_asset(type_norm, name, args, platform)?;
    check_compiled_asset(type_norm, name, args)
}

/// Run all semantic validation on a fully expanded world. Collects every
/// problem found (per-asset arg errors, unresolved cross-references, and
/// graphics-rule violations) so the caller can report them in a single pass.
/// `platform` is the shader platform the world is cooked for; the shader-backed
/// types are checked against it.
pub(crate) fn check_world(
    assets: &[WorldJsonlAsset],
    platform: Platform,
) -> Result<(), Vec<String>> {
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
        .find(|a| a.asset_type.to_lowercase().replace('_', "") == "variables")
        .map(|a| behavior::DeclaredVars::from_args(&a.args))
        .unwrap_or_default();

    for asset in assets {
        let type_norm = asset.asset_type.to_lowercase().replace('_', "");
        let checked = if type_norm == "behavior" {
            behavior::check_with_vars(&asset.name, &asset.args, &declared_vars)
        } else {
            check_authored_asset(&type_norm, &asset.name, &asset.args, platform)
        };
        if let Err(e) = checked {
            errors.push(e);
        }
        if let Err(e) = check_compiled_asset(&type_norm, &asset.name, &asset.args) {
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

    fn asset(name: &str, asset_type: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
        }
    }

    #[test]
    fn graphics_config_with_full_render_stack_passes_graphics_rules() {
        let assets = vec![
            asset("gfx", "GraphicsConfig", serde_json::json!({})),
            asset("win", "Window", serde_json::json!({})),
            asset(
                "scene_shader",
                "Shader",
                serde_json::json!({
                    "vertex": {"sources": {"metal": "x.metal", "hlsl": "x.hlsl", "glsl": "x.glsl"}},
                    "fragment": {"sources": {"metal": "x.metal", "hlsl": "x.hlsl", "glsl": "x.glsl"}}
                }),
            ),
        ];
        assert!(check_world(&assets, Platform::Metal).is_ok());
    }

    #[test]
    fn per_asset_and_cross_reference_errors_both_collected() {
        // Prop with no mesh/model/prefab (per-asset error) plus a Material
        // with a missing albedo texture (cross-reference error).
        let assets = vec![
            asset("bad_prop", "Prop", serde_json::json!({})),
            asset("bad_mat", "Material", serde_json::json!({"albedo":"ghost"})),
        ];
        let errs = check_world(&assets, Platform::Metal).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad_prop")));
        assert!(errs.iter().any(|e| e.contains("ghost")));
    }

    // The composed pass surfaces a compile-backed error (unknown texture
    // generator) alongside a pure one (Prop with no source) -- both check sets
    // run in one collection.
    #[test]
    fn composed_checks_collect_pure_and_compile_backed_errors() {
        let assets = vec![
            asset("bad_prop", "Prop", serde_json::json!({})),
            asset(
                "bad_tex",
                "Texture",
                serde_json::json!({"generator": "not_a_generator"}),
            ),
        ];
        let errs = check_world(&assets, Platform::Metal).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad_prop")));
        assert!(errs.iter().any(|e| e.contains("not_a_generator")));
    }

    // Every spelling of a compile-backed asset type reaches the same check.
    #[test]
    fn check_asset_routes_each_type_alias() {
        for alias in ["cubemaptexture", "cubemap"] {
            let args = serde_json::json!({"source": "studio.png"});
            let err = check_asset(alias, "c", &args, Platform::Metal).unwrap_err();
            assert!(err.contains("Radiance .hdr"), "{alias}: {err}");
        }
        for alias in ["environmentmap", "envmap", "ibl"] {
            let args = serde_json::json!({"generator": "aurora"});
            let err = check_asset(alias, "e", &args, Platform::Metal).unwrap_err();
            assert!(
                err.contains("unknown EnvironmentMap generator"),
                "{alias}: {err}"
            );
        }
    }

    #[test]
    fn check_asset_runs_both_check_sets() {
        // A pure check arm.
        assert!(check_asset("prop", "p", &serde_json::json!({}), Platform::Metal).is_err());
        // A compile-backed arm.
        assert!(
            check_asset(
                "texture",
                "t",
                &serde_json::json!({"generator": "not_a_generator"}),
                Platform::Metal,
            )
            .is_err()
        );
        // A type neither set knows is fine.
        assert!(check_asset("window", "w", &serde_json::json!({}), Platform::Metal).is_ok());
    }
}
