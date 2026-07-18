// Semantic validation of an expanded world: per-asset arg checks, cross-asset
// reference checks, and world-shape rules (crate::check::shape). Structural
// validation (name/type present, known type, unique names) happens earlier in
// crate::world::load_world.
//
// The checks here are pure JSON-shape validation. A few asset types validate
// by running their compiler (mesh generators, texture generators, ...); those
// compilers live in concinnity-cook, which passes them into `check_world_with`
// as the per-asset `extra` hook and composes the full check set behind its own
// entry points.

pub mod anim_graph;
pub(crate) mod asset_refs;
pub mod cross_reference;
pub mod instanced_prop;
pub mod prop;
pub mod scene_reel;
pub mod sdf_volume;
pub mod shader;
pub(crate) mod shape;
pub mod voxel_chunk;
pub mod voxel_world;

use crate::world::WorldJsonlAsset;

// A per-asset check supplied by the caller, run alongside the built-in ones:
// (normalized type, asset name, args) -> error message on failure. cook uses
// this to plug in its compile-backed checks.
pub type ExtraAssetCheck<'a> = &'a dyn Fn(&str, &str, &serde_json::Value) -> Result<(), String>;

// Print each validation error in CLI form and collapse them into a single
// io::Error. Shared by the `cn test` command and the build orchestrator so a
// failed world surfaces every problem in one pass.
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

// The pure per-asset checks. Types whose validation runs their compiler
// (mesh/proceduralmesh, texture, cubemap, environment map) are not handled
// here; cook covers them through the `extra` hook.
pub fn check_asset(type_norm: &str, name: &str, args: &serde_json::Value) -> Result<(), String> {
    match type_norm {
        "animgraph" => anim_graph::check(name, args),
        "shaderstage" => shader::check(name, args),
        "prop" => prop::check(name, args),
        "scenereel" | "scenreel" => scene_reel::check(name, args),
        "sdfvolume" | "sdf" => sdf_volume::check(name, args),
        "voxelchunk" | "chunk" => voxel_chunk::check(name, args),
        "voxelworld" => voxel_world::check(name, args),
        "instancedprop" | "instanced" => instanced_prop::check(name, args),
        _ => Ok(()),
    }
}

// Run all semantic validation on a fully expanded world, with the caller's
// extra per-asset checks folded into the same pass. Collects every problem
// found (per-asset arg errors, unresolved cross-references, and graphics-rule
// violations) so the caller can report them in a single pass.
pub fn check_world_with(
    assets: &[WorldJsonlAsset],
    extra: ExtraAssetCheck,
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

    for asset in assets {
        let type_norm = asset.asset_type.to_lowercase().replace('_', "");
        if let Err(e) = check_asset(&type_norm, &asset.name, &asset.args) {
            errors.push(e);
        }
        if let Err(e) = extra(&type_norm, &asset.name, &asset.args) {
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

// `check_world_with` with no extra checks: the pure-validation subset. Callers
// that have the compilers available (cook) compose theirs in instead.
pub fn check_world(assets: &[WorldJsonlAsset]) -> Result<(), Vec<String>> {
    check_world_with(assets, &|_, _, _| Ok(()))
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
    fn graphics_config_without_vertex_stage_is_an_error() {
        let assets = vec![asset("gfx", "GraphicsConfig", serde_json::json!({}))];
        let errs = check_world(&assets).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("vertex ShaderStage")));
    }

    #[test]
    fn graphics_config_with_full_render_stack_passes_graphics_rules() {
        let assets = vec![
            asset("gfx", "GraphicsConfig", serde_json::json!({})),
            asset("win", "Window", serde_json::json!({})),
            asset(
                "vert",
                "ShaderStage",
                serde_json::json!({
                    "kind": "vertex",
                    "sources": {"metal": "x.metal", "hlsl": "x.hlsl", "glsl": "x.glsl"}
                }),
            ),
        ];
        assert!(check_world(&assets).is_ok());
    }

    #[test]
    fn per_asset_and_cross_reference_errors_both_collected() {
        // Prop with no mesh/model/prefab (per-asset error) plus a Material
        // with a missing albedo texture (cross-reference error).
        let assets = vec![
            asset("bad_prop", "Prop", serde_json::json!({})),
            asset("bad_mat", "Material", serde_json::json!({"albedo":"ghost"})),
        ];
        let errs = check_world(&assets).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad_prop")));
        assert!(errs.iter().any(|e| e.contains("ghost")));
    }

    #[test]
    fn extra_checks_fold_into_the_same_error_pass() {
        let assets = vec![asset("t", "Texture", serde_json::json!({}))];
        let errs = check_world_with(&assets, &|type_norm, name, _args| {
            if type_norm == "texture" {
                Err(format!("Asset '{name}': extra check fired"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert!(errs.iter().any(|e| e.contains("extra check fired")));
    }
}
