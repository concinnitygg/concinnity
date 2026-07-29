// Cook's semantic-validation surface: the pure checks live in
// concinnity-world (re-exported below); this module adds the per-asset checks
// that validate by running a compiler (mesh generators, texture generators,
// cubemap/environment-map sources) and composes both sets behind the same
// `check_asset` / `check_world` entry points callers have always used.

pub(crate) mod cubemap_texture;
pub(crate) mod environment_map;
pub(crate) mod mesh;
pub(crate) mod texture;

pub use concinnity_world::check::{behavior, cross_reference, fault, report_validation_errors};

use crate::world::WorldJsonlAsset;

// The compile-backed per-asset checks only cook can run. Fed into
// concinnity-world's `check_world_with` as the extra hook.
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

// The full per-asset check: the pure world-crate checks plus the
// compile-backed ones above.
pub fn check_asset(type_norm: &str, name: &str, args: &serde_json::Value) -> Result<(), String> {
    concinnity_world::check::check_asset(type_norm, name, args)?;
    check_compiled_asset(type_norm, name, args)
}

// Run all semantic validation on a fully expanded world: the world crate's
// pure checks, cross-references, and graphics rules, with the compile-backed
// checks folded into the same error-collection pass.
pub fn check_world(assets: &[WorldJsonlAsset]) -> Result<(), Vec<String>> {
    concinnity_world::check::check_world_with(assets, &check_compiled_asset)
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

    // The composed pass surfaces a compile-backed error (unknown texture
    // generator) alongside a pure world-crate error (Prop with no source) --
    // both check sets run in one collection.
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
        let errs = check_world(&assets).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad_prop")));
        assert!(errs.iter().any(|e| e.contains("not_a_generator")));
    }

    // Every spelling of a compile-backed asset type reaches the same check.
    #[test]
    fn check_asset_routes_each_type_alias() {
        for alias in ["cubemaptexture", "cubemap"] {
            let args = serde_json::json!({"source": "studio.png"});
            let err = check_asset(alias, "c", &args).unwrap_err();
            assert!(err.contains("Radiance .hdr"), "{alias}: {err}");
        }
        for alias in ["environmentmap", "envmap", "ibl"] {
            let args = serde_json::json!({"generator": "aurora"});
            let err = check_asset(alias, "e", &args).unwrap_err();
            assert!(
                err.contains("unknown EnvironmentMap generator"),
                "{alias}: {err}"
            );
        }
    }

    #[test]
    fn check_asset_runs_both_check_sets() {
        // A pure check arm.
        assert!(check_asset("prop", "p", &serde_json::json!({})).is_err());
        // A compile-backed arm.
        assert!(
            check_asset(
                "texture",
                "t",
                &serde_json::json!({"generator": "not_a_generator"})
            )
            .is_err()
        );
        // A type neither set knows is fine.
        assert!(check_asset("window", "w", &serde_json::json!({})).is_ok());
    }
}
