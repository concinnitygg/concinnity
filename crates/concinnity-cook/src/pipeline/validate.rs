//! The compile-free half of the pipeline: resolve every asset's type and args
//! and run the structural checks, without producing a payload.

use std::path::Path;

use crate::asset_api::{self, AssetRequest};
use crate::ecs::asset_id;

use super::errors_to_io;

/// Validate a single asset's type and generator without running the full build
/// pipeline. Called by the server on each world_add so the LLM gets per-asset
/// feedback without waiting for a WebSocket round-trip.
///
/// Checks:
///
/// - asset type is registered (via `asset_api::create_asset_def`)
/// - per-type structural checks via `crate::check`
///
/// Shader assets are not compiled here; use the validate_shader tool for that.
pub fn validate_asset(
    asset_type: &str,
    name: &str,
    args: &serde_json::Value,
) -> Result<(), String> {
    // Single-asset validation has no surrounding world to intern against; the
    // resulting ids are throwaway. Reset so calls do not accumulate entries.
    // Clear the resource handle map too: with no world there are no handles, so
    // a resource reference falls back to the interner (parses without resolving
    // to a real slot, which single-asset validation never needs).
    asset_id::reset_interner();
    crate::resource_handles::reset_resource_handles();
    let type_norm = asset_type.to_lowercase().replace('_', "");

    // Build-time types are valid in world.jsonl; they are consumed by expansion
    // functions before the runtime asset registry sees them.
    if matches!(
        type_norm.as_str(),
        "environment"
            | "lightrig"
            | "materialpalette"
            | "camerashot"
            | "prefab"
            | "sceneimport"
            | "characterschema"
            | "charactermodel"
    ) {
        return Ok(());
    }

    // A resource asset never builds a component def; validate it as a known type
    // with a structural check instead of routing through `create_asset_def`.
    if crate::registry::RegisteredType::parse(asset_type).is_some_and(|t| t.is_resource()) {
        crate::check::check_asset(&type_norm, name, args)?;
        return Ok(());
    }

    let req = AssetRequest {
        asset_type: asset_type.to_string(),
        args: Some(args.clone()),
    };
    asset_api::create_asset_def(&req).map_err(|e| format!("Asset '{}': {}", name, e))?;

    crate::check::check_asset(&type_norm, name, args)?;

    Ok(())
}

/// Validate world JSONL without running compilation. Runs the full front half
/// of the pipeline (load, expand, semantic checks) plus a per-asset type/args
/// resolution, but stops short of compiling payloads: intended for fast
/// server-side pre-deploy checks where shader compilation is not needed.
/// `assets_dir` is the asset search root the expansion passes resolve their
/// sources and presets against. Every problem found is reported in a single
/// newline-joined error.
pub fn validate_world_jsonl(content: &str, assets_dir: Option<&Path>) -> std::io::Result<()> {
    let loaded = crate::build_only::prepare_world(content, assets_dir).map_err(errors_to_io)?;

    let mut errors: Vec<String> = Vec::new();
    for asset in &loaded.assets {
        // A resource asset does not build a component def, so skip the component
        // resolution for it.
        if crate::registry::RegisteredType::parse(&asset.asset_type)
            .is_some_and(|t| t.is_resource())
        {
            continue;
        }
        let req = AssetRequest {
            asset_type: asset.asset_type.clone(),
            args: Some(asset.args.clone()),
        };
        if let Err(e) = asset_api::create_asset_def(&req) {
            errors.push(format!("Asset '{}': {}", asset.name, e));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors_to_io(errors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The visual_novel demo world (in concinnity-infra/worlds) exercises
    // Sprite + Screen + KeyBinding together. Validating it here catches asset
    // registration / pipeline regressions before we ship the world.
    #[test]
    fn visual_novel_world_validates() {
        // Inline a representative subset of the world so the test stays
        // hermetic (no infra path lookup needed). Covers: an initial Screen,
        // a Sprite under that screen's prefix, a TextLabel under it, a
        // HitRegion firing screen:show on another Screen, and a KeyBinding to
        // toggle a third (modal) Screen.
        let world = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}
{"name":"f","type":"Font","args":{"size_px":20}}
{"name":"title_menu","type":"Screen","args":{"initial":true}}
{"name":"title_menu_bg","type":"Sprite","args":{"x":0,"y":0,"width":640,"height":360,"tint":[0.1,0.1,0.1,1]}}
{"name":"title_menu_lbl","type":"TextLabel","args":{"font":"f","content":"Start","x":260,"y":160}}
{"name":"title_menu_btn","type":"HitRegion","args":{"x":260,"y":156,"width":120,"height":40,"label":"title_menu_lbl","action":"screen:show:vn_page_1"}}
{"name":"vn_page_1","type":"Screen","args":{}}
{"name":"vn_page_1_text","type":"TextLabel","args":{"font":"f","content":"hello","x":40,"y":40}}
{"name":"vn_page_1_next","type":"HitRegion","args":{"x":0,"y":0,"width":640,"height":360,"action":"screen:show:title_menu"}}
{"name":"pause_menu","type":"Screen","args":{}}
{"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":640,"height":360,"tint":[0,0,0,0.6]}}
{"name":"esc","type":"KeyBinding","args":{"key":"Escape","action":"screen:toggle:pause_menu"}}
"#;
        validate_world_jsonl(world, None).expect("visual_novel-shaped world should validate");
    }

    #[test]
    fn validate_asset_accepts_build_time_expansion_types() {
        // Build-time types are expanded before the runtime registry sees
        // them, so they validate structurally regardless of args.
        for ty in [
            "SceneImport",
            "Environment",
            "LightRig",
            "Prefab",
            "CharacterSchema",
            "CharacterModel",
        ] {
            validate_asset(ty, "x", &serde_json::json!({}))
                .unwrap_or_else(|e| panic!("{ty} should validate: {e}"));
        }
    }

    // A resource-only type never builds a component def, so it is validated
    // through the structural check alone rather than `create_asset_def`.
    #[test]
    fn validate_asset_routes_resource_only_types_past_the_component_registry() {
        validate_asset("AudioClip", "clip", &serde_json::json!({"source": "a.wav"}))
            .expect("a source-backed AudioClip validates");
        let err = validate_asset("Texture", "tex", &serde_json::json!({"generator": "nope"}))
            .expect_err("an unknown texture generator is rejected");
        assert!(err.contains("nope"), "got: {err}");
    }

    // A type that resolves through `create_asset_def` still has to satisfy its
    // structural check, and a clean asset returns Ok.
    #[test]
    fn validate_asset_runs_the_structural_check_after_type_resolution() {
        validate_asset("Scene", "day", &serde_json::json!({})).expect("a Scene validates");
        // A Prop resolves as a type but has no mesh source to render.
        let err = validate_asset("Prop", "empty_prop", &serde_json::json!({}))
            .expect_err("a source-less Prop is rejected");
        assert!(err.contains("empty_prop"), "got: {err}");
    }

    #[test]
    fn validate_asset_unknown_type_mentions_the_asset_name() {
        let err =
            validate_asset("Bogus", "my_thing", &serde_json::json!({})).expect_err("unknown type");
        assert!(err.contains("my_thing"), "got: {err}");
    }

    #[test]
    fn validate_asset_bad_args_mention_the_asset_name() {
        // `generator` must be a string; a number fails args deserialization.
        let err = validate_asset(
            "ProceduralMesh",
            "bad_mesh",
            &serde_json::json!({"generator": 5}),
        )
        .expect_err("bad args");
        assert!(err.contains("bad_mesh"), "got: {err}");
    }

    // The per-asset resolution pass reports every asset that fails, not just
    // the first, and a clean world returns Ok.
    #[test]
    fn validate_world_jsonl_collects_every_resolution_failure() {
        let world = concat!(
            r#"{"name":"first","type":"ProceduralMesh","args":{"generator":"box"}}"#,
            "\n",
            r#"{"name":"clip","type":"AudioClip","args":{"source":"a.wav"}}"#,
            "\n",
        );
        validate_world_jsonl(world, None).expect("a resolvable world validates");

        // Args of the wrong shape survive the structural world checks and are
        // rejected when the def is built.
        let bad = concat!(
            r#"{"name":"t1","type":"PointLight","args":{"intensity":"soon"}}"#,
            "\n",
            r#"{"name":"t2","type":"PointLight","args":{"intensity":"later"}}"#,
            "\n",
        );
        let err = validate_world_jsonl(bad, None).expect_err("mistyped args do not resolve");
        let msg = err.to_string();
        assert!(msg.contains("Asset 't1'"), "got: {msg}");
        assert!(msg.contains("Asset 't2'"), "got: {msg}");
    }
}
