// Entry point for all build-time JSON-level world expansion.
// Operates purely on serde_json::Value; no type registry or blob compilation.

use std::path::Path;

use super::app_config::apply_app_config;
use super::camera_shot::expand_camera_shots;
use super::character_model::expand_character_models;
use super::companion::inject_companions;
use super::light_rig::expand_light_rigs;
use super::main_menu::expand_main_menus;
use super::material_palette::expand_material_palettes;
use super::menu_defaults::inject_menu_defaults;
use super::option_select::expand_option_selects;
use super::panel::expand_panels;
use super::prefab::expand_prefabs;
use super::room::expand_room_textures;
use super::scene_import::expand_scene_imports;
use super::slider::expand_sliders;
use super::story::expand_stories;
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::load_world;

// Shared helpers used across expansion submodules.

// The entry's registered type, or `None` unless its "type" is an exact name.
pub(crate) fn registered_type(v: &serde_json::Value) -> Option<RegisteredType> {
    v.get("type")
        .and_then(|t| t.as_str())
        .and_then(RegisteredType::parse)
}

pub(crate) fn asset_name(v: &serde_json::Value) -> String {
    asset_name_str(v).to_string()
}

// Borrowing form of `asset_name`, for scans that only compare.
pub(crate) fn asset_name_str(v: &serde_json::Value) -> &str {
    v.get("name").and_then(|n| n.as_str()).unwrap_or("")
}

/// One asset added to the world by an injection pass rather than authored or
/// macro-expanded. Recorded in world-lock.json so the user can see every
/// default and copy its entry into world.jsonl as an override.
#[derive(Debug, Clone)]
pub struct InjectedAsset {
    /// The injected asset's name.
    pub name: String,
    /// The asset's registry type name.
    pub asset_type: String,
    /// The args the injection supplied.
    pub args: serde_json::Value,
    /// The injection pass (an EngineDefaults flag name or "companion"), so
    /// listings can say where a default came from.
    pub injected_by: &'static str,
}

/// One asset a macro expansion produced from an authored entry, recorded so
/// listings can group generated assets by what produced them and offer to copy
/// one into world.jsonl as an override.
#[derive(Debug, Clone)]
pub struct GeneratedAsset {
    /// The generated asset's name.
    pub name: String,
    /// The asset's registry type name.
    pub asset_type: String,
    /// The authored asset that generated it (a SceneImport's name).
    pub generated_by: String,
}

/// One generated asset the world declares itself: the authored entry is a
/// sparse patch merged over the generated args (see `shadow::merge_args`), so a
/// line in world.jsonl overrides exactly the fields it names and tracks the
/// expansion for the rest. Recorded so listings can show the override for what
/// it is rather than leaving the generated asset unaccounted for.
#[derive(Debug, Clone)]
pub struct ShadowedAsset {
    /// The shadowed asset's name.
    pub name: String,
    /// The asset's registry type name.
    pub asset_type: String,
    /// The authored asset whose expansion it patches.
    pub generated_by: String,
    /// The args the expansion produced before the authored patch was merged:
    /// the template baseline a per-field override is measured against.
    pub args: serde_json::Value,
}

// What the expansion passes added, generated, and skipped during one run.
#[derive(Debug, Default)]
pub(crate) struct ExpandReport {
    pub injected: Vec<InjectedAsset>,
    pub generated: Vec<GeneratedAsset>,
    pub shadowed: Vec<ShadowedAsset>,
}

impl ExpandReport {
    pub(crate) fn record(
        &mut self,
        name: &str,
        asset_type: &str,
        args: serde_json::Value,
        injected_by: &'static str,
    ) {
        self.injected.push(InjectedAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
            injected_by,
        });
    }

    pub(crate) fn record_generated(&mut self, name: &str, asset_type: &str, generated_by: &str) {
        self.generated.push(GeneratedAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            generated_by: generated_by.to_string(),
        });
    }

    // Idempotent: a name can be checked by more than one pass (both HUDs test the
    // shared font), and the same override must not be listed twice. The first
    // record's args win: the earliest pass to produce the asset is its template.
    pub(crate) fn record_shadowed(
        &mut self,
        name: &str,
        asset_type: &str,
        generated_by: &str,
        args: serde_json::Value,
    ) {
        if self.shadowed.iter().any(|s| s.name == name) {
            return;
        }
        self.shadowed.push(ShadowedAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            generated_by: generated_by.to_string(),
            args,
        });
    }
}

// Run all expansion passes in order. Mutates the asset list in place and
// reports what the injection passes added. `assets_dir` is the asset search
// root the source-reading passes (scene imports, presets) resolve against.
// Returns an error only when a hard failure occurs (e.g. prefab cycle or
// missing prefab reference).
pub(crate) fn expand_world(
    assets: &mut Vec<serde_json::Value>,
    assets_dir: Option<&Path>,
) -> Result<ExpandReport, String> {
    let mut report = ExpandReport::default();
    // The assets the world declares itself, snapshotted before any pass runs:
    // a generated entry landing on one of these names is the user's patch of
    // it, while a collision with anything added later is a conflict between
    // two expansions.
    let authored: std::collections::HashMap<String, RegisteredType> = assets
        .iter()
        .filter_map(|v| Some((asset_name(v), registered_type(v)?)))
        .filter(|(n, _)| !n.is_empty())
        .collect();
    // Imports expand first so the assets they generate (materials, meshes,
    // props, a framed camera) flow through every later pass, including
    // companion injection.
    expand_scene_imports(assets, &mut report, assets_dir)?;
    // Stories expand to External UI assets (Screens, TextLabels, HitRegions)
    // that need no further expansion but must exist before companion
    // injection so their TextLabels pull in GraphicsConfig + Font companions.
    expand_stories(assets)?;
    expand_camera_shots(assets, assets_dir);
    // Character models become the skinned meshes they emit, under their own
    // names, so every later pass (companions, references) sees a SkinnedMesh.
    expand_character_models(assets)?;
    expand_light_rigs(assets, assets_dir);
    expand_material_palettes(assets, assets_dir);
    expand_prefabs(assets, &authored, &mut report, assets_dir)?;
    expand_room_textures(assets);
    // First companion round: materialize the GraphicsConfig render marker (and
    // its Window / Shader stack) implied by everything authored or
    // expanded above, so the defaults pass can key off "this world renders".
    inject_companions(assets, &mut report);
    // The AppConfig asset (at most one) names the world for distribution and,
    // when a Window authored no title, fills it so a running game shows its own
    // name. Runs after the first companion round so a rendering world's injected
    // Window is present to receive the title.
    apply_app_config(assets, &mut report)?;
    // The engine defaults stated in build-only terms: the StatHud a MainMenu
    // world drives, and a story world's pause MainMenu. Runs before menu
    // expansion so an injected MainMenu expands like an authored one. Every
    // other default is injected at world start.
    inject_menu_defaults(assets, &mut report)?;
    // Menus expand to External UI assets (Screen / Sprite / TextLabel /
    // HitRegion / KeyBinding) that need no further expansion, but whose
    // TextLabels must still pull in their GraphicsConfig + Font companions, so
    // this runs before the second companion round.
    expand_main_menus(assets)?;
    // Menus emit OptionSelect rows for their settings sub-screen; expand those to
    // their primitives (TextLabels + HitRegion) before companion injection so
    // the generated TextLabels pull in their Font.
    expand_option_selects(assets)?;
    // Menus also emit Slider rows (continuous settings); expand those to their
    // primitives (TextLabels + Sprites + HitRegion) on the same footing, before
    // companion injection.
    expand_sliders(assets)?;
    // Panels expand to a background Sprite (+ title TextLabel), also before the
    // second companion round so those pull in their GraphicsConfig / Font.
    expand_panels(assets)?;
    // Second companion round: companions for the assets the defaults and menu
    // passes added. Idempotent for everything round one already covered.
    inject_companions(assets, &mut report);
    Ok(report)
}

/// Load and structurally validate a world.jsonl string, then run all
/// expansion passes, resolving bare source filenames under `assets_dir`.
/// Returns the fully expanded asset list. Does not run semantic validation;
/// see `crate::build_only::prepare_world` for the full build-pipeline front half.
pub fn expand_world_from_str(
    content: &str,
    assets_dir: Option<&Path>,
) -> std::io::Result<Vec<serde_json::Value>> {
    let mut assets = load_world(content)
        .map_err(|errs| std::io::Error::new(std::io::ErrorKind::InvalidData, errs.join("\n")))?;

    let _ = expand_world(&mut assets, assets_dir)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    Ok(assets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_type_parses_an_exact_name() {
        let v = serde_json::json!({"type": "MaterialPalette"});
        assert_eq!(registered_type(&v), Some(RegisteredType::MaterialPalette));
        let v = serde_json::json!({"type": "Camera3D"});
        assert_eq!(registered_type(&v), Some(RegisteredType::Camera3D));
    }

    #[test]
    fn registered_type_rejects_inexact_spellings() {
        for ty in [
            "materialpalette",
            "Material_Palette",
            "camera3d",
            "CAMERA3D",
        ] {
            let v = serde_json::json!({"type": ty});
            assert_eq!(registered_type(&v), None, "{ty}");
        }
    }

    #[test]
    fn registered_type_missing_or_unknown_is_none() {
        assert_eq!(registered_type(&serde_json::json!({"name": "x"})), None);
        assert_eq!(
            registered_type(&serde_json::json!({"type": "Logger"})),
            None
        );
        assert_eq!(registered_type(&serde_json::json!({"type": 3})), None);
    }

    #[test]
    fn asset_name_extracts_name() {
        let v = serde_json::json!({"name": "my_asset", "type": "Logger"});
        assert_eq!(asset_name(&v), "my_asset");
    }

    #[test]
    fn asset_name_missing_returns_empty() {
        let v = serde_json::json!({"type": "Logger"});
        assert_eq!(asset_name(&v), "");
    }

    // Every pass's failure aborts the run and surfaces its own message, so a
    // broken entry is reported by the pass that understands it.
    #[test]
    fn a_failing_pass_aborts_the_whole_expansion() {
        for (asset, needle) in [
            (
                serde_json::json!({"name":"s","type":"SceneImport","args":{}}),
                "SceneImport 's': missing `source`",
            ),
            (
                serde_json::json!({"name":"t","type":"StoryImport","args":{}}),
                "StoryImport 't': missing `source`",
            ),
            (
                serde_json::json!({"name":"p","type":"Prop","args":{"prefab":"ghost"}}),
                "prefab 'ghost' not found",
            ),
            (
                serde_json::json!({"type":"MainMenu","args":{}}),
                "MainMenu: missing `name`",
            ),
            (
                serde_json::json!({"type":"OptionSelect","args":{}}),
                "OptionSelect: missing `name`",
            ),
            (
                serde_json::json!({"type":"Slider","args":{}}),
                "Slider: missing `name`",
            ),
            (
                serde_json::json!({"type":"Panel","args":{}}),
                "Panel: missing `name`",
            ),
        ] {
            let mut assets = vec![asset.clone()];
            let err = expand_world(&mut assets, None).unwrap_err();
            assert!(err.contains(needle), "{asset} -> {err}");
        }
    }

    #[test]
    fn a_second_engine_defaults_entry_aborts_the_expansion() {
        let mut assets = vec![
            serde_json::json!({"name":"a","type":"EngineDefaults","args":{}}),
            serde_json::json!({"name":"b","type":"EngineDefaults","args":{}}),
        ];
        let err = expand_world(&mut assets, None).unwrap_err();
        assert!(err.contains("at most one"), "{err}");
    }

    #[test]
    fn a_window_that_cannot_take_the_app_config_title_aborts_the_expansion() {
        let mut assets = vec![
            serde_json::json!({"name":"app","type":"AppConfig","args":{"name":"My Game"}}),
            serde_json::json!({"name":"win","type":"Window","args":[]}),
        ];
        let err = expand_world(&mut assets, None).unwrap_err();
        assert!(err.contains("Window 'win'"), "{err}");
        assert!(err.contains("args must be an object"), "{err}");
    }

    // The string entry point reports both the structural failures `load_world`
    // finds and the expansion failures that follow it.
    #[test]
    fn expand_world_from_str_surfaces_load_and_expansion_errors() {
        let malformed = expand_world_from_str("not json at all\n", None).unwrap_err();
        assert_eq!(malformed.kind(), std::io::ErrorKind::InvalidData);
        assert!(!malformed.to_string().is_empty());

        let broken = r#"{"name":"p","type":"Prop","args":{"prefab":"ghost"}}"#;
        let err = expand_world_from_str(broken, None).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("ghost"), "{err}");
    }

    // Every entry a generator emits or an injection adds must name its type
    // exactly, because the typed world after expansion accepts nothing else.
    // One world declares every build-only type plus a renderable asset; a
    // second leaves the pause menu to the story injection.
    #[test]
    fn every_expanded_entry_names_an_exact_registered_type() {
        use crate::authoring::registry::AssetOrigin;

        let dir = tempfile::tempdir().unwrap();
        let mut gltf = crate::import::glb::test_fixtures::static_triangle_json();
        gltf["buffers"][0]["uri"] = "geo.bin".into();
        std::fs::write(
            dir.path().join("geo.bin"),
            crate::import::glb::test_fixtures::static_triangle_bin(),
        )
        .unwrap();
        let scene = dir.path().join("tri.gltf");
        std::fs::write(&scene, serde_json::to_vec(&gltf).unwrap()).unwrap();
        let story = dir.path().join("story.md");
        std::fs::write(
            &story,
            "---\ntitle: Road\ncharacters:\n  guide: Guide\n---\n\n# start\n\n\
             **guide:** Which way?\n\n- [Left](#end)\n- [Right](#end)\n\n# end\n\nDone.\n",
        )
        .unwrap();
        let scene = scene.to_str().unwrap();
        let story = story.to_str().unwrap();

        let full = vec![
            serde_json::json!({"name":"box_mesh","type":"ProceduralMesh","args":{"generator":"box"}}),
            serde_json::json!({"name":"stone","type":"Material","args":{}}),
            serde_json::json!({"name":"rig","type":"LightRig","args":{"preset":"rig_outdoor_sun_fill"}}),
            serde_json::json!({"name":"palette","type":"MaterialPalette","args":{"entries":[{"alias":"rock"}]}}),
            serde_json::json!({"name":"shot","type":"CameraShot","args":{"position":[0,2,6]}}),
            serde_json::json!({"name":"crate_prefab","type":"Prefab","args":{"props":[
                {"kind":"prop","name":"body","mesh":"box_mesh","material":"stone"},
                {"kind":"point_light","name":"lamp","position":[0,1,0]}]}}),
            serde_json::json!({"name":"crate_a","type":"Prop","args":{"prefab":"crate_prefab"}}),
            serde_json::json!({"name":"imported","type":"SceneImport","args":{"source":scene}}),
            serde_json::json!({"name":"pause","type":"MainMenu","args":{}}),
            serde_json::json!({"name":"settings","type":"Panel","args":{"title":"Settings"}}),
            serde_json::json!({"name":"exposure","type":"Slider","args":{"setting":"exposure","label":"Exposure"}}),
            serde_json::json!({"name":"vsync","type":"OptionSelect","args":{"setting":"vsync","label":"Vsync"}}),
            serde_json::json!({"name":"tale","type":"StoryImport","args":{"source":story}}),
            serde_json::json!({"name":"sk","type":"CharacterSchema","args":{
                "joints":[{"name":"root"}],"regions":[{"name":"all","joints":["root"]}]}}),
            serde_json::json!({"name":"body_model","type":"CharacterModel","args":{"schema":"sk","source":"hero.glb"}}),
            serde_json::json!({"name":"hall","type":"Room","args":{"wall_texture":"brick"}}),
            serde_json::json!({"name":"app","type":"AppConfig","args":{"name":"Typed"}}),
            serde_json::json!({"name":"defaults","type":"EngineDefaults","args":{}}),
        ];
        let story_only =
            vec![serde_json::json!({"name":"tale","type":"StoryImport","args":{"source":story}})];

        for world in [full, story_only] {
            let mut assets = world;
            let report = expand_world(&mut assets, None).expect("the world expands");
            assert!(!report.injected.is_empty());
            for v in &assets {
                let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let parsed = RegisteredType::parse(ty)
                    .unwrap_or_else(|| panic!("'{}' has inexact type '{ty}'", asset_name(v)));
                // A schema stays in the world for the character bake to read.
                assert!(
                    parsed == RegisteredType::CharacterSchema
                        || parsed.registration().origin != AssetOrigin::BuildOnly,
                    "'{}' survived expansion as {ty}",
                    asset_name(v)
                );
            }
        }
    }

    #[test]
    fn expand_world_from_str_injects_companions() {
        let content = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#;
        let assets = expand_world_from_str(content, None).unwrap();
        assert!(
            assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::GraphicsConfig))
        );
        // GraphicsConfig pulls in a Window companion.
        assert!(
            assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::Window))
        );
    }

    #[test]
    fn bare_main_menu_world_expands_and_pulls_companions() {
        let content = r#"{"name":"main_menu","type":"MainMenu"}"#;
        let assets = expand_world_from_str(content, None).unwrap();
        // The MainMenu is gone, replaced by its UI assets.
        assert!(
            !assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::MainMenu))
        );
        assert!(
            assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::Screen))
        );
        assert!(
            assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::HitRegion))
        );
        // The generated TextLabels pull in GraphicsConfig + a Font companion.
        assert!(
            assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::TextLabel))
        );
        assert!(
            assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::GraphicsConfig))
        );
        assert!(
            assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::Font))
        );
    }
}
