/// The build-time world surface: the expansion passes, preset loading, and the
/// build front-half orchestrator (prepare_world = load + expand + validate).
/// The authored world model it works on -- world.jsonl I/O, `WorldJsonlAsset`,
/// $include resolution, and structural validation (`load_world`) -- is
/// `crate::authoring::world`.
pub mod preset;

use crate::authoring::world::{WorldJsonlAsset, load_world};

pub(crate) mod app_config;
pub(crate) mod camera_shot;
pub(crate) mod character_model;
pub(crate) mod companion;
pub(crate) mod companion_specs;

pub(crate) mod light_rig;
pub(crate) mod main_menu;
pub(crate) mod material_palette;
pub(crate) mod menu_defaults;
pub(crate) mod option_select;
pub(crate) mod panel;
pub(crate) mod prefab;
pub(crate) mod room;
pub(crate) mod scene_import;
pub(crate) mod slider;
pub(crate) mod story;
pub use story::validate_story_source;
pub(crate) mod ui_spec;

pub(crate) mod expand;
mod provenance;
pub use provenance::Provenance;
pub(crate) mod shadow;
pub use shadow::merge_args;

pub(crate) use expand::expand_world;
pub use expand::{GeneratedAsset, InjectedAsset, ShadowedAsset, expand_world_from_str};
/// A world.jsonl that has been loaded, structurally validated, expanded, and
/// semantically checked: everything the compile stage needs, computed once.
pub struct LoadedWorld {
    /// The same assets as typed entries, consumed by the build pipeline.
    pub assets: Vec<WorldJsonlAsset>,
    /// Assets added by the injection passes (companions, engine defaults),
    /// recorded in world-lock.json so the user can see and override them.
    pub injected: Vec<InjectedAsset>,
    /// Assets a macro expansion produced, paired with the authored asset that
    /// produced them, so listings can group them by source.
    pub generated: Vec<GeneratedAsset>,
    /// Generated assets the world declares a patch of; the merged result is in
    /// `assets` and each record carries the pre-merge generated args.
    pub shadowed: Vec<ShadowedAsset>,
    /// Names declared in the world file itself (pre-expansion), for
    /// provenance listings.
    pub authored: Vec<String>,
}

/// Run the read-only front half of the build pipeline: parse and structurally
/// validate the world (`load_world`), expand all build-time assets, then run
/// semantic validation (`crate::check::check_world`). Returns everything the
/// compile stage needs, computed exactly once. Errors from every stage are
/// collected, so the caller gets the full picture in a single pass.
///
/// `assets_dir` is the asset search root the expansion passes resolve bare
/// source filenames and preset names against; `None` leaves them unresolved.
/// `platform` is the shader platform the world is cooked for; the shader-backed
/// types are validated against it.
pub fn prepare_world(
    content: &str,
    assets_dir: Option<&std::path::Path>,
    platform: concinnity_core::platform::Platform,
) -> Result<LoadedWorld, Vec<String>> {
    let mut expanded = load_world(content)?;
    let authored: Vec<String> = expanded
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
        .map(str::to_string)
        .collect();
    let report = expand_world(&mut expanded, assets_dir).map_err(|e| vec![e])?;
    // The expansion is the work this half of the build produces cache entries
    // for, so its segment is written here rather than left to a compile that a
    // check-only run never reaches.
    crate::cache::flush();

    let assets: Vec<WorldJsonlAsset> = expanded.iter().map(WorldJsonlAsset::from_value).collect();

    crate::check::check_world(&assets, platform)?;

    Ok(LoadedWorld {
        assets,
        injected: report.injected,
        generated: report.generated,
        shadowed: report.shadowed,
        authored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::platform::Platform;

    // The model-layer tests (load_world, resolve_includes, asset_name_from_path)
    // live in `crate::authoring::world` with the code; this covers cook's
    // front-half orchestration on top of it.
    #[test]
    fn prepare_world_expands_and_validates() {
        let content = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#;
        let loaded = prepare_world(content, None, Platform::Metal).unwrap();
        // GraphicsConfig pulls in its companions, so the prepared world holds
        // more than the single declared asset.
        assert!(loaded.assets.len() > 1);
        assert!(
            loaded
                .assets
                .iter()
                .any(|a| a.asset_type == "GraphicsConfig")
        );
        // The authored names are captured before expansion, so the injected
        // companions are not mistaken for what the world declared.
        assert_eq!(loaded.authored, vec!["gfx".to_string()]);
    }

    // An expansion failure is reported as the single error it is, rather than
    // being swallowed on the way to semantic validation.
    #[test]
    fn prepare_world_reports_an_expansion_failure() {
        let content = r#"{"name":"p","type":"Prop","args":{"prefab":"ghost"}}"#;
        let errs = prepare_world(content, None, Platform::Metal)
            .err()
            .unwrap_or_default();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("ghost"), "{errs:?}");
    }

    // Semantic validation runs on the expanded world, so a dangling reference
    // that survives expansion still fails the build.
    #[test]
    fn prepare_world_reports_semantic_errors() {
        let content = r#"{"name":"prop","type":"Prop","args":{"mesh":"nope"}}"#;
        let errs = prepare_world(content, None, Platform::Metal)
            .err()
            .unwrap_or_default();
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("nope")), "{errs:?}");
    }

    // The asset search root is the caller's, so one world expands differently
    // under two roots in the same process, and under none it falls back to the
    // type defaults. This is what the root being a parameter rather than a
    // process-wide anchor buys.
    #[test]
    fn prepare_world_expands_presets_from_the_root_it_is_given() {
        fn rig_root(intensity: f64) -> tempfile::TempDir {
            let dir = tempfile::tempdir().unwrap();
            let rigs = dir.path().join("light_rigs");
            std::fs::create_dir_all(&rigs).unwrap();
            std::fs::write(
                rigs.join("dusk.json"),
                serde_json::to_vec(&serde_json::json!({
                    "args": {"lights": [{"kind": "directional", "name": "key", "intensity": intensity}]}
                }))
                .unwrap(),
            )
            .unwrap();
            dir
        }
        fn key_intensity(loaded: &LoadedWorld) -> Option<f64> {
            loaded
                .assets
                .iter()
                .find(|a| a.name == "rig_key")?
                .args
                .get("intensity")?
                .as_f64()
        }

        let content = r#"{"name":"rig","type":"LightRig","args":{"preset":"dusk"}}"#;
        let bright = rig_root(3.5);
        let dim = rig_root(0.25);

        assert_eq!(
            key_intensity(&prepare_world(content, Some(bright.path()), Platform::Metal).unwrap()),
            Some(3.5)
        );
        assert_eq!(
            key_intensity(&prepare_world(content, Some(dim.path()), Platform::Metal).unwrap()),
            Some(0.25)
        );
        // No root: the preset is never found, so the rig expands to nothing.
        assert_eq!(
            key_intensity(&prepare_world(content, None, Platform::Metal).unwrap()),
            None
        );
    }
}
