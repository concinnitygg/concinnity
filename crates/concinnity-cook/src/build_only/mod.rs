/// The build-time world surface: the expansion passes, preset loading, and the
/// build front-half orchestrator (prepare_world = load + expand + validate).
/// The authored world model it works on -- world.jsonl I/O, `WorldJsonlAsset`,
/// and structural validation (`load_world`) -- is `crate::authoring::world`.
pub mod preset;

use crate::authoring::world::{WorldJsonlAsset, WorldSource, load_world};

pub(crate) mod app_config;
pub(crate) mod camera_shot;
pub(crate) mod character_model;
pub(crate) mod companion;
pub(crate) mod companion_specs;
pub mod include;

pub(crate) mod light_rig;
pub(crate) mod main_menu;
pub(crate) mod material_palette;
pub(crate) mod membership;
pub(crate) mod menu_defaults;
pub(crate) mod option_select;
pub(crate) mod panel;
pub(crate) mod prefab;
pub(crate) mod room;
pub(crate) mod row_setting;
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
    /// The handles of the entries in the world file itself (pre-expansion),
    /// for provenance listings: each `$id`, else the entry's label.
    pub authored: Vec<String>,
}

/// Run the read-only front half of the build pipeline: parse and structurally
/// validate the world (`load_world`), expand all build-time assets, then run
/// semantic validation (`crate::check::check_world`). Returns everything the
/// compile stage needs, computed exactly once. Errors from every stage are
/// collected, so the caller gets the full picture in a single pass.
///
/// `source` is the world text, with the file it was read from when an
/// `Include` in it should resolve beside that file. `assets_dir` is the asset
/// search root the expansion passes resolve bare source filenames and preset
/// names against; `None` leaves them unresolved.
pub fn prepare_world<'a>(
    source: impl Into<WorldSource<'a>>,
    assets_dir: Option<&std::path::Path>,
) -> Result<LoadedWorld, Vec<String>> {
    let mut expanded = load_world(source.into())?;
    let authored: Vec<String> = expanded
        .iter()
        .filter_map(crate::authoring::world::entry_id)
        .map(str::to_string)
        .collect();
    let report = expand_world(&mut expanded, assets_dir).map_err(|e| vec![e])?;
    // The expansion is the work this half of the build produces cache entries
    // for, so its segment is written here rather than left to a compile that a
    // check-only run never reaches.
    crate::cache::flush();

    let assets = typed_assets(&expanded)?;

    crate::check::check_world(&assets)?;

    Ok(LoadedWorld {
        assets,
        injected: report.injected,
        generated: report.generated,
        shadowed: report.shadowed,
        authored,
    })
}

// Type every expanded entry, reporting each one whose name or type does not
// parse, so a generated entry with a bad type fails instead of matching nothing.
fn typed_assets(values: &[serde_json::Value]) -> Result<Vec<WorldJsonlAsset>, Vec<String>> {
    let mut errors = Vec::new();
    let assets = values
        .iter()
        .filter_map(|v| {
            WorldJsonlAsset::from_value(v)
                .map_err(|e| errors.push(e))
                .ok()
        })
        .collect();
    if errors.is_empty() {
        Ok(assets)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::registry::RegisteredType;

    #[test]
    fn typed_assets_reports_every_generated_entry_with_a_bad_type() {
        let expanded = [
            serde_json::json!({"type": "Prop", "args": {"$id": "ok"}}),
            serde_json::json!({"type": "pointlight", "args": {"$id": "gen_a"}}),
            serde_json::json!({"type": "Point_Light", "args": {"$id": "gen_b"}}),
        ];
        let errs = typed_assets(&expanded).err().unwrap_or_default();
        assert_eq!(errs.len(), 2, "{errs:?}");
        assert!(errs[0].contains("'gen_a'") && errs[0].contains("'pointlight'"));
        assert!(errs[1].contains("'gen_b'") && errs[1].contains("'Point_Light'"));
    }

    // The model-layer tests (load_world, resolve_includes, asset_name_from_path)
    // live in `crate::authoring::world` with the code; this covers cook's
    // front-half orchestration on top of it.
    #[test]
    fn prepare_world_expands_and_validates() {
        let content = r#"["GraphicsConfig",{"$id":"gfx"}]"#;
        let loaded = prepare_world(content, None).unwrap();
        // GraphicsConfig pulls in its companions, so the prepared world holds
        // more than the single declared asset.
        assert!(loaded.assets.len() > 1);
        assert!(
            loaded
                .assets
                .iter()
                .any(|a| a.asset_type == RegisteredType::GraphicsConfig)
        );
        // The authored names are captured before expansion, so the injected
        // companions are not mistaken for what the world declared.
        assert_eq!(loaded.authored, vec!["gfx".to_string()]);
    }

    // An expansion failure is reported as the single error it is, rather than
    // being swallowed on the way to semantic validation.
    #[test]
    fn prepare_world_reports_an_expansion_failure() {
        let content = r#"["Prop",{"$id":"p","prefab":"ghost"}]"#;
        let errs = prepare_world(content, None).err().unwrap_or_default();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("ghost"), "{errs:?}");
    }

    // Semantic validation runs on the expanded world, so a dangling reference
    // that survives expansion still fails the build.
    #[test]
    fn prepare_world_reports_semantic_errors() {
        let content = r#"["Prop",{"$id":"prop","mesh":"nope"}]"#;
        let errs = prepare_world(content, None).err().unwrap_or_default();
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
                .find(|a| a.id == "rig_key")?
                .args
                .get("intensity")?
                .as_f64()
        }

        let content = r#"["LightRig",{"$id":"rig","preset":"dusk"}]"#;
        let bright = rig_root(3.5);
        let dim = rig_root(0.25);

        assert_eq!(
            key_intensity(&prepare_world(content, Some(bright.path())).unwrap()),
            Some(3.5)
        );
        assert_eq!(
            key_intensity(&prepare_world(content, Some(dim.path())).unwrap()),
            Some(0.25)
        );
        // No root: the preset is never found, so the rig expands to nothing.
        assert_eq!(key_intensity(&prepare_world(content, None).unwrap()), None);
    }
}
