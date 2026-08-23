/// The build-time world surface: the expansion passes, preset loading, and the
/// build front-half orchestrator (prepare_world = load + expand + validate).
/// The authored world model itself -- world.jsonl I/O, `WorldJsonlAsset`,
/// $include resolution, and structural validation (`load_world`) -- lives in
/// concinnity-world and is re-exported here so cook code and downstream
/// consumers keep resolving `world::...` paths.
pub mod preset;

pub use concinnity_world::world::{
    WORLD_JSONL, WorldJsonlAsset, asset_name_from_path, find_world_jsonl, known_names, load_world,
    parse_world_jsonl, patch_world_jsonl, patch_world_jsonl_to, resolve_includes,
    write_world_jsonl,
};

pub(crate) mod application;
pub(crate) mod camera_shot;
pub(crate) mod character_model;
pub(crate) mod companion;
pub(crate) mod companion_specs;
pub(crate) mod defaults;

pub(crate) mod light_rig;
pub(crate) mod main_menu;
pub(crate) mod material_palette;
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
pub fn prepare_world(content: &str) -> Result<LoadedWorld, Vec<String>> {
    let mut expanded = load_world(content)?;
    let authored: Vec<String> = expanded
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
        .map(str::to_string)
        .collect();
    let report = expand_world(&mut expanded).map_err(|e| vec![e])?;

    let assets: Vec<WorldJsonlAsset> = expanded.iter().map(WorldJsonlAsset::from_value).collect();

    crate::check::check_world(&assets)?;

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

    // The model-layer tests (load_world, resolve_includes, asset_name_from_path)
    // live in concinnity-world with the code; this covers cook's front-half
    // orchestration on top of it.
    #[test]
    fn prepare_world_expands_and_validates() {
        let content = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#;
        let loaded = prepare_world(content).unwrap();
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
        let errs = prepare_world(content).err().unwrap_or_default();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("ghost"), "{errs:?}");
    }

    // Semantic validation runs on the expanded world, so a dangling reference
    // that survives expansion still fails the build.
    #[test]
    fn prepare_world_reports_semantic_errors() {
        let content = r#"{"name":"prop","type":"Prop","args":{"mesh":"nope"}}"#;
        let errs = prepare_world(content).err().unwrap_or_default();
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("nope")), "{errs:?}");
    }
}
