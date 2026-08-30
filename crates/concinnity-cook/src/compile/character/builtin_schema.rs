// The schema the cook ships for its bundled base humanoid, addressed by the
// reserved name `builtin:humanoid`, and schema lookup by name against a
// world's CharacterSchema assets.

use crate::authoring::registry::build_only::CharacterSchema;
use crate::authoring::world::WorldJsonlAsset;
use std::sync::OnceLock;

/// `CharacterModel.schema` value of the bundled humanoid schema.
pub const HUMANOID_SCHEMA: &str = "builtin:humanoid";

const HUMANOID_JSON: &str = include_str!("../../../assets/humanoid_schema.json");

/// The bundled humanoid schema, parsed once per process.
pub fn humanoid() -> &'static CharacterSchema {
    static SCHEMA: OnceLock<CharacterSchema> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        serde_json::from_str(HUMANOID_JSON).expect("the bundled humanoid schema parses")
    })
}

/// The schema `name` refers to: the bundled one for its reserved name, else
/// the `CharacterSchema` asset of that name in `assets`.
pub(crate) fn resolve(name: &str, assets: &[WorldJsonlAsset]) -> Result<CharacterSchema, String> {
    if name == HUMANOID_SCHEMA {
        return Ok(humanoid().clone());
    }
    let asset = assets
        .iter()
        .find(|a| a.name == name && a.asset_type.eq_ignore_ascii_case("CharacterSchema"))
        .ok_or_else(|| {
            format!("schema '{name}' is not a CharacterSchema asset in this world (or {HUMANOID_SCHEMA})")
        })?;
    serde_json::from_value(asset.args.clone()).map_err(|e| format!("CharacterSchema '{name}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_schema_is_consistent_and_describes_the_base_humanoid() {
        let s = humanoid();
        assert!(
            s.consistency_errors().is_empty(),
            "{:?}",
            s.consistency_errors()
        );
        assert_eq!(s.joints.len(), 25);
        assert_eq!(s.keys.len(), 13);
        assert_eq!(s.required_target_names().len(), 22);
        assert_eq!(s.proportion_groups.len(), 10);
        assert!(s.synthesized.len() >= 25);
        assert_eq!(s.panel.len(), 5);
        assert_eq!(s.presets.len(), 4);
        // Every generator the runner knows appears at least once.
        for g in [
            "girth",
            "taper",
            "bulge",
            "mirror",
            "blend_mask",
            "surface_offset",
        ] {
            assert!(s.synthesized.iter().any(|t| t.generator == g), "{g}");
        }
        // Every key and group sits in a region some section shows.
        let shown: Vec<&str> = s
            .panel
            .iter()
            .flat_map(|p| p.regions.iter().map(String::as_str))
            .collect();
        for key in s.all_keys() {
            assert!(shown.contains(&key.region.as_str()), "{}", key.name);
        }
        for group in &s.proportion_groups {
            assert!(shown.contains(&group.region.as_str()), "{}", group.name);
        }
    }

    #[test]
    fn schemas_resolve_by_reserved_or_asset_name() {
        let assets = vec![WorldJsonlAsset {
            name: "mine".into(),
            asset_type: "CharacterSchema".into(),
            args: serde_json::json!({"joints": [{"name": "root"}]}),
        }];
        assert_eq!(resolve(HUMANOID_SCHEMA, &assets).unwrap().joints.len(), 25);
        assert_eq!(resolve("mine", &assets).unwrap().joints.len(), 1);
        let err = resolve("other", &assets).unwrap_err();
        assert!(err.contains("'other' is not a CharacterSchema"), "{err}");
    }
}
