// Build-time expansion: CharacterModel -> a SkinnedMesh of the same name. The
// placement, material, capsule, LOD request and spawn reserve move across as
// they are; the schema (resolved here, so an unknown one fails the
// expansion) and the source ride in a `character_model` arg the skinned-mesh
// import pass turns into geometry. Naming the mesh after the model is what lets a
// CharacterShape or Animation target either.

use super::expand::{asset_name, type_norm};
use crate::authoring::registry::build_only::CharacterModel;
use crate::authoring::world::WorldJsonlAsset;
use crate::compile::character::import::CharacterModelArg;

// The fields a CharacterModel shares with the SkinnedMesh it becomes.
const PASSED_THROUGH: [&str; 8] = [
    "material",
    "position",
    "rotation_deg",
    "scale",
    "lod_levels",
    "lod_distances",
    "max_instances",
    "capsule",
];

pub(crate) fn expand_character_models(
    asset_values: &mut Vec<serde_json::Value>,
) -> Result<(), String> {
    let schemas: Vec<WorldJsonlAsset> = asset_values
        .iter()
        .filter(|v| type_norm(v) == "characterschema")
        .map(WorldJsonlAsset::from_value)
        .collect();
    let mut result = Vec::with_capacity(asset_values.len());
    for value in asset_values.drain(..) {
        if type_norm(&value) != "charactermodel" {
            result.push(value);
            continue;
        }
        let name = asset_name(&value);
        let args = value
            .get("args")
            .and_then(|a| a.as_object())
            .cloned()
            .unwrap_or_default();
        let mut model_args = args.clone();
        // The material reference resolves on the emitted mesh, not here.
        model_args.remove("material");
        let model: CharacterModel = serde_json::from_value(serde_json::Value::Object(model_args))
            .map_err(|e| format!("CharacterModel '{name}': {e}"))?;
        let arg = CharacterModelArg::resolve(&name, model, &schemas)?;
        let mut mesh = serde_json::Map::new();
        for key in PASSED_THROUGH {
            if let Some(v) = args.get(key) {
                mesh.insert(key.to_string(), v.clone());
            }
        }
        mesh.insert(
            "character_model".to_string(),
            serde_json::to_value(&arg).map_err(|e| format!("CharacterModel '{name}': {e}"))?,
        );
        result.push(serde_json::json!({
            "name": name,
            "type": "SkinnedMesh",
            "args": serde_json::Value::Object(mesh),
        }));
    }
    *asset_values = result;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_becomes_a_same_named_skinned_mesh_with_its_schema_inlined() {
        let mut assets = vec![
            serde_json::json!({"name": "sk", "type": "CharacterSchema",
                "args": {"joints": [{"name": "root"}], "regions": [{"name": "all", "joints": ["root"]}]}}),
            serde_json::json!({"name": "body", "type": "CharacterModel", "args": {
                "schema": "sk", "material": "skin", "position": [0, 1, 0],
                "source": "hero.glb", "lod_levels": 2,
                "capsule": {"half_height": 0.8, "radius": 0.3}, "max_instances": 2}}),
            serde_json::json!({"name": "other", "type": "Prop", "args": {}}),
        ];
        expand_character_models(&mut assets).expect("expand");
        assert_eq!(assets.len(), 3);
        let mesh = &assets[1];
        assert_eq!(mesh["name"], "body");
        assert_eq!(mesh["type"], "SkinnedMesh");
        let args = &mesh["args"];
        assert_eq!(args["material"], "skin");
        assert_eq!(args["position"], serde_json::json!([0, 1, 0]));
        assert_eq!(args["capsule"]["radius"], 0.3);
        assert_eq!(args["max_instances"], 2);
        assert_eq!(args["lod_levels"], 2);
        assert!(args.get("source").is_none());
        let arg: CharacterModelArg =
            serde_json::from_value(args["character_model"].clone()).unwrap();
        assert_eq!(arg.schema.joints[0].name, "root");
        assert_eq!(arg.model.source, "hero.glb");
        assert_eq!(assets[2]["type"], "Prop");
    }

    #[test]
    fn an_unknown_schema_or_bad_args_fail_the_expansion() {
        let mut assets = vec![serde_json::json!({"name": "body", "type": "CharacterModel",
            "args": {"schema": "ghost", "source": "a.glb"}})];
        let err = expand_character_models(&mut assets).unwrap_err();
        assert!(err.contains("'ghost' is not a CharacterSchema"), "{err}");
        let mut assets = vec![serde_json::json!({"name": "body", "type": "CharacterModel",
            "args": {"source": 7}})];
        let err = expand_character_models(&mut assets).unwrap_err();
        assert!(err.starts_with("CharacterModel 'body':"), "{err}");
    }
}
