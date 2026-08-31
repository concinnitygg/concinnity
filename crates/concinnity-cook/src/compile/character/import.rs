// Builds a CharacterModel's geometry: the source is imported, checked against
// the schema, and given the schema's synthesized targets. The result is an
// ordinary skinned import; lower levels of detail come from the skinned
// mesh's own `lod_levels` decimation, which keeps every vertex (so every
// target and skin weight) and only shortens the index list.

use std::path::Path;

use super::builtin_schema;
use super::synthesize::{MorphSet, synthesize};
use super::validate;
use crate::authoring::registry::build_only::CharacterModel;
use crate::authoring::registry::build_only::CharacterSchema;
use crate::authoring::world::WorldJsonlAsset;
use crate::import::glb::{ImportedSkinnedMesh, import_skinned_from_doc};
use crate::import::gltf_source::GltfDoc;

// Import `model`'s source against `schema`.
pub(crate) fn import_model(
    name: &str,
    schema: &CharacterSchema,
    model: &CharacterModel,
    assets_dir: Option<&Path>,
) -> Result<ImportedSkinnedMesh, String> {
    let errors = validate::model_errors(model);
    if !errors.is_empty() {
        return Err(format!("CharacterModel '{name}': {}", errors.join("; ")));
    }
    let source = &model.source;
    let doc = GltfDoc::parse_file(&crate::import::glb::resolve_source(source, assets_dir))?;
    let mut mesh = import_skinned_from_doc(&doc, source, model.skin_index)?;
    let errors = validate::source_errors(schema, &mesh.skeleton, &mesh.morph_target_names);
    if !errors.is_empty() {
        return Err(format!(
            "CharacterModel '{name}': source '{source}' does not conform to the schema:\n  {}",
            errors.join("\n  ")
        ));
    }
    let mut morphs = MorphSet {
        names: std::mem::take(&mut mesh.morph_target_names),
        deltas: std::mem::take(&mut mesh.morph_deltas),
    };
    synthesize(
        schema,
        &mesh.skeleton,
        &mesh.vertices,
        &mesh.indices,
        &mut morphs,
    )
    .map_err(|e| format!("CharacterModel '{name}': source '{source}': {e}"))?;
    mesh.morph_target_names = morphs.names;
    mesh.morph_deltas = morphs.deltas;
    Ok(mesh)
}

// The `character_model` arg a CharacterModel expands into on its SkinnedMesh:
// the model itself plus the resolved schema, so the payload cache key covers
// both.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CharacterModelArg {
    pub schema: CharacterSchema,
    pub model: CharacterModel,
}

impl CharacterModelArg {
    pub(crate) fn resolve(
        name: &str,
        model: CharacterModel,
        assets: &[WorldJsonlAsset],
    ) -> Result<Self, String> {
        let schema = builtin_schema::resolve(&model.schema, assets)
            .map_err(|e| format!("CharacterModel '{name}': {e}"))?;
        let errors = schema.consistency_errors();
        if !errors.is_empty() {
            return Err(format!(
                "CharacterModel '{name}': schema '{}' is inconsistent:\n  {}",
                model.schema,
                errors.join("\n  ")
            ));
        }
        Ok(Self { schema, model })
    }

    // The file the model reads.
    pub(crate) fn source_files(&self) -> Vec<String> {
        if self.model.source.is_empty() {
            Vec::new()
        } else {
            vec![self.model.source.clone()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::registry::build_only::KeyPolarity;
    use crate::authoring::registry::build_only::SchemaJoint;
    use crate::authoring::registry::build_only::SchemaKey;

    #[test]
    fn a_non_conforming_source_is_refused_with_the_reasons() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("body.glb");
        std::fs::write(&path, crate::import::glb::test_fixtures::skinned_glb()).expect("write");
        let source: String = path.to_str().expect("utf-8 path").into();
        let mut schema = builtin_schema::humanoid().clone();
        schema.joints.push(SchemaJoint {
            name: "tail".into(),
            parent: "hips".into(),
            optional: false,
        });
        schema.keys.push(SchemaKey {
            name: "wings".into(),
            polarity: KeyPolarity::Bipolar,
            ..Default::default()
        });
        let model = CharacterModel {
            source,
            ..Default::default()
        };
        let err = import_model("body", &schema, &model, None)
            .err()
            .expect("refused");
        assert!(err.contains("missing joint 'tail'"), "{err}");
        assert!(
            err.contains("missing shape key pair 'wings+' / 'wings-'"),
            "{err}"
        );
        let err = import_model("body", &schema, &CharacterModel::default(), None)
            .err()
            .expect("refused");
        assert!(err.contains("no source"), "{err}");
    }

    #[test]
    fn the_arg_resolves_its_schema_and_lists_the_source_file() {
        let model = CharacterModel {
            source: "hero.glb".into(),
            ..Default::default()
        };
        let arg = CharacterModelArg::resolve("body", model.clone(), &[]).unwrap();
        assert_eq!(arg.schema.joints.len(), 25);
        assert_eq!(arg.source_files(), ["hero.glb"]);
        let bad = CharacterModel {
            schema: "mine".into(),
            ..model
        };
        let err = CharacterModelArg::resolve("body", bad, &[]).unwrap_err();
        assert!(err.contains("'mine' is not a CharacterSchema"), "{err}");
        let assets = vec![WorldJsonlAsset {
            name: "mine".into(),
            asset_type: "CharacterSchema".into(),
            args: serde_json::json!({"regions": [{"name": "r", "joints": ["ghost"]}]}),
        }];
        let bad = CharacterModel {
            schema: "mine".into(),
            ..Default::default()
        };
        let err = CharacterModelArg::resolve("body", bad, &assets).unwrap_err();
        assert!(
            err.contains("inconsistent") && err.contains("ghost"),
            "{err}"
        );
        assert!(
            CharacterModelArg::resolve("b", CharacterModel::default(), &[])
                .unwrap()
                .source_files()
                .is_empty()
        );
    }
}
