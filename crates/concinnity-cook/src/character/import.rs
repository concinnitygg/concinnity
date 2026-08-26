// Builds a CharacterModel's geometry: the source is imported, checked against
// the schema, and given the schema's synthesized targets. The result is an
// ordinary skinned import; lower levels of detail come from the skinned
// mesh's own `lod_levels` decimation, which keeps every vertex (so every
// target and skin weight) and only shortens the index list.

use std::path::Path;

use super::builtin_schema;
use super::synthesize::{MorphSet, synthesize};
use super::validate;
use crate::components::{CharacterModel, CharacterSchema};
use crate::glb::{ImportedSkinnedMesh, import_skinned_from_doc};
use crate::gltf_source::GltfDoc;
use concinnity_world::world::WorldJsonlAsset;

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
    let doc = GltfDoc::parse_file(&crate::glb::resolve_source(source, assets_dir))?;
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
    use crate::components::{KeyPolarity, SchemaJoint, SchemaKey};

    // The customize_character example's body, the only real one in the tree.
    // Absent from a crate checkout outside the workspace, in which case the
    // tests that need it report that and pass.
    fn example_body() -> Option<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/customize_character/base_humanoid.glb");
        if path.is_file() {
            path.to_str().map(str::to_string)
        } else {
            eprintln!("customize_character example glb not found; skipping");
            None
        }
    }

    #[test]
    fn the_example_humanoid_imports_with_synthesized_targets() {
        let Some(source) = example_body() else {
            return;
        };
        let model = CharacterModel {
            schema: builtin_schema::HUMANOID_SCHEMA.into(),
            source,
            ..Default::default()
        };
        let schema = builtin_schema::humanoid();
        let g = import_model("body", schema, &model, None).expect("import");
        assert_eq!(g.skeleton.len(), 25);
        assert!(g.vertices.len() > 15_000 && g.vertices.len() < 25_000);
        let expected_targets = 22
            + schema
                .synthesized
                .iter()
                .map(|t| match t.polarity {
                    KeyPolarity::Unipolar => 1,
                    KeyPolarity::Bipolar => 2,
                })
                .sum::<usize>();
        assert_eq!(g.morph_target_names.len(), expected_targets);
        assert_eq!(g.morph_deltas.len(), expected_targets * g.vertices.len());
        let n = g.vertices.len();
        let t = g
            .morph_target_names
            .iter()
            .position(|n| n == "thigh_girth+")
            .unwrap();
        let moved = g.morph_deltas[t * n..(t + 1) * n]
            .iter()
            .filter(|d| d.position != [0.0; 3])
            .count();
        assert!(moved > 500 && moved < n, "{moved} of {n}");
    }

    // The synthesized targets each move a few percent of the body, but the
    // authored keys are body-wide (`weight` touches 95% of vertices, `muscle`
    // 55%, `leg_weight` 62%), which puts the real fill at about 15% of the
    // dense (target, vertex) grid: 5.3x smaller as measured, 4x asserted.
    #[test]
    fn the_example_humanoid_payload_stores_morphs_several_times_smaller_than_dense() {
        let Some(source) = example_body() else {
            return;
        };
        let model = CharacterModel {
            schema: builtin_schema::HUMANOID_SCHEMA.into(),
            source,
            lod_levels: 3,
            ..Default::default()
        };
        let g = import_model("body", builtin_schema::humanoid(), &model, None).expect("import");
        let payload = crate::geometry::compile_skinned_mesh_payload_with_lods(
            &g.vertices,
            &g.indices,
            &g.skeleton,
            &g.morph_target_names,
            &g.morph_deltas,
            &crate::geometry::SkinnedLods {
                levels: 3,
                distances: &[],
            },
        )
        .expect("payload");
        let p = crate::gfx::mesh_payload::deserialise_skinned_with_lods(&payload).expect("read");
        assert_eq!(p.lods.len(), 2);
        let dense_bytes = g.morph_deltas.len() * 24;
        let sparse_bytes = p.morphs.offsets.len() * 4 + p.morphs.entries.len() * 28;
        assert!(
            sparse_bytes * 4 <= dense_bytes,
            "sparse morph block {sparse_bytes} B vs dense {dense_bytes} B"
        );
        assert!(
            payload.len() * 3 <= dense_bytes,
            "whole payload {} B vs dense morph block {dense_bytes} B",
            payload.len()
        );
        let back: Vec<crate::components::MorphDelta> = p
            .morphs
            .to_dense()
            .iter()
            .map(|d| crate::components::MorphDelta {
                position: d.position,
                normal: d.normal,
            })
            .collect();
        let lost = g
            .morph_deltas
            .iter()
            .zip(&back)
            .filter(|(a, b)| a != b)
            .filter(|(a, _)| {
                a.position
                    .iter()
                    .chain(a.normal.iter())
                    .any(|x| x.abs() > crate::gfx::mesh_payload::MORPH_DELTA_EPSILON)
            })
            .count();
        assert_eq!(lost, 0, "every significant delta survives the sparse form");
    }

    #[test]
    fn a_non_conforming_source_is_refused_with_the_reasons() {
        let Some(source) = example_body() else {
            return;
        };
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
