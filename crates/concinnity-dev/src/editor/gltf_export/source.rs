// src/editor/gltf_export/source.rs
//
// Feeding the writer from the working world: prepare + compile the entries in
// memory and pull the named SkinnedMesh's compiled payload, which is the
// composed mesh (glTF import, CharacterModel expansion, synthesized targets).
// A `bake: true` CharacterShape targeting the mesh is compiled with the flag
// disarmed so the morph targets survive into the payload; the export-time bake
// then folds the same shape into the vertices on request.

use super::ExportMesh;
use crate::components::CharacterShape;
use crate::gfx::mesh_payload::{SkinnedPayload, deserialise_skinned_with_lods};
use crate::world::WorldJsonlAsset;
use concinnity_core::ecs::ResourceKind;
use concinnity_core::geometry::payload_joints_to_defs;

// Compile `content` (a world.jsonl string) and export the named skinned mesh
// as GLB bytes. With `bake`, the current CharacterShape targeting the mesh is
// folded into the vertices and bind pose instead of exporting morph targets.
pub(crate) fn export_world_mesh(content: &str, mesh: &str, bake: bool) -> Result<Vec<u8>, String> {
    let loaded = concinnity_cook::prepare_world(content, crate::project::assets_dir().as_deref())
        .map_err(|errs| errs.join("; "))?;
    let mut assets = loaded.assets;
    let entry = assets
        .iter()
        .find(|a| a.name == mesh)
        .ok_or_else(|| format!("no asset named '{mesh}' in the world"))?;
    if entry.asset_type != "SkinnedMesh" {
        return Err(format!(
            "'{mesh}' is a {}, not a skinned mesh",
            entry.asset_type
        ));
    }
    let shape = shape_targeting(&assets, mesh);
    if bake && shape.is_none() {
        return Err(format!("no CharacterShape targets '{mesh}' to bake"));
    }
    for a in assets.iter_mut() {
        if a.asset_type == "CharacterShape"
            && a.args.get("target").and_then(|t| t.as_str()) == Some(mesh)
            && let Some(obj) = a.args.as_object_mut()
        {
            obj.insert("bake".into(), serde_json::Value::Bool(false));
        }
    }
    let result = concinnity_cook::build_compiled(
        assets,
        crate::project::assets_dir().as_deref(),
        None,
        crate::cook_platform(),
    )
    .map_err(|e| e.to_string())?;
    let bytes = result
        .resource_payload(ResourceKind::SkinnedMesh, mesh)
        .ok_or_else(|| format!("'{mesh}' compiled without a skinned payload"))?;
    let payload = deserialise_skinned_with_lods(bytes)?;
    let mut export = export_mesh_from_payload(mesh, payload);
    if bake && let Some(shape) = &shape {
        super::bake::bake_shape(&mut export, shape);
    }
    super::export_glb(&export)
}

// The first CharacterShape whose target is `mesh`, with only the fields the
// export bake reads.
fn shape_targeting(assets: &[WorldJsonlAsset], mesh: &str) -> Option<CharacterShape> {
    assets
        .iter()
        .find(|a| {
            a.asset_type == "CharacterShape"
                && a.args.get("target").and_then(|t| t.as_str()) == Some(mesh)
        })
        .map(|a| CharacterShape {
            sliders: serde_json::from_value(a.args.get("sliders").cloned().unwrap_or_default())
                .unwrap_or_default(),
            proportions: serde_json::from_value(
                a.args.get("proportions").cloned().unwrap_or_default(),
            )
            .unwrap_or_default(),
            ..Default::default()
        })
}

// The export form of a compiled skinned payload: LOD0 geometry, the bind
// skeleton, and the dense morph set. Uniform white vertex colour (the
// payload's fill-in default) is left out of the file.
fn export_mesh_from_payload(name: &str, p: SkinnedPayload) -> ExportMesh {
    let colors = if p.vertices.iter().all(|v| v.color == [1.0, 1.0, 1.0]) {
        Vec::new()
    } else {
        p.vertices.iter().map(|v| v.color).collect()
    };
    ExportMesh {
        name: name.to_string(),
        positions: p.vertices.iter().map(|v| v.pos).collect(),
        normals: p.vertices.iter().map(|v| v.normal).collect(),
        uvs: p.vertices.iter().map(|v| v.uv).collect(),
        colors,
        joints: p.vertices.iter().map(|v| v.joints).collect(),
        weights: p.vertices.iter().map(|v| v.weights).collect(),
        indices: p.indices,
        skeleton: payload_joints_to_defs(p.joints),
        morph_target_names: p.morphs.names.clone(),
        morph_deltas: p.morphs.to_dense(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A two-joint prism with a `wide` target and a live shape, as one
    // world.jsonl string. `baking` arms the shape's bake flag.
    fn prism_world(baking: bool) -> String {
        let vertices: Vec<serde_json::Value> = [
            ([0.0, 0.0, 0.0], 0),
            ([1.0, 0.0, 0.0], 0),
            ([0.0, 1.0, 0.0], 1),
        ]
        .iter()
        .map(|(pos, joint)| {
            serde_json::json!({"pos": pos, "joints": [joint, 0, 0, 0],
                "weights": [1.0, 0.0, 0.0, 0.0]})
        })
        .collect();
        let mesh = serde_json::json!({
            "name": "prism", "type": "SkinnedMesh",
            "args": {
                "vertices": vertices,
                "indices": [0, 1, 2],
                "skeleton": [
                    {"name": "root", "parent": -1},
                    {"name": "tip", "parent": 0, "translation": [0.0, 1.0, 0.0]}
                ],
                "morph_target_names": ["wide"],
                "morph_deltas": [
                    {"position": [1.0, 0.0, 0.0]},
                    {"position": [1.0, 0.0, 0.0]},
                    {"position": [1.0, 0.0, 0.0]}
                ],
                "scale": [1.0, 1.0, 1.0]
            }
        });
        let shape = serde_json::json!({
            "name": "shape", "type": "CharacterShape",
            "args": {"target": "prism", "bake": baking,
                "sliders": [{"name": "wide", "value": 0.5}]}
        });
        format!("{mesh}\n{shape}\n")
    }

    fn json_chunk(glb: &[u8]) -> serde_json::Value {
        let json_len = u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        serde_json::from_slice(&glb[20..20 + json_len]).expect("JSON chunk parses")
    }

    #[test]
    fn a_world_mesh_exports_with_its_targets_and_joints() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();
        let glb = export_world_mesh(&prism_world(false), "prism", false).expect("export");
        let doc = json_chunk(&glb);
        assert_eq!(
            doc["meshes"][0]["extras"]["targetNames"],
            serde_json::json!(["wide"])
        );
        let names: Vec<&str> = doc["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|n| n["name"].as_str())
            .collect();
        assert_eq!(names, ["root", "tip", "prism"]);
        // The compile normalised the geometry and baked normals in.
        let attrs = &doc["meshes"][0]["primitives"][0]["attributes"];
        assert!(attrs.get("NORMAL").is_some());
        assert!(attrs.get("JOINTS_0").is_some());
    }

    #[test]
    fn a_baking_shape_is_disarmed_so_targets_still_export() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();
        let glb = export_world_mesh(&prism_world(true), "prism", false).expect("export");
        let doc = json_chunk(&glb);
        assert_eq!(
            doc["meshes"][0]["extras"]["targetNames"],
            serde_json::json!(["wide"])
        );
    }

    #[test]
    fn a_baked_export_folds_the_shape_and_drops_the_targets() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();
        let glb = export_world_mesh(&prism_world(false), "prism", true).expect("export");
        let doc = json_chunk(&glb);
        assert!(doc["meshes"][0].get("extras").is_none());
        assert!(doc["meshes"][0]["primitives"][0].get("targets").is_none());
        // wide at 0.5 moved every vertex +0.5 in X: min/max shift with it.
        let pos = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
            .as_u64()
            .unwrap() as usize;
        let min_x = doc["accessors"][pos]["min"][0].as_f64().unwrap();
        assert!((min_x - 0.5).abs() < 1e-5, "{min_x}");
    }

    #[test]
    fn export_errors_name_the_missing_pieces() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();
        let err = export_world_mesh(&prism_world(false), "ghost", false).unwrap_err();
        assert!(err.contains("no asset named 'ghost'"), "{err}");
        let err = export_world_mesh(&prism_world(false), "shape", false).unwrap_err();
        assert!(err.contains("not a skinned mesh"), "{err}");
        // Baking needs a shape: point at a world without one.
        let bare = prism_world(false);
        let bare: String = bare.lines().take(1).collect::<Vec<_>>().join("\n");
        let err = export_world_mesh(&bare, "prism", true).unwrap_err();
        assert!(err.contains("no CharacterShape targets"), "{err}");
    }
}
