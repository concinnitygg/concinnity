// src/editor/gltf_export/mod.rs
//
// glTF 2.0 (.glb) export of a skinned mesh: geometry, skeleton, and the full
// morph-target set, written so Blender and the engine's own glTF importer read
// back the same joint order, joint names, and shape-key names. The writer is
// pure data-to-bytes; `source.rs` feeds it from a compiled world and the
// console's /export command writes the result beside the project.

mod bake;
mod buffer;
mod container;
mod json;
mod source;

pub(crate) use source::export_world_mesh;

use crate::components::SkeletonJoint;
use crate::gfx::mesh_payload::MorphDelta;
use concinnity_core::components::build_skeleton_from_joint_defs;
use concinnity_core::gfx::transform::{Mat4, mat4_affine_inverse};

// Everything one exported mesh carries. Attribute lists are parallel to
// `positions`; the empty ones (`normals` / `uvs` / `colors`) are omitted from
// the file. `morph_deltas` is dense and target-major, like the asset form.
pub(crate) struct ExportMesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub indices: Vec<u16>,
    pub skeleton: Vec<SkeletonJoint>,
    pub morph_target_names: Vec<String>,
    pub morph_deltas: Vec<MorphDelta>,
}

impl ExportMesh {
    fn validate(&self) -> Result<(), String> {
        let n = self.positions.len();
        if n == 0 {
            return Err("mesh has no vertices".to_string());
        }
        for (label, len) in [
            ("normals", self.normals.len()),
            ("uvs", self.uvs.len()),
            ("colors", self.colors.len()),
        ] {
            if len != 0 && len != n {
                return Err(format!("{label} has {len} entries for {n} vertices"));
            }
        }
        if !self.skeleton.is_empty() && (self.joints.len() != n || self.weights.len() != n) {
            return Err(format!(
                "joint bindings have {} / {} entries for {} vertices",
                self.joints.len(),
                self.weights.len(),
                n
            ));
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(format!(
                "{} indices do not form triangles",
                self.indices.len()
            ));
        }
        if let Some(bad) = self.indices.iter().find(|&&i| i as usize >= n) {
            return Err(format!("index {bad} out of range ({n} vertices)"));
        }
        for (i, j) in self.skeleton.iter().enumerate() {
            if j.parent >= i as i32 {
                return Err(format!(
                    "joint {i} ('{}') has parent {} at or after it",
                    j.name, j.parent
                ));
            }
        }
        let joint_count = self.skeleton.len();
        if !self.skeleton.is_empty()
            && let Some(bad) = self
                .joints
                .iter()
                .flatten()
                .find(|&&j| j as usize >= joint_count)
        {
            return Err(format!(
                "vertex joint {bad} out of range ({joint_count} joints)"
            ));
        }
        let expected = self.morph_target_names.len() * n;
        if self.morph_deltas.len() != expected {
            return Err(format!(
                "morph_deltas has {} entries; {} target(s) x {} vertices requires {}",
                self.morph_deltas.len(),
                self.morph_target_names.len(),
                n,
                expected
            ));
        }
        Ok(())
    }
}

// One inverse bind matrix per joint: the inverse of the joint's world-space
// bind transform, which glTF skins require alongside the node hierarchy.
fn inverse_bind_matrices(skeleton: &[SkeletonJoint]) -> Vec<Mat4> {
    let s = build_skeleton_from_joint_defs(skeleton);
    let mut world = Vec::new();
    s.world_matrices_into(s.bind_locals(), &mut world);
    world.iter().map(|m| mat4_affine_inverse(*m)).collect()
}

// Serialise `mesh` into a GLB byte stream.
pub(crate) fn export_glb(mesh: &ExportMesh) -> Result<Vec<u8>, String> {
    mesh.validate()?;
    let n = mesh.positions.len();
    let skinned = !mesh.skeleton.is_empty();
    let mut buf = buffer::BinBuffer::default();
    let acc = json::MeshAccessors {
        position: buf.push_vec3(&mesh.positions, true),
        normal: (!mesh.normals.is_empty()).then(|| buf.push_vec3(&mesh.normals, false)),
        uv: (!mesh.uvs.is_empty()).then(|| buf.push_vec2(&mesh.uvs)),
        color: (!mesh.colors.is_empty()).then(|| buf.push_vec3(&mesh.colors, false)),
        joints: skinned.then(|| buf.push_u16_vec4(&mesh.joints)),
        weights: skinned.then(|| buf.push_vec4(&mesh.weights)),
        indices: buf.push_indices(&mesh.indices),
        inverse_bind: skinned.then(|| buf.push_mat4(&inverse_bind_matrices(&mesh.skeleton))),
        targets: (0..mesh.morph_target_names.len())
            .map(|t| {
                let block = &mesh.morph_deltas[t * n..(t + 1) * n];
                let positions: Vec<[f32; 3]> = block.iter().map(|d| d.position).collect();
                let normals: Vec<[f32; 3]> = block.iter().map(|d| d.normal).collect();
                (
                    buf.push_vec3(&positions, true),
                    buf.push_vec3(&normals, false),
                )
            })
            .collect(),
    };
    let doc = json::document(
        &mesh.name,
        &mesh.skeleton,
        &mesh.morph_target_names,
        &acc,
        &buf,
    );
    let json_bytes = serde_json::to_vec(&doc).map_err(|e| format!("serialise glTF json: {e}"))?;
    Ok(container::wrap_glb(json_bytes, buf.bytes))
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;

    // A two-triangle quad on a two-joint chain with two morph targets:
    // `wide` (unipolar, +X on every vertex) and `lift+` (upper vertices +Y,
    // with a normal tilt).
    pub(crate) fn quad_mesh() -> ExportMesh {
        let positions = vec![
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 1.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let deltas = |t: usize, v: usize| -> MorphDelta {
            match (t, v) {
                (0, _) => MorphDelta {
                    position: [0.1, 0.0, 0.0],
                    normal: [0.0; 3],
                },
                (1, 2) | (1, 3) => MorphDelta {
                    position: [0.0, 0.2, 0.0],
                    normal: [0.0, 0.1, 0.0],
                },
                _ => MorphDelta::default(),
            }
        };
        ExportMesh {
            name: "quad".to_string(),
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            colors: Vec::new(),
            joints: vec![[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 4],
            indices: vec![0, 1, 2, 2, 1, 3],
            skeleton: vec![
                SkeletonJoint {
                    name: "root".to_string(),
                    parent: -1,
                    ..Default::default()
                },
                SkeletonJoint {
                    name: "top".to_string(),
                    parent: 0,
                    translation: [0.0, 1.0, 0.0],
                    ..Default::default()
                },
            ],
            morph_target_names: vec!["wide".to_string(), "lift+".to_string()],
            morph_deltas: (0..2)
                .flat_map(|t| (0..4).map(move |v| deltas(t, v)))
                .collect(),
            positions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::quad_mesh;
    use super::*;

    // The parsed JSON chunk of an exported GLB.
    fn json_chunk(glb: &[u8]) -> serde_json::Value {
        let json_len = u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        serde_json::from_slice(&glb[20..20 + json_len]).expect("JSON chunk parses")
    }

    #[test]
    fn a_quad_exports_every_expected_accessor() {
        let glb = export_glb(&quad_mesh()).expect("export");
        let doc = json_chunk(&glb);
        // POSITION, NORMAL, TEXCOORD_0, JOINTS_0, WEIGHTS_0, indices, IBM,
        // plus 2 accessors per target.
        assert_eq!(doc["accessors"].as_array().unwrap().len(), 7 + 4);
        assert_eq!(doc["bufferViews"].as_array().unwrap().len(), 7 + 4);
        let attrs = &doc["meshes"][0]["primitives"][0]["attributes"];
        assert!(attrs.get("COLOR_0").is_none());
        assert_eq!(
            doc["meshes"][0]["extras"]["targetNames"],
            serde_json::json!(["wide", "lift+"])
        );
        // POSITION bounds cover the quad.
        let pos = &doc["accessors"][attrs["POSITION"].as_u64().unwrap() as usize];
        assert_eq!(pos["min"], serde_json::json!([-0.5, 0.0, 0.0]));
        assert_eq!(pos["max"], serde_json::json!([0.5, 1.0, 0.0]));
        // Every morph POSITION accessor carries bounds too.
        for (p, _) in [(7usize, 8usize), (9, 10)] {
            assert!(
                doc["accessors"][p].get("min").is_some(),
                "target accessor {p}"
            );
        }
    }

    #[test]
    fn inverse_bind_matrices_undo_the_bind_pose() {
        let mesh = quad_mesh();
        let ibm = inverse_bind_matrices(&mesh.skeleton);
        assert_eq!(ibm.len(), 2);
        // Root binds at the origin: identity.
        assert_eq!(ibm[0][3][1], 0.0);
        // The top joint binds 1 up in Y; its inverse carries -1.
        assert!((ibm[1][3][1] + 1.0).abs() < 1e-6);
        assert!((ibm[1][0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn validation_rejects_inconsistent_data() {
        let mut short_deltas = quad_mesh();
        short_deltas.morph_deltas.pop();
        assert!(
            export_glb(&short_deltas)
                .unwrap_err()
                .contains("morph_deltas")
        );

        let mut bad_index = quad_mesh();
        bad_index.indices[0] = 9;
        assert!(export_glb(&bad_index).unwrap_err().contains("out of range"));

        let mut child_first = quad_mesh();
        child_first.skeleton[0].parent = 1;
        assert!(export_glb(&child_first).unwrap_err().contains("parent"));

        let mut bad_binding = quad_mesh();
        bad_binding.joints[0] = [7, 0, 0, 0];
        assert!(
            export_glb(&bad_binding)
                .unwrap_err()
                .contains("vertex joint")
        );

        let mut ragged = quad_mesh();
        ragged.normals.pop();
        assert!(export_glb(&ragged).unwrap_err().contains("normals"));
    }

    #[test]
    fn an_export_round_trips_through_the_cook_importer() {
        let mesh = quad_mesh();
        let glb = export_glb(&mesh).expect("export");
        let doc =
            concinnity_cook::import::gltf_source::GltfDoc::from_slice(&glb, None, "roundtrip")
                .expect("exported GLB parses");
        let back = concinnity_cook::import::mesh_reimport::decode_skinned_inline_from_parsed_glb(
            &doc,
            "roundtrip",
            0,
        )
        .expect("cook importer accepts the export");
        assert_eq!(back.vertices.len(), mesh.positions.len());
        assert_eq!(back.indices, mesh.indices);
        // Joint order and names survive: the export's node order is already
        // parents-before-children, so the importer's reorder is an identity.
        let names: Vec<&str> = back.skeleton.iter().map(|j| j.name.as_str()).collect();
        assert_eq!(names, ["root", "top"]);
        assert_eq!(back.skeleton[1].parent, 0);
        assert!(
            back.skeleton[1]
                .translation
                .iter()
                .zip(&mesh.skeleton[1].translation)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        // Vertex bindings and weights are unchanged.
        for (b, (j, w)) in back
            .vertices
            .iter()
            .zip(mesh.joints.iter().zip(&mesh.weights))
        {
            assert_eq!(
                b.joints,
                [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32]
            );
            assert_eq!(b.weights, *w);
        }
        // Target names and per-target deltas within float tolerance.
        assert_eq!(back.morph_target_names, mesh.morph_target_names);
        assert_eq!(back.morph_deltas.len(), mesh.morph_deltas.len());
        for (b, d) in back.morph_deltas.iter().zip(&mesh.morph_deltas) {
            for c in 0..3 {
                assert!((b.position[c] - d.position[c]).abs() < 1e-6);
                assert!((b.normal[c] - d.normal[c]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn the_container_is_aligned_and_sized() {
        let glb = export_glb(&quad_mesh()).expect("export");
        assert_eq!(&glb[0..4], b"glTF");
        let total = u32::from_le_bytes([glb[8], glb[9], glb[10], glb[11]]) as usize;
        assert_eq!(total, glb.len());
        assert!(glb.len().is_multiple_of(4));
    }
}
