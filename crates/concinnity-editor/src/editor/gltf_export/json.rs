// src/editor/gltf_export/json.rs
//
// glTF 2.0 JSON emission: nodes for the joint hierarchy, one skinned mesh
// with its morph targets (named via the `extras.targetNames` convention the
// engine's importer and Blender both read), a skin, and the bufferView /
// accessor tables packed by `buffer.rs`.

use serde_json::{Value, json};

use super::buffer::BinBuffer;
use crate::components::SkeletonJoint;
use concinnity_core::gfx::transform::{decompose, trs_matrix};

// Accessor indices for one mesh, as returned by the `BinBuffer` pushes.
// Optional attributes are omitted from the JSON when `None`.
pub(crate) struct MeshAccessors {
    pub position: usize,
    pub normal: Option<usize>,
    pub uv: Option<usize>,
    pub color: Option<usize>,
    pub joints: Option<usize>,
    pub weights: Option<usize>,
    pub indices: usize,
    pub inverse_bind: Option<usize>,
    // One (position, normal) accessor pair per morph target, in target order.
    pub targets: Vec<(usize, usize)>,
}

// The complete glTF document. Joint node index == joint index; the mesh node
// comes after the joints so the importer's topological reorder is an identity
// remap and a round trip preserves the engine's joint order.
pub(crate) fn document(
    name: &str,
    skeleton: &[SkeletonJoint],
    target_names: &[String],
    acc: &MeshAccessors,
    buffer: &BinBuffer,
) -> Value {
    let mut nodes: Vec<Value> = skeleton.iter().map(joint_node).collect();
    for (i, j) in skeleton.iter().enumerate() {
        if j.parent >= 0 {
            let children = nodes[j.parent as usize]
                .as_object_mut()
                .expect("joint node is an object")
                .entry("children")
                .or_insert_with(|| json!([]));
            children
                .as_array_mut()
                .expect("children is an array")
                .push(json!(i));
        }
    }

    let mesh_node = nodes.len();
    let mut mesh_node_json = json!({ "name": name, "mesh": 0 });
    if !skeleton.is_empty() {
        mesh_node_json["skin"] = json!(0);
    }
    nodes.push(mesh_node_json);

    let mut scene_nodes: Vec<usize> = skeleton
        .iter()
        .enumerate()
        .filter(|(_, j)| j.parent < 0)
        .map(|(i, _)| i)
        .collect();
    scene_nodes.push(mesh_node);

    let mut root = json!({
        "asset": { "version": "2.0", "generator": "concinnity-editor" },
        "scene": 0,
        "scenes": [{ "nodes": scene_nodes }],
        "nodes": nodes,
        "meshes": [mesh(name, target_names, acc)],
        "buffers": [{ "byteLength": buffer.bytes.len() }],
        "bufferViews": buffer
            .views
            .iter()
            .map(|v| json!({ "buffer": 0, "byteOffset": v.offset, "byteLength": v.len }))
            .collect::<Vec<_>>(),
        "accessors": buffer.accessors.iter().map(accessor).collect::<Vec<_>>(),
    });
    if !skeleton.is_empty() {
        let mut skin = json!({ "joints": (0..skeleton.len()).collect::<Vec<_>>() });
        if let Some(ibm) = acc.inverse_bind {
            skin["inverseBindMatrices"] = json!(ibm);
        }
        if let Some(first_root) = skeleton.iter().position(|j| j.parent < 0) {
            skin["skeleton"] = json!(first_root);
        }
        root["skins"] = json!([skin]);
    }
    root
}

fn joint_node(j: &SkeletonJoint) -> Value {
    // The engine stores joint rotation as YXZ Euler degrees; glTF nodes take a
    // unit quaternion. Build the rotation-only matrix and decompose it.
    let (_, rotation, _) = decompose(trs_matrix([0.0; 3], j.rotation_deg, [1.0; 3]));
    let mut node = json!({
        "translation": j.translation,
        "rotation": rotation,
        "scale": j.scale,
    });
    if !j.name.is_empty() {
        node["name"] = json!(j.name);
    }
    node
}

fn mesh(name: &str, target_names: &[String], acc: &MeshAccessors) -> Value {
    let mut attributes = json!({ "POSITION": acc.position });
    let mut set = |key: &str, a: Option<usize>| {
        if let Some(a) = a {
            attributes[key] = json!(a);
        }
    };
    set("NORMAL", acc.normal);
    set("TEXCOORD_0", acc.uv);
    set("COLOR_0", acc.color);
    set("JOINTS_0", acc.joints);
    set("WEIGHTS_0", acc.weights);

    let mut primitive = json!({ "attributes": attributes, "indices": acc.indices });
    if !acc.targets.is_empty() {
        primitive["targets"] = acc
            .targets
            .iter()
            .map(|(p, n)| json!({ "POSITION": p, "NORMAL": n }))
            .collect();
    }
    let mut mesh = json!({ "name": name, "primitives": [primitive] });
    if !target_names.is_empty() {
        mesh["extras"] = json!({ "targetNames": target_names });
    }
    mesh
}

fn accessor(a: &super::buffer::Accessor) -> Value {
    let mut out = json!({
        "bufferView": a.view,
        "componentType": a.component_type,
        "count": a.count,
        "type": a.element_type,
    });
    if let Some(min) = &a.min {
        out["min"] = json!(min);
    }
    if let Some(max) = &a.max {
        out["max"] = json!(max);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(name: &str, parent: i32) -> SkeletonJoint {
        SkeletonJoint {
            name: name.into(),
            parent,
            translation: [0.0, 1.0, 0.0],
            ..Default::default()
        }
    }

    fn accessors() -> (MeshAccessors, BinBuffer) {
        let mut buf = BinBuffer::default();
        let position = buf.push_vec3(&[[0.0, 0.0, 0.0]], true);
        let indices = buf.push_indices(&[0, 0, 0]);
        (
            MeshAccessors {
                position,
                normal: None,
                uv: None,
                color: None,
                joints: None,
                weights: None,
                indices,
                inverse_bind: None,
                targets: Vec::new(),
            },
            buf,
        )
    }

    #[test]
    fn joints_become_nodes_with_children_and_the_mesh_node_rides_behind() {
        let skeleton = [joint("root", -1), joint("mid", 0), joint("tip", 1)];
        let (acc, buf) = accessors();
        let doc = document("body", &skeleton, &[], &acc, &buf);
        let nodes = doc["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 4);
        assert_eq!(nodes[0]["name"], "root");
        assert_eq!(nodes[0]["children"], json!([1]));
        assert_eq!(nodes[1]["children"], json!([2]));
        assert_eq!(nodes[3]["mesh"], 0);
        assert_eq!(nodes[3]["skin"], 0);
        // Scene roots: the root joint and the mesh node only.
        assert_eq!(doc["scenes"][0]["nodes"], json!([0, 3]));
        assert_eq!(doc["skins"][0]["joints"], json!([0, 1, 2]));
        assert_eq!(doc["skins"][0]["skeleton"], 0);
    }

    #[test]
    fn a_rotated_joint_exports_the_equivalent_quaternion() {
        let mut j = joint("root", -1);
        j.rotation_deg = [0.0, 90.0, 0.0];
        let (acc, buf) = accessors();
        let doc = document("m", &[j], &[], &acc, &buf);
        let q: Vec<f64> = doc["nodes"][0]["rotation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        // 90 degrees of yaw about +Y: (0, sin 45, 0, cos 45).
        assert!(
            (q[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "{q:?}"
        );
        assert!(
            (q[3] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "{q:?}"
        );
        assert!(q[0].abs() < 1e-6 && q[2].abs() < 1e-6, "{q:?}");
    }

    #[test]
    fn morph_targets_emit_accessor_pairs_and_extras_target_names() {
        let (mut acc, buf) = accessors();
        acc.targets = vec![(2, 3), (4, 5)];
        let names = ["wide".to_string(), "lean+".to_string()];
        let doc = document("m", &[], &names, &acc, &buf);
        let prim = &doc["meshes"][0]["primitives"][0];
        assert_eq!(prim["targets"][0], json!({ "POSITION": 2, "NORMAL": 3 }));
        assert_eq!(prim["targets"][1], json!({ "POSITION": 4, "NORMAL": 5 }));
        assert_eq!(
            doc["meshes"][0]["extras"]["targetNames"],
            json!(["wide", "lean+"])
        );
        // No skeleton: no skin, and the mesh node carries none.
        assert!(doc.get("skins").is_none());
        assert!(doc["nodes"][0].get("skin").is_none());
    }

    #[test]
    fn optional_attributes_are_omitted_when_absent() {
        let (acc, buf) = accessors();
        let doc = document("m", &[], &[], &acc, &buf);
        let attrs = &doc["meshes"][0]["primitives"][0]["attributes"];
        assert_eq!(attrs["POSITION"], 0);
        for key in ["NORMAL", "TEXCOORD_0", "COLOR_0", "JOINTS_0", "WEIGHTS_0"] {
            assert!(attrs.get(key).is_none(), "{key} should be absent");
        }
        // Accessor table: POSITION carries bounds, indices none.
        let accs = doc["accessors"].as_array().unwrap();
        assert!(accs[0].get("min").is_some() && accs[0].get("max").is_some());
        assert!(accs[1].get("min").is_none());
        assert_eq!(doc["buffers"][0]["byteLength"], buf.bytes.len());
    }
}
