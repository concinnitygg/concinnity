// src/fbx/skin.rs
//
// Skinned-mesh import from binary FBX: reads Deformer(Skin) / SubDeformer
// (Cluster) objects, builds a parents-before-children skeleton, and assembles
// skinned vertices with top-4 normalized weights per control point.
//
// The runtime recomputes inverse-bind matrices from the SkeletonJoint chain, so a
// weighted joint's locals must multiply out to exactly the bind world its
// weights were bound against -- the cluster's `TransformLink`. Joints without
// a cluster (unweighted parents in the chain) extend the chain with their
// scene-pose transform instead.

use std::collections::{HashMap, HashSet};

use fbxcel::tree::v7400::NodeHandle;

use super::{
    arr_f64, arr_i32, attr_i64, attr_str, local_matrices, node_scene_local, object_id, object_name,
    transform_point,
};
use crate::components::{SkeletonJoint, SkinnedVertexData, VertexData};
use crate::gfx::skinning::{
    IDENTITY, Mat4, decompose, euler_yxz_from_quat, mat4_affine_inverse, mat4_mul,
};
use crate::glb::ImportedSkinnedMesh;

// The skinning-relevant slice of an FBX document: the first skinned geometry,
// its skeleton, and its per-control-point weights.
pub(super) struct FbxSkin {
    pub(crate) geometry_id: i64,
    // Mesh world at bind (cluster `Transform` x geometric offset): baked into
    // vertex positions so they live in the same frame as the joint worlds.
    pub(crate) mesh_bind: Mat4,
    // Meters per file unit; joint translations are already normalized by it,
    // and evaluated animation translations must be too.
    pub(crate) unit_scale: f32,
    pub joints: Vec<SkeletonJoint>,
    // Model object id -> joint index, for resolving animation curve targets.
    pub(crate) model_to_joint: HashMap<i64, usize>,
    // Control-point index -> accumulated (joint, weight) pairs.
    pub weights: HashMap<i32, Vec<(usize, f32)>>,
}

// Locate the `skin_index`-th skinned geometry (in Geometry declaration order)
// and extract its skeleton + weights.
pub(super) fn parse_skin(
    root: NodeHandle<'_>,
    path: &str,
    skin_index: u32,
) -> Result<FbxSkin, String> {
    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| format!("'{path}': FBX has no Objects section"))?;

    let mut model_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut geom_ids: HashSet<i64> = HashSet::new();
    let mut geom_order: Vec<i64> = Vec::new();
    let mut skin_ids: HashSet<i64> = HashSet::new();
    let mut cluster_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    for c in objects.children() {
        let Some(id) = object_id(&c) else { continue };
        match c.name() {
            "Model" => {
                model_by_id.insert(id, c);
            }
            "Geometry" => {
                geom_ids.insert(id);
                geom_order.push(id);
            }
            "Deformer" => match c.attributes().get(2).and_then(attr_str) {
                Some("Skin") => {
                    skin_ids.insert(id);
                }
                Some("Cluster") => {
                    cluster_by_id.insert(id, c);
                }
                _ => {}
            },
            _ => {}
        }
    }
    if skin_ids.is_empty() {
        return Err(format!(
            "'{path}': FBX has no skin deformer (no skinned mesh)"
        ));
    }

    // Connections. A bone Model is an OO child of both its parent Model (the
    // hierarchy) and its Cluster (the deformer link), so the hierarchy map
    // only accepts Model (or scene-root) parents.
    let mut model_parent: HashMap<i64, i64> = HashMap::new();
    let mut skin_of_geometry: HashMap<i64, i64> = HashMap::new();
    let mut geometry_model: HashMap<i64, i64> = HashMap::new();
    let mut clusters_of_skin: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut bone_of_cluster: HashMap<i64, i64> = HashMap::new();
    if let Some(conns) = root.first_child_by_name("Connections") {
        for c in conns.children_by_name("C") {
            let a = c.attributes();
            if a.first().and_then(attr_str) != Some("OO") {
                continue;
            }
            let (Some(child), Some(parent)) =
                (a.get(1).and_then(attr_i64), a.get(2).and_then(attr_i64))
            else {
                continue;
            };
            if model_by_id.contains_key(&child) {
                if cluster_by_id.contains_key(&parent) {
                    bone_of_cluster.insert(parent, child);
                } else if parent == 0 || model_by_id.contains_key(&parent) {
                    model_parent.insert(child, parent);
                }
            } else if skin_ids.contains(&child) && geom_ids.contains(&parent) {
                skin_of_geometry.insert(parent, child);
            } else if cluster_by_id.contains_key(&child) && skin_ids.contains(&parent) {
                clusters_of_skin.entry(parent).or_default().push(child);
            } else if geom_ids.contains(&child) && model_by_id.contains_key(&parent) {
                geometry_model.insert(child, parent);
            }
        }
    }

    let skinned: Vec<(i64, i64)> = geom_order
        .iter()
        .filter_map(|g| skin_of_geometry.get(g).map(|s| (*g, *s)))
        .collect();
    if skinned.is_empty() {
        return Err(format!("'{path}': no geometry is bound to a skin deformer"));
    }
    let count = skinned.len();
    let (geometry_id, skin_id) = *skinned.get(skin_index as usize).ok_or_else(|| {
        format!(
            "'{path}': skin_index {skin_index} out of range (file has {count} skinned mesh{})",
            if count == 1 { "" } else { "es" }
        )
    })?;
    let cluster_ids = clusters_of_skin.remove(&skin_id).unwrap_or_default();
    if cluster_ids.is_empty() {
        return Err(format!("'{path}': skin deformer has no clusters"));
    }

    // Per-cluster data, in connection order (stable across rebuilds).
    struct Cluster {
        bone: i64,
        indexes: Vec<i32>,
        weights: Vec<f64>,
        transform_link: Option<Mat4>,
        transform: Option<Mat4>,
    }
    let mut clusters: Vec<Cluster> = Vec::new();
    for cid in &cluster_ids {
        let node = cluster_by_id[cid];
        let Some(&bone) = bone_of_cluster.get(cid) else {
            continue;
        };
        clusters.push(Cluster {
            bone,
            indexes: node
                .first_child_by_name("Indexes")
                .as_ref()
                .and_then(arr_i32)
                .map(|a| a.to_vec())
                .unwrap_or_default(),
            weights: node
                .first_child_by_name("Weights")
                .as_ref()
                .and_then(arr_f64)
                .map(|a| a.to_vec())
                .unwrap_or_default(),
            transform_link: node
                .first_child_by_name("TransformLink")
                .as_ref()
                .and_then(arr_f64)
                .and_then(mat4_from_flat),
            transform: node
                .first_child_by_name("Transform")
                .as_ref()
                .and_then(arr_f64)
                .and_then(mat4_from_flat),
        });
    }
    if clusters.is_empty() {
        return Err(format!("'{path}': skin clusters have no bone links"));
    }

    // Skeleton node set: every cluster bone plus its Model ancestors, in
    // first-seen order, then sorted parents-before-children.
    let mut seen: HashSet<i64> = HashSet::new();
    let mut nodes: Vec<i64> = Vec::new();
    for c in &clusters {
        let mut id = c.bone;
        let mut chain: Vec<i64> = Vec::new();
        loop {
            if seen.contains(&id) {
                break;
            }
            seen.insert(id);
            chain.push(id);
            match model_parent.get(&id) {
                Some(&p) if p != 0 && model_by_id.contains_key(&p) => id = p,
                _ => break,
            }
        }
        // Ancestors were discovered leaf-first; keep root-first insertion so
        // the topological pass below tends to emit in one sweep.
        for id in chain.into_iter().rev() {
            nodes.push(id);
        }
    }

    // Parents-before-children order over the node set.
    let mut order: Vec<i64> = Vec::with_capacity(nodes.len());
    let mut emitted: HashSet<i64> = HashSet::new();
    loop {
        let before = order.len();
        for &id in &nodes {
            if emitted.contains(&id) {
                continue;
            }
            let ready = match model_parent.get(&id) {
                Some(p) => !seen.contains(p) || emitted.contains(p),
                None => true,
            };
            if ready {
                emitted.insert(id);
                order.push(id);
            }
        }
        if order.len() == before {
            break;
        }
    }
    for &id in &nodes {
        if !emitted.contains(&id) {
            order.push(id);
        }
    }

    // Bind worlds: a cluster bone uses its TransformLink verbatim; a joint
    // without one (an unweighted parent) extends the chain with its scene
    // pose. The whole rig is then normalized to meters by left-multiplying a
    // uniform scale, which only alters ROOT joint locals: every child local
    // (and therefore every authored animation curve) stays exactly as the
    // file expressed it, while weighted joints reproduce their bind worlds
    // regardless of what the scene pose says.
    let unit_scale = super::unit_scale_to_meters(&root);
    let tl_by_bone: HashMap<i64, Mat4> = clusters
        .iter()
        .filter_map(|c| c.transform_link.map(|m| (c.bone, m)))
        .collect();
    let mut model_to_joint: HashMap<i64, usize> = HashMap::new();
    let mut worlds: Vec<Mat4> = Vec::new();
    let mut parents: Vec<Option<usize>> = Vec::new();
    for &id in &order {
        let parent_joint = model_parent
            .get(&id)
            .and_then(|p| model_to_joint.get(p))
            .copied();
        let parent_world = parent_joint.map_or(IDENTITY, |j| worlds[j]);
        let world = match tl_by_bone.get(&id) {
            Some(&m) => m,
            None => mat4_mul(
                parent_world,
                model_by_id
                    .get(&id)
                    .map_or(IDENTITY, |m| node_scene_local(m)),
            ),
        };
        model_to_joint.insert(id, worlds.len());
        worlds.push(world);
        parents.push(parent_joint);
    }
    let mut joints: Vec<SkeletonJoint> = Vec::new();
    for (i, &id) in order.iter().enumerate() {
        let local = match parents[i] {
            Some(j) => mat4_mul(mat4_affine_inverse(worlds[j]), worlds[i]),
            None => uniform_scale_pre(unit_scale, worlds[i]),
        };
        let (translation, rotation, scale) = decompose(local);
        joints.push(SkeletonJoint {
            name: model_by_id.get(&id).map(object_name).unwrap_or_default(),
            parent: parents[i].map_or(-1, |j| j as i32),
            translation,
            rotation_deg: euler_yxz_from_quat(rotation),
            scale,
        });
    }

    // Control-point weights, joint-indexed.
    let mut weights: HashMap<i32, Vec<(usize, f32)>> = HashMap::new();
    for c in &clusters {
        let joint = model_to_joint[&c.bone];
        for (i, &cp) in c.indexes.iter().enumerate() {
            let w = c.weights.get(i).copied().unwrap_or(0.0) as f32;
            if w > 0.0 {
                weights.entry(cp).or_default().push((joint, w));
            }
        }
    }

    // Mesh bind frame: the file stores each cluster's `Transform` relative to
    // its bone's `TransformLink`, so the mesh world at bind is TL x Transform
    // (identical across clusters); the mesh model's geometric offset applies
    // after.
    let geometric = geometry_model
        .get(&geometry_id)
        .and_then(|m| model_by_id.get(m))
        .map_or(IDENTITY, |m| local_matrices(m).1);
    let mesh_world = clusters
        .iter()
        .find_map(|c| match (c.transform_link, c.transform) {
            (Some(tl), Some(t)) => Some(mat4_mul(tl, t)),
            _ => None,
        })
        .unwrap_or(IDENTITY);
    let mesh_bind = mat4_mul(mesh_world, geometric);

    Ok(FbxSkin {
        geometry_id,
        mesh_bind,
        unit_scale,
        joints,
        model_to_joint,
        weights,
    })
}

// Import the `skin_index`-th skinned mesh of a binary FBX into the inline
// `SkinnedMesh` fields, mirroring the glTF importer's output shape.
pub(crate) fn import_skinned_fbx(
    path: &str,
    skin_index: u32,
) -> Result<ImportedSkinnedMesh, String> {
    let tree = super::load_tree(path)?;
    let root = tree.root();
    let skin = parse_skin(root, path, skin_index)?;

    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| format!("'{path}': FBX has no Objects section"))?;
    let geom = objects
        .children()
        .find(|c| c.name() == "Geometry" && object_id(c) == Some(skin.geometry_id))
        .ok_or_else(|| format!("'{path}': skinned geometry object is missing"))?;

    let (verts, indices32, control_points) = extract_skinned_geometry(&geom)
        .ok_or_else(|| format!("'{path}': skinned geometry has no polygon data"))?;

    let mut vertices: Vec<SkinnedVertexData> = Vec::with_capacity(verts.len());
    for (v, cp) in verts.iter().zip(&control_points) {
        let (joints, weights) = top4_weights(skin.weights.get(cp));
        let world = transform_point(skin.mesh_bind, v.pos);
        vertices.push(SkinnedVertexData {
            pos: [
                world[0] * skin.unit_scale,
                world[1] * skin.unit_scale,
                world[2] * skin.unit_scale,
            ],
            color: [1.0, 1.0, 1.0],
            uv: v.uv,
            joints,
            weights,
        });
    }

    let mut indices: Vec<u16> = Vec::with_capacity(indices32.len());
    for i in indices32 {
        if i > u16::MAX as u32 {
            return Err(format!(
                "'{path}': skinned mesh exceeds the {}-vertex u16 index limit",
                u16::MAX
            ));
        }
        indices.push(i as u16);
    }

    Ok(ImportedSkinnedMesh {
        vertices,
        indices,
        skeleton: skin.joints,
        // FBX blend shapes are not imported; glTF is the morph path.
        morph_target_names: Vec::new(),
        morph_deltas: Vec::new(),
    })
}

// Left-multiply a uniform scale: `S(f) * m`. Scales the rotation/scale block
// and the translation together, i.e. re-expresses the matrix in units f times
// larger.
pub(super) fn uniform_scale_pre(f: f32, m: Mat4) -> Mat4 {
    let mut out = m;
    for col in &mut out {
        for c in col.iter_mut().take(3) {
            *c *= f;
        }
    }
    out
}

// Reinterpret 16 FBX doubles as the engine's Mat4. FBX stores row-vector
// matrices with translation at elements 12..14, which is byte-compatible with
// the engine's column-vector layout.
pub(super) fn mat4_from_flat(a: &[f64]) -> Option<Mat4> {
    if a.len() != 16 {
        return None;
    }
    let mut m = IDENTITY;
    for (i, v) in a.iter().enumerate() {
        m[i / 4][i % 4] = *v as f32;
    }
    Some(m)
}

// Top-4 weights for one control point, normalized; an unweighted point binds
// fully to joint 0 (matching the glTF importer's defaults).
fn top4_weights(list: Option<&Vec<(usize, f32)>>) -> ([u32; 4], [f32; 4]) {
    let Some(list) = list else {
        return ([0; 4], [1.0, 0.0, 0.0, 0.0]);
    };
    let mut sorted = list.clone();
    sorted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    sorted.truncate(4);
    let sum: f32 = sorted.iter().map(|(_, w)| w).sum();
    if sum <= 0.0 {
        return ([0; 4], [1.0, 0.0, 0.0, 0.0]);
    }
    let mut joints = [0u32; 4];
    let mut weights = [0.0f32; 4];
    for (i, (j, w)) in sorted.iter().enumerate() {
        joints[i] = *j as u32;
        weights[i] = w / sum;
    }
    (joints, weights)
}

// Triangulate every polygon of a geometry into one vertex/index buffer,
// deduplicating by (control point, UV index) and reporting each emitted
// vertex's control point so cluster weights (stored per control point) can be
// scattered onto the final vertices. The skinned path keeps all material
// groups together: a SkinnedMesh is a single primitive.
fn extract_skinned_geometry(geom: &NodeHandle) -> Option<(Vec<VertexData>, Vec<u32>, Vec<i32>)> {
    let positions = geom
        .first_child_by_name("Vertices")
        .as_ref()
        .and_then(arr_f64)?;
    let pvi = geom
        .first_child_by_name("PolygonVertexIndex")
        .as_ref()
        .and_then(arr_i32)?;

    let uv_layer = geom
        .children_by_name("LayerElementUV")
        .find(|l| super::child_str(l, "Name") == Some("TextureUV"))
        .or_else(|| geom.children_by_name("LayerElementUV").next());
    let (uv, uv_index, uv_indexed) = match uv_layer {
        Some(l) => (
            l.first_child_by_name("UV").as_ref().and_then(arr_f64),
            l.first_child_by_name("UVIndex").as_ref().and_then(arr_i32),
            super::child_str(&l, "ReferenceInformationType") == Some("IndexToDirect"),
        ),
        None => (None, None, false),
    };

    let mut dedup: HashMap<(i32, i32), u32> = HashMap::new();
    let mut vertices: Vec<VertexData> = Vec::new();
    let mut control_points: Vec<i32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut corners: Vec<u32> = Vec::new();

    for (pv, &raw) in pvi.iter().enumerate() {
        let (cp, end) = super::decode_pvi(raw);
        let uv_key = if uv_indexed {
            uv_index.and_then(|ui| ui.get(pv)).copied().unwrap_or(0)
        } else {
            pv as i32
        };
        let out = match dedup.get(&(cp, uv_key)) {
            Some(&i) => i,
            None => {
                let c = (cp as usize) * 3;
                let pos = [
                    positions.get(c).copied().unwrap_or(0.0) as f32,
                    positions.get(c + 1).copied().unwrap_or(0.0) as f32,
                    positions.get(c + 2).copied().unwrap_or(0.0) as f32,
                ];
                let i = vertices.len() as u32;
                vertices.push(VertexData {
                    pos,
                    color: [1.0, 1.0, 1.0],
                    uv: super::lookup_uv(uv, uv_indexed, uv_index, pv),
                });
                control_points.push(cp);
                dedup.insert((cp, uv_key), i);
                i
            }
        };
        corners.push(out);
        if !end {
            continue;
        }
        for k in 1..corners.len().saturating_sub(1) {
            indices.push(corners[0]);
            indices.push(corners[k]);
            indices.push(corners[k + 1]);
        }
        corners.clear();
    }

    if vertices.is_empty() {
        return None;
    }
    Some((vertices, indices, control_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fbx::fixtures as fx;
    use crate::fbx::fixtures::assert_vec3_eq;

    #[test]
    fn mat4_from_flat_places_translation_in_the_fourth_column() {
        let mut flat = [0.0f64; 16];
        flat[0] = 1.0;
        flat[5] = 1.0;
        flat[10] = 1.0;
        flat[15] = 1.0;
        flat[12] = 3.0;
        flat[13] = -2.0;
        flat[14] = 7.0;
        let m = mat4_from_flat(&flat).expect("16 elements");
        let p = transform_point(m, [0.0, 0.0, 0.0]);
        assert!((p[0] - 3.0).abs() < 1e-6);
        assert!((p[1] + 2.0).abs() < 1e-6);
        assert!((p[2] - 7.0).abs() < 1e-6);
        assert!(mat4_from_flat(&flat[..12]).is_none());
    }

    #[test]
    fn top4_weights_sorts_truncates_and_normalizes() {
        let list = vec![(3, 0.1f32), (1, 0.4), (2, 0.3), (0, 0.15), (4, 0.05)];
        let (joints, weights) = top4_weights(Some(&list));
        // Largest four survive, ordered by weight.
        assert_eq!(joints, [1, 2, 0, 3]);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(weights[0] > weights[1] && weights[1] > weights[2]);
    }

    #[test]
    fn top4_weights_defaults_an_unweighted_point_to_joint_zero() {
        assert_eq!(top4_weights(None), ([0; 4], [1.0, 0.0, 0.0, 0.0]));
        let zero = vec![(2, 0.0f32)];
        assert_eq!(top4_weights(Some(&zero)), ([0; 4], [1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn top4_weights_breaks_ties_by_joint_index() {
        let list = vec![(9, 0.5f32), (2, 0.5)];
        let (joints, _) = top4_weights(Some(&list));
        assert_eq!(joints[0], 2, "equal weights order by joint index");
        assert_eq!(joints[1], 9);
    }

    fn import(doc: fx::Doc) -> Result<ImportedSkinnedMesh, String> {
        let file = doc.write();
        import_skinned_fbx(file.path(), 0)
    }

    #[test]
    fn import_skinned_fbx_bakes_the_bind_frame_into_the_vertices() {
        let mesh = import(fx::two_bone_rig(100.0)).expect("skinned import");

        // The mesh model's +X geometric offset is baked in; the cluster
        // Transform x TransformLink product is identity for this rig.
        assert_eq!(mesh.vertices.len(), 3);
        assert_vec3_eq(mesh.vertices[0].pos, [1.0, 0.0, 0.0]);
        assert_vec3_eq(mesh.vertices[1].pos, [2.0, 0.0, 0.0]);
        assert_vec3_eq(mesh.vertices[2].pos, [1.0, 1.0, 0.0]);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert!(mesh.morph_target_names.is_empty());
        assert!(mesh.morph_deltas.is_empty());
    }

    #[test]
    fn import_skinned_fbx_rebuilds_joint_locals_from_the_cluster_bind_worlds() {
        let mesh = import(fx::two_bone_rig(100.0)).expect("skinned import");

        assert_eq!(mesh.skeleton.len(), 2);
        let root = &mesh.skeleton[0];
        assert_eq!(root.name, "Root");
        assert_eq!(root.parent, -1);
        assert_vec3_eq(root.translation, [0.0, 0.0, 0.0]);
        assert_vec3_eq(root.scale, [1.0, 1.0, 1.0]);

        // Tip's TransformLink is two units up; relative to Root that is its local.
        let tip = &mesh.skeleton[1];
        assert_eq!(tip.name, "Tip");
        assert_eq!(tip.parent, 0);
        assert_vec3_eq(tip.translation, [0.0, 2.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_normalizes_the_top_four_cluster_weights() {
        let mesh = import(fx::two_bone_rig(100.0)).expect("skinned import");

        assert_eq!(mesh.vertices[0].joints, [0, 0, 0, 0]);
        assert_eq!(mesh.vertices[0].weights, [1.0, 0.0, 0.0, 0.0]);
        // Control point 1 is bound half to each joint.
        assert_eq!(mesh.vertices[1].joints, [0, 1, 0, 0]);
        assert_eq!(mesh.vertices[1].weights, [0.5, 0.5, 0.0, 0.0]);
        assert_eq!(mesh.vertices[2].joints, [1, 0, 0, 0]);
        assert_eq!(mesh.vertices[2].weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_normalizes_a_centimeter_rig_to_meters() {
        let mesh = import(fx::two_bone_rig(1.0)).expect("skinned import");

        assert_vec3_eq(mesh.vertices[0].pos, [0.01, 0.0, 0.0]);
        assert_vec3_eq(mesh.vertices[1].pos, [0.02, 0.0, 0.0]);
        // Only the root local carries the unit compensation.
        assert_vec3_eq(mesh.skeleton[0].scale, [0.01, 0.01, 0.01]);
        assert_vec3_eq(mesh.skeleton[1].translation, [0.0, 2.0, 0.0]);
        assert_vec3_eq(mesh.skeleton[1].scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn import_skinned_fbx_ignores_zero_and_missing_cluster_weights() {
        let mut doc = fx::two_bone_rig(100.0);
        // The Root cluster claims control point 2 but supplies no weight for it.
        doc.replace_object(
            fx::ROOT_CLUSTER_ID,
            fx::cluster(
                fx::ROOT_CLUSTER_ID,
                "RootCluster",
                vec![0, 1, 2],
                vec![1.0, 0.5],
            )
            .child(fx::transform_link(fx::flat_translation([0.0, 0.0, 0.0])))
            .child(fx::transform(fx::flat_translation([0.0, 0.0, 0.0]))),
        );
        let mesh = import(doc).expect("skinned import");
        assert_eq!(mesh.vertices[2].joints, [1, 0, 0, 0]);
        assert_eq!(mesh.vertices[2].weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_drops_clusters_without_a_bone_link() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.drop_connection(fx::TIP_BONE_ID, fx::TIP_CLUSTER_ID);
        let mesh = import(doc).expect("skinned import");

        assert_eq!(mesh.skeleton.len(), 1);
        assert_eq!(mesh.skeleton[0].name, "Root");
        // The Tip-only control point loses its weights and falls back to joint 0.
        assert_eq!(mesh.vertices[2].joints, [0, 0, 0, 0]);
        assert_eq!(mesh.vertices[2].weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_keeps_a_bind_frame_of_identity_without_cluster_transforms() {
        let mut doc = fx::two_bone_rig(100.0);
        for (id, name, indexes, weights, link) in [
            (
                fx::ROOT_CLUSTER_ID,
                "RootCluster",
                vec![0, 1],
                vec![1.0, 0.5],
                0.0,
            ),
            (
                fx::TIP_CLUSTER_ID,
                "TipCluster",
                vec![1, 2],
                vec![0.5, 1.0],
                2.0,
            ),
        ] {
            doc.replace_object(
                id,
                fx::cluster(id, name, indexes, weights)
                    .child(fx::transform_link(fx::flat_translation([0.0, link, 0.0]))),
            );
        }
        let mesh = import(doc).expect("skinned import");
        // Without a cluster Transform the bind frame is just the geometric offset.
        assert_vec3_eq(mesh.vertices[0].pos, [1.0, 0.0, 0.0]);
        assert_vec3_eq(mesh.skeleton[1].translation, [0.0, 2.0, 0.0]);
    }

    // A rig whose weighted bone hangs off an unweighted parent: the parent has
    // no cluster, so its bind world comes from its scene pose.
    fn rig_with_an_unclustered_parent() -> fx::Doc {
        const HIPS_ID: i64 = 302;
        fx::Doc {
            unit_scale_factor: 100.0,
            objects: vec![
                fx::geometry(
                    fx::GEOMETRY_ID,
                    "mesh",
                    vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    vec![0, 1, -3],
                ),
                fx::model(fx::MESH_MODEL_ID, "MeshNode", "Mesh"),
                fx::model(HIPS_ID, "Hips", "LimbNode").child(fx::properties70(vec![fx::p_vec3(
                    "Lcl Translation",
                    [0.0, 1.0, 0.0],
                )])),
                fx::model(fx::ROOT_BONE_ID, "Root", "LimbNode"),
                fx::skin_deformer(fx::SKIN_ID, "Skin"),
                fx::cluster(
                    fx::ROOT_CLUSTER_ID,
                    "RootCluster",
                    vec![0, 1, 2],
                    vec![1.0, 1.0, 1.0],
                )
                .child(fx::transform_link(fx::flat_translation([0.0, 3.0, 0.0])))
                .child(fx::transform(fx::flat_translation([0.0, -1.0, 0.0]))),
            ],
            connections: vec![
                fx::oo(fx::MESH_MODEL_ID, 0),
                fx::oo(HIPS_ID, 0),
                fx::oo(fx::ROOT_BONE_ID, HIPS_ID),
                fx::oo(fx::SKIN_ID, fx::GEOMETRY_ID),
                fx::oo(fx::ROOT_CLUSTER_ID, fx::SKIN_ID),
                fx::oo(fx::ROOT_BONE_ID, fx::ROOT_CLUSTER_ID),
            ],
        }
    }

    #[test]
    fn import_skinned_fbx_extends_the_chain_with_unclustered_parent_scene_poses() {
        let mesh = import(rig_with_an_unclustered_parent()).expect("skinned import");

        assert_eq!(mesh.skeleton.len(), 2);
        // Parents come before children even though only the child has a cluster.
        assert_eq!(mesh.skeleton[0].name, "Hips");
        assert_eq!(mesh.skeleton[0].parent, -1);
        assert_vec3_eq(mesh.skeleton[0].translation, [0.0, 1.0, 0.0]);
        assert_eq!(mesh.skeleton[1].name, "Root");
        assert_eq!(mesh.skeleton[1].parent, 0);
        // Root's cluster bind world is 3 up; relative to Hips that is 2.
        assert_vec3_eq(mesh.skeleton[1].translation, [0.0, 2.0, 0.0]);
        // The bind frame is TransformLink x Transform; with no Geometry -> Model
        // connection there is no geometric offset on top of it.
        assert_vec3_eq(mesh.vertices[0].pos, [0.0, 2.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_binds_the_first_skinned_geometry() {
        let mut doc = fx::two_bone_rig(100.0);
        // An earlier, unskinned geometry must not win.
        doc.objects.insert(
            0,
            fx::geometry(
                101,
                "decoration",
                vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                vec![0, 1, -3],
            ),
        );
        let mesh = import(doc).expect("skinned import");
        assert_vec3_eq(mesh.vertices[0].pos, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_reads_the_texture_uv_layer() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.attach(
            fx::GEOMETRY_ID,
            fx::uv_layer("Lightmap", vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9], None),
        );
        doc.attach(
            fx::GEOMETRY_ID,
            fx::uv_layer("TextureUV", vec![0.25, 0.0, 0.5, 0.0, 0.75, 0.0], None),
        );
        let mesh = import(doc).expect("skinned import");
        assert_eq!(mesh.vertices[0].uv, [0.25, 1.0]);
        assert_eq!(mesh.vertices[1].uv, [0.5, 1.0]);
        assert_eq!(mesh.vertices[2].uv, [0.75, 1.0]);
    }

    #[test]
    fn import_skinned_fbx_deduplicates_corners_sharing_an_indexed_uv() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.replace_object(
            fx::GEOMETRY_ID,
            fx::geometry(
                fx::GEOMETRY_ID,
                "quad",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![0, 1, -3, 0, 2, -4],
            )
            .child(fx::uv_layer(
                "TextureUV",
                vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                Some(vec![0, 1, 2, 0, 2, 3]),
            )),
        );
        let mesh = import(doc).expect("skinned import");
        // Six polygon-vertices collapse onto four (control point, uv) pairs.
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(mesh.vertices[3].uv, [0.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_rejects_meshes_past_the_u16_index_limit() {
        let triangles = (u16::MAX as usize + 3) / 3;
        let mut pvi = Vec::with_capacity(triangles * 3);
        for _ in 0..triangles {
            pvi.extend_from_slice(&[0, 0, !0]);
        }
        let mut doc = fx::two_bone_rig(100.0);
        doc.replace_object(
            fx::GEOMETRY_ID,
            fx::geometry(fx::GEOMETRY_ID, "huge", vec![0.0, 0.0, 0.0], pvi),
        );
        let err = import(doc).err().expect("index limit");
        assert!(err.contains("u16 index limit"), "got: {err}");
    }

    #[test]
    fn import_skinned_fbx_reports_geometry_without_polygon_data() {
        let empty = [
            fx::object("Geometry", fx::GEOMETRY_ID, "mesh", "Mesh"),
            // Control points without polygons.
            fx::object("Geometry", fx::GEOMETRY_ID, "mesh", "Mesh")
                .child(fx::node("Vertices").arr_f64(vec![0.0, 0.0, 0.0])),
            // Declared but empty polygon data emits no vertices either.
            fx::geometry(fx::GEOMETRY_ID, "mesh", vec![0.0, 0.0, 0.0], Vec::new()),
        ];
        for geom in empty {
            let mut doc = fx::two_bone_rig(100.0);
            doc.replace_object(fx::GEOMETRY_ID, geom);
            let err = import(doc).err().expect("no polygon data");
            assert!(err.contains("has no polygon data"), "got: {err}");
        }
    }

    #[test]
    fn import_skinned_fbx_ignores_other_deformers_and_malformed_connections() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.objects
            .push(fx::object("Deformer", 450, "shape", "BlendShape"));
        // An object node without an id is skipped by the object scan.
        doc.objects.push(fx::node("Deformer"));
        doc.connections.extend([
            // A connection record without its object ids.
            fx::node("C").text("OO"),
            fx::oo(450, fx::GEOMETRY_ID),
            // A bone parented to an object that is not in the file.
            fx::oo(fx::ROOT_BONE_ID, 999),
        ]);
        let mesh = import(doc).expect("skinned import");
        assert_eq!(mesh.skeleton.len(), 2);
        assert_eq!(mesh.skeleton[0].name, "Root");
        assert_eq!(mesh.vertices.len(), 3);
    }

    #[test]
    fn import_skinned_fbx_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing.fbx");
        let err = import_skinned_fbx(path.to_str().expect("path"), 0)
            .err()
            .expect("missing file");
        assert!(err.contains("could not open"), "got: {err}");
    }

    #[test]
    fn import_skinned_fbx_accepts_a_bone_with_no_parent_connection() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.drop_connection(fx::ROOT_BONE_ID, 0);
        let mesh = import(doc).expect("skinned import");
        assert_eq!(mesh.skeleton.len(), 2);
        assert_eq!(mesh.skeleton[0].name, "Root");
        assert_eq!(mesh.skeleton[0].parent, -1);
        assert_eq!(mesh.skeleton[1].parent, 0);
    }

    #[test]
    fn import_skinned_fbx_reports_a_skin_in_a_file_without_connections() {
        let doc = fx::two_bone_rig(100.0);
        let file = fx::write(vec![fx::objects(doc.objects)]);
        let err = import_skinned_fbx(file.path(), 0)
            .err()
            .expect("no connections");
        assert!(
            err.contains("no geometry is bound to a skin deformer"),
            "got: {err}"
        );
    }

    // Two bones that each claim the other as parent. No topological order
    // exists, so the ordering pass has to fall back to discovery order.
    fn rig_with_a_parent_cycle() -> fx::Doc {
        const ALPHA_ID: i64 = 300;
        const BETA_ID: i64 = 301;
        const ALPHA_CLUSTER: i64 = 401;
        const BETA_CLUSTER: i64 = 402;
        fx::Doc {
            unit_scale_factor: 100.0,
            objects: vec![
                fx::geometry(
                    fx::GEOMETRY_ID,
                    "mesh",
                    vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    vec![0, 1, -3],
                ),
                fx::model(ALPHA_ID, "Alpha", "LimbNode"),
                fx::model(BETA_ID, "Beta", "LimbNode"),
                fx::skin_deformer(fx::SKIN_ID, "Skin"),
                fx::cluster(ALPHA_CLUSTER, "AlphaCluster", vec![0], vec![1.0])
                    .child(fx::transform_link(fx::flat_translation([0.0, 7.0, 0.0]))),
                fx::cluster(BETA_CLUSTER, "BetaCluster", vec![1, 2], vec![1.0, 1.0])
                    .child(fx::transform_link(fx::flat_translation([0.0, 5.0, 0.0]))),
            ],
            connections: vec![
                fx::oo(ALPHA_ID, BETA_ID),
                fx::oo(BETA_ID, ALPHA_ID),
                fx::oo(fx::SKIN_ID, fx::GEOMETRY_ID),
                fx::oo(ALPHA_CLUSTER, fx::SKIN_ID),
                fx::oo(BETA_CLUSTER, fx::SKIN_ID),
                fx::oo(ALPHA_ID, ALPHA_CLUSTER),
                fx::oo(BETA_ID, BETA_CLUSTER),
            ],
        }
    }

    #[test]
    fn import_skinned_fbx_emits_a_cyclic_parent_chain_in_discovery_order() {
        let mesh = import(rig_with_a_parent_cycle()).expect("skinned import");

        assert_eq!(mesh.skeleton.len(), 2);
        // Alpha's chain walk discovers Beta first, so Beta leads and becomes
        // the root; Alpha then resolves against it.
        assert_eq!(mesh.skeleton[0].name, "Beta");
        assert_eq!(mesh.skeleton[0].parent, -1);
        assert_vec3_eq(mesh.skeleton[0].translation, [0.0, 5.0, 0.0]);
        assert_eq!(mesh.skeleton[1].name, "Alpha");
        assert_eq!(mesh.skeleton[1].parent, 0);
        assert_vec3_eq(mesh.skeleton[1].translation, [0.0, 2.0, 0.0]);
    }

    #[test]
    fn import_skinned_fbx_reports_a_file_without_an_objects_section() {
        let file = fx::write(vec![fx::node("Definitions")]);
        let err = import_skinned_fbx(file.path(), 0)
            .err()
            .expect("no Objects section");
        assert!(err.contains("FBX has no Objects section"), "got: {err}");
    }

    #[test]
    fn import_skinned_fbx_reports_a_file_without_a_skin_deformer() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(100, "mesh", vec![0.0; 9], vec![0, 1, -3]),
            fx::model(200, "Mesh", "Mesh"),
        ];
        doc.connections = vec![fx::oo(100, 200)];
        let err = import(doc).err().expect("no skin deformer");
        assert!(err.contains("no skin deformer"), "got: {err}");
    }

    #[test]
    fn import_skinned_fbx_reports_a_skin_bound_to_no_geometry() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.drop_connection(fx::SKIN_ID, fx::GEOMETRY_ID);
        let err = import(doc).err().expect("unbound skin");
        assert!(
            err.contains("no geometry is bound to a skin deformer"),
            "got: {err}"
        );
    }

    #[test]
    fn import_skinned_fbx_reports_a_skin_without_clusters() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.drop_connection(fx::ROOT_CLUSTER_ID, fx::SKIN_ID);
        doc.drop_connection(fx::TIP_CLUSTER_ID, fx::SKIN_ID);
        let err = import(doc).err().expect("cluster-less skin");
        assert!(err.contains("skin deformer has no clusters"), "got: {err}");
    }

    #[test]
    fn import_skinned_fbx_reports_clusters_without_bone_links() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.drop_connection(fx::ROOT_BONE_ID, fx::ROOT_CLUSTER_ID);
        doc.drop_connection(fx::TIP_BONE_ID, fx::TIP_CLUSTER_ID);
        let err = import(doc).err().expect("bone-less clusters");
        assert!(
            err.contains("skin clusters have no bone links"),
            "got: {err}"
        );
    }

    #[test]
    fn skin_index_selects_each_skinned_geometry_in_declaration_order() {
        let file = fx::two_part_rig(100.0).write();

        // The first skinned geometry is the body triangle, weighted across
        // both bones; the second is the hair triangle, bound to Tip alone.
        let body = import_skinned_fbx(file.path(), 0).expect("body import");
        let hair = import_skinned_fbx(file.path(), 1).expect("hair import");
        assert_eq!(body.vertices.len(), 3);
        assert_eq!(hair.vertices.len(), 3);
        assert_ne!(body.vertices[0].pos, hair.vertices[0].pos);

        // Both parts carry their own copy of the shared skeleton.
        let names = |m: &ImportedSkinnedMesh| -> Vec<String> {
            m.skeleton.iter().map(|j| j.name.clone()).collect()
        };
        assert_eq!(names(&body), vec!["Root".to_string(), "Tip".to_string()]);
        assert_eq!(names(&hair), names(&body));

        // The hair is fully weighted to Tip, so its own cluster drives it.
        let tip = hair
            .skeleton
            .iter()
            .position(|j| j.name == "Tip")
            .expect("Tip joint") as u32;
        for v in &hair.vertices {
            assert_eq!(v.joints[0], tip);
            assert!((v.weights[0] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn skin_index_past_the_last_skinned_geometry_errors() {
        let file = fx::two_part_rig(100.0).write();
        let err = import_skinned_fbx(file.path(), 2)
            .err()
            .expect("only two skins");
        assert!(
            err.contains("skin_index 2 out of range") && err.contains("2 skinned meshes"),
            "got: {err}"
        );

        // A single-skin file pluralizes its count correctly too.
        let one = fx::two_bone_rig(100.0).write();
        let err = import_skinned_fbx(one.path(), 1)
            .err()
            .expect("only one skin");
        assert!(err.contains("1 skinned mesh)"), "got: {err}");
    }
}
