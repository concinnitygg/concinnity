// src/fbx/skin.rs
//
// Skinned-mesh import from binary FBX: reads Deformer(Skin) / SubDeformer
// (Cluster) objects, builds a parents-before-children skeleton, and assembles
// skinned vertices with top-4 normalized weights per control point.
//
// The runtime recomputes inverse-bind matrices from the JointDef chain, so a
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
use crate::assets::{JointDef, SkinnedVertexData, VertexData};
use crate::gfx::skinning::{
    IDENTITY, Mat4, decompose, euler_yxz_from_quat, mat4_affine_inverse, mat4_mul,
};
use crate::glb::ImportedSkinnedMesh;

// The skinning-relevant slice of an FBX document: the first skinned geometry,
// its skeleton, and its per-control-point weights.
pub(super) struct FbxSkin {
    pub geometry_id: i64,
    // Mesh world at bind (cluster `Transform` x geometric offset): baked into
    // vertex positions so they live in the same frame as the joint worlds.
    pub mesh_bind: Mat4,
    // Meters per file unit; joint translations are already normalized by it,
    // and evaluated animation translations must be too.
    pub unit_scale: f32,
    pub joints: Vec<JointDef>,
    // Model object id -> joint index, for resolving animation curve targets.
    pub model_to_joint: HashMap<i64, usize>,
    // Control-point index -> accumulated (joint, weight) pairs.
    pub weights: HashMap<i32, Vec<(usize, f32)>>,
}

// Locate the first skinned geometry and extract its skeleton + weights.
pub(super) fn parse_skin(root: NodeHandle<'_>, path: &str) -> Result<FbxSkin, String> {
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

    let (geometry_id, skin_id) = geom_order
        .iter()
        .find_map(|g| skin_of_geometry.get(g).map(|s| (*g, *s)))
        .ok_or_else(|| format!("'{path}': no geometry is bound to a skin deformer"))?;
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
    let mut joints: Vec<JointDef> = Vec::new();
    for (i, &id) in order.iter().enumerate() {
        let local = match parents[i] {
            Some(j) => mat4_mul(mat4_affine_inverse(worlds[j]), worlds[i]),
            None => uniform_scale_pre(unit_scale, worlds[i]),
        };
        let (translation, rotation, scale) = decompose(local);
        joints.push(JointDef {
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

// Import the first skinned mesh of a binary FBX into the inline `SkinnedMesh`
// fields, mirroring the glTF importer's output shape.
pub fn import_skinned_fbx(path: &str) -> Result<ImportedSkinnedMesh, String> {
    let tree = super::load_tree(path)?;
    let root = tree.root();
    let skin = parse_skin(root, path)?;

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
}
