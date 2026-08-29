// src/editor/gltf_export/bake.rs
//
// Baking a CharacterShape into an ExportMesh before writing: slider weights
// are folded into the vertex positions and normals, the morph targets dropped,
// and the bind pose rewritten through the proportion layer with the vertices
// re-skinned onto it. Mirrors the cook's bake pass (character/bake.rs) on the
// export form so a baked file needs no shape work in the target tool.

use super::ExportMesh;
use crate::components::{CharacterShape, SkeletonJoint};
use concinnity_core::components::build_skeleton_from_joint_defs;
use concinnity_core::gfx::proportions::ProportionLayer;
use concinnity_core::gfx::transform::{Mat4, decompose, euler_yxz_from_quat};
use concinnity_core::math::vec3;

fn transform_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    let mut out = [m[3][0], m[3][1], m[3][2]];
    for (c, col) in m.iter().enumerate().take(3) {
        out[0] += col[0] * p[c];
        out[1] += col[1] * p[c];
        out[2] += col[2] * p[c];
    }
    out
}

fn transform_direction(m: &Mat4, d: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (c, col) in m.iter().enumerate().take(3) {
        out[0] += col[0] * d[c];
        out[1] += col[1] * d[c];
        out[2] += col[2] * d[c];
    }
    out
}

fn normalized(v: [f32; 3]) -> [f32; 3] {
    let len = vec3::length(v);
    if len > 1e-6 {
        vec3::scale(v, 1.0 / len)
    } else {
        v
    }
}

// Fold `shape` into `mesh` in place: morphs into the vertices, proportions
// into the bind pose. The morph target set is consumed.
pub(crate) fn bake_shape(mesh: &mut ExportMesh, shape: &CharacterShape) {
    let weights = shape.resolve_sliders(&mesh.morph_target_names).weights;
    let n = mesh.positions.len();
    for (t, w) in weights.iter().enumerate() {
        if *w == 0.0 {
            continue;
        }
        let Some(block) = mesh.morph_deltas.get(t * n..(t + 1) * n) else {
            break;
        };
        for (p, d) in mesh.positions.iter_mut().zip(block) {
            vec3::vec3_add(p, vec3::scale(d.position, *w));
        }
        for (nrm, d) in mesh.normals.iter_mut().zip(block) {
            vec3::vec3_add(nrm, vec3::scale(d.normal, *w));
        }
    }
    for nrm in &mut mesh.normals {
        *nrm = normalized(*nrm);
    }
    mesh.morph_target_names = Vec::new();
    mesh.morph_deltas = Vec::new();

    let skeleton = build_skeleton_from_joint_defs(&mesh.skeleton);
    let layer = ProportionLayer::resolve(&skeleton, &shape.proportions);
    if layer.is_empty() {
        return;
    }
    let mut locals: Vec<Mat4> = skeleton.bind_locals().to_vec();
    layer.apply(&mut locals);
    // The matrices carrying a bind-pose vertex onto the proportioned bind.
    let mut skin = Vec::new();
    skeleton.skinning_matrices_into(&locals, &mut skin);
    for v in 0..n {
        let sum: f32 = mesh.weights[v].iter().sum();
        if sum <= 1e-6 {
            continue;
        }
        let mut pos = [0.0_f32; 3];
        let mut nrm = [0.0_f32; 3];
        for k in 0..4 {
            let w = mesh.weights[v][k] / sum;
            if w == 0.0 {
                continue;
            }
            let Some(m) = skin.get(mesh.joints[v][k] as usize) else {
                continue;
            };
            vec3::vec3_add(
                &mut pos,
                vec3::scale(transform_point(m, mesh.positions[v]), w),
            );
            if let Some(base) = mesh.normals.get(v) {
                vec3::vec3_add(&mut nrm, vec3::scale(transform_direction(m, *base), w));
            }
        }
        mesh.positions[v] = pos;
        if let Some(slot) = mesh.normals.get_mut(v) {
            *slot = normalized(nrm);
        }
    }
    mesh.skeleton = mesh
        .skeleton
        .iter()
        .zip(&locals)
        .zip(skeleton.bind_locals())
        .map(|((j, local), bind)| {
            if local == bind {
                return j.clone();
            }
            let (translation, quat, scale) = decompose(*local);
            SkeletonJoint {
                name: j.name.clone(),
                parent: j.parent,
                translation,
                rotation_deg: euler_yxz_from_quat(quat),
                scale,
            }
        })
        .collect();
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::quad_mesh;
    use super::*;
    use crate::components::{JointProportion, ShapeSlider};

    #[test]
    fn sliders_fold_into_positions_and_targets_are_dropped() {
        let mut mesh = quad_mesh();
        let shape = CharacterShape {
            sliders: vec![
                ShapeSlider {
                    name: "wide".into(),
                    value: 0.5,
                },
                ShapeSlider {
                    name: "lift".into(),
                    value: 1.0,
                },
            ],
            ..Default::default()
        };
        bake_shape(&mut mesh, &shape);
        // wide at 0.5 adds 0.05 in X everywhere; lift+ adds 0.2 Y on top verts.
        assert!(
            (mesh.positions[0][0] + 0.45).abs() < 1e-6,
            "{:?}",
            mesh.positions[0]
        );
        assert!(
            (mesh.positions[2][1] - 1.2).abs() < 1e-6,
            "{:?}",
            mesh.positions[2]
        );
        assert!(mesh.morph_target_names.is_empty() && mesh.morph_deltas.is_empty());
        // The lifted normal tilted by its delta and was re-normalised.
        let nrm = mesh.normals[2];
        assert!((vec3::length(nrm) - 1.0).abs() < 1e-5);
        assert!(nrm[1] > 0.0, "{nrm:?}");
        // No proportions: the skeleton is untouched.
        assert_eq!(mesh.skeleton[1].translation, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn proportions_rewrite_the_bind_pose_and_reskin_the_vertices() {
        let mut mesh = quad_mesh();
        let shape = CharacterShape {
            proportions: vec![JointProportion {
                joint: "root".into(),
                scale: 1.0,
                length: 0.5,
            }],
            ..Default::default()
        };
        bake_shape(&mut mesh, &shape);
        // The top joint moved 0.5 along its bind direction (+Y)...
        assert!((mesh.skeleton[1].translation[1] - 1.5).abs() < 1e-5);
        // ...and the vertices bound to it followed.
        assert!(
            (mesh.positions[2][1] - 1.5).abs() < 1e-5,
            "{:?}",
            mesh.positions[2]
        );
        // Vertices on the root stayed put.
        assert!((mesh.positions[0][1]).abs() < 1e-5);
        // A rigid translation leaves the normals alone.
        assert!((mesh.normals[2][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn an_unresolved_slider_changes_nothing() {
        let mut mesh = quad_mesh();
        let before = mesh.positions.clone();
        let shape = CharacterShape {
            sliders: vec![ShapeSlider {
                name: "tail".into(),
                value: 1.0,
            }],
            ..Default::default()
        };
        bake_shape(&mut mesh, &shape);
        assert_eq!(mesh.positions, before);
        assert!(mesh.morph_target_names.is_empty(), "targets still drop");
    }
}
