// Baking a CharacterShape into its target: the slider weights are applied to
// the vertices and the morph targets dropped, the bind pose is rewritten
// through the proportion layer (with the vertices re-skinned to it), the
// capsule follows, and the shape asset is consumed. The mesh that comes out
// is a plain SkinnedMesh with no shape work left for the runtime.

use crate::components::{CharacterShape, MorphDelta, SkeletonJoint, SkinnedMesh};
use crate::ecs::Component;
use concinnity_core::components::build_skeleton_from_joint_defs;
use concinnity_core::gfx::proportions::ProportionLayer;
use concinnity_core::gfx::transform::{self, Mat4};
use concinnity_core::math::vec3;
use concinnity_world::world::WorldJsonlAsset;

// What a bake changed, for the build log.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BakeReport {
    pub targets_applied: usize,
    pub joints_changed: usize,
}

// Positions with the weighted morph deltas folded in.
pub(crate) fn morphed_positions(mesh: &SkinnedMesh, weights: &[f32]) -> Vec<[f32; 3]> {
    let n = mesh.vertices.len();
    let mut out: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.pos).collect();
    for (t, w) in weights.iter().enumerate() {
        if *w == 0.0 {
            continue;
        }
        let Some(block) = mesh.morph_deltas.get(t * n..(t + 1) * n) else {
            break;
        };
        for (p, d) in out.iter_mut().zip(block) {
            vec3::vec3_add(p, vec3::scale(d.position, *w));
        }
    }
    out
}

fn transform_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    let mut out = [m[3][0], m[3][1], m[3][2]];
    for (c, col) in m.iter().enumerate().take(3) {
        out[0] += col[0] * p[c];
        out[1] += col[1] * p[c];
        out[2] += col[2] * p[c];
    }
    out
}

// The proportioned bind pose: every joint's new local pose and the skin
// matrices that carry a bind-pose vertex onto it.
struct ProportionedBind {
    joints: Vec<SkeletonJoint>,
    skin: Vec<Mat4>,
    height_ratio: f32,
    root_scale: f32,
    changed: usize,
}

fn proportioned_bind(joints: &[SkeletonJoint], shape: &CharacterShape) -> ProportionedBind {
    let skeleton = build_skeleton_from_joint_defs(joints);
    let layer = ProportionLayer::resolve(&skeleton, &shape.proportions);
    let mut locals: Vec<Mat4> = skeleton.bind_locals().to_vec();
    layer.apply(&mut locals);
    let mut skin = Vec::new();
    skeleton.skinning_matrices_into(&locals, &mut skin);
    let mut changed = 0;
    let new_joints = joints
        .iter()
        .zip(&locals)
        .zip(skeleton.bind_locals())
        .map(|((j, local), bind)| {
            if local == bind {
                return j.clone();
            }
            changed += 1;
            let (translation, quat, scale) = transform::decompose(*local);
            SkeletonJoint {
                name: j.name.clone(),
                parent: j.parent,
                translation,
                rotation_deg: transform::euler_yxz_from_quat(quat),
                scale,
            }
        })
        .collect();
    ProportionedBind {
        joints: new_joints,
        skin,
        height_ratio: layer.height_ratio(&skeleton),
        root_scale: layer.root_scale(&skeleton),
        changed,
    }
}

// Flatten `shape` into `mesh` (whose `skeleton` rides `joints`), in place.
pub(crate) fn bake(
    shape: &CharacterShape,
    mesh: &mut SkinnedMesh,
    joints: &mut Vec<SkeletonJoint>,
) -> BakeReport {
    let resolved = shape.resolve_sliders(&mesh.morph_target_names);
    let targets_applied = resolved.weights.iter().filter(|w| **w != 0.0).count();
    let positions = morphed_positions(mesh, &resolved.weights);
    let bind = proportioned_bind(joints, shape);
    for (v, p) in mesh.vertices.iter_mut().zip(positions) {
        let sum: f32 = v.weights.iter().sum();
        v.pos = if sum <= 1e-6 || bind.changed == 0 {
            p
        } else {
            let mut out = [0.0_f32; 3];
            for k in 0..4 {
                let w = v.weights[k] / sum;
                if w == 0.0 {
                    continue;
                }
                let Some(m) = bind.skin.get(v.joints[k] as usize) else {
                    continue;
                };
                vec3::vec3_add(&mut out, vec3::scale(transform_point(m, p), w));
            }
            out
        };
    }
    mesh.morph_target_names = Vec::new();
    mesh.morph_deltas = Vec::<MorphDelta>::new();
    if let Some(c) = &mut mesh.capsule {
        c.half_height *= bind.height_ratio;
        c.radius *= bind.root_scale;
    }
    *joints = bind.joints;
    BakeReport {
        targets_applied,
        joints_changed: bind.changed,
    }
}

fn is_baking(a: &WorldJsonlAsset) -> bool {
    a.asset_type == CharacterShape::NAME
        && a.args.get("bake").and_then(|v| v.as_bool()) == Some(true)
}

// The args of the baking shape that targets `mesh`, if one does.
pub(crate) fn baking_shape_args(
    assets: &[WorldJsonlAsset],
    mesh: &str,
) -> Option<serde_json::Value> {
    assets
        .iter()
        .find(|a| is_baking(a) && a.args.get("target").and_then(|v| v.as_str()) == Some(mesh))
        .map(|a| a.args.clone())
}

// Run every `bake: true` CharacterShape against its target and drop the
// shape from the list. Runs after the mesh import passes. A target whose
// payload the cache already holds (so no geometry is inline) was baked when
// that payload was compiled; only its capsule, which rides the args, is
// redone, from the payload's skeleton (`cached` looks the bytes up).
pub(crate) fn bake_shapes<'c>(
    assets: &mut Vec<WorldJsonlAsset>,
    cached: impl Fn(&str) -> Option<&'c [u8]>,
) -> std::io::Result<()> {
    let invalid = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    let baked: Vec<usize> = assets
        .iter()
        .enumerate()
        .filter(|(_, a)| is_baking(a))
        .map(|(i, _)| i)
        .collect();
    for &i in baked.iter().rev() {
        let name = assets[i].name.clone();
        let shape = CharacterShape {
            sliders: serde_json::from_value(
                assets[i].args.get("sliders").cloned().unwrap_or_default(),
            )
            .unwrap_or_default(),
            proportions: serde_json::from_value(
                assets[i]
                    .args
                    .get("proportions")
                    .cloned()
                    .unwrap_or_default(),
            )
            .unwrap_or_default(),
            ..Default::default()
        };
        let target = assets[i]
            .args
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let Some(t) = assets.iter().position(|a| a.name == target) else {
            return Err(invalid(format!(
                "Asset '{name}': bake target '{target}' is not in the world"
            )));
        };
        let args = assets[t]
            .args
            .as_object_mut()
            .ok_or_else(|| invalid(format!("Asset '{target}': args is not a JSON object")))?;
        let mut mesh: SkinnedMesh = serde_json::from_value(serde_json::Value::Object(args.clone()))
            .map_err(|e| invalid(format!("Asset '{target}': {e}")))?;
        let encode = |v: serde_json::Result<serde_json::Value>| {
            v.map_err(|e| {
                invalid(format!(
                    "Asset '{target}': failed to encode baked mesh: {e}"
                ))
            })
        };
        let mut joints: Vec<SkeletonJoint> = args
            .get("skeleton")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| invalid(format!("Asset '{target}': invalid skeleton: {e}")))?
            .unwrap_or_default();
        if mesh.vertices.is_empty() {
            let bytes = cached(&target).ok_or_else(|| {
                invalid(format!(
                    "Asset '{name}': bake target '{target}' has no geometry to bake"
                ))
            })?;
            let payload = concinnity_core::gfx::mesh_payload::deserialise_skinned_with_lods(bytes)
                .map_err(|e| invalid(format!("Asset '{target}': cached payload: {e}")))?;
            joints = crate::geometry::payload_joints_to_defs(payload.joints);
            let report = bake(&shape, &mut mesh, &mut joints);
            if let Some(c) = &mesh.capsule {
                args.insert("capsule".into(), encode(serde_json::to_value(c))?);
            }
            tracing::info!(
                "Asset '{name}': '{target}' is baked in its cached payload ({} joint(s) changed)",
                report.joints_changed
            );
            assets.remove(i);
            continue;
        }
        let report = bake(&shape, &mut mesh, &mut joints);
        args.insert(
            "vertices".into(),
            encode(serde_json::to_value(&mesh.vertices))?,
        );
        args.insert("skeleton".into(), encode(serde_json::to_value(&joints))?);
        args.remove("morph_target_names");
        args.remove("morph_deltas");
        if let Some(c) = &mesh.capsule {
            args.insert("capsule".into(), encode(serde_json::to_value(c))?);
        }
        tracing::info!(
            "Asset '{name}': baked into '{target}' ({} target(s), {} joint(s) changed)",
            report.targets_applied,
            report.joints_changed
        );
        assets.remove(i);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{CharacterCapsule, JointProportion, ShapeSlider, SkinnedVertexData};

    fn joint(name: &str, parent: i32, y: f32) -> SkeletonJoint {
        SkeletonJoint {
            name: name.into(),
            parent,
            translation: [0.0, y, 0.0],
            ..Default::default()
        }
    }

    fn vertex(pos: [f32; 3], joints: [u32; 4], weights: [f32; 4]) -> SkinnedVertexData {
        SkinnedVertexData {
            pos,
            color: [1.0; 3],
            uv: [0.0; 2],
            joints,
            weights,
        }
    }

    // root at 0, mid at 1, tip at 2; three vertices on the chain.
    fn mesh() -> (SkinnedMesh, Vec<SkeletonJoint>) {
        let joints = vec![
            joint("root", -1, 0.0),
            joint("mid", 0, 1.0),
            joint("tip", 1, 1.0),
        ];
        let mesh = SkinnedMesh {
            vertices: vec![
                vertex([0.0, 0.0, 0.0], [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]),
                vertex([0.0, 1.5, 0.0], [1, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]),
                vertex([0.0, 2.0, 0.0], [1, 2, 0, 0], [0.5, 0.5, 0.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
            morph_target_names: vec!["wide".into(), "lean+".into(), "lean-".into()],
            morph_deltas: (0..9)
                .map(|i| MorphDelta {
                    position: [
                        if i < 3 {
                            1.0
                        } else if i < 6 {
                            0.0
                        } else {
                            -1.0
                        },
                        0.0,
                        0.0,
                    ],
                    normal: [0.0; 3],
                })
                .collect(),
            capsule: Some(CharacterCapsule {
                half_height: 1.0,
                radius: 0.5,
            }),
            ..Default::default()
        };
        (mesh, joints)
    }

    #[test]
    fn sliders_fold_into_the_positions_and_targets_are_dropped() {
        let (mut mesh, mut joints) = mesh();
        let shape = CharacterShape {
            sliders: vec![
                ShapeSlider {
                    name: "wide".into(),
                    value: 0.5,
                },
                ShapeSlider {
                    name: "lean".into(),
                    value: -1.0,
                },
            ],
            ..Default::default()
        };
        let report = bake(&shape, &mut mesh, &mut joints);
        assert_eq!(
            report,
            BakeReport {
                targets_applied: 2,
                joints_changed: 0
            }
        );
        // wide at 0.5 (+0.5 x) and lean- at 1.0 (-1 x).
        assert_eq!(mesh.vertices[0].pos, [-0.5, 0.0, 0.0]);
        assert!(mesh.morph_target_names.is_empty() && mesh.morph_deltas.is_empty());
        assert_eq!(joints.len(), 3);
        assert_eq!(mesh.capsule.unwrap().half_height, 1.0);
    }

    #[test]
    fn proportions_rewrite_the_bind_pose_and_reskin_the_vertices() {
        let (mut mesh, mut joints) = mesh();
        let shape = CharacterShape {
            proportions: vec![JointProportion {
                joint: "mid".into(),
                scale: 1.0,
                length: 0.5,
            }],
            ..Default::default()
        };
        let report = bake(&shape, &mut mesh, &mut joints);
        assert_eq!(report.joints_changed, 1, "only tip's local moves");
        assert!((joints[2].translation[1] - 1.5).abs() < 1e-5);
        assert_eq!(joints[1].translation[1], 1.0);
        // A vertex bound to mid stays; one half-bound to tip moves half the
        // offset; the capsule follows the height ratio (2.5 / 2).
        assert!((mesh.vertices[1].pos[1] - 1.5).abs() < 1e-5);
        assert!(
            (mesh.vertices[2].pos[1] - 2.25).abs() < 1e-5,
            "{:?}",
            mesh.vertices[2].pos
        );
        assert!((mesh.capsule.as_ref().unwrap().half_height - 1.25).abs() < 1e-5);
        assert_eq!(mesh.capsule.unwrap().radius, 0.5);
    }

    #[test]
    fn a_root_scale_scales_the_capsule_radius() {
        let (mut mesh, mut joints) = mesh();
        let shape = CharacterShape {
            proportions: vec![JointProportion {
                joint: "root".into(),
                scale: 1.2,
                length: 0.0,
            }],
            ..Default::default()
        };
        bake(&shape, &mut mesh, &mut joints);
        assert!((joints[0].scale[0] - 1.2).abs() < 1e-5);
        assert!((mesh.capsule.as_ref().unwrap().radius - 0.6).abs() < 1e-5);
        assert!(
            (mesh.vertices[1].pos[1] - 1.8).abs() < 1e-5,
            "{:?}",
            mesh.vertices[1].pos
        );
    }

    #[test]
    fn the_pass_consumes_baked_shapes_and_rewrites_the_target_args() {
        let (mesh, joints) = mesh();
        let mut args = serde_json::to_value(&mesh).unwrap();
        args["skeleton"] = serde_json::to_value(&joints).unwrap();
        let mut assets = vec![
            WorldJsonlAsset {
                name: "body".into(),
                asset_type: "SkinnedMesh".into(),
                args,
            },
            WorldJsonlAsset {
                name: "shape".into(),
                asset_type: "CharacterShape".into(),
                args: serde_json::json!({"target": "body", "bake": true,
                    "sliders": [{"name": "wide", "value": 1.0}]}),
            },
            WorldJsonlAsset {
                name: "live".into(),
                asset_type: "CharacterShape".into(),
                args: serde_json::json!({"target": "body"}),
            },
        ];
        bake_shapes(&mut assets, |_| None).expect("bake");
        assert_eq!(assets.len(), 2);
        assert!(assets.iter().all(|a| a.name != "shape"));
        let body = &assets[0].args;
        assert!(body.get("morph_deltas").is_none());
        assert_eq!(body["vertices"][0]["pos"][0], 1.0);
        assert_eq!(body["skeleton"].as_array().unwrap().len(), 3);
        let mut missing = vec![WorldJsonlAsset {
            name: "shape".into(),
            asset_type: "CharacterShape".into(),
            args: serde_json::json!({"target": "ghost", "bake": true}),
        }];
        let err = bake_shapes(&mut missing, |_| None).unwrap_err();
        assert!(
            err.to_string().contains("'ghost' is not in the world"),
            "{err}"
        );
    }

    #[test]
    fn a_cached_target_only_redoes_its_capsule_from_the_payload_skeleton() {
        let (mesh, joints) = mesh();
        let payload = crate::geometry::compile_skinned_mesh_payload_with_lods(
            &mesh.vertices,
            &mesh.indices,
            &joints,
            &[],
            &[],
            &crate::geometry::SkinnedLods {
                levels: 1,
                distances: &[],
            },
        )
        .unwrap();
        let shape_args = serde_json::json!({"target": "body", "bake": true,
            "proportions": [{"joint": "mid", "length": 0.5}]});
        let mut assets = vec![
            WorldJsonlAsset {
                name: "body".into(),
                asset_type: "SkinnedMesh".into(),
                args: serde_json::json!({"source": "hero.glb", "capsule": {"half_height": 1.0, "radius": 0.5}}),
            },
            WorldJsonlAsset {
                name: "shape".into(),
                asset_type: "CharacterShape".into(),
                args: shape_args.clone(),
            },
        ];
        assert_eq!(baking_shape_args(&assets, "body"), Some(shape_args));
        assert_eq!(baking_shape_args(&assets, "other"), None);
        bake_shapes(&mut assets, |name| {
            (name == "body").then_some(payload.as_slice())
        })
        .expect("bake");
        assert_eq!(assets.len(), 1);
        let body = &assets[0].args;
        assert!(
            body.get("vertices").is_none(),
            "the cached payload is left alone"
        );
        assert!((body["capsule"]["half_height"].as_f64().unwrap() - 1.25).abs() < 1e-5);
        let mut assets = vec![
            WorldJsonlAsset {
                name: "body".into(),
                asset_type: "SkinnedMesh".into(),
                args: serde_json::json!({"source": "hero.glb"}),
            },
            WorldJsonlAsset {
                name: "shape".into(),
                asset_type: "CharacterShape".into(),
                args: serde_json::json!({"target": "body", "bake": true}),
            },
        ];
        let err = bake_shapes(&mut assets, |_| None).unwrap_err();
        assert!(err.to_string().contains("no geometry to bake"), "{err}");
    }
}
