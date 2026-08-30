// Animation import from binary FBX: AnimationStack / AnimationLayer /
// AnimationCurveNode / AnimationCurve objects are resolved onto the skeleton
// of the file's first skinned mesh, evaluated, and baked at a uniform sample
// rate into the same `ImportedAnimation` form the glTF importer produces, so
// everything downstream (desugar, runtime clips, root motion) is shared.
//
// Curve keys are interpolated linearly. Blender and Mixamo exports bake dense
// per-frame keys, so tangent shaping carries no extra information there; a
// sparsely keyed cubic curve from another tool bakes with linear error.
//
// Rotations honour each node's RotationOrder and PreRotation. Nonzero
// pivots / offsets / PostRotation are outside the supported envelope and log
// a warning instead of silently mis-posing.

use std::collections::{BTreeMap, HashMap};

use fbxcel::tree::v7400::NodeHandle;

use super::{
    arr_f32, arr_i64, attr_i64, attr_str, object_id, object_name, prop_scalar, prop_vec3,
    rot_ordered, rot_ordered_xyz,
};
use crate::gfx::transform::{decompose, euler_yxz_from_quat, mat4_mul};
use crate::import::glb::{ImportedAnimation, ImportedAnimationTrack, ImportedKeyframe};

// FBX time unit: one second is 46,186,158,000 KTime ticks.
const KTIME_PER_SEC: f64 = 46_186_158_000.0;

// Names of every animation stack (clip) in declaration order.
pub(crate) fn fbx_animation_names(path: &str) -> Result<Vec<String>, String> {
    let tree = super::load_tree(path)?;
    let root = tree.root();
    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| format!("'{path}': FBX has no Objects section"))?;
    Ok(objects
        .children()
        .filter(|c| c.name() == "AnimationStack")
        .map(|c| object_name(&c))
        .collect())
}

/// Import one animation clip, selected by `animation_name` (precedence) or
/// `animation_index`, baked at `sample_rate` keys per second and resolved
/// against the `skin_index`-th skinned mesh's skeleton. Curves targeting nodes
/// that are not skeleton joints are dropped, mirroring the glTF importer.
pub fn import_fbx_animation(
    path: &str,
    animation_index: u32,
    animation_name: &str,
    sample_rate: f32,
    skin_index: u32,
) -> Result<ImportedAnimation, String> {
    let tree = super::load_tree(path)?;
    let root = tree.root();
    let skin = super::skin::parse_skin(root, path, skin_index)?;
    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| format!("'{path}': FBX has no Objects section"))?;

    // Object indexes. Layers and curve nodes also carry their declaration
    // rank, which orders channel assignment below.
    let mut stacks: Vec<(i64, NodeHandle)> = Vec::new();
    let mut layer_rank: HashMap<i64, usize> = HashMap::new();
    let mut curve_node_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut curve_node_rank: HashMap<i64, usize> = HashMap::new();
    let mut curve_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut model_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    for c in objects.children() {
        let Some(id) = object_id(&c) else { continue };
        match c.name() {
            "AnimationStack" => stacks.push((id, c)),
            "AnimationLayer" => {
                let rank = layer_rank.len();
                layer_rank.insert(id, rank);
            }
            "AnimationCurveNode" => {
                let rank = curve_node_rank.len();
                curve_node_rank.insert(id, rank);
                curve_node_by_id.insert(id, c);
            }
            "AnimationCurve" => {
                curve_by_id.insert(id, c);
            }
            "Model" => {
                model_by_id.insert(id, c);
            }
            _ => {}
        }
    }
    if stacks.is_empty() {
        return Err(format!("'{path}': FBX has no animation stacks"));
    }

    // Select the stack.
    let stack_idx = if !animation_name.is_empty() {
        stacks
            .iter()
            .position(|(_, n)| object_name(n) == animation_name)
            .ok_or_else(|| {
                format!(
                    "'{}': no animation named '{}' (file has {} clip{})",
                    path,
                    animation_name,
                    stacks.len(),
                    if stacks.len() == 1 { "" } else { "s" }
                )
            })?
    } else {
        let i = animation_index as usize;
        if i >= stacks.len() {
            return Err(format!(
                "'{}': animation_index {} out of range (file has {} animation{})",
                path,
                animation_index,
                stacks.len(),
                if stacks.len() == 1 { "" } else { "s" }
            ));
        }
        i
    };
    let (stack_id, stack_node) = stacks[stack_idx];

    // Connections: layer -> stack, curve node -> layer (both OO), curve ->
    // curve node (OP, axis property) and curve node -> model (OP, transform
    // property).
    let mut stack_of_layer: HashMap<i64, i64> = HashMap::new();
    let mut layer_of_curve_node: HashMap<i64, i64> = HashMap::new();
    let mut node_target: HashMap<i64, (i64, String)> = HashMap::new();
    let mut node_axis_curves: HashMap<i64, [Option<i64>; 3]> = HashMap::new();
    if let Some(conns) = root.first_child_by_name("Connections") {
        for c in conns.children_by_name("C") {
            let a = c.attributes();
            let ty = a.first().and_then(attr_str).unwrap_or("");
            let (Some(child), Some(parent)) =
                (a.get(1).and_then(attr_i64), a.get(2).and_then(attr_i64))
            else {
                continue;
            };
            match ty {
                "OO" => {
                    if layer_rank.contains_key(&child) {
                        stack_of_layer.insert(child, parent);
                    } else if curve_node_by_id.contains_key(&child) {
                        layer_of_curve_node.insert(child, parent);
                    }
                }
                "OP" => {
                    let prop = a.get(3).and_then(attr_str).unwrap_or("");
                    if curve_by_id.contains_key(&child) && curve_node_by_id.contains_key(&parent) {
                        let slot = match prop {
                            "d|X" => 0,
                            "d|Y" => 1,
                            "d|Z" => 2,
                            _ => continue,
                        };
                        node_axis_curves.entry(parent).or_default()[slot] = Some(child);
                    } else if curve_node_by_id.contains_key(&child)
                        && model_by_id.contains_key(&parent)
                        && matches!(prop, "Lcl Translation" | "Lcl Rotation" | "Lcl Scaling")
                    {
                        node_target.insert(child, (parent, prop.to_string()));
                    }
                }
                _ => {}
            }
        }
    }

    // Per-model channels for the selected stack: first curve node seen per
    // (model, property), walking the file's declaration order. Multiple layers
    // on one stack are not blended; the first layer's curves win.
    struct Channels {
        translation: Option<i64>,
        rotation: Option<i64>,
        scale: Option<i64>,
    }
    let mut in_stack: Vec<(usize, usize, i64)> = node_target
        .keys()
        .filter_map(|&node_id| {
            let layer = layer_of_curve_node.get(&node_id)?;
            if stack_of_layer.get(layer) != Some(&stack_id) {
                return None;
            }
            Some((
                *layer_rank.get(layer)?,
                *curve_node_rank.get(&node_id)?,
                node_id,
            ))
        })
        .collect();
    in_stack.sort_unstable();

    let mut per_model: BTreeMap<i64, Channels> = BTreeMap::new();
    for (_, _, node_id) in in_stack {
        let (model_id, prop) = &node_target[&node_id];
        let entry = per_model.entry(*model_id).or_insert(Channels {
            translation: None,
            rotation: None,
            scale: None,
        });
        let slot = match prop.as_str() {
            "Lcl Translation" => &mut entry.translation,
            "Lcl Rotation" => &mut entry.rotation,
            _ => &mut entry.scale,
        };
        if slot.is_none() {
            *slot = Some(node_id);
        }
    }

    // Clip window from the stack, falling back to the covered key range.
    let p70 = stack_node.first_child_by_name("Properties70");
    let start_kt = p70.as_ref().and_then(|p| prop_scalar(p, "LocalStart"));
    let stop_kt = p70.as_ref().and_then(|p| prop_scalar(p, "LocalStop"));
    let (start_kt, stop_kt) = match (start_kt, stop_kt) {
        (Some(a), Some(b)) if b > a => (a, b),
        _ => key_time_range(&node_axis_curves, &curve_by_id),
    };
    let duration = ((stop_kt - start_kt) / KTIME_PER_SEC).max(1e-3) as f32;

    // Bake each animated joint at the uniform rate.
    let rate = if sample_rate > 0.0 { sample_rate } else { 30.0 };
    let samples = ((duration * rate).ceil() as usize + 1).max(2);
    let mut tracks: Vec<ImportedAnimationTrack> = Vec::new();
    for (model_id, channels) in &per_model {
        let Some(&joint) = skin.model_to_joint.get(model_id) else {
            // Curves on a non-joint node (camera, prop): drop, like glTF.
            continue;
        };
        let model = model_by_id[model_id];
        warn_unsupported_transform_props(&model, path);

        let model_p70 = model.first_child_by_name("Properties70");
        let pre_rotation = model_p70
            .as_ref()
            .and_then(|p| prop_vec3(p, "PreRotation"))
            .unwrap_or([0.0; 3]);
        let rotation_order = model_p70
            .as_ref()
            .and_then(|p| prop_scalar(p, "RotationOrder"))
            .unwrap_or(0.0) as i32;

        let read = |id: Option<i64>| -> Option<AxisCurves<'_>> {
            id.map(|id| AxisCurves::read(id, &curve_node_by_id, &node_axis_curves, &curve_by_id))
        };
        let translation = read(channels.translation);
        let rotation = read(channels.rotation);
        let scale = read(channels.scale);

        let bind = &skin.joints[joint];
        let mut keys: Vec<ImportedKeyframe> = Vec::with_capacity(samples);
        // Constant for the whole track: the PreRotation matrix would otherwise be
        // rebuilt from three sin_cos pairs on every sampled keyframe.
        let pre_rotation_mat = rot_ordered_xyz([pre_rotation[0], pre_rotation[1], pre_rotation[2]]);
        for s in 0..samples {
            let time = (s as f32 / rate).min(duration);
            let kt = start_kt + (time as f64) * KTIME_PER_SEC;
            let mut pose = crate::gfx::skeleton::JointPose {
                translation: bind.translation,
                rotation_deg: bind.rotation_deg,
                scale: bind.scale,
            };
            if let Some(t) = &translation {
                pose.translation = t.eval3(kt);
            }
            if let Some(r) = &rotation {
                let euler = r.eval3(kt);
                let m = mat4_mul(
                    pre_rotation_mat,
                    rot_ordered(
                        [euler[0] as f64, euler[1] as f64, euler[2] as f64],
                        rotation_order,
                    ),
                );
                let (_, quat, _) = decompose(m);
                pose.rotation_deg = euler_yxz_from_quat(quat);
            }
            if let Some(sc) = &scale {
                pose.scale = sc.eval3(kt);
            }
            // The skeleton is normalized to meters by a uniform scale on the
            // root locals (see skin.rs); an animated root channel must fold
            // the same scale in or it would snap back to file units. Channels
            // left at bind are already normalized.
            if bind.parent < 0 {
                if translation.is_some() {
                    for c in &mut pose.translation {
                        *c *= skin.unit_scale;
                    }
                }
                if scale.is_some() {
                    for c in &mut pose.scale {
                        *c *= skin.unit_scale;
                    }
                }
            }
            keys.push(ImportedKeyframe { time, pose });
        }
        tracks.push(ImportedAnimationTrack { joint, keys });
    }
    tracks.sort_by_key(|t| t.joint);

    Ok(ImportedAnimation {
        name: object_name(&stack_node),
        duration,
        tracks,
        // FBX blend-shape channels are not imported; glTF is the morph path.
        morph_track: Vec::new(),
    })
}

// The three per-axis curves of one AnimationCurveNode plus its default
// values, ready for evaluation.
struct AxisCurves<'a> {
    axes: [Option<Curve<'a>>; 3],
    defaults: [f32; 3],
}

impl<'a> AxisCurves<'a> {
    fn read(
        node_id: i64,
        nodes: &HashMap<i64, NodeHandle<'a>>,
        axis_curves: &HashMap<i64, [Option<i64>; 3]>,
        curves: &HashMap<i64, NodeHandle<'a>>,
    ) -> Self {
        let p70 = nodes
            .get(&node_id)
            .and_then(|n| n.first_child_by_name("Properties70"));
        let default = |name: &str| -> f32 {
            p70.as_ref()
                .and_then(|p| prop_scalar(p, name))
                .unwrap_or(0.0) as f32
        };
        let defaults = [default("d|X"), default("d|Y"), default("d|Z")];
        let ids = axis_curves.get(&node_id).copied().unwrap_or_default();
        let axes = ids.map(|id| id.and_then(|id| curves.get(&id)).and_then(Curve::read));
        Self { axes, defaults }
    }

    fn eval3(&self, kt: f64) -> [f32; 3] {
        [
            self.axes[0]
                .as_ref()
                .map_or(self.defaults[0], |c| c.eval(kt)),
            self.axes[1]
                .as_ref()
                .map_or(self.defaults[1], |c| c.eval(kt)),
            self.axes[2]
                .as_ref()
                .map_or(self.defaults[2], |c| c.eval(kt)),
        ]
    }
}

// One AnimationCurve's key data.
struct Curve<'a> {
    times: &'a [i64],
    values: &'a [f32],
}

impl<'a> Curve<'a> {
    fn read(node: &NodeHandle<'a>) -> Option<Self> {
        let times = node
            .first_child_by_name("KeyTime")
            .as_ref()
            .and_then(arr_i64)?;
        let values = node
            .first_child_by_name("KeyValueFloat")
            .as_ref()
            .and_then(arr_f32)?;
        if times.is_empty() || times.len() != values.len() {
            return None;
        }
        Some(Self { times, values })
    }

    // Linear interpolation between the surrounding keys, clamped at the ends.
    fn eval(&self, kt: f64) -> f32 {
        let n = self.times.len();
        let after = self.times.partition_point(|&t| (t as f64) <= kt);
        if after == 0 {
            return self.values[0];
        }
        if after >= n {
            return self.values[n - 1];
        }
        let t0 = self.times[after - 1] as f64;
        let t1 = self.times[after] as f64;
        let v0 = self.values[after - 1];
        let v1 = self.values[after];
        if t1 <= t0 {
            return v1;
        }
        let f = ((kt - t0) / (t1 - t0)) as f32;
        v0 + (v1 - v0) * f
    }
}

// Widest key-time range across every referenced curve, used when a stack
// declares no LocalStart/LocalStop window.
fn key_time_range(
    node_axis_curves: &HashMap<i64, [Option<i64>; 3]>,
    curves: &HashMap<i64, NodeHandle>,
) -> (f64, f64) {
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for ids in node_axis_curves.values() {
        for id in ids.iter().flatten() {
            let Some(node) = curves.get(id) else { continue };
            let Some(times) = node
                .first_child_by_name("KeyTime")
                .as_ref()
                .and_then(arr_i64)
            else {
                continue;
            };
            if let (Some(&first), Some(&last)) = (times.first(), times.last()) {
                lo = lo.min(first as f64);
                hi = hi.max(last as f64);
            }
        }
    }
    if lo < hi { (lo, hi) } else { (0.0, 0.0) }
}

// Transform features outside the supported envelope: warn instead of
// silently mis-posing the joint.
fn warn_unsupported_transform_props(model: &NodeHandle, path: &str) {
    let Some(p70) = model.first_child_by_name("Properties70") else {
        return;
    };
    for name in [
        "PostRotation",
        "RotationPivot",
        "ScalingPivot",
        "RotationOffset",
        "ScalingOffset",
    ] {
        if let Some(v) = prop_vec3(&p70, name)
            && v.iter().any(|c| c.abs() > 1e-6)
        {
            tracing::warn!(
                "'{}': node '{}' uses {} ({:?}); this transform feature is not applied \
                 and the pose may be off",
                path,
                object_name(model),
                name,
                v
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::fbx::fixtures as fx;
    use crate::import::fbx::fixtures::assert_vec3_eq;

    #[test]
    fn curve_eval_interpolates_and_clamps() {
        let times: Vec<i64> = vec![0, 46_186_158_000, 92_372_316_000];
        let values: Vec<f32> = vec![0.0, 10.0, 20.0];
        let c = Curve {
            times: &times,
            values: &values,
        };
        assert_eq!(c.eval(-5.0), 0.0, "clamps before the first key");
        assert_eq!(c.eval(0.0), 0.0);
        let half = 0.5 * KTIME_PER_SEC;
        assert!((c.eval(half) - 5.0).abs() < 1e-4, "midpoint lerps");
        assert_eq!(c.eval(3.0 * KTIME_PER_SEC), 20.0, "clamps after the last");
    }

    #[test]
    fn curve_eval_handles_a_single_key() {
        let times: Vec<i64> = vec![100];
        let values: Vec<f32> = vec![7.0];
        let c = Curve {
            times: &times,
            values: &values,
        };
        assert_eq!(c.eval(0.0), 7.0);
        assert_eq!(c.eval(1e12), 7.0);
    }

    #[test]
    fn ktime_constant_matches_one_second() {
        // The FBX SDK defines 46,186,158,000 KTime units per second; a drift
        // here would silently stretch every imported clip.
        assert_eq!(KTIME_PER_SEC, 46_186_158_000.0);
    }

    const ONE_SECOND: i64 = KTIME_PER_SEC as i64;
    const STACK: i64 = 500;
    const LAYER: i64 = 510;
    const TRANSLATION_NODE: i64 = 520;
    const TRANSLATION_CURVE: i64 = 530;
    const ROTATION_NODE: i64 = 540;
    const SCALE_NODE: i64 = 560;
    const SECOND_STACK: i64 = 501;
    const SECOND_LAYER: i64 = 511;
    const SECOND_NODE: i64 = 521;

    // The two-bone rig plus a one-second clip whose only curve sweeps the root
    // joint's X translation from 0 to 10. Y and Z keep the curve node defaults.
    fn animated_rig(unit_scale_factor: f64) -> fx::Doc {
        let mut doc = fx::two_bone_rig(unit_scale_factor);
        doc.objects.extend([
            fx::anim_stack(STACK, "Take 001").child(fx::properties70(vec![
                fx::p_time("LocalStart", 0),
                fx::p_time("LocalStop", ONE_SECOND),
            ])),
            fx::anim_layer(LAYER, "BaseLayer"),
            fx::anim_curve_node(TRANSLATION_NODE, "T", [1.0, 2.0, 3.0]),
            fx::anim_curve(TRANSLATION_CURVE, vec![0, ONE_SECOND], vec![0.0, 10.0]),
        ]);
        doc.connections.extend([
            fx::oo(LAYER, STACK),
            fx::oo(TRANSLATION_NODE, LAYER),
            fx::op(TRANSLATION_CURVE, TRANSLATION_NODE, "d|X"),
            fx::op(TRANSLATION_NODE, fx::ROOT_BONE_ID, "Lcl Translation"),
        ]);
        doc
    }

    // A second clip on its own layer, translating the Tip joint instead.
    fn add_second_clip(doc: &mut fx::Doc) {
        doc.objects.extend([
            fx::anim_stack(SECOND_STACK, "Second").child(fx::properties70(vec![
                fx::p_time("LocalStart", 0),
                fx::p_time("LocalStop", ONE_SECOND),
            ])),
            fx::anim_layer(SECOND_LAYER, "SecondLayer"),
            fx::anim_curve_node(SECOND_NODE, "T", [100.0, 0.0, 0.0]),
        ]);
        doc.connections.extend([
            fx::oo(SECOND_LAYER, SECOND_STACK),
            fx::oo(SECOND_NODE, SECOND_LAYER),
            fx::op(SECOND_NODE, fx::TIP_BONE_ID, "Lcl Translation"),
        ]);
    }

    // The rotation of a baked pose, read as the image of +X relative to the
    // pose origin so the joint's bind translation drops out.
    fn rotated_x_axis(key: &ImportedKeyframe) -> [f32; 3] {
        let m = key.pose.to_matrix();
        let origin = crate::import::fbx::transform_point(m, [0.0, 0.0, 0.0]);
        let tip = crate::import::fbx::transform_point(m, [1.0, 0.0, 0.0]);
        [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]]
    }

    #[test]
    fn fbx_animation_names_lists_stacks_in_declaration_order() {
        let mut doc = animated_rig(100.0);
        add_second_clip(&mut doc);
        let file = doc.write();
        assert_eq!(
            fbx_animation_names(file.path()).expect("names"),
            vec!["Take 001".to_string(), "Second".to_string()]
        );
    }

    #[test]
    fn fbx_animation_names_reports_a_file_without_an_objects_section() {
        let file = fx::write(vec![fx::node("Definitions")]);
        let err = fbx_animation_names(file.path()).expect_err("no Objects section");
        assert!(err.contains("FBX has no Objects section"), "got: {err}");
    }

    #[test]
    fn import_fbx_animation_bakes_a_translation_channel_at_the_sample_rate() {
        let file = animated_rig(100.0).write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");

        assert_eq!(anim.name, "Take 001");
        assert!((anim.duration - 1.0).abs() < 1e-4, "{}", anim.duration);
        assert!(anim.morph_track.is_empty());
        assert_eq!(anim.tracks.len(), 1);
        assert_eq!(anim.tracks[0].joint, 0);

        let keys = &anim.tracks[0].keys;
        assert_eq!(keys.len(), 31);
        assert!((keys[0].time - 0.0).abs() < 1e-6);
        // The keyed axis follows the curve; the others hold the node defaults.
        assert_vec3_eq(keys[0].pose.translation, [0.0, 2.0, 3.0]);
        assert!((keys[15].time - 0.5).abs() < 1e-4);
        assert_vec3_eq(keys[15].pose.translation, [5.0, 2.0, 3.0]);
        assert!((keys[30].time - 1.0).abs() < 1e-4);
        assert_vec3_eq(keys[30].pose.translation, [10.0, 2.0, 3.0]);
        // Unanimated channels stay at the bind pose.
        assert_vec3_eq(keys[0].pose.rotation_deg, [0.0, 0.0, 0.0]);
        assert_vec3_eq(keys[0].pose.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn import_fbx_animation_falls_back_to_thirty_hertz() {
        let file = animated_rig(100.0).write();
        let slow = import_fbx_animation(file.path(), 0, "", 0.0, 0).expect("animation");
        assert_eq!(slow.tracks[0].keys.len(), 31);
        let dense = import_fbx_animation(file.path(), 0, "", 60.0, 0).expect("animation");
        assert_eq!(dense.tracks[0].keys.len(), 61);
    }

    #[test]
    fn import_fbx_animation_selects_a_clip_by_name_and_by_index() {
        let mut doc = animated_rig(100.0);
        add_second_clip(&mut doc);
        let file = doc.write();

        let by_name = import_fbx_animation(file.path(), 0, "Second", 30.0, 0).expect("by name");
        assert_eq!(by_name.name, "Second");
        // Only the selected stack's channels are baked.
        assert_eq!(by_name.tracks.len(), 1);
        assert_eq!(by_name.tracks[0].joint, 1);
        assert_vec3_eq(
            by_name.tracks[0].keys[0].pose.translation,
            [100.0, 0.0, 0.0],
        );

        let by_index = import_fbx_animation(file.path(), 1, "", 30.0, 0).expect("by index");
        assert_eq!(by_index.name, "Second");

        let first = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("first");
        assert_eq!(first.name, "Take 001");
        assert_eq!(first.tracks.len(), 1);
        assert_eq!(first.tracks[0].joint, 0);
    }

    #[test]
    fn import_fbx_animation_emits_one_track_per_animated_joint_in_joint_order() {
        let mut doc = animated_rig(100.0);
        // A second channel on the same stack, targeting the deeper joint.
        doc.objects
            .push(fx::anim_curve_node(SECOND_NODE, "T", [4.0, 0.0, 0.0]));
        doc.connections.extend([
            fx::oo(SECOND_NODE, LAYER),
            fx::op(SECOND_NODE, fx::TIP_BONE_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");

        assert_eq!(anim.tracks.len(), 2);
        assert_eq!(anim.tracks[0].joint, 0);
        assert_eq!(anim.tracks[1].joint, 1);
        assert_vec3_eq(anim.tracks[1].keys[0].pose.translation, [4.0, 0.0, 0.0]);
    }

    #[test]
    fn import_fbx_animation_drops_curves_targeting_non_joint_nodes() {
        let mut doc = animated_rig(100.0);
        doc.objects
            .push(fx::anim_curve_node(SECOND_NODE, "T", [7.0, 0.0, 0.0]));
        doc.connections.extend([
            fx::oo(SECOND_NODE, LAYER),
            // The mesh node is not part of the skeleton.
            fx::op(SECOND_NODE, fx::MESH_MODEL_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        assert_eq!(anim.tracks.len(), 1);
        assert_eq!(anim.tracks[0].joint, 0);
    }

    #[test]
    fn import_fbx_animation_ignores_unknown_curve_and_target_properties() {
        let mut doc = animated_rig(100.0);
        doc.objects
            .push(fx::anim_curve_node(SECOND_NODE, "V", [9.0, 9.0, 9.0]));
        doc.connections.extend([
            // A fourth axis has no slot.
            fx::op(TRANSLATION_CURVE, TRANSLATION_NODE, "d|W"),
            fx::oo(SECOND_NODE, LAYER),
            // Only the three transform properties are animatable.
            fx::op(SECOND_NODE, fx::ROOT_BONE_ID, "Visibility"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        assert_eq!(anim.tracks.len(), 1);
        assert_vec3_eq(anim.tracks[0].keys[30].pose.translation, [10.0, 2.0, 3.0]);
    }

    #[test]
    fn import_fbx_animation_folds_the_unit_scale_into_animated_root_channels() {
        let file = animated_rig(1.0).write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        let keys = &anim.tracks[0].keys;
        // File units are centimeters, so the root's evaluated translation is
        // re-expressed in meters alongside the bind pose.
        assert_vec3_eq(keys[30].pose.translation, [0.1, 0.02, 0.03]);
        // No scale channel: the bind scale already carries the compensation.
        assert_vec3_eq(keys[30].pose.scale, [0.01, 0.01, 0.01]);
    }

    #[test]
    fn import_fbx_animation_scales_an_animated_root_scale_channel() {
        // The root carries a scale channel and no translation channel.
        let mut doc = fx::two_bone_rig(1.0);
        doc.objects.extend([
            fx::anim_stack(STACK, "Take 001").child(fx::properties70(vec![
                fx::p_time("LocalStart", 0),
                fx::p_time("LocalStop", ONE_SECOND),
            ])),
            fx::anim_layer(LAYER, "BaseLayer"),
            fx::anim_curve_node(SCALE_NODE, "S", [2.0, 2.0, 2.0]),
        ]);
        doc.connections.extend([
            fx::oo(LAYER, STACK),
            fx::oo(SCALE_NODE, LAYER),
            fx::op(SCALE_NODE, fx::ROOT_BONE_ID, "Lcl Scaling"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        let key = &anim.tracks[0].keys[0];
        assert_vec3_eq(key.pose.scale, [0.02, 0.02, 0.02]);
        // The unanimated translation is already normalized by the bind pose.
        assert_vec3_eq(key.pose.translation, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn import_fbx_animation_ignores_malformed_and_unrelated_connections() {
        let mut doc = animated_rig(100.0);
        // An object node without an id is skipped by the object scan.
        doc.objects.push(fx::node("AnimationCurve"));
        doc.connections.extend([
            fx::node("C").text("OO"),
            fx::node("C")
                .text("PP")
                .int64(TRANSLATION_CURVE)
                .int64(TRANSLATION_NODE),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        assert_eq!(anim.tracks.len(), 1);
        assert_vec3_eq(anim.tracks[0].keys[30].pose.translation, [10.0, 2.0, 3.0]);
    }

    #[test]
    fn import_fbx_animation_skips_key_less_curves_when_deriving_the_window() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.objects.extend([
            fx::anim_stack(STACK, "Take 001"),
            fx::anim_layer(LAYER, "BaseLayer"),
            fx::anim_curve_node(TRANSLATION_NODE, "T", [0.0, 0.0, 0.0]),
            fx::anim_curve(TRANSLATION_CURVE, vec![0, ONE_SECOND], vec![0.0, 10.0]),
            // Values without key times contribute nothing to the range.
            fx::object("AnimationCurve", 531, "", "")
                .child(fx::node("KeyValueFloat").arr_f32(vec![1.0, 2.0])),
            // Neither does a declared but empty key list.
            fx::anim_curve(532, Vec::new(), Vec::new()),
        ]);
        doc.connections.extend([
            fx::oo(LAYER, STACK),
            fx::oo(TRANSLATION_NODE, LAYER),
            fx::op(TRANSLATION_CURVE, TRANSLATION_NODE, "d|X"),
            fx::op(531, TRANSLATION_NODE, "d|Y"),
            fx::op(532, TRANSLATION_NODE, "d|Z"),
            fx::op(TRANSLATION_NODE, fx::ROOT_BONE_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        assert!((anim.duration - 1.0).abs() < 1e-4, "{}", anim.duration);
        assert_vec3_eq(anim.tracks[0].keys[30].pose.translation, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn import_fbx_animation_leaves_child_joint_channels_in_file_units() {
        let mut doc = animated_rig(1.0);
        doc.objects
            .push(fx::anim_curve_node(SECOND_NODE, "T", [4.0, 0.0, 0.0]));
        doc.connections.extend([
            fx::oo(SECOND_NODE, LAYER),
            fx::op(SECOND_NODE, fx::TIP_BONE_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        let tip = anim
            .tracks
            .iter()
            .find(|t| t.joint == 1)
            .expect("tip track");
        assert_vec3_eq(tip.keys[0].pose.translation, [4.0, 0.0, 0.0]);
    }

    #[test]
    fn import_fbx_animation_applies_the_node_prerotation() {
        let mut doc = animated_rig(100.0);
        doc.attach(
            fx::TIP_BONE_ID,
            fx::properties70(vec![fx::p_vec3("PreRotation", [0.0, 0.0, 90.0])]),
        );
        doc.objects
            .push(fx::anim_curve_node(ROTATION_NODE, "R", [0.0, 0.0, 0.0]));
        doc.connections.extend([
            fx::oo(ROTATION_NODE, LAYER),
            fx::op(ROTATION_NODE, fx::TIP_BONE_ID, "Lcl Rotation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        let tip = anim
            .tracks
            .iter()
            .find(|t| t.joint == 1)
            .expect("tip track");
        // With a zero rotation channel the pose is the PreRotation alone.
        assert_vec3_eq(rotated_x_axis(&tip.keys[0]), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn import_fbx_animation_honours_the_node_rotation_order() {
        let mut doc = animated_rig(100.0);
        doc.attach(
            fx::TIP_BONE_ID,
            fx::properties70(vec![fx::p_scalar("RotationOrder", 5.0)]),
        );
        doc.objects
            .push(fx::anim_curve_node(ROTATION_NODE, "R", [90.0, 0.0, 90.0]));
        doc.connections.extend([
            fx::oo(ROTATION_NODE, LAYER),
            fx::op(ROTATION_NODE, fx::TIP_BONE_ID, "Lcl Rotation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        let tip = anim
            .tracks
            .iter()
            .find(|t| t.joint == 1)
            .expect("tip track");
        // ZYX applies Z first: +X to +Y, then Rx90 to +Z. XYZ would stop at +Y.
        assert_vec3_eq(rotated_x_axis(&tip.keys[0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn import_fbx_animation_warns_about_unsupported_transform_properties() {
        let mut doc = animated_rig(100.0);
        doc.attach(
            fx::ROOT_BONE_ID,
            fx::properties70(vec![
                fx::p_vec3("PostRotation", [0.0, 0.0, 45.0]),
                fx::p_vec3("RotationPivot", [1.0, 0.0, 0.0]),
                // A zero-valued pivot is within the supported envelope.
                fx::p_vec3("ScalingPivot", [0.0, 0.0, 0.0]),
            ]),
        );
        let file = doc.write();
        let (anim, warnings) =
            fx::count_warnings(|| import_fbx_animation(file.path(), 0, "", 30.0, 0));
        let anim = anim.expect("animation");

        assert_eq!(warnings, 2, "only the two nonzero properties warn");
        // The unsupported features are reported, not applied.
        assert_vec3_eq(anim.tracks[0].keys[30].pose.translation, [10.0, 2.0, 3.0]);
    }

    #[test]
    fn import_fbx_animation_is_silent_for_supported_transforms() {
        let file = animated_rig(100.0).write();
        let (anim, warnings) =
            fx::count_warnings(|| import_fbx_animation(file.path(), 0, "", 30.0, 0));
        anim.expect("animation");
        assert_eq!(warnings, 0);
    }

    #[test]
    fn import_fbx_animation_evaluates_all_three_axis_curves() {
        let mut doc = animated_rig(100.0);
        doc.objects.extend([
            fx::anim_curve(531, vec![0, ONE_SECOND], vec![0.0, 20.0]),
            fx::anim_curve(532, vec![0, ONE_SECOND], vec![0.0, 30.0]),
        ]);
        doc.connections.extend([
            fx::op(531, TRANSLATION_NODE, "d|Y"),
            fx::op(532, TRANSLATION_NODE, "d|Z"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");

        let keys = &anim.tracks[0].keys;
        assert_vec3_eq(keys[0].pose.translation, [0.0, 0.0, 0.0]);
        assert_vec3_eq(keys[15].pose.translation, [5.0, 10.0, 15.0]);
        assert_vec3_eq(keys[30].pose.translation, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn import_fbx_animation_does_not_blend_multiple_layers() {
        let mut doc = animated_rig(100.0);
        // A second layer on the same stack, keyed differently on the same
        // channel of the same joint.
        doc.objects.extend([
            fx::anim_layer(SECOND_LAYER, "OverrideLayer"),
            fx::anim_curve_node(SECOND_NODE, "T", [7.0, 8.0, 9.0]),
            fx::anim_curve(531, vec![0, ONE_SECOND], vec![0.0, 100.0]),
        ]);
        doc.connections.extend([
            fx::oo(SECOND_LAYER, STACK),
            fx::oo(SECOND_NODE, SECOND_LAYER),
            fx::op(531, SECOND_NODE, "d|X"),
            fx::op(SECOND_NODE, fx::ROOT_BONE_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");

        // The first layer wins the channel outright: neither the override's
        // curve nor its defaults reach the baked pose, and nothing is summed.
        assert_eq!(anim.tracks.len(), 1);
        assert_vec3_eq(anim.tracks[0].keys[30].pose.translation, [10.0, 2.0, 3.0]);
    }

    #[test]
    fn import_fbx_animation_resolves_layer_order_by_declaration_not_object_id() {
        // The base layer is declared first but carries the higher object id, as
        // do its curve node and curve, so an id-ordered walk picks the override.
        let mut doc = fx::two_bone_rig(100.0);
        doc.objects.extend([
            fx::anim_stack(STACK, "Take 001").child(fx::properties70(vec![
                fx::p_time("LocalStart", 0),
                fx::p_time("LocalStop", ONE_SECOND),
            ])),
            fx::anim_layer(SECOND_LAYER, "BaseLayer"),
            fx::anim_curve_node(SECOND_NODE, "T", [1.0, 2.0, 3.0]),
            fx::anim_curve(531, vec![0, ONE_SECOND], vec![0.0, 10.0]),
            fx::anim_layer(LAYER, "OverrideLayer"),
            fx::anim_curve_node(TRANSLATION_NODE, "T", [7.0, 8.0, 9.0]),
            fx::anim_curve(TRANSLATION_CURVE, vec![0, ONE_SECOND], vec![0.0, 100.0]),
        ]);
        doc.connections.extend([
            fx::oo(SECOND_LAYER, STACK),
            fx::oo(SECOND_NODE, SECOND_LAYER),
            fx::op(531, SECOND_NODE, "d|X"),
            fx::op(SECOND_NODE, fx::ROOT_BONE_ID, "Lcl Translation"),
            fx::oo(LAYER, STACK),
            fx::oo(TRANSLATION_NODE, LAYER),
            fx::op(TRANSLATION_CURVE, TRANSLATION_NODE, "d|X"),
            fx::op(TRANSLATION_NODE, fx::ROOT_BONE_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");

        assert_eq!(anim.tracks.len(), 1);
        assert_vec3_eq(anim.tracks[0].keys[30].pose.translation, [10.0, 2.0, 3.0]);
    }

    #[test]
    fn animation_imports_report_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing.fbx");
        let path = path.to_str().expect("path");
        let err = fbx_animation_names(path).expect_err("names");
        assert!(err.contains("could not open"), "got: {err}");
        let err = import_fbx_animation(path, 0, "", 30.0, 0).expect_err("clip");
        assert!(err.contains("could not open"), "got: {err}");
    }

    #[test]
    fn import_fbx_animation_requires_a_skinned_mesh() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(100, "mesh", vec![0.0; 9], vec![0, 1, -3]),
            fx::model(200, "Mesh", "Mesh"),
            fx::anim_stack(STACK, "Take 001"),
        ];
        doc.connections = vec![fx::oo(100, 200)];
        let file = doc.write();
        let err = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect_err("no skin");
        assert!(err.contains("no skin deformer"), "got: {err}");
    }

    #[test]
    fn import_fbx_animation_derives_the_window_from_the_key_range() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.objects.extend([
            // An inverted LocalStart/LocalStop window is not usable.
            fx::anim_stack(STACK, "Take 001").child(fx::properties70(vec![
                fx::p_time("LocalStart", ONE_SECOND),
                fx::p_time("LocalStop", 0),
            ])),
            fx::anim_layer(LAYER, "BaseLayer"),
            fx::anim_curve_node(TRANSLATION_NODE, "T", [0.0, 0.0, 0.0]),
            fx::anim_curve(
                TRANSLATION_CURVE,
                vec![ONE_SECOND / 2, 3 * ONE_SECOND / 2],
                vec![4.0, 8.0],
            ),
        ]);
        doc.connections.extend([
            fx::oo(LAYER, STACK),
            fx::oo(TRANSLATION_NODE, LAYER),
            fx::op(TRANSLATION_CURVE, TRANSLATION_NODE, "d|X"),
            fx::op(TRANSLATION_NODE, fx::ROOT_BONE_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");

        assert!((anim.duration - 1.0).abs() < 1e-4, "{}", anim.duration);
        let keys = &anim.tracks[0].keys;
        // Sampling starts at the first key, not at time zero.
        assert_vec3_eq(keys[0].pose.translation, [4.0, 0.0, 0.0]);
        assert_vec3_eq(keys[30].pose.translation, [8.0, 0.0, 0.0]);
    }

    #[test]
    fn import_fbx_animation_gives_a_curveless_stack_a_minimal_window() {
        let mut doc = fx::two_bone_rig(100.0);
        doc.objects.extend([
            fx::anim_stack(STACK, "Empty"),
            fx::anim_layer(LAYER, "BaseLayer"),
            fx::anim_curve_node(TRANSLATION_NODE, "T", [1.0, 2.0, 3.0]),
        ]);
        doc.connections.extend([
            fx::oo(LAYER, STACK),
            fx::oo(TRANSLATION_NODE, LAYER),
            fx::op(TRANSLATION_NODE, fx::ROOT_BONE_ID, "Lcl Translation"),
        ]);
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");

        assert!((anim.duration - 1e-3).abs() < 1e-9, "{}", anim.duration);
        // Two keys is the floor, both holding the curve node defaults.
        assert_eq!(anim.tracks[0].keys.len(), 2);
        assert_vec3_eq(anim.tracks[0].keys[0].pose.translation, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn import_fbx_animation_falls_back_to_defaults_for_malformed_curves() {
        let malformed = [
            // Fewer values than key times.
            fx::anim_curve(TRANSLATION_CURVE, vec![0, ONE_SECOND], vec![5.0]),
            // No keys at all.
            fx::anim_curve(TRANSLATION_CURVE, Vec::new(), Vec::new()),
            // InputKey times without values.
            fx::object("AnimationCurve", TRANSLATION_CURVE, "", "")
                .child(fx::node("KeyTime").arr_i64(vec![0, ONE_SECOND])),
        ];
        for curve in malformed {
            let mut doc = animated_rig(100.0);
            doc.replace_object(TRANSLATION_CURVE, curve);
            let file = doc.write();
            let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
            // X reverts to the curve node's `d|X` default instead of the curve.
            assert_vec3_eq(anim.tracks[0].keys[0].pose.translation, [1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn import_fbx_animation_treats_a_propertyless_curve_node_as_zero() {
        let mut doc = animated_rig(100.0);
        doc.replace_object(
            TRANSLATION_NODE,
            fx::object("AnimationCurveNode", TRANSLATION_NODE, "T", ""),
        );
        let file = doc.write();
        let anim = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect("animation");
        // The curve still drives X; the axes without one default to zero.
        assert_vec3_eq(anim.tracks[0].keys[30].pose.translation, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn import_fbx_animation_reports_a_file_without_animation_stacks() {
        let file = fx::two_bone_rig(100.0).write();
        let err = import_fbx_animation(file.path(), 0, "", 30.0, 0).expect_err("no stacks");
        assert!(err.contains("FBX has no animation stacks"), "got: {err}");
    }

    #[test]
    fn import_fbx_animation_reports_an_unknown_clip_name() {
        let file = animated_rig(100.0).write();
        let err = import_fbx_animation(file.path(), 0, "Nope", 30.0, 0).expect_err("unknown name");
        assert!(
            err.contains("no animation named 'Nope' (file has 1 clip)"),
            "got: {err}"
        );

        let mut doc = animated_rig(100.0);
        add_second_clip(&mut doc);
        let file = doc.write();
        let err = import_fbx_animation(file.path(), 0, "Nope", 30.0, 0).expect_err("unknown name");
        assert!(err.contains("(file has 2 clips)"), "got: {err}");
    }

    #[test]
    fn import_fbx_animation_reports_an_out_of_range_index() {
        let file = animated_rig(100.0).write();
        let err = import_fbx_animation(file.path(), 5, "", 30.0, 0).expect_err("out of range");
        assert!(
            err.contains("animation_index 5 out of range (file has 1 animation)"),
            "got: {err}"
        );

        let mut doc = animated_rig(100.0);
        add_second_clip(&mut doc);
        let file = doc.write();
        let err = import_fbx_animation(file.path(), 5, "", 30.0, 0).expect_err("out of range");
        assert!(err.contains("(file has 2 animations)"), "got: {err}");
    }
}
