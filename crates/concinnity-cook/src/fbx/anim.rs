// src/fbx/anim.rs
//
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

use std::collections::HashMap;

use fbxcel::tree::v7400::NodeHandle;

use super::{
    arr_f32, arr_i64, attr_i64, attr_str, object_id, object_name, prop_scalar, prop_vec3,
    rot_ordered, rot_xyz,
};
use crate::gfx::skinning::{decompose, euler_yxz_from_quat, mat4_mul};
use crate::glb::{ImportedAnimation, ImportedAnimationTrack, ImportedKeyframe};

// FBX time unit: one second is 46,186,158,000 KTime ticks.
const KTIME_PER_SEC: f64 = 46_186_158_000.0;

// Names of every animation stack (clip) in declaration order.
pub fn fbx_animation_names(path: &str) -> Result<Vec<String>, String> {
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

// Import one animation clip, selected by `animation_name` (precedence) or
// `animation_index`, baked at `sample_rate` keys per second. Curves targeting
// nodes that are not skeleton joints are dropped, mirroring the glTF
// importer.
pub fn import_fbx_animation(
    path: &str,
    animation_index: u32,
    animation_name: &str,
    sample_rate: f32,
) -> Result<ImportedAnimation, String> {
    let tree = super::load_tree(path)?;
    let root = tree.root();
    let skin = super::skin::parse_skin(root, path)?;
    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| format!("'{path}': FBX has no Objects section"))?;

    // Object indexes.
    let mut stacks: Vec<(i64, NodeHandle)> = Vec::new();
    let mut layer_ids: Vec<i64> = Vec::new();
    let mut curve_node_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut curve_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut model_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    for c in objects.children() {
        let Some(id) = object_id(&c) else { continue };
        match c.name() {
            "AnimationStack" => stacks.push((id, c)),
            "AnimationLayer" => layer_ids.push(id),
            "AnimationCurveNode" => {
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
                    if layer_ids.contains(&child) {
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
    // (model, property). Multiple layers on one stack are not blended; the
    // first layer's curves win.
    struct Channels {
        translation: Option<i64>,
        rotation: Option<i64>,
        scale: Option<i64>,
    }
    let mut per_model: HashMap<i64, Channels> = HashMap::new();
    let mut model_order: Vec<i64> = Vec::new();
    for (&node_id, (model_id, prop)) in &node_target {
        let in_stack = layer_of_curve_node
            .get(&node_id)
            .and_then(|l| stack_of_layer.get(l))
            == Some(&stack_id);
        if !in_stack {
            continue;
        }
        let entry = per_model.entry(*model_id).or_insert_with(|| {
            model_order.push(*model_id);
            Channels {
                translation: None,
                rotation: None,
                scale: None,
            }
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
    model_order.sort();

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
    for model_id in model_order {
        let Some(&joint) = skin.model_to_joint.get(&model_id) else {
            // Curves on a non-joint node (camera, prop): drop, like glTF.
            continue;
        };
        let channels = &per_model[&model_id];
        let model = model_by_id[&model_id];
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
        for s in 0..samples {
            let time = (s as f32 / rate).min(duration);
            let kt = start_kt + (time as f64) * KTIME_PER_SEC;
            let mut pose = crate::gfx::skinning::JointPose {
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
                    rot_xyz([pre_rotation[0], pre_rotation[1], pre_rotation[2]]),
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
}
