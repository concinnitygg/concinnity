// Character-shape schema: slider-driven morph weights and per-joint proportions
// that deform a SkinnedMesh at runtime.

use crate::ecs::SkinnedMeshHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::de_opt_skinned_mesh_handle;
use alloc::string::String;
use alloc::vec::Vec;

/// One named shape value in `[-1, 1]`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ShapeSlider {
    /// Slider name; matched against the target mesh's morph-target names.
    pub name: String,
    /// Slider value, clamped to `[-1, 1]`.
    pub value: f32,
}

/// One joint's proportion change.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct JointProportion {
    /// Name of the joint in the target mesh's `skeleton`.
    pub joint: String,
    /// Uniform scale applied to the joint (and, through the hierarchy,
    /// everything below it). `1` leaves it alone.
    pub scale: f32,
    /// Extra length along the bone, in model units: every child joint is
    /// pushed that far along its bind direction from this joint. `0` leaves
    /// it alone.
    pub length: f32,
}

impl Default for JointProportion {
    fn default() -> Self {
        Self {
            joint: String::new(),
            scale: 1.0,
            length: 0.0,
        }
    }
}

/// Shape sliders and joint proportions applied to one [SkinnedMesh](#skinnedmesh).
///
/// Every characteristic of the shape is data on the mesh, not code: a slider
/// drives one or two of the mesh's morph targets, and a proportion scales or
/// lengthens one joint of its skeleton. The deformation is static and sits
/// under any [Animation](#animation) playing on the same mesh: clip morph
/// tracks are added on top of the slider weights, and clip poses are
/// re-proportioned every frame.
///
/// **Sliders** resolve to morph targets by name. A target named exactly
/// `name` is unipolar and receives the slider value clamped to `[0, 1]`. A
/// pair named `name+` / `name-` is bipolar: a positive value drives `name+`,
/// a negative value drives `name-` by its magnitude. A slider with no matching
/// target is reported as a build warning and ignored.
///
/// **Proportions** resolve to joints by name. `scale` is uniform (the
/// skinning shaders transform normals with the plain joint matrix, so a
/// non-uniform scale would shade incorrectly) and propagates to the joint's
/// descendants; `length` moves only the joint's children along the bone, so a
/// longer thigh does not also stretch the shin. Proportions change the posed
/// skeleton, not the bind pose, so clips with translation tracks on the
/// affected joints fight them; keep such rigs rotation-only. When the mesh
/// declares a `capsule`, the capsule's half-height follows the skeleton's
/// height change and its radius follows the root joint's scale.
///
/// `target` may name a [CharacterModel](#charactermodel) as well as a
/// `SkinnedMesh`; the model's emitted mesh is what the shape deforms.
///
/// **Baking.** With `bake` set, the build flattens the shape into its target:
/// the sliders' deformation is applied to the vertices and the morph targets
/// dropped, the bind pose is rewritten through the proportions, the capsule
/// is resized, and this asset is consumed. The result is a plain `SkinnedMesh`
/// with no per-frame shape work, for characters that never change shape.
///
/// ```rust
/// # use concinnity_core::components::{CharacterShape, JointProportion, ShapeSlider};
/// CharacterShape {
///     sliders: vec![ShapeSlider { name: "weight".into(), value: 0.5 }],
///     proportions: vec![JointProportion { joint: "spine".into(), scale: 1.1, length: 0.0 }],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CharacterShape {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [SkinnedMesh](#skinnedmesh) this shape deforms.
    #[serde(deserialize_with = "de_opt_skinned_mesh_handle")]
    pub target: Option<SkinnedMeshHandle>,
    /// Named shape values, each resolved to the mesh's morph targets.
    pub sliders: Vec<ShapeSlider>,
    /// Per-joint scale and length changes.
    pub proportions: Vec<JointProportion>,
    /// Flatten the shape into the target mesh at build time and drop this
    /// asset, instead of deforming at runtime.
    pub bake: bool,
}

/// Morph weights resolved from a shape's sliders against a mesh's morph-target
/// names, plus the slider names that matched nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedSliders {
    /// One weight per morph target, in target order.
    pub weights: Vec<f32>,
    /// Slider names with neither a unipolar nor a bipolar target.
    pub unresolved: Vec<String>,
}

impl CharacterShape {
    /// Resolve `sliders` against `target_names` (the mesh's morph targets in
    /// target order). Several sliders naming the same target accumulate; the
    /// result is clamped to `[0, 1]` per target.
    pub fn resolve_sliders(&self, target_names: &[String]) -> ResolvedSliders {
        let mut out = ResolvedSliders {
            weights: alloc::vec![0.0; target_names.len()],
            unresolved: Vec::new(),
        };
        let find = |name: &str, suffix: &str| {
            target_names
                .iter()
                .position(|t| t.strip_suffix(suffix).is_some_and(|base| base == name))
        };
        for slider in &self.sliders {
            let value = slider.value.clamp(-1.0, 1.0);
            let plus = find(&slider.name, "+");
            let minus = find(&slider.name, "-");
            if plus.is_some() || minus.is_some() {
                if let Some(i) = plus {
                    out.weights[i] += value.max(0.0);
                }
                if let Some(i) = minus {
                    out.weights[i] += (-value).max(0.0);
                }
            } else if let Some(i) = find(&slider.name, "") {
                out.weights[i] += value.max(0.0);
            } else {
                out.unresolved.push(slider.name.clone());
            }
        }
        for w in &mut out.weights {
            *w = w.clamp(0.0, 1.0);
        }
        out
    }

    /// The proportion joint names that `has_joint` does not know.
    pub fn unresolved_joints(&self, has_joint: impl Fn(&str) -> bool) -> Vec<String> {
        self.proportions
            .iter()
            .filter(|p| !has_joint(&p.joint))
            .map(|p| p.joint.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| String::from(*s)).collect()
    }

    fn shape(sliders: &[(&str, f32)]) -> CharacterShape {
        CharacterShape {
            sliders: sliders
                .iter()
                .map(|(n, v)| ShapeSlider {
                    name: String::from(*n),
                    value: *v,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn unipolar_slider_drives_the_target_of_the_same_name() {
        let r = shape(&[("weight", 0.6)]).resolve_sliders(&names(&["height", "weight"]));
        assert_eq!(r.weights, [0.0, 0.6]);
        assert!(r.unresolved.is_empty());
        // A negative value on a unipolar target contributes nothing.
        let r = shape(&[("weight", -0.6)]).resolve_sliders(&names(&["weight"]));
        assert_eq!(r.weights, [0.0]);
    }

    #[test]
    fn bipolar_slider_splits_by_sign() {
        let targets = names(&["jaw-", "jaw+"]);
        let r = shape(&[("jaw", 0.25)]).resolve_sliders(&targets);
        assert_eq!(r.weights, [0.0, 0.25]);
        let r = shape(&[("jaw", -0.75)]).resolve_sliders(&targets);
        assert_eq!(r.weights, [0.75, 0.0]);
        // The pair takes precedence over a same-named unipolar target.
        let r = shape(&[("jaw", 0.5)]).resolve_sliders(&names(&["jaw", "jaw+"]));
        assert_eq!(r.weights, [0.0, 0.5]);
    }

    #[test]
    fn unresolved_sliders_are_reported_not_fatal() {
        let r = shape(&[("nose", 1.0), ("weight", 2.0)]).resolve_sliders(&names(&["weight"]));
        assert_eq!(r.unresolved, names(&["nose"]));
        // Values clamp to the slider range before resolving.
        assert_eq!(r.weights, [1.0]);
    }

    #[test]
    fn repeated_sliders_accumulate_and_clamp() {
        let r = shape(&[("w", 0.7), ("w", 0.7)]).resolve_sliders(&names(&["w"]));
        assert_eq!(r.weights, [1.0]);
    }

    #[test]
    fn unresolved_joints_are_listed() {
        let s = CharacterShape {
            proportions: vec![
                JointProportion {
                    joint: "spine".into(),
                    ..Default::default()
                },
                JointProportion {
                    joint: "tail".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(s.unresolved_joints(|j| j == "spine"), names(&["tail"]));
    }

    #[test]
    fn a_shape_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let s: CharacterShape = serde_json::from_str(
            r#"{"target":"hero","sliders":[{"name":"jaw","value":-0.5}],
                "proportions":[{"joint":"thigh.L","scale":1.05,"length":0.1}]}"#,
        )
        .unwrap();
        assert_eq!(s.target, Some(SkinnedMeshHandle(4)));
        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: CharacterShape = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.target, Some(SkinnedMeshHandle(4)));
        assert_eq!(back.sliders, s.sliders);
        assert_eq!(back.proportions, s.proportions);
        assert_eq!(back.proportions[0].scale, 1.05);
        assert!(!back.bake, "runtime deformation is the default");
        assert_eq!(back.asset_id, AssetId::default());
        // A blank proportion is the identity.
        let p = JointProportion::default();
        assert_eq!((p.scale, p.length), (1.0, 0.0));
    }
}
