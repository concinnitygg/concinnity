//! Resolves a `CharacterShape` against its target mesh: slider names to morph
//! weights, joint names to a proportion layer, and the capsule dimensions that
//! follow the proportioned skeleton.

use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::animation::proportions::ProportionLayer;
use crate::animation::skeleton::Skeleton;
use crate::components::{CharacterCapsule, Identity, SkeletonPose};
use crate::ecs::{PipelineContext, SkinnedMeshHandle};
use crate::gfx::render_types::SkinnedIndex;

/// The static layers one shape contributes to its mesh's pose.
pub struct ShapeLayers {
    /// The morph weights the pose starts from; empty when every slider is zero.
    pub morph_base: Vec<f32>,
    /// The per-joint proportions.
    pub proportions: ProportionLayer,
}

/// Every `CharacterShape` in the world, keyed by its target. The components
/// stay in the world so an editor can keep editing them.
pub fn collect(ctx: &PipelineContext<'_>) -> HashMap<SkinnedMeshHandle, CharacterShape> {
    let mut map = HashMap::new();
    for (entity, shape) in ctx.query_with_entity::<CharacterShape>() {
        let id = ctx.get::<Identity>(entity).map(|i| i.id());
        if let Some(target) = shape.target {
            if map.insert(target, shape.clone()).is_some() {
                tracing::warn!(
                    "CharacterShape {id:?}: its target already has a shape; the later one wins"
                );
            }
        } else {
            tracing::warn!("CharacterShape {id:?} has no target SkinnedMesh, ignored");
        }
    }
    map
}

/// Resolve `shape` against the mesh's morph-target names and skeleton, warning
/// about each slider or joint name the mesh does not have.
pub fn resolve(shape: &CharacterShape, skeleton: &Skeleton, morph_names: &[String]) -> ShapeLayers {
    for name in &shape.resolve_sliders(morph_names).unresolved {
        tracing::warn!(
            "CharacterShape on mesh {:?}: slider '{}' matches no morph target of it",
            shape.target.map(super::super::ecs::handle::Handle::index),
            name
        );
    }
    for joint in shape.unresolved_joints(|name| skeleton.joint_index(name).is_some()) {
        tracing::warn!(
            "CharacterShape on mesh {:?}: joint '{}' is not in its skeleton",
            shape.target.map(super::super::ecs::handle::Handle::index),
            joint
        );
    }
    layers(shape, skeleton, morph_names)
}

/// The layers alone, with no reporting: the per-frame path an editor preview
/// takes while a slider is dragged.
pub fn layers(shape: &CharacterShape, skeleton: &Skeleton, morph_names: &[String]) -> ShapeLayers {
    let sliders = shape.resolve_sliders(morph_names);
    let morph_base = if sliders.weights.iter().any(|w| *w != 0.0) {
        sliders.weights
    } else {
        Vec::new()
    };
    ShapeLayers {
        morph_base,
        proportions: ProportionLayer::resolve(skeleton, &shape.proportions),
    }
}

/// The pose for a mesh, seeded through its shape layers when it has a shape.
pub fn seed_pose(
    handle: SkinnedMeshHandle,
    skinned_index: SkinnedIndex,
    skeleton: Skeleton,
    layers: Option<ShapeLayers>,
) -> SkeletonPose {
    let pose = SkeletonPose::new(handle, skinned_index, skeleton);
    match layers {
        Some(l) => pose.with_shape(l.morph_base, l.proportions),
        None => pose,
    }
}

/// Capsule dimensions `(half_height, radius)` following the proportioned
/// skeleton: the half-height scales with the skeleton's height, the radius with
/// the root joint's scale.
pub fn proportioned_capsule(
    capsule: &CharacterCapsule,
    skeleton: &Skeleton,
    proportions: &ProportionLayer,
) -> (f32, f32) {
    (
        capsule.half_height * proportions.height_ratio(skeleton),
        capsule.radius * proportions.root_scale(skeleton),
    )
}

/// One named shape value in `[-1, 1]`.
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct ShapeSlider {
    /// Slider name; matched against the target mesh's morph-target names.
    pub name: String,
    /// Slider value, clamped to `[-1, 1]`.
    pub value: f32,
}

/// One joint's proportion change.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct JointProportion {
    /// Name of the joint in the target mesh's `skeleton`.
    pub joint: String,
    /// Uniform scale applied to the joint (and, through the hierarchy,
    /// everything below it). `1` leaves it alone.
    #[asset(default = 1.0)]
    pub scale: f32,
    /// Extra length along the bone, in model units: every child joint is
    /// pushed that far along its bind direction from this joint. `0` leaves
    /// it alone.
    pub length: f32,
}

/// Shape sliders and joint proportions applied to one [`SkinnedMesh`](#skinnedmesh).
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
/// `target` may name a [`CharacterModel`](#charactermodel) as well as a
/// `SkinnedMesh`; the model's emitted mesh is what the shape deforms.
///
/// **Baking.** With `bake` set, the build flattens the shape into its target:
/// the sliders' deformation is applied to the vertices and the morph targets
/// dropped, the bind pose is rewritten through the proportions, the capsule
/// is resized, and this asset is consumed. The result is a plain `SkinnedMesh`
/// with no per-frame shape work, for characters that never change shape.
///
/// ```rust
/// # use concinnity_engine::components::{CharacterShape, JointProportion, ShapeSlider};
/// CharacterShape {
///     sliders: vec![ShapeSlider { name: "weight".into(), value: 0.5 }],
///     proportions: vec![JointProportion { joint: "spine".into(), scale: 1.1, length: 0.0 }],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct CharacterShape {
    /// The [`SkinnedMesh`](#skinnedmesh) this shape deforms.
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
pub(crate) mod tests {
    use super::*;
    use crate::animation::skeleton::{Joint, JointPose};
    use crate::components::{JointProportion, ShapeSlider};
    use alloc::string::ToString;
    use alloc::vec;

    /// A root joint and a head two units above it.
    pub(crate) fn chain() -> Skeleton {
        let joint = |name: &str, parent: Option<usize>, y: f32| Joint {
            name: name.to_string(),
            parent,
            bind: JointPose {
                translation: [0.0, y, 0.0],
                ..Default::default()
            },
        };
        Skeleton::new(vec![joint("root", None, 0.0), joint("head", Some(0), 2.0)])
    }

    #[test]
    fn resolves_sliders_and_joints_against_the_mesh() {
        let shape = CharacterShape {
            sliders: vec![
                ShapeSlider {
                    name: "jaw".into(),
                    value: -0.5,
                },
                ShapeSlider {
                    name: "missing".into(),
                    value: 1.0,
                },
            ],
            proportions: vec![JointProportion {
                joint: "root".into(),
                scale: 1.5,
                length: 0.0,
            }],
            ..Default::default()
        };
        let skeleton = chain();
        let names = vec!["jaw+".to_string(), "jaw-".to_string()];
        let layers = resolve(&shape, &skeleton, &names);
        assert_eq!(layers.morph_base, [0.0, 0.5]);
        assert!(!layers.proportions.is_empty());
        let (half, radius) = proportioned_capsule(
            &CharacterCapsule {
                half_height: 1.0,
                radius: 0.4,
            },
            &skeleton,
            &layers.proportions,
        );
        assert!((half - 1.5).abs() < 1e-5 && (radius - 0.6).abs() < 1e-5);
        let pose = seed_pose(
            SkinnedMeshHandle::new(0),
            SkinnedIndex(0),
            skeleton,
            Some(layers),
        );
        assert_eq!(pose.morph_weights, [0.0, 0.5]);
        assert_eq!(pose.joint_matrices[0][0][0], 1.5);
    }

    #[test]
    fn all_zero_sliders_leave_no_base_layer() {
        let shape = CharacterShape {
            sliders: vec![ShapeSlider {
                name: "jaw".into(),
                value: 0.0,
            }],
            ..Default::default()
        };
        let layers = resolve(&shape, &chain(), &["jaw".to_string()]);
        assert!(layers.morph_base.is_empty());
        let pose = seed_pose(
            SkinnedMeshHandle::new(0),
            SkinnedIndex(0),
            chain(),
            Some(layers),
        );
        assert!(pose.morph_weights.is_empty());
    }

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
}
