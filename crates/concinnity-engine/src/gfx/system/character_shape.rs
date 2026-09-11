// Resolves each CharacterShape against its target mesh at init: slider names
// to morph weights, joint names to a proportion layer, and the capsule
// dimensions that follow the proportioned skeleton.

use std::collections::HashMap;

use crate::components::{CharacterCapsule, CharacterShape, SkeletonPose};
use crate::ecs::{PipelineContext, SkinnedMeshHandle};
use crate::gfx::proportions::ProportionLayer;
use crate::gfx::skeleton::Skeleton;

// The static layers one shape contributes to its mesh's pose.
pub(crate) struct ShapeLayers {
    pub(crate) morph_base: Vec<f32>,
    pub(crate) proportions: ProportionLayer,
}

// Every CharacterShape in the world, keyed by its target. The components stay
// in the world so an editor can keep editing them.
pub(super) fn collect(ctx: &PipelineContext) -> HashMap<SkinnedMeshHandle, CharacterShape> {
    let mut map = HashMap::new();
    for shape in ctx.query::<CharacterShape>() {
        match shape.target {
            Some(target) => {
                if map.insert(target, shape.clone()).is_some() {
                    tracing::warn!(
                        "CharacterShape '{}': its target already has a shape; the later one wins",
                        shape.asset_id
                    );
                }
            }
            None => tracing::warn!(
                "CharacterShape '{}' has no target SkinnedMesh, ignored",
                shape.asset_id
            ),
        }
    }
    map
}

// Resolve `shape` against the mesh's morph-target names and skeleton.
// Unresolved names were already warned about at build time; they are logged
// again here in case the mesh changed since.
pub(super) fn resolve(
    shape: &CharacterShape,
    skeleton: &Skeleton,
    morph_names: &[String],
) -> ShapeLayers {
    for name in &shape.resolve_sliders(morph_names).unresolved {
        tracing::warn!(
            "CharacterShape '{}': slider '{}' matches no morph target of its mesh",
            shape.asset_id,
            name
        );
    }
    for joint in shape.unresolved_joints(|name| skeleton.joint_index(name).is_some()) {
        tracing::warn!(
            "CharacterShape '{}': joint '{}' is not in its mesh's skeleton",
            shape.asset_id,
            joint
        );
    }
    layers(shape, skeleton, morph_names)
}

// The layers alone, with no reporting: the per-frame path an editor preview
// takes while a slider is dragged.
pub(crate) fn layers(
    shape: &CharacterShape,
    skeleton: &Skeleton,
    morph_names: &[String],
) -> ShapeLayers {
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

// The pose for a mesh, seeded through its shape layers when it has a shape.
pub(super) fn seed_pose(
    handle: SkinnedMeshHandle,
    skinned_index: usize,
    skeleton: Skeleton,
    layers: Option<ShapeLayers>,
) -> SkeletonPose {
    let pose = SkeletonPose::new(handle, skinned_index, skeleton);
    match layers {
        Some(l) => pose.with_shape(l.morph_base, l.proportions),
        None => pose,
    }
}

// Capsule dimensions following the proportioned skeleton: the half-height
// scales with the skeleton's height, the radius with the root joint's scale.
pub(crate) fn proportioned_capsule(
    capsule: &CharacterCapsule,
    skeleton: &Skeleton,
    proportions: &ProportionLayer,
) -> (f32, f32) {
    (
        capsule.half_height * proportions.height_ratio(skeleton),
        capsule.radius * proportions.root_scale(skeleton),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{JointProportion, ShapeSlider};
    use crate::gfx::skeleton::{Joint, JointPose};

    fn chain() -> Skeleton {
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
        let pose = seed_pose(SkinnedMeshHandle(0), 0, skeleton, Some(layers));
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
        let pose = seed_pose(SkinnedMeshHandle(0), 0, chain(), Some(layers));
        assert!(pose.morph_weights.is_empty());
    }
}
