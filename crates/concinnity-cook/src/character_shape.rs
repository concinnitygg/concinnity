// Build-time check of every CharacterShape against its target SkinnedMesh:
// slider names that match no morph target and joint names not in the skeleton
// are warnings (the shape still builds; the entry is ignored at runtime).
// Runs after the mesh import passes so imported targets and skeletons are
// inline in the args.

use crate::components::CharacterShape;
use crate::ecs::Component;
use concinnity_world::world::WorldJsonlAsset;

// The unresolved slider and joint names of one shape, given its target mesh's
// args (`morph_target_names` and `skeleton`).
pub(crate) fn unresolved_names(
    shape: &CharacterShape,
    mesh_args: &serde_json::Value,
) -> (Vec<String>, Vec<String>) {
    let strings = |field: &str, key: Option<&str>| -> Vec<String> {
        mesh_args
            .get(field)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| match key {
                        Some(k) => v.get(k).and_then(|n| n.as_str()),
                        None => v.as_str(),
                    })
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    };
    let targets = strings("morph_target_names", None);
    let joints = strings("skeleton", Some("name"));
    let sliders = shape.resolve_sliders(&targets).unresolved;
    let unresolved_joints = shape.unresolved_joints(|j| joints.iter().any(|n| n == j));
    (sliders, unresolved_joints)
}

// Warn for every CharacterShape whose names do not resolve against its target.
// Only the name-bearing fields are read: the `target` handle resolver is not
// installed this early in the build, and the name string is all that is needed.
pub(crate) fn warn_unresolved(assets: &[WorldJsonlAsset]) {
    for asset in assets {
        if asset.asset_type != CharacterShape::NAME {
            continue;
        }
        let field = |name: &str| {
            asset
                .args
                .get(name)
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        };
        let shape = CharacterShape {
            sliders: serde_json::from_value(field("sliders")).unwrap_or_default(),
            proportions: serde_json::from_value(field("proportions")).unwrap_or_default(),
            ..Default::default()
        };
        let target = asset
            .args
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let Some(mesh) = assets.iter().find(|a| a.name == target) else {
            continue;
        };
        // A target served from the payload cache keeps its pre-import args,
        // with no targets or skeleton inline to check against.
        if mesh.args.get("vertices").is_none() {
            continue;
        }
        let (sliders, joints) = unresolved_names(&shape, &mesh.args);
        for name in sliders {
            tracing::warn!(
                "Asset '{}': slider '{}' matches no morph target of SkinnedMesh '{}'",
                asset.name,
                name,
                target
            );
        }
        for name in joints {
            tracing::warn!(
                "Asset '{}': joint '{}' is not in the skeleton of SkinnedMesh '{}'",
                asset.name,
                name,
                target
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{JointProportion, ShapeSlider};

    #[test]
    fn names_resolve_against_the_inline_mesh_args() {
        let mesh = serde_json::json!({
            "morph_target_names": ["weight", "jaw+", "jaw-"],
            "skeleton": [{"name": "root"}, {"name": "spine", "parent": 0}]
        });
        let shape = CharacterShape {
            sliders: vec![
                ShapeSlider {
                    name: "weight".into(),
                    value: 0.5,
                },
                ShapeSlider {
                    name: "jaw".into(),
                    value: -0.5,
                },
                ShapeSlider {
                    name: "nose".into(),
                    value: 0.5,
                },
            ],
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
        let (sliders, joints) = unresolved_names(&shape, &mesh);
        assert_eq!(sliders, ["nose"]);
        assert_eq!(joints, ["tail"]);
    }

    #[test]
    fn a_mesh_without_targets_or_skeleton_resolves_nothing() {
        let shape = CharacterShape {
            sliders: vec![ShapeSlider {
                name: "weight".into(),
                value: 0.5,
            }],
            ..Default::default()
        };
        let (sliders, joints) = unresolved_names(&shape, &serde_json::json!({}));
        assert_eq!(sliders, ["weight"]);
        assert!(joints.is_empty());
    }
}
