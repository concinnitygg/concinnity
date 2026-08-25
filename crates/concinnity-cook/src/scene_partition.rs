// Scene ownership derivation for blob packing: which payload-bearing assets
// are reachable exclusively from one scene's members, and therefore pack into
// that scene's payload group. Runs on the expanded world after
// scene-membership resolution, walking the generic reference graph
// (concinnity_world::refs) from every component root.

use crate::world::WorldJsonlAsset;
use std::collections::HashMap;

// Where an asset's payload packs: the global set loaded with the world, or one
// scene's group. Derived, never authored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Owner {
    Global,
    // Index into `ScenePartition::scenes`.
    Scene(usize),
}

pub(crate) struct ScenePartition {
    // Scene asset names in declaration order.
    pub scenes: Vec<String>,
    // Asset name -> derived owner. Assets referenced by nothing are Global.
    owner_of: HashMap<String, Owner>,
}

impl ScenePartition {
    pub(crate) fn owner(&self, asset_name: &str) -> Owner {
        self.owner_of
            .get(asset_name)
            .copied()
            .unwrap_or(Owner::Global)
    }
}

// Ownership label lattice: unlabeled < one scene < Global (top). An asset
// reached from two different scenes, or from any scene-less root, is Global.
fn merge(a: Option<Owner>, b: Owner) -> Owner {
    match (a, b) {
        (None, next) => next,
        (Some(Owner::Global), _) | (_, Owner::Global) => Owner::Global,
        (Some(Owner::Scene(s)), Owner::Scene(n)) if s == n => Owner::Scene(s),
        (Some(Owner::Scene(_)), Owner::Scene(_)) => Owner::Global,
    }
}

// Reference targets (never roots): their owner derives entirely from who
// references them. Resource assets plus the component types that only exist
// to be referenced (mesh sources and Model). Everything else is a root whose
// label is fixed -- labels never merge INTO a root, so a logic asset naming a
// scene prop (an AnimationGraph target, a Behavior show/hide) does not drag that
// prop's resources into the global set.
fn is_reference_target(type_norm: &str) -> bool {
    concinnity_world::registry::RegisteredType::all()
        .iter()
        .filter(|t| t.is_resource())
        .any(|t| t.as_str().to_lowercase() == type_norm)
        || matches!(
            type_norm,
            "proceduralmesh" | "voxelchunk" | "file" | "model" | "shader"
        )
}

pub(crate) fn partition_scenes(assets: &[WorldJsonlAsset]) -> ScenePartition {
    let norm = |t: &str| t.to_lowercase().replace('_', "");

    let scenes: Vec<String> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "scene")
        .map(|a| a.name.clone())
        .collect();
    let scene_index: HashMap<&str, usize> = scenes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let index_of: HashMap<&str, usize> = assets
        .iter()
        .enumerate()
        .map(|(i, a)| (a.name.as_str(), i))
        .collect();
    let edges: Vec<Vec<usize>> = assets
        .iter()
        .map(|a| {
            concinnity_world::refs::referenced_names(a)
                .iter()
                .filter_map(|n| index_of.get(n.as_str()).copied())
                .collect()
        })
        .collect();

    // Seed every root with its own label: a Prop carrying a resolved scene
    // membership labels its scene; every other root is Global. Reference
    // targets start unlabeled and take whatever reaches them.
    let is_target: Vec<bool> = assets
        .iter()
        .map(|a| is_reference_target(&norm(&a.asset_type)))
        .collect();
    let mut labels: Vec<Option<Owner>> = vec![None; assets.len()];
    let mut worklist: Vec<usize> = Vec::new();
    for (i, asset) in assets.iter().enumerate() {
        if is_target[i] {
            continue;
        }
        let label = if norm(&asset.asset_type) == "prop" {
            asset
                .args
                .get("scene")
                .and_then(|v| v.as_str())
                .and_then(|s| scene_index.get(s).copied())
                .map_or(Owner::Global, Owner::Scene)
        } else {
            Owner::Global
        };
        labels[i] = Some(label);
        worklist.push(i);
    }

    // Propagate labels through the reference closure to a fixpoint, merging
    // only into reference targets (root labels are fixed seeds). Labels only
    // move up the lattice, so each target re-enters the worklist at most twice
    // (unlabeled -> scene -> Global) and termination is guaranteed.
    while let Some(i) = worklist.pop() {
        let Some(label) = labels[i] else { continue };
        for &target in &edges[i] {
            if !is_target[target] {
                continue;
            }
            let merged = merge(labels[target], label);
            if labels[target] != Some(merged) {
                labels[target] = Some(merged);
                worklist.push(target);
            }
        }
    }

    let owner_of = assets
        .iter()
        .zip(&labels)
        .filter_map(|(a, l)| l.map(|owner| (a.name.clone(), owner)))
        .collect();

    ScenePartition { scenes, owner_of }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, ty: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args,
        }
    }

    fn scene(name: &str) -> WorldJsonlAsset {
        asset(name, "Scene", serde_json::json!({}))
    }

    fn prop(name: &str, scene: Option<&str>, args: serde_json::Value) -> WorldJsonlAsset {
        let mut args = args;
        if let (Some(s), serde_json::Value::Object(m)) = (scene, &mut args) {
            m.insert("scene".into(), serde_json::Value::String(s.into()));
        }
        asset(name, "Prop", args)
    }

    #[test]
    fn exclusive_chain_is_scene_owned() {
        // day prop -> material -> texture: all three payload carriers owned by
        // "day" (the material and texture are reached only from day's prop).
        let assets = vec![
            scene("day"),
            prop(
                "day_wall",
                Some("day"),
                serde_json::json!({"mesh":"wall_mesh","material":"wall_mat"}),
            ),
            asset("wall_mesh", "ProceduralMesh", serde_json::json!({})),
            asset(
                "wall_mat",
                "Material",
                serde_json::json!({"albedo":"wall_tex"}),
            ),
            asset("wall_tex", "Texture", serde_json::json!({})),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("wall_mesh"), Owner::Scene(0));
        assert_eq!(p.owner("wall_mat"), Owner::Scene(0));
        assert_eq!(p.owner("wall_tex"), Owner::Scene(0));
    }

    #[test]
    fn a_shader_referenced_only_by_a_scene_material_is_scene_owned() {
        // day prop -> material -> shader: the shader's lifetime follows the
        // material's, so it loads and unloads with the scene. The world's own
        // shader (a root nothing references) stays global.
        let assets = vec![
            scene("day"),
            prop(
                "day_wall",
                Some("day"),
                serde_json::json!({"mesh":"wall_mesh","material":"wall_mat"}),
            ),
            asset("wall_mesh", "ProceduralMesh", serde_json::json!({})),
            asset(
                "wall_mat",
                "Material",
                serde_json::json!({"shader":"wall_shader"}),
            ),
            asset(
                "wall_shader",
                "Shader",
                serde_json::json!({"vertex":{"source":"w.metal"},"fragment":{"source":"w.metal"}}),
            ),
            asset(
                "scene_shader",
                "Shader",
                serde_json::json!({"vertex":{"source":"s.metal"},"fragment":{"source":"s.metal"}}),
            ),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("wall_shader"), Owner::Scene(0));
        assert_eq!(p.owner("scene_shader"), Owner::Global);
    }

    #[test]
    fn shared_between_scenes_is_global() {
        let assets = vec![
            scene("day"),
            scene("night"),
            prop(
                "day_a",
                Some("day"),
                serde_json::json!({"mesh":"shared_mesh"}),
            ),
            prop(
                "night_a",
                Some("night"),
                serde_json::json!({"mesh":"shared_mesh"}),
            ),
            asset("shared_mesh", "ProceduralMesh", serde_json::json!({})),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("shared_mesh"), Owner::Global);
    }

    #[test]
    fn sceneless_prop_reference_forces_global() {
        let assets = vec![
            scene("day"),
            prop("day_a", Some("day"), serde_json::json!({"mesh":"m"})),
            prop("everywhere", None, serde_json::json!({"mesh":"m"})),
            asset("m", "ProceduralMesh", serde_json::json!({})),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("m"), Owner::Global);
    }

    #[test]
    fn transitive_sharing_propagates_through_intermediates() {
        // Both scenes reference the same material; its texture is Global even
        // though each scene reaches it through a different path.
        let assets = vec![
            scene("day"),
            scene("night"),
            prop("day_a", Some("day"), serde_json::json!({"material":"mat"})),
            prop(
                "night_a",
                Some("night"),
                serde_json::json!({"material":"mat"}),
            ),
            asset("mat", "Material", serde_json::json!({"albedo":"tex"})),
            asset("tex", "Texture", serde_json::json!({})),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("mat"), Owner::Global);
        assert_eq!(p.owner("tex"), Owner::Global);
    }

    #[test]
    fn unreferenced_resource_is_global() {
        let assets = vec![
            scene("day"),
            asset("orphan", "Texture", serde_json::json!({})),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("orphan"), Owner::Global);
    }

    #[test]
    fn no_scenes_means_everything_global() {
        let assets = vec![
            prop("a", None, serde_json::json!({"mesh":"m"})),
            asset("m", "ProceduralMesh", serde_json::json!({})),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("m"), Owner::Global);
    }

    #[test]
    fn logic_reference_to_a_scene_prop_does_not_globalize_its_resources() {
        // A Behavior naming a day prop (show/hide) must not drag the prop's
        // exclusive resources into the global set: roots keep their labels.
        let assets = vec![
            scene("day"),
            prop("day_a", Some("day"), serde_json::json!({"mesh":"m"})),
            asset("m", "ProceduralMesh", serde_json::json!({})),
            asset(
                "toggle",
                "Behavior",
                serde_json::json!({"on":{"interact":"day_a"},"do":[{"hide":{"target":{"named":"day_a"}}}]}),
            ),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("m"), Owner::Scene(0));
    }

    #[test]
    fn model_expands_ownership_to_submesh_resources() {
        let assets = vec![
            scene("day"),
            prop("day_a", Some("day"), serde_json::json!({"model":"mdl"})),
            asset(
                "mdl",
                "Model",
                serde_json::json!({"meshes":[{"mesh":"m0","material":"mat0"}]}),
            ),
            asset("m0", "Mesh", serde_json::json!({})),
            asset("mat0", "Material", serde_json::json!({"albedo":"t0"})),
            asset("t0", "Texture", serde_json::json!({})),
        ];
        let p = partition_scenes(&assets);
        assert_eq!(p.owner("m0"), Owner::Scene(0));
        assert_eq!(p.owner("mat0"), Owner::Scene(0));
        assert_eq!(p.owner("t0"), Owner::Scene(0));
    }
}
