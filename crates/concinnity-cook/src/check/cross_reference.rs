//! Cross-asset name reference validation. Runs on the fully expanded world and
//! checks that every named reference (Prop -> Mesh, Material -> Texture, etc.)
//! resolves to an asset of the right kind, and that Prop parent chains are
//! acyclic. Every problem found is collected: validation never stops at the
//! first error, so the caller can report them all in one pass.
//!
//! Reference fields are validated generically from the tables the registry
//! derives from each schema's field types (`validate_registry_refs`), so typing
//! a field `Ref<T>` IS enforcing it -- the same table drives the editor's Ref
//! pickers. Only what a field type cannot state remains hand-written: each such
//! asset implements `CrossReferenced` in `asset_refs` (the polymorphic mesh
//! sources, references inside enum variants, required-ness) and
//! `cross_refs_for` dispatches to it by type. A hand impl must not re-check a
//! derived field, or the problem reports twice.

use std::collections::{HashMap, HashSet};

use super::asset_refs::{CrossRef, CrossReferenced, RefKind};
use crate::authoring::field_path::string_leaves;
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// Dispatch reference extraction by asset type. Every arm delegates
// to a `CrossReferenced` impl in the named asset's file.
pub(crate) fn cross_refs_for(
    asset_type: RegisteredType,
    name: &str,
    args: &serde_json::Value,
) -> Vec<CrossRef> {
    use concinnity_core::components::{
        AnimationGraph, Behavior, Camera3D, InstancedProp, Model, PhysicsJoint, Prop, VoxelChunk,
        VoxelWorld,
    };
    match asset_type {
        RegisteredType::AnimationGraph => AnimationGraph::cross_refs(name, args),
        RegisteredType::Behavior => Behavior::cross_refs(name, args),
        RegisteredType::Camera3D => Camera3D::cross_refs(name, args),
        RegisteredType::Prop => Prop::cross_refs(name, args),
        RegisteredType::Model => Model::cross_refs(name, args),
        RegisteredType::InstancedProp => InstancedProp::cross_refs(name, args),
        RegisteredType::VoxelChunk => VoxelChunk::cross_refs(name, args),
        RegisteredType::VoxelWorld => VoxelWorld::cross_refs(name, args),
        RegisteredType::PhysicsJoint => PhysicsJoint::cross_refs(name, args),
        _ => Vec::new(),
    }
}

// The assets a reference can name: those with a `$id`. An anonymous asset's
// label addresses it in tools, never from another entry.
fn referenceable(assets: &[WorldJsonlAsset]) -> impl Iterator<Item = &WorldJsonlAsset> {
    assets.iter().filter(|a| !a.is_anonymous())
}

// Resolve every reference field the registry derives: each non-empty string
// at the field's path must be a declared asset of one of its target types, or
// of any type when the field names none. Name-sets are built once per distinct
// target; every target names a real declarable type (guarded by
// `ref_fields_name_real_target_types`).
fn validate_registry_refs(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let mut scopes: HashMap<&str, HashSet<&str>> = HashMap::new();
    for ty in RegisteredType::all() {
        for field in ty.ref_fields() {
            for &target in field.targets {
                scopes.entry(target).or_insert_with(|| {
                    let target = RegisteredType::parse(target)
                        .expect("ref_fields_name_real_target_types guards every target");
                    referenceable(assets)
                        .filter(|a| a.asset_type == target)
                        .map(|a| a.id.as_str())
                        .collect()
                });
            }
        }
    }
    let any: HashSet<&str> = referenceable(assets).map(|a| a.id.as_str()).collect();

    for asset in assets {
        for field in asset.asset_type.ref_fields() {
            for (location, referenced) in string_leaves(&asset.args, &field.path) {
                let resolves = if field.targets.is_empty() {
                    any.contains(referenced)
                } else {
                    field.targets.iter().any(|t| scopes[t].contains(referenced))
                };
                if resolves {
                    continue;
                }
                errors.push(format!(
                    "{} '{}': {} '{}' not found, add {} asset with that `$id`",
                    asset.asset_type.as_str(),
                    asset.id,
                    location,
                    referenced,
                    one_of(field.targets)
                ));
            }
        }
    }
}

// "a Prop", or "a Prop or a SkyRotation" for a field with several targets, or
// "an" for a field that accepts any asset.
fn one_of(targets: &[&str]) -> String {
    let named: Vec<String> = targets.iter().map(|t| format!("a {t}")).collect();
    match named.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
        None => "an".to_string(),
    }
}

// The name-sets of an expanded world, one per `RefKind`. Built once per
// validation pass; `contains` answers whether a reference resolves.
struct RefScope<'a> {
    mesh_sources: HashSet<&'a str>,
    scenes: HashSet<&'a str>,
    animations: HashSet<&'a str>,
    audio_clips: HashSet<&'a str>,
    screens: HashSet<&'a str>,
    trigger_volumes: HashSet<&'a str>,
    all_names: HashSet<&'a str>,
}

impl<'a> RefScope<'a> {
    fn build(assets: &'a [WorldJsonlAsset]) -> Self {
        // Names of every asset of the given type.
        let by_type = |asset_type: RegisteredType| -> HashSet<&'a str> {
            referenceable(assets)
                .filter(|a| a.asset_type == asset_type)
                .map(|a| a.id.as_str())
                .collect()
        };

        // Mesh, ProceduralMesh, VoxelChunk, and mesh-kind File are all valid
        // mesh sources; the same classifier the build's mesh-source handle
        // assignment uses, so the two never disagree on what a `.mesh` name may
        // resolve to.
        let mesh_sources = referenceable(assets)
            .filter(|a| crate::authoring::resource_type::is_mesh_source(a.asset_type, &a.args))
            .map(|a| a.id.as_str())
            .collect();

        RefScope {
            mesh_sources,
            scenes: by_type(RegisteredType::Scene),
            animations: by_type(RegisteredType::Animation),
            audio_clips: by_type(RegisteredType::AudioClip),
            screens: by_type(RegisteredType::Screen),
            trigger_volumes: by_type(RegisteredType::TriggerVolume),
            all_names: referenceable(assets).map(|a| a.id.as_str()).collect(),
        }
    }

    // True when `name` is satisfied by an asset of the given reference kind.
    fn contains(&self, kind: RefKind, name: &str) -> bool {
        match kind {
            RefKind::MeshSource => self.mesh_sources.contains(name),
            RefKind::Scene => self.scenes.contains(name),
            RefKind::Animation => self.animations.contains(name),
            RefKind::AudioClip => self.audio_clips.contains(name),
            RefKind::Screen => self.screens.contains(name),
            RefKind::TriggerVolume => self.trigger_volumes.contains(name),
            RefKind::AnyAsset => self.all_names.contains(name),
        }
    }
}

// Validate cross-asset name references on the expanded world.
pub(crate) fn validate_cross_references(assets: &[WorldJsonlAsset]) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let scope = RefScope::build(assets);

    validate_registry_refs(assets, &mut errors);

    for asset in assets {
        for cross_ref in cross_refs_for(asset.asset_type, &asset.id, &asset.args) {
            match cross_ref {
                CrossRef::Resolve {
                    kind,
                    target,
                    error,
                } => {
                    if !scope.contains(kind, &target) {
                        errors.push(error);
                    }
                }
                CrossRef::Issue(msg) => errors.push(msg),
            }
        }
    }

    // Detect cycles in the Prop parent chain. This is a graph-global pass, so
    // it stays in the validator rather than the per-asset trait.
    let prop_parent_map: std::collections::HashMap<&str, &str> = assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Prop)
        .filter_map(|a| {
            let parent = a
                .args
                .get("parent")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())?;
            Some((a.id.as_str(), parent))
        })
        .collect();

    for &start in prop_parent_map.keys() {
        let mut visited = std::collections::HashSet::new();
        let mut current = start;
        visited.insert(current);
        while let Some(&parent) = prop_parent_map.get(current) {
            if !visited.insert(parent) {
                errors.push(format!(
                    "Prop '{}': parent chain contains a cycle (via '{}')",
                    start, parent
                ));
                break;
            }
            current = parent;
        }
    }

    check_graph_ownership(assets, &mut errors);
    check_follow_targets(assets, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// AnimationGraph ownership rules. A graph owns its target mesh's animation set:
// at most one graph per SkinnedMesh, every clip a graph references must
// actually target that mesh, and every Animation targeting a graph-driven
// mesh must be referenced by the graph (a loose clip would silently never
// play, since the graph decides what runs). These need the whole world, so
// they live here rather than in the per-asset checks.
fn check_graph_ownership(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let graphs: Vec<&WorldJsonlAsset> = assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::AnimationGraph)
        .collect();
    if graphs.is_empty() {
        return;
    }

    // Animation name -> its target SkinnedMesh name.
    let clip_targets: std::collections::HashMap<&str, &str> = assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Animation)
        .map(|a| {
            let target = a.args.get("target").and_then(|v| v.as_str()).unwrap_or("");
            (a.id.as_str(), target)
        })
        .collect();

    let mut owner_by_mesh: std::collections::HashMap<&str, &str> = Default::default();
    for graph in &graphs {
        let mesh = graph
            .args
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if mesh.is_empty() {
            continue; // missing target already reported by cross_refs
        }
        if let Some(other) = owner_by_mesh.insert(mesh, graph.id.as_str()) {
            errors.push(format!(
                "AnimationGraph '{}': SkinnedMesh '{}' is already driven by AnimationGraph '{}'; \
                 a mesh can have at most one graph",
                graph.id, mesh, other
            ));
        }

        let mut referenced: HashSet<String> = HashSet::new();
        for state in graph
            .args
            .get("states")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[])
        {
            // Single-clip and blendspace members alike; a state naming no
            // clips at all is already reported by cross_refs.
            for clip in super::asset_refs::state_clip_names(state) {
                if let Some(&clip_target) = clip_targets.get(clip.as_str())
                    && clip_target != mesh
                {
                    errors.push(format!(
                        "AnimationGraph '{}': clip '{}' targets SkinnedMesh '{}', not the graph's \
                         target '{}'",
                        graph.id, clip, clip_target, mesh
                    ));
                }
                referenced.insert(clip);
            }
        }

        for (&clip, &clip_target) in &clip_targets {
            if clip_target == mesh && !referenced.contains(clip) {
                errors.push(format!(
                    "Animation '{}': targets SkinnedMesh '{}', which AnimationGraph '{}' drives, \
                     but no graph state references it; add a state for it or remove the clip",
                    clip, mesh, graph.id
                ));
            }
        }
    }
}

// Third-person follow rules. The followed SkinnedMesh must declare a
// `capsule` (the controller moves its character capsule), and an explicitly
// named speed parameter must exist on an AnimationGraph driving that mesh. An
// omitted `speed_parameter` (the "speed" default) is not enforced, so a
// graph-less direct-drive character still builds; the runtime warns and
// skips the writes instead. These need the whole world, so they live here
// rather than in the per-asset checks.
fn check_follow_targets(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    for camera in assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Camera3D)
    {
        let Some(follow) = camera
            .args
            .get("controller")
            .and_then(|c| c.get("follow"))
            .filter(|f| !f.is_null())
        else {
            continue;
        };
        let target = follow.get("target").and_then(|v| v.as_str()).unwrap_or("");
        if target.is_empty() {
            continue; // missing target already reported by cross_refs
        }

        let has_capsule = assets.iter().any(|a| {
            a.asset_type == RegisteredType::SkinnedMesh
                && a.id == target
                && a.args.get("capsule").is_some_and(|c| !c.is_null())
        });
        if !has_capsule {
            errors.push(format!(
                "Camera3D '{}': follow target SkinnedMesh '{}' has no `capsule`; the \
                 third-person controller needs a character capsule to move",
                camera.id, target
            ));
        }

        if let Some(param) = follow.get("speed_parameter").and_then(|v| v.as_str())
            && !param.is_empty()
        {
            let declared = assets.iter().any(|a| {
                a.asset_type == RegisteredType::AnimationGraph
                    && a.args.get("target").and_then(|v| v.as_str()) == Some(target)
                    && a.args
                        .get("parameters")
                        .and_then(|p| p.as_array())
                        .is_some_and(|params| {
                            params
                                .iter()
                                .any(|p| p.get("name").and_then(|n| n.as_str()) == Some(param))
                        })
            });
            if !declared {
                errors.push(format!(
                    "Camera3D '{}': no AnimationGraph on follow target '{}' declares the speed \
                     parameter '{}'",
                    camera.id, target, param
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: name.to_string(),
            asset_type,
            args,
        }
    }

    // Joins every cross-reference error into one string so a test can assert
    // on a substring regardless of how many problems were reported.
    fn err_text(assets: &[WorldJsonlAsset]) -> String {
        validate_cross_references(assets).unwrap_err().join("\n")
    }

    #[test]
    fn valid_references_pass() {
        let assets = vec![
            asset(
                "my_mesh",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "my_tex",
                RegisteredType::Texture,
                serde_json::json!({"generator":"brick"}),
            ),
            asset(
                "my_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"my_mesh","texture":"my_tex"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn prop_missing_mesh_fails() {
        let assets = vec![asset(
            "my_prop",
            RegisteredType::Prop,
            serde_json::json!({"mesh":"missing_mesh"}),
        )];
        assert!(err_text(&assets).contains("missing_mesh"));
    }

    #[test]
    fn prop_missing_material_fails() {
        let assets = vec![
            asset(
                "my_mesh",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "my_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"my_mesh","material":"no_mat"}),
            ),
        ];
        assert!(err_text(&assets).contains("no_mat"));
    }

    #[test]
    fn prop_model_valid_passes() {
        let assets = vec![
            asset(
                "body",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[0.4,0.4,0.4]}),
            ),
            asset(
                "mat_wood",
                RegisteredType::Material,
                serde_json::json!({"roughness":0.7}),
            ),
            asset(
                "crate_model",
                RegisteredType::Model,
                serde_json::json!({"meshes":[{"mesh":"body","material":"mat_wood"}]}),
            ),
            asset(
                "crate_a",
                RegisteredType::Prop,
                serde_json::json!({"model":"crate_model"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn model_missing_mesh_fails() {
        let assets = vec![asset(
            "my_model",
            RegisteredType::Model,
            serde_json::json!({"meshes":[{"mesh":"ghost_mesh"}]}),
        )];
        assert!(err_text(&assets).contains("ghost_mesh"));
    }

    #[test]
    fn prop_missing_model_fails() {
        let assets = vec![asset(
            "my_prop",
            RegisteredType::Prop,
            serde_json::json!({"model":"ghost_model"}),
        )];
        assert!(err_text(&assets).contains("ghost_model"));
    }

    #[test]
    fn file_mesh_counts_as_mesh_source() {
        let assets = vec![
            asset(
                "room_obj",
                RegisteredType::File,
                serde_json::json!({"path":"assets/room.obj","kind":"obj"}),
            ),
            asset(
                "my_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"room_obj"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn file_non_mesh_does_not_count_as_mesh_source() {
        let assets = vec![
            asset(
                "wall_png",
                RegisteredType::File,
                serde_json::json!({"path":"assets/wall.png","kind":"png"}),
            ),
            asset(
                "my_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"wall_png"}),
            ),
        ];
        assert!(err_text(&assets).contains("wall_png"));
    }

    #[test]
    fn raw_mesh_counts_as_mesh_source() {
        let assets = vec![
            asset(
                "inline_mesh",
                RegisteredType::Mesh,
                serde_json::json!({"vertices":[],"indices":[]}),
            ),
            asset(
                "my_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"inline_mesh"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn material_missing_albedo_fails() {
        let assets = vec![asset(
            "my_mat",
            RegisteredType::Material,
            serde_json::json!({"albedo":"no_tex"}),
        )];
        assert!(err_text(&assets).contains("no_tex"));
    }

    #[test]
    fn material_missing_normal_map_fails() {
        let assets = vec![asset(
            "my_mat",
            RegisteredType::Material,
            serde_json::json!({"normal_map":"no_nrm"}),
        )];
        assert!(err_text(&assets).contains("no_nrm"));
    }

    #[test]
    fn material_valid_normal_map_passes() {
        let assets = vec![
            asset(
                "nrm_tex",
                RegisteredType::Texture,
                serde_json::json!({"generator":"solid","color":[128,128,255,255]}),
            ),
            asset(
                "my_mat",
                RegisteredType::Material,
                serde_json::json!({"normal_map":"nrm_tex"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn empty_optional_references_pass() {
        let assets = vec![
            asset(
                "my_mesh",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "my_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"my_mesh"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn prop_valid_parent_passes() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "frame",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","position":[0,0,0]}),
            ),
            asset(
                "panel",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"frame"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    // `parent` declares two targets: a prop may hang off another prop or off
    // the celestial-sphere pivot, and resolving in either is enough.
    #[test]
    fn prop_parented_to_the_sky_rotation_passes() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "sky",
                RegisteredType::SkyRotation,
                serde_json::json!({"axis":[1,0,0]}),
            ),
            asset(
                "moon",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"sky"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    // A `parent` that resolves to neither target names both in the failure.
    #[test]
    fn a_multi_target_ref_failure_names_every_target() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "moon",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"nothing"}),
            ),
        ];
        let text = err_text(&assets);
        assert!(text.contains("a Prop or a SkyRotation"), "{text}");
    }

    #[test]
    fn prop_missing_parent_fails() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "panel",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"ghost_prop"}),
            ),
        ];
        assert!(err_text(&assets).contains("ghost_prop"));
    }

    #[test]
    fn prop_parent_cycle_fails() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "a",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"b"}),
            ),
            asset(
                "b",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"a"}),
            ),
        ];
        assert!(err_text(&assets).contains("cycle"));
    }

    #[test]
    fn prop_parent_chain_no_cycle_passes() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "root",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box"}),
            ),
            asset(
                "mid",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"root"}),
            ),
            asset(
                "leaf",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","parent":"mid"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn scene_camera_shot_valid_passes() {
        let assets = vec![
            asset("cam", RegisteredType::Camera3D, serde_json::json!({})),
            asset(
                "day",
                RegisteredType::Scene,
                serde_json::json!({"camera_shot":"cam"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn scene_camera_shot_missing_fails() {
        let assets = vec![asset(
            "day",
            RegisteredType::Scene,
            serde_json::json!({"camera_shot":"ghost_cam"}),
        )];
        assert!(err_text(&assets).contains("ghost_cam"));
    }

    #[test]
    fn voxel_chunk_counts_as_mesh_source() {
        let assets = vec![
            asset(
                "air",
                RegisteredType::BlockType,
                serde_json::json!({"solid":false}),
            ),
            asset("stone", RegisteredType::BlockType, serde_json::json!({})),
            asset(
                "chunk",
                RegisteredType::VoxelChunk,
                serde_json::json!({
                    "palette":["air","stone"],
                    "dim":[1,1,1],
                    "blocks":[1],
                }),
            ),
            asset(
                "p",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"chunk"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn voxel_chunk_unknown_block_type_fails() {
        let assets = vec![
            asset(
                "air",
                RegisteredType::BlockType,
                serde_json::json!({"solid":false}),
            ),
            asset(
                "chunk",
                RegisteredType::VoxelChunk,
                serde_json::json!({
                    "palette":["air","ghost_block"],
                    "dim":[1,1,1],
                    "blocks":[0],
                }),
            ),
        ];
        assert!(err_text(&assets).contains("ghost_block"));
    }

    #[test]
    fn instanced_prop_valid_references_pass() {
        let assets = vec![
            asset(
                "rock_mesh",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"sphere","radius":0.4,"rings":6,"segments":8}),
            ),
            asset(
                "tex_rock",
                RegisteredType::Texture,
                serde_json::json!({"generator":"stone"}),
            ),
            asset(
                "mat_rock",
                RegisteredType::Material,
                serde_json::json!({"albedo":"tex_rock"}),
            ),
            asset(
                "rocks",
                RegisteredType::InstancedProp,
                serde_json::json!({
                    "mesh":"rock_mesh",
                    "material":"mat_rock",
                    "instances":[
                        {"position":[1.0, 0.0, 2.0]},
                        {"position":[3.0, 0.0, -1.0]}
                    ]
                }),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn instanced_prop_missing_mesh_fails() {
        let assets = vec![asset(
            "rocks",
            RegisteredType::InstancedProp,
            serde_json::json!({"mesh":"ghost_mesh","instances":[]}),
        )];
        assert!(err_text(&assets).contains("ghost_mesh"));
    }

    #[test]
    fn instanced_prop_empty_mesh_field_fails() {
        let assets = vec![asset(
            "rocks",
            RegisteredType::InstancedProp,
            serde_json::json!({"instances":[]}),
        )];
        assert!(err_text(&assets).contains("`mesh` field is required"));
    }

    #[test]
    fn instanced_prop_missing_material_fails() {
        let assets = vec![
            asset(
                "rock_mesh",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[0.5,0.5,0.5]}),
            ),
            asset(
                "rocks",
                RegisteredType::InstancedProp,
                serde_json::json!({"mesh":"rock_mesh","material":"ghost_mat","instances":[]}),
            ),
        ];
        assert!(err_text(&assets).contains("ghost_mat"));
    }

    #[test]
    fn instanced_prop_voxel_chunk_mesh_passes() {
        let assets = vec![
            asset(
                "air",
                RegisteredType::BlockType,
                serde_json::json!({"solid":false}),
            ),
            asset("stone", RegisteredType::BlockType, serde_json::json!({})),
            asset(
                "chunk",
                RegisteredType::VoxelChunk,
                serde_json::json!({
                    "palette":["air","stone"],
                    "dim":[1,1,1],
                    "blocks":[1],
                }),
            ),
            asset(
                "rocks",
                RegisteredType::InstancedProp,
                serde_json::json!({"mesh":"chunk","instances":[]}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn joint_valid_two_body_passes() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "a",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","collider":{"shape":"cuboid","half_extents":[1,1,1]}}),
            ),
            asset(
                "b",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","collider":{"shape":"cuboid","half_extents":[1,1,1]}}),
            ),
            asset(
                "j",
                RegisteredType::PhysicsJoint,
                serde_json::json!({"kind":"revolute","body_a":"a","body_b":"b"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn joint_world_anchor_passes_without_body_b() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset(
                "bob",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"box","collider":{"shape":"ball","radius":0.3}}),
            ),
            asset(
                "pendulum",
                RegisteredType::PhysicsJoint,
                serde_json::json!({"kind":"revolute","body_a":"bob","anchor_b":[0,5,0]}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    #[test]
    fn joint_missing_body_a_fails() {
        let assets = vec![asset(
            "j",
            RegisteredType::PhysicsJoint,
            serde_json::json!({"kind":"fixed","body_b":"b"}),
        )];
        assert!(err_text(&assets).contains("body_a"));
    }

    #[test]
    fn joint_unknown_body_b_fails() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset("a", RegisteredType::Prop, serde_json::json!({"mesh":"box"})),
            asset(
                "j",
                RegisteredType::PhysicsJoint,
                serde_json::json!({"kind":"fixed","body_a":"a","body_b":"ghost"}),
            ),
        ];
        assert!(err_text(&assets).contains("ghost"));
    }

    #[test]
    fn joint_unknown_kind_fails() {
        let assets = vec![
            asset(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
            ),
            asset("a", RegisteredType::Prop, serde_json::json!({"mesh":"box"})),
            asset(
                "j",
                RegisteredType::PhysicsJoint,
                serde_json::json!({"kind":"frumpus","body_a":"a"}),
            ),
        ];
        assert!(err_text(&assets).contains("frumpus"));
    }

    // A minimal skinned world: mesh + two clips + a graph referencing both.
    fn graph_world() -> Vec<WorldJsonlAsset> {
        vec![
            asset("hero", RegisteredType::SkinnedMesh, serde_json::json!({})),
            asset(
                "idle",
                RegisteredType::Animation,
                serde_json::json!({"target":"hero"}),
            ),
            asset(
                "run",
                RegisteredType::Animation,
                serde_json::json!({"target":"hero"}),
            ),
            asset(
                "g",
                RegisteredType::AnimationGraph,
                serde_json::json!({
                    "target":"hero",
                    "states":[
                        {"name":"idle","clip":"idle"},
                        {"name":"run","clip":"run"}
                    ]
                }),
            ),
        ]
    }

    #[test]
    fn anim_graph_valid_references_pass() {
        assert!(validate_cross_references(&graph_world()).is_ok());
    }

    #[test]
    fn anim_graph_missing_target_mesh_fails() {
        let mut assets = graph_world();
        assets.remove(0);
        assert!(err_text(&assets).contains("'hero' not found"));
    }

    #[test]
    fn anim_graph_missing_clip_fails() {
        let mut assets = graph_world();
        assets[3].args["states"][1]["clip"] = serde_json::json!("ghost_clip");
        assert!(err_text(&assets).contains("ghost_clip"));
    }

    #[test]
    fn anim_graph_unreferenced_clip_on_target_fails() {
        let mut assets = graph_world();
        assets.push(asset(
            "wave",
            RegisteredType::Animation,
            serde_json::json!({"target":"hero"}),
        ));
        assert!(err_text(&assets).contains("no graph state references it"));
    }

    #[test]
    fn anim_graph_clip_targeting_other_mesh_fails() {
        let mut assets = graph_world();
        assets.push(asset(
            "other",
            RegisteredType::SkinnedMesh,
            serde_json::json!({}),
        ));
        assets.push(asset(
            "other_idle",
            RegisteredType::Animation,
            serde_json::json!({"target":"other"}),
        ));
        assets[3].args["states"][1]["clip"] = serde_json::json!("other_idle");
        let errs = err_text(&assets);
        assert!(errs.contains("not the graph's"));
        // The displaced 'run' clip is now also unreferenced.
        assert!(errs.contains("no graph state references it"));
    }

    #[test]
    fn two_graphs_on_one_mesh_fails() {
        let mut assets = graph_world();
        assets.push(asset(
            "g2",
            RegisteredType::AnimationGraph,
            serde_json::json!({
                "target":"hero",
                "states":[{"name":"idle","clip":"idle"},{"name":"run","clip":"run"}]
            }),
        ));
        assert!(err_text(&assets).contains("at most one graph"));
    }

    #[test]
    fn blend_members_count_as_referenced_clips() {
        let mut assets = graph_world();
        assets[3].args["states"] = serde_json::json!([
            {"name": "locomotion", "blend": {"kind": "blend1d", "parameter": "speed",
             "points": [
                 {"value": 0.0, "clip": "idle"},
                 {"value": 5.0, "clip": "run"}
             ]}}
        ]);
        assets[3].args["parameters"] = serde_json::json!([{"name": "speed"}]);
        assert!(validate_cross_references(&assets).is_ok());

        // A blend member resolving to a ghost clip still fails.
        assets[3].args["states"][0]["blend"]["points"][1]["clip"] = serde_json::json!("ghost_clip");
        let errs = err_text(&assets);
        assert!(errs.contains("ghost_clip"));
        // And the displaced 'run' clip is now unreferenced.
        assert!(errs.contains("no graph state references it"));
    }

    #[test]
    fn clips_without_graph_stay_unowned_and_pass() {
        let assets = vec![
            asset("hero", RegisteredType::SkinnedMesh, serde_json::json!({})),
            asset(
                "idle",
                RegisteredType::Animation,
                serde_json::json!({"target":"hero"}),
            ),
        ];
        assert!(validate_cross_references(&assets).is_ok());
    }

    // A third-person world: a capsuled skinned mesh, a graph declaring the
    // speed parameter, its clip, and a camera following the mesh.
    fn follow_world() -> Vec<WorldJsonlAsset> {
        vec![
            asset(
                "hero",
                RegisteredType::SkinnedMesh,
                serde_json::json!({"capsule":{"half_height":0.5,"radius":0.3}}),
            ),
            asset(
                "walk",
                RegisteredType::Animation,
                serde_json::json!({"target":"hero"}),
            ),
            asset(
                "g",
                RegisteredType::AnimationGraph,
                serde_json::json!({
                    "target":"hero",
                    "parameters":[{"name":"speed"}],
                    "states":[{"name":"walk","clip":"walk"}]
                }),
            ),
            asset(
                "cam",
                RegisteredType::Camera3D,
                serde_json::json!({"controller":{"follow":{
                    "target":"hero","speed_parameter":"speed"
                }}}),
            ),
        ]
    }

    #[test]
    fn follow_valid_references_pass() {
        assert!(validate_cross_references(&follow_world()).is_ok());
    }

    #[test]
    fn follow_missing_target_fails() {
        let mut assets = follow_world();
        assets[3].args["controller"]["follow"]["target"] = serde_json::json!("ghost");
        assert!(err_text(&assets).contains("'ghost' not found"));
    }

    #[test]
    fn follow_without_target_field_fails() {
        let mut assets = follow_world();
        assets[3].args["controller"]["follow"] = serde_json::json!({});
        assert!(err_text(&assets).contains("`controller.follow.target` is required"));
    }

    #[test]
    fn follow_target_without_capsule_fails() {
        let mut assets = follow_world();
        assets[0].args = serde_json::json!({});
        assert!(err_text(&assets).contains("has no `capsule`"));
    }

    #[test]
    fn follow_explicit_speed_parameter_must_be_declared() {
        let mut assets = follow_world();
        assets[3].args["controller"]["follow"]["speed_parameter"] = serde_json::json!("velocity");
        assert!(err_text(&assets).contains("declares the speed parameter 'velocity'"));
    }

    #[test]
    fn follow_omitted_speed_parameter_is_not_enforced() {
        // No graph at all: a direct-drive character with the defaulted
        // "speed" name still builds; the runtime warns instead.
        let mut assets = follow_world();
        assets[3].args["controller"]["follow"]
            .as_object_mut()
            .unwrap()
            .remove("speed_parameter");
        assets.remove(2); // drop the graph
        assets.remove(1); // and its clip (now unowned, which is fine)
        assert!(validate_cross_references(&assets).is_ok());
    }

    // The drift guard for the derived ref contract: EVERY reference field the
    // registry derives -- component, resource or build-only schema, at any
    // nesting depth -- is actually validated. A world holding only the
    // referencing asset with a dangling name at that path must report it.
    #[test]
    fn every_registry_ref_field_is_validated() {
        let mut checked = 0;
        for &ty in RegisteredType::all() {
            for field in ty.ref_fields() {
                let mut args = serde_json::json!({});
                let mut cursor = &mut args;
                for segment in field.path.split('.') {
                    cursor = &mut cursor[segment];
                }
                *cursor = serde_json::Value::String("ghost_ref".to_string());
                let probe = asset("probe", ty, args);
                let errs = validate_cross_references(&[probe]).unwrap_err();
                assert!(
                    errs.iter()
                        .any(|e| e.contains("ghost_ref") && e.contains(&field.path)),
                    "{}.{} (-> {:?}): dangling reference not reported; got {errs:?}",
                    ty.as_str(),
                    field.path,
                    field.targets
                );
                checked += 1;
            }
        }
        assert!(
            checked > 100,
            "only {checked} reference fields were derived"
        );
    }

    #[test]
    fn behavior_named_entities_resolve_at_any_nesting_depth() {
        let mesh = asset(
            "box_mesh",
            RegisteredType::ProceduralMesh,
            serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
        );
        let prop = asset(
            "door",
            RegisteredType::Prop,
            serde_json::json!({"mesh":"box_mesh"}),
        );
        // A `named` buried inside an if/for_each body still resolves.
        let ok = asset(
            "open",
            RegisteredType::Behavior,
            serde_json::json!({"on":"start","do":[
                {"if":{"cond":{"bool":true},"then":[{"hide":{"target":{"named":"door"}}}]}}
            ]}),
        );
        assert!(validate_cross_references(&[mesh.clone(), prop.clone(), ok]).is_ok());

        let ghost = asset(
            "open",
            RegisteredType::Behavior,
            serde_json::json!({"on":"start","do":[
                {"if":{"cond":{"bool":true},"then":[{"hide":{"target":{"named":"ghost_door"}}}]}}
            ]}),
        );
        assert!(err_text(&[mesh, prop, ghost]).contains("ghost_door"));
    }

    #[test]
    fn behavior_scene_node_does_not_double_report_its_inner_field() {
        let scene = asset("level2", RegisteredType::Scene, serde_json::json!({}));
        let ok = asset(
            "advance",
            RegisteredType::Behavior,
            serde_json::json!({"on":"start","do":[{"scene":{"scene":"level2"}}]}),
        );
        assert!(validate_cross_references(&[scene.clone(), ok]).is_ok());

        let ghost = asset(
            "advance",
            RegisteredType::Behavior,
            serde_json::json!({"on":"start","do":[{"scene":{"scene":"ghost_level"}}]}),
        );
        let errs = err_text(&[scene, ghost]);
        assert_eq!(
            errs.matches("ghost_level").count(),
            1,
            "the node verb and its inner field must not both report; got {errs:?}"
        );
    }

    #[test]
    fn behavior_enter_volume_resolves_against_trigger_volumes() {
        let volume = asset("zone", RegisteredType::TriggerVolume, serde_json::json!({}));
        let ok = asset(
            "opens",
            RegisteredType::Behavior,
            serde_json::json!({"on":{"enter":"zone"},"do":[]}),
        );
        assert!(validate_cross_references(&[volume.clone(), ok]).is_ok());

        let ghost = asset(
            "opens",
            RegisteredType::Behavior,
            serde_json::json!({"on":{"exit":"ghost_zone"},"do":[]}),
        );
        assert!(err_text(&[volume, ghost]).contains("ghost_zone"));
    }

    #[test]
    fn behavior_node_refs_are_validated() {
        let mesh = asset(
            "box_mesh",
            RegisteredType::ProceduralMesh,
            serde_json::json!({"generator":"box","half_extents":[1,1,1]}),
        );
        let prop = asset(
            "crate",
            RegisteredType::Prop,
            serde_json::json!({"mesh":"box_mesh"}),
        );
        let ok = asset(
            "spawner_rule",
            RegisteredType::Behavior,
            serde_json::json!({"do":[{"spawn":{"template":"crate"}}]}),
        );
        assert!(validate_cross_references(&[mesh.clone(), prop.clone(), ok]).is_ok());

        let ghost = asset(
            "spawner_rule",
            RegisteredType::Behavior,
            serde_json::json!({"do":[{"despawn":{"target":{"named":"ghost_prop"}}}]}),
        );
        assert!(err_text(&[mesh, prop, ghost]).contains("ghost_prop"));
    }

    #[test]
    fn all_errors_are_collected() {
        // A prop with three independent bad references should report all three.
        let assets = vec![asset(
            "broken",
            RegisteredType::Prop,
            serde_json::json!({"mesh":"no_mesh","material":"no_mat","scene":"no_scene"}),
        )];
        let errs = validate_cross_references(&assets).unwrap_err();
        assert_eq!(errs.len(), 3);
    }
}
