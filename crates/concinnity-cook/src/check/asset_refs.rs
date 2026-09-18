// Per-asset cross-reference declarations for what a field type cannot state:
// the polymorphic mesh sources, references inside enum variants (a behavior's
// nodes, a blendspace's members), and required-ness (a missing mandatory field
// is an authoring error, not an absent optional). Each such asset implements
// `CrossReferenced`; the validator in `cross_reference.rs` resolves each
// `RefKind` to the matching set of asset names and detects Prop parent cycles.
// A field typed `Ref<T>` or with a resource handle is validated generically
// from the registry's derived table instead (`validate_registry_refs`); an impl
// here must not re-check one, or the problem reports twice.
//
// This is build-time-only authoring logic; the asset data structs it operates
// on, and their runtime `Component` impls, live in
// concinnity-core.

use concinnity_core::components::{
    AnimationGraph, Behavior, Camera3D, InstancedProp, Model, PhysicsJoint, PhysicsJointKind, Prop,
    VoxelChunk, VoxelWorld,
};

// The category of asset a structured name reference must resolve to.
// Reference kinds are deliberately not 1:1 with asset types: `MeshSource`
// accepts several types and `AnyAsset` accepts every declared name.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RefKind {
    // Mesh, ProceduralMesh, VoxelChunk, or a mesh-kind File.
    MeshSource,
    Scene,
    Animation,
    AudioClip,
    Screen,
    TriggerVolume,
    // Any declared asset, whatever its type (runtime targets like a despawned
    // entity or a spawn template are addressed by bare name).
    AnyAsset,
}

// One item produced by a referencing asset's `cross_refs`.
pub(crate) enum CrossRef {
    // `target` must resolve to an asset in `kind`'s name-set; if it does not,
    // `error` is collected verbatim.
    Resolve {
        kind: RefKind,
        target: String,
        error: String,
    },
    // A problem the asset detected on its own: a missing required field, a
    // malformed array entry, an empty list. Collected verbatim.
    Issue(String),
}

// Implemented by every asset type that references other assets by name.
// `cross_refs` extracts those references (and any structural problems) from
// the asset's args; the resolver resolves each `Resolve` against the world.
pub(crate) trait CrossReferenced {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef>;
}

// Every Animation name a state's raw JSON references: its `clip`, or all of its
// blendspace members. Serves reference validation over the raw world;
// empty/missing names are skipped.
pub(crate) fn state_clip_names(state: &serde_json::Value) -> Vec<String> {
    let mut names: Vec<String> = state
        .get("clip")
        .and_then(|v| v.as_str())
        .filter(|clip| !clip.is_empty())
        .map(str::to_string)
        .into_iter()
        .collect();
    if let Some(blend) = state.get("blend") {
        names.extend(blend_clip_names(blend));
    }
    names
}

// The Animation names a blendspace's members reference: its 1D `points` or its
// 2D `rows`. Empty/missing names are skipped.
fn blend_clip_names(blend: &serde_json::Value) -> Vec<String> {
    let points = blend
        .get("points")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|point| point.get("clip"));
    let cells = blend
        .get("rows")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|row| row.as_array())
        .flatten();
    points
        .chain(cells)
        .filter_map(|v| v.as_str())
        .filter(|clip| !clip.is_empty())
        .map(str::to_string)
        .collect()
}

// A required reference left unset: absent, null, or an empty string. An
// integer is an already-resolved id, so it counts as set.
fn is_blank(value: Option<&serde_json::Value>) -> bool {
    match value {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::String(s)) => s.is_empty(),
        Some(_) => false,
    }
}

impl CrossReferenced for AnimationGraph {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        // `target` and each state's `clip` resolve generically; the target's
        // required-ness and the blendspace members, which sit in an enum
        // variant, stay here.
        let mut refs = Vec::new();
        if is_blank(args.get("target")) {
            refs.push(CrossRef::Issue(format!(
                "AnimationGraph '{name}': `target` field is required (the SkinnedMesh to animate)"
            )));
        }
        let states = args
            .get("states")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        for (i, state) in states.iter().enumerate() {
            let state_name = state.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let label = if state_name.is_empty() {
                format!("state #{i}")
            } else {
                format!("state '{state_name}'")
            };
            let clips = state_clip_names(state);
            if clips.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "AnimationGraph '{name}': {label} names no Animation (set `clip`, or `blend` \
                     members)"
                )));
            }
            for clip in state.get("blend").map(blend_clip_names).unwrap_or_default() {
                refs.push(CrossRef::Resolve {
                    error: format!("AnimationGraph '{name}': {label} clip '{clip}' not found"),
                    kind: RefKind::Animation,
                    target: clip,
                });
            }
        }
        refs
    }
}

impl CrossReferenced for Camera3D {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let Some(follow) = args
            .get("controller")
            .and_then(|c| c.get("follow"))
            .filter(|f| !f.is_null())
        else {
            return Vec::new();
        };
        // The target resolves generically; only its required-ness is here.
        if is_blank(follow.get("target")) {
            return vec![CrossRef::Issue(format!(
                "Camera3D '{name}': `controller.follow.target` is required (the SkinnedMesh to follow)"
            ))];
        }
        Vec::new()
    }
}

impl CrossReferenced for Prop {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        // The typed references (model, material, scene, parent) resolve
        // through the derived table; only the polymorphic mesh source stays
        // here. A Model takes precedence over a Mesh, so
        // the mesh is checked only when no model is set.
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        if !arg("model").is_empty() {
            return Vec::new();
        }
        let mesh_ref = arg("mesh");
        if mesh_ref.is_empty() {
            return Vec::new();
        }
        vec![CrossRef::Resolve {
            kind: RefKind::MeshSource,
            target: mesh_ref.to_string(),
            error: format!(
                "Prop '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that `$id`",
                name, mesh_ref
            ),
        }]
    }
}

impl CrossReferenced for Model {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();

        if let Some(meshes) = args.get("meshes").and_then(|v| v.as_array()) {
            for (i, sub) in meshes.iter().enumerate() {
                let sub_mesh = sub.get("mesh").and_then(|v| v.as_str()).unwrap_or("");
                if sub_mesh.is_empty() {
                    refs.push(CrossRef::Issue(format!(
                        "Model '{}': submesh[{}] is missing a 'mesh' field",
                        name, i
                    )));
                } else {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::MeshSource,
                        target: sub_mesh.to_string(),
                        error: format!(
                            "Model '{}': submesh[{}] mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that `$id`",
                            name, i, sub_mesh
                        ),
                    });
                }
            }
        }

        refs
    }
}

impl CrossReferenced for InstancedProp {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        // The material resolves through the derived table; the mesh stays
        // here for its required-ness and its polymorphic target set.
        let mesh_ref = args.get("mesh").and_then(|v| v.as_str()).unwrap_or("");
        if mesh_ref.is_empty() {
            return vec![CrossRef::Issue(format!(
                "InstancedProp '{}': `mesh` field is required",
                name
            ))];
        }
        vec![CrossRef::Resolve {
            kind: RefKind::MeshSource,
            target: mesh_ref.to_string(),
            error: format!(
                "InstancedProp '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, VoxelChunk, or File (obj) asset with that `$id`",
                name, mesh_ref
            ),
        }]
    }
}

impl CrossReferenced for VoxelChunk {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();

        let palette = args
            .get("palette")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        // Each named entry resolves generically; an entry that names nothing
        // is caught here.
        for (i, entry) in palette.iter().enumerate() {
            if entry.as_str().unwrap_or("").is_empty() && !entry.is_u64() {
                refs.push(CrossRef::Issue(format!(
                    "VoxelChunk '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            }
        }

        refs
    }
}

impl CrossReferenced for VoxelWorld {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();

        let palette = args
            .get("palette")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        // Each named entry resolves generically; an entry that names nothing
        // is caught here.
        for (i, entry) in palette.iter().enumerate() {
            if entry.as_str().unwrap_or("").is_empty() && !entry.is_u64() {
                refs.push(CrossRef::Issue(format!(
                    "VoxelWorld '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            }
        }

        refs
    }
}

impl CrossReferenced for PhysicsJoint {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        // body_a / body_b resolve through the derived table; only the kind
        // check and body_a's required-ness stay here.
        let arg_str = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let kind = arg_str("kind");
        if !kind.is_empty() && PhysicsJointKind::from_str_norm(kind).is_none() {
            refs.push(CrossRef::Issue(format!(
                "PhysicsJoint '{name}': unknown kind '{kind}' (expected one of {})",
                PhysicsJointKind::ACCEPTED.join(" | ")
            )));
        }

        if arg_str("body_a").is_empty() {
            refs.push(CrossRef::Issue(format!(
                "PhysicsJoint '{name}': `body_a` is required, name of a Prop with a collider"
            )));
        }

        refs
    }
}

impl CrossReferenced for Behavior {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();
        walk_behavior_nodes(args.get("do"), name, &mut refs);

        if let Some(source) = args.get("on") {
            // A `variable` source watching an unnamed variable never fires.
            if let Some(var) = source.get("variable")
                && var.as_str().unwrap_or("").is_empty()
            {
                refs.push(CrossRef::Issue(format!(
                    "Behavior '{name}': `variable` source requires a variable name"
                )));
            }
            for verb in ["enter", "exit"] {
                match source.get(verb) {
                    Some(serde_json::Value::String(target)) if !target.is_empty() => {
                        refs.push(CrossRef::Resolve {
                            kind: RefKind::TriggerVolume,
                            target: target.clone(),
                            error: format!(
                                "Behavior '{name}': `{verb}` volume '{target}' not found, \
                                 add a TriggerVolume asset with that `$id`"
                            ),
                        });
                    }
                    Some(serde_json::Value::String(_)) | Some(serde_json::Value::Null) => {
                        refs.push(CrossRef::Issue(format!(
                            "Behavior '{name}': `{verb}` source requires a TriggerVolume name"
                        )));
                    }
                    _ => {}
                }
            }
            match source.get("interact") {
                Some(serde_json::Value::String(target)) if !target.is_empty() => {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::AnyAsset,
                        target: target.clone(),
                        error: format!("Behavior '{name}': `interact` target '{target}' not found"),
                    });
                }
                Some(serde_json::Value::String(_)) | Some(serde_json::Value::Null) => {
                    refs.push(CrossRef::Issue(format!(
                        "Behavior '{name}': `interact` source requires an entity name"
                    )));
                }
                _ => {}
            }
        }

        refs
    }
}

// Every asset name a behavior body references, at any nesting depth. Nodes and
// expressions are both single-key objects, so one descent covers both: `named`
// expressions anywhere, plus the four nodes carrying an asset field.
fn walk_behavior_nodes(value: Option<&serde_json::Value>, name: &str, refs: &mut Vec<CrossRef>) {
    // One node field: required-ness plus resolution against `kind`'s name-set.
    // An integer value is an already-resolved id and passes.
    fn field(
        node: &serde_json::Value,
        verb: &str,
        key: &str,
        kind: RefKind,
        name: &str,
        refs: &mut Vec<CrossRef>,
    ) {
        match node.get(key) {
            Some(serde_json::Value::String(target)) if !target.is_empty() => {
                refs.push(CrossRef::Resolve {
                    kind,
                    target: target.clone(),
                    error: format!("Behavior '{name}': {verb} {key} '{target}' not found"),
                });
            }
            None | Some(serde_json::Value::String(_)) | Some(serde_json::Value::Null) => {
                refs.push(CrossRef::Issue(format!(
                    "Behavior '{name}': `{verb}` node requires `{key}`"
                )));
            }
            _ => {}
        }
    }

    let Some(value) = value else { return };
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                walk_behavior_nodes(Some(item), name, refs);
            }
        }
        serde_json::Value::Object(map) => {
            for (key, body) in map {
                if key == "named" {
                    match body {
                        serde_json::Value::String(target) if !target.is_empty() => {
                            refs.push(CrossRef::Resolve {
                                kind: RefKind::AnyAsset,
                                target: target.clone(),
                                error: format!(
                                    "Behavior '{name}': `named` entity '{target}' not found"
                                ),
                            });
                        }
                        serde_json::Value::String(_) | serde_json::Value::Null => {
                            refs.push(CrossRef::Issue(format!(
                                "Behavior '{name}': `named` requires an entity name"
                            )));
                        }
                        _ => {}
                    }
                    continue;
                }
                // Only an object body is a node; the same words appear as
                // inner field names carrying plain strings.
                if body.is_object() {
                    match key.as_str() {
                        "spawn" => field(body, "spawn", "template", RefKind::AnyAsset, name, refs),
                        "sound" => field(body, "sound", "clip", RefKind::AudioClip, name, refs),
                        "scene" => field(body, "scene", "scene", RefKind::Scene, name, refs),
                        "screen" => field(body, "screen", "screen", RefKind::Screen, name, refs),
                        _ => {}
                    }
                }
                walk_behavior_nodes(Some(body), name, refs);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // (resolve count, issue count) in a cross-ref list. CrossRef has no
    // PartialEq, so tests match on the variant rather than compare values.
    fn tally(refs: &[CrossRef]) -> (usize, usize) {
        let mut resolves = 0;
        let mut issues = 0;
        for r in refs {
            match r {
                CrossRef::Resolve { .. } => resolves += 1,
                CrossRef::Issue(_) => issues += 1,
            }
        }
        (resolves, issues)
    }

    // Whether the list contains a Resolve to `target` of the given kind.
    fn resolves_to(refs: &[CrossRef], kind: RefKind, target: &str) -> bool {
        refs.iter().any(|r| match r {
            CrossRef::Resolve {
                kind: k, target: t, ..
            } => std::mem::discriminant(k) == std::mem::discriminant(&kind) && t == target,
            CrossRef::Issue(_) => false,
        })
    }

    #[test]
    fn voxel_world_and_chunk_cross_refs_palette() {
        // Named entries resolve through the derived table, so only an entry
        // naming nothing is caught here; a resolved id is not one.
        let refs = VoxelWorld::cross_refs("ow", &json!({"palette": ["", "grass", 3]}));
        assert_eq!(tally(&refs), (0, 1));

        let chunk = VoxelChunk::cross_refs("c", &json!({"palette": ["stone", null]}));
        assert_eq!(tally(&chunk), (0, 1));
    }

    #[test]
    fn prop_cross_refs_model_takes_precedence_over_mesh() {
        // The typed refs (model, material, parent) resolve through the
        // derived table, so only the mesh source is extracted here, and only
        // when no model claims the prop.
        let refs = Prop::cross_refs("p", &json!({"model": "m", "mesh": "mesh_skipped"}));
        assert_eq!(tally(&refs), (0, 0));
        // With no model, the mesh path is used instead.
        let mesh_only = Prop::cross_refs("p", &json!({"mesh": "only_mesh"}));
        assert!(resolves_to(&mesh_only, RefKind::MeshSource, "only_mesh"));
    }

    #[test]
    fn model_cross_refs_submeshes_and_missing_field() {
        let refs = Model::cross_refs(
            "mdl",
            &json!({"meshes": [{"mesh": "m0", "material": "mat0"}, {}]}),
        );
        // submesh0 -> a mesh Resolve (its material resolves through the
        // derived table); submesh1 -> missing-mesh Issue.
        assert_eq!(tally(&refs), (1, 1));
        assert!(resolves_to(&refs, RefKind::MeshSource, "m0"));
    }

    fn graph_json() -> serde_json::Value {
        json!({
            "target": "hero",
            "parameters": [{"name": "speed", "default": 0.5}],
            "initial": "idle",
            "states": [
                {"name": "idle", "clip": "hero_idle"},
                {"name": "run", "clip": "hero_run", "rate": 1.5, "loop_override": false}
            ]
        })
    }

    fn blend1d_graph_json() -> serde_json::Value {
        json!({
            "target": "hero",
            "parameters": [{"name": "speed", "default": 0.0}],
            "states": [
                {"name": "locomotion", "blend": {"kind": "blend1d", "parameter": "speed",
                 "sync": true,
                 "points": [
                     {"value": 0.0, "clip": "idle"},
                     {"value": 1.6, "clip": "walk"},
                     {"value": 5.0, "clip": "run"}
                 ]}}
            ]
        })
    }

    fn blend2d_graph_json() -> serde_json::Value {
        json!({
            "target": "hero",
            "parameters": [{"name": "speed"}, {"name": "strafe"}],
            "states": [
                {"name": "locomotion", "blend": {"kind": "blend2d",
                 "parameter_x": "speed", "parameter_y": "strafe",
                 "x_values": [0.0, 5.0], "y_values": [-1.0, 1.0],
                 "rows": [["run_l", "run_l"], ["run_r", "run_r"]]}}
            ]
        })
    }

    #[test]
    fn anim_graph_cross_refs_leave_target_and_state_clips_to_the_table() {
        // Both are typed reference fields, so the derived table resolves them.
        assert!(AnimationGraph::cross_refs("g", &graph_json()).is_empty());
    }

    #[test]
    fn anim_graph_cross_refs_flag_missing_target_and_clip() {
        let refs = AnimationGraph::cross_refs("g", &json!({"states":[{"name":"idle"}]}));
        let issues: Vec<_> = refs
            .iter()
            .filter_map(|r| match r {
                CrossRef::Issue(msg) => Some(msg.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(issues.len(), 2);
        assert!(issues[0].contains("target"));
        assert!(issues[1].contains("clip"));
    }

    #[test]
    fn anim_graph_cross_refs_cover_blend_members() {
        // Members sit in an enum variant, out of the table's reach.
        let refs = AnimationGraph::cross_refs("g", &blend1d_graph_json());
        assert_eq!(refs.len(), 3);
        assert!(refs.iter().all(|r| matches!(r, CrossRef::Resolve { .. })));
        assert!(resolves_to(&refs, RefKind::Animation, "walk"));

        let refs = AnimationGraph::cross_refs("g", &blend2d_graph_json());
        assert_eq!(refs.len(), 4);
    }

    #[test]
    fn a_follow_camera_needs_a_target_it_leaves_to_the_table() {
        let issues = |target: serde_json::Value| {
            Camera3D::cross_refs(
                "cam",
                &json!({"controller": {"follow": {"target": target}}}),
            )
        };
        assert_eq!(tally(&issues(json!(""))), (0, 1));
        assert_eq!(tally(&issues(json!(null))), (0, 1));
        assert!(issues(json!("hero")).is_empty());
        assert!(issues(json!(4)).is_empty());
        assert!(Camera3D::cross_refs("cam", &json!({"controller": {}})).is_empty());
    }

    #[test]
    fn state_clip_names_walks_clip_points_and_rows() {
        let names = state_clip_names(&json!({"clip":"solo"}));
        assert_eq!(names, vec!["solo"]);
        let names = state_clip_names(&blend1d_graph_json()["states"][0]);
        assert_eq!(names, vec!["idle", "walk", "run"]);
        let names = state_clip_names(&blend2d_graph_json()["states"][0]);
        assert_eq!(names, vec!["run_l", "run_l", "run_r", "run_r"]);
        assert!(state_clip_names(&json!({})).is_empty());
    }
}
