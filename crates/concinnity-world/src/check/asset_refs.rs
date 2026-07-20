// src/check/asset_refs.rs
//
// Per-asset cross-reference declarations for the STRUCTURED references a flat
// registry `refs:` pair cannot express: lists (Model submeshes, SceneReel
// scenes, voxel palettes), the polymorphic mesh sources, nested fields
// (Camera3D's follow controller), and required-ness (a missing mandatory
// field is an authoring error, not an absent optional). Each such asset
// implements `CrossReferenced`; the validator in `cross_reference.rs` resolves
// each `RefKind` to the matching set of asset names and detects Prop parent
// cycles. Flat references belong in the registry's `refs:` metadata instead
// (validated generically by `validate_registry_refs`); an impl here must not
// re-check a registry-declared field, or the problem reports twice.
//
// This is build-time-only authoring logic; the asset data structs it operates
// on live in concinnity-asset and their runtime `Component` impls in
// concinnity-core.

use crate::assets::{
    AnimGraph, Camera3D, InstancedProp, Joint, JointKind, Model, Prop, Reaction, SceneReel,
    VoxelChunk, VoxelWorld,
};

// The category of asset a structured name reference must resolve to.
// Reference kinds are deliberately not 1:1 with asset types: `MeshSource`
// accepts several types and `AnyAsset` accepts every declared name.
#[derive(Debug, Clone, Copy)]
pub enum RefKind {
    // Mesh, ProceduralMesh, VoxelChunk, or a mesh-kind File.
    MeshSource,
    Material,
    Scene,
    BlockType,
    SkinnedMesh,
    Animation,
    AudioClip,
    Screen,
    TriggerVolume,
    // Any declared asset, whatever its type (runtime targets like a despawned
    // entity or a spawn template are addressed by bare name).
    AnyAsset,
}

// One item produced by a referencing asset's `cross_refs`.
pub enum CrossRef {
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
pub trait CrossReferenced {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef>;
}

// Every Animation name a state's raw JSON references: its `clip`, or all of its
// blendspace members. Serves reference validation over the raw world;
// empty/missing names are skipped.
pub(crate) fn state_clip_names(state: &serde_json::Value) -> Vec<String> {
    let mut names = Vec::new();
    let mut push = |v: Option<&serde_json::Value>| {
        if let Some(clip) = v.and_then(|v| v.as_str())
            && !clip.is_empty()
        {
            names.push(clip.to_string());
        }
    };
    push(state.get("clip"));
    if let Some(blend) = state.get("blend") {
        for point in blend
            .get("points")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[])
        {
            push(point.get("clip"));
        }
        for row in blend
            .get("rows")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[])
        {
            for cell in row.as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                push(Some(cell));
            }
        }
    }
    names
}

impl CrossReferenced for AnimGraph {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();
        match args.get("target").and_then(|v| v.as_str()).unwrap_or("") {
            "" => refs.push(CrossRef::Issue(format!(
                "AnimGraph '{name}': `target` field is required (the SkinnedMesh to animate)"
            ))),
            target => refs.push(CrossRef::Resolve {
                kind: RefKind::SkinnedMesh,
                target: target.to_string(),
                error: format!("AnimGraph '{name}': target SkinnedMesh '{target}' not found"),
            }),
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
                    "AnimGraph '{name}': {label} names no Animation (set `clip`, or `blend` \
                     members)"
                )));
            }
            for clip in clips {
                refs.push(CrossRef::Resolve {
                    error: format!("AnimGraph '{name}': {label} clip '{clip}' not found"),
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
        match follow.get("target").and_then(|v| v.as_str()).unwrap_or("") {
            "" => vec![CrossRef::Issue(format!(
                "Camera3D '{name}': `controller.follow.target` is required (the SkinnedMesh to follow)"
            ))],
            target => vec![CrossRef::Resolve {
                kind: RefKind::SkinnedMesh,
                target: target.to_string(),
                error: format!("Camera3D '{name}': follow target SkinnedMesh '{target}' not found"),
            }],
        }
    }
}

impl CrossReferenced for Prop {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        // The flat references (model, material, texture, scene, parent) are
        // registry-declared and resolved generically; only the polymorphic
        // mesh source stays here. A Model takes precedence over a Mesh, so
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
                "Prop '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that name",
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
                            "Model '{}': submesh[{}] mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that name",
                            name, i, sub_mesh
                        ),
                    });
                }

                let sub_mat = sub.get("material").and_then(|v| v.as_str()).unwrap_or("");
                if !sub_mat.is_empty() {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::Material,
                        target: sub_mat.to_string(),
                        error: format!(
                            "Model '{}': submesh[{}] material '{}' not found, add a Material asset with that name",
                            name, i, sub_mat
                        ),
                    });
                }
            }
        }

        refs
    }
}

impl CrossReferenced for SceneReel {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();

        if let Some(entries) = args.get("scenes").and_then(|v| v.as_array()) {
            if entries.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "SceneReel '{}': scenes list is empty",
                    name
                )));
            }
            for (i, entry) in entries.iter().enumerate() {
                let scene_ref = entry.as_str().unwrap_or("");
                if scene_ref.is_empty() {
                    refs.push(CrossRef::Issue(format!(
                        "SceneReel '{}': scenes[{}] is not a valid scene name string",
                        name, i
                    )));
                } else {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::Scene,
                        target: scene_ref.to_string(),
                        error: format!(
                            "SceneReel '{}': scenes[{}] references unknown scene '{}', add a Scene asset with that name",
                            name, i, scene_ref
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
        // The flat references (material, texture) are registry-declared and
        // resolved generically; the mesh stays here for its required-ness and
        // its polymorphic target set.
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
                "InstancedProp '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, VoxelChunk, or File (obj) asset with that name",
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
        for (i, entry) in palette.iter().enumerate() {
            let bt_name = entry.as_str().unwrap_or("");
            if bt_name.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "VoxelChunk '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            } else {
                refs.push(CrossRef::Resolve {
                    kind: RefKind::BlockType,
                    target: bt_name.to_string(),
                    error: format!(
                        "VoxelChunk '{}': palette[{}] BlockType '{}' not found, add a BlockType asset with that name",
                        name, i, bt_name
                    ),
                });
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
        for (i, entry) in palette.iter().enumerate() {
            let bt_name = entry.as_str().unwrap_or("");
            if bt_name.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "VoxelWorld '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            } else {
                refs.push(CrossRef::Resolve {
                    kind: RefKind::BlockType,
                    target: bt_name.to_string(),
                    error: format!(
                        "VoxelWorld '{}': palette[{}] BlockType '{}' not found, add a BlockType asset with that name",
                        name, i, bt_name
                    ),
                });
            }
        }

        refs
    }
}

impl CrossReferenced for Joint {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        // body_a / body_b resolution is registry-declared and generic; only
        // the kind check and body_a's required-ness stay here.
        let arg_str = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let kind = arg_str("kind");
        if !kind.is_empty() && JointKind::from_str_norm(kind).is_none() {
            refs.push(CrossRef::Issue(format!(
                "Joint '{name}': unknown kind '{kind}' (expected one of fixed | revolute | spherical | prismatic)"
            )));
        }

        if arg_str("body_a").is_empty() {
            refs.push(CrossRef::Issue(format!(
                "Joint '{name}': `body_a` is required, name of a Prop with a collider"
            )));
        }

        refs
    }
}

impl CrossReferenced for Reaction {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();

        // One action field: required-ness plus resolution against `kind`'s
        // name-set. An integer value is an already-resolved id and passes; a
        // missing or empty field is an authoring error when required.
        let field = |action: &serde_json::Value,
                     verb: &str,
                     key: &str,
                     kind: RefKind,
                     required: bool,
                     refs: &mut Vec<CrossRef>| {
            match action.get(key) {
                Some(serde_json::Value::String(target)) if !target.is_empty() => {
                    refs.push(CrossRef::Resolve {
                        kind,
                        target: target.clone(),
                        error: format!("Reaction '{name}': {verb} {key} '{target}' not found"),
                    });
                }
                None | Some(serde_json::Value::String(_)) | Some(serde_json::Value::Null)
                    if required =>
                {
                    refs.push(CrossRef::Issue(format!(
                        "Reaction '{name}': `{verb}` action requires `{key}`"
                    )));
                }
                _ => {}
            }
        };

        let actions = args
            .get("actions")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        for action in actions {
            if let Some(spawn) = action.get("spawn") {
                field(
                    spawn,
                    "spawn",
                    "template",
                    RefKind::AnyAsset,
                    true,
                    &mut refs,
                );
            }
            if let Some(despawn) = action.get("despawn") {
                field(
                    despawn,
                    "despawn",
                    "target",
                    RefKind::AnyAsset,
                    true,
                    &mut refs,
                );
            }
            if let Some(reparent) = action.get("reparent") {
                field(
                    reparent,
                    "reparent",
                    "child",
                    RefKind::AnyAsset,
                    true,
                    &mut refs,
                );
                field(
                    reparent,
                    "reparent",
                    "parent",
                    RefKind::AnyAsset,
                    false,
                    &mut refs,
                );
            }
            if let Some(sound) = action.get("sound") {
                field(sound, "sound", "clip", RefKind::AudioClip, true, &mut refs);
            }
            if let Some(scene) = action.get("scene") {
                field(scene, "scene", "scene", RefKind::Scene, true, &mut refs);
            }
            if let Some(screen) = action.get("screen") {
                field(screen, "screen", "screen", RefKind::Screen, true, &mut refs);
            }
            if let Some(set) = action.get("set")
                && set
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .is_empty()
            {
                refs.push(CrossRef::Issue(format!(
                    "Reaction '{name}': `set` action requires a variable `name`"
                )));
            }
        }

        if let Some(source) = args.get("on") {
            // A `variable` source watching an unnamed variable never fires.
            if let Some(var) = source.get("variable")
                && var.as_str().unwrap_or("").is_empty()
            {
                refs.push(CrossRef::Issue(format!(
                    "Reaction '{name}': `variable` source requires a variable name"
                )));
            }
            // An enter/exit source must name a declared TriggerVolume.
            for verb in ["enter", "exit"] {
                match source.get(verb) {
                    Some(serde_json::Value::String(target)) if !target.is_empty() => {
                        refs.push(CrossRef::Resolve {
                            kind: RefKind::TriggerVolume,
                            target: target.clone(),
                            error: format!(
                                "Reaction '{name}': `{verb}` volume '{target}' not found, \
                                 add a TriggerVolume asset with that name"
                            ),
                        });
                    }
                    Some(serde_json::Value::String(_)) | Some(serde_json::Value::Null) => {
                        refs.push(CrossRef::Issue(format!(
                            "Reaction '{name}': `{verb}` source requires a TriggerVolume name"
                        )));
                    }
                    _ => {}
                }
            }
        }

        refs
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
        // The flat material ref is registry-declared, so only the palette list
        // is extracted here: an empty entry is an Issue, "grass" resolves.
        let refs = VoxelWorld::cross_refs("ow", &json!({"palette": ["", "grass"]}));
        assert_eq!(tally(&refs), (1, 1));
        assert!(resolves_to(&refs, RefKind::BlockType, "grass"));

        let chunk = VoxelChunk::cross_refs("c", &json!({"palette": ["stone", ""]}));
        assert_eq!(tally(&chunk), (1, 1));
        assert!(resolves_to(&chunk, RefKind::BlockType, "stone"));
    }

    #[test]
    fn prop_cross_refs_model_takes_precedence_over_mesh() {
        // The flat refs (model, material, texture, parent) are
        // registry-declared, so only the mesh source is extracted here, and
        // only when no model claims the prop.
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
        // submesh0 -> mesh + material Resolves; submesh1 -> missing-mesh Issue.
        assert_eq!(tally(&refs), (2, 1));
        assert!(resolves_to(&refs, RefKind::MeshSource, "m0"));
        assert!(resolves_to(&refs, RefKind::Material, "mat0"));
    }

    #[test]
    fn scene_reel_cross_refs_scenes_list() {
        // Empty scenes list -> one Issue.
        assert_eq!(
            tally(&SceneReel::cross_refs("r", &json!({"scenes": []}))),
            (0, 1)
        );
        let refs = SceneReel::cross_refs("r", &json!({"scenes": ["a", ""]}));
        assert_eq!(tally(&refs), (1, 1));
        assert!(resolves_to(&refs, RefKind::Scene, "a"));
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
    fn anim_graph_cross_refs_cover_target_and_clips() {
        let refs = AnimGraph::cross_refs("g", &graph_json());
        // One target resolve + two clip resolves.
        assert_eq!(refs.len(), 3);
        assert!(refs.iter().all(|r| matches!(r, CrossRef::Resolve { .. })));
    }

    #[test]
    fn anim_graph_cross_refs_flag_missing_target_and_clip() {
        let refs = AnimGraph::cross_refs("g", &json!({"states":[{"name":"idle"}]}));
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
        let refs = AnimGraph::cross_refs("g", &blend1d_graph_json());
        // One target resolve + three point-clip resolves.
        assert_eq!(refs.len(), 4);
        assert!(refs.iter().all(|r| matches!(r, CrossRef::Resolve { .. })));

        let refs = AnimGraph::cross_refs("g", &blend2d_graph_json());
        // One target resolve + four grid-cell resolves.
        assert_eq!(refs.len(), 5);
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
