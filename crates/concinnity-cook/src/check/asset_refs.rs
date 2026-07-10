// src/check/asset_refs.rs
//
// Per-asset cross-reference declarations. Every referencing asset (Prop ->
// Mesh, Material -> Texture, etc.) declares its own references by implementing
// `CrossReferenced`. The validator in `cross_reference.rs` consumes these
// declarations: it resolves each `RefKind` to the matching set of asset names
// and detects Prop parent cycles. Adding a new referencing asset means writing
// one impl here plus one dispatch arm in `cross_refs_for`.
//
// This is build-time-only authoring logic; the asset data structs it operates
// on live in concinnity-asset and their runtime `Component` impls in
// concinnity-core.

use crate::assets::{
    AnimGraph, Camera3D, DebugHud, Decal, FpsCounter, InstancedProp, Joint, JointKind, Material,
    Model, ParticleEmitter, Prop, Scene, SceneReel, StatHud, VoxelChunk, VoxelWorld,
};

// The category of asset a name reference must resolve to. Reference kinds are
// deliberately not 1:1 with asset types: `MeshSource` accepts several types
// and `CameraShot` resolves to `Camera3D` names.
#[derive(Debug, Clone, Copy)]
pub enum RefKind {
    // Mesh, ProceduralMesh, VoxelChunk, or a mesh-kind File.
    MeshSource,
    Texture,
    Material,
    Model,
    Prop,
    Scene,
    // A camera shot target, resolves to declared `Camera3D` names.
    CameraShot,
    BlockType,
    TextLabel,
    SkinnedMesh,
    Animation,
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

// Shared extractor for HUD-style assets whose args hold optional TextLabel
// references: one `Resolve` per non-empty label field.
pub fn label_refs(
    asset_type: &str,
    name: &str,
    args: &serde_json::Value,
    fields: &[&str],
) -> Vec<CrossRef> {
    let mut refs = Vec::new();
    for field in fields {
        let target = args.get(field).and_then(|v| v.as_str()).unwrap_or("");
        if target.is_empty() {
            continue;
        }
        refs.push(CrossRef::Resolve {
            kind: RefKind::TextLabel,
            target: target.to_string(),
            error: format!(
                "{} '{}': {} '{}' not found, add a TextLabel asset with that name",
                asset_type, name, field, target
            ),
        });
    }
    refs
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
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        // A Model takes precedence over a Mesh; only the one in effect is checked.
        let model_ref = arg("model");
        let mesh_ref = arg("mesh");
        if !model_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Model,
                target: model_ref.to_string(),
                error: format!(
                    "Prop '{}': model '{}' not found, add a Model asset with that name",
                    name, model_ref
                ),
            });
        } else if !mesh_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::MeshSource,
                target: mesh_ref.to_string(),
                error: format!(
                    "Prop '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that name",
                    name, mesh_ref
                ),
            });
        }

        let mat_ref = arg("material");
        if !mat_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Material,
                target: mat_ref.to_string(),
                error: format!(
                    "Prop '{}': material '{}' not found, add a Material asset with that name",
                    name, mat_ref
                ),
            });
        }

        let tex_ref = arg("texture");
        if !tex_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex_ref.to_string(),
                error: format!(
                    "Prop '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex_ref
                ),
            });
        }

        let parent_ref = arg("parent");
        if !parent_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Prop,
                target: parent_ref.to_string(),
                error: format!(
                    "Prop '{}': parent '{}' not found, add a Prop asset with that name",
                    name, parent_ref
                ),
            });
        }

        refs
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

impl CrossReferenced for Scene {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let mut refs = Vec::new();

        let shot_ref = args
            .get("camera_shot")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !shot_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::CameraShot,
                target: shot_ref.to_string(),
                error: format!(
                    "Scene '{}': camera_shot '{}' not found, add a CameraShot or Camera3D asset with that name",
                    name, shot_ref
                ),
            });
        }

        refs
    }
}

impl CrossReferenced for InstancedProp {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let mesh_ref = arg("mesh");
        if mesh_ref.is_empty() {
            refs.push(CrossRef::Issue(format!(
                "InstancedProp '{}': `mesh` field is required",
                name
            )));
        } else {
            refs.push(CrossRef::Resolve {
                kind: RefKind::MeshSource,
                target: mesh_ref.to_string(),
                error: format!(
                    "InstancedProp '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, VoxelChunk, or File (obj) asset with that name",
                    name, mesh_ref
                ),
            });
        }

        let mat_ref = arg("material");
        if !mat_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Material,
                target: mat_ref.to_string(),
                error: format!(
                    "InstancedProp '{}': material '{}' not found, add a Material asset with that name",
                    name, mat_ref
                ),
            });
        }

        let tex_ref = arg("texture");
        if !tex_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex_ref.to_string(),
                error: format!(
                    "InstancedProp '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex_ref
                ),
            });
        }

        refs
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

        if let Some(mat) = args.get("material").and_then(|v| v.as_str())
            && !mat.is_empty()
        {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Material,
                target: mat.to_string(),
                error: format!(
                    "VoxelWorld '{}': material '{}' not found, add a Material asset with that name",
                    name, mat
                ),
            });
        }

        refs
    }
}

impl CrossReferenced for Material {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let albedo = arg("albedo");
        if !albedo.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: albedo.to_string(),
                error: format!(
                    "Material '{}': albedo texture '{}' not found, add a Texture asset with that name",
                    name, albedo
                ),
            });
        }

        let normal_map = arg("normal_map");
        if !normal_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: normal_map.to_string(),
                error: format!(
                    "Material '{}': normal_map texture '{}' not found, add a Texture asset with that name",
                    name, normal_map
                ),
            });
        }

        let emissive_map = arg("emissive_map");
        if !emissive_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: emissive_map.to_string(),
                error: format!(
                    "Material '{}': emissive_map texture '{}' not found, add a Texture asset with that name",
                    name, emissive_map
                ),
            });
        }

        let orm_map = arg("orm_map");
        if !orm_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: orm_map.to_string(),
                error: format!(
                    "Material '{}': orm_map texture '{}' not found, add a Texture asset with that name",
                    name, orm_map
                ),
            });
        }

        let albedo_secondary = arg("albedo_secondary");
        if !albedo_secondary.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: albedo_secondary.to_string(),
                error: format!(
                    "Material '{}': albedo_secondary texture '{}' not found, add a Texture asset with that name",
                    name, albedo_secondary
                ),
            });
        }

        let normal_secondary = arg("normal_secondary");
        if !normal_secondary.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: normal_secondary.to_string(),
                error: format!(
                    "Material '{}': normal_secondary texture '{}' not found, add a Texture asset with that name",
                    name, normal_secondary
                ),
            });
        }

        refs
    }
}

impl CrossReferenced for Decal {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();
        let tex = arg("texture");
        if !tex.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex.to_string(),
                error: format!(
                    "Decal '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex
                ),
            });
        }
        refs
    }
}

impl CrossReferenced for ParticleEmitter {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();
        let tex = arg("texture");
        if !tex.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex.to_string(),
                error: format!(
                    "ParticleEmitter '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex
                ),
            });
        }
        refs
    }
}

impl CrossReferenced for Joint {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        let arg_str = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let kind = arg_str("kind");
        if !kind.is_empty() && JointKind::from_str_norm(kind).is_none() {
            refs.push(CrossRef::Issue(format!(
                "Joint '{name}': unknown kind '{kind}' (expected one of fixed | revolute | spherical | prismatic)"
            )));
        }

        let body_a = arg_str("body_a");
        if body_a.is_empty() {
            refs.push(CrossRef::Issue(format!(
                "Joint '{name}': `body_a` is required, name of a Prop with a collider"
            )));
        } else {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Prop,
                target: body_a.to_string(),
                error: format!("Joint '{name}': body_a '{body_a}' is not a declared Prop"),
            });
        }

        let body_b = arg_str("body_b");
        if !body_b.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Prop,
                target: body_b.to_string(),
                error: format!("Joint '{name}': body_b '{body_b}' is not a declared Prop"),
            });
        }

        refs
    }
}

impl CrossReferenced for StatHud {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        label_refs(
            "StatHud",
            name,
            args,
            &["fps_label", "vram_label", "ev_label", "edr_label"],
        )
    }
}

impl CrossReferenced for DebugHud {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        label_refs(
            "DebugHud",
            name,
            args,
            &["passes_label", "mouse_label", "camera_label"],
        )
    }
}

impl CrossReferenced for FpsCounter {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef> {
        label_refs("FpsCounter", name, args, &["label"])
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
    fn material_cross_refs_cover_all_texture_slots() {
        let refs = Material::cross_refs(
            "mat",
            &json!({
                "albedo": "a",
                "normal_map": "n",
                "emissive_map": "e",
                "orm_map": "o",
                "albedo_secondary": "a2",
                "normal_secondary": "n2",
            }),
        );
        assert_eq!(tally(&refs), (6, 0));
        assert!(resolves_to(&refs, RefKind::Texture, "a"));
        assert!(resolves_to(&refs, RefKind::Texture, "n2"));
        // No texture fields -> no refs.
        assert_eq!(tally(&Material::cross_refs("mat", &json!({}))), (0, 0));
    }

    #[test]
    fn voxel_world_and_chunk_cross_refs_palette() {
        let refs = VoxelWorld::cross_refs(
            "ow",
            &json!({"palette": ["", "grass"], "material": "mat_ground"}),
        );
        // Empty entry -> one Issue; "grass" -> BlockType; material -> Material.
        assert_eq!(tally(&refs), (2, 1));
        assert!(resolves_to(&refs, RefKind::BlockType, "grass"));
        assert!(resolves_to(&refs, RefKind::Material, "mat_ground"));

        let chunk = VoxelChunk::cross_refs("c", &json!({"palette": ["stone", ""]}));
        assert_eq!(tally(&chunk), (1, 1));
        assert!(resolves_to(&chunk, RefKind::BlockType, "stone"));
    }

    #[test]
    fn prop_cross_refs_model_takes_precedence_over_mesh() {
        let refs = Prop::cross_refs(
            "p",
            &json!({
                "model": "m",
                "mesh": "mesh_skipped",
                "material": "mat",
                "texture": "t",
                "parent": "par",
            }),
        );
        assert!(resolves_to(&refs, RefKind::Model, "m"));
        assert!(!resolves_to(&refs, RefKind::MeshSource, "mesh_skipped"));
        assert!(resolves_to(&refs, RefKind::Material, "mat"));
        assert!(resolves_to(&refs, RefKind::Texture, "t"));
        assert!(resolves_to(&refs, RefKind::Prop, "par"));
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
    fn scene_and_scene_reel_cross_refs() {
        assert!(resolves_to(
            &Scene::cross_refs("s", &json!({"camera_shot": "shot"})),
            RefKind::CameraShot,
            "shot",
        ));
        // Empty scenes list -> one Issue.
        assert_eq!(
            tally(&SceneReel::cross_refs("r", &json!({"scenes": []}))),
            (0, 1)
        );
        let refs = SceneReel::cross_refs("r", &json!({"scenes": ["a", ""]}));
        assert_eq!(tally(&refs), (1, 1));
        assert!(resolves_to(&refs, RefKind::Scene, "a"));
    }

    #[test]
    fn decal_and_particle_emitter_texture_refs() {
        assert!(resolves_to(
            &Decal::cross_refs("d", &json!({"texture": "t"})),
            RefKind::Texture,
            "t",
        ));
        assert_eq!(tally(&Decal::cross_refs("d", &json!({}))), (0, 0));
        assert!(resolves_to(
            &ParticleEmitter::cross_refs("pe", &json!({"texture": "spark"})),
            RefKind::Texture,
            "spark",
        ));
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
