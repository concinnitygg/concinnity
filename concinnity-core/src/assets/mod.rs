// src/assets/mod.rs
//
// Asset type definitions: one pure-data component per file. Systems are not
// assets: every system is internal client code (see the client's
// `World::build_internal_systems`), driven by the presence of the components
// defined here. The client re-exports this module under the historical
// `crate::assets::*` paths.

// Component data types.
mod anim_graph;
mod anim_params;
mod animation;
mod application;
pub mod audio_clip;
mod audio_command;
mod audio_cue;
mod audio_emitter;
mod block_type;
mod camera3d;
mod camera_probe;
mod camera_shot;
mod character_rig;
mod color_lut;
mod controls_command;
mod cubemap_texture;
mod decal;
mod despawn_request;
mod directional_light;
mod engine_defaults;
mod environment_map;
mod file;
mod font;
mod frame_input;
mod glass_panel;
mod graphics_config;
mod ground_probes;
mod hit_region;
mod input_key;
pub mod instanced_prop;
mod joint;
mod key_binding;
mod layout_container;
mod lifetime;
mod light_rig;
mod main_menu;
mod material;
mod material_palette;
mod mesh;
mod model;
mod option_select;
mod panel;
mod particle_emitter;
mod physics_config;
mod play_cue;
mod point_light;
mod post_process_config;
mod prefab;
pub mod procedural_mesh;
mod prop;
mod prop_body;
mod reflection_probe;
mod reparent_request;
mod rigid_body;
mod room;
mod root_motion_event;
mod scene;
mod scene_command;
mod scene_import;
mod scene_reel;
mod scroll_panel;
pub mod sdf_volume;
mod setting_command;
pub mod shader_stage;
mod skeleton_pose;
mod skinned_mesh;
mod slider;
mod spawn_request;
mod spawner;
mod sprite;
mod story;
mod story_command;
mod story_import;
mod streaming_config;
mod text_input;
mod text_label;
mod texture;
mod view;
mod view_command;
mod view_shown;
mod volumetric_fog;
mod voxel_chunk;
mod voxel_world;
mod water_surface;
mod window;

// Per-instance components an entity is composed from: its placement, render
// description, collision, hierarchy, and gameplay tags.
mod children;
mod collider;
mod global_transform;
mod held;
mod interactable;
mod mesh_renderer;
mod model_renderer;
mod parent;
mod pickup;
mod render_handle;
mod scene_member;
mod transform;

// HUD-overlay request components. Declaring one runs the matching internal
// overlay behavior (in the client crate); all are pure data here.
mod debug_hud;
mod fps_counter;
mod stat_hud;

pub use anim_graph::{
    AnimGraph, GraphBlend, GraphBlendPoint, GraphCondition, GraphIkChain, GraphParam, GraphState,
    GraphTransition,
};
pub use anim_params::AnimParams;
pub use animation::Animation;
pub use application::Application;
pub use audio_clip::AudioClip;
pub use audio_command::AudioCommand;
pub use audio_cue::{AudioCue, CueKind};
pub use audio_emitter::AudioEmitter;
pub use block_type::BlockType;
pub use camera_probe::CameraProbe;
pub use camera_shot::CameraShot;
pub use camera3d::{Camera3D, CameraController, FollowController, FollowDrive};
pub use character_rig::CharacterRig;
pub use color_lut::ColorLut;
pub use controls_command::ControlsCommand;
pub use cubemap_texture::CubemapTexture;
pub use decal::Decal;
pub use despawn_request::DespawnRequest;
pub use directional_light::DirectionalLight;
pub use engine_defaults::EngineDefaults;
pub use environment_map::EnvironmentMap;
pub use file::{File, FileKind};
pub use font::Font;
pub use frame_input::FrameInput;
pub use glass_panel::GlassPanel;
pub use graphics_config::GraphicsConfig;
#[allow(unused_imports)]
pub use graphics_config::ShadowUpdate;
pub use ground_probes::{GroundProbe, GroundProbes};
pub use hit_region::HitRegion;
pub use input_key::Key;
pub use instanced_prop::InstancedProp;
pub use joint::{Joint, JointKind};
pub use key_binding::KeyBinding;
pub use layout_container::{Justify, LabelBox, LayoutContainer, LayoutRow, Placement};
pub use lifetime::Lifetime;
pub use light_rig::LightRig;
pub use main_menu::{MainMenu, MainMenuItem, SettingsProfile};
pub use material::Material;
pub use material_palette::MaterialPalette;
pub use mesh::{Mesh, VertexData};
pub use model::{Model, SubMeshRef};
pub use option_select::OptionSelect;
pub use panel::Panel;
pub use particle_emitter::ParticleEmitter;
pub use physics_config::PhysicsConfig;
pub use play_cue::PlayCue;
pub use point_light::PointLight;
#[allow(unused_imports)]
pub use post_process_config::AaMode;
#[allow(unused_imports)]
pub use post_process_config::IndirectLighting;
pub use post_process_config::PostProcessConfig;
#[allow(unused_imports)]
pub use post_process_config::ReflectionBlurResolution;
#[allow(unused_imports)]
pub use post_process_config::SsgiResolution;
#[allow(unused_imports)]
pub use post_process_config::UpscaleQuality;
#[allow(unused_imports)]
pub use post_process_config::UpscalerBackend;
pub use prefab::Prefab;
pub use procedural_mesh::ProceduralMesh;
pub use prop::Prop;
pub use root_motion_event::RootMotion;
// `PropCollider` is re-exported for tests / future consumers; the crate
// currently only uses it through `Prop.collider`, so the re-export is unused
// at compile time outside of the test module.
#[allow(unused_imports)]
pub use prop::PropCollider;
pub use prop_body::PropBody;
pub use reflection_probe::ReflectionProbe;
pub use reparent_request::ReparentRequest;
pub use rigid_body::RigidBody;
pub use room::Room;
pub use scene::Scene;
pub use scene_command::SceneCommand;
pub use scene_import::SceneImport;
pub use scene_reel::SceneReel;
pub use scroll_panel::{ScrollGroup, ScrollPanel, ScrollRow};
pub use sdf_volume::SdfVolume;
pub use setting_command::{SettingCommand, SettingOp};
// Re-exported for the Metal raymarch encoder; non-Metal builds reach
// the asset through `SdfVolume` only.
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub use sdf_volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR, SDF_PARAMS_LEN};
pub use shader_stage::{ShaderKind, ShaderStage};
pub use skeleton_pose::SkeletonPose;
pub use skinned_mesh::{
    CharacterCapsule, JointDef, SkinnedMesh, SkinnedVertexData, build_skeleton_from_joint_defs,
};
pub use slider::Slider;
pub use spawn_request::SpawnRequest;
pub use spawner::Spawner;
pub use sprite::{Sprite, SpriteFit};
pub use story::{
    CmpOp, Story, StoryChoice, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use story_command::StoryCommand;
pub use story_import::StoryImport;
pub use streaming_config::StreamingConfig;
pub use text_input::TextInput;
pub use text_label::{TextAlign, TextLabel};
pub use texture::Texture;
pub use view::View;
pub use view_command::ViewCommand;
pub use view_shown::ViewShown;
pub use volumetric_fog::VolumetricFog;
pub use voxel_chunk::VoxelChunk;
pub use voxel_world::VoxelWorld;
pub use water_surface::WaterSurface;
// Re-exported for the Metal water encoder; non-Metal builds reach the
// asset through `WaterSurface` only.
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub use water_surface::{MAX_WATER_WAVES, WaterWave};
pub use window::{Window, WindowArgs, WindowMode};

// Per-instance components an entity is composed from.
pub use children::Children;
pub use collider::Collider;
pub use global_transform::GlobalTransform;
pub use held::Held;
pub use interactable::Interactable;
pub use mesh_renderer::MeshRenderer;
pub use model_renderer::ModelRenderer;
pub use parent::Parent;
pub use pickup::Pickup;
pub use render_handle::RenderHandle;
pub use scene_member::SceneMember;
pub use transform::Transform;

// HUD-overlay request components; their behavior lives in the client crate.
pub use debug_hud::DebugHud;
pub use fps_counter::FpsCounter;
pub use stat_hud::StatHud;

#[cfg(test)]
mod tests {
    // Uniform, low-level checks over the small data-only asset types: their
    // derive impls, custom Defaults, arg round-trips, injection hooks,
    // source_path branches, and cross-reference declarations. Kept in one place
    // because the checks are identical in shape across many one-file components.
    use super::*;
    use crate::build::{Platform, SourceBacked};
    use crate::check::cross_reference::{CrossRef, CrossReferenced, RefKind};
    use crate::ecs::asset_id::AssetId;
    use crate::ecs::{Component, PayloadLocator};
    use serde_json::json;

    // Round-trip an asset's default args through JSON and its Component hooks.
    // One call executes the type's Default, from_args, to_args, inject_name,
    // inject_locator, and registration.
    fn exercise<C: Component>() {
        let reg = C::registration();
        assert_eq!(reg.type_name, C::NAME);
        let args = <C::Args as Default>::default();
        let value = serde_json::to_value(&args).expect("default args serialize");
        let back: C::Args = serde_json::from_value(value).expect("default args deserialize");
        let mut comp = C::from_args(back);
        comp.inject_name(AssetId::default());
        comp.inject_locator(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 0,
        });
        serde_json::to_value(comp.to_args()).expect("to_args serialize");
    }

    #[test]
    fn simple_assets_round_trip_defaults() {
        exercise::<Texture>();
        exercise::<Font>();
        exercise::<CubemapTexture>();
        exercise::<ColorLut>();
        exercise::<Mesh>();
        exercise::<Room>();
        exercise::<Scene>();
        exercise::<Model>();
        exercise::<ProceduralMesh>();
        exercise::<AudioClip>();
        exercise::<EnvironmentMap>();
        exercise::<WaterSurface>();
        // More data-only components sharing the same shape.
        exercise::<Material>();
        exercise::<Decal>();
        exercise::<ParticleEmitter>();
        exercise::<VoxelWorld>();
        exercise::<VoxelChunk>();
        exercise::<SceneReel>();
    }

    #[test]
    fn source_backed_source_path_branches() {
        // Procedural generator -> no source file; file-backed -> the path;
        // neither -> None. Covers both arms of the generator-gated types.
        assert_eq!(
            Texture::source_path(&json!({"generator": "checker"}), Platform::Metal),
            None
        );
        assert_eq!(
            Texture::source_path(&json!({"source": "img.png"}), Platform::Metal),
            Some("img.png".to_string())
        );
        assert_eq!(Texture::source_path(&json!({}), Platform::Metal), None);

        assert_eq!(
            EnvironmentMap::source_path(&json!({"generator": "sky"}), Platform::Metal),
            None
        );
        assert_eq!(
            EnvironmentMap::source_path(&json!({"source": "env.hdr"}), Platform::Metal),
            Some("env.hdr".to_string())
        );

        // Font keys its source under `path`, not `source`.
        assert_eq!(
            Font::source_path(&json!({"path": "f.ttf"}), Platform::Metal),
            Some("f.ttf".to_string())
        );
        assert_eq!(Font::source_path(&json!({}), Platform::Metal), None);

        // The remaining source-keyed types share one shape: present -> Some,
        // absent -> None.
        assert_eq!(
            CubemapTexture::source_path(&json!({"source": "c"}), Platform::Metal),
            Some("c".to_string())
        );
        assert_eq!(
            CubemapTexture::source_path(&json!({}), Platform::Metal),
            None
        );
        assert_eq!(
            ColorLut::source_path(&json!({"source": "l"}), Platform::Metal),
            Some("l".to_string())
        );
        assert_eq!(ColorLut::source_path(&json!({}), Platform::Metal), None);
        assert_eq!(
            Mesh::source_path(&json!({"source": "m.obj"}), Platform::Metal),
            Some("m.obj".to_string())
        );
        assert_eq!(Mesh::source_path(&json!({}), Platform::Metal), None);
        assert_eq!(
            AudioClip::source_path(&json!({"source": "a.wav"}), Platform::Metal),
            Some("a.wav".to_string())
        );
        assert_eq!(AudioClip::source_path(&json!({}), Platform::Metal), None);
    }

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
        assert!(
            VoxelWorld::companions(&json!({}), &[])
                .iter()
                .any(|c| c.asset_type == "GraphicsConfig")
        );

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
        assert!(
            Prop::companions(&json!({}), &[])
                .iter()
                .any(|c| c.asset_type == "GraphicsConfig")
        );
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
}
