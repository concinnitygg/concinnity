// src/assets/mod.rs
//
// Asset type definitions and the runtime components they map to. Most
// components are pure data whose `Component` impl is generated from the
// registry (see `cn_impl_components!` in `ecs::registry`); the modules kept
// here define a runtime struct distinct from its authored args, an extension
// trait, a build-time `SourceBacked` binding, or a helper the generated impl
// can't express. Systems are not assets: every system is internal client code
// (see the client's `World::build_internal_systems`), driven by the presence of
// the components defined here. The client re-exports this module under the
// historical `crate::assets::*` paths.

// Component data types.
mod anim_graph;
mod anim_params;
mod animation;
pub mod audio_clip;
mod audio_command;
mod camera3d;
mod camera_probe;
mod character_rig;
mod color_lut;
mod controls_command;
mod cubemap_texture;
mod despawn_request;
mod environment_map;
mod file;
mod font;
mod frame_input;
mod geometry;
mod ground_probes;
mod input_key;
mod lifetime;
mod mesh;
mod play_cue;
mod post_process_config;
pub mod procedural_mesh;
mod reparent_request;
mod room;
mod root_motion_event;
mod scene_command;
pub mod sdf_volume;
mod setting_command;
pub mod shader_stage;
mod skeleton_pose;
mod skinned_mesh;
mod spawn_request;
mod spawner;
mod story_command;
mod texture;
mod view_command;
mod view_shown;

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

// Named `from_args` validators referenced by the generated `Component` impls
// (see `cn_impl_components!`, invoked in `ecs::registry`). One free function per
// asset that clamps or normalizes its authored args at load time.
pub(crate) mod validate;

// Serde / default / round-trip coverage for the generated data-only
// components, gathered here after their per-type modules were removed.
#[cfg(test)]
mod component_tests;

pub use anim_graph::{
    AnimGraph, GraphBlend, GraphBlendPoint, GraphCondition, GraphIkChain, GraphParam, GraphState,
    GraphTransition,
};
pub use anim_params::AnimParams;
pub use animation::Animation;
pub use audio_command::AudioCommand;
pub use camera_probe::CameraProbe;
pub use camera3d::Camera3D;
pub use character_rig::CharacterRig;
#[allow(unused_imports)]
pub use concinnity_asset::AaMode;
pub use concinnity_asset::Application;
pub use concinnity_asset::AudioClip;
pub use concinnity_asset::AudioEmitter;
pub use concinnity_asset::BlockType;
pub use concinnity_asset::CameraShot;
pub use concinnity_asset::ColorLut;
pub use concinnity_asset::CubemapTexture;
pub use concinnity_asset::Decal;
pub use concinnity_asset::DirectionalLight;
pub use concinnity_asset::EngineDefaults;
pub use concinnity_asset::EnvironmentMap;
pub use concinnity_asset::Font;
pub use concinnity_asset::GlassPanel;
pub use concinnity_asset::GraphicsConfig;
pub use concinnity_asset::HitRegion;
#[allow(unused_imports)]
pub use concinnity_asset::IndirectLighting;
pub use concinnity_asset::KeyBinding;
pub use concinnity_asset::LightRig;
pub use concinnity_asset::Material;
pub use concinnity_asset::MaterialPalette;
pub use concinnity_asset::OptionSelect;
pub use concinnity_asset::Panel;
pub use concinnity_asset::ParticleEmitter;
pub use concinnity_asset::PhysicsConfig;
pub use concinnity_asset::PointLight;
pub use concinnity_asset::PostProcessConfig;
pub use concinnity_asset::Prefab;
pub use concinnity_asset::ProceduralMesh;
pub use concinnity_asset::Prop;
#[allow(unused_imports)]
pub use concinnity_asset::ReflectionBlurResolution;
#[allow(unused_imports)]
pub use concinnity_asset::ShadowUpdate;
#[allow(unused_imports)]
pub use concinnity_asset::SsgiResolution;
#[allow(unused_imports)]
pub use concinnity_asset::UpscaleQuality;
#[allow(unused_imports)]
pub use concinnity_asset::UpscalerBackend;
pub use concinnity_asset::{AudioCue, CueKind};
pub use concinnity_asset::{Camera3DArgs, CameraController, FollowController, FollowDrive};
pub use concinnity_asset::{FileArgs, FileKind};
pub use concinnity_asset::{InstanceTransform, InstancedProp};
pub use concinnity_asset::{Joint, JointKind};
pub use concinnity_asset::{Justify, LabelBox, LayoutContainer, LayoutRow, Placement};
pub use concinnity_asset::{MainMenu, MainMenuItem, SettingsProfile};
pub use concinnity_asset::{Mesh, VertexData};
pub use concinnity_asset::{Model, SubMeshRef};
pub use controls_command::ControlsCommand;
pub use despawn_request::DespawnRequest;
pub use file::File;
pub use frame_input::FrameInput;
pub use geometry::{GlassPanelGeometry, InstancedPropGeometry, PropGeometry};
pub use ground_probes::{GroundProbe, GroundProbes};
pub use input_key::Key;
pub use lifetime::Lifetime;
pub use play_cue::PlayCue;
pub use post_process_config::PostProcessResolve;
pub use root_motion_event::RootMotion;
// `PropCollider` is re-exported for tests / future consumers; the crate
// currently only uses it through `Prop.collider`, so the re-export is unused
// at compile time outside of the test module.
pub use concinnity_asset::PropBody;
#[allow(unused_imports)]
pub use concinnity_asset::PropCollider;
pub use concinnity_asset::ReflectionProbe;
pub use concinnity_asset::RigidBody;
pub use concinnity_asset::RoomArgs;
pub use concinnity_asset::Scene;
pub use concinnity_asset::SceneImport;
pub use concinnity_asset::SceneReel;
pub use concinnity_asset::{ScrollGroup, ScrollPanel, ScrollRow};
pub use reparent_request::ReparentRequest;
pub use room::Room;
pub use scene_command::SceneCommand;
pub use sdf_volume::SdfVolume;
pub use setting_command::{SettingCommand, SettingOp};
// Re-exported for the Metal raymarch encoder; non-Metal builds reach
// the asset through `SdfVolume` only.
pub use concinnity_asset::Slider;
pub use concinnity_asset::SpawnerArgs;
pub use concinnity_asset::StoryImport;
pub use concinnity_asset::StreamingConfig;
pub use concinnity_asset::TextInput;
pub use concinnity_asset::Texture;
pub use concinnity_asset::View;
pub use concinnity_asset::VolumetricFog;
pub use concinnity_asset::VoxelChunk;
pub use concinnity_asset::VoxelWorld;
pub use concinnity_asset::{CharacterCapsule, JointDef, SkinnedMesh, SkinnedVertexData};
pub use concinnity_asset::{
    CmpOp, Story, StoryChoice, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use concinnity_asset::{Sprite, SpriteFit};
pub use concinnity_asset::{TextAlign, TextLabel};
pub use concinnity_asset::{WaterSurface, WaterWave};
pub use concinnity_asset::{Window, WindowArgs, WindowMode};
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub use sdf_volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR, SDF_PARAMS_LEN};
pub use shader_stage::{ShaderKind, ShaderStage, ShaderStageExt};
pub use skeleton_pose::SkeletonPose;
pub use skinned_mesh::{SkinnedMeshGeometry, build_skeleton_from_joint_defs};
pub use spawn_request::SpawnRequest;
pub use spawner::Spawner;
pub use story_command::StoryCommand;
pub use view_command::ViewCommand;
pub use view_shown::ViewShown;
// `MAX_WATER_WAVES` stays in core (used by the water-surface validator);
// re-exported for the Metal water encoder, which also reaches `WaterWave`
// through the asset crate.
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub use validate::MAX_WATER_WAVES;

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
pub use concinnity_asset::DebugHud;
pub use concinnity_asset::FpsCounter;
pub use concinnity_asset::StatHud;

#[cfg(test)]
mod tests {
    // Uniform, low-level checks over the small data-only asset types: their
    // derive impls, custom Defaults, arg round-trips, injection hooks,
    // source_path branches, and cross-reference declarations. Kept in one place
    // because the checks are identical in shape across many one-file components.
    use super::*;
    use crate::build::{Platform, SourceBacked};
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
}
