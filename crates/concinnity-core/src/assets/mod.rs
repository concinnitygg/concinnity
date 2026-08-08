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
mod application;
mod audio_command;
mod camera3d;
mod camera_probe;
mod character_rig;
mod contact_event;
mod controls_command;
mod despawn_request;
mod file;
mod frame_input;
mod gamepad_button;
mod gamepad_map;
mod geometry;
mod ground_probes;
mod input_key;
mod interact_signal;
mod lifetime;
mod nav_direction;
mod play_cue;
mod post_process_config;
pub mod procedural_mesh;
mod reparent_request;
mod room;
mod root_motion_event;
mod scene_command;
mod screen_command;
mod screen_shown;
pub mod sdf_volume;
mod setting_command;
pub mod shader;
mod skeleton_pose;
mod skinned_mesh;
mod spawn_request;
mod spawner;
mod story_command;
mod target;
mod visibility_request;
mod volume_event;

// Per-instance components an entity is composed from: its placement, render
// description, collision, hierarchy, and gameplay tags.
mod body_dynamics;
mod children;
mod collider;
mod global_transform;
mod held;
mod hidden;
mod interactable;
mod mesh_renderer;
mod model_renderer;
mod parent;
mod pickup;
mod render_handle;
mod scene_member;
mod transform;

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
pub use application::Application;
pub use audio_command::AudioCommand;
pub use camera_probe::CameraProbe;
pub use camera3d::Camera3D;
pub use character_rig::CharacterRig;
pub use concinnity_asset::AaMode;
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
pub use concinnity_asset::IndirectLighting;
pub use concinnity_asset::KeyBinding;
pub use concinnity_asset::LightRig;
pub use concinnity_asset::LoadingOverlay;
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
pub use concinnity_asset::PropBody;
pub use concinnity_asset::PropCollider;
pub use concinnity_asset::RectAreaLight;
pub use concinnity_asset::ReflectionBlurResolution;
pub use concinnity_asset::ReflectionProbe;
pub use concinnity_asset::RigidBody;
pub use concinnity_asset::RoomArgs;
pub use concinnity_asset::Scene;
pub use concinnity_asset::SceneImport;
pub use concinnity_asset::ShadowUpdate;
pub use concinnity_asset::Slider;
pub use concinnity_asset::SpawnerArgs;
pub use concinnity_asset::SpotLight;
pub use concinnity_asset::SsgiResolution;
pub use concinnity_asset::StoryImport;
pub use concinnity_asset::StreamingConfig;
pub use concinnity_asset::TextInput;
pub use concinnity_asset::Texture;
pub use concinnity_asset::UpscaleQuality;
pub use concinnity_asset::UpscalerBackend;
pub use concinnity_asset::VolumetricFog;
pub use concinnity_asset::VoxelChunk;
pub use concinnity_asset::VoxelWorld;
pub use concinnity_asset::{AppLimits, ApplicationArgs};
pub use concinnity_asset::{AudioCue, CueKind};
pub use concinnity_asset::{
    Behavior, BehaviorSource, Expr, Literal, LocalDecl, Node, QueryDecl, VarDecl, Variables,
};
pub use concinnity_asset::{Camera3DArgs, CameraController, FollowController, FollowDrive};
pub use concinnity_asset::{
    CharacterCapsule, JointDef, MorphDelta, SkinnedMesh, SkinnedVertexData,
};
pub use concinnity_asset::{
    CmpOp, Story, StoryChoice, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryPlayback, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use concinnity_asset::{FileArgs, FileKind};
pub use concinnity_asset::{InstanceTransform, InstancedProp};
pub use concinnity_asset::{Joint, JointKind};
pub use concinnity_asset::{Justify, LabelBox, LayoutContainer, LayoutRow, Placement};
pub use concinnity_asset::{MainMenu, MainMenuItem, SettingsProfile};
pub use concinnity_asset::{Mesh, VertexData};
pub use concinnity_asset::{Model, SubMeshRef};
pub use concinnity_asset::{Screen, ScreenInput};
pub use concinnity_asset::{ScrollGroup, ScrollPanel, ScrollRow};
pub use concinnity_asset::{Sprite, SpriteFit};
pub use concinnity_asset::{TextAlign, TextLabel};
pub use concinnity_asset::{TriggerFilter, TriggerVolume};
pub use concinnity_asset::{WaterSurface, WaterWave};
pub use concinnity_asset::{Window, WindowArgs, WindowMode};
pub use contact_event::ContactEvent;
pub use controls_command::ControlsCommand;
pub use despawn_request::DespawnRequest;
pub use file::File;
pub use frame_input::FrameInput;
pub use gamepad_button::GamepadButton;
pub use gamepad_map::{GamepadAction, GamepadMap};
pub use geometry::{
    GlassPanelGeometry, InstancedPropGeometry, PropGeometry, RectAreaLightGeometry,
    SPOT_MAX_ANGLE_DEG, SpotLightGeometry,
};
pub use ground_probes::{GroundProbe, GroundProbes};
pub use input_key::Key;
pub use interact_signal::InteractSignal;
pub use lifetime::Lifetime;
pub use nav_direction::NavDirection;
pub use play_cue::PlayCue;
pub use post_process_config::PostProcessResolve;
pub use reparent_request::ReparentRequest;
pub use room::Room;
pub use root_motion_event::RootMotion;
pub use scene_command::SceneCommand;
pub use screen_command::ScreenCommand;
pub use screen_shown::ScreenShown;
pub use sdf_volume::SdfVolume;
pub use setting_command::{SettingCommand, SettingOp};
pub use shader::{Shader, ShaderKind, ShaderPayload, StageSource, StageSourceExt};
pub use skeleton_pose::SkeletonPose;
pub use skinned_mesh::{SkinnedMeshGeometry, build_skeleton_from_joint_defs};
pub use spawn_request::SpawnRequest;
pub use spawner::Spawner;
pub use story_command::StoryCommand;
pub use target::Target;
pub use visibility_request::VisibilityRequest;
pub use volume_event::VolumeEvent;
// `MAX_WATER_WAVES` stays in core: the build-side water-surface validator
// (concinnity-world) reaches it on every backend, and the Metal water encoder
// also reaches `WaterWave` through the asset crate.
/// Maximum number of waves per water surface. Shared by the render backends'
/// wave uniforms and the build-side water validator.
pub const MAX_WATER_WAVES: usize = 4;

// Per-instance components an entity is composed from.
pub use body_dynamics::BodyDynamics;
pub use children::Children;
pub use collider::Collider;
pub use global_transform::GlobalTransform;
pub use held::Held;
pub use hidden::Hidden;
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

// The file-name extension of a path (the chars after the last `.` of its final
// component), or `None` when that component has no extension. A pure, no_std
// stand-in for `Path::new(p).extension().and_then(|e| e.to_str())` over the
// asset-relative source strings the asset types carry.
pub(crate) fn path_extension(path: &str) -> Option<&str> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => Some(ext),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    // Uniform, low-level checks over the small data-only asset types: their
    // derive impls, custom Defaults, arg round-trips, injection hooks,
    // source_path branches, and cross-reference declarations. Kept in one place
    // because the checks are identical in shape across many one-file components.
    use super::*;
    use crate::ecs::asset_id::AssetId;
    use crate::ecs::{Component, PayloadLocator};

    // Round-trip an asset's defaults through its baked form and the Component
    // hooks. One call executes the type's Default, serialization, `from_baked`,
    // `inject_name`, and `inject_locator`.
    fn exercise<C: Component + Default + serde::Serialize>() {
        let bytes = postcard::to_allocvec(&C::default()).expect("default serializes");
        let mut comp = C::from_baked(&bytes).expect("baked bytes deserialize");
        comp.inject_name(AssetId::default());
        comp.inject_locator(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 0,
        });
    }

    #[test]
    fn path_extension_matches_std_path_semantics() {
        assert_eq!(path_extension("foo.metal"), Some("metal"));
        assert_eq!(path_extension("shaders/pbr.hlsl"), Some("hlsl"));
        assert_eq!(path_extension("a.b.glsl"), Some("glsl"));
        // No extension, a dotfile, and a dotted directory with an extensionless
        // file all resolve to None, matching `Path::extension`.
        assert_eq!(path_extension("plain"), None);
        assert_eq!(path_extension(".bashrc"), None);
        assert_eq!(path_extension("dir.v2/plain"), None);
    }

    #[test]
    fn simple_assets_round_trip_defaults() {
        exercise::<Scene>();
        exercise::<Model>();
        exercise::<ProceduralMesh>();
        exercise::<WaterSurface>();
        // More data-only components sharing the same shape.
        exercise::<Decal>();
        exercise::<ParticleEmitter>();
        exercise::<VoxelWorld>();
        exercise::<VoxelChunk>();
    }
}
