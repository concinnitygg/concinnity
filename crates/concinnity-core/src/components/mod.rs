//! Every component type the runtime can store: the ones an authored world
//! declares and the ones only the runtime mints, plus the resources the cook
//! compiles into the blob's resource stream.
//!
//! Each module here is one asset, both halves together: the authored data
//! schema a world.jsonl declares and whatever runtime behavior it needs -- a
//! runtime struct distinct from its authored args, an extension trait, a
//! build-time `SourceBacked` binding, or a helper the generated `Component`
//! impl can't express. Most components are pure data whose impl is generated
//! from the registry (see `cn_impl_components!` in `ecs::registry`).
//!
//! The authoring-only vocabulary -- the types a world declares and the cook
//! expands away, and the authored args schemas that diverge from the component
//! they bake into -- is not named here. It lives in
//! [`cook`], and the registry half of it in
//! `concinnity_world::registry::build_only`.
//!
//! Systems are not components: every system is internal code (see
//! `World::start`), driven by the presence of the components defined here. The
//! client re-exports this module under the historical `crate::components::*`
//! paths.

// Component data types.
mod animation;
mod animation_graph;
mod animation_params;
mod app_config;
mod audio_bus;
mod audio_clip;
mod audio_command;
mod audio_cue;
mod audio_emitter;
mod audio_occlusion_probe;
mod behavior;
mod block_type;
mod camera3d;
mod camera_probe;
mod character_rig;
mod character_shape;
mod color_lut;
mod contact_event;
mod controls_command;
mod cubemap_texture;
mod debug_hud;
mod decal;
mod despawn_request;
mod directional_light;
mod entity_target;
mod environment_map;
mod file;
mod font;
mod fps_counter;
mod frame_input;
mod gamepad_button;
mod gamepad_map;
mod geometry;
mod glass_panel;
mod graphics_config;
mod ground_probes;
mod hit_region;
mod input_key;
mod instanced_prop;
mod interact_event;
mod key_binding;
mod layout_container;
mod lifetime;
mod loading_overlay;
mod material;
mod mesh;
mod model;
mod nav_direction;
mod particle_emitter;
mod physics_config;
mod physics_joint;
mod play_cue;
mod point_light;
mod post_process_config;
pub mod procedural_mesh;
mod prop;
mod prop_body;
mod rect_area_light;
mod reflection_probe;
mod reparent_request;
mod rigid_body;
mod room;
mod root_motion_event;
mod scene;
mod scene_command;
mod screen;
mod screen_command;
mod screen_shown;
mod scroll_panel;
pub mod sdf_volume;
mod setting_command;
pub mod shader;
mod skeleton_pose;
mod skinned_mesh;
mod spawn_request;
mod spawner;
mod spot_light;
mod sprite;
mod stat_hud;
mod story;
mod story_command;
mod streaming_config;
mod text_input;
mod text_label;
mod texture;
mod trigger_volume;
mod variables;
mod visibility_request;
mod volume_event;
mod volumetric_fog;
mod voxel_chunk;
mod voxel_world;
mod water_surface;
mod window;

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
mod prop_instance;
mod render_handle;
mod scene_member;
mod transform;

pub mod cook;
pub mod stored;

// Serde / default / round-trip coverage for the generated data-only
// components, gathered here after their per-type modules were removed.
#[cfg(test)]
mod component_tests;

pub use animation::{Animation, AnimationTrack, Keyframe, MorphKey};
pub use animation_graph::{
    AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph, AnimationIkChain,
    AnimationParam, AnimationState, AnimationTransition,
};
pub use animation_params::AnimationParams;
pub use app_config::AppConfig;
pub use audio_bus::AudioBus;
pub use audio_clip::AudioClip;
pub use audio_command::{AudioCommand, AudioTarget};
pub use audio_cue::AudioCue;
pub use audio_cue::CueKind;
pub use audio_emitter::AudioEmitter;
pub use audio_emitter::Rolloff;
pub use audio_occlusion_probe::AudioOcclusionProbe;
pub use behavior::Behavior;
pub use behavior::BehaviorExpr;
pub use behavior::BehaviorLiteral;
pub use behavior::BehaviorLocal;
pub use behavior::BehaviorNode;
pub use behavior::BehaviorQuery;
pub use behavior::BehaviorSource;
pub use block_type::BlockType;
pub use camera_probe::CameraProbe;
pub use camera3d::Camera3D;
pub use camera3d::CameraController;
pub use camera3d::FollowController;
pub use camera3d::FollowDrive;
pub use character_rig::CharacterRig;
pub use character_shape::CharacterShape;
pub use character_shape::JointProportion;
pub use character_shape::ResolvedSliders;
pub use character_shape::ShapeSlider;
pub use color_lut::ColorLut;
pub use contact_event::ContactEvent;
pub use controls_command::ControlsCommand;
pub use cubemap_texture::CubemapTexture;
pub use decal::Decal;
pub use despawn_request::DespawnRequest;
pub use directional_light::DirectionalLight;
pub use entity_target::EntityTarget;
pub use environment_map::EnvironmentMap;
pub use file::File;
pub use file::FileKind;
pub use font::Font;
pub use frame_input::FrameInput;
pub use gamepad_button::GamepadButton;
pub use gamepad_map::{GamepadAction, GamepadMap};
pub use geometry::{
    GlassPanelGeometry, InstancedPropGeometry, RectAreaLightGeometry, SPOT_MAX_ANGLE_DEG,
    SpotLightGeometry,
};
pub use glass_panel::GlassPanel;
pub use graphics_config::GraphicsConfig;
pub use graphics_config::ShadowUpdate;
pub use ground_probes::{GroundProbe, GroundProbes};
pub use hit_region::HitRegion;
pub use input_key::InputKey;
pub use instanced_prop::InstanceTransform;
pub use instanced_prop::InstancedProp;
pub use interact_event::InteractEvent;
pub use key_binding::KeyBinding;
pub use layout_container::Justify;
pub use layout_container::LabelBox;
pub use layout_container::LabelPlacement;
pub use layout_container::LayoutContainer;
pub use layout_container::LayoutRow;
pub use lifetime::Lifetime;
pub use loading_overlay::LoadingOverlay;
pub use material::Material;
pub use mesh::Mesh;
pub use mesh::VertexData;
pub use model::Model;
pub use model::SubMeshRef;
pub use nav_direction::NavDirection;
pub use particle_emitter::ParticleEmitter;
pub use physics_config::PhysicsConfig;
pub use physics_joint::PhysicsJoint;
pub use physics_joint::PhysicsJointKind;
pub use play_cue::PlayCue;
pub use point_light::PointLight;
pub use post_process_config::AaMode;
pub use post_process_config::IndirectLighting;
pub use post_process_config::PostProcessConfig;
pub use post_process_config::PostProcessResolve;
pub use post_process_config::ReflectionBlurResolution;
pub use post_process_config::SsgiResolution;
pub use post_process_config::UpscaleQuality;
pub use post_process_config::UpscalerBackend;
pub use procedural_mesh::ProceduralMesh;
pub use prop::Prop;
pub use prop::PropCollider;
pub use prop_body::PropBody;
pub use rect_area_light::RectAreaLight;
pub use reflection_probe::ReflectionProbe;
pub use reparent_request::ReparentRequest;
pub use rigid_body::RigidBody;
pub use room::Room;
pub use root_motion_event::RootMotionEvent;
pub use scene::Scene;
pub use scene_command::SceneCommand;
pub use screen::Screen;
pub use screen::ScreenInput;
pub use screen_command::ScreenCommand;
pub use screen_shown::ScreenShown;
pub use scroll_panel::ScrollGroup;
pub use scroll_panel::ScrollPanel;
pub use scroll_panel::ScrollRow;
pub use sdf_volume::SdfVolume;
pub use setting_command::{SettingCommand, SettingOp};
pub use shader::{Shader, ShaderKind, ShaderPayload, StageSource, StageSourceExt};
pub use skeleton_pose::SkeletonPose;
pub use skinned_mesh::CharacterCapsule;
pub use skinned_mesh::MorphDelta;
pub use skinned_mesh::SkeletonJoint;
pub use skinned_mesh::SkinnedMesh;
pub use skinned_mesh::SkinnedVertexData;
pub use skinned_mesh::{SkinnedMeshGeometry, build_skeleton_from_joint_defs};
pub use spawn_request::SpawnRequest;
pub use spawner::Spawner;
pub use spot_light::SpotLight;
pub use sprite::Sprite;
pub use sprite::SpriteFit;
pub use story::Story;
pub use story::StoryChoice;
pub use story::StoryCompareOp;
pub use story::StoryCondition;
pub use story::StoryGate;
pub use story::StoryImage;
pub use story::StoryNode;
pub use story::StoryOp;
pub use story::StoryPage;
pub use story::StoryPlayback;
pub use story::StoryReload;
pub use story::StoryScaffold;
pub use story::StorySpeaker;
pub use story::StoryStage;
pub use story_command::StoryCommand;
pub use streaming_config::StreamingConfig;
pub use text_input::TextInput;
pub use text_label::TextAlign;
pub use text_label::TextLabel;
pub use texture::Texture;
pub use trigger_volume::TriggerFilter;
pub use trigger_volume::TriggerVolume;
pub use variables::VariableDecl;
pub use variables::Variables;
pub use visibility_request::VisibilityRequest;
pub use volume_event::VolumeEvent;
pub use volumetric_fog::VolumetricFog;
pub use voxel_chunk::VoxelChunk;
pub use voxel_world::VoxelWorld;
pub use water_surface::MAX_WATER_WAVES;
pub use water_surface::WaterSurface;
pub use water_surface::WaterWave;
pub use window::Window;
pub use window::WindowMode;

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
pub use prop_instance::PropInstance;
pub use render_handle::RenderHandle;
pub use scene_member::SceneMember;
pub use transform::Transform;

// HUD-overlay request components; their behavior lives in the client crate.
pub use debug_hud::DebugHud;
pub use fps_counter::FpsCounter;
pub use stat_hud::StatHud;

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

// Bounds and capacities the engine reads off the schema. Not vocabulary: they
// declare nothing, so they stay out of both namespaces.
pub use post_process_config::{DEFAULT_SSGI_RAYS, DEFAULT_SSGI_STEPS};
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
    // `inject_name`, `inject_locator`, and the frame-exactness check.
    fn exercise<C: Component + Default + serde::Serialize>() {
        let bytes = postcard::to_allocvec(&C::default()).expect("default serializes");
        let mut comp = C::from_baked(&bytes).expect("baked bytes deserialize");
        comp.inject_name(AssetId::default());
        comp.inject_locator(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 0,
        });

        // A record written by a schema carrying a field this build no longer
        // reads leaves the tail of its frame unread. `from_baked` takes the
        // whole frame or fails.
        let mut widened = bytes.clone();
        widened.push(0);
        assert!(
            C::from_baked(&widened).is_err(),
            "{} accepted a record with an unread trailing byte",
            core::any::type_name::<C>()
        );
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
        exercise::<Decal>();
        exercise::<CharacterShape>();
        exercise::<ParticleEmitter>();
        exercise::<VoxelWorld>();
        exercise::<VoxelChunk>();
    }
}
