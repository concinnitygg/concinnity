//! Every component type the runtime can store: the ones an authored world
//! declares and the ones only the runtime mints, plus the resources the cook
//! compiles into the blob's resource stream.
//!
//! Each module here is one asset, both halves together: the authored data
//! schema a world.jsonl declares and whatever runtime behavior it needs -- a
//! runtime struct distinct from its authored args, methods over its fields, a
//! build-time `SourceBacked` binding, or a helper the generated `Component`
//! impl can't express. Most components are pure data whose impl is generated
//! from the registry (see `cn_impl_components!` in `ecs::registry`).
//!
//! The authoring-only vocabulary -- the types a world declares and the cook
//! expands away, and the authored args schemas that diverge from the component
//! they bake into -- is not named here. It lives in
//! [`cook`], and the registry half of it in
//! `concinnity_cook::authoring::registry::build_only`.
//!
//! Systems are not components: every system is internal code (see
//! `World::start`), driven by the presence of the components defined here.

mod animation;
mod app;
pub mod assets;
mod audio;
mod camera;
mod entity;
pub mod environment;
mod gameplay;
mod geometry;
mod graphics;
mod input;
mod light;
mod physics;
mod requests;
pub mod ui;
mod vocabulary;

pub mod cook;
pub mod stored;
pub mod validate;

#[cfg(test)]
mod component_tests;
#[cfg(test)]
mod registry_tests;

pub use animation::character_rig::CharacterRig;
pub use animation::character_shape::{
    CharacterShape, JointProportion, ResolvedSliders, ShapeSlider,
};
pub use animation::clip::{Animation, AnimationTrack, Keyframe, MorphKey};
pub use animation::graph::{
    AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph, AnimationIkChain,
    AnimationParam, AnimationState, AnimationTransition,
};
pub use animation::ground_probes::{GroundProbe, GroundProbes};
pub use animation::params::AnimationParams;
pub use animation::root_motion_event::RootMotionEvent;
pub use animation::skeleton_pose::SkeletonPose;
pub use animation::skinned_mesh::{
    CharacterCapsule, MorphDelta, SkeletonJoint, SkinnedMesh, SkinnedVertexData,
    build_skeleton_from_joint_defs,
};
pub use app::app_config::{AppConfig, FramePriority};
pub use app::engine_defaults::EngineDefaults;
pub use app::window::{Window, WindowMode};
pub use assets::decal::Decal;
pub use assets::file::{File, FileKind};
pub use assets::font::Font;
pub use assets::glass_panel::GlassPanel;
pub use assets::material::Material;
pub use assets::mesh::{Mesh, VertexData};
pub use assets::model::{Model, SubMeshRef};
pub use assets::particle_emitter::ParticleEmitter;
pub use assets::procedural_mesh::ProceduralMesh;
pub use assets::room::Room;
pub use assets::scene::{Scene, SceneTransition};
pub use assets::sdf_volume::SdfVolume;
pub use assets::shader::{Shader, ShaderPrograms, ShaderSource, ShaderStage};
pub use assets::sprite::{Sprite, SpriteFit};
pub use assets::texture::Texture;
pub use audio::bus::AudioBus;
pub use audio::clip::AudioClip;
pub use audio::command::{AudioCommand, AudioTarget};
pub use audio::cue::{AudioCue, CueKind};
pub use audio::emitter::{AudioEmitter, Rolloff};
pub use audio::occlusion_probe::AudioOcclusionProbe;
pub use audio::play_cue::PlayCue;
pub use camera::camera3d::{Camera3D, CameraController, FollowController, FollowDrive};
pub use camera::probe::CameraProbe;
pub use camera::track::{
    CameraTrack, CameraTrackArgs, CameraTravel, CameraTravelKey, CameraTurn, CameraTurnKey, Ease,
};
pub use concinnity_derive::Vocabulary;
pub use entity::entity_target::EntityTarget;
pub use entity::instanced_prop::{InstanceTransform, InstancedProp};
pub use entity::lifetime::Lifetime;
pub use entity::prop::{PrefabTemplate, Prop, PropCollider, PropColliderShape, PropParent};
pub use environment::atmosphere::Atmosphere;
pub use environment::block_type::BlockType;
pub use environment::grass::Grass;
pub use environment::night_sky::NightSky;
pub use environment::planet::Planet;
pub use environment::sky_rotation::SkyRotation;
pub use environment::terrain::{Terrain, TerrainLayer};
pub use environment::voxel_chunk::VoxelChunk;
pub use environment::voxel_world::VoxelWorld;
pub use environment::water_surface::{MAX_WATER_WAVES, WaterShape, WaterSurface, WaterWave};
pub use environment::wind::Wind;
pub use gameplay::behavior::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery,
    BehaviorSource,
};
pub use gameplay::interact_event::InteractEvent;
pub use gameplay::spawner::Spawner;
pub use gameplay::story::{
    Story, StoryChoice, StoryCompareOp, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryPlayback, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use gameplay::story_command::StoryCommand;
pub use gameplay::variables::{VariableDecl, Variables};
pub use geometry::SPOT_MAX_ANGLE_DEG;
pub use graphics::ambient_occlusion::{AmbientOcclusion, AmbientOcclusionOverrides};
pub use graphics::anti_aliasing::{
    AaMode, AntiAliasing, AntiAliasingOverrides, HDR_MULTISAMPLE_COUNT, UpscaleQuality,
    UpscalerBackend, hdr_sample_count,
};
pub use graphics::background::{Background, BackgroundOverrides};
pub use graphics::bloom::{Bloom, BloomOverrides};
pub use graphics::color_lut::ColorLut;
pub use graphics::culling::{Culling, CullingOverrides};
pub use graphics::display::{Display, DisplayOverrides};
pub use graphics::exposure::{Exposure, ExposureOverrides};
pub use graphics::frame_pacing::{FramePacing, FramePacingOverrides};
pub use graphics::graphics_config::{GRAPHICS_SECTIONS, GraphicsConfig};
pub use graphics::pass_resolution::PassResolution;
pub use graphics::planet_detail::{PlanetDetail, PlanetDetailOverrides};
pub use graphics::reflections::{Reflections, ReflectionsOverrides};
pub use graphics::shadows::{ShadowUpdate, Shadows, ShadowsOverrides};
pub use graphics::streaming_config::StreamingConfig;
pub use graphics::texture_filtering::{TextureFiltering, TextureFilteringOverrides};
pub use graphics::vignette::{Vignette, VignetteOverrides};
pub use graphics::volumetric_fog::VolumetricFog;
pub use input::controls_command::ControlsCommand;
pub use input::frame_input::FrameInput;
pub use input::gamepad_button::GamepadButton;
pub use input::gamepad_map::{GamepadAction, GamepadMap};
pub use input::input_key::InputKey;
pub use input::key_binding::KeyBinding;
pub use input::key_event::{KeyEvent, KeyMods, KeyPress};
pub use input::nav_direction::NavDirection;
pub use light::cubemap_texture::CubemapTexture;
pub use light::directional::DirectionalLight;
pub use light::environment_map::EnvironmentMap;
pub use light::indirect_lighting::{
    IndirectLighting, IndirectLightingOverrides, IndirectTechnique,
};
pub use light::point::PointLight;
pub use light::rect_area::RectAreaLight;
pub use light::reflection_probe::ReflectionProbe;
pub use light::spot::SpotLight;
pub use physics::config::PhysicsConfig;
pub use physics::contact_event::ContactEvent;
pub use physics::joint::{PhysicsJoint, PhysicsJointKind};
pub use physics::prop_body::PropBody;
pub use physics::rigid_body::RigidBody;
pub use physics::trigger_volume::{TriggerFilter, TriggerVolume};
pub use physics::volume_event::VolumeEvent;
pub use requests::despawn_request::DespawnRequest;
pub use requests::reparent_request::ReparentRequest;
pub use requests::scene_command::SceneCommand;
pub use requests::setting_command::{SettingCommand, SettingOp};
pub use requests::spawn_request::SpawnRequest;
pub use requests::visibility_request::VisibilityRequest;
pub use requests::wake_request::WakeRequest;
pub use ui::authored_action::{AuthoredAction, NamedAction};
pub use ui::hit_region::HitRegion;
pub use ui::layout_container::{Justify, LabelBox, LabelPlacement, LayoutContainer, LayoutRow};
pub use ui::loading_overlay::LoadingOverlay;
pub use ui::screen::{Screen, ScreenInput};
pub use ui::screen_command::ScreenCommand;
pub use ui::screen_shown::ScreenShown;
pub use ui::scroll_panel::{ScrollElement, ScrollGroup, ScrollPanel, ScrollRow};
pub use ui::text_input::TextInput;
pub use ui::text_label::{ColorRun, ColorRunError, TextAlign, TextLabel, color_run_error};
pub use ui::ui_action::{SettingVerb, UiAction};
pub use vocabulary::Vocabulary;
pub(crate) use vocabulary::vocabulary_synonyms;

// Per-instance components an entity is composed from.
pub use entity::children::Children;
pub use entity::global_transform::GlobalTransform;
pub use entity::hidden::Hidden;
pub use entity::identity::Identity;
pub use entity::mesh_renderer::MeshRenderer;
pub use entity::model_renderer::ModelRenderer;
pub use entity::parent::Parent;
pub use entity::prop_instance::PropInstance;
pub use entity::render_handle::RenderHandle;
pub use entity::scene_member::SceneMember;
pub use entity::transform::Transform;
pub use gameplay::held::Held;
pub use gameplay::interactable::Interactable;
pub use gameplay::pickup::Pickup;
pub use physics::body_dynamics::BodyDynamics;
pub use physics::collider::Collider;

pub use camera::benchmark_config::{BenchmarkConfig, BenchmarkPhase, CameraPose, FrameCapture};

// HUD-overlay request components; their behavior lives in the client crate.
pub use ui::debug_hud::DebugHud;
pub use ui::fps_counter::FpsCounter;
pub use ui::stat_hud::StatHud;

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
pub use light::indirect_lighting::DEFAULT_SSGI_RAYS;

#[cfg(test)]
mod tests {
    use super::*;

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
}
