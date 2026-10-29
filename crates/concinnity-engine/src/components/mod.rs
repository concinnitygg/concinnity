//! Every component type the runtime can store: the ones an authored world
//! declares and the ones only the runtime mints, plus the resources the cook
//! compiles into the blob's resource stream.
//!
//! Being a component is a role, not a place: each type is defined in the
//! module that owns its concept (`Transform` in [`crate::transform`],
//! `AudioEmitter` in [`crate::audio`], `GraphicsConfig` in
//! [`crate::settings`], and so on), beside the runtime behavior it needs. What
//! makes it a component is its row in the registry (`define_components!` in
//! `ecs::registry`), which generates most `Component` impls. This module only
//! gathers them into the one flat namespace worlds and the cook name them by.
//!
//! The authoring-only vocabulary -- the types a world declares and the cook
//! expands away, and the authored args schemas that diverge from the component
//! they bake into -- is not named here. It lives in
//! [`cook`], and the registry half of it in
//! `concinnity_cook::authoring::registry::build_only`.
//!
//! Systems are not components: every system is internal code (see
//! `World::start`), driven by the presence of the components named here.

mod vocabulary;

pub mod cook;
pub mod stored;
pub mod validate;

#[cfg(test)]
mod component_tests;
#[cfg(test)]
mod registry_tests;

pub use crate::animation::character_rig::CharacterRig;
pub use crate::animation::character_shape::{
    CharacterShape, JointProportion, ResolvedSliders, ShapeSlider,
};
pub use crate::animation::clip::{Animation, AnimationTrack, Keyframe, MorphKey};
pub use crate::animation::graph::{
    AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph, AnimationIkChain,
    AnimationParam, AnimationState, AnimationTransition,
};
pub use crate::animation::ground_probes::{GroundProbe, GroundProbes};
pub use crate::animation::params::AnimationParams;
pub use crate::animation::root_motion_event::RootMotionEvent;
pub use crate::animation::skeleton_pose::SkeletonPose;
pub use crate::app::config::{AppConfig, FramePriority};
pub use crate::audio::bus::AudioBus;
pub use crate::audio::command::{AudioCommand, AudioTarget};
pub use crate::audio::cue::{AudioCue, CueKind};
pub use crate::audio::emitter::{AudioEmitter, Rolloff};
pub use crate::audio::occlusion_probe::AudioOcclusionProbe;
pub use crate::audio::play_cue::PlayCue;
pub use crate::behavior::definition::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery,
    BehaviorSource,
};
pub use crate::behavior::variables::{VariableDecl, Variables};
pub use crate::camera::camera3d::{Camera3D, CameraController, FollowController, FollowDrive};
pub use crate::camera::probe::{CameraProbe, Followed};
pub use crate::camera::track::{
    CameraBoom, CameraBoomKey, CameraTrack, CameraTrackArgs, CameraTravel, CameraTravelKey,
    CameraTurn, CameraTurnKey, Ease,
};
pub use crate::controller::interact_event::InteractEvent;
pub use crate::defaults::engine::EngineDefaults;
pub use crate::ecs::entity_target::EntityTarget;
pub use crate::geometry::block_type::BlockType;
pub use crate::geometry::procedural_mesh::ProceduralMesh;
pub use crate::geometry::room::Room;
pub use crate::geometry::voxel_chunk::VoxelChunk;
pub use crate::geometry::voxel_world::VoxelWorld;
pub use crate::hud::loading_overlay::LoadingOverlay;
pub use crate::input::controls_command::ControlsCommand;
pub use crate::input::frame_input::FrameInput;
pub use crate::input::gamepad_button::GamepadButton;
pub use crate::input::gamepad_map::{GamepadAction, GamepadMap};
pub use crate::input::input_key::InputKey;
pub use crate::input::key_binding::KeyBinding;
pub use crate::input::key_event::{KeyEvent, KeyMods, KeyPress};
pub use crate::input::nav_direction::NavDirection;
pub use crate::physics::buoyancy::Buoyancy;
pub use crate::physics::config::PhysicsConfig;
pub use crate::physics::contact_event::ContactEvent;
pub use crate::physics::drive_script::{DriveKey, DriveScript};
pub use crate::physics::joint::{PhysicsJoint, PhysicsJointKind};
pub use crate::physics::prop_body::PropBody;
pub use crate::physics::rigid_body::RigidBody;
pub use crate::physics::rolling_drive::{RollingControl, RollingDrive};
pub use crate::physics::trigger_volume::{TriggerFilter, TriggerVolume};
pub use crate::physics::volume_event::VolumeEvent;
pub use crate::physics::wake_request::WakeRequest;
pub use crate::planet::{GroundCover, Planet, PlanetGround, PlanetSea};
pub use crate::render::decal::Decal;
pub use crate::render::environment::atmosphere::Atmosphere;
pub use crate::render::environment::grass::Grass;
pub use crate::render::environment::night_sky::NightSky;
pub use crate::render::environment::volumetric_fog::VolumetricFog;
pub use crate::render::instanced_prop::{InstanceTransform, InstancedProp};
pub use crate::render::light::directional::DirectionalLight;
pub use crate::render::light::point::PointLight;
pub use crate::render::light::rect_area::RectAreaLight;
pub use crate::render::light::spot::SPOT_MAX_ANGLE_DEG;
pub use crate::render::light::spot::SpotLight;
pub use crate::render::model::{Model, SubMeshRef};
pub use crate::render::particles::emitter::ParticleEmitter;
pub use crate::render::probe::prop_reflection_probe::PropReflectionProbe;
pub use crate::render::probe::reflection_probe::ReflectionProbe;
pub use crate::render::scene::command::SceneCommand;
pub use crate::render::scene::{Scene, SceneTransition};
pub use crate::render::sdf::volume::SdfVolume;
pub use crate::render::shader::world_shader::{Shader, ShaderPrograms, ShaderSource, ShaderStage};
pub use crate::render::transparent::glass_panel::GlassPanel;
pub use crate::resource::audio_clip::AudioClip;
pub use crate::resource::color_lut::ColorLut;
pub use crate::resource::cubemap_texture::CubemapTexture;
pub use crate::resource::environment_map::EnvironmentMap;
pub use crate::resource::file::{File, FileKind};
pub use crate::resource::font::Font;
pub use crate::resource::material::Material;
pub use crate::resource::mesh::{Mesh, VertexData};
pub use crate::resource::skinned_mesh::{
    CharacterCapsule, MorphDelta, SkeletonJoint, SkinnedMesh, SkinnedVertexData,
    build_skeleton_from_joint_defs,
};
pub use crate::resource::texture::Texture;
pub use crate::settings::command::{SettingCommand, SettingOp};
pub use crate::settings::graphics::ambient_occlusion::{
    AmbientOcclusion, AmbientOcclusionOverrides,
};
pub use crate::settings::graphics::anti_aliasing::{
    AaMode, AntiAliasing, AntiAliasingOverrides, HDR_MULTISAMPLE_COUNT, UpscaleQuality,
    UpscalerBackend, hdr_sample_count,
};
pub use crate::settings::graphics::background::{Background, BackgroundOverrides};
pub use crate::settings::graphics::bloom::{Bloom, BloomOverrides};
pub use crate::settings::graphics::config::{GRAPHICS_SECTIONS, GraphicsConfig};
pub use crate::settings::graphics::culling::{Culling, CullingOverrides};
pub use crate::settings::graphics::display::{Display, DisplayOverrides};
pub use crate::settings::graphics::exposure::{Exposure, ExposureOverrides};
pub use crate::settings::graphics::frame_pacing::{FramePacing, FramePacingOverrides};
pub use crate::settings::graphics::indirect_lighting::{
    IndirectLighting, IndirectLightingOverrides, IndirectTechnique,
};
pub use crate::settings::graphics::pass_resolution::PassResolution;
pub use crate::settings::graphics::planet_detail::{PlanetDetail, PlanetDetailOverrides};
pub use crate::settings::graphics::reflections::{Reflections, ReflectionsOverrides};
pub use crate::settings::graphics::shadows::{ShadowUpdate, Shadows, ShadowsOverrides};
pub use crate::settings::graphics::streaming::StreamingConfig;
pub use crate::settings::graphics::texture_filtering::{
    TextureFiltering, TextureFilteringOverrides,
};
pub use crate::settings::graphics::vignette::{Vignette, VignetteOverrides};
pub use crate::sky::rotation::SkyRotation;
pub use crate::spawn::despawn_request::DespawnRequest;
pub use crate::spawn::lifetime::Lifetime;
pub use crate::spawn::prop::{PrefabTemplate, Prop, PropCollider, PropColliderShape, PropParent};
pub use crate::spawn::reparent_request::ReparentRequest;
pub use crate::spawn::spawn_request::SpawnRequest;
pub use crate::spawn::spawner::Spawner;
pub use crate::spawn::visibility_request::VisibilityRequest;
pub use crate::story::command::StoryCommand;
pub use crate::story::definition::{
    Story, StoryChoice, StoryCompareOp, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryPlayback, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use crate::terrain::{GroundRule, Terrain, TerrainLayer};
pub use crate::ui::authored_action::{AuthoredAction, NamedAction};
pub use crate::ui::hit_region::HitRegion;
pub use crate::ui::layout_container::{
    Justify, LabelBox, LabelPlacement, LayoutContainer, LayoutRow,
};
pub use crate::ui::screen::{Screen, ScreenInput};
pub use crate::ui::screen_command::ScreenCommand;
pub use crate::ui::screen_shown::ScreenShown;
pub use crate::ui::scroll_panel::{ScrollElement, ScrollGroup, ScrollPanel, ScrollRow};
pub use crate::ui::sprite::{Sprite, SpriteFit};
pub use crate::ui::text_input::TextInput;
pub use crate::ui::text_label::{ColorRun, ColorRunError, TextAlign, TextLabel, color_run_error};
pub use crate::ui::ui_action::{SettingVerb, UiAction};
pub use crate::water::{MAX_WATER_WAVES, WaterSurface, WaterWave};
pub use crate::wind::Wind;
pub use crate::window::{Window, WindowMode};
pub use concinnity_derive::Vocabulary;
pub use vocabulary::Vocabulary;
pub(crate) use vocabulary::vocabulary_synonyms;

// Per-instance components an entity is composed from.
pub use crate::controller::held::Held;
pub use crate::controller::interactable::Interactable;
pub use crate::controller::pickup::Pickup;
pub use crate::ecs::identity::Identity;
pub use crate::physics::body_dynamics::BodyDynamics;
pub use crate::physics::collider::Collider;
pub use crate::render::mesh_renderer::MeshRenderer;
pub use crate::render::model_renderer::ModelRenderer;
pub use crate::render::scene::member::SceneMember;
pub use crate::render::slots::render_handle::RenderHandle;
pub use crate::spawn::hidden::Hidden;
pub use crate::spawn::prop_instance::PropInstance;
pub use crate::transform::GlobalTransform;
pub use crate::transform::Transform;
pub use crate::transform::children::Children;
pub use crate::transform::parent::Parent;

pub use crate::benchmark::PlanetTour;
pub use crate::benchmark::config::{BenchmarkConfig, BenchmarkPhase, BenchmarkRoute, CameraPose};
pub use crate::screenshot::Screenshot;

// HUD-overlay request components; their behavior lives in the client crate.
pub use crate::hud::debug_hud::DebugHud;
pub use crate::hud::fps_counter::FpsCounter;
pub use crate::hud::stat_hud::StatHud;

// Bounds and capacities the engine reads off the schema. Not vocabulary: they
// declare nothing, so they stay out of both namespaces.
pub use crate::settings::graphics::indirect_lighting::DEFAULT_SSGI_RAYS;
