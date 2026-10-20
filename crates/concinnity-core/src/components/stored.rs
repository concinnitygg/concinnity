//! The stored vocabulary: every type a world can hold as a component.
//!
//! Membership is the component registry's `stored` group. Named here are the ones a running world mints and the ones whose
//! runtime struct differs from the authored args it bakes from. What the cook
//! consumes instead of storing is [`super::cook`].
//!
//! This is the module `concinnity::components` globs, so a type reaches the
//! framework's runtime namespace by reaching this one.

pub use crate::components::animation::character_shape::{
    CharacterShape, JointProportion, ResolvedSliders, ShapeSlider,
};
pub use crate::components::app::app_config::FramePriority;
pub use crate::components::app::engine_defaults::EngineDefaults;
pub use crate::components::app::window::{Window, WindowMode};
pub use crate::components::assets::decal::Decal;
pub use crate::components::assets::file::FileKind;
pub use crate::components::assets::glass_panel::GlassPanel;
pub use crate::components::assets::model::{Model, SubMeshRef};
pub use crate::components::assets::particle_emitter::ParticleEmitter;
pub use crate::components::assets::procedural_mesh::ProceduralMesh;
pub use crate::components::assets::scene::{Scene, SceneTransition};
pub use crate::components::assets::sdf_volume::SdfVolume;
pub use crate::components::assets::shader::{Shader, ShaderPrograms, ShaderStage};
pub use crate::components::assets::sprite::{Sprite, SpriteFit};
pub use crate::components::audio::bus::AudioBus;
pub use crate::components::audio::cue::{AudioCue, CueKind};
pub use crate::components::audio::emitter::{AudioEmitter, Rolloff};
pub use crate::components::camera::benchmark_config::{
    BenchmarkConfig, BenchmarkPhase, CameraPose, FrameCapture,
};
pub use crate::components::camera::camera3d::{CameraController, FollowController, FollowDrive};
pub use crate::components::camera::track::{
    CameraTrack, CameraTravel, CameraTravelKey, CameraTurn, CameraTurnKey, Ease,
};
pub use crate::components::entity::instanced_prop::{InstanceTransform, InstancedProp};
pub use crate::components::entity::prop::{Prop, PropCollider, PropColliderShape};
pub use crate::components::environment::atmosphere::Atmosphere;
pub use crate::components::environment::block_type::BlockType;
pub use crate::components::environment::grass::Grass;
pub use crate::components::environment::night_sky::NightSky;
pub use crate::components::environment::planet::Planet;
pub use crate::components::environment::sky_rotation::SkyRotation;
pub use crate::components::environment::terrain::{Terrain, TerrainLayer};
pub use crate::components::environment::voxel_chunk::VoxelChunk;
pub use crate::components::environment::voxel_world::VoxelWorld;
pub use crate::components::environment::water_surface::{WaterShape, WaterSurface, WaterWave};
pub use crate::components::environment::wind::Wind;
pub use crate::components::gameplay::behavior::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery,
    BehaviorSource,
};
pub use crate::components::gameplay::story::{
    Story, StoryChoice, StoryCompareOp, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryPlayback, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use crate::components::gameplay::variables::{VariableDecl, Variables};
pub use crate::components::graphics::ambient_occlusion::{
    AmbientOcclusion, AmbientOcclusionOverrides,
};
pub use crate::components::graphics::anti_aliasing::{
    AaMode, AntiAliasing, AntiAliasingOverrides, UpscaleQuality, UpscalerBackend,
};
pub use crate::components::graphics::background::{Background, BackgroundOverrides};
pub use crate::components::graphics::bloom::{Bloom, BloomOverrides};
pub use crate::components::graphics::culling::{Culling, CullingOverrides};
pub use crate::components::graphics::display::{Display, DisplayOverrides};
pub use crate::components::graphics::exposure::{Exposure, ExposureOverrides};
pub use crate::components::graphics::frame_pacing::{FramePacing, FramePacingOverrides};
pub use crate::components::graphics::graphics_config::GraphicsConfig;
pub use crate::components::graphics::pass_resolution::PassResolution;
pub use crate::components::graphics::planet_detail::{PlanetDetail, PlanetDetailOverrides};
pub use crate::components::graphics::reflections::{Reflections, ReflectionsOverrides};
pub use crate::components::graphics::shadows::{ShadowUpdate, Shadows, ShadowsOverrides};
pub use crate::components::graphics::streaming_config::StreamingConfig;
pub use crate::components::graphics::texture_filtering::{
    TextureFiltering, TextureFilteringOverrides,
};
pub use crate::components::graphics::vignette::{Vignette, VignetteOverrides};
pub use crate::components::graphics::volumetric_fog::VolumetricFog;
pub use crate::components::input::key_binding::KeyBinding;
pub use crate::components::input::key_event::{KeyEvent, KeyMods, KeyPress};
pub use crate::components::light::directional::DirectionalLight;
pub use crate::components::light::indirect_lighting::{
    IndirectLighting, IndirectLightingOverrides, IndirectTechnique,
};
pub use crate::components::light::point::PointLight;
pub use crate::components::light::rect_area::RectAreaLight;
pub use crate::components::light::reflection_probe::ReflectionProbe;
pub use crate::components::light::spot::SpotLight;
pub use crate::components::physics::config::PhysicsConfig;
pub use crate::components::physics::joint::{PhysicsJoint, PhysicsJointKind};
pub use crate::components::physics::prop_body::PropBody;
pub use crate::components::physics::rigid_body::RigidBody;
pub use crate::components::physics::trigger_volume::{TriggerFilter, TriggerVolume};
pub use crate::components::ui::debug_hud::DebugHud;
pub use crate::components::ui::fps_counter::FpsCounter;
pub use crate::components::ui::hit_region::HitRegion;
pub use crate::components::ui::layout_container::{
    Justify, LabelBox, LabelPlacement, LayoutContainer, LayoutRow,
};
pub use crate::components::ui::loading_overlay::LoadingOverlay;
pub use crate::components::ui::screen::{Screen, ScreenInput};
pub use crate::components::ui::scroll_panel::{ScrollGroup, ScrollPanel, ScrollRow};
pub use crate::components::ui::stat_hud::StatHud;
pub use crate::components::ui::text_input::TextInput;
pub use crate::components::ui::text_label::{TextAlign, TextLabel};

pub use super::{
    Animation, AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph,
    AnimationIkChain, AnimationParam, AnimationParams, AnimationState, AnimationTrack,
    AnimationTransition, AppConfig, AudioCommand, AudioOcclusionProbe, AudioTarget, BodyDynamics,
    Camera3D, CameraProbe, CharacterRig, Children, Collider, ContactEvent, ControlsCommand,
    DespawnRequest, EntityTarget, File, FrameInput, GamepadAction, GamepadButton, GamepadMap,
    GlobalTransform, GroundProbe, GroundProbes, Held, Hidden, Identity, InputKey, InteractEvent,
    Interactable, Keyframe, Lifetime, MeshRenderer, ModelRenderer, MorphKey, NavDirection, Parent,
    Pickup, PlayCue, PropInstance, RenderHandle, ReparentRequest, Room, RootMotionEvent,
    SceneCommand, SceneMember, ScreenCommand, ScreenShown, SettingCommand, SettingOp, SkeletonPose,
    SpawnRequest, Spawner, StoryCommand, Transform, VisibilityRequest, VolumeEvent, WakeRequest,
};
