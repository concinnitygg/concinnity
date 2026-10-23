//! The stored vocabulary: every type a world can hold as a component.
//!
//! Membership is the component registry's `stored` group. Named here are the ones a running world mints and the ones whose
//! runtime struct differs from the authored args it bakes from. What the cook
//! consumes instead of storing is [`super::cook`].
//!
//! This is the module `concinnity::components` globs, so a type reaches the
//! framework's runtime namespace by reaching this one.

pub use crate::animation::character_shape::{
    CharacterShape, JointProportion, ResolvedSliders, ShapeSlider,
};
pub use crate::app::config::FramePriority;
pub use crate::audio::bus::AudioBus;
pub use crate::audio::cue::{AudioCue, CueKind};
pub use crate::audio::emitter::{AudioEmitter, Rolloff};
pub use crate::behavior::definition::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery,
    BehaviorSource,
};
pub use crate::behavior::variables::{VariableDecl, Variables};
pub use crate::benchmark::config::{BenchmarkConfig, BenchmarkPhase, CameraPose, FrameCapture};
pub use crate::camera::camera3d::{CameraController, FollowController, FollowDrive};
pub use crate::camera::track::{
    CameraBoom, CameraBoomKey, CameraTrack, CameraTravel, CameraTravelKey, CameraTurn,
    CameraTurnKey, Ease,
};
pub use crate::defaults::engine::EngineDefaults;
pub use crate::geometry::block_type::BlockType;
pub use crate::geometry::procedural_mesh::ProceduralMesh;
pub use crate::geometry::voxel_chunk::VoxelChunk;
pub use crate::geometry::voxel_world::VoxelWorld;
pub use crate::hud::debug_hud::DebugHud;
pub use crate::hud::fps_counter::FpsCounter;
pub use crate::hud::loading_overlay::LoadingOverlay;
pub use crate::hud::stat_hud::StatHud;
pub use crate::input::key_binding::KeyBinding;
pub use crate::input::key_event::{KeyEvent, KeyMods, KeyPress};
pub use crate::physics::buoyancy::Buoyancy;
pub use crate::physics::config::PhysicsConfig;
pub use crate::physics::joint::{PhysicsJoint, PhysicsJointKind};
pub use crate::physics::prop_body::PropBody;
pub use crate::physics::rigid_body::RigidBody;
pub use crate::physics::rolling_drive::RollingDrive;
pub use crate::physics::trigger_volume::{TriggerFilter, TriggerVolume};
pub use crate::planet::{GroundCover, Planet};
pub use crate::render::decal::Decal;
pub use crate::render::environment::atmosphere::Atmosphere;
pub use crate::render::environment::grass::Grass;
pub use crate::render::environment::night_sky::NightSky;
pub use crate::render::environment::volumetric_fog::VolumetricFog;
pub use crate::render::instanced_prop::{InstanceTransform, InstancedProp};
pub use crate::render::light::directional::DirectionalLight;
pub use crate::render::light::point::PointLight;
pub use crate::render::light::rect_area::RectAreaLight;
pub use crate::render::light::spot::SpotLight;
pub use crate::render::model::{Model, SubMeshRef};
pub use crate::render::particles::emitter::ParticleEmitter;
pub use crate::render::probe::reflection_probe::ReflectionProbe;
pub use crate::render::scene::{Scene, SceneTransition};
pub use crate::render::sdf::volume::SdfVolume;
pub use crate::render::shader::world_shader::{Shader, ShaderPrograms, ShaderStage};
pub use crate::render::transparent::glass_panel::GlassPanel;
pub use crate::resource::file::FileKind;
pub use crate::settings::graphics::ambient_occlusion::{
    AmbientOcclusion, AmbientOcclusionOverrides,
};
pub use crate::settings::graphics::anti_aliasing::{
    AaMode, AntiAliasing, AntiAliasingOverrides, UpscaleQuality, UpscalerBackend,
};
pub use crate::settings::graphics::background::{Background, BackgroundOverrides};
pub use crate::settings::graphics::bloom::{Bloom, BloomOverrides};
pub use crate::settings::graphics::config::GraphicsConfig;
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
pub use crate::spawn::prop::{Prop, PropCollider, PropColliderShape};
pub use crate::story::definition::{
    Story, StoryChoice, StoryCompareOp, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryPlayback, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use crate::terrain::{GroundRule, Terrain, TerrainLayer};
pub use crate::ui::hit_region::HitRegion;
pub use crate::ui::layout_container::{
    Justify, LabelBox, LabelPlacement, LayoutContainer, LayoutRow,
};
pub use crate::ui::screen::{Screen, ScreenInput};
pub use crate::ui::scroll_panel::{ScrollGroup, ScrollPanel, ScrollRow};
pub use crate::ui::sprite::{Sprite, SpriteFit};
pub use crate::ui::text_input::TextInput;
pub use crate::ui::text_label::{TextAlign, TextLabel};
pub use crate::water::{WaterShape, WaterSurface, WaterWave};
pub use crate::wind::Wind;
pub use crate::window::{Window, WindowMode};

pub use super::{
    Animation, AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph,
    AnimationIkChain, AnimationParam, AnimationParams, AnimationState, AnimationTrack,
    AnimationTransition, AppConfig, AudioCommand, AudioOcclusionProbe, AudioTarget, BodyDynamics,
    Camera3D, CameraProbe, CharacterRig, Children, Collider, ContactEvent, ControlsCommand,
    DespawnRequest, EntityTarget, File, FrameInput, GamepadAction, GamepadButton, GamepadMap,
    GlobalTransform, GroundProbe, GroundProbes, Held, Hidden, Identity, InputKey, InteractEvent,
    Interactable, Keyframe, Lifetime, MeshRenderer, ModelRenderer, MorphKey, NavDirection, Parent,
    Pickup, PlayCue, PropInstance, RenderHandle, ReparentRequest, RollingControl, Room,
    RootMotionEvent, SceneCommand, SceneMember, ScreenCommand, ScreenShown, SettingCommand,
    SettingOp, SkeletonPose, SpawnRequest, Spawner, StoryCommand, Transform, VisibilityRequest,
    VolumeEvent, WakeRequest,
};
