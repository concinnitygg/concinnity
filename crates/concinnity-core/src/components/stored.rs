//! The stored vocabulary: every type a world can hold as a component.
//!
//! Membership is the component registry's `stored` group. Named here are the ones a running world mints and the ones whose
//! runtime struct differs from the authored args it bakes from. What the cook
//! consumes instead of storing is [`super::cook`].
//!
//! This is the module `concinnity::components` globs, so a type reaches the
//! framework's runtime namespace by reaching this one.

pub use super::audio_bus::AudioBus;
pub use super::audio_cue::{AudioCue, CueKind};
pub use super::audio_emitter::{AudioEmitter, Rolloff};
pub use super::behavior::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery,
    BehaviorSource,
};
pub use super::block_type::BlockType;
pub use super::camera3d::{CameraController, FollowController, FollowDrive};
pub use super::character_shape::{CharacterShape, JointProportion, ResolvedSliders, ShapeSlider};
pub use super::debug_hud::DebugHud;
pub use super::decal::Decal;
pub use super::directional_light::DirectionalLight;
pub use super::engine_defaults::EngineDefaults;
pub use super::file::FileKind;
pub use super::fps_counter::FpsCounter;
pub use super::glass_panel::GlassPanel;
pub use super::graphics_config::{GraphicsConfig, ShadowUpdate};
pub use super::hit_region::HitRegion;
pub use super::instanced_prop::{InstanceTransform, InstancedProp};
pub use super::key_binding::KeyBinding;
pub use super::layout_container::{Justify, LabelBox, LabelPlacement, LayoutContainer, LayoutRow};
pub use super::loading_overlay::LoadingOverlay;
pub use super::model::{Model, SubMeshRef};
pub use super::particle_emitter::ParticleEmitter;
pub use super::physics_config::PhysicsConfig;
pub use super::physics_joint::{PhysicsJoint, PhysicsJointKind};
pub use super::point_light::PointLight;
pub use super::post_process_config::{
    AaMode, IndirectLighting, PostProcessConfig, ReflectionBlurResolution, SsgiResolution,
    UpscaleQuality, UpscalerBackend,
};
pub use super::procedural_mesh::ProceduralMesh;
pub use super::prop::{Prop, PropCollider};
pub use super::prop_body::PropBody;
pub use super::rect_area_light::RectAreaLight;
pub use super::reflection_probe::ReflectionProbe;
pub use super::rigid_body::RigidBody;
pub use super::scene::Scene;
pub use super::screen::{Screen, ScreenInput};
pub use super::scroll_panel::{ScrollGroup, ScrollPanel, ScrollRow};
pub use super::sdf_volume::SdfVolume;
pub use super::shader::{Shader, ShaderKind, ShaderPayload, StageSource};
pub use super::spot_light::SpotLight;
pub use super::sprite::{Sprite, SpriteFit};
pub use super::stat_hud::StatHud;
pub use super::story::{
    Story, StoryChoice, StoryCompareOp, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryPlayback, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use super::streaming_config::StreamingConfig;
pub use super::text_input::TextInput;
pub use super::text_label::{TextAlign, TextLabel};
pub use super::trigger_volume::{TriggerFilter, TriggerVolume};
pub use super::variables::{VariableDecl, Variables};
pub use super::volumetric_fog::VolumetricFog;
pub use super::voxel_chunk::VoxelChunk;
pub use super::voxel_world::VoxelWorld;
pub use super::water_surface::{WaterSurface, WaterWave};
pub use super::window::{Window, WindowMode};

pub use super::{
    Animation, AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph,
    AnimationIkChain, AnimationParam, AnimationParams, AnimationState, AnimationTrack,
    AnimationTransition, AppConfig, AudioCommand, AudioOcclusionProbe, AudioTarget, BodyDynamics,
    Camera3D, CameraProbe, CharacterRig, Children, Collider, ContactEvent, ControlsCommand,
    DespawnRequest, EntityTarget, File, FrameInput, GamepadAction, GamepadButton, GamepadMap,
    GlobalTransform, GroundProbe, GroundProbes, Held, Hidden, InputKey, InteractEvent,
    Interactable, Keyframe, Lifetime, MeshRenderer, ModelRenderer, MorphKey, NavDirection, Parent,
    Pickup, PlayCue, PropInstance, RenderHandle, ReparentRequest, Room, RootMotionEvent,
    SceneCommand, SceneMember, ScreenCommand, ScreenShown, SettingCommand, SettingOp, SkeletonPose,
    SpawnRequest, Spawner, StoryCommand, Transform, VisibilityRequest, VolumeEvent,
};
