//! The stored vocabulary: every type a world can hold as a component.
//!
//! Membership is the component registry's `stored` group. The schema half
//! arrives whole from concinnity-asset; named here are the components core
//! defines itself, which are the ones a running world mints and the ones whose
//! runtime struct differs from the authored args it bakes from. What the cook
//! consumes instead of storing is [`concinnity_asset::cook`].
//!
//! This is the module `concinnity::components` globs, so a type reaches the
//! framework's runtime namespace by reaching this one.

pub use concinnity_asset::components::*;

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
