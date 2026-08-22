//! Concinnity is a graphics application framework: construct an [`App`],
//! populate its [`World`] with components, and run it on the engine's
//! runtime loop.
//!
//! # Creating an application
//!
//! A [`World`] is constructed first, then handed to an [`App`], which runs
//! it.
//!
//! ```no_run
//! use concinnity::assets::{GraphicsConfig, TextLabel};
//! use concinnity::{App, World};
//!
//! fn main() -> std::io::Result<()> {
//!     let mut world = World::new();
//!     world.add_component(GraphicsConfig::default());
//!     world.add_component(TextLabel {
//!         content: "Hello, world!".to_string(),
//!         ..Default::default()
//!     });
//!
//!     App::from_world(world).run()
//! }
//! ```
//!
//! A [`GraphicsConfig`](assets::GraphicsConfig) is what gives the app a
//! window.
//!
//! # Features
//!
//! The default build is the runtime alone: the world loop, the renderer, and
//! the asset vocabulary above, which is all the example needs.
//!
//! `cook` adds the `cook` module, which compiles authored assets into a
//! runnable [`World`] in process. It pulls in the asset importers (glTF, FBX,
//! textures, fonts), so a shipped application that plays an already-compiled
//! world should leave it off.
//!
//! `vulkan` selects the Vulkan backend where the platform default is Metal or
//! DirectX.

pub use concinnity_engine::App;
pub use concinnity_engine::ecs::World;
pub use concinnity_memory::install_global_allocator;

/// The runtime asset vocabulary (`Application`, `Camera3D`, `Room`,
/// `DirectionalLight`, ...), each addable to a [`World`] as a component.
pub mod assets {
    // Named one by one rather than glob-imported. The glob also re-exported
    // concinnity-core's `procedural_mesh`, `sdf_volume` and `shader` module
    // paths, and the extension traits the renderer derives geometry through
    // (`SpotLightGeometry`, `PostProcessResolve`, ...), none of which an
    // application authoring assets has any use for. `asset_exports` checks the
    // list stays complete.
    pub use concinnity_engine::assets::{
        AaMode, AnimGraph, AnimParams, Animation, AppLimits, Application, ApplicationArgs,
        AudioBus, AudioClip, AudioCommand, AudioCue, AudioEmitter, AudioOcclusionProbe,
        AudioTarget, Behavior, BehaviorSource, BlockType, BodyDynamics, Camera3D, Camera3DArgs,
        CameraController, CameraProbe, CameraShot, CharacterCapsule, CharacterRig, Children, CmpOp,
        Collider, ColorLut, ContactEvent, ControlsCommand, CubemapTexture, CueKind, DebugHud,
        Decal, DespawnRequest, DirectionalLight, EngineDefaults, EnvironmentMap, Expr, File,
        FileArgs, FileKind, FollowController, FollowDrive, Font, FpsCounter, FrameInput,
        GamepadAction, GamepadButton, GamepadMap, GlassPanel, GlobalTransform, GraphBlend,
        GraphBlendPoint, GraphCondition, GraphIkChain, GraphParam, GraphState, GraphTransition,
        GraphicsConfig, GroundProbe, GroundProbes, Held, Hidden, HitRegion, IndirectLighting,
        InstanceTransform, InstancedProp, InteractSignal, Interactable, Joint, JointDef, JointKind,
        Justify, Key, KeyBinding, LabelBox, LayoutContainer, LayoutRow, Lifetime, LightRig,
        Literal, LoadingOverlay, LocalDecl, MainMenu, MainMenuItem, Material, MaterialPalette,
        Mesh, MeshRenderer, Model, ModelRenderer, MorphDelta, NavDirection, Node, OptionSelect,
        Panel, Parent, ParticleEmitter, PhysicsConfig, Pickup, Placement, PlayCue, PointLight,
        PostProcessConfig, Prefab, ProceduralMesh, Prop, PropBody, PropCollider, QueryDecl,
        RectAreaLight, ReflectionBlurResolution, ReflectionProbe, RenderHandle, ReparentRequest,
        RigidBody, Rolloff, Room, RoomArgs, RootMotion, Scene, SceneCommand, SceneImport,
        SceneMember, Screen, ScreenCommand, ScreenInput, ScreenShown, ScrollGroup, ScrollPanel,
        ScrollRow, SdfVolume, SettingCommand, SettingOp, SettingsProfile, Shader, ShaderKind,
        ShaderPayload, ShadowUpdate, SkeletonPose, SkinnedMesh, SkinnedVertexData, Slider,
        SpawnRequest, Spawner, SpawnerArgs, SpotLight, Sprite, SpriteFit, SsgiResolution,
        StageSource, StatHud, Story, StoryChoice, StoryCommand, StoryCondition, StoryGate,
        StoryImage, StoryImport, StoryNode, StoryOp, StoryPage, StoryPlayback, StoryReload,
        StoryScaffold, StorySpeaker, StoryStage, StreamingConfig, SubMeshRef, Target, TextAlign,
        TextInput, TextLabel, Texture, Transform, TriggerFilter, TriggerVolume, UpscaleQuality,
        UpscalerBackend, VarDecl, Variables, VertexData, VisibilityRequest, VolumeEvent,
        VolumetricFog, VoxelChunk, VoxelWorld, WaterSurface, WaterWave, Window, WindowArgs,
        WindowMode,
    };
}

#[cfg(feature = "cook")]
pub mod cook;

#[cfg(test)]
mod asset_exports;

#[cfg(test)]
mod tests {
    use super::assets::TextLabel;
    use super::{App, World};

    // The documented headless case: the same starter world without a
    // GraphicsConfig, which is what lets a test or a simulation-only tool
    // drive a world with no window.
    #[test]
    fn a_world_without_graphics_starts_headless() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".to_string(),
            ..Default::default()
        });

        let mut app = App::from_world(world);
        assert_eq!(app.start(), Ok(()));
    }
}
