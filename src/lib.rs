//! Concinnity is a graphics application framework: construct an `App`,
//! populate its [`World`] with components, and run it on the engine's
//! runtime loop.
//!
//! # Running an application
//!
//! The most common use-case for a client will be running an `App` from
//! locally saved binary data.
//!
//! ```no_run
//! use concinnity::App;
//!
//! fn main() {
//!     App::from_blob("my_game.cnb")
//!         .expect("my_game.cnb holds a compiled world")
//!         .run()
//!         .expect("the app runs");
//! }
//! ```
//!
//! The `cook` module manages this world binary data.
//!
//! # Features
//!
//! The default build is the runtime alone: the world loop, the renderer, and
//! the asset vocabulary above, which is all the example needs.
//!
//! `std` is that runtime, and it is on by default. Turning it off leaves the
//! asset vocabulary and a [`World`] to build with it, so a `no_std` crate can
//! assemble world content where no runtime exists. What it drops is everything
//! that runs: there is no `App` and no `cook`, and a [`World`] carries
//! components but nothing to step them.
//!
//! `cook` adds the `cook` module, which compiles authored assets into a
//! runnable [`World`] in process, or writes them to a blob file for a shipped
//! application to play. It pulls in the asset importers (glTF, FBX, textures,
//! fonts), so an application that only plays an already-compiled world should
//! leave it off.
//!
//! `vulkan` selects the Vulkan backend where the platform default is Metal or
//! DirectX.

#![cfg_attr(not(feature = "std"), no_std)]

// The test harness is a std program whichever tier is built, so the `no_std`
// build still gets one.
#[cfg(all(test, not(feature = "std")))]
extern crate std;

#[cfg(feature = "std")]
mod app;
mod world;

#[cfg(feature = "std")]
pub use app::App;
pub use world::World;

/// The runtime asset vocabulary (`AppConfig`, `Camera3D`, `Room`,
/// `DirectionalLight`, ...), each addable to a [`World`] as a component.
pub mod assets {
    // Named one by one rather than glob-imported. The glob also re-exported
    // concinnity-core's `procedural_mesh`, `sdf_volume` and `shader` module
    // paths, and the extension traits the renderer derives geometry through
    // (`SpotLightGeometry`, `PostProcessResolve`, ...), none of which an
    // application authoring assets has any use for. `asset_exports` checks the
    // list stays complete.
    pub use concinnity_core::components::{
        AaMode, Animation, AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph,
        AnimationIkChain, AnimationParam, AnimationParams, AnimationState, AnimationTrack,
        AnimationTransition, AppConfig, AppConfigArgs, AudioBus, AudioClip, AudioCommand, AudioCue,
        AudioEmitter, AudioOcclusionProbe, AudioTarget, Behavior, BehaviorExpr, BehaviorLiteral,
        BehaviorLocal, BehaviorNode, BehaviorQuery, BehaviorSource, BlockType, BodyDynamics,
        Camera3D, Camera3DArgs, CameraController, CameraProbe, CameraShot, CharacterCapsule,
        CharacterModel, CharacterRig, CharacterSchema, CharacterShape, Children, Collider,
        ColorLut, ContactEvent, ControlsCommand, CubemapTexture, CueKind, DebugHud, Decal,
        DespawnRequest, DirectionalLight, EngineDefaults, EntityTarget, EnvironmentMap, File,
        FileArgs, FileKind, FollowController, FollowDrive, Font, FpsCounter, FrameInput,
        GamepadAction, GamepadButton, GamepadMap, GlassPanel, GlobalTransform, GraphicsConfig,
        GroundProbe, GroundProbes, Held, Hidden, HitRegion, IndirectLighting, InputKey,
        InstanceTransform, InstancedProp, InteractEvent, Interactable, JointProportion, Justify,
        KeyBinding, KeyPolarity, Keyframe, LabelBox, LabelPlacement, LayoutContainer, LayoutRow,
        Lifetime, LightRig, LoadingOverlay, MainMenu, MainMenuItem, Material, MaterialPalette,
        Mesh, MeshRenderer, Model, ModelRenderer, MorphDelta, MorphKey, NavDirection, OptionSelect,
        PaletteEntry, Panel, PanelSection, Parent, ParticleEmitter, PhysicsConfig, PhysicsJoint,
        PhysicsJointKind, Pickup, PlayCue, PointLight, PostProcessConfig, Prefab, PrefabEntry,
        PrefabKind, ProceduralMesh, Prop, PropBody, PropCollider, PropInstance, ProportionGroup,
        RectAreaLight, ReflectionBlurResolution, ReflectionProbe, RenderHandle, ReparentRequest,
        ResolvedSliders, RigidBody, Rolloff, Room, RoomArgs, RootMotionEvent, Scene, SceneCommand,
        SceneImport, SceneMember, SchemaJoint, SchemaKey, SchemaRegion, Screen, ScreenCommand,
        ScreenInput, ScreenShown, ScrollGroup, ScrollPanel, ScrollRow, SdfVolume, SettingCommand,
        SettingOp, SettingsProfile, Shader, ShaderKind, ShaderPayload, ShadowUpdate, ShapePreset,
        ShapeSlider, SkeletonJoint, SkeletonPose, SkinnedMesh, SkinnedVertexData, Slider,
        SpawnRequest, Spawner, SpawnerArgs, SpotLight, Sprite, SpriteFit, SsgiResolution,
        StageSource, StatHud, Story, StoryChoice, StoryCommand, StoryCompareOp, StoryCondition,
        StoryGate, StoryImage, StoryImport, StoryNode, StoryOp, StoryPage, StoryPlayback,
        StoryReload, StoryScaffold, StorySpeaker, StoryStage, StreamingConfig, SubMeshRef,
        SynthParams, SynthesizedTarget, TextAlign, TextInput, TextLabel, Texture, Transform,
        TriggerFilter, TriggerVolume, UpscaleQuality, UpscalerBackend, VariableDecl, Variables,
        VertexData, VisibilityRequest, VolumeEvent, VolumetricFog, VoxelChunk, VoxelWorld,
        WaterSurface, WaterWave, Window, WindowMode,
    };
}

#[cfg(feature = "cook")]
pub mod cook;

#[cfg(test)]
mod asset_exports;

#[cfg(test)]
mod tests {
    use super::World;
    use super::assets::TextLabel;

    // The starter world's first two lines, on whichever tier is built. Without
    // `std` this is the whole of what the facade offers, so it is also the
    // check that a world is constructible with no runtime behind it.
    #[test]
    fn a_world_holds_the_components_added_to_it() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        let inner = world.inner();
        assert_eq!(inner.component_count(), 1);
        assert_eq!(
            inner.query::<TextLabel>().next().unwrap().content,
            "Hello, world!"
        );
    }

    // The documented headless case: the same starter world without a
    // GraphicsConfig, which is what lets a test or a simulation-only tool
    // drive a world with no window.
    #[cfg(feature = "std")]
    #[test]
    fn a_world_without_graphics_starts_headless() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        let mut app = super::App::from_world(world);
        assert_eq!(app.inner_mut().start(), Ok(()));
    }
}
