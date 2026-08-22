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
//! `std` is that runtime, and it is on by default. Turning it off leaves the
//! asset vocabulary and a [`World`] to build with it, so a `no_std` crate can
//! assemble world content where no runtime exists. What it drops is everything
//! that runs: there is no `App`, no `cook`, and no `install_global_allocator`,
//! and a [`World`] carries components and resources but no systems, `start`, or
//! `step`. `concinnity_engine::ecs::World` is [`From`] the one built here, so
//! handing the content to a runtime elsewhere is one conversion.
//!
//! `cook` adds the `cook` module, which compiles authored assets into a
//! runnable [`World`] in process. It pulls in the asset importers (glTF, FBX,
//! textures, fonts), so a shipped application that plays an already-compiled
//! world should leave it off.
//!
//! `vulkan` selects the Vulkan backend where the platform default is Metal or
//! DirectX.

#![cfg_attr(not(feature = "std"), no_std)]

// The test harness is a std program whichever tier is built, so the `no_std`
// build still gets one.
#[cfg(all(test, not(feature = "std")))]
extern crate std;

#[cfg(feature = "std")]
pub use concinnity_engine::App;
#[cfg(feature = "std")]
pub use concinnity_engine::ecs::World;
#[cfg(feature = "std")]
pub use concinnity_memory::install_global_allocator;

// Without the runtime there is nothing to run a world, so the world is its data
// half alone -- the same components and resources, minus the systems.
#[cfg(not(feature = "std"))]
pub use concinnity_core::ecs::World;

/// The runtime asset vocabulary (`Application`, `Camera3D`, `Room`,
/// `DirectionalLight`, ...), each addable to a [`World`] as a component.
pub mod assets {
    // Named one by one rather than glob-imported. The glob also re-exported
    // concinnity-core's `procedural_mesh`, `sdf_volume` and `shader` module
    // paths, and the extension traits the renderer derives geometry through
    // (`SpotLightGeometry`, `PostProcessResolve`, ...), none of which an
    // application authoring assets has any use for. `asset_exports` checks the
    // list stays complete.
    pub use concinnity_core::assets::{
        AaMode, Animation, AnimationBlend, AnimationBlendPoint, AnimationCondition, AnimationGraph,
        AnimationIkChain, AnimationParam, AnimationParams, AnimationState, AnimationTrack,
        AnimationTransition, AppLimits, Application, ApplicationArgs, AudioBus, AudioClip,
        AudioCommand, AudioCue, AudioEmitter, AudioOcclusionProbe, AudioTarget, Behavior,
        BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery, BehaviorSource,
        BlockType, BodyDynamics, Camera3D, Camera3DArgs, CameraController, CameraProbe, CameraShot,
        CharacterCapsule, CharacterRig, Children, Collider, ColorLut, ContactEvent,
        ControlsCommand, CubemapTexture, CueKind, DebugHud, Decal, DespawnRequest,
        DirectionalLight, EngineDefaults, EntityTarget, EnvironmentMap, File, FileArgs, FileKind,
        FollowController, FollowDrive, Font, FpsCounter, FrameInput, GamepadAction, GamepadButton,
        GamepadMap, GlassPanel, GlobalTransform, GraphicsConfig, GroundProbe, GroundProbes, Held,
        Hidden, HitRegion, IndirectLighting, InputKey, InstanceTransform, InstancedProp,
        InteractEvent, Interactable, Justify, KeyBinding, Keyframe, LabelBox, LabelPlacement,
        LayoutContainer, LayoutRow, Lifetime, LightRig, LoadingOverlay, MainMenu, MainMenuItem,
        Material, MaterialPalette, Mesh, MeshRenderer, Model, ModelRenderer, MorphDelta, MorphKey,
        NavDirection, OptionSelect, PaletteEntry, Panel, Parent, ParticleEmitter, PhysicsConfig,
        PhysicsJoint, PhysicsJointKind, Pickup, PlayCue, PointLight, PostProcessConfig, Prefab,
        PrefabEntry, PrefabKind, ProceduralMesh, Prop, PropBody, PropCollider, RectAreaLight,
        ReflectionBlurResolution, ReflectionProbe, RenderHandle, ReparentRequest, RigidBody,
        Rolloff, Room, RoomArgs, RootMotionEvent, Scene, SceneCommand, SceneImport, SceneMember,
        Screen, ScreenCommand, ScreenInput, ScreenShown, ScrollGroup, ScrollPanel, ScrollRow,
        SdfVolume, SettingCommand, SettingOp, SettingsProfile, Shader, ShaderKind, ShaderPayload,
        ShadowUpdate, SkeletonJoint, SkeletonPose, SkinnedMesh, SkinnedVertexData, Slider,
        SpawnRequest, Spawner, SpawnerArgs, SpotLight, Sprite, SpriteFit, SsgiResolution,
        StageSource, StatHud, Story, StoryChoice, StoryCommand, StoryCompareOp, StoryCondition,
        StoryGate, StoryImage, StoryImport, StoryNode, StoryOp, StoryPage, StoryPlayback,
        StoryReload, StoryScaffold, StorySpeaker, StoryStage, StreamingConfig, SubMeshRef,
        TextAlign, TextInput, TextLabel, Texture, Transform, TriggerFilter, TriggerVolume,
        UpscaleQuality, UpscalerBackend, VariableDecl, Variables, VertexData, VisibilityRequest,
        VolumeEvent, VolumetricFog, VoxelChunk, VoxelWorld, WaterSurface, WaterWave, Window,
        WindowMode,
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

        assert_eq!(world.component_count(), 1);
        assert_eq!(
            world.query::<TextLabel>().next().unwrap().content,
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
        assert_eq!(app.start(), Ok(()));
    }

    // The tier handoff: a world built where no runtime exists becomes the
    // engine's world by conversion, with its content intact.
    #[cfg(feature = "std")]
    #[test]
    fn a_data_only_world_converts_into_the_engines_world() {
        let mut data = concinnity_core::ecs::World::new();
        data.add_component(TextLabel {
            content: "carried over".into(),
            ..Default::default()
        });

        let world: World = data.into();
        assert_eq!(world.component_count(), 1);
        assert_eq!(
            world.query::<TextLabel>().next().unwrap().content,
            "carried over"
        );
    }
}
