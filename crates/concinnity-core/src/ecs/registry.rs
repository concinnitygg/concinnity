// src/ecs/registry.rs
//
// Single source of truth for the renderer-free half of the engine's asset
// registry: every Component type paired with its stable u8 discriminant.
//
// The list lives in one macro, `for_each_component!`, so both registries built
// from it stay in lockstep: the runtime value enum + ECS storage
// (`define_components!`, invoked below in this crate) and the authoring metadata
// registry (`ComponentType`, invoked in the build crate from the same list).
//
// Components are pure data, registered with one entry each. There is no system
// registry: every system is internal client code, constructed at runtime from
// world content (see the client's `World::build_internal_systems`), never
// declared in a world or serialized to a blob. The runtime `SystemAsset` enum
// that holds the constructed systems is generated client-side from each
// system's `System` behavior impl, in the client crate's `ecs::registry`.
//
// Each component's discriminant (its on-disk blob tag and in-memory
// `ComponentId`) is assigned by its position in this list: the runtime
// `define_components!` builds a `#[repr(u8)] ComponentTag` enum whose variants
// are these entries in order, so the tag is the list position. Discriminants are
// therefore not hand-written and not a stable on-disk contract: a build
// regenerates the blob, so the blob and the engine that loads it always agree.
// Reordering the list changes every tag, which is only safe alongside a rebuild
// (`cn build`), which the workflow always does. The tag must stay in 0..128 (the
// `ComponentMask` ceiling); the list is far shorter, so position keeps it there.

use crate::define_components;
use crate::ecs::{AssetKind, BlobAssetDef, Component, PayloadLocator, RecordKind};
use crate::result::CnResult;

// The one component list. `$cb` is a macro that receives the `Variant => Type`
// entries and expands to whatever registry it builds from them. Type paths are
// absolute so the list resolves from any crate that consumes it.
#[macro_export]
macro_rules! for_each_component {
    ($cb:ident) => {
        $cb! {
            Window            => $crate::assets::Window,
            GraphicsConfig    => $crate::assets::GraphicsConfig,
            ShaderStage       => $crate::assets::ShaderStage,
            Camera3D          => $crate::assets::Camera3D,
            Mesh              => $crate::assets::Mesh,
            FrameInput        => $crate::assets::FrameInput,
            Texture           => $crate::assets::Texture,
            Prop              => $crate::assets::Prop,
            RigidBody         => $crate::assets::RigidBody,
            PropBody          => $crate::assets::PropBody,
            Room              => $crate::assets::Room,
            Material          => $crate::assets::Material,
            DirectionalLight  => $crate::assets::DirectionalLight,
            PointLight        => $crate::assets::PointLight,
            ProceduralMesh    => $crate::assets::ProceduralMesh,
            Model             => $crate::assets::Model,
            Scene             => $crate::assets::Scene,
            SceneReel         => $crate::assets::SceneReel,
            Font              => $crate::assets::Font,
            TextLabel         => $crate::assets::TextLabel,
            LightRig          => $crate::assets::LightRig,
            MaterialPalette   => $crate::assets::MaterialPalette,
            CameraShot        => $crate::assets::CameraShot,
            Prefab            => $crate::assets::Prefab,
            HitRegion         => $crate::assets::HitRegion,
            File              => $crate::assets::File,
            BlockType         => $crate::assets::BlockType,
            VoxelChunk        => $crate::assets::VoxelChunk,
            InstancedProp     => $crate::assets::InstancedProp,
            CubemapTexture    => $crate::assets::CubemapTexture,
            EnvironmentMap    => $crate::assets::EnvironmentMap,
            PostProcessConfig => $crate::assets::PostProcessConfig,
            ColorLut          => $crate::assets::ColorLut,
            SkinnedMesh       => $crate::assets::SkinnedMesh,
            Animation         => $crate::assets::Animation,
            SkeletonPose      => $crate::assets::SkeletonPose,
            StreamingConfig   => $crate::assets::StreamingConfig,
            VoxelWorld        => $crate::assets::VoxelWorld,
            AudioClip         => $crate::assets::AudioClip,
            AudioEmitter      => $crate::assets::AudioEmitter,
            Sprite            => $crate::assets::Sprite,
            KeyBinding        => $crate::assets::KeyBinding,
            View              => $crate::assets::View,
            Decal             => $crate::assets::Decal,
            VolumetricFog     => $crate::assets::VolumetricFog,
            Joint             => $crate::assets::Joint,
            ParticleEmitter   => $crate::assets::ParticleEmitter,
            WaterSurface      => $crate::assets::WaterSurface,
            SdfVolume         => $crate::assets::SdfVolume,
            GlassPanel        => $crate::assets::GlassPanel,
            LayoutContainer   => $crate::assets::LayoutContainer,
            PhysicsConfig     => $crate::assets::PhysicsConfig,
            FpsCounter        => $crate::assets::FpsCounter,
            StatHud           => $crate::assets::StatHud,
            SceneImport       => $crate::assets::SceneImport,
            MainMenu          => $crate::assets::MainMenu,
            OptionSelect      => $crate::assets::OptionSelect,
            Slider            => $crate::assets::Slider,
            ScrollPanel       => $crate::assets::ScrollPanel,
            ReflectionProbe   => $crate::assets::ReflectionProbe,
            Transform         => $crate::assets::Transform,
            MeshRenderer      => $crate::assets::MeshRenderer,
            ModelRenderer     => $crate::assets::ModelRenderer,
            Collider          => $crate::assets::Collider,
            Interactable      => $crate::assets::Interactable,
            Pickup            => $crate::assets::Pickup,
            Parent            => $crate::assets::Parent,
            Children          => $crate::assets::Children,
            SceneMember       => $crate::assets::SceneMember,
            GlobalTransform   => $crate::assets::GlobalTransform,
            RenderHandle      => $crate::assets::RenderHandle,
            Held              => $crate::assets::Held,
            Lifetime          => $crate::assets::Lifetime,
            Spawner           => $crate::assets::Spawner,
            DebugHud          => $crate::assets::DebugHud,
            EngineDefaults    => $crate::assets::EngineDefaults,
            StoryImport       => $crate::assets::StoryImport,
            AudioCue          => $crate::assets::AudioCue,
            Story             => $crate::assets::Story,
            Application       => $crate::assets::Application,
            AnimGraph         => $crate::assets::AnimGraph,
            AnimParams        => $crate::assets::AnimParams,
            CharacterRig      => $crate::assets::CharacterRig,
            GroundProbes      => $crate::assets::GroundProbes,
            CameraProbe       => $crate::assets::CameraProbe,
            TextInput         => $crate::assets::TextInput,
            Panel             => $crate::assets::Panel,
        }
    };
}

// The runtime half: the `ComponentTag` enum, the `ComponentAsset` value enum,
// its blob loader, and the ECS storage. The authoring `ComponentType` registry
// is built from the same list in the build crate.
crate::for_each_component!(define_components);

// Retired discriminants that must not be reintroduced as components. Each is now
// an Events<T> queue: SceneCommand, ViewCommand, SettingCommand, ControlsCommand,
// AudioCommand. Systems (once declarable, discriminants 128..255) are all
// internal now: GraphicsSystem, FpsCounter, Camera3DSystem, PhysicsSystem,
// UiInputSystem, AnimationSystem, AudioSystem, StatHud. FpsCounter / StatHud
// became components; PhysicsSystem's world config became the `PhysicsConfig`
// component. These names are kept as history; the component list above no longer
// carries their numbers.
