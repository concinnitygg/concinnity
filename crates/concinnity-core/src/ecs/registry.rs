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
// Component discriminants live in 0..128. Discriminants are stable on disk --
// do not reorder or repurpose existing entries. The 128..255 range was once
// used by declarable systems; those are all retired (see the note below) and
// must not be reused.

use crate::define_components;
use crate::ecs::{AssetKind, BlobAssetDef, Component, PayloadLocator};
use crate::result::CnResult;

// The one component list. `$cb` is a macro that receives the `Variant => Type,
// disc` entries and expands to whatever registry it builds from them. Type paths
// are absolute so the list resolves from any crate that consumes it.
#[macro_export]
macro_rules! for_each_component {
    ($cb:ident) => {
        $cb! {
            Window            => $crate::assets::Window,            1,
            GraphicsConfig    => $crate::assets::GraphicsConfig,    2,
            ShaderStage       => $crate::assets::ShaderStage,       3,
            Camera3D          => $crate::assets::Camera3D,          4,
            Mesh              => $crate::assets::Mesh,              5,
            FrameInput        => $crate::assets::FrameInput,        6,
            Texture           => $crate::assets::Texture,           7,
            Prop              => $crate::assets::Prop,              8,
            RigidBody         => $crate::assets::RigidBody,         9,
            PropBody          => $crate::assets::PropBody,          10,
            Room              => $crate::assets::Room,              11,
            Material          => $crate::assets::Material,          12,
            DirectionalLight  => $crate::assets::DirectionalLight,  13,
            PointLight        => $crate::assets::PointLight,        14,
            ProceduralMesh    => $crate::assets::ProceduralMesh,    15,
            Model             => $crate::assets::Model,             16,
            Scene             => $crate::assets::Scene,             17,
            SceneReel         => $crate::assets::SceneReel,         18,
            Font              => $crate::assets::Font,              19,
            TextLabel         => $crate::assets::TextLabel,         20,
            LightRig          => $crate::assets::LightRig,          21,
            MaterialPalette   => $crate::assets::MaterialPalette,   22,
            CameraShot        => $crate::assets::CameraShot,        23,
            Prefab            => $crate::assets::Prefab,            24,
            HitRegion         => $crate::assets::HitRegion,         25,
            File              => $crate::assets::File,              27,
            BlockType         => $crate::assets::BlockType,         28,
            VoxelChunk        => $crate::assets::VoxelChunk,        29,
            InstancedProp     => $crate::assets::InstancedProp,     30,
            CubemapTexture    => $crate::assets::CubemapTexture,    31,
            EnvironmentMap    => $crate::assets::EnvironmentMap,    32,
            PostProcessConfig => $crate::assets::PostProcessConfig, 33,
            ColorLut          => $crate::assets::ColorLut,          34,
            SkinnedMesh       => $crate::assets::SkinnedMesh,       35,
            Animation         => $crate::assets::Animation,         36,
            SkeletonPose      => $crate::assets::SkeletonPose,      37,
            StreamingConfig   => $crate::assets::StreamingConfig,   38,
            VoxelWorld        => $crate::assets::VoxelWorld,        39,
            AudioClip         => $crate::assets::AudioClip,         40,
            AudioEmitter      => $crate::assets::AudioEmitter,      41,
            Sprite            => $crate::assets::Sprite,            42,
            KeyBinding        => $crate::assets::KeyBinding,        43,
            View              => $crate::assets::View,              44,
            Decal             => $crate::assets::Decal,             46,
            VolumetricFog     => $crate::assets::VolumetricFog,     47,
            Joint             => $crate::assets::Joint,             48,
            ParticleEmitter   => $crate::assets::ParticleEmitter,   49,
            WaterSurface      => $crate::assets::WaterSurface,      50,
            SdfVolume         => $crate::assets::SdfVolume,         51,
            GlassPanel        => $crate::assets::GlassPanel,        52,
            LayoutContainer   => $crate::assets::LayoutContainer,   53,
            PhysicsConfig     => $crate::assets::PhysicsConfig,     54,
            FpsCounter        => $crate::assets::FpsCounter,        55,
            StatHud           => $crate::assets::StatHud,           56,
            SceneImport       => $crate::assets::SceneImport,       57,
            MainMenu          => $crate::assets::MainMenu,          58,
            OptionSelect      => $crate::assets::OptionSelect,      59,
            Slider            => $crate::assets::Slider,            61,
            ScrollPanel       => $crate::assets::ScrollPanel,       62,
            ReflectionProbe   => $crate::assets::ReflectionProbe,   65,
            Transform         => $crate::assets::Transform,         66,
            MeshRenderer      => $crate::assets::MeshRenderer,      67,
            ModelRenderer     => $crate::assets::ModelRenderer,     68,
            Collider          => $crate::assets::Collider,          69,
            Interactable      => $crate::assets::Interactable,      70,
            Pickup            => $crate::assets::Pickup,            71,
            Parent            => $crate::assets::Parent,            72,
            Children          => $crate::assets::Children,          73,
            SceneMember       => $crate::assets::SceneMember,       74,
            GlobalTransform   => $crate::assets::GlobalTransform,   75,
            RenderHandle      => $crate::assets::RenderHandle,      76,
            Held              => $crate::assets::Held,              77,
            Lifetime          => $crate::assets::Lifetime,          78,
            Spawner           => $crate::assets::Spawner,           79,
            DebugHud          => $crate::assets::DebugHud,          80,
            EngineDefaults    => $crate::assets::EngineDefaults,    81,
            StoryImport       => $crate::assets::StoryImport,       82,
            AudioCue          => $crate::assets::AudioCue,          83,
            Story             => $crate::assets::Story,             84,
            Application       => $crate::assets::Application,       85,
            AnimGraph         => $crate::assets::AnimGraph,         86,
            AnimParams        => $crate::assets::AnimParams,        87,
            CharacterRig      => $crate::assets::CharacterRig,      88,
            GroundProbes      => $crate::assets::GroundProbes,      89,
            CameraProbe       => $crate::assets::CameraProbe,       90,
            TextInput         => $crate::assets::TextInput,         91,
            Panel             => $crate::assets::Panel,             92,
        }
    };
}

// The runtime half: the `ComponentAsset` value enum, its blob loader, and the
// ECS storage. The authoring `ComponentType` registry is built from the same
// list in the build crate.
crate::for_each_component!(define_components);

// Retired component discriminants (0..128), stable on disk; never reuse. Each
// is now an Events<T> queue, not a component; all were RuntimeOnly (never
// serialized), so no blob references the gaps:
//   26 SceneCommand, 45 ViewCommand, 60 SettingCommand, 63 ControlsCommand,
//   64 AudioCommand
//
// Retired system discriminants (128..255), stable on disk; never reuse:
//   130 GraphicsSystem, 131 FpsCounter, 141 Camera3DSystem, 142 PhysicsSystem,
//   143 UiInputSystem, 145 AnimationSystem, 146 AudioSystem, 147 StatHud
// These were declarable systems before systems became internal. FpsCounter /
// StatHud became components (discs 55 / 56); PhysicsSystem's world config became
// the `PhysicsConfig` component (disc 54).
