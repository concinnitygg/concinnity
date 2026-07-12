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
            Window            => $crate::assets::Window { gen, external },
            GraphicsConfig    => $crate::assets::GraphicsConfig { gen, external },
            ShaderStage       => $crate::assets::ShaderStage { manual },
            Camera3D          => $crate::assets::Camera3D { manual },
            Mesh              => $crate::assets::Mesh { gen, external, compiled, id },
            FrameInput        => $crate::assets::FrameInput { gen, runtime },
            Prop              => $crate::assets::Prop { gen, external, id, validate: prop },
            RigidBody         => $crate::assets::RigidBody { gen, external, validate: rigid_body },
            PropBody          => $crate::assets::PropBody { gen, external },
            Room              => $crate::assets::Room { manual },
            DirectionalLight  => $crate::assets::DirectionalLight { gen, external, baked, validate: directional_light },
            PointLight        => $crate::assets::PointLight { gen, external, baked, validate: point_light },
            ProceduralMesh    => $crate::assets::ProceduralMesh { gen, external, compiled, id },
            Model             => $crate::assets::Model { gen, external, id },
            Scene             => $crate::assets::Scene { gen, external, id },
            SceneReel         => $crate::assets::SceneReel { gen, external, id },
            TextLabel         => $crate::assets::TextLabel { gen, external, id, refs: [("font", "Font"), ("view", "View")] },
            LightRig          => $crate::assets::LightRig { gen, build_only },
            MaterialPalette   => $crate::assets::MaterialPalette { gen, build_only },
            CameraShot        => $crate::assets::CameraShot { gen, build_only },
            Prefab            => $crate::assets::Prefab { gen, build_only },
            HitRegion         => $crate::assets::HitRegion { gen, external, refs: [("label", "TextLabel"), ("view", "View")] },
            File              => $crate::assets::File { manual },
            BlockType         => $crate::assets::BlockType { gen, external, id },
            VoxelChunk        => $crate::assets::VoxelChunk { gen, external, compiled, id, validate: voxel_chunk },
            InstancedProp     => $crate::assets::InstancedProp { gen, external, id, validate: instanced_prop },
            PostProcessConfig => $crate::assets::PostProcessConfig { manual },
            SkinnedMesh       => $crate::assets::SkinnedMesh { manual },
            Animation         => $crate::assets::Animation { gen, external, id },
            SkeletonPose      => $crate::assets::SkeletonPose { runtime, build: skeleton_pose },
            StreamingConfig   => $crate::assets::StreamingConfig { gen, external },
            VoxelWorld        => $crate::assets::VoxelWorld { gen, external },
            AudioEmitter      => $crate::assets::AudioEmitter { gen, external, refs: [("clip", "AudioClip"), ("prop", "Prop")] },
            Sprite            => $crate::assets::Sprite { gen, external, id, refs: [("texture", "Texture"), ("view", "View")] },
            KeyBinding        => $crate::assets::KeyBinding { gen, external },
            View              => $crate::assets::View { gen, external, id },
            Decal             => $crate::assets::Decal { gen, external, id, validate: decal, refs: [("texture", "Texture")] },
            VolumetricFog     => $crate::assets::VolumetricFog { gen, external, baked, validate: volumetric_fog },
            Joint             => $crate::assets::Joint { gen, external, id, validate: joint },
            ParticleEmitter   => $crate::assets::ParticleEmitter { gen, external, id, validate: particle_emitter, refs: [("texture", "Texture")] },
            WaterSurface      => $crate::assets::WaterSurface { gen, external, id, validate: water_surface },
            SdfVolume         => $crate::assets::SdfVolume { manual },
            GlassPanel        => $crate::assets::GlassPanel { gen, external, id, validate: glass_panel },
            LayoutContainer   => $crate::assets::LayoutContainer { gen, external },
            PhysicsConfig     => $crate::assets::PhysicsConfig { gen, external },
            FpsCounter        => $crate::assets::FpsCounter { gen, external, refs: [("label", "TextLabel")] },
            StatHud           => $crate::assets::StatHud { gen, external },
            SceneImport       => $crate::assets::SceneImport { gen, build_only },
            MainMenu          => $crate::assets::MainMenu { gen, build_only },
            OptionSelect      => $crate::assets::OptionSelect { gen, build_only },
            Slider            => $crate::assets::Slider { gen, build_only },
            ScrollPanel       => $crate::assets::ScrollPanel { gen, external },
            ReflectionProbe   => $crate::assets::ReflectionProbe { gen, external, baked, validate: reflection_probe },
            Transform         => $crate::assets::Transform { runtime },
            MeshRenderer      => $crate::assets::MeshRenderer { runtime },
            ModelRenderer     => $crate::assets::ModelRenderer { runtime },
            Collider          => $crate::assets::Collider { runtime },
            Interactable      => $crate::assets::Interactable { runtime },
            Pickup            => $crate::assets::Pickup { runtime },
            Parent            => $crate::assets::Parent { runtime },
            Children          => $crate::assets::Children { runtime },
            SceneMember       => $crate::assets::SceneMember { runtime },
            GlobalTransform   => $crate::assets::GlobalTransform { runtime },
            RenderHandle      => $crate::assets::RenderHandle { runtime },
            Held              => $crate::assets::Held { runtime },
            Lifetime          => $crate::assets::Lifetime { runtime },
            Spawner           => $crate::assets::Spawner { manual },
            DebugHud          => $crate::assets::DebugHud { gen, external },
            EngineDefaults    => $crate::assets::EngineDefaults { gen, build_only },
            StoryImport       => $crate::assets::StoryImport { gen, build_only },
            AudioCue          => $crate::assets::AudioCue { gen, external, refs: [("clip", "AudioClip"), ("view", "View")] },
            Story             => $crate::assets::Story { gen, external, id },
            Application       => $crate::assets::Application { gen, external },
            AnimGraph         => $crate::assets::AnimGraph { gen, external, id },
            AnimParams        => $crate::assets::AnimParams { runtime, build: anim_params },
            CharacterRig      => $crate::assets::CharacterRig { runtime, build: character_rig },
            GroundProbes      => $crate::assets::GroundProbes { runtime },
            CameraProbe       => $crate::assets::CameraProbe { runtime },
            TextInput         => $crate::assets::TextInput { gen, external, id, refs: [("font", "Font"), ("view", "View")] },
            Panel             => $crate::assets::Panel { gen, build_only },
        }
    };
}

// The resource-asset list: asset types that are compiled into the blob's
// resource stream and addressed at runtime by a per-kind handle, rather than
// stored as ECS components. These have left `for_each_component!` (no
// `ComponentTag`, no `ComponentAsset`, no `impl Component`): the runtime keeps
// them in per-kind resource tables owned by the system that uses them, not in a
// component column. Cook still recognizes them as declarable asset types (it
// builds a `ResourceAssetType` from this list), compiles their payload, assigns
// their handle, and emits a resource record. Each entry is
// `Variant => Type { resource: <ResourceKind>, <flags...> }`.
//
// This is the asset-registry / component-registry split the P5 design calls for,
// applied one kind at a time; AudioClip left first, then Texture, then the GPU
// resource kinds (CubemapTexture, ...).
#[macro_export]
macro_rules! for_each_resource_asset {
    ($cb:ident) => {
        $cb! {
            AudioClip => $crate::assets::AudioClip { resource: AudioClip, compiled },
            Texture => $crate::assets::Texture { resource: Texture, compiled },
            CubemapTexture => $crate::assets::CubemapTexture { resource: CubemapTexture, compiled },
            EnvironmentMap => $crate::assets::EnvironmentMap { resource: EnvironmentMap, compiled },
            ColorLut => $crate::assets::ColorLut { resource: ColorLut, compiled },
            Font => $crate::assets::Font { resource: Font, compiled },
            Material => $crate::assets::Material { resource: Material, data },
        }
    };
}

// The runtime half: the `ComponentTag` enum, the `ComponentAsset` value enum,
// its blob loader, and the ECS storage. The authoring `ComponentType` registry
// is built from the same list in the build crate.
crate::for_each_component!(define_components);

// Generate the trivial `impl Component` blocks from the shared component list.
//
// Most components are pure data whose `Component` impl is mechanical: a NAME
// equal to the type name, `Args = Self`, `to_args` = clone, an identity (or
// named-validator) `from_args`, and the optional payload / identity / reference
// hooks. Runtime-only components are just as mechanical the other way: an empty
// `Args` and a placeholder `from_args`. Rather than hand-write one such block
// per file, each list entry in `for_each_component!` carries a compact `{ ... }`
// metadata block and this macro expands it into the impl. Entries whose impl is
// genuinely bespoke (a distinct authored `Args`, an extension trait, or a real
// `from_args` translation) mark themselves `{ manual }` and keep their impl.
//
// Metadata grammar (inside the braces):
//   manual                      -- skip; the impl is hand-written elsewhere
//   gen, <flags...>             -- pass-through (`Args = Self`) impl from flags:
//     external | build_only | runtime -- the `ORIGIN`
//     compiled                  -- `PAYLOAD = Compiled` + an `inject_locator`
//                                  that stores into `self.locator`
//     baked                     -- `BAKED = true`
//     id                        -- an `inject_name` that stores into
//                                  `self.asset_id`
//     validate: <fn>            -- `from_args` calls `assets::validate::<fn>`
//                                  (defaults to identity)
//     refs: [ ("field", "Type"), ... ] -- the `ref_fields` list
//   runtime                     -- RuntimeOnly impl: empty `RuntimeArgs`,
//                                  `from_args` = `Self::default()`
//   runtime, build: <fn>        -- as `runtime` but `from_args` calls
//                                  `assets::runtime_component::<fn>` (for the
//                                  few RuntimeOnly types without `Default`)
macro_rules! cn_impl_components {
    // Entry point: expand one impl per list entry.
    ( $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? ) => {
        $( cn_impl_components!(@one $variant $ty { $($meta)* }); )+
    };

    // Bespoke impls opt out here.
    (@one $variant:ident $ty:path { manual }) => {};

    // Generated impls: seed an empty method accumulator and the default
    // (`identity`) from_args marker, then consume the flag list one token at a
    // time. The from_args marker travels as `identity` or `validate $fn` rather
    // than as an expanded expression so the eventual `args` in the body is
    // written in the same expansion as the `from_args(args)` parameter (macro
    // hygiene ties a name to the expansion that wrote it).
    (@one $variant:ident $ty:path { gen $($flags:tt)* }) => {
        cn_impl_components!(@munch $variant $ty [] { identity } $($flags)*);
    };

    // RuntimeOnly components: never authored in a world and never round-tripped
    // through a blob (their owning systems build the real instances). They all
    // share the empty `RuntimeArgs`, so `from_args` only needs to yield a valid
    // placeholder -- `Self::default()`, or a named constructor for the few types
    // that do not derive `Default`.
    (@one $variant:ident $ty:path { runtime }) => {
        impl $crate::ecs::Component for $ty {
            const NAME: &'static str = stringify!($variant);
            const ORIGIN: $crate::ecs::AssetOrigin = $crate::ecs::AssetOrigin::RuntimeOnly;
            type Args = $crate::assets::RuntimeArgs;
            fn to_args(&self) -> Self::Args {
                $crate::assets::RuntimeArgs::default()
            }
            fn from_args(_: Self::Args) -> Self {
                Self::default()
            }
        }
    };
    (@one $variant:ident $ty:path { runtime, build: $f:ident }) => {
        impl $crate::ecs::Component for $ty {
            const NAME: &'static str = stringify!($variant);
            const ORIGIN: $crate::ecs::AssetOrigin = $crate::ecs::AssetOrigin::RuntimeOnly;
            type Args = $crate::assets::RuntimeArgs;
            fn to_args(&self) -> Self::Args {
                $crate::assets::RuntimeArgs::default()
            }
            fn from_args(_: Self::Args) -> Self {
                $crate::assets::runtime_component::$f()
            }
        }
    };

    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , external $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)* const ORIGIN: $crate::ecs::AssetOrigin = $crate::ecs::AssetOrigin::External;]
            { $($fa)* } $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , build_only $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)* const ORIGIN: $crate::ecs::AssetOrigin = $crate::ecs::AssetOrigin::BuildOnly;]
            { $($fa)* } $($rest)*);
    };
    // A pass-through (`Args = Self`) component that is nonetheless RuntimeOnly,
    // e.g. the per-frame input snapshot: keeps clone/identity but is never
    // authored.
    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , runtime $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)* const ORIGIN: $crate::ecs::AssetOrigin = $crate::ecs::AssetOrigin::RuntimeOnly;]
            { $($fa)* } $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , compiled $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)*
             const PAYLOAD: $crate::ecs::AssetPayload = $crate::ecs::AssetPayload::Compiled;
             fn inject_locator(&mut self, locator: $crate::ecs::PayloadLocator) {
                 self.locator = Some(locator);
             }]
            { $($fa)* } $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , baked $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)* const BAKED: bool = true;]
            { $($fa)* } $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , id $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)* fn inject_name(&mut self, id: $crate::ecs::asset_id::AssetId) {
                 self.asset_id = id;
             }]
            { $($fa)* } $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , validate: $f:ident $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)*]
            { validate $f } $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] { $($fa:tt)* } , refs: [ $( ($fld:literal, $tgt:literal) ),+ $(,)? ] $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)* fn ref_fields() -> &'static [(&'static str, &'static str)] {
                 &[ $( ($fld, $tgt) ),+ ]
             }]
            { $($fa)* } $($rest)*);
    };

    // No flags left: emit the impl. Two terminal arms select the from_args body
    // by marker. Each writes the `from_args(args)` parameter and its body in the
    // same arm, so the body's `args` binds to that parameter (writing `args` in
    // a separate expansion would leave it unresolved).
    (@munch $variant:ident $ty:path [$($body:tt)*] { identity }) => {
        impl $crate::ecs::Component for $ty {
            const NAME: &'static str = stringify!($variant);
            type Args = Self;
            $($body)*
            fn to_args(&self) -> Self {
                self.clone()
            }
            fn from_args(args: Self) -> Self {
                args
            }
        }
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] { validate $f:ident }) => {
        impl $crate::ecs::Component for $ty {
            const NAME: &'static str = stringify!($variant);
            type Args = Self;
            $($body)*
            fn to_args(&self) -> Self {
                self.clone()
            }
            fn from_args(args: Self) -> Self {
                $crate::assets::validate::$f(args)
            }
        }
    };
}

// The generated trivial `impl Component` blocks: one per list entry marked
// `gen`, expanded from its metadata. Entries marked `manual` keep the
// hand-written impl in their own `assets` module. Emitted here (rather than in
// `assets`) so the macro is in textual scope, alongside `define_components`.
crate::for_each_component!(cn_impl_components);

// Retired discriminants that must not be reintroduced as components. Each is now
// an Events<T> queue: SceneCommand, ViewCommand, SettingCommand, ControlsCommand,
// AudioCommand. Systems (once declarable, discriminants 128..255) are all
// internal now: GraphicsSystem, FpsCounter, Camera3DSystem, PhysicsSystem,
// UiInputSystem, AnimationSystem, AudioSystem, StatHud. FpsCounter / StatHud
// became components; PhysicsSystem's world config became the `PhysicsConfig`
// component. These names are kept as history; the component list above no longer
// carries their numbers.
