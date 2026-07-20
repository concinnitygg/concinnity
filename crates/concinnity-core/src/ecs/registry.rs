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
use crate::ecs::{BlobAssetDef, Component, PayloadLocator};
use crate::result::CnResult;

// The one component list. `$cb` is a macro that receives the `Variant => Type`
// entries and expands to whatever registry it builds from them. Type paths are
// absolute so the list resolves from any crate that consumes it.
#[macro_export]
macro_rules! for_each_component {
    ($cb:ident) => {
        $cb! {
            Window            => $crate::assets::Window { gen, external, singleton },
            GraphicsConfig    => $crate::assets::GraphicsConfig { gen, external, singleton, renders },
            ShaderStage       => $crate::assets::ShaderStage { manual, external, compiled },
            Camera3D          => $crate::assets::Camera3D { manual, external, useful_blank, args: Camera3DArgs },
            FrameInput        => $crate::assets::FrameInput { gen, runtime },
            Prop              => $crate::assets::Prop { gen, external, id, renders, validate: prop, refs: [("model", "Model"), ("material", "Material"), ("texture", "Texture"), ("scene", "Scene"), ("parent", "Prop")] },
            RigidBody         => $crate::assets::RigidBody { gen, external, validate: rigid_body },
            PropBody          => $crate::assets::PropBody { gen, external },
            Room              => $crate::assets::Room { manual, external, compiled, useful_blank, args: RoomArgs, refs: [("texture", "Texture"), ("wall_texture", "Texture"), ("floor_texture", "Texture"), ("ceiling_texture", "Texture")] },
            DirectionalLight  => $crate::assets::DirectionalLight { gen, external, useful_blank, validate: directional_light },
            PointLight        => $crate::assets::PointLight { gen, external, useful_blank, validate: point_light },
            SpotLight         => $crate::assets::SpotLight { gen, external, useful_blank, validate: spot_light },
            RectAreaLight     => $crate::assets::RectAreaLight { gen, external, useful_blank, validate: rect_area_light },
            ProceduralMesh    => $crate::assets::ProceduralMesh { gen, external, compiled, id },
            Model             => $crate::assets::Model { gen, external, id },
            Scene             => $crate::assets::Scene { gen, external, id, refs: [("camera_shot", "Camera3D")] },
            SceneReel         => $crate::assets::SceneReel { gen, external, id },
            TextLabel         => $crate::assets::TextLabel { gen, external, id, useful_blank, renders, refs: [("font", "Font"), ("screen", "Screen")] },
            LightRig          => $crate::assets::LightRig { gen, build_only },
            MaterialPalette   => $crate::assets::MaterialPalette { gen, build_only },
            CameraShot        => $crate::assets::CameraShot { gen, build_only },
            Prefab            => $crate::assets::Prefab { gen, build_only },
            HitRegion         => $crate::assets::HitRegion { gen, external, useful_blank, refs: [("label", "TextLabel"), ("screen", "Screen")] },
            File              => $crate::assets::File { manual, external, compiled, args: FileArgs },
            BlockType         => $crate::assets::BlockType { gen, external, id, useful_blank },
            VoxelChunk        => $crate::assets::VoxelChunk { gen, external, compiled, id, validate: voxel_chunk },
            InstancedProp     => $crate::assets::InstancedProp { gen, external, id, renders, validate: instanced_prop, refs: [("material", "Material"), ("texture", "Texture")] },
            PostProcessConfig => $crate::assets::PostProcessConfig { manual, external, singleton },
            Animation         => $crate::assets::Animation { gen, external, id },
            SkeletonPose      => $crate::assets::SkeletonPose { runtime, build: skeleton_pose },
            StreamingConfig   => $crate::assets::StreamingConfig { gen, external, singleton },
            VoxelWorld        => $crate::assets::VoxelWorld { gen, external, renders, refs: [("material", "Material")] },
            AudioEmitter      => $crate::assets::AudioEmitter { gen, external, useful_blank, refs: [("clip", "AudioClip"), ("prop", "Prop")] },
            Sprite            => $crate::assets::Sprite { gen, external, id, useful_blank, renders, refs: [("texture", "Texture"), ("screen", "Screen")] },
            KeyBinding        => $crate::assets::KeyBinding { gen, external, useful_blank, refs: [("screen", "Screen")] },
            Screen            => $crate::assets::Screen { gen, external, id, useful_blank, refs: [("focus", "TextInput")] },
            Decal             => $crate::assets::Decal { gen, external, id, useful_blank, validate: decal, refs: [("texture", "Texture")] },
            VolumetricFog     => $crate::assets::VolumetricFog { gen, external, useful_blank, validate: volumetric_fog },
            Joint             => $crate::assets::Joint { gen, external, id, validate: joint, refs: [("body_a", "Prop"), ("body_b", "Prop")] },
            ParticleEmitter   => $crate::assets::ParticleEmitter { gen, external, id, useful_blank, validate: particle_emitter, refs: [("texture", "Texture")] },
            WaterSurface      => $crate::assets::WaterSurface { gen, external, id, useful_blank, renders, validate: water_surface },
            SdfVolume         => $crate::assets::SdfVolume { manual, external, compiled, renders, validate: sdf_volume },
            GlassPanel        => $crate::assets::GlassPanel { gen, external, id, useful_blank, validate: glass_panel },
            LayoutContainer   => $crate::assets::LayoutContainer { gen, external, renders },
            PhysicsConfig     => $crate::assets::PhysicsConfig { gen, external, singleton },
            FpsCounter        => $crate::assets::FpsCounter { gen, external, useful_blank, refs: [("label", "TextLabel")] },
            StatHud           => $crate::assets::StatHud { gen, external, renders, refs: [("fps_label", "TextLabel"), ("vram_label", "TextLabel"), ("ram_label", "TextLabel"), ("ev_label", "TextLabel"), ("edr_label", "TextLabel")] },
            SceneImport       => $crate::assets::SceneImport { gen, build_only },
            MainMenu          => $crate::assets::MainMenu { gen, build_only, renders },
            OptionSelect      => $crate::assets::OptionSelect { gen, build_only },
            Slider            => $crate::assets::Slider { gen, build_only },
            ScrollPanel       => $crate::assets::ScrollPanel { gen, external, refs: [("screen", "Screen")] },
            ReflectionProbe   => $crate::assets::ReflectionProbe { gen, external, useful_blank, validate: reflection_probe },
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
            Spawner           => $crate::assets::Spawner { manual, external, args: SpawnerArgs },
            DebugHud          => $crate::assets::DebugHud { gen, external, renders, refs: [("passes_label", "TextLabel"), ("mouse_label", "TextLabel"), ("camera_label", "TextLabel"), ("sys_label", "TextLabel")] },
            EngineDefaults    => $crate::assets::EngineDefaults { gen, build_only },
            StoryImport       => $crate::assets::StoryImport { gen, build_only },
            AudioCue          => $crate::assets::AudioCue { gen, external, useful_blank, refs: [("clip", "AudioClip"), ("screen", "Screen")] },
            Story             => $crate::assets::Story { gen, external, id },
            Application       => $crate::assets::Application { manual, external, singleton, args: ApplicationArgs },
            AnimGraph         => $crate::assets::AnimGraph { gen, external, id },
            AnimParams        => $crate::assets::AnimParams { runtime, build: anim_params },
            CharacterRig      => $crate::assets::CharacterRig { runtime, build: character_rig },
            GroundProbes      => $crate::assets::GroundProbes { runtime },
            CameraProbe       => $crate::assets::CameraProbe { runtime },
            TextInput         => $crate::assets::TextInput { gen, external, id, useful_blank, renders, refs: [("font", "Font"), ("screen", "Screen")] },
            Panel             => $crate::assets::Panel { gen, build_only },
            Reaction          => $crate::assets::Reaction { gen, external, id },
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
            EnvironmentMap => $crate::assets::EnvironmentMap { resource: EnvironmentMap, compiled, renders },
            ColorLut => $crate::assets::ColorLut { resource: ColorLut, compiled },
            Font => $crate::assets::Font { resource: Font, compiled, useful_blank },
            Material => $crate::assets::Material { resource: Material, data, useful_blank },
            Mesh => $crate::assets::Mesh { resource: Mesh, compiled },
            SkinnedMesh => $crate::assets::SkinnedMesh { resource: SkinnedMesh, compiled },
        }
    };
}

// The runtime half: the `ComponentTag` enum, the `ComponentAsset` value enum,
// its blob loader, and the ECS storage. The authoring `ComponentType` registry
// is built from the same list in the build crate.
crate::for_each_component!(define_components);

// Generate the trivial `impl Component` blocks from the shared component list.
//
// The runtime trait is small: a NAME, a `from_baked` blob loader, and the
// optional identity / payload injection hooks. Most components are pure data
// whose impl is mechanical, generated here from each list entry's compact
// `{ ... }` metadata block. Entries whose impl is bespoke mark themselves
// `manual` and keep their impl; their trailing flags (origin, args type, refs)
// are authoring metadata consumed only by the build-side registry in
// concinnity-world.
//
// Metadata grammar (inside the braces):
//   manual, <flags...>          -- skip; the impl is hand-written elsewhere
//   gen, <flags...>             -- generated impl:
//     external | build_only | runtime -- the authoring origin (world-side only)
//     compiled                  -- an `inject_locator` that stores into
//                                  `self.locator` (and marks the payload
//                                  world-side)
//     id                        -- an `inject_name` that stores into
//                                  `self.asset_id`
//     singleton                 -- at most one instance belongs to a world
//                                  (world-side only)
//     useful_blank              -- meaningful when declared with only default
//                                  args, so authoring tools offer a plain add
//                                  (world-side only)
//     renders                   -- presence implies the world renders; drives
//                                  the GraphicsConfig companion injection at
//                                  build time (world-side only)
//     validate: <fn>            -- the bake-time validator (world-side only)
//     refs: [ ("field", "Type"), ... ] -- the reference fields (world-side only)
//     args: <Type>              -- the authored args schema when it differs
//                                  from the component (world-side only)
//   runtime [, build: <fn>]     -- RuntimeOnly: never authored, never in a
//                                  blob; the impl is NAME + the default
//                                  (rejecting) `from_baked`.
macro_rules! cn_impl_components {
    // Entry point: expand one impl per list entry.
    ( $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? ) => {
        $( cn_impl_components!(@one $variant $ty { $($meta)* }); )+
    };

    // Bespoke impls opt out here; trailing flags are world-side metadata.
    (@one $variant:ident $ty:path { manual $($rest:tt)* }) => {};

    // Generated impls: seed an empty method accumulator, then consume the flag
    // list one token at a time. Only `compiled` and `id` contribute runtime
    // code; the authoring flags are consumed (and used) by the world registry.
    (@one $variant:ident $ty:path { gen $($flags:tt)* }) => {
        cn_impl_components!(@munch $variant $ty [] $($flags)*);
    };

    // RuntimeOnly components: never authored in a world and never stored in a
    // blob, so the default (rejecting) `from_baked` is correct.
    (@one $variant:ident $ty:path { runtime $($rest:tt)* }) => {
        impl $crate::ecs::Component for $ty {
            const NAME: &'static str = stringify!($variant);
        }
    };

    (@munch $variant:ident $ty:path [$($body:tt)*] , compiled $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)*
             fn inject_locator(&mut self, locator: $crate::ecs::PayloadLocator) {
                 self.locator = Some(locator);
             }]
            $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] , id $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty
            [$($body)* fn inject_name(&mut self, id: $crate::ecs::asset_id::AssetId) {
                 self.asset_id = id;
             }]
            $($rest)*);
    };
    // Authoring-only flags: consumed here, used by the world registry.
    (@munch $variant:ident $ty:path [$($body:tt)*] , validate: $f:ident $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty [$($body)*] $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] , refs: [ $( ($fld:literal, $tgt:literal) ),+ $(,)? ] $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty [$($body)*] $($rest)*);
    };
    (@munch $variant:ident $ty:path [$($body:tt)*] , $flag:ident $($rest:tt)*) => {
        cn_impl_components!(@munch $variant $ty [$($body)*] $($rest)*);
    };

    // No flags left: emit the impl. The baked blob record carries the
    // serialized component itself.
    (@munch $variant:ident $ty:path [$($body:tt)*]) => {
        impl $crate::ecs::Component for $ty {
            const NAME: &'static str = stringify!($variant);
            $($body)*
            fn from_baked(bytes: &[u8]) -> Result<Self, $crate::result::CnResult> {
                Ok(postcard::from_bytes(bytes)?)
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
// an Events<T> queue: SceneCommand, ScreenCommand (once ViewCommand), SettingCommand, ControlsCommand,
// AudioCommand. Systems (once declarable, discriminants 128..255) are all
// internal now: GraphicsSystem, FpsCounter, Camera3DSystem, PhysicsSystem,
// UiInputSystem, AnimationSystem, AudioSystem, StatHud. FpsCounter / StatHud
// became components; PhysicsSystem's world config became the `PhysicsConfig`
// component. These names are kept as history; the component list above no longer
// carries their numbers.
