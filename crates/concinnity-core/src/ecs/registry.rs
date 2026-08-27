// src/ecs/registry.rs
//
// Single source of truth for the renderer-free half of the engine's asset
// registry: every Component type paired with its stable u8 discriminant.
//
// The list lives in one macro, `for_each_component!`, so both registries built
// from it stay in lockstep: the runtime value enum + ECS storage
// (`define_components!`, invoked below in this crate) and the authoring metadata
// registry (`RegisteredType`, invoked in the build crate from the same list).
//
// It arrives in two groups, and which group an entry is in is the whole of what
// separates a component from a resource:
//
//   stored    has a `ComponentTag`, a `ComponentAsset` variant, a column, an
//             `impl Component`, and an `impl RuntimeComponent`, so a world can
//             hold one. Split further by the entry's own origin flag into
//             `external` (declared in a world, survives into a blob) and
//             `runtime` (only ever minted by a running world).
//   resource  declared in a world and compiled into the blob's resource stream,
//             addressed at runtime by a per-kind handle rather than a column.
//             `impl ResourceAsset`; the entry names its `ResourceKind`. Carries
//             no origin flag, because being in the group is the origin.
//
// A third group -- the types a world declares and the cook expands away before a
// blob is written -- is the authoring vocabulary, which this crate does not
// name: its list lives in `concinnity_world::registry::build_only`, and the
// authoring registry composes the two by passing it through the `$extra` tail
// below.
//
// Components are pure data, registered with one entry each. There is no system
// registry: every system is internal code, constructed at runtime from world
// content (see `World::start`), never declared in a world or serialized to a
// blob. The table that gates and orders the constructed systems is the caller's:
// this crate's headless table, or the client crate's `ecs::registry`.
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

/// The one component list. `$cb` is a macro that receives the `Variant => Type`
/// entries and expands to whatever registry it builds from them. Type paths are
/// absolute so the list resolves from any crate that consumes it.
///
/// The `$cb; $extra` form prepends arbitrary tokens to what `$cb` receives, so a
/// consumer holding a group this crate does not name (the authoring-only
/// vocabulary in concinnity-world) can hand its own list to the same callback.
#[macro_export]
macro_rules! for_each_component {
    ($cb:ident) => { $crate::for_each_component!($cb;); };
    ($cb:ident; $($extra:tt)*) => {
        $cb! {
            $($extra)*
            // Stored: every type with a column, a `ComponentTag`, and a
            // `ComponentAsset` variant. `external` entries are declared in a
            // world and survive into a blob; `runtime` entries are only ever
            // minted by a running world.
            stored: {
                Window            => $crate::components::Window { gen, external, singleton, consumed },
                GraphicsConfig    => $crate::components::GraphicsConfig { gen, external, singleton, renders, consumed },
                Shader            => $crate::components::Shader { manual, external, compiled, consumed },
                Camera3D          => $crate::components::Camera3D { manual, external, useful_blank, args: Camera3D },
                FrameInput        => $crate::components::FrameInput { gen, runtime },
                Prop              => $crate::components::Prop { gen, external, id, renders, validate: prop, refs: [("model", "Model"), ("material", "Material"), ("texture", "Texture"), ("scene", "Scene"), ("parent", "Prop")], consumed: PropInstance },
                RigidBody         => $crate::components::RigidBody { gen, external, validate: rigid_body },
                PropBody          => $crate::components::PropBody { gen, external, consumed },
                Room              => $crate::components::Room { manual, external, compiled, useful_blank, args: Room, refs: [("texture", "Texture"), ("wall_texture", "Texture"), ("floor_texture", "Texture"), ("ceiling_texture", "Texture")], consumed },
                DirectionalLight  => $crate::components::DirectionalLight { gen, external, useful_blank, validate: directional_light },
                PointLight        => $crate::components::PointLight { gen, external, useful_blank, validate: point_light },
                SpotLight         => $crate::components::SpotLight { gen, external, useful_blank, validate: spot_light },
                RectAreaLight     => $crate::components::RectAreaLight { gen, external, useful_blank, validate: rect_area_light },
                ProceduralMesh    => $crate::components::ProceduralMesh { gen, external, compiled, id },
                Model             => $crate::components::Model { gen, external, id, consumed },
                Scene             => $crate::components::Scene { gen, external, id, refs: [("camera_shot", "Camera3D")], consumed },
                TextLabel         => $crate::components::TextLabel { gen, external, id, useful_blank, renders, refs: [("font", "Font"), ("screen", "Screen")] },
                HitRegion         => $crate::components::HitRegion { gen, external, useful_blank, refs: [("label", "TextLabel"), ("screen", "Screen")], consumed },
                File              => $crate::components::File { manual, external, compiled, args: File, consumed },
                BlockType         => $crate::components::BlockType { gen, external, id, useful_blank, consumed },
                VoxelChunk        => $crate::components::VoxelChunk { gen, external, compiled, id, validate: voxel_chunk, consumed },
                InstancedProp     => $crate::components::InstancedProp { gen, external, id, renders, validate: instanced_prop, refs: [("material", "Material"), ("texture", "Texture")], consumed },
                PostProcessConfig => $crate::components::PostProcessConfig { manual, external, singleton, consumed },
                Animation         => $crate::components::Animation { gen, external, id, consumed },
                SkeletonPose      => $crate::components::SkeletonPose { runtime, build: skeleton_pose },
                StreamingConfig   => $crate::components::StreamingConfig { gen, external, singleton, consumed },
                VoxelWorld        => $crate::components::VoxelWorld { gen, external, renders, refs: [("material", "Material")], consumed },
                AudioEmitter      => $crate::components::AudioEmitter { gen, external, useful_blank, refs: [("clip", "AudioClip"), ("prop", "Prop")] },
                Sprite            => $crate::components::Sprite { gen, external, id, useful_blank, renders, refs: [("texture", "Texture"), ("screen", "Screen")] },
                KeyBinding        => $crate::components::KeyBinding { gen, external, useful_blank, refs: [("screen", "Screen")], consumed },
                Screen            => $crate::components::Screen { gen, external, id, useful_blank, refs: [("focus", "TextInput")], consumed },
                Decal             => $crate::components::Decal { gen, external, id, useful_blank, validate: decal, refs: [("texture", "Texture")], consumed },
                VolumetricFog     => $crate::components::VolumetricFog { gen, external, useful_blank, validate: volumetric_fog, consumed },
                PhysicsJoint             => $crate::components::PhysicsJoint { gen, external, id, validate: joint, refs: [("body_a", "Prop"), ("body_b", "Prop")], consumed },
                ParticleEmitter   => $crate::components::ParticleEmitter { gen, external, id, useful_blank, validate: particle_emitter, refs: [("texture", "Texture")], consumed },
                WaterSurface      => $crate::components::WaterSurface { gen, external, id, useful_blank, renders, validate: water_surface, consumed },
                SdfVolume         => $crate::components::SdfVolume { manual, external, compiled, renders, validate: sdf_volume, consumed },
                GlassPanel        => $crate::components::GlassPanel { gen, external, id, useful_blank, validate: glass_panel, consumed },
                LayoutContainer   => $crate::components::LayoutContainer { gen, external, renders },
                PhysicsConfig     => $crate::components::PhysicsConfig { gen, external, singleton },
                FpsCounter        => $crate::components::FpsCounter { gen, external, useful_blank, refs: [("label", "TextLabel")] },
                StatHud           => $crate::components::StatHud { gen, external, renders, refs: [("fps_label", "TextLabel"), ("vram_label", "TextLabel"), ("ram_label", "TextLabel"), ("ev_label", "TextLabel"), ("edr_label", "TextLabel")] },
                ScrollPanel       => $crate::components::ScrollPanel { gen, external, refs: [("screen", "Screen")], consumed },
                ReflectionProbe   => $crate::components::ReflectionProbe { gen, external, useful_blank, validate: reflection_probe },
                Transform         => $crate::components::Transform { runtime },
                PropInstance      => $crate::components::PropInstance { runtime },
                MeshRenderer      => $crate::components::MeshRenderer { runtime },
                ModelRenderer     => $crate::components::ModelRenderer { runtime },
                Collider          => $crate::components::Collider { runtime },
                BodyDynamics      => $crate::components::BodyDynamics { runtime },
                Interactable      => $crate::components::Interactable { runtime },
                Pickup            => $crate::components::Pickup { runtime },
                Parent            => $crate::components::Parent { runtime },
                Children          => $crate::components::Children { runtime },
                SceneMember       => $crate::components::SceneMember { runtime },
                GlobalTransform   => $crate::components::GlobalTransform { runtime },
                RenderHandle      => $crate::components::RenderHandle { runtime },
                Held              => $crate::components::Held { runtime },
                Lifetime          => $crate::components::Lifetime { runtime },
                Spawner           => $crate::components::Spawner { manual, external, args: Spawner },
                DebugHud          => $crate::components::DebugHud { gen, external, renders, refs: [("passes_label", "TextLabel"), ("mouse_label", "TextLabel"), ("camera_label", "TextLabel"), ("sys_label", "TextLabel")] },
                AudioCue          => $crate::components::AudioCue { gen, external, useful_blank, refs: [("clip", "AudioClip"), ("screen", "Screen")] },
                Story             => $crate::components::Story { gen, external, id },
                AppConfig         => $crate::components::AppConfig { manual, external, singleton, args: AppConfig },
                AnimationGraph         => $crate::components::AnimationGraph { gen, external, id, consumed },
                AnimationParams        => $crate::components::AnimationParams { runtime, build: anim_params },
                CharacterRig      => $crate::components::CharacterRig { runtime, build: character_rig },
                GroundProbes      => $crate::components::GroundProbes { runtime },
                CameraProbe       => $crate::components::CameraProbe { runtime },
                TextInput         => $crate::components::TextInput { gen, external, id, useful_blank, renders, refs: [("font", "Font"), ("screen", "Screen")] },
                Behavior          => $crate::components::Behavior { gen, external, id, useful_blank },
                Variables         => $crate::components::Variables { gen, external, singleton },
                TriggerVolume     => $crate::components::TriggerVolume { gen, external, id, useful_blank },
                Hidden            => $crate::components::Hidden { runtime },
                LoadingOverlay    => $crate::components::LoadingOverlay { gen, external, singleton, renders, refs: [("screen", "Screen"), ("backdrop", "Sprite"), ("track", "Sprite"), ("fill", "Sprite"), ("label", "TextLabel")] },
                AudioOcclusionProbe => $crate::components::AudioOcclusionProbe { runtime },
                CharacterShape    => $crate::components::CharacterShape { gen, external, id, refs: [("target", "SkinnedMesh")] },
            },

            // Resource: declared in a world and compiled into the blob's
            // resource stream, addressed at runtime by a per-kind handle
            // rather than stored in a column. Each entry names the dense
            // handle space it is assigned into.
            resource: {
                AudioClip => $crate::components::AudioClip { resource: AudioClip, compiled },
                Texture => $crate::components::Texture { resource: Texture, compiled },
                CubemapTexture => $crate::components::CubemapTexture { resource: CubemapTexture, compiled },
                EnvironmentMap => $crate::components::EnvironmentMap { resource: EnvironmentMap, compiled, renders },
                ColorLut => $crate::components::ColorLut { resource: ColorLut, compiled },
                Font => $crate::components::Font { resource: Font, compiled, useful_blank },
                Material => $crate::components::Material { resource: Material, data, useful_blank, refs: [("albedo", "Texture"), ("normal_map", "Texture"), ("emissive_map", "Texture"), ("orm_map", "Texture"), ("albedo_secondary", "Texture"), ("normal_secondary", "Texture"), ("shader", "Shader")] },
                Mesh => $crate::components::Mesh { resource: Mesh, compiled },
                SkinnedMesh => $crate::components::SkinnedMesh { resource: SkinnedMesh, compiled, renders },
            },
        }
    };
}

// The runtime half: the `ComponentTag` enum, the `ComponentAsset` value enum,
// its blob loader, and the ECS storage. The authoring `RegisteredType` registry
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
//     external | runtime        -- the authoring origin (world-side only)
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
//     consumed [: <Type>]       -- a load-time pass drains this column during
//                                  `World::start`, so it holds nothing from
//                                  the first tick; `: <Type>` names the
//                                  runtime component that survives in its
//                                  place (world-side only)
//     validate: <fn>            -- the bake-time validator (world-side only)
//     refs: [ ("field", "Type"), ... ] -- the reference fields (world-side only)
//     args: <Asset>             -- names the asset whose authored schema
//                                  differs from the component it bakes into;
//                                  the schema is that asset's `cook` form
//                                  (world-side only)
//   runtime [, build: <fn>]     -- RuntimeOnly: never authored, never in a
//                                  blob; the impl is NAME + the default
//                                  (rejecting) `from_baked`.
macro_rules! cn_impl_components {
    // Entry point: one impl per stored entry. The resource group is skipped
    // whole -- a resource is never loaded from a component record.
    (
        stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
        resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? } $(,)?
    ) => {
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
    (@munch $variant:ident $ty:path [$($body:tt)*] , consumed: $surviving:ident $($rest:tt)*) => {
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
                Ok($crate::blob::decode_exact(bytes)?)
            }
        }
    };
}

// The generated trivial `impl Component` blocks: one per list entry marked
// `gen`, expanded from its metadata. Entries marked `manual` keep the
// hand-written impl in their own `components` module. Emitted here (rather than
// in `components`) so the macro is in textual scope, alongside
// `define_components`.
crate::for_each_component!(cn_impl_components);

#[cfg(test)]
mod tests {
    use crate::ecs::ComponentTag;

    #[test]
    fn an_undrained_component_survives_as_itself() {
        assert_eq!(
            ComponentTag::Transform.surviving_tag(),
            Some(ComponentTag::Transform)
        );
        assert_eq!(
            ComponentTag::Behavior.surviving_tag(),
            Some(ComponentTag::Behavior)
        );
    }

    #[test]
    fn a_consumed_component_survives_as_nothing() {
        assert_eq!(ComponentTag::Screen.surviving_tag(), None);
        assert_eq!(ComponentTag::PropBody.surviving_tag(), None);
    }

    // The one entry whose `consumed` flag names a replacement: the flag is what
    // keeps "Prop" a usable behavior scope after decomposition drains it.
    #[test]
    fn a_consumed_component_can_name_its_replacement() {
        assert_eq!(
            ComponentTag::Prop.surviving_tag(),
            Some(ComponentTag::PropInstance)
        );
    }
}
