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
// name: its list lives in `concinnity_cook::authoring::registry::build_only`, and the
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
/// vocabulary in concinnity-cook) can hand its own list to the same callback.
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
                Camera3D          => $crate::components::Camera3D { manual, external, useful_blank, live, args: Camera3D },
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
                TextLabel         => $crate::components::TextLabel { gen, external, id, useful_blank, renders, live, refs: [("font", "Font"), ("screen", "Screen")] },
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
                Sprite            => $crate::components::Sprite { gen, external, id, useful_blank, renders, live, refs: [("texture", "Texture"), ("screen", "Screen")] },
                KeyBinding        => $crate::components::KeyBinding { gen, external, useful_blank, refs: [("screen", "Screen")], consumed },
                Screen            => $crate::components::Screen { gen, external, id, useful_blank, refs: [("focus", "TextInput")], consumed },
                Decal             => $crate::components::Decal { gen, external, id, useful_blank, validate: decal, refs: [("texture", "Texture")], consumed },
                VolumetricFog     => $crate::components::VolumetricFog { gen, external, useful_blank, validate: volumetric_fog, consumed },
                PhysicsJoint             => $crate::components::PhysicsJoint { gen, external, id, validate: joint, refs: [("body_a", "Prop"), ("body_b", "Prop")], consumed },
                ParticleEmitter   => $crate::components::ParticleEmitter { gen, external, id, useful_blank, validate: particle_emitter, refs: [("texture", "Texture")], consumed },
                WaterSurface      => $crate::components::WaterSurface { gen, external, id, useful_blank, renders, validate: water_surface, consumed },
                SdfVolume         => $crate::components::SdfVolume { manual, external, compiled, renders, validate_for: sdf_volume, consumed },
                GlassPanel        => $crate::components::GlassPanel { gen, external, id, useful_blank, validate: glass_panel, consumed },
                LayoutContainer   => $crate::components::LayoutContainer { gen, external, renders, live },
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
                TextInput         => $crate::components::TextInput { gen, external, id, useful_blank, renders, live, refs: [("font", "Font"), ("screen", "Screen")] },
                Behavior          => $crate::components::Behavior { gen, external, id, useful_blank, live },
                Variables         => $crate::components::Variables { gen, external, singleton, live },
                TriggerVolume     => $crate::components::TriggerVolume { gen, external, id, useful_blank },
                Hidden            => $crate::components::Hidden { runtime },
                LoadingOverlay    => $crate::components::LoadingOverlay { gen, external, singleton, renders, refs: [("screen", "Screen"), ("backdrop", "Sprite"), ("track", "Sprite"), ("fill", "Sprite"), ("label", "TextLabel")] },
                AudioOcclusionProbe => $crate::components::AudioOcclusionProbe { runtime },
                CharacterShape    => $crate::components::CharacterShape { gen, external, id, refs: [("target", "SkinnedMesh")] },
                EngineDefaults    => $crate::components::EngineDefaults { gen, external, singleton, consumed },
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
// concinnity-cook.
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
//     live                      -- the running world re-reads this column
//                                  every frame, so overwriting a component in
//                                  place takes effect without reloading the
//                                  world. Carries a second obligation: no
//                                  build-time expansion may read the type's
//                                  args, because an in-place write skips the
//                                  expansion entirely (world-side only)
//     consumed [: <Type>]       -- a load-time pass drains this column during
//                                  `World::start`, so it holds nothing from
//                                  the first tick; `: <Type>` names the
//                                  runtime component that survives in its
//                                  place (world-side only)
//     validate: <fn>            -- the bake-time validator (world-side only)
//     validate_for: <fn>        -- the bake-time validator for entries whose
//                                  clamp depends on the shader platform the
//                                  world is cooked for; it takes that platform
//                                  alongside the value (world-side only)
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
    (@munch $variant:ident $ty:path [$($body:tt)*] , validate_for: $f:ident $($rest:tt)*) => {
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
    use crate::blob::BlobAssetDef;
    use crate::components::{Prop, Transform};
    use crate::ecs::asset_id::AssetId;
    use crate::ecs::{
        AssetKind, ComponentAsset, ComponentStorage, ComponentTag, PayloadLocator, ResourceKind,
    };
    use crate::result::CnResult;
    use alloc::vec::Vec;

    // Both halves of the shared list, so the tests below drive the generated
    // per-entry arms over every entry rather than a hand-picked sample that
    // goes stale as the list grows.
    macro_rules! registry_names {
        (
            stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
            resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? } $(,)?
        ) => {
            const STORED: &[(ComponentTag, &str)] =
                &[$( (ComponentTag::$variant, stringify!($variant)) ),+];
            const RESOURCES: &[&str] = &[$( stringify!($rvariant) ),+];
        };
    }
    crate::for_each_component!(registry_names);

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

    // The tag, its authored name, and the enum discriminant are one fact in
    // three forms: the name round-trips through `parse`, and the discriminant
    // is the entry's position in the list.
    #[test]
    fn every_tag_names_itself_and_parses_back() {
        for (i, (tag, name)) in STORED.iter().enumerate() {
            assert_eq!(tag.as_str(), *name);
            assert_eq!(ComponentTag::parse(name), Some(*tag));
            assert_eq!(*tag as u8, i as u8, "{name} is not at its list position");
        }
    }

    #[test]
    fn a_name_no_component_carries_parses_to_nothing() {
        assert_eq!(ComponentTag::parse("NotAComponent"), None);
        assert_eq!(ComponentTag::parse(""), None);
    }

    // Every entry resolves, and the whole list stays inside the mask ceiling
    // that makes a tag usable as a ComponentId.
    #[test]
    fn every_tag_resolves_a_surviving_tag_and_fits_the_mask() {
        for (tag, name) in STORED {
            let surviving = tag.surviving_tag();
            assert!(
                surviving.is_none_or(|s| STORED.iter().any(|(t, _)| *t == s)),
                "{name} survives as a tag that is not in the list"
            );
        }
        assert!(
            STORED.len() < 128,
            "the list outgrew the ComponentMask ceiling"
        );
    }

    #[test]
    fn every_resource_name_resolves_to_a_handle_space() {
        for name in RESOURCES {
            assert!(
                ResourceKind::parse(name).is_some(),
                "{name} is a resource entry with no handle space"
            );
        }
        assert_eq!(ResourceKind::parse("Transform"), None);
        assert_eq!(ResourceKind::parse("NotAResource"), None);
    }

    #[test]
    fn a_loaded_component_reports_the_type_it_holds() {
        let asset = ComponentAsset::from(Transform::default());
        assert_eq!(asset.type_name(), "Transform");
        assert_eq!(asset.tag(), ComponentTag::Transform);
    }

    // Injection is a no-op for a type that overrides neither hook, so it lands
    // on the value without changing what the value holds.
    #[test]
    fn injection_dispatches_to_the_variant_it_holds() {
        let mut asset = ComponentAsset::from(Transform::default());
        asset.inject_name(AssetId(3));
        asset.inject_locator(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 0,
        });
        assert_eq!(asset.tag(), ComponentTag::Transform);
    }

    // A type declaring no clamp comes back unchanged; one declaring a clamp is
    // returned through it, and a `validate_for` entry reads the platform.
    #[test]
    fn validation_runs_only_where_an_entry_declares_a_clamp() {
        use crate::components::SdfVolume;
        use crate::platform::Platform;

        let plain = ComponentAsset::from(Transform::default()).validated(Platform::Metal);
        assert_eq!(plain.tag(), ComponentTag::Transform);

        let clamped = ComponentAsset::from(Prop::default()).validated(Platform::Metal);
        assert_eq!(clamped.tag(), ComponentTag::Prop);

        let volume = SdfVolume {
            fragment_shaders: Some(
                [
                    ("metal".into(), "blob.metal".into()),
                    ("hlsl".into(), "blob.hlsl".into()),
                ]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        };
        let ComponentAsset::SdfVolume(baked) =
            ComponentAsset::from(volume).validated(Platform::Hlsl)
        else {
            panic!("the value keeps its variant through validation");
        };
        assert_eq!(baked.fragment_shader, "blob.hlsl");
    }

    fn baked(discriminant: u8, args_bytes: Vec<u8>, name: Option<AssetId>) -> BlobAssetDef {
        BlobAssetDef {
            name,
            kind: AssetKind::Component,
            discriminant,
            args_bytes,
            payload: None,
        }
    }

    // The record's discriminant picks the type, the bytes rebuild the value,
    // and a named record has its identity injected on the way out.
    #[test]
    fn a_baked_record_loads_as_the_component_its_discriminant_names() {
        let prop = Prop {
            position: [1.0, 2.0, 3.0],
            ..Prop::default()
        };
        let bytes = postcard::to_allocvec(&prop).expect("a prop encodes");
        let asset =
            ComponentAsset::from_baked(&baked(ComponentTag::Prop as u8, bytes, Some(AssetId(7))))
                .expect("the record loads");
        let ComponentAsset::Prop(loaded) = asset else {
            panic!("expected a prop");
        };
        assert_eq!(loaded.position, [1.0, 2.0, 3.0]);
        assert_eq!(loaded.asset_id, AssetId(7));
    }

    #[test]
    fn a_record_no_tag_claims_is_rejected() {
        assert_eq!(
            ComponentAsset::from_baked(&baked(u8::MAX, Vec::new(), None)).err(),
            Some(CnResult::AssetInvalidType)
        );
    }

    // Storage dispatch: pushing through the value enum lands in the column the
    // variant names, replacing overwrites it in place, and an entity holding no
    // component of that type reports so rather than gaining one.
    #[test]
    fn the_value_enum_pushes_replaces_and_counts_through_its_column() {
        let mut storage = ComponentStorage::default();
        let entity = storage.push(ComponentAsset::from(Transform::default()));
        assert_eq!(
            storage.entities_with_tag(ComponentTag::Transform as u8),
            &[entity]
        );
        assert_eq!(
            storage.component_census(),
            alloc::vec![(ComponentTag::Transform as u8, 1)]
        );

        let moved = Transform {
            position: [4.0, 5.0, 6.0],
            ..Transform::default()
        };
        assert!(storage.replace(entity, ComponentAsset::from(moved)));
        assert_eq!(
            storage.get::<Transform>(entity).map(|t| t.position),
            Some([4.0, 5.0, 6.0])
        );

        // The entity carries no Prop, so there is nothing of that type to
        // overwrite.
        assert!(!storage.replace(entity, ComponentAsset::from(Prop::default())));
        assert!(storage.entities_with_tag(u8::MAX).is_empty());
    }
}
