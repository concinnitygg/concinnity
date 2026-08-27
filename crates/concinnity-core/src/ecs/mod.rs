//! The renderer-free half of the engine's ECS: the storage mechanism, the
//! per-tick context systems see, and the asset identity + registry layer the
//! build / validate pipeline and the client runtime share.
//!
//! The storage mechanism is closed-world and carries no engine domain type. It
//! provides the generic primitives only: entities ([`Entity`], [`Entities`]),
//! typed storage columns ([`Column`]), change ticks ([`Tick`]), component masks
//! ([`ComponentMask`]) and a join index ([`JoinIndex`]), resources
//! ([`Resources`]), events ([`Events`]), and the per-system access sets
//! ([`Access`]) the scheduler uses to run two systems concurrently. Nothing in
//! that half knows about meshes, blobs, or rendering, and none of it stores a
//! type the project did not register at compile time: there is no TypeId-keyed
//! type erasure and no open-world insert of arbitrary external types.
//!
//! The concrete component set is registered in `registry` and expanded by
//! [`define_components!`](crate::define_components), which pairs the asset-enum
//! dispatch with the storage half from
//! [`define_component_storage!`](crate::define_component_storage).
//!
//! On top of that sit the pieces the engine reaches for: the [`Component`]
//! metadata trait, the plain data types the registry and blob format are built
//! from ([`AssetOrigin`], [`AssetPayload`], [`PayloadLocator`],
//! [`BlobAssetDef`], [`AssetKind`]), the [`System`] behavior trait, and the
//! [`World`]: the components, resources, events, payloads, profile, and frame
//! scratch a tick reads and writes, plus the systems that run over them and
//! their schedule. The system table itself ([`SystemTable`]) names a host's own
//! system types, so it is written in the client crate, whose `ecs` module
//! re-exports everything here under the historical `crate::ecs::*` paths
//! alongside the `ComponentAsset` value enum.
//!
//! The interner that assigns asset identities keeps a per-thread table and
//! lives in concinnity-host. The authoring `Registration` record lives in
//! concinnity-world, constructed from the trait's metadata consts.

pub mod access_check;
pub mod asset_id;

mod access;
mod built_system;
mod clock;
mod column;
mod component;
mod context;
mod define_components;
mod entity;
mod entity_by_name;
mod event;
mod event_store;
mod frame;
mod headless;
mod join;
mod mask;
mod payload_store;
mod protocol;
mod registry;
mod resource;
mod storage;
mod system;
mod system_entry;
mod tick;
mod waves;
mod world;

#[cfg(test)]
mod join_bench;
#[cfg(test)]
mod storage_bench;
#[cfg(test)]
mod world_run_tests;

// The storage primitives. `Column`, `Entities`, `JoinIndex` and `AtomicTick`
// are named by the expansion of `define_component_storage!`, so they are public
// here for every crate that expands it.
pub use access::Access;
pub use column::{Column, ColumnTicks};
pub use entity::{Entities, Entity};
pub use event::{EventCursor, Events};
pub use event_store::EventStore;
pub use join::JoinIndex;
pub use mask::{ComponentId, ComponentMask};
pub use resource::Resources;
pub use tick::{AtomicTick, MAX_CHANGE_AGE, Tick};

// The runtime-facing component contract and the metadata enums the registry and
// the blob format are built from.
pub use component::{AssetOrigin, AssetPayload, Component, ResourceAsset, RuntimeComponent};

// Systems' view of the world during a tick.
pub use context::PipelineContext;

// Per-frame facilities carried on `PipelineContext`. Re-exported by the client
// `ecs` module under the historical `crate::ecs::*` paths, like the rest.
// `Arena` comes with it so a crate that only builds a context (the physics and
// audio subsystems, and every test world) can name the scratch type without
// taking its own dependency on the allocation layer.
pub use concinnity_memory::Arena;
pub use frame::{FrameContext, FrameVec};

// Renderer-free resources the runtime systems publish and read to coordinate a
// tick (menu state, frame-rate cap, HUD prefs, cursor + dropdown views), plus
// the world's cook-counted physics reservation. They name no renderer type, so
// they live here where the physics / audio subsystem crates can reach them; the
// client `ecs` module re-exports them under the historical `crate::ecs::*`
// paths.
pub use protocol::{
    CursorShape, CursorState, DesiredCursor, DropdownView, ExecutionTrace, FlyCam, FrameRateCap,
    GpuMemoryPressure, HiddenAssets, HudLayers, HudPrefs, MenuActive, MenuOverride, OpenDropdown,
    OverlayImage, OverlayImages, PickEntry, PickIndex, ScheduleMode, ScreenStack, SimTiming,
    TraceEvent, TracePath, TracePaths, TraceRequest, TraceStep, TraceVal, TransientSaves,
    ViewOverrides, WorldLines, WorldPhysicsBudget,
};

// The runtime behavior trait every engine system implements + its per-step
// control signal. Renderer-free (they name only `PipelineContext`), so they live
// here for the physics / audio subsystem crates; the client `ecs` module
// re-exports them under the historical `crate::ecs::*` paths and its
// `define_systems!` table names the gate that builds each one.
pub use system::{StepResult, System};

// The name -> Entity index the load-time Prop decomposition pass publishes.
// Renderer-free, so the physics / audio subsystem crates can resolve a name
// reference to an Entity through it; the client `ecs::decompose` module
// re-exports it under the historical `crate::ecs::decompose::EntityByName` path.
pub use entity_by_name::EntityByName;

// The payload-access seam systems reach through: keeps the storage mechanism
// free of blob file I/O (`concinnity_host::store`'s `BlobData` is the runtime
// implementor).
pub use payload_store::{NoPayloads, PayloadStore};

// A world, the systems built over it, and the table a host starts it from: one
// entry per system in run order, plus the load-time passes bracketing them.
// This crate writes one such table itself, listing the simulation systems it
// owns, for a world that runs with no host beyond it.
pub use built_system::BuiltSystem;
pub use headless::HEADLESS_SYSTEMS;
pub use system_entry::{SystemEntry, SystemTable};
pub use world::{ScratchStats, World};

// The host-installed monotonic clock the step loop times systems with.
pub use clock::Clock;

// Runtime asset-registry types, generated by the macros in `define_components`
// (invoked in `registry`). Re-exported here so the rest of the crate (and the
// client, which re-exports this module under `crate::ecs::*`) can keep using
// the historical `crate::ecs::*` paths. The authoring `RegisteredType` registry
// is built from the same component list in the build crate. Systems have no
// registry here: they are built client-side from the `System` behavior trait.
pub use registry::{ComponentAsset, ComponentSlot, ComponentStorage, ComponentTag};

// Points to an asset's compiled binary payload within the data blob files.
// Defined in the schema crate because blob-backed asset structs carry it as a
// `#[serde(skip)]` field; re-exported here under its historical path.
pub use concinnity_asset::PayloadLocator;

// Per-kind resource handles (dense per-kind indices into the runtime resource
// tables), defined in the schema crate alongside `AssetId`. Cook assigns them;
// components and the resource tables address resources by them.
pub use concinnity_asset::{
    AudioClipHandle, ColorLutHandle, CubemapTextureHandle, EnvironmentMapHandle, FontHandle,
    MaterialHandle, MeshHandle, ShaderHandle, SkinnedMeshHandle, TextureHandle,
    de_audio_clip_handle_vec, de_opt_audio_clip_handle, de_opt_font_handle, de_opt_material_handle,
    de_opt_mesh_handle, de_opt_shader_handle, de_opt_skinned_mesh_handle, de_opt_texture_handle,
    set_audio_clip_handle_resolver, set_font_handle_resolver, set_material_handle_resolver,
    set_mesh_handle_resolver, set_shader_handle_resolver, set_skinned_mesh_handle_resolver,
    set_texture_handle_resolver,
};

// The blob record schema (the component defs stream + the resource stream) is
// owned by the `blob` format module; re-exported here so the runtime, cook, and
// the registry macros keep naming `ecs::{BlobAssetDef, ResourceKind, ...}`
// unchanged.
pub use crate::blob::{
    AssetKind, BlobAssetDef, BlobMeta, MeshBoundsRecord, PhysicsBudgetRecord, ResourceKind,
    ResourceRecord, SceneGroup,
};
