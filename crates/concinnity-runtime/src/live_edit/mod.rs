//! Live editing of a running world: the source catalogs a hot reload works
//! from, and the render backend handoff that carries a window and device into a
//! rebuilt world.
//!
//! Every entry point here assumes one ordering, which the runtime's own systems
//! rely on:
//!
//! - When the launch request arms the dev loop, graphics init captures the
//!   source catalogs ([`hot_reload_sources`]) and parks them as world
//!   resources beside the render backend (core's `Parked` slot), with the
//!   texture-name map and the pushed fog settings.
//! - A host reaches the parked backend only on the main thread, between world
//!   steps; from inside a step it is taken.
//! - A [`PendingBackend`] inserted into a freshly built world is consumed by
//!   that world's graphics init, which reloads the world into it instead of
//!   building a new backend.

use concinnity_engine::render::backend::RenderBackend;

pub mod hot_reload_sources;
pub mod shader_overrides;

/// A render backend transplanted out of a previous world, inserted into a
/// freshly built world so its graphics init reuses the live GPU device and
/// window. Init calls `RenderBackend::reload_world` on it when the swapchain
/// config is unchanged, and otherwise idles and drops it and builds a new one.
pub struct PendingBackend(pub Box<dyn RenderBackend>);
