// src/ecs/world_queries.rs
//
// Queries over a world that only a renderer-bearing runtime can answer. The
// world itself is concinnity-core's and names no backend, no GPU profile, and
// no streaming pool; each of these reads one of the resources this crate's
// render band parks there, or the systems it built.

use crate::app::budget::{MemoryBudget, ThreadBudget};
use crate::app::mem_drift::MemoryDrift;
use crate::ecs::{ActiveRenderBackend, World};
use crate::gfx::backend::{GpuProfile, RenderBackend};
use crate::gfx::streaming_system::{StreamingPressure, StreamingState, StreamingStats};
use concinnity_host::store::paths::StateTree;

/// Whether the world needs a renderer. True when it declares a
/// `GraphicsConfig` (pre-`start`) or has a constructed `GraphicsSystem`
/// (post-`start`, after the config component has been drained), so callers can
/// decide on the render loop regardless of timing.
pub fn renders(world: &World) -> bool {
    world
        .query::<crate::components::GraphicsConfig>()
        .next()
        .is_some()
        || world.systems().iter().any(|s| {
            s.downcast_ref::<crate::gfx::graphics_system::GraphicsSystem>()
                .is_some()
        })
}

/// Per-pool `(resident, pending, unloaded)` streaming counts from the parked
/// `StreamingState` (StreamingSystem drives it against the backend each
/// frame). `None` before graphics init parks it, and from inside a system
/// step, which takes the state out. Read by the `cn debug` server's
/// `streaming` command and the editor's Health panel.
pub fn streaming_stats(world: &World) -> Option<StreamingStats> {
    world
        .resource::<StreamingState>()
        .map(|s| s.streaming_stats())
}

/// Live process-RAM back-off pressure on streaming, published by
/// StreamingSystem on its throttled RSS sample. `None` before the first sample
/// or when no `MemoryBudget` / RSS is available (the valve is inert).
pub fn streaming_pressure(world: &World) -> Option<StreamingPressure> {
    world.resource::<StreamingPressure>().copied()
}

/// Long-session memory drift, folded from the same throttled sample as the
/// back-off valve. `None` until the session settles enough for a baseline, and
/// for the same reasons `streaming_pressure` is absent.
pub fn memory_drift(world: &World) -> Option<MemoryDrift> {
    world.resource::<MemoryDrift>().copied()
}

/// The detected GPU's capability + memory profile, published by graphics init.
/// `None` before init runs, and `GpuProfile::UNKNOWN` when the backend could
/// not classify the device.
pub fn gpu_profile(world: &World) -> Option<GpuProfile> {
    world.resource::<GpuProfile>().copied()
}

/// The state tree `App::start` published: where this world reads and writes.
/// `None` for a world running against no tree, which is a world that touches no
/// disk. What every system reads instead of resolving a path of its own.
pub fn state_tree(world: &World) -> Option<&StateTree> {
    world.resource::<StateTree>()
}

/// The process thread budget App published at start. `None` before `App::start`
/// installs it. Read by the `cn debug` server's `budget` command.
pub fn thread_budget(world: &World) -> Option<ThreadBudget> {
    world.resource::<ThreadBudget>().copied()
}

/// The world's memory budget, once `start` has published one.
pub fn memory_budget(world: &World) -> Option<MemoryBudget> {
    world.resource::<MemoryBudget>().copied()
}

/// Take the live render backend out of the world's parked slot, leaving the
/// world backend-less. The `cn editor` live SAVE swap transplants it into the
/// rebuilt world (via a `PendingBackend` resource) so the edit applies without
/// recreating the OS window / re-initialising the GPU device. `None` when the
/// world never built a backend (or it was already yielded).
pub fn take_render_backend(world: &mut World) -> Option<Box<dyn RenderBackend>> {
    world
        .resource_mut::<ActiveRenderBackend>()
        .and_then(|slot| slot.0.take())
}

/// Disjoint mutable borrows of the system list and the parked render backend,
/// for the `cn debug` hot-reload drive: it applies backend edits through a
/// system's init-captured bookkeeping, so it needs both at once. The backend is
/// `None` while a step has it taken (never the case between ticks, where the
/// drive runs) or when no backend was built.
pub fn systems_and_render_backend(
    world: &mut World,
) -> (
    &mut [crate::ecs::BuiltSystem],
    Option<&mut (dyn RenderBackend + 'static)>,
) {
    let (systems, resources) = world.systems_and_resources();
    let backend = resources
        .get_mut::<ActiveRenderBackend>()
        .and_then(|slot| slot.0.as_deref_mut());
    (systems, backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A GraphicsConfig marks a rendering world. `renders` reports it before
    // `start` (while the component is present), the pre-start signal callers
    // use to choose the render loop. (The post-start GraphicsSystem path can't
    // be unit-tested here: its `init` builds the GPU backend.)
    #[test]
    fn graphics_config_makes_world_render() {
        let mut world = World::new();
        assert!(!renders(&world));
        world.add_component(crate::components::GraphicsConfig::default());
        assert!(renders(&world));
    }

    // The streaming readouts are `None` until graphics init parks the state, so
    // a world that never built a backend reports nothing rather than panicking.
    #[test]
    fn streaming_readouts_are_absent_before_graphics_init() {
        let world = World::new();
        assert!(streaming_stats(&world).is_none());
        assert!(streaming_pressure(&world).is_none());
    }

    // A world that never built a backend has none to yield, and the disjoint
    // borrow still hands back the (empty) system list.
    #[test]
    fn render_backend_accessors_without_a_backend() {
        let mut world = World::new();
        assert!(take_render_backend(&mut world).is_none());

        let (systems, backend) = systems_and_render_backend(&mut world);
        assert!(systems.is_empty());
        assert!(backend.is_none());
    }
}
