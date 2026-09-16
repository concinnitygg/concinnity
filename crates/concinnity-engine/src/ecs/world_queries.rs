//! Queries over a world that only a renderer-bearing runtime can answer. The
//! world itself is concinnity-core's and names no backend, no GPU profile, and
//! no streaming pool; each of these reads one of the resources this crate's
//! render band parks there, or the systems it built.

use concinnity_core::components::GraphicsConfig;
use concinnity_core::ecs::World;
use concinnity_core::render::backend::{GpuProfile, RenderBackend};
use concinnity_host::store::paths::StateTree;

use crate::animation::AnimationSystem;
use crate::app::budget::{MemoryBudget, ThreadBudget};
use crate::app::mem_drift::MemoryDrift;
use crate::ecs::ActiveRenderBackend;
use crate::gfx::streaming::system::{StreamingPressure, StreamingState, StreamingStats};
use crate::gfx::system::hot_reload_sources::HotReloadSources;
use crate::gfx::system::parked::{PushedFogSettings, TextureNameSlots};

/// Whether the world needs a renderer. True when it declares a
/// `GraphicsConfig` (pre-`start`) or has a constructed `GraphicsSystem`
/// (post-`start`, after the config component has been drained), so callers can
/// decide on the render loop regardless of timing.
pub fn renders(world: &World) -> bool {
    world.query::<GraphicsConfig>().next().is_some()
        || world.systems().iter().any(|s| {
            s.downcast_ref::<crate::gfx::system::GraphicsSystem>()
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
/// recreating the OS window / re-initializing the GPU device. `None` when the
/// world never built a backend (or it was already yielded).
pub fn take_render_backend(world: &mut World) -> Option<Box<dyn RenderBackend>> {
    world
        .resource_mut::<ActiveRenderBackend>()
        .and_then(|slot| slot.0.take())
}

/// Disjoint borrows of the parked render backend and the init-captured state the
/// `cn debug` drives edit the running world through. Each is `None` when init did
/// not park it; the backend also while a step has it taken, which is never the
/// case between ticks, where the drives run.
pub struct RenderHandoff<'a> {
    /// The live render backend.
    pub backend: Option<&'a mut (dyn RenderBackend + 'static)>,
    /// The fog settings last pushed to the backend.
    pub fog: Option<&'a mut PushedFogSettings>,
    /// The texture-name map, parked only under hot-reload capture.
    pub texture_slots: Option<&'a TextureNameSlots>,
}

/// Borrow the parked backend and the state beside it at once.
pub fn render_handoff(world: &mut World) -> RenderHandoff<'_> {
    let (_, resources) = world.systems_and_resources();
    let (backend, fog, texture_slots) =
        resources.get_disjoint_mut::<ActiveRenderBackend, PushedFogSettings, TextureNameSlots>();
    RenderHandoff {
        backend: backend.and_then(|slot| slot.0.as_deref_mut()),
        fog,
        texture_slots: texture_slots.map(|slots| &*slots),
    }
}

/// Take the hot-reload source catalogs graphics init parked, leaving none behind.
/// `None` under `cn run`, or when no file-backed asset or world.jsonl was declared.
pub fn take_hot_reload_sources(world: &mut World) -> Option<HotReloadSources> {
    world.remove_resource::<HotReloadSources>()
}

/// The world's AnimationSystem, when one was built.
pub fn animation_system_mut(world: &mut World) -> Option<&mut AnimationSystem> {
    world
        .systems_mut()
        .iter_mut()
        .find_map(|system| system.downcast_mut::<AnimationSystem>())
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
        world.add_component(GraphicsConfig::default());
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

    // A world that never built a backend has none to yield, and the handoff
    // borrow reports every parked piece absent.
    #[test]
    fn render_backend_accessors_without_a_backend() {
        let mut world = World::new();
        assert!(take_render_backend(&mut world).is_none());
        assert!(take_hot_reload_sources(&mut world).is_none());
        assert!(animation_system_mut(&mut world).is_none());

        let handoff = render_handoff(&mut world);
        assert!(handoff.backend.is_none());
        assert!(handoff.fog.is_none());
        assert!(handoff.texture_slots.is_none());
    }

    // The handoff reaches the parked fog and texture-name map together, and the
    // source catalogs are taken exactly once.
    #[test]
    fn render_handoff_borrows_the_parked_state_together() {
        let mut world = World::new();
        world.insert_resource(PushedFogSettings(None));
        world.insert_resource(TextureNameSlots(
            [(concinnity_core::ecs::asset_id::AssetId(3), 7)].into(),
        ));
        world.insert_resource(HotReloadSources::default());

        let handoff = render_handoff(&mut world);
        assert!(handoff.backend.is_none());
        let slots = handoff.texture_slots.expect("texture-name map parked");
        assert_eq!(slots.0.len(), 1);
        let fog = concinnity_core::render::volumetric_fog::resolve_asset(
            &concinnity_core::components::VolumetricFog {
                enabled: true,
                ..Default::default()
            },
        );
        handoff.fog.expect("fog parked").0 = fog;
        assert!(world.resource::<PushedFogSettings>().unwrap().0.is_some());

        assert!(take_hot_reload_sources(&mut world).is_some());
        assert!(take_hot_reload_sources(&mut world).is_none());
    }
}
