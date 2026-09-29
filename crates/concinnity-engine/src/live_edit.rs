//! Live editing of a running world: the previews that apply an authoring edit
//! without a rebuild, the source catalogs and parked state a hot reload works
//! from, and the render backend handoff that carries a window and device into a
//! rebuilt world.
//!
//! Every entry point here assumes one ordering, which the engine's own systems
//! rely on:
//!
//! - When the launch request arms the dev loop, graphics init captures the
//!   source catalogs ([`hot_reload_sources`], [`shader_sources`],
//!   [`sdf_field_sources`]) and parks them with the state in [`parked`] as
//!   world resources beside the render backend.
//! - A host calls in only on the main thread, between world steps, before the
//!   next step runs. The backend is parked there, so [`render_handoff`] and
//!   [`shader_reload_flag`] reach it; from inside a step it is taken.
//! - A preview records its backend calls into the frame's render op queue,
//!   which submission replays before the next draw, and keeps the matching ECS
//!   state in step so the running world agrees with what is drawn.
//! - A [`PendingBackend`] inserted into a freshly built world is consumed by
//!   that world's graphics init, which reloads the world into it instead of
//!   building a new backend.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use concinnity_core::ecs::World;
use concinnity_core::render::backend::RenderBackend;

use crate::ecs::ActiveRenderBackend;
use crate::gfx::system::hot_reload_sources::HotReloadSources;
use crate::gfx::system::parked::{PushedFogSettings, TextureNameSlots};

pub use crate::animation::AnimationReloadEntry;
pub use crate::ecs::take_render_backend;
pub use crate::gfx::system::{hot_reload_sources, parked, sdf_field_sources, shader_sources};
pub use crate::gfx::{draw_preview, lighting_preview, material_preview, shape_preview};

/// A render backend transplanted out of a previous world, inserted into a
/// freshly built world so its graphics init reuses the live GPU device and
/// window. Init calls `RenderBackend::reload_world` on it when the swapchain
/// config is unchanged, and otherwise idles and drops it and builds a new one.
pub struct PendingBackend(pub Box<dyn RenderBackend>);

/// Disjoint borrows of the parked render backend and the init-captured state
/// beside it. Each is `None` when init did not park it; the backend also while
/// a step has it taken.
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

/// Take the source catalogs graphics init parked, leaving none behind. `None`
/// when capture was off, or when the world declared no file-backed asset.
pub fn take_hot_reload_sources(world: &mut World) -> Option<HotReloadSources> {
    world.remove_resource::<HotReloadSources>()
}

/// The parked backend's shader-reload flag, which a host sets to have the
/// backend recompile its shaders before the next draw. `None` when no backend
/// is parked or it does not reload shaders.
pub fn shader_reload_flag(world: &World) -> Option<Arc<AtomicBool>> {
    world
        .resource::<ActiveRenderBackend>()
        .and_then(|slot| slot.0.as_ref())
        .and_then(|backend| backend.shader_reload_flag())
}

#[cfg(test)]
mod tests {
    use super::*;

    // A world that never built a backend has none to yield or flag, and the
    // handoff borrow reports every parked piece absent.
    #[test]
    fn accessors_without_a_backend() {
        let mut world = World::new();
        assert!(shader_reload_flag(&world).is_none());
        assert!(take_render_backend(&mut world).is_none());
        assert!(take_hot_reload_sources(&mut world).is_none());

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
