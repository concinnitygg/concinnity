//! Borrows of the render backend a running world parks between steps, and the
//! state graphics init parks beside it. A caller reaches them only on the main
//! thread, between world steps; from inside a step the backend is taken.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use concinnity_engine::ecs::{ActiveRenderBackend, World};
use concinnity_engine::render::backend::RenderBackend;
use concinnity_engine::render::environment::volumetric_fog::PushedFogSettings;
use concinnity_engine::render::world_load::source_maps::TextureNameSlots;

/// Disjoint borrows of the parked render backend and the init-captured state
/// beside it. Each is `None` when init did not park it; the backend also while
/// a step has it taken.
pub(crate) struct RenderHandoff<'a> {
    pub(crate) backend: Option<&'a mut (dyn RenderBackend + 'static)>,
    pub(crate) fog: Option<&'a mut PushedFogSettings>,
    // Parked only under hot-reload capture.
    pub(crate) texture_slots: Option<&'a TextureNameSlots>,
}

/// Borrow the parked backend and the state beside it at once.
pub(crate) fn render_handoff(world: &mut World) -> RenderHandoff<'_> {
    let (_, resources) = world.systems_and_resources();
    let (backend, fog, texture_slots) =
        resources.get_disjoint_mut::<ActiveRenderBackend, PushedFogSettings, TextureNameSlots>();
    RenderHandoff {
        backend: backend.and_then(|slot| slot.0.as_deref_mut()),
        fog,
        texture_slots: texture_slots.map(|slots| &*slots),
    }
}

/// The parked backend's shader-reload flag, which a host sets to have the
/// backend recompile its shaders before the next draw. `None` when no backend
/// is parked or it does not reload shaders.
pub(crate) fn shader_reload_flag(world: &World) -> Option<Arc<AtomicBool>> {
    world
        .resource::<ActiveRenderBackend>()
        .and_then(|slot| slot.0.as_ref())
        .and_then(|backend| backend.live_edit_ref()?.shader_reload_flag())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::VolumetricFog;
    use concinnity_engine::ecs::asset_id::AssetId;
    use concinnity_engine::render::environment::volumetric_fog::resolve_asset;

    // A world that never built a backend has none to yield or flag, and the
    // handoff borrow reports every parked piece absent.
    #[test]
    fn accessors_without_a_backend() {
        let mut world = World::new();
        assert!(shader_reload_flag(&world).is_none());

        let handoff = render_handoff(&mut world);
        assert!(handoff.backend.is_none());
        assert!(handoff.fog.is_none());
        assert!(handoff.texture_slots.is_none());
    }

    // The handoff reaches the parked fog and texture-name map together.
    #[test]
    fn render_handoff_borrows_the_parked_state_together() {
        let mut world = World::new();
        world.insert_resource(PushedFogSettings(None));
        world.insert_resource(TextureNameSlots([(AssetId(3), 7)].into()));
        let handoff = render_handoff(&mut world);
        assert!(handoff.backend.is_none());
        let slots = handoff.texture_slots.expect("texture-name map parked");
        assert_eq!(slots.0.len(), 1);
        let fog = resolve_asset(&VolumetricFog {
            enabled: true,
            ..Default::default()
        });
        handoff.fog.expect("fog parked").0 = fog;
        assert!(world.resource::<PushedFogSettings>().unwrap().0.is_some());
    }
}
