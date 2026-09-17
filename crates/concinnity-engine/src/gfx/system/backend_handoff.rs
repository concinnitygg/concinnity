// The render backend's construction, and the handoff of it and the init-built
// runtime state to the world's shared resources.

use concinnity_core::ecs::PipelineContext;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::backend_init::BackendInit;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::volumetric_fog::FogSettings;

use super::GraphicsSystem;
use super::hot_reload_sources::HotReloadSources;
use super::parked::{PushedFogSettings, TextureNameSlots};

// What init hands to the world beside the backend.
pub(super) struct RuntimeHandoff<'a> {
    pub(super) draw_object_count: usize,
    pub(super) frames_in_flight: usize,
    pub(super) skinned_pool_reservations: &'a [(usize, usize)],
    pub(super) fog: Option<FogSettings>,
    // Present only under hot-reload capture.
    pub(super) texture_name_slots: Option<TextureNameSlots>,
    pub(super) hot_reload_sources: Option<HotReloadSources>,
}

impl GraphicsSystem {
    // Build the backend for this world. A live editor swap may have parked the
    // pre-edit world's backend as a `PendingBackend`: it reloads the new world
    // in place, keeping the window and device, when it can hot-swap and the
    // swapchain config (pixel format, frames in flight, EDR) is unchanged.
    // Otherwise it is idled and dropped, and a new backend is built (a rare
    // one-frame flash). DirectX and Vulkan report no hot-swap config.
    pub(super) fn build_backend(
        &mut self,
        ctx: &mut PipelineContext,
        backend_init: BackendInit<'_>,
    ) -> RenderResult<Box<dyn RenderBackend>> {
        let reuse_backend = match ctx
            .resources
            .remove::<crate::ecs::PendingBackend>()
            .map(|p| p.0)
        {
            Some(backend) if backend.hot_swap_config() == Some(backend_init.swapchain_config()) => {
                Some(backend)
            }
            Some(backend) => {
                backend.wait_idle();
                None
            }
            None => None,
        };
        if let Some(mut backend) = reuse_backend {
            backend.reload_world(backend_init)?;
            tracing::info!(
                "GraphicsSystem: reused live backend (world reloaded in place, window kept)"
            );
            return Ok(backend);
        }
        // Tests inject a mock backend factory through `test_hooks`; production
        // always routes to the compile-time-selected real backend.
        #[cfg(test)]
        {
            match self.test_hooks.as_mut() {
                Some(hooks) => (hooks.backend_factory)(backend_init),
                // A test builds no real device: one that forgets its hooks
                // fails its own assertions instead of opening a window.
                None => Err(concinnity_core::render::error::RenderError::Other(
                    "no test backend factory".into(),
                )),
            }
        }
        #[cfg(not(test))]
        {
            crate::device::init_backend(backend_init)
        }
    }

    // Park the backend and the init-built runtime state in the world's shared
    // resources, where each per-step user takes and returns them.
    pub(super) fn park_runtime_state(
        &mut self,
        ctx: &mut PipelineContext,
        handoff: RuntimeHandoff<'_>,
    ) {
        // Seed the frame extraction's viewport from the live backend;
        // FrameInput refreshes it once InputSystem starts publishing.
        self.viewport = self
            .backend
            .as_ref()
            .map(|b| b.logical_size())
            .unwrap_or((0.0, 0.0));
        // OverlaySystem shapes the overlay draw list from these each frame.
        ctx.insert_resource(crate::gfx::overlay::OverlayAssets {
            fonts: std::mem::take(&mut self.loaded_fonts),
            sprite_texture_slots: std::mem::take(&mut self.sprite_texture_slots),
            debug_hud_chips: std::mem::take(&mut self.debug_hud_chips),
            stat_hud_chips: std::mem::take(&mut self.stat_hud_chips),
            clip_rects: std::mem::take(&mut self.clip_rects),
            initial_viewport: self.viewport,
        });

        self.publish_streaming_state(ctx, handoff.frames_in_flight);

        // The op queue backend effects accumulate into, and the slot-allocation
        // authority: the draw-slot free list seeded with the build-time draw
        // count, and the skinned instance pool with the pre-reserved copies.
        ctx.insert_resource(crate::ecs::ActiveRenderQueues(Some(
            crate::ecs::RenderQueues {
                ops: Default::default(),
                slots: crate::gfx::render_slots::RenderSlots::new(
                    handoff.draw_object_count,
                    self.caps.reuses_build_slots,
                    handoff.skinned_pool_reservations,
                ),
            },
        )));

        ctx.insert_resource(crate::ecs::ActiveRenderBackend(self.backend.take()));
        // Beside it, the state tooling edits the running world through: the fog
        // a reload dedupes against, and under hot-reload capture the
        // texture-name map and the source catalogs.
        ctx.insert_resource(PushedFogSettings(handoff.fog));
        if let Some(slots) = handoff.texture_name_slots {
            ctx.insert_resource(slots);
        }
        if let Some(sources) = handoff.hot_reload_sources {
            ctx.insert_resource(sources);
        }

        // SettingsSystem jumps the scene flow and this system ticks it.
        ctx.insert_resource(crate::ecs::ActiveSceneFlow::new(self.scene_flow.take()));
    }
}
