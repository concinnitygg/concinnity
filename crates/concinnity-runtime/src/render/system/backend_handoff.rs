// The render backend's construction, and the handoff of it and the init-built
// runtime state to the world's shared resources.

use concinnity_core::ecs::PipelineContext;
use concinnity_core::gfx::render_types::SkinnedIndex;
use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::backend::error::{RenderError, RenderResult};
use concinnity_core::render::backend::init::BackendInit;
use concinnity_core::render::environment::volumetric_fog::{FogSettings, PushedFogSettings};
use concinnity_core::render::frame::queues::{ActiveRenderQueues, RenderQueues};
use concinnity_core::render::overlay::OverlayAssets;
use concinnity_core::render::scene_flow::ActiveSceneFlow;
use concinnity_core::render::slots::render_slots::RenderSlots;
use concinnity_core::render::world_load::source_maps::TextureNameSlots;

use super::GraphicsSystem;
use crate::live_edit::hot_reload_sources::HotReloadSources;

// What init hands to the world beside the backend.
pub(super) struct RuntimeHandoff<'a> {
    pub(super) draw_object_count: usize,
    pub(super) frames_in_flight: usize,
    pub(super) skinned_pool_reservations: &'a [(SkinnedIndex, SkinnedIndex)],
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
    #[cfg_attr(
        not(test),
        expect(
            clippy::needless_pass_by_ref_mut,
            reason = "test builds swap in a hook through `self`"
        )
    )]
    pub(super) fn build_backend(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        backend_init: BackendInit<'_>,
    ) -> RenderResult<Box<dyn RenderBackend>> {
        let reuse_backend = match ctx
            .resources
            .remove::<crate::live_edit::PendingBackend>()
            .map(|p| p.0)
        {
            Some(backend) if backend_init.reloads_into(backend.as_ref()) => Some(backend),
            Some(backend) => {
                backend.wait_idle();
                None
            }
            None => None,
        };
        if let Some(mut backend) = reuse_backend {
            backend
                .live_edit()
                .ok_or(RenderError::Unsupported { op: "reload_world" })?
                .reload_world(backend_init)?;
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
                None => Err(RenderError::Other("no test backend factory".into())),
            }
        }
        #[cfg(not(test))]
        {
            crate::platform::device::init_backend(backend_init)
        }
    }

    // Park the backend and the init-built runtime state in the world's shared
    // resources, where each per-step user takes and returns them.
    pub(super) fn park_runtime_state(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        handoff: RuntimeHandoff<'_>,
    ) {
        // Seed the frame extraction's viewport from the live backend;
        // FrameInput refreshes it once InputSystem starts publishing.
        let viewport = self
            .backend
            .as_ref()
            .map_or((0.0, 0.0), |b| b.logical_size());
        self.extractor.set_viewport(viewport);
        // OverlaySystem shapes the overlay draw list from these each frame.
        ctx.insert_resource(OverlayAssets {
            fonts: std::mem::take(&mut self.loaded_fonts),
            sprite_texture_slots: std::mem::take(&mut self.sprite_texture_slots),
            debug_hud_chips: std::mem::take(&mut self.debug_hud_chips),
            stat_hud_chips: std::mem::take(&mut self.stat_hud_chips),
            clip_rects: std::mem::take(&mut self.clip_rects),
            initial_viewport: viewport,
        });

        self.publish_streaming_state(ctx, handoff.frames_in_flight);

        // The op queue backend effects accumulate into, and the slot-allocation
        // authority: the draw-slot free list seeded with the build-time draw
        // count, and the skinned instance pool with the pre-reserved copies.
        ctx.insert_resource(ActiveRenderQueues::new(RenderQueues {
            ops: Default::default(),
            slots: RenderSlots::new(
                handoff.draw_object_count,
                self.caps.reuses_build_slots,
                handoff.skinned_pool_reservations,
            ),
        }));

        ctx.insert_resource(concinnity_core::ecs::Parked(self.backend.take()));
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
        ctx.insert_resource(ActiveSceneFlow::new(self.scene_flow.take()));
    }
}
