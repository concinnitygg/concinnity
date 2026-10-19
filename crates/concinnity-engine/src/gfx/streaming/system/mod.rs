//! StreamingSystem: runs core's streaming pools against their load threads and
//! publishes the camera-relative view the draw consumes. Each step hands every
//! pool the loads its threads finished, drives the pools (core's
//! `StreamingState`), then sends the loads the drive requested.
//!
//! Scheduled after the cameras and before GraphicsSystem, so the view is this
//! tick's pose, a chunk world's view rebase is ready for this same frame's
//! submit, and any recorded texture / mesh upload lands before the draw.
//! GraphicsSystem's init builds the pools and their threads and parks them here
//! as the `StreamingRuntime` resource.

use concinnity_core::bake::texture::TextureImage;
use concinnity_core::components::{Camera3D, FrameInput};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{PipelineContext, StepResult, System};
use concinnity_core::planet::{TileId, TileMesh};
use concinnity_core::render::extract::CameraRelativeView;
use concinnity_core::render::overlay::OverlayFrame;
use concinnity_core::render::queues::ActiveRenderQueues;
use concinnity_core::render::scene_flow;
use concinnity_core::render::scene_residency;
use concinnity_core::render::stream_pools::{
    ChunkLoad, ChunkRequest, DecodedMesh, DriveView, ItemLoad, LoadPool, PlanetView,
    StreamingState, TileCamera,
};

use crate::ecs::RenderOpFailures;
use crate::gfx::streaming::shader::ShaderWarmup;
use crate::worker::{Job, Worker};

const IDENTITY4: [[f32; 4]; 4] = concinnity_core::transform::IDENTITY;

// The load threads of each streaming pool, `Some` where that pool streams.
#[derive(Default)]
pub(crate) struct StreamWorkers {
    pub(crate) texture: Option<Worker<usize, ItemLoad<TextureImage>>>,
    pub(crate) mesh: Option<Worker<usize, ItemLoad<DecodedMesh>>>,
    pub(crate) chunk: Option<Worker<ChunkRequest, ChunkLoad>>,
    pub(crate) planet: Option<Worker<TileId, (TileId, TileMesh)>>,
}

// Pair a pool with its worker, when both exist.
fn paired<'a, P, Req, Res>(
    worker: &'a mut Option<Worker<Req, Res>>,
    pool: Option<&'a mut P>,
) -> Option<(&'a mut Worker<Req, Res>, &'a mut P)>
where
    P: LoadPool<Request = Req, Result = Res>,
    Req: Job<Key = Req>,
    Res: Send + 'static,
{
    worker.as_mut().zip(pool)
}

impl StreamWorkers {
    // Hand each pool the loads its threads finished.
    fn deliver(&mut self, state: &mut StreamingState) {
        if let Some((w, p)) = paired(&mut self.texture, state.texture_mut()) {
            w.deliver(p);
        }
        if let Some((w, p)) = paired(&mut self.mesh, state.mesh_mut()) {
            w.deliver(p);
        }
        if let Some((w, p)) = paired(&mut self.chunk, state.chunk_pool_mut()) {
            w.deliver(p);
        }
        if let Some((w, p)) = paired(&mut self.planet, state.planet_mut()) {
            w.deliver(p);
        }
    }

    // Send the loads the drive requested.
    fn send(&mut self, state: &mut StreamingState) {
        if let Some((w, p)) = paired(&mut self.texture, state.texture_mut()) {
            w.send_requests(p);
        }
        if let Some((w, p)) = paired(&mut self.mesh, state.mesh_mut()) {
            w.send_requests(p);
        }
        if let Some((w, p)) = paired(&mut self.chunk, state.chunk_pool_mut()) {
            w.send_requests(p);
        }
        if let Some((w, p)) = paired(&mut self.planet, state.planet_mut()) {
            w.send_requests(p);
        }
    }
}

// The streaming pools GraphicsSystem's init builds, their load threads and
// deferred shader builds, and the long-session memory drift, parked as a
// resource that StreamingSystem drives each step.
pub(crate) struct StreamingRuntime {
    pub(crate) state: StreamingState,
    pub(crate) workers: StreamWorkers,
    pub(crate) shader_warmup: Option<ShaderWarmup>,
    // Long-session memory drift, folded from the pressure valve's throttled
    // sample. Purely reported: it names what grew, and never moves the valve.
    drift: concinnity_core::budget::drift::DriftTracker,
    // Last verdict logged, so a steady session states its reading once.
    last_drift_verdict: Option<concinnity_core::budget::drift::DriftVerdict>,
}

impl std::fmt::Debug for StreamingRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingRuntime")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl StreamingRuntime {
    pub(crate) fn new(
        state: StreamingState,
        workers: StreamWorkers,
        shader_warmup: Option<ShaderWarmup>,
    ) -> Self {
        Self {
            state,
            workers,
            shader_warmup,
            drift: Default::default(),
            last_drift_verdict: None,
        }
    }

    // One frame of every pool, a planet's tiles included: hand them the
    // finished loads, drive them, then send what they requested.
    fn drive(
        &mut self,
        queues: &mut concinnity_core::render::queues::RenderQueues,
        view: DriveView<'_>,
    ) -> ([[f32; 4]; 4], [f32; 3]) {
        self.workers.deliver(&mut self.state);
        let out = self.state.drive(queues, view, self.shader_warmup.as_mut());
        self.workers.send(&mut self.state);
        out
    }

    // Fold the pressure valve's RSS sample into the long-session drift tracker,
    // stating the reading whenever it changes. Reports only: nothing here moves
    // a byte budget.
    fn sample_drift(
        &mut self,
        rss: Option<u64>,
        budget: u64,
    ) -> Option<concinnity_core::budget::drift::MemoryDrift> {
        use concinnity_core::budget::drift::DriftVerdict;
        let rss = rss?;
        let heap_live = concinnity_core::memory::stats()?.live_bytes;
        let drift = self.drift.sample(
            rss,
            heap_live,
            budget,
            concinnity_host::clock::monotonic_micros(),
        )?;

        if self.last_drift_verdict != Some(drift.verdict) {
            self.last_drift_verdict = Some(drift.verdict);
            let heap_mib = drift.heap_growth_bytes / (1024 * 1024);
            let outside_mib = drift.outside_heap_growth_bytes / (1024 * 1024);
            let minutes = drift.window_secs / 60;
            let reading = drift.verdict.label();
            if drift.verdict == DriftVerdict::Settled {
                tracing::info!(
                    "memory drift: {reading}: heap {heap_mib:+} MiB, outside heap {outside_mib:+} MiB over {minutes} min"
                );
            } else {
                tracing::warn!(
                    "memory drift: {reading}: heap {heap_mib:+} MiB, outside heap {outside_mib:+} MiB over {minutes} min"
                );
            }
        }
        Some(drift)
    }
}

// Drives texture / mesh / chunk residency against the streaming budgets.
#[derive(Debug, Default)]
pub(crate) struct StreamingSystem {
    // Scene-status scratch reused across frames, compared against the
    // published `SceneResidencyStatus` before republishing.
    scene_status_scratch: Vec<(AssetId, scene_residency::SceneLoadState, f32)>,
}

impl StreamingSystem {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl System for StreamingSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No parked state (graphics init has not succeeded): nothing to drive,
        // and GraphicsSystem is not drawing either, so no view to publish.
        if !ctx.resources.contains::<StreamingRuntime>() {
            return StepResult::Continue;
        }

        // Everything the drive needs from the wider context, gathered first so
        // the runtime below is borrowed in place.
        let ram_budget = ctx
            .resource::<concinnity_core::budget::MemoryBudget>()
            .map(|b| b.budget_bytes);
        let planet = ctx
            .resource::<concinnity_core::planet::PlanetFrame>()
            .copied();
        // The camera the draw will use (written by the camera controller last
        // tick) is the absolute fallback when no chunk streaming rebases it.
        let (view_matrix, cam_pos, fov_y_degrees) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.view_matrix, c.position, c.fov_y_degrees))
            .unwrap_or((IDENTITY4, [0.0; 3], 0.0));
        // The viewport the input system sampled last tick: with the field of
        // view, how large a planet's ground looks.
        let viewport_height = ctx.resource::<FrameInput>().map_or(0.0, |i| i.viewport[1]);
        let finest_cell_size = f64::from(
            concinnity_core::settings::system::planet_finest_cell_size(ctx.resources),
        );
        // Peek (not take) the overlay's world-hidden flag: OverlaySystem
        // published it first this tick and GraphicsSystem takes it later.
        let world_hidden = ctx
            .resource::<OverlayFrame>()
            .map(|o| o.world_hidden)
            .unwrap_or(false);

        let pins = ctx
            .resource::<scene_flow::ActiveSceneFlow>()
            .and_then(|slot| slot.flow.as_ref())
            .map(scene_flow::SceneFlow::streaming_pins);
        let scene_pins = pins.as_ref().map(scene_flow::ScenePins::as_slice);
        let transient_pool_bytes = ctx.profile.render.transient_pool_bytes;

        // Throttled process-RAM back-off sample (~2x/sec), skipped entirely
        // when no `MemoryBudget` is published or RSS is unavailable.
        {
            let mut pressure_sample = None;
            let mut drift_sample = None;
            let Some(runtime) = ctx.resources.get_mut::<StreamingRuntime>() else {
                return StepResult::Continue;
            };
            if runtime.state.pressure_due()
                && let Some(budget) = ram_budget
            {
                let rss = crate::app::sysmem::process_resident_bytes();
                pressure_sample = runtime.state.sample_pressure(rss, budget);
                drift_sample = runtime.sample_drift(rss, budget);
            }
            if let Some(pressure) = pressure_sample {
                ctx.insert_resource(pressure);
            }
            if let Some(drift) = drift_sample {
                ctx.insert_resource(drift);
            }
        }

        // The recording surfaces graphics init published beside this state.
        let Some(mut queues) = ActiveRenderQueues::take(ctx.resources) else {
            // Should not happen (published together with this state): publish
            // the absolute view so the draw is still driven.
            ctx.insert_resource(CameraRelativeView {
                view: view_matrix,
                cam_pos,
            });
            return StepResult::Continue;
        };
        let failures = ctx.resources.remove::<RenderOpFailures>();

        let Some(runtime) = ctx.resources.get_mut::<StreamingRuntime>() else {
            ActiveRenderQueues::put(ctx.resources, queues);
            return StepResult::Continue;
        };

        // Roll back the ops that failed at the previous frame's replay before
        // planning, so this frame sees the corrected residency.
        if let Some(failures) = failures {
            runtime
                .state
                .apply_op_failures(&failures.0, &mut queues.slots);
        }

        // A planet's tiles, around the camera in the frame the planet system
        // last published.
        let planet = planet.map(|frame| PlanetView {
            frame,
            camera: TileCamera {
                position: cam_pos,
                points_per_radian: concinnity_core::planet::points_per_radian(
                    fov_y_degrees,
                    viewport_height,
                ),
                finest_cell_size,
            },
        });
        let (view, cam_pos) = runtime.drive(
            &mut queues,
            DriveView {
                cam_pos,
                view: view_matrix,
                world_hidden,
                scene_pins,
                planet,
            },
        );

        // Refresh each pool's device footprint under the shared tags.
        runtime.state.publish_footprint(transient_pool_bytes);
        let have_residency = match runtime.state.scene_residency() {
            Some(residency) => {
                residency.status_into(&mut self.scene_status_scratch);
                true
            }
            None => false,
        };

        ActiveRenderQueues::put(ctx.resources, queues);
        ctx.insert_resource(CameraRelativeView { view, cam_pos });
        // Republish the per-scene load status when it changed, so menus and
        // loading screens can read scene progress without touching the pools.
        if have_residency {
            match ctx.resource_mut::<scene_residency::SceneResidencyStatus>() {
                Some(published) => {
                    if published.scenes != self.scene_status_scratch {
                        published.scenes.clone_from(&self.scene_status_scratch);
                    }
                }
                None => {
                    ctx.insert_resource(scene_residency::SceneResidencyStatus {
                        scenes: self.scene_status_scratch.clone(),
                    });
                }
            }
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests;
