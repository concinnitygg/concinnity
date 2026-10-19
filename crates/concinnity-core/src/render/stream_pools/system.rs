// The streaming pools as a system: each step hands every pool the loads its
// host finished, drives the pools, sends the loads the drive requested, and
// publishes the camera-relative view the draw consumes and the per-scene load
// status. The loads themselves cross to the host through `StreamLoaders`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

use super::{DriveView, PlanetView, ShaderPump, StreamingState, StreamingStats, TileCamera};
use crate::budget::drift::{DriftTracker, DriftVerdict, MemoryDrift};
use crate::budget::{MemoryBudget, ResidentMemory};
use crate::components::{Camera3D, FrameInput};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Clock, PipelineContext, StepResult, System, World};
use crate::planet::{PlanetFrame, points_per_radian};
use crate::render::extract::CameraRelativeView;
use crate::render::ops::OpFailure;
use crate::render::overlay::OverlayFrame;
use crate::render::queues::{ActiveRenderQueues, RenderQueues};
use crate::render::render_mode::RenderMode;
use crate::render::scene_flow::{ActiveSceneFlow, SceneFlow, ScenePins};
use crate::render::scene_residency::{SceneLoadState, SceneResidencyStatus};
use crate::settings::system::planet_finest_cell_size;
use crate::transform::IDENTITY;

/// Where a host's load threads meet the pools.
pub trait StreamLoaders: Send {
    /// Hand each pool the loads that finished since the last frame.
    fn deliver(&mut self, state: &mut StreamingState);

    /// Send the loads the drive requested.
    fn send(&mut self, state: &mut StreamingState);

    /// The host's deferred shader builds, when it has any.
    fn shaders(&mut self) -> Option<&mut dyn ShaderPump> {
        None
    }
}

// Recorded backend effects that failed at replay and need rolling back on the
// pools (a streamed-mesh upload refused by a full region, a chunk add). The
// frame outcome leaves them; the next streaming step drains them first.
#[derive(Debug, Default)]
pub(crate) struct RenderOpFailures(pub(crate) Vec<OpFailure>);

/// The streaming pools, the host's loaders and the long-session memory drift,
/// published as a resource the streaming system drives each step.
pub struct StreamingRuntime {
    /// Every pool's policy and residency.
    pub state: StreamingState,
    loaders: Box<dyn StreamLoaders>,
    // Folded from the pressure valve's throttled sample. Purely reported: it
    // names what grew, and never moves the valve.
    drift: DriftTracker,
    // Last verdict logged, so a steady session states its reading once.
    last_drift_verdict: Option<DriftVerdict>,
}

impl fmt::Debug for StreamingRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamingRuntime")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl StreamingRuntime {
    /// The pools in `state`, loading through `loaders`.
    pub fn new(state: StreamingState, loaders: Box<dyn StreamLoaders>) -> Self {
        Self {
            state,
            loaders,
            drift: DriftTracker::default(),
            last_drift_verdict: None,
        }
    }

    /// Per-pool `(resident, pending, unloaded)` counts of the runtime
    /// published in `world`, or `None` before graphics init publishes it.
    pub fn stats_in(world: &World) -> Option<StreamingStats> {
        world
            .resource::<Self>()
            .map(|runtime| runtime.state.streaming_stats())
    }

    /// One frame of every pool, a planet's tiles included: hand them the
    /// finished loads, drive them, then send what they requested. Returns the
    /// view and camera position the draw takes.
    pub fn drive(
        &mut self,
        queues: &mut RenderQueues,
        view: DriveView<'_>,
    ) -> ([[f32; 4]; 4], [f32; 3]) {
        self.loaders.deliver(&mut self.state);
        let out = self.state.drive(queues, view, self.loaders.shaders());
        self.loaders.send(&mut self.state);
        out
    }

    // Fold the pressure valve's RSS sample into the drift tracker, stating the
    // reading whenever it changes. Reports only: nothing here moves a budget.
    fn sample_drift(&mut self, rss: Option<u64>, budget: u64, now_us: u64) -> Option<MemoryDrift> {
        let heap_live = crate::memory::stats()?.live_bytes;
        let drift = self.drift.sample(rss?, heap_live, budget, now_us)?;
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

/// Drives texture, mesh, chunk, planet-tile and shader residency against the
/// streaming budgets, and publishes the camera-relative view the draw takes.
#[derive(Debug, Default)]
pub struct StreamingSystem {
    // Compared against the published `SceneResidencyStatus` before
    // republishing.
    scene_status_scratch: Vec<(AssetId, SceneLoadState, f32)>,
}

impl StreamingSystem {
    /// The system for a world that runs with a window.
    pub fn for_world(world: &World) -> Option<Self> {
        RenderMode::renders_in(world).then(Self::default)
    }

    // The throttled (~2x/sec) RAM back-off and drift sample, skipped when no
    // `MemoryBudget` is published.
    fn sample_memory(ctx: &mut PipelineContext) {
        let budget = ctx.resource::<MemoryBudget>().map(|b| b.budget_bytes);
        let rss = ctx.resource::<ResidentMemory>().copied();
        let now_us = ctx.resource::<Clock>().map_or(0, Clock::read);
        let Some(runtime) = ctx.resources.get_mut::<StreamingRuntime>() else {
            return;
        };
        let Some(budget) = budget.filter(|_| runtime.state.pressure_due()) else {
            return;
        };
        let rss = rss.and_then(|r| r.read());
        let pressure = runtime.state.sample_pressure(rss, budget);
        let drift = runtime.sample_drift(rss, budget, now_us);
        if let Some(pressure) = pressure {
            ctx.insert_resource(pressure);
        }
        if let Some(drift) = drift {
            ctx.insert_resource(drift);
        }
    }

    // Republish the per-scene load status when it changed, so menus and
    // loading screens read scene progress without touching the pools.
    fn publish_scene_status(&self, ctx: &mut PipelineContext) {
        match ctx.resource_mut::<SceneResidencyStatus>() {
            Some(published) => {
                if published.scenes != self.scene_status_scratch {
                    published.scenes.clone_from(&self.scene_status_scratch);
                }
            }
            None => {
                ctx.insert_resource(SceneResidencyStatus {
                    scenes: self.scene_status_scratch.clone(),
                });
            }
        }
    }
}

impl System for StreamingSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No runtime (graphics init has not succeeded): nothing to drive, and
        // nothing is drawn either, so no view to publish.
        if !ctx.resources.contains::<StreamingRuntime>() {
            return StepResult::Continue;
        }

        // The camera the draw will use (written by the controllers last tick)
        // is the absolute fallback when no chunk streaming rebases it.
        let (view_matrix, cam_pos, fov_y_degrees) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.view_matrix, c.position, c.fov_y_degrees))
            .unwrap_or((IDENTITY, [0.0; 3], 0.0));
        let planet = ctx
            .resource::<PlanetFrame>()
            .copied()
            .map(|frame| PlanetView {
                frame,
                camera: TileCamera {
                    position: cam_pos,
                    // With the viewport the input sampled last tick: how large a
                    // planet's ground looks.
                    points_per_radian: points_per_radian(
                        fov_y_degrees,
                        ctx.resource::<FrameInput>().map_or(0.0, |i| i.viewport[1]),
                    ),
                    finest_cell_size: f64::from(planet_finest_cell_size(ctx.resources)),
                },
            });
        // Peek (not take) the overlay's world-hidden flag: the frame submit
        // takes the overlay frame later this tick.
        let world_hidden = ctx
            .resource::<OverlayFrame>()
            .is_some_and(|o| o.world_hidden);
        let pins = ctx
            .resource::<ActiveSceneFlow>()
            .and_then(|slot| slot.flow.as_ref())
            .map(SceneFlow::streaming_pins);
        let scene_pins = pins.as_ref().map(ScenePins::as_slice);
        let transient_pool_bytes = ctx.profile.render.transient_pool_bytes;

        Self::sample_memory(ctx);

        // The recording surfaces published beside the runtime.
        let Some(mut queues) = ActiveRenderQueues::take(ctx.resources) else {
            // Published together with the runtime, so not expected: publish
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
        runtime.state.publish_footprint(transient_pool_bytes);
        let residency = runtime.state.scene_residency();
        if let Some(residency) = residency {
            residency.status_into(&mut self.scene_status_scratch);
        }
        let has_residency = residency.is_some();

        ActiveRenderQueues::put(ctx.resources, queues);
        ctx.insert_resource(CameraRelativeView { view, cam_pos });
        if has_residency {
            self.publish_scene_status(ctx);
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests;
