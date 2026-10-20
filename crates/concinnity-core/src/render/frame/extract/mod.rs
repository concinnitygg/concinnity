//! Frame extraction: everything one frame's draw consumes, read from world
//! state into the owned [`RenderSnapshot`]. All world reads on the frame path
//! happen here; the backend is not borrowed, so extraction cannot reach the GPU
//! and submission cannot reach the world.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::components::{
    Camera3D, CharacterRig, DirectionalLight, FrameInput, GlobalTransform, RenderHandle,
    SkeletonPose, Transform,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{
    Entity, FlyCam, FrameCaptureRequest, FrameTime, HiddenAssets, InputSampledAt, MenuOverride,
    PickEntry, PickIndex, PipelineContext, SimTiming, ViewOverrides,
};
use crate::gfx::frustum;
use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
use crate::planet::{FrameRebases, RebaseCursor};
use crate::render::environment::atmosphere::AtmosphereSettings;
use crate::render::frame::queues::ActiveRenderQueues;
use crate::render::frame::snapshot::{FrameScalars, RenderSnapshot, SceneOpRecorder};
use crate::render::overlay::{OverlayFrame, OverlayRecycle};
use crate::render::scene_flow::{self, ActiveSceneFlow, SceneVisibilityScratch};
use crate::render::temporal::history_reset::{
    FrameClock, HistoryResetCauses, HistoryResetTracker, HistoryView, PendingHistoryReset,
};
use crate::sky::SkyOrientation;
use crate::transform::propagation::{self, TransformCache};

mod atmosphere;
#[cfg(test)]
mod bench;
mod grass_benders;
mod lights;
mod lines;
mod model_push;
mod screenshot;

pub use lights::gather_lights;
pub use screenshot::ScreenshotBurst;

use atmosphere::AtmosphereDrive;
use model_push::ModelPushCache;

const IDENTITY4: [[f32; 4]; 4] = crate::transform::IDENTITY;

// The model matrix pushed for an editor-hidden object's draw slots: zero
// linear part and translation, so every vertex collapses to a degenerate
// point and nothing rasterizes (shadow passes included).
const HIDDEN_MODEL: [[f32; 4]; 4] = [[0.0; 4], [0.0; 4], [0.0; 4], [0.0, 0.0, 0.0, 1.0]];

/// The view and camera position the frame draws with, published by the
/// streaming drive just before extraction: the absolute values from
/// `Camera3D` when no chunk world is streaming, or both rebased onto the chunk
/// render origin when one is.
#[derive(Debug, Clone, Copy)]
pub struct CameraRelativeView {
    /// The view matrix the frame draws with.
    pub view: [[f32; 4]; 4],
    /// The camera position in the same space.
    pub cam_pos: [f32; 3],
}

/// One viewport-pick candidate captured at init: the prop's asset id, its
/// entity (the live `GlobalTransform` source), and its local-space bounds.
#[derive(Debug, Clone, Copy)]
pub struct PickCandidate {
    /// The prop's asset id.
    pub asset_id: AssetId,
    /// The entity whose transform places it.
    pub entity: Entity,
    /// Local-space bounds, minimum corner.
    pub local_min: [f32; 3],
    /// Local-space bounds, maximum corner.
    pub local_max: [f32; 3],
}

/// What a frame that never drew leaves the next one to reset: its own causes,
/// and a cut for a move of the simulated frame its history was never carried
/// through.
fn skipped_causes(frame: &FrameScalars) -> HistoryResetCauses {
    if frame.rebase.is_some() {
        frame.history_reset.union(HistoryResetCauses::CAMERA_CUT)
    } else {
        frame.history_reset
    }
}

// The camera the frame is extracted against: projection and absolute pose from
// `Camera3D`, and the view and position the draw uses, which the streaming
// drive may have rebased onto the chunk render origin.
struct FrameCamera {
    entity: Option<Entity>,
    fov_y_radians: f32,
    near: f32,
    view_distance: Option<f32>,
    view: [[f32; 4]; 4],
    position: [f32; 3],
    draw_view: [[f32; 4]; 4],
    draw_position: [f32; 3],
}

impl FrameCamera {
    fn read(ctx: &PipelineContext<'_>) -> Self {
        let mut camera = ctx.query_with_entity::<Camera3D>().next().map_or(
            Self {
                entity: None,
                fov_y_radians: core::f32::consts::FRAC_PI_4,
                near: 0.05,
                view_distance: None,
                view: IDENTITY4,
                position: [0.0; 3],
                draw_view: IDENTITY4,
                draw_position: [0.0; 3],
            },
            |(entity, c)| Self {
                entity: Some(entity),
                fov_y_radians: c.fov_y_degrees.to_radians(),
                near: c.near,
                view_distance: c.view_distance,
                view: c.view_matrix,
                position: c.position,
                draw_view: c.view_matrix,
                draw_position: c.position,
            },
        );
        if let Some(relative) = ctx.resource::<CameraRelativeView>() {
            camera.draw_view = relative.view;
            camera.draw_position = relative.cam_pos;
        }
        camera
    }
}

/// The cross-frame state extraction keeps: change gates, the shaders' clock,
/// the temporal-history tracker and the per-frame drives the world declared.
#[derive(Default)]
pub struct FrameExtractor {
    // Seconds of frame time rendered, the shaders' clock.
    render_secs: f64,
    menu_mode: bool,
    grows_grass: bool,
    atmosphere: Option<AtmosphereDrive>,
    screenshot: Option<ScreenshotBurst>,
    scene_visibility: SceneVisibilityScratch,
    pick_candidates: Vec<PickCandidate>,
    transform_cache: TransformCache,
    // The sky rotation the directional-light set was last carried at. `None`
    // until the first frame, so a world whose sky never turns carries it
    // exactly once.
    pushed_sky_rows: Option<[[f32; 4]; 3]>,
    rebases: RebaseCursor,
    history_reset: HistoryResetTracker,
    model_push: ModelPushCache<DrawIndex>,
    skinned_model_push: ModelPushCache<SkinnedIndex>,
    // Logical viewport size the line builder maps ribbon widths with. Seeded
    // from the backend at init, refreshed each frame from `FrameInput`.
    viewport: (f32, f32),
}

impl core::fmt::Debug for FrameExtractor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FrameExtractor")
            .field("render_secs", &self.render_secs)
            .field("pick_candidates", &self.pick_candidates.len())
            .finish_non_exhaustive()
    }
}

impl FrameExtractor {
    /// A togglable menu coexists with a controlled camera, so each frame
    /// captures the cursor for the camera unless a menu screen is active.
    pub fn set_menu_mode(&mut self, on: bool) {
        self.menu_mode = on;
    }

    /// The world grows grass, so each frame gathers what tramples it.
    pub fn set_grows_grass(&mut self, on: bool) {
        self.grows_grass = on;
    }

    /// The world's atmosphere, which each frame places about the planet and
    /// which drives the sun light.
    pub fn set_atmosphere(&mut self, settings: Option<AtmosphereSettings>) {
        self.atmosphere = settings.map(AtmosphereDrive::new);
    }

    /// The logical viewport the line builder starts from.
    pub fn set_viewport(&mut self, viewport: (f32, f32)) {
        self.viewport = viewport;
    }

    /// The viewport-pick candidates captured at init.
    pub fn pick_candidates(&self) -> &[PickCandidate] {
        &self.pick_candidates
    }

    /// The viewport-pick candidates, refilled at init when the editor opted
    /// in.
    pub fn pick_candidates_mut(&mut self) -> &mut Vec<PickCandidate> {
        &mut self.pick_candidates
    }

    /// The per-entity scene-visibility scratch, shared with scene setup.
    pub fn scene_visibility_mut(&mut self) -> &mut SceneVisibilityScratch {
        &mut self.scene_visibility
    }

    /// Start a screenshot burst; the frames after this one capture it.
    pub fn start_screenshots(&mut self, burst: Option<ScreenshotBurst>) {
        self.screenshot = burst;
    }

    /// Hand a dropped frame's history reset to the next frame extracted.
    pub fn reissue_history_reset(&mut self, frame: &FrameScalars) {
        self.history_reset.reissue(skipped_causes(frame));
    }

    /// Fill `snap` with everything frame `frame` draws.
    pub fn extract(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        snap: &mut RenderSnapshot,
        frame: u64,
    ) {
        snap.clear();

        // The backend effects the earlier systems recorded this tick (spawn
        // slot ops, settings appliers, streaming uploads) move into the
        // snapshot; submission replays them in record order before the draw.
        if let Some(queues) = ctx
            .resources
            .get_mut::<ActiveRenderQueues>()
            .and_then(|slot| slot.0.as_mut())
        {
            queues.ops.drain_into(&mut snap.ops);
        }
        snap.capture = ctx
            .resources
            .remove::<FrameCaptureRequest>()
            .map(|request| request.0);
        screenshot::schedule(&mut self.screenshot, ctx, snap, frame);

        // The frame pacer runs before the world steps, so the frame time here
        // already reflects the capped interval.
        let frame_time = ctx.resource::<FrameTime>().copied().unwrap_or_default();
        self.render_secs += f64::from(frame_time.dt.max(0.0));
        let camera = FrameCamera::read(ctx);

        // The overlay draw list + resolved menu state the overlay build made
        // earlier this tick, taken so a stale build is never redrawn.
        let overlay = ctx.resources.take::<OverlayFrame>().unwrap_or_default();
        // The editor's menu-state override also drives the backend's menu mode
        // (the overlay build already folded it into `menu_active`).
        let menu_override = ctx.resource::<MenuOverride>().and_then(|m| m.0);
        self.write_ui_state(ctx, snap, &overlay, menu_override.is_some());

        self.push_models(ctx, snap);
        // Skipped while a menu is open: animation is frozen and the skinned
        // draw is skipped behind an opaque menu. The editor's override keeps
        // the world drawn, and its edits reseed poses and move templates.
        if !overlay.menu_active || menu_override.is_some() {
            self.push_skinned(ctx, snap);
        }
        self.advance_scene_fade(ctx, snap, frame_time.elapsed);
        self.build_lines(ctx, snap, &camera);

        snap.frame = self.frame_scalars(ctx, snap, &camera, &overlay, frame_time.dt);
        // Adopt the overlay draw list wholesale and hand the spent one back to
        // the overlay build, which recycles its buffers into the next build.
        let spent = core::mem::replace(&mut snap.text_calls, overlay.calls);
        ctx.insert_resource(OverlayRecycle(spent));
    }

    // The cursor and menu-mode state the backend applies before the draw.
    fn write_ui_state(
        &self,
        ctx: &PipelineContext<'_>,
        snap: &mut RenderSnapshot,
        overlay: &OverlayFrame,
        menu_overridden: bool,
    ) {
        // Hide the system cursor while an in-engine cursor sprite is shown
        // (edge-triggered in the backend, so this is cheap every frame).
        snap.ui.cursor_hidden = overlay.want_ui_cursor;
        // The override also needs the backend in menu mode so a click with a
        // freed cursor fires a UI action instead of re-capturing the camera; a
        // genuine menu-mode world already had this set at init.
        snap.ui.menu_mode = menu_overridden.then_some(true);
        // In menu mode, capture the cursor for the camera unless a menu is
        // active. The editor's fly camera captures despite the active override:
        // the world stays frozen while the editor drives Camera3D from the
        // captured deltas.
        snap.ui.camera_capture = (self.menu_mode || menu_overridden).then(|| {
            let fly = ctx.resource::<FlyCam>().is_some_and(|f| f.0);
            !overlay.menu_active || fly
        });
    }

    // Updated model matrices for every entity whose transform changed since
    // last frame, queued for its draw slots, plus what follows the same
    // transforms: grass benders and the editor's pick index. The cached
    // propagation skips the resolve when no Transform / Parent changed; the
    // push gate drops slots whose matrix is unchanged, so a static scene sends
    // nothing.
    fn push_models(&mut self, ctx: &mut PipelineContext<'_>, snap: &mut RenderSnapshot) {
        propagation::propagate_transforms_cached(ctx, &mut self.transform_cache);
        for (_entity, global, handle) in ctx.join2::<GlobalTransform, RenderHandle>() {
            for &slot in &handle.draws {
                self.model_push
                    .push_changed(&mut snap.models, slot, global.0);
            }
        }
        if self.grows_grass {
            grass_benders::gather_into(ctx, &mut snap.grass_benders);
        }
        if !self.pick_candidates.is_empty() {
            self.refresh_pick_index(ctx, snap);
        }
    }

    // The latest skinned poses the animation wrote on the previous tick, and
    // the skinned models that moved. An untouched pose keeps its last upload.
    fn push_skinned(&mut self, ctx: &mut PipelineContext<'_>, snap: &mut RenderSnapshot) {
        for pose in ctx.query_mut::<SkeletonPose>() {
            if !pose.updated {
                continue;
            }
            snap.poses.push(pose.skinned_index, &pose.joint_matrices);
            if !pose.morph_weights.is_empty() {
                snap.morphs.push(pose.skinned_index, &pose.morph_weights);
            }
            pose.updated = false;
        }
        // Runtime-spawned skinned instances carry a Transform the model
        // follows; the authored templates keep the model baked at load.
        for (_entity, pose, transform) in ctx.join2::<SkeletonPose, Transform>() {
            self.skinned_model_push.push_changed(
                &mut snap.skinned_models,
                pose.skinned_index,
                transform.model_matrix(),
            );
        }
        // Rig-driven meshes follow their capsule (`moved` persists across a
        // menu pause, so no motion is lost while uploads are skipped).
        for rig in ctx.query_mut::<CharacterRig>() {
            if rig.moved {
                self.skinned_model_push.push(
                    &mut snap.skinned_models,
                    rig.skinned_index,
                    rig.model(),
                );
                rig.moved = false;
            }
        }
    }

    // Advance an in-flight scene fade, recording its fade / visibility effects
    // for submission. The visibility pairs are rebuilt only while a transition
    // is actually running.
    fn advance_scene_fade(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        snap: &mut RenderSnapshot,
        elapsed: f32,
    ) {
        let fading = ctx
            .resource::<ActiveSceneFlow>()
            .and_then(|f| f.flow.as_ref())
            .is_some_and(|f| !matches!(f.fade, scene_flow::FadePhase::None));
        if !fading {
            return;
        }
        scene_flow::refresh_visibility_snapshot(ctx, &mut self.scene_visibility);
        if let Some(slot) = ctx.resources.get_mut::<ActiveSceneFlow>() {
            let flow_elapsed = slot.elapsed(elapsed);
            let mut recorder = SceneOpRecorder(&mut snap.scene_ops);
            scene_flow::tick_transitions(
                &mut slot.flow,
                &self.scene_visibility.visibility,
                flow_elapsed,
                &mut recorder,
            );
        }
    }

    // World-space lines published this frame, expanded into ribbon geometry
    // against the same camera the frame draws with.
    fn build_lines(
        &mut self,
        ctx: &PipelineContext<'_>,
        snap: &mut RenderSnapshot,
        camera: &FrameCamera,
    ) {
        // The logical viewport the line builder maps ribbon widths with, from
        // the FrameInput published after the last draw. `[0, 0]` (backend not
        // ready yet) keeps the seed.
        if let Some(input) = ctx.query::<FrameInput>().next()
            && input.viewport != [0.0, 0.0]
        {
            self.viewport = (input.viewport[0], input.viewport[1]);
        }
        let (draw, absolute) = (camera.draw_position, camera.position);
        lines::build_into(
            ctx,
            lines::LineFrame {
                view: camera.draw_view,
                cam_pos: draw,
                // Lines are authored in absolute world space; a streaming
                // voxel world draws rebased onto the chunk render origin.
                rebase: [
                    draw[0] - absolute[0],
                    draw[1] - absolute[1],
                    draw[2] - absolute[2],
                ],
                fov_y_radians: camera.fov_y_radians,
                near: camera.near,
                viewport: self.viewport,
            },
            &mut snap.lines,
        );
    }

    // The frame-wide scalars: the sky and lights, a planet rebase, and whether
    // the temporal history still matches this frame.
    fn frame_scalars(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        snap: &mut RenderSnapshot,
        camera: &FrameCamera,
        overlay: &OverlayFrame,
        dt: f32,
    ) -> FrameScalars {
        // The view mode + show flags the dev tools published; a shipped
        // runtime has no resource and renders the lit default.
        let view = ctx.resource::<ViewOverrides>().copied().unwrap_or_default();

        // The sky's rotation reaches the GPU twice: as the rows every cubemap
        // sample is taken through, and as the directions the directional lights
        // are carried to. Carried only when the sky has moved.
        let sky = ctx.resource::<SkyOrientation>().copied();
        let sky_rot = sky.map_or(SkyOrientation::IDENTITY_ROWS, |s| s.sample_rows());
        let mut directional = None;
        if let Some(sky) = sky
            && self.pushed_sky_rows != Some(sky_rot)
        {
            self.pushed_sky_rows = Some(sky_rot);
            directional = Some(lights::lights_under_sky(
                ctx.query::<DirectionalLight>(),
                &sky,
            ));
        }
        // An atmosphere lights the scene with the sun as it reaches the camera
        // through the air, so the set is re-pushed whenever that light changes.
        let atmosphere = self.atmosphere.as_mut().map(|drive| {
            let (frame, set) = drive.frame(ctx, camera.position, directional);
            if set.is_some() {
                directional = set;
            }
            frame
        });

        // A planet world's frame moved since the last drawn frame: the previous
        // view is carried into the new frame before it is compared, and the
        // local lights packed at init are moved and rewritten in place.
        let rebase = self.rebases.take(ctx.resource::<FrameRebases>());
        if let Some(rebase) = &rebase {
            self.history_reset.rebase(rebase);
            if let Some((data, uniforms)) = lights::carry_local_lights(ctx, rebase) {
                snap.ops.record(move |backend| {
                    if let Err(e) = backend.move_local_lights(&data, &uniforms) {
                        tracing::error!("moving the local lights: {e}");
                    }
                });
            }
        }

        // Whether the history the temporal passes accumulated still matches
        // this frame. Compared in absolute world space, so a chunk rebase is
        // not a cut.
        self.history_reset.request(PendingHistoryReset::take(ctx));
        let scene = ctx
            .resource::<ActiveSceneFlow>()
            .and_then(|f| f.flow.as_ref())
            .map(|f| f.current);
        let timing = ctx.resource::<SimTiming>().copied().unwrap_or_default();
        let history_reset = self.history_reset.observe(
            HistoryView {
                camera: camera.entity,
                position: camera.position,
                view: camera.view,
                fov_y_radians: camera.fov_y_radians,
                scene,
            },
            FrameClock {
                dt,
                ticks: timing.ticks,
                tick_dt: timing.tick_dt,
            },
        );

        FrameScalars {
            elapsed: self.render_secs,
            fov_y_radians: camera.fov_y_radians,
            near: camera.near,
            view_distance: camera.view_distance,
            view: camera.draw_view,
            cam_pos: camera.draw_position,
            view_mode: view.mode,
            show: view.show,
            world_hidden: overlay.world_hidden,
            menu_active: overlay.menu_active,
            sky_rot,
            directional,
            history_reset,
            rebase,
            atmosphere,
            sunlight: lights::sunlight(ctx),
            input_sampled_us: ctx.resource::<InputSampledAt>().map(|at| at.0),
        }
    }

    // Refresh the editor's viewport-pick index from the freshly propagated
    // transforms, refilling the published index in place. Editor-session
    // hidden objects get the degenerate model matrix (nothing draws) and stay
    // out of the index; re-derived every frame, so un-hiding shows up as a
    // changed matrix and is re-sent.
    fn refresh_pick_index(&mut self, ctx: &mut PipelineContext<'_>, snap: &mut RenderSnapshot) {
        let mut entries = ctx
            .resources
            .take::<PickIndex>()
            .map(|index| index.entries)
            .unwrap_or_default();
        entries.clear();
        {
            let empty = BTreeSet::new();
            let hidden = ctx.resource::<HiddenAssets>().map_or(&empty, |h| &h.0);
            for c in self
                .pick_candidates
                .iter()
                .filter(|c| hidden.contains(&c.asset_id))
            {
                if let Some(handle) = ctx.get::<RenderHandle>(c.entity) {
                    for &slot in &handle.draws {
                        self.model_push
                            .push_changed(&mut snap.models, slot, HIDDEN_MODEL);
                    }
                }
            }
            entries.extend(
                self.pick_candidates
                    .iter()
                    .filter(|c| !hidden.contains(&c.asset_id))
                    .filter_map(|c| {
                        let global = ctx.get::<GlobalTransform>(c.entity)?;
                        let (bb_min, bb_max) =
                            frustum::transform_aabb(c.local_min, c.local_max, global.0);
                        Some(PickEntry {
                            asset_id: c.asset_id,
                            bb_min,
                            bb_max,
                        })
                    }),
            );
        }
        // The slot `take` left behind is refilled, so the resource is boxed
        // only on the first frame.
        ctx.insert_resource(PickIndex { entries });
    }
}

#[cfg(test)]
mod tests;
