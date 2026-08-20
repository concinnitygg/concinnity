// GraphicsSystem per-frame step: extraction of the frame's draw inputs from
// world state into the owned RenderSnapshot, then submission of that snapshot
// to the backend (see `submit`). Asset streaming + the camera-relative screen
// rebase run in StreamingSystem, scheduled just before this system.

use super::*;
use crate::assets::{Camera3D, HitRegion, Sprite, TextLabel, WindowMode};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult};
use crate::gfx::snapshot::{FrameScalars, RenderSnapshot, SceneOpRecorder};
use crate::gfx::{scene_flow, setting_action, settings, transform_propagation};
// The settings-row helpers this system's init-time captures share with the
// SettingCommand drain (which now lives in `settings_system`).
use crate::gfx::settings_system::rows::{
    DISABLED_ROW_COLOR, capture_row_labels, expand_dim_set, set_label_content, set_rows_grayed,
    set_sprite_x,
};

// The model matrix pushed for an editor-hidden object's draw slots: zero
// linear part and translation, so every vertex collapses to a degenerate
// point and nothing rasterizes (shadow passes included).
const HIDDEN_MODEL: [[f32; 4]; 4] = [[0.0; 4], [0.0; 4], [0.0; 4], [0.0, 0.0, 0.0, 1.0]];

// Deposit a sampled input packet for InputSystem (scheduled right after
// GraphicsSystem), merging onto an unconsumed one so no edge is lost.
fn deposit_input(ctx: &mut PipelineContext, packet: crate::gfx::input::InputPacket) {
    match ctx.resource_mut::<crate::ecs::InputMailbox>() {
        Some(mailbox) => mailbox.deposit(packet),
        None => {
            let mut mailbox = crate::ecs::InputMailbox::default();
            mailbox.deposit(packet);
            ctx.insert_resource(mailbox);
        }
    }
}

// The engine-owned skinned instance pool's free count, for the profiler's
// pool chip.
fn skinned_pool_free(ctx: &PipelineContext) -> u32 {
    ctx.resource::<crate::ecs::ActiveRenderQueues>()
        .and_then(|slot| slot.0.as_ref())
        .map(|queues| queues.slots.skinned_free() as u32)
        .unwrap_or(0)
}

// Record a device-memory failure in the shared [`GpuMemoryPressure`] resource,
// where the streaming valve can observe it.
fn publish_memory_pressure(ctx: &mut PipelineContext, frame: u64) {
    use crate::ecs::GpuMemoryPressure;
    match ctx.resource_mut::<GpuMemoryPressure>() {
        Some(pressure) => {
            pressure.events += 1;
            pressure.last_frame = frame;
        }
        None => {
            ctx.insert_resource(GpuMemoryPressure {
                events: 1,
                last_frame: frame,
            });
        }
    }
}

impl GraphicsSystem {
    pub(super) fn run_step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        if self.failed {
            // Stop, not Done: `Done` only retires this system and lets the world
            // keep stepping the rest, which for a failed init means the process
            // spins on headlessly with no window and no frame pacing. The run
            // has nothing left to present, so it ends.
            return StepResult::Stop;
        }
        // Pipelined frames: the backend lives with the render half on the
        // main thread; extract and send the snapshot instead of submitting.
        if ctx.resources.contains::<crate::ecs::PipelinedFrames>() {
            return self.run_step_pipelined(ctx);
        }
        // Take the parked backend for this step (see `ActiveRenderBackend`);
        // it is a plain local from here on, so `ctx` stays freely borrowable.
        let Some(mut backend) = crate::ecs::ActiveRenderBackend::take(ctx.resources) else {
            return StepResult::Done;
        };

        // Fill the frame snapshot from world state, then submit it. The
        // snapshot is taken out of `self` so extraction can borrow the change
        // gates beside it; its buffers ride along and keep their capacity.
        let mut snapshot = std::mem::take(&mut self.snapshot);
        self.extract(ctx, &mut snapshot);
        let outcome =
            super::submit::submit(&mut self.frame_policy, &mut snapshot, backend.as_mut());
        self.snapshot = snapshot;

        if outcome.memory_pressure || outcome.replay.memory_pressure {
            publish_memory_pressure(ctx, self.frame_count);
        }
        // Op failures the simulation side must roll back (streamed-mesh
        // uploads, chunk adds); StreamingSystem drains them next tick.
        if !outcome.replay.failures.is_empty() {
            match ctx.resource_mut::<crate::ecs::RenderOpFailures>() {
                Some(pending) => pending.0.extend_from_slice(&outcome.replay.failures),
                None => {
                    ctx.insert_resource(crate::ecs::RenderOpFailures(outcome.replay.failures));
                }
            }
        }
        if let Some(mut stats) = outcome.render_stats {
            // Publish this frame's render stats for the profiler overlay.
            // Backends without GPU-timed stats return the trait's default
            // (all zeros), which the HUD displays as "--". The skinned pool
            // chip reads the engine-owned instance pool.
            stats.skinned_pool_free = skinned_pool_free(ctx);
            ctx.profile.render = stats;
        }

        let mut result = outcome.result;
        if result == StepResult::Continue {
            self.frame_count += 1;
            if let Some(max) = self.max_frames
                && self.frame_count >= max
            {
                tracing::info!("GraphicsSystem: max_frames ({}) reached", max);
                backend.wait_idle();
                // Stop, not Done: `Done` only retires this system and lets the
                // world keep stepping the rest, which for a frame cap means the
                // renderer shuts down and the process spins on headlessly. Stop
                // ends the run, matching what closing the window does.
                result = StepResult::Stop;
            }
        }

        // Sample the window input the draw's event pump just produced and
        // deposit it for InputSystem (scheduled right after this system), so
        // sampling keeps the freshness it had when InputSystem polled the
        // backend itself.
        if result == StepResult::Continue {
            deposit_input(
                ctx,
                crate::gfx::input::InputPacket::sample(backend.as_mut()),
            );
        }

        crate::ecs::ActiveRenderBackend::put(ctx.resources, backend);
        result
    }

    // The pipelined step: extract into a recycled snapshot, hand it to the
    // render half (a rendezvous send, so at most one frame is in flight), and
    // apply the feedback the render half produced for the previous frame --
    // which the completed send guarantees has already arrived. A closed
    // channel in either direction means the render half stopped.
    fn run_step_pipelined(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let Some(pipe) = crate::ecs::PipelinedFrames::take(ctx.resources) else {
            return StepResult::Done;
        };
        let mut snapshot = std::mem::take(&mut self.snapshot);
        self.extract(ctx, &mut snapshot);
        if pipe.snapshot_tx.send(snapshot).is_err() {
            return StepResult::Stop;
        }
        // The rendezvous send completed, so the render half has this snapshot
        // and submits it unconditionally: count the frame here, once per world
        // step, the way the serial path and StreamingSystem's clock do. Counting
        // drained feedback instead would tie the count to a non-blocking
        // `try_recv`, and a frame-capped run would render `max` or `max + 1`
        // frames depending on which side won the race.
        self.frame_count += 1;

        let mut stop = false;
        while let Ok(feedback) = pipe.feedback_rx.try_recv() {
            stop |= feedback.stop;
            deposit_input(ctx, feedback.input);
            if feedback.replay.memory_pressure {
                publish_memory_pressure(ctx, self.frame_count);
            }
            if !feedback.replay.failures.is_empty() {
                match ctx.resource_mut::<crate::ecs::RenderOpFailures>() {
                    Some(pending) => pending.0.extend_from_slice(&feedback.replay.failures),
                    None => {
                        ctx.insert_resource(crate::ecs::RenderOpFailures(feedback.replay.failures));
                    }
                }
            }
            let mut stats = feedback.render_stats;
            stats.skinned_pool_free = skinned_pool_free(ctx);
            ctx.profile.render = stats;
            self.snapshot = feedback.recycled;
        }

        if !stop
            && let Some(max) = self.max_frames
            && self.frame_count >= max
        {
            tracing::info!("GraphicsSystem: max_frames ({}) reached", max);
            stop = true;
        }

        crate::ecs::PipelinedFrames::put(ctx.resources, pipe);
        if stop {
            StepResult::Stop
        } else {
            StepResult::Continue
        }
    }

    // Fill `snap` with everything this frame's draw consumes. All world-state
    // reads on the frame path happen here; the backend is not borrowed, so
    // extraction cannot reach the GPU and submission cannot reach the world.
    pub(crate) fn extract(&mut self, ctx: &mut PipelineContext, snap: &mut RenderSnapshot) {
        snap.clear();

        // The backend effects the earlier systems recorded this tick (spawn
        // slot ops, settings appliers, streaming uploads) move into the
        // snapshot; submission replays them in record order before the draw.
        if let Some(queues) = ctx
            .resources
            .get_mut::<crate::ecs::ActiveRenderQueues>()
            .and_then(|slot| slot.0.as_mut())
        {
            queues.ops.drain_into(&mut snap.ops);
        }

        // The FPS-cap pacer runs at the App level before the world steps (see
        // `app::pacing`), so `elapsed` here already reflects the capped
        // interval.
        let elapsed = self
            .start_time
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0);

        // read projection from Camera3D; the screen + camera position for the
        // draw come from StreamingSystem's `CameraRelativeView` (published just
        // before this system: the absolute values from Camera3D when no chunk
        // world is streaming, or both rebased onto the chunk render origin when
        // one is). Fall back to the absolute Camera3D values if the resource is
        // absent (a unit test driving this system without StreamingSystem).
        let (fov_y_radians, near, far, view_matrix, cam_pos) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| {
                (
                    c.fov_y_degrees.to_radians(),
                    c.near,
                    c.far,
                    c.view_matrix,
                    c.position,
                )
            })
            .unwrap_or((
                std::f32::consts::FRAC_PI_4,
                0.05,
                200.0,
                IDENTITY4,
                [0.0; 3],
            ));
        let (final_view, final_cam_pos) = ctx
            .resource::<crate::gfx::streaming_system::CameraRelativeView>()
            .map(|c| (c.view, c.cam_pos))
            .unwrap_or((view_matrix, cam_pos));

        // The overlay draw list + resolved menu state OverlaySystem built
        // earlier this tick, taken so a stale build is never redrawn.
        let overlay = ctx
            .resources
            .take::<crate::gfx::overlay::OverlayFrame>()
            .unwrap_or_default();
        let menu_active = overlay.menu_active;
        // The editor's menu-state override also drives the backend's menu mode
        // (OverlaySystem already folded it into `menu_active`).
        let menu_override = ctx.resource::<crate::ecs::MenuOverride>().and_then(|m| m.0);

        // Hide the system cursor while an in-engine cursor sprite is shown
        // (edge-triggered in the backend, so this is cheap every frame).
        snap.ui.cursor_hidden = overlay.want_ui_cursor;
        // The `MenuOverride` driver also needs the backend in menu mode so a
        // click with a freed cursor fires a UI action instead of re-capturing
        // the camera; a genuine menu-mode world already had this set at init.
        snap.ui.menu_mode = if menu_override.is_some() {
            Some(true)
        } else {
            None
        };
        // In menu mode (a MainMenu over a controlled camera, or an editor
        // override), capture the cursor for the camera unless a menu is active.
        // The editor's fly camera captures despite the active override: the
        // world stays frozen while the editor drives Camera3D from the
        // captured deltas. Edge-triggered in the backend, so this is cheap
        // every frame and a no-op in a plain first-person world.
        snap.ui.camera_capture = if self.menu_mode || menu_override.is_some() {
            let fly = ctx.resource::<crate::ecs::FlyCam>().is_some_and(|f| f.0);
            Some(!menu_active || fly)
        } else {
            None
        };

        // Runtime decal / emitter spawn and asset / shader / world.jsonl
        // hot-reload (`cn debug` only) are driven from the binary's
        // `DebugHook::tick` between world steps, not here. `cn run` has no
        // debug hook, so this per-frame path never touches them.

        // Lifetime/Spawner ticks and the spawn / despawn / reparent drains run
        // in SpawnSystem, scheduled earlier this tick, so the churn is already
        // applied when transforms are gathered below.

        // Gather updated model matrices for any entity whose transform changed
        // since last frame (physics, camera interact, reparent): resolve each
        // entity's GlobalTransform from Transform + Parent (top-down so parents
        // propagate to children), then queue it for the entity's GPU draw
        // slots. The cached path reuses its scratch and skips the resolve
        // entirely when no Transform / Parent changed; the push gate drops
        // slots whose matrix is unchanged, so a static scene sends nothing.
        transform_propagation::propagate_transforms_cached(ctx, &mut self.transform_cache);
        for (_entity, global, handle) in
            ctx.join2::<crate::assets::GlobalTransform, crate::assets::RenderHandle>()
        {
            for &slot in &handle.draws {
                self.model_push
                    .push_changed(&mut snap.models, slot as usize, global.0);
            }
        }

        // Refresh the editor's viewport-pick index from the freshly
        // propagated transforms. Candidates exist only when the editor
        // opted in at init (an injected PickIndex resource); a shipped
        // runtime skips this entirely. A despawned entity simply drops
        // out of the index.
        if !self.pick_candidates.is_empty() {
            let entries: Vec<crate::ecs::PickEntry> = {
                // Editor-session hidden objects: overwrite their just-queued
                // model matrices with the degenerate one (nothing draws) and
                // keep them out of the pick index. Re-derived every frame, so
                // clearing the set restores them immediately: the overwrite
                // goes through the push gate, so un-hiding shows up as a
                // changed matrix and is re-sent.
                let empty = std::collections::BTreeSet::new();
                let hidden = ctx
                    .resource::<crate::ecs::HiddenAssets>()
                    .map(|h| &h.0)
                    .unwrap_or(&empty);
                for c in self
                    .pick_candidates
                    .iter()
                    .filter(|c| hidden.contains(&c.asset_id))
                {
                    if let Some(handle) = ctx.get::<crate::assets::RenderHandle>(c.entity) {
                        for &slot in &handle.draws {
                            self.model_push.push_changed(
                                &mut snap.models,
                                slot as usize,
                                HIDDEN_MODEL,
                            );
                        }
                    }
                }
                self.pick_candidates
                    .iter()
                    .filter(|c| !hidden.contains(&c.asset_id))
                    .filter_map(|c| {
                        let global = ctx.get::<crate::assets::GlobalTransform>(c.entity)?;
                        let (bb_min, bb_max) =
                            crate::gfx::frustum::transform_aabb(c.local_min, c.local_max, global.0);
                        Some(crate::ecs::PickEntry {
                            asset_id: c.asset_id,
                            bb_min,
                            bb_max,
                        })
                    })
                    .collect()
            };
            ctx.insert_resource(crate::ecs::PickIndex { entries });
        }

        // Copy out the latest skinned poses. AnimationSystem wrote them into
        // the SkeletonPose components on the previous tick (flagging each pose
        // it touched); the one-frame lag is invisible at animation rates. An
        // untouched pose keeps its last upload, so an unanimated mesh uploads
        // its bind pose once and never again. Skipped while a menu is open:
        // animation is frozen, so nothing is flagged anyway and the skinned
        // draw is skipped behind an opaque menu.
        if !menu_active {
            for pose in ctx.query_mut::<crate::assets::SkeletonPose>() {
                if !pose.updated {
                    continue;
                }
                snap.poses.push(pose.skinned_index, &pose.joint_matrices);
                if !pose.morph_weights.is_empty() {
                    snap.morphs.push(pose.skinned_index, &pose.morph_weights);
                }
                pose.updated = false;
            }
            // Queue the model matrix for skinned instances that carry a
            // Transform (the runtime-spawned ones), so a moved instance
            // follows it. The authored templates have no Transform and keep
            // the model baked into their draw object at load.
            for (_entity, pose, transform) in
                ctx.join2::<crate::assets::SkeletonPose, crate::assets::Transform>()
            {
                self.skinned_model_push.push_changed(
                    &mut snap.skinned_models,
                    pose.skinned_index,
                    transform.model_matrix(),
                );
            }
            // Move rig-driven meshes to their capsule's resolved position
            // (PhysicsSystem wrote it on the previous tick; `moved` persists
            // across a menu pause, so no motion is lost while uploads are
            // skipped).
            for rig in ctx.query_mut::<crate::assets::CharacterRig>() {
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

        // SceneCommand / SettingCommand application lives in SettingsSystem,
        // scheduled just before this system, so a change is already on the
        // backend for this frame's submit and a scene jump has primed the
        // flow below.

        // Advance any in-flight scene fade, sourcing visibility from the live
        // per-entity components and recording the resulting fade / visibility
        // effects for submission. An idle fade is a no-op in
        // `tick_transitions`, so the visibility pairs are rebuilt only while a
        // transition is actually running. The flow is the shared
        // `ActiveSceneFlow` resource SettingsSystem also jumps; its `epoch` is
        // the shared clock for the fade timing.
        let fading = ctx
            .resource::<crate::ecs::ActiveSceneFlow>()
            .and_then(|f| f.flow.as_ref())
            .is_some_and(|f| !matches!(f.fade, scene_flow::FadePhase::None));
        if fading {
            super::scene::refresh_visibility_snapshot(ctx, &mut self.scene_visibility);
            if let Some(slot) = ctx.resources.get_mut::<crate::ecs::ActiveSceneFlow>() {
                let flow_elapsed = slot.epoch.elapsed().as_secs_f32();
                let mut recorder = SceneOpRecorder(&mut snap.scene_ops);
                scene_flow::tick_transitions(
                    &mut slot.flow,
                    &self.scene_visibility.visibility,
                    flow_elapsed,
                    &mut recorder,
                );
            }
        }

        // The logical viewport the line builder maps ribbon widths with:
        // refreshed from the FrameInput InputSystem published after the last
        // draw (the same source all overlay layout uses), seeded from the
        // backend at init. `[0, 0]` (backend not ready yet) keeps the seed.
        if let Some(input) = ctx.query::<crate::assets::FrameInput>().next()
            && input.viewport != [0.0, 0.0]
        {
            self.viewport = (input.viewport[0], input.viewport[1]);
        }

        // World-space lines published this frame (the `cn editor` origin
        // axes), expanded into ribbon geometry against the same camera the
        // frame draws with. Empty when nothing published any, which keeps the
        // pass out of the render graph entirely.
        super::lines::build_into(
            ctx,
            super::lines::LineFrame {
                view: final_view,
                cam_pos: final_cam_pos,
                // Lines are authored in absolute world space; a streaming
                // voxel world draws rebased onto the chunk render origin, so
                // shift them into that same space.
                rebase: [
                    final_cam_pos[0] - cam_pos[0],
                    final_cam_pos[1] - cam_pos[1],
                    final_cam_pos[2] - cam_pos[2],
                ],
                fov_y_radians,
                near,
                viewport: self.viewport,
            },
            &mut snap.lines,
        );

        // The editor's view mode + show flags, when published; a shipped
        // runtime has no resource and renders the lit default.
        let view = ctx
            .resource::<crate::ecs::ViewOverrides>()
            .copied()
            .unwrap_or_default();

        snap.frame = FrameScalars {
            elapsed,
            fov_y_radians,
            near,
            far,
            view: final_view,
            cam_pos: final_cam_pos,
            view_mode: view.mode,
            show: view.show,
            world_hidden: overlay.world_hidden,
            menu_active,
        };
        // Adopt the overlay draw list wholesale and hand the spent one back to
        // OverlaySystem, which recycles its buffers into the next build.
        let spent = std::mem::replace(&mut snap.text_calls, overlay.calls);
        ctx.insert_resource(crate::gfx::overlay::OverlayRecycle(spent));
    }

    // Capture each slider row's runtime bookkeeping from its drag HitRegion +
    // handle Sprite, then sync the handle position and value label to the live
    // value. Runs once at init, before UiInputSystem drains the HitRegions and
    // hides the screen elements. The HitRegions / Sprites are still present here.
    pub(super) fn init_sliders(&mut self, ctx: &mut PipelineContext) {
        let sprite_w: std::collections::HashMap<AssetId, f32> = ctx
            .query::<Sprite>()
            .map(|s| (s.asset_id, s.width))
            .collect();
        let mut sliders: Vec<SliderViz> = Vec::new();
        for r in ctx.query::<HitRegion>() {
            let Some(key) = setting_action::key_with_verb(&r.action, "drag") else {
                continue;
            };
            let (Some(handle_id), Some(value_id)) = (r.drag_handle, r.label) else {
                continue;
            };
            let handle_w = sprite_w.get(&handle_id).copied().unwrap_or(0.0);
            sliders.push(SliderViz {
                key: key.to_string(),
                track_x: r.x,
                track_w: r.width,
                handle_w,
                handle_id,
                value_id,
            });
        }
        // Sync each slider's handle + value label to its live value. One
        // persisted snapshot serves every controls slider (they read the store,
        // not the render params), so the capture never re-reads it per row.
        let persisted = self.persisted_settings();
        for s in &sliders {
            let Some(value) = self.slider_current_value(&s.key, &persisted) else {
                continue;
            };
            let frac = settings::slider_fraction(&s.key, value).unwrap_or(0.0);
            let hx = s.track_x + frac.clamp(0.0, 1.0) * (s.track_w - s.handle_w).max(0.0);
            set_sprite_x(ctx, s.handle_id, hx);
            set_label_content(
                ctx,
                s.value_id,
                &settings::format_slider_value(&s.key, value),
            );
        }
        self.sliders = sliders;
    }

    // Capture each key-rebind row's bookkeeping from its `setting:key_*:rebind`
    // HitRegion, then sync each value label to the live bound key. Runs once at
    // init (after the keymap is seeded), before UiInputSystem drains the
    // HitRegions; they are still present here.
    pub(super) fn init_rebind_rows(&mut self, ctx: &mut PipelineContext) {
        let mut rows: Vec<RebindViz> = Vec::new();
        let mut pad_rows: Vec<super::PadRebindViz> = Vec::new();
        for r in ctx.query::<HitRegion>() {
            let (Some(key), Some(value_id)) =
                (setting_action::key_with_verb(&r.action, "rebind"), r.label)
            else {
                continue;
            };
            // A `key_*` setting is a keyboard rebind row; a `pad_*` setting is
            // a gamepad rebind row.
            if let Some(action) = crate::gfx::keymap::Bindable::from_setting_key(key) {
                rows.push(RebindViz { action, value_id });
            } else if let Some(action) = crate::assets::GamepadAction::from_setting_key(key) {
                pad_rows.push(super::PadRebindViz { action, value_id });
            }
        }
        // Sync each value label to the live binding (persisted or default).
        for row in &rows {
            let name = self.keymap.get(row.action).display_name();
            set_label_content(ctx, row.value_id, name);
        }
        for row in &pad_rows {
            let name = self.gamepad_map.get(row.action).display_name();
            set_label_content(ctx, row.value_id, name);
        }
        self.rebind_rows = rows;
        self.pad_rebind_rows = pad_rows;
    }

    // Capture each cycle row's setting key -> value-label id, so a runtime change
    // can relabel a row other than the one clicked (the master preset relabels
    // its dependents; a quality-toggle change relabels the master row). Runs at
    // init, before UiInputSystem drains the HitRegions (GraphicsSystem.init runs
    // first), since they cannot be re-queried once drained.
    pub(super) fn init_cycle_value_labels(&mut self, ctx: &mut PipelineContext) {
        let mut labels = std::collections::HashMap::new();
        for r in ctx.query::<HitRegion>() {
            if let (Some(key), Some(value_id)) = (setting_action::cycle_key(&r.action), r.label) {
                labels.insert(key.to_string(), value_id);
            }
        }
        self.cycle_value_labels = labels;
    }

    // Capture each ScrollPanel's per-element clip band (reference space) so the
    // draw path scissors scroll-content elements to their panel and off-band
    // rows do not bleed over the chrome. Runs at init, before UiInputSystem
    // drains the ScrollPanels (GraphicsSystem.init runs first); the panels are
    // still queryable here. Every element listed in any row maps to its panel's
    // content band.
    pub(super) fn init_clip_rects(&mut self, ctx: &mut PipelineContext) {
        let mut clips: std::collections::HashMap<AssetId, [f32; 4]> =
            std::collections::HashMap::new();
        for panel in ctx.query::<crate::assets::ScrollPanel>() {
            let band = [panel.x, panel.y, panel.width, panel.height];
            for row in &panel.rows {
                for &id in &row.elements {
                    clips.insert(id, band);
                }
            }
        }
        self.clip_rects = clips;
    }

    // Gray out and disable every settings row whose feature the device cannot
    // provide (e.g. ray-traced reflections on a GPU without hardware ray
    // tracing). Runs once at init after the backend reports `self.caps`, while
    // the HitRegions / TextLabels / ScrollPanels are still present (before
    // UiInputSystem drains them). A disabled HitRegion is dropped by
    // UiInputSystem so it never hovers or fires; the row's labels are recolored
    // to a muted gray so it reads as unavailable.
    pub(super) fn apply_capability_gating(&mut self, ctx: &mut PipelineContext) {
        let caps = self.caps;
        // Mark each unavailable setting's region(s) disabled and collect their
        // value-label ids (both stepper regions of a row reference its value
        // label, so this is the row's anchor into the scroll element list).
        let mut gated_value_labels: std::collections::HashSet<AssetId> =
            std::collections::HashSet::new();
        for r in ctx.query_mut::<HitRegion>() {
            let Some(rest) = r.action.strip_prefix("setting:") else {
                continue;
            };
            let Some(key) = rest.split(':').next() else {
                continue;
            };
            if settings::setting_available(key, &caps) {
                continue;
            }
            r.disabled = true;
            if let Some(label) = r.label {
                gated_value_labels.insert(label);
            }
        }
        if gated_value_labels.is_empty() {
            return;
        }
        // Snapshot each scroll row's element id list (owned, so the ScrollPanel
        // borrow ends before the TextLabel write below), then expand the gated
        // value labels to every element of the rows that contain them.
        let rows: Vec<Vec<AssetId>> = ctx
            .query::<crate::assets::ScrollPanel>()
            .flat_map(|p| p.rows.iter().map(|r| r.elements.clone()))
            .collect();
        let dim = expand_dim_set(&gated_value_labels, &rows);
        for l in ctx.query_mut::<TextLabel>() {
            if dim.contains(&l.asset_id) {
                l.color = DISABLED_ROW_COLOR;
            }
        }
    }

    // Capture the show_fps / show_vram row labels (with their authored colors)
    // so the master "Display performance stats" toggle can gray them out at
    // runtime and restore them, and apply the initial gray from the resolved
    // toggle. Runs once at init while the HitRegions / ScrollPanels are present
    // (before UiInputSystem drains them), the same window
    // `apply_capability_gating` and the other init-time row captures use.
    pub(super) fn capture_perf_sub_rows(&mut self, ctx: &mut PipelineContext) {
        self.perf_sub_row_labels = capture_row_labels(ctx, &["show_fps", "show_vram"]);
        set_rows_grayed(ctx, &self.perf_sub_row_labels, !self.perf_stats);
    }

    // Capture the Resolution row's labels and apply the initial gray from the
    // resolved window mode: the row only applies in fullscreen (windowed sizes
    // come from the window itself, borderless covers the display), so it is
    // grayed + inert in the other modes. Same init window as the perf rows.
    pub(super) fn capture_resolution_row(&mut self, ctx: &mut PipelineContext) {
        self.resolution_row_labels = capture_row_labels(ctx, &["resolution"]);
        set_rows_grayed(
            ctx,
            &self.resolution_row_labels,
            self.window_args.mode != WindowMode::Fullscreen,
        );
    }

    // The current user-facing value of a slider setting, derived from the live
    // post-process params, or from `persisted` for the settings this system
    // does not hold live. `None` for a key it does not own at all.
    fn slider_current_value(&self, key: &str, persisted: &crate::config::Settings) -> Option<f32> {
        let stored = match key {
            "exposure" => self.post_process.exposure,
            "bloom_intensity" => self.post_process.bloom_intensity,
            "bloom_threshold" => self.post_process.bloom_threshold,
            "bloom_knee" => self.post_process.bloom_knee,
            "vignette" => self.post_process.vignette,
            "lut_strength" => self.post_process.lut_strength,
            "ambient_intensity" => self.ambient_intensity,
            // Per-feature sub-quality sliders read from the stored PostProcessConfig.
            "ssao_radius" => self.post_config.ssao_radius,
            "ssao_intensity" => self.post_config.ssao_intensity,
            "ssr_intensity" => self.post_config.ssr_intensity,
            "ssr_max_distance" => self.post_config.ssr_max_distance,
            "ssgi_intensity" => self.post_config.ssgi_intensity,
            "ssgi_max_distance" => self.post_config.ssgi_max_distance,
            "auto_exposure_min_ev" => self.post_config.auto_exposure_min_ev,
            "auto_exposure_max_ev" => self.post_config.auto_exposure_max_ev,
            "auto_exposure_speed" => self.post_config.auto_exposure_speed,
            // The controls sliders live in the controls store, not the render
            // params; read the persisted value or the engine default.
            "mouse_sensitivity" => persisted
                .controls
                .mouse_sensitivity
                .unwrap_or(settings::DEFAULT_MOUSE_SENSITIVITY),
            "gamepad_look_sensitivity" => persisted
                .controls
                .gamepad_look_sensitivity
                .unwrap_or(settings::DEFAULT_GAMEPAD_LOOK_SENSITIVITY),
            "gamepad_deadzone" => persisted
                .controls
                .gamepad_deadzone
                .unwrap_or(settings::DEFAULT_GAMEPAD_DEADZONE),
            // FOV lives in the graphics store (degrees); read the persisted value
            // or the authored default.
            "fov" => persisted.graphics.fov.unwrap_or(settings::DEFAULT_FOV),
            _ => return None,
        };
        // Invert `slider_apply_value` to the user-facing value (exposure: 2^ev ->
        // EV; mouse sensitivity: radians/pixel -> 1..100).
        Some(settings::slider_recover_value(key, stored))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::assets::{GlobalTransform, RenderHandle, SkeletonPose};
    use crate::blob::BlobData;
    use crate::ecs::{ComponentStorage, Resources, SkinnedMeshHandle};
    use crate::gfx::overlay::OverlayFrame;
    use crate::gfx::profile::FrameProfile;
    use crate::gfx::snapshot::SceneOp;

    // Owns the storage a PipelineContext borrows from; extraction never reads
    // the blob, so it stays empty.
    struct ExtractWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
        scratch: crate::ecs::Arena,
    }

    impl ExtractWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::empty(),
                profile: FrameProfile::default(),
                resources: Resources::new(),
                scratch: crate::ecs::Arena::with_capacity(64 * 1024),
            }
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: crate::ecs::FrameContext::new(&self.scratch),
            }
        }
    }

    fn extract_once(gs: &mut GraphicsSystem, world: &mut ExtractWorld) -> RenderSnapshot {
        let mut snap = RenderSnapshot::default();
        gs.extract(&mut world.ctx(), &mut snap);
        snap
    }

    fn translated(x: f32) -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, 0.0, 0.0, 1.0],
        ]
    }

    fn bare_pose(skinned_index: usize, joints: usize) -> SkeletonPose {
        SkeletonPose {
            mesh_id: SkinnedMeshHandle(0),
            skinned_index,
            skeleton: crate::gfx::skinning::Skeleton::new(Vec::new()),
            joint_matrices: vec![translated(1.0); joints],
            morph_weights: Vec::new(),
            updated: true,
            scratch: Default::default(),
        }
    }

    // A changed model matrix lands in the snapshot once per draw slot; an
    // unchanged frame extracts nothing for the same entity.
    #[test]
    fn extraction_gathers_changed_models_and_gates_repeats() {
        let mut world = ExtractWorld::new();
        {
            let mut ctx = world.ctx();
            let e = ctx.components.spawn();
            ctx.insert(e, GlobalTransform(translated(2.0)));
            ctx.insert(
                e,
                RenderHandle {
                    draws: [3, 4].into(),
                },
            );
        }
        let mut gs = GraphicsSystem::new();
        let snap = extract_once(&mut gs, &mut world);
        assert_eq!(
            snap.models,
            vec![(3, translated(2.0)), (4, translated(2.0))]
        );

        let snap = extract_once(&mut gs, &mut world);
        assert!(snap.models.is_empty(), "static frame re-extracts nothing");
    }

    // An updated pose is copied into the snapshot spans and its handshake flag
    // cleared, so the next frame extracts nothing for it. Morph weights ride
    // along only when present.
    #[test]
    fn extraction_copies_updated_poses_and_clears_the_flag() {
        let mut world = ExtractWorld::new();
        {
            let mut ctx = world.ctx();
            let e = ctx.components.spawn();
            let mut pose = bare_pose(5, 2);
            pose.morph_weights = vec![0.25, 0.75];
            ctx.insert(e, pose);
        }
        let mut gs = GraphicsSystem::new();
        let snap = extract_once(&mut gs, &mut world);
        let poses: Vec<(usize, usize)> = snap.poses.iter().map(|(idx, m)| (idx, m.len())).collect();
        assert_eq!(poses, vec![(5, 2)]);
        let morphs: Vec<(usize, Vec<f32>)> = snap
            .morphs
            .iter()
            .map(|(idx, w)| (idx, w.to_vec()))
            .collect();
        assert_eq!(morphs, vec![(5, vec![0.25, 0.75])]);

        let snap = extract_once(&mut gs, &mut world);
        assert!(snap.poses.is_empty(), "consumed pose is not re-extracted");
        assert!(snap.morphs.is_empty());
    }

    // While a menu is open the skinned families are skipped entirely, and the
    // frame scalars + overlay adoption reflect the menu state.
    #[test]
    fn extraction_skips_skinned_families_while_a_menu_is_open() {
        let mut world = ExtractWorld::new();
        {
            let mut ctx = world.ctx();
            let e = ctx.components.spawn();
            ctx.insert(e, bare_pose(0, 1));
            ctx.insert_resource(OverlayFrame {
                calls: Vec::new(),
                want_ui_cursor: true,
                menu_active: true,
                world_hidden: true,
            });
        }
        let mut gs = GraphicsSystem::new();
        let snap = extract_once(&mut gs, &mut world);
        assert!(snap.poses.is_empty(), "menu freezes pose extraction");
        assert!(snap.frame.menu_active);
        assert!(snap.frame.world_hidden);
        assert!(snap.ui.cursor_hidden);
        let parked = world
            .resources
            .get::<OverlayFrame>()
            .expect("the slot stays parked after the take");
        assert!(
            parked.calls.is_empty() && !parked.menu_active && !parked.world_hidden,
            "the overlay build is consumed so a stale one is never redrawn"
        );

        // The pose is still flagged, so closing the menu extracts it.
        let snap = extract_once(&mut gs, &mut world);
        assert_eq!(snap.poses.iter().count(), 1);
    }

    // The camera-relative view published by StreamingSystem wins over the
    // absolute Camera3D values; projection parameters come from the camera.
    #[test]
    fn extraction_prefers_the_camera_relative_view() {
        let mut world = ExtractWorld::new();
        {
            let mut ctx = world.ctx();
            let e = ctx.components.spawn();
            ctx.insert(
                e,
                crate::assets::Camera3D {
                    fov_y_degrees: 90.0,
                    near: 0.1,
                    far: 500.0,
                    view_matrix: translated(1.0),
                    position: [1.0, 2.0, 3.0],
                    yaw: 0.0,
                    pitch: 0.0,
                    desired_move: [0.0; 3],
                    jump_requested: false,
                    interact_requested: false,
                    controller: None,
                },
            );
            ctx.insert_resource(crate::gfx::streaming_system::CameraRelativeView {
                view: translated(7.0),
                cam_pos: [7.0, 0.0, 0.0],
            });
        }
        let mut gs = GraphicsSystem::new();
        let snap = extract_once(&mut gs, &mut world);
        assert_eq!(snap.frame.view, translated(7.0));
        assert_eq!(snap.frame.cam_pos, [7.0, 0.0, 0.0]);
        assert!((snap.frame.fov_y_radians - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(snap.frame.near, 0.1);
        assert_eq!(snap.frame.far, 500.0);
    }

    // An in-flight fade records its effects as scene ops instead of touching a
    // backend; an idle flow records nothing.
    #[test]
    fn extraction_records_fade_effects_as_scene_ops() {
        let mut world = ExtractWorld::new();
        {
            let mut ctx = world.ctx();
            ctx.insert_resource(crate::ecs::ActiveSceneFlow {
                flow: Some(scene_flow::SceneFlow {
                    scenes: vec![AssetId(1), AssetId(2)],
                    current: AssetId(1),
                    fade: scene_flow::FadePhase::ToBlack {
                        started_at: 0.0,
                        next: AssetId(2),
                    },
                }),
                epoch: std::time::Instant::now(),
            });
        }
        let mut gs = GraphicsSystem::new();
        let snap = extract_once(&mut gs, &mut world);
        assert!(
            matches!(snap.scene_ops.first(), Some(SceneOp::SetFade(_))),
            "an in-flight fade records its fade value"
        );

        // Clear the fade: nothing is recorded on an idle flow.
        if let Some(slot) = world.resources.get_mut::<crate::ecs::ActiveSceneFlow>() {
            slot.flow.as_mut().unwrap().fade = scene_flow::FadePhase::None;
        }
        let snap = extract_once(&mut gs, &mut world);
        assert!(snap.scene_ops.is_empty());
    }

    // The editor's menu override resolves to submit intents; without it (a
    // shipped runtime, no menu-mode world) both intents stay None.
    #[test]
    fn extraction_resolves_menu_override_into_ui_intents() {
        let mut world = ExtractWorld::new();
        let mut gs = GraphicsSystem::new();
        let snap = extract_once(&mut gs, &mut world);
        assert_eq!(snap.ui.menu_mode, None);
        assert_eq!(snap.ui.camera_capture, None);

        world
            .resources
            .insert(crate::ecs::MenuOverride(Some(false)));
        let snap = extract_once(&mut gs, &mut world);
        assert_eq!(snap.ui.menu_mode, Some(true));
        assert_eq!(
            snap.ui.camera_capture,
            Some(true),
            "no active menu screen, so the camera captures"
        );
    }

    // Editor-session hidden objects overwrite their queued models with the
    // degenerate matrix and drop out of the published pick index; without the
    // editor's resources nothing of the sort is extracted.
    #[test]
    fn extraction_applies_editor_hides_and_pick_index_only_when_opted_in() {
        let mut world = ExtractWorld::new();
        let entity = {
            let mut ctx = world.ctx();
            let e = ctx.components.spawn();
            ctx.insert(e, GlobalTransform(translated(2.0)));
            ctx.insert(e, RenderHandle { draws: [9].into() });
            e
        };
        // A shipped runtime: no candidates, so no pick index is published.
        let mut gs = GraphicsSystem::new();
        let snap = extract_once(&mut gs, &mut world);
        assert_eq!(snap.models, vec![(9, translated(2.0))]);
        assert!(world.resources.get::<crate::ecs::PickIndex>().is_none());

        // The editor opted in and hid the asset: the queued model is
        // overwritten in order and the index excludes it.
        gs.pick_candidates.push(super::super::PickCandidate {
            asset_id: AssetId(42),
            entity,
            local_min: [-1.0; 3],
            local_max: [1.0; 3],
        });
        world.resources.insert(crate::ecs::HiddenAssets(
            [AssetId(42)].into_iter().collect(),
        ));
        {
            let mut ctx = world.ctx();
            if let Some(g) = ctx.get_mut::<GlobalTransform>(entity) {
                g.0 = translated(3.0);
            }
        }
        let snap = extract_once(&mut gs, &mut world);
        assert_eq!(
            snap.models,
            vec![(9, translated(3.0)), (9, HIDDEN_MODEL)],
            "the hide overwrite follows the move so it wins on the backend"
        );
        let index = world.resources.get::<crate::ecs::PickIndex>().unwrap();
        assert!(index.entries.is_empty(), "a hidden asset is not pickable");
    }

    // The pipelined step: the extracted snapshot crosses the channel, the
    // render half's feedback is applied (input mailbox, stats, recycled
    // buffer), and a feedback-flagged stop propagates as StepResult::Stop.
    #[test]
    fn pipelined_step_sends_the_snapshot_and_applies_feedback() {
        let mut world = ExtractWorld::new();
        {
            let mut ctx = world.ctx();
            let e = ctx.components.spawn();
            ctx.insert(e, GlobalTransform(translated(2.0)));
            ctx.insert(e, RenderHandle { draws: [3].into() });
        }
        let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel(0);
        let (feedback_tx, feedback_rx) = std::sync::mpsc::channel();
        world.resources.insert(crate::ecs::PipelinedFrames(Some(
            crate::ecs::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        )));
        // Stand-in render half: consume each snapshot and reply with a
        // feedback whose second frame flags a stop.
        let consumer = std::thread::spawn(move || {
            for stop in [false, true] {
                let snapshot = snapshot_rx.recv().expect("a snapshot arrives");
                let mut input = crate::gfx::input::InputPacket::default();
                input.raw.jump = true;
                let render_stats = crate::gfx::profile::RenderStats {
                    draw_calls: 7,
                    ..Default::default()
                };
                feedback_tx
                    .send(crate::gfx::feedback::FrameFeedback {
                        input,
                        render_stats,
                        replay: Default::default(),
                        recycled: snapshot,
                        stop,
                    })
                    .expect("feedback is consumed");
            }
        });

        let mut gs = GraphicsSystem::new();
        // Frame 0: the send completes, but its feedback may or may not have
        // landed yet; either way the step continues.
        assert_eq!(gs.run_step(&mut world.ctx()), StepResult::Continue);
        // Frame 1: the rendezvous completing proves feedback 0 arrived; the
        // second feedback carries the stop, applied this frame or next.
        let mut result = gs.run_step(&mut world.ctx());
        if result == StepResult::Continue {
            result = gs.run_step(&mut world.ctx());
        }
        assert_eq!(result, StepResult::Stop, "the feedback stop propagates");
        consumer.join().expect("the stand-in render half exits");

        let mut ctx = world.ctx();
        assert!(
            ctx.resource_mut::<crate::ecs::InputMailbox>()
                .and_then(|m| m.0.take())
                .is_some_and(|p| p.raw.jump),
            "the feedback's input packet reached the mailbox"
        );
        assert_eq!(
            ctx.profile.render.draw_calls, 7,
            "the feedback's render stats reached the profile"
        );
    }

    // With the render half gone (channel closed), the pipelined step stops
    // instead of blocking or panicking.
    #[test]
    fn pipelined_step_stops_when_the_render_half_is_gone() {
        let mut world = ExtractWorld::new();
        let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel(0);
        let (_feedback_tx, feedback_rx) = std::sync::mpsc::channel();
        world.resources.insert(crate::ecs::PipelinedFrames(Some(
            crate::ecs::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        )));
        drop(snapshot_rx);
        let mut gs = GraphicsSystem::new();
        assert_eq!(gs.run_step(&mut world.ctx()), StepResult::Stop);
    }

    // A gated value label pulls in every element of the scroll row that holds
    // it (the row's background, name, value, and stepper glyphs), so the whole
    // row grays out; unrelated rows are untouched.
    #[test]
    fn dim_set_expands_a_gated_value_label_to_its_whole_row() {
        let value = AssetId(3);
        let gated: HashSet<AssetId> = [value].into_iter().collect();
        let rows = vec![
            // Row A: bg, name, prev_glyph, value, next_glyph (value is gated).
            vec![AssetId(1), AssetId(2), value, AssetId(4), AssetId(5)],
            // Row B: an unrelated row.
            vec![AssetId(10), AssetId(11)],
        ];
        let dim = expand_dim_set(&gated, &rows);
        for id in [1, 2, 3, 4, 5] {
            assert!(dim.contains(&AssetId(id)), "row A element {id} should dim");
        }
        assert!(!dim.contains(&AssetId(10)), "an unrelated row stays lit");
        assert!(!dim.contains(&AssetId(11)), "an unrelated row stays lit");
    }

    // With no scroll rows (a hand-authored menu outside a panel), only the gated
    // value label itself dims -- a graceful fallback, not a panic.
    #[test]
    fn dim_set_without_rows_falls_back_to_the_value_label() {
        let gated: HashSet<AssetId> = [AssetId(7)].into_iter().collect();
        assert_eq!(expand_dim_set(&gated, &[]), gated);
    }
}
