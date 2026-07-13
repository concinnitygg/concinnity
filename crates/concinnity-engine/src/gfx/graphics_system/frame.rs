// GraphicsSystem per-frame step: transform/pose upload, settings application,
// streaming updates, scene-reel ticking, and the backend draw call.

use super::helpers::*;
use super::*;
use crate::assets::{Camera3D, HitRegion, Sprite, TextLabel, WindowMode};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult};
use crate::gfx::backend::{ChunkMesh, FrameParams};
use crate::gfx::{draw_list, scene_reel, settings};
// The settings-row helpers this system's init-time captures share with the
// SettingCommand drain (which now lives in `settings_system`).
use crate::gfx::settings_system::rows::{
    DISABLED_ROW_COLOR, capture_row_labels, cycle_next_key_of, expand_dim_set, rebind_key_of,
    set_label_content, set_rows_grayed, set_sprite_x, slider_key_of,
};

impl GraphicsSystem {
    pub(super) fn run_step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        if self.failed {
            return StepResult::Done;
        }
        // Take the parked backend for this step (see `ActiveRenderBackend`);
        // it is a plain local from here on, so `ctx` stays freely borrowable.
        let Some(mut backend) = crate::ecs::ActiveRenderBackend::take(ctx.resources) else {
            return StepResult::Done;
        };
        let result = self.step_with_backend(ctx, backend.as_mut());
        crate::ecs::ActiveRenderBackend::put(ctx.resources, backend);
        result
    }

    fn step_with_backend(
        &mut self,
        ctx: &mut PipelineContext,
        backend: &mut dyn crate::gfx::backend::RenderBackend,
    ) -> StepResult {
        // The FPS-cap pacer runs at the App level before the world steps (see
        // `app::pacing`), so `elapsed` here already reflects the capped
        // interval.
        let elapsed = self
            .start_time
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0);

        // read projection and view from Camera3D; view_matrix was written by
        // Camera3DSystem on the previous tick
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

        // The overlay draw list + resolved menu state OverlaySystem built
        // earlier this tick, taken so a stale build is never redrawn.
        let overlay = ctx
            .resources
            .remove::<crate::gfx::overlay::OverlayFrame>()
            .unwrap_or_default();
        let menu_active = overlay.menu_active;
        // The editor's menu-state override also drives the backend's menu mode
        // below (OverlaySystem already folded it into `menu_active`).
        let menu_override = ctx.resource::<crate::ecs::MenuOverride>().and_then(|m| m.0);

        // Hide the system cursor while an in-engine cursor sprite is shown
        // (edge-triggered in the backend, so this is cheap every frame).
        backend.set_ui_cursor_hidden(overlay.want_ui_cursor);

        // The `MenuOverride` driver also needs the backend in menu mode so a
        // click with a freed cursor fires a UI action instead of re-capturing the
        // camera; a genuine menu-mode world already had this set at init.
        if menu_override.is_some() {
            backend.set_menu_mode(true);
        }
        // In menu mode (a MainMenu over a controlled camera, or an editor
        // override), capture the cursor for the camera unless a menu is active.
        // Edge-triggered in the backend, so this is cheap every frame and a no-op
        // in a plain first-person world.
        if self.menu_mode || menu_override.is_some() {
            backend.set_camera_capture(!menu_active);
        }

        // Runtime decal / emitter spawn (`cn debug` only) is drained + dispatched
        // from the binary's `DebugHook::tick` (see `crate::debug::runtime_spawn`),
        // not here. `cn run` has no debug hook, so this step never touches it.

        // Asset / shader / world.jsonl hot-reload (`cn debug` only) is driven
        // from the binary's `DebugHook::tick` (see the `debug` module), not
        // here: it reaches the reload passes through
        // `GraphicsSystem::hot_reload_drive`. `cn run` has no debug hook, so
        // this per-frame path is reload-free.

        let result = {
            {
                if backend.window_closed() {
                    tracing::info!("GraphicsSystem: window closed");
                    backend.wait_idle();
                    return StepResult::Stop;
                }

                // Lifetime/Spawner ticks and the spawn / despawn / reparent
                // drains run in SpawnSystem, scheduled immediately before this
                // system, so the churn is already applied when transforms are
                // pushed below.

                // Push updated model matrices for any entity whose transform
                // changed since last frame (physics, camera interact, reparent):
                // resolve each entity's GlobalTransform from Transform + Parent
                // (top-down so parents propagate to children), then push it to the
                // entity's GPU draw slots.
                draw_list::propagate_transforms(ctx);
                for (_entity, global, handle) in
                    ctx.join2::<crate::assets::GlobalTransform, crate::assets::RenderHandle>()
                {
                    for &slot in &handle.draws {
                        backend.update_model(slot as usize, global.0);
                    }
                }

                // Push the latest skinned poses to the GPU. AnimationSystem
                // wrote them into the SkeletonPose components on the previous
                // tick; the one-frame lag is invisible at animation rates.
                // Skipped while a menu is open: animation is frozen, so the
                // poses are unchanged and the last upload still stands (the
                // skinned draw is skipped behind an opaque menu anyway).
                if !menu_active {
                    for pose in ctx.query::<crate::assets::SkeletonPose>() {
                        backend.update_skinned_pose(pose.skinned_index, &pose.joint_matrices);
                    }
                    // Push the model matrix for skinned instances that carry a
                    // Transform (the runtime-spawned ones), so a moved instance
                    // follows it. The authored templates have no Transform and keep
                    // the model baked into their draw object at load.
                    for (_entity, pose, transform) in
                        ctx.join2::<crate::assets::SkeletonPose, crate::assets::Transform>()
                    {
                        backend.update_skinned_model(pose.skinned_index, transform.model_matrix());
                    }
                    // Move rig-driven meshes to their capsule's resolved
                    // position (PhysicsSystem wrote it on the previous tick;
                    // `moved` persists across a menu pause, so no motion is
                    // lost while uploads are skipped).
                    for rig in ctx.query_mut::<crate::assets::CharacterRig>() {
                        if rig.moved {
                            backend.update_skinned_model(rig.skinned_index, rig.model());
                            rig.moved = false;
                        }
                    }
                }

                // SceneCommand / SettingCommand application lives in
                // SettingsSystem, scheduled just before this system, so a
                // change is already on the backend for this frame's submit
                // and a scene jump has primed the reel below.

                // Advance the SceneReel and apply fade / visibility changes,
                // sourcing visibility from the live per-entity components (the
                // snapshot is rebuilt each frame the reel exists). The reel is
                // the shared `ActiveSceneReel` resource SettingsSystem also
                // jumps; its `epoch` is the shared clock for the fade timing.
                let reel_active = ctx
                    .resource::<crate::ecs::ActiveSceneReel>()
                    .is_some_and(|r| r.reel.is_some());
                if reel_active {
                    let (draws, scenes) = super::scene::decomposed_visibility_snapshot(ctx);
                    if let Some(slot) = ctx.resources.get_mut::<crate::ecs::ActiveSceneReel>() {
                        let reel_elapsed = slot.epoch.elapsed().as_secs_f32();
                        scene_reel::tick_reel(
                            &mut slot.reel,
                            &draws,
                            &scenes,
                            reel_elapsed,
                            backend,
                        );
                    }
                }

                // Drive albedo-texture streaming: re-score every slot by
                // camera distance, dispatch this frame's background loads
                // within budget, then apply completed uploads + evictions.
                // Each backend's update_texture_slot rewrites whichever
                // descriptors / argument-buffers sample that slot so it
                // takes effect on this same draw_frame.
                if !overlay.world_hidden
                    && let Some(streamer) = &mut self.texture_streamer
                {
                    streamer.update_scores(cam_pos, self.frame_count);
                    for slot in streamer.plan_and_dispatch() {
                        if let Err(e) = backend.evict_texture_slot(slot) {
                            tracing::warn!("GraphicsSystem: texture evict slot {}: {}", slot, e);
                        }
                    }
                    streamer.drain_completed(self.frame_count, |slot, w, h, px| {
                        if let Err(e) = backend.update_texture_slot(slot, w, h, px) {
                            tracing::warn!("GraphicsSystem: texture upload slot {}: {}", slot, e);
                        }
                    });
                    // Surface streaming progress periodically so a headless
                    // run can confirm textures are coming resident.
                    if self.frame_count.is_multiple_of(120) {
                        let (resident, pending, unloaded) = streamer.stats();
                        tracing::info!(
                            "GraphicsSystem: texture streaming -- {} resident, {} pending, {} unloaded",
                            resident,
                            pending,
                            unloaded
                        );
                    }
                }

                // Drive mesh-geometry streaming: re-score each streamed mesh
                // by camera distance, dispatch this frame's background loads,
                // then apply completed geometry uploads + evictions. A mesh is
                // skipped in every pass until its geometry region is resident.
                if !overlay.world_hidden
                    && let Some(streamer) = &mut self.mesh_streamer
                {
                    streamer.update_scores(cam_pos, self.frame_count);
                    // A runtime eviction's freed space must not be reused
                    // until the in-flight command buffers that drew it retire.
                    let retire_frame = self.frame_count + self.frames_in_flight as u64;
                    for stream_id in streamer.plan_and_dispatch() {
                        if let Some(&draw_idx) = self.mesh_stream_draw_indices.get(stream_id)
                            && let Err(e) = backend.evict_mesh(draw_idx, retire_frame)
                        {
                            tracing::warn!("GraphicsSystem: mesh evict draw {}: {}", draw_idx, e);
                        }
                    }
                    let draw_indices = &self.mesh_stream_draw_indices;
                    let frame = self.frame_count;
                    streamer.drain_completed(self.frame_count, |stream_id, verts, idxs| {
                        match draw_indices.get(stream_id) {
                            // Return the upload result so the streamer can roll
                            // a transient seed-full miss back to Unloaded and
                            // retry it once freed regions reclaim, rather than
                            // marking the mesh resident with no GPU geometry.
                            Some(&draw_idx) => backend.upload_mesh(draw_idx, verts, idxs, frame),
                            None => Ok(()),
                        }
                    });
                    if self.frame_count.is_multiple_of(120) {
                        let (resident, pending, unloaded) = streamer.stats();
                        tracing::info!(
                            "GraphicsSystem: mesh streaming -- {} resident, {} pending, {} unloaded",
                            resident,
                            pending,
                            unloaded
                        );
                    }
                }

                // Drive infinite-world chunk streaming: generate + upload the
                // chunks entering the camera's view window and remove those
                // that have left it. None unless a VoxelWorld was declared.
                //
                // Camera-relative rendering: chunk geometry is placed
                // relative to a render origin that follows the camera's chunk,
                // and the view + camera position handed to the backend are
                // rebased onto the same origin. The world transform is
                // unchanged -- it is just evaluated from small coordinates, so
                // an unbounded world renders without large-coordinate jitter.
                // `final_view` / `final_cam_pos` stay absolute when no
                // VoxelWorld is streaming, leaving a non-voxel world
                // byte-for-byte unchanged.
                let mut final_view = view_matrix;
                let mut final_cam_pos = cam_pos;
                if let Some(cs) = &mut self.chunk_stream {
                    let camera_chunk = cs.streamer.camera_chunk(cam_pos);
                    let retire_frame = self.frame_count + self.frames_in_flight as u64;
                    for coord in cs.streamer.plan_and_dispatch(camera_chunk) {
                        if let Some(draw_idx) = cs.draws.remove(&coord)
                            && let Err(e) = backend.remove_chunk_mesh(draw_idx, retire_frame)
                        {
                            tracing::warn!(
                                "GraphicsSystem: chunk remove ({},{}): {}",
                                coord.x,
                                coord.z,
                                e
                            );
                        }
                    }
                    // The camera crossed into a new chunk: move the render
                    // origin to it and rebase every resident chunk's model
                    // matrix. `prev_draw_models` is deliberately left alone --
                    // the rebase is exact, so a stationary chunk shows zero TAA
                    // velocity across the shift.
                    if camera_chunk != cs.origin_chunk {
                        for (&coord, &draw_idx) in &cs.draws {
                            let model =
                                chunk_model_matrix(coord, camera_chunk, cs.chunk_w, cs.chunk_d);
                            if let Err(e) = backend.set_chunk_model(draw_idx, model) {
                                tracing::warn!(
                                    "GraphicsSystem: chunk rebase ({},{}): {}",
                                    coord.x,
                                    coord.z,
                                    e
                                );
                            }
                        }
                        cs.origin_chunk = camera_chunk;
                    }
                    let frame = self.frame_count;
                    let (chunk_w, chunk_d) = (cs.chunk_w, cs.chunk_d);
                    let (tex, nm, mat) = (cs.texture_slot, cs.normal_map_slot, cs.material);
                    let mut added: Vec<(crate::gfx::chunk_coord::ChunkCoord, usize)> = Vec::new();
                    cs.streamer.drain_completed(|coord, verts, idxs| {
                        let model = chunk_model_matrix(coord, camera_chunk, chunk_w, chunk_d);
                        match backend.add_chunk_mesh(ChunkMesh {
                            verts,
                            idxs,
                            model,
                            texture_slot: tex,
                            normal_map_slot: nm,
                            material: mat,
                            frame,
                        }) {
                            Ok(draw_idx) => added.push((coord, draw_idx)),
                            Err(e) => tracing::warn!(
                                "GraphicsSystem: chunk add ({},{}): {}",
                                coord.x,
                                coord.z,
                                e
                            ),
                        }
                    });
                    for (coord, draw_idx) in added {
                        cs.draws.insert(coord, draw_idx);
                    }
                    // Rebase the view + camera onto the render origin so the
                    // origin-relative chunk geometry above transforms exactly.
                    let (ox, oz) = camera_chunk.origin_world(cs.chunk_w, cs.chunk_d);
                    let origin = [ox, 0.0, oz];
                    final_view =
                        crate::gfx::chunk_coord::camera_relative_view(view_matrix, cam_pos, origin);
                    final_cam_pos = [cam_pos[0] - ox, cam_pos[1], cam_pos[2] - oz];
                    if self.frame_count.is_multiple_of(120) {
                        let (resident, pending) = cs.streamer.stats();
                        let (near, far) = cs.streamer.detail_counts();
                        tracing::info!(
                            "GraphicsSystem: chunk streaming -- {} resident ({} full, {} impostor), {} pending",
                            resident,
                            near,
                            far,
                            pending
                        );
                    }
                }

                // On Metal, pump_ns_events runs inside draw_frame, so update_view
                // is called first so any key/mouse events that arrived since the
                // last tick are in InputState before InputSystem's take_input()
                // (scheduled right after this system) snapshots and clears it.
                backend.update_view(final_view);
                match backend.draw_frame(FrameParams {
                    elapsed,
                    fov_y_radians,
                    near,
                    far,
                    cam_pos: final_cam_pos,
                    text_calls: &overlay.calls,
                    world_hidden: overlay.world_hidden,
                }) {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("GraphicsSystem: draw_frame: {}", e);
                        backend.wait_idle();
                        return StepResult::Stop;
                    }
                }

                // Publish this frame's render stats for the profiler overlay.
                // Backends without GPU-timed stats return the trait's default
                // (all zeros), which the HUD displays as "--".
                ctx.profile.render = backend.render_stats();

                // Input sampling + FrameInput publish live in InputSystem,
                // which the schedule runs immediately after this system.

                StepResult::Continue
            }
        };

        if result == StepResult::Continue {
            self.frame_count += 1;
            if let Some(max) = self.max_frames
                && self.frame_count >= max
            {
                tracing::info!("GraphicsSystem: max_frames ({}) reached", max);
                backend.wait_idle();
                return StepResult::Done;
            }
        }

        result
    }

    // Capture each slider row's runtime bookkeeping from its drag HitRegion +
    // handle Sprite, then sync the handle position and value label to the live
    // value. Runs once at init, before UiInputSystem drains the HitRegions and
    // hides the view elements. The HitRegions / Sprites are still present here.
    pub(super) fn init_sliders(&mut self, ctx: &mut PipelineContext) {
        let sprite_w: std::collections::HashMap<AssetId, f32> = ctx
            .query::<Sprite>()
            .map(|s| (s.asset_id, s.width))
            .collect();
        let mut sliders: Vec<SliderViz> = Vec::new();
        for r in ctx.query::<HitRegion>() {
            let Some(key) = slider_key_of(&r.action) else {
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
        // Sync each slider's handle + value label to its live value.
        for s in &sliders {
            let Some(value) = self.slider_current_value(&s.key) else {
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
        for r in ctx.query::<HitRegion>() {
            let Some(key) = rebind_key_of(&r.action) else {
                continue;
            };
            let (Some(action), Some(value_id)) =
                (crate::gfx::keymap::Bindable::from_setting_key(key), r.label)
            else {
                continue;
            };
            rows.push(RebindViz { action, value_id });
        }
        // Sync each value label to the live bound key (persisted or default).
        for row in &rows {
            let name = self.keymap.get(row.action).display_name();
            set_label_content(ctx, row.value_id, name);
        }
        self.rebind_rows = rows;
    }

    // Capture each cycle row's setting key -> value-label id, so a runtime change
    // can relabel a row other than the one clicked (the master preset relabels
    // its dependents; a quality-toggle change relabels the master row). Runs at
    // init, before UiInputSystem drains the HitRegions (GraphicsSystem.init runs
    // first), since they cannot be re-queried once drained.
    pub(super) fn init_cycle_value_labels(&mut self, ctx: &mut PipelineContext) {
        let mut labels = std::collections::HashMap::new();
        for r in ctx.query::<HitRegion>() {
            if let (Some(key), Some(value_id)) = (cycle_next_key_of(&r.action), r.label) {
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
    // post-process params. `None` for a key this system does not own.
    fn slider_current_value(&self, key: &str) -> Option<f32> {
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
            // Mouse sensitivity lives in the controls store (radians/pixel), not
            // the render params; read the persisted value or the authored default.
            "mouse_sensitivity" => crate::config::Settings::load()
                .controls
                .mouse_sensitivity
                .unwrap_or(settings::DEFAULT_MOUSE_SENSITIVITY),
            // FOV lives in the graphics store (degrees); read the persisted value
            // or the authored default.
            "fov" => crate::config::Settings::load()
                .graphics
                .fov
                .unwrap_or(settings::DEFAULT_FOV),
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
