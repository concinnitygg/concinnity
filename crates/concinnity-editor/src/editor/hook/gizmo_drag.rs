// src/editor/hook/gizmo_drag.rs
//
// EditorHook: the gizmo drive. One handle skeleton, three modes (T/R/S keys):
// translate drags the tip along its world axis, scale drags it to stretch the
// axis parameter ratio, rotate turns the mouse around the projected origin.
// While the button is held the selected entity's live `Transform` changes (the
// renderer, pick index, and selection ring all follow, so the scene previews
// the edit without a rebuild); releasing commits the mode's arg to the
// authored entry as ONE undo step (`mark_changed` snapshots the pre-drag entry
// list); Escape cancels and restores the start state.

use super::*;
use crate::assets::{Camera3D, GlobalTransform, Transform};
use gizmo::GizmoMode;

// Committed values are rounded so world.jsonl stays readable: positions and
// scales to 3 decimals, angles to 1.
fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

fn round1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

// A rotate grab needs a usable lever arm around the origin (the angle is
// unstable at the center; a camera-facing axis projects its tip there).
const MIN_ROTATE_ARM_PX: f32 = 15.0;

// Scale factors are clamped away from zero (no mirroring) and runaway growth.
const SCALE_FACTOR_RANGE: (f32, f32) = (0.01, 100.0);

pub(super) struct GizmoDrag {
    axis: usize,
    // The mode at the press; a mode key mid-drag must not morph the drag.
    mode: GizmoMode,
    // `Transform` fields at the press: the drag base and the cancel target.
    start_position: [f32; 3],
    start_rotation: [f32; 3],
    start_scale: [f32; 3],
    // Translate / Scale: the axis parameter under the cursor at the press, so
    // the object keeps its grab offset instead of snapping to the cursor.
    grab_t: f32,
    // Rotate: the previous frame's screen angle about the gizmo origin and
    // the rotation accumulated so far. Accumulating per-frame wrapped steps
    // keeps a long drag continuous across the atan2 seam.
    last_angle: f32,
    accum_deg: f32,
}

impl EditorHook {
    // The selected asset's editable target for `mode`, resolved fresh every
    // use (rebuilds re-mint entities): the authored entry index and its live
    // entity. `None` when the gizmo must not show: a generated asset (no
    // entry to write back), a type without the mode's arg, a missing or
    // parented entity (the drag works in world axes; a rotated parent would
    // skew it), or no Transform to edit.
    fn gizmo_target(&self, world: &World, mode: GizmoMode) -> Option<(usize, crate::ecs::Entity)> {
        let name = self.selected.as_deref()?;
        let idx = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))?;
        let ty = entry_type(self.entries.get(idx)?)?;
        let merged = form::working_args(ty, Some(&self.entry_args(idx)));
        if !merged.get(mode.arg_key()).is_some_and(|p| p.is_array()) {
            return None;
        }
        let id = crate::ecs::asset_id::name_table()
            .iter()
            .position(|n| n == name)?;
        let entity = world
            .resource::<concinnity_core::ecs::EntityByName>()?
            .get(AssetId(id as u32))?;
        if world.get::<crate::assets::Parent>(entity).is_some() {
            return None;
        }
        world.get::<Transform>(entity)?;
        Some((idx, entity))
    }

    // The entity's world origin: the propagated transform when the renderer
    // has produced one, else the raw component (a headless test world).
    fn gizmo_origin(world: &World, entity: crate::ecs::Entity) -> Option<[f32; 3]> {
        world
            .get::<GlobalTransform>(entity)
            .map(|g| [g.0[3][0], g.0[3][1], g.0[3][2]])
            .or_else(|| world.get::<Transform>(entity).map(|t| t.position))
    }

    // The gizmo's screen layout this frame, when the selection is editable in
    // the current mode.
    pub(super) fn gizmo_layout(&self, world: &World, vp: [f32; 2]) -> Option<gizmo::Layout> {
        let (_, entity) = self.gizmo_target(world, self.gizmo_mode)?;
        let origin = Self::gizmo_origin(world, entity)?;
        let cam = world.query::<Camera3D>().next()?;
        gizmo::layout(&cam.view_matrix, cam.fov_y_degrees.to_radians(), vp, origin)
    }

    // Offer an unclaimed press to the gizmo: `true` when a tip handle takes it
    // and a drag begins. Runs before the viewport pick, so grabbing a handle
    // never re-picks the object behind it.
    pub(super) fn try_gizmo_press(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        let mouse = [input.mouse_x, input.mouse_y];
        let mode = self.gizmo_mode;
        let Some(layout) = self.gizmo_layout(world, vp) else {
            return false;
        };
        let Some(axis) = gizmo::hit_axis(&layout, mouse) else {
            return false;
        };
        let Some((_, entity)) = self.gizmo_target(world, mode) else {
            return false;
        };
        let Some(start) = world.get::<Transform>(entity).cloned() else {
            return false;
        };
        let mut drag = GizmoDrag {
            axis,
            mode,
            start_position: start.position,
            start_rotation: start.rotation_deg,
            start_scale: start.scale,
            grab_t: 0.0,
            last_angle: 0.0,
            accum_deg: 0.0,
        };
        match mode {
            GizmoMode::Translate | GizmoMode::Scale => {
                let Some(ray) = pick::camera_ray(world, vp, mouse) else {
                    return false;
                };
                // The drag line stays anchored at the press-time position, so
                // each frame's parameter is measured against a stable line.
                let Some(t) = gizmo::axis_drag_t(start.position, gizmo::AXES[axis], &ray) else {
                    return false;
                };
                // A scale ratio needs a non-degenerate base parameter.
                if mode == GizmoMode::Scale && t.abs() < 1e-4 {
                    return false;
                }
                drag.grab_t = t;
            }
            GizmoMode::Rotate => {
                let (dx, dy) = (mouse[0] - layout.origin[0], mouse[1] - layout.origin[1]);
                if (dx * dx + dy * dy).sqrt() < MIN_ROTATE_ARM_PX {
                    return false;
                }
                drag.last_angle = dy.atan2(dx);
            }
        }
        self.gizmo_drag = Some(drag);
        true
    }

    // Per-frame drag drive: follow the cursor while the button is held,
    // cancel on Escape, commit on release.
    pub(super) fn drive_gizmo_drag(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let Some(drag) = &self.gizmo_drag else {
            return;
        };
        let mode = drag.mode;
        if input.escape {
            let (pos, rot, scale) = (drag.start_position, drag.start_rotation, drag.start_scale);
            if let Some((_, entity)) = self.gizmo_target(world, mode)
                && let Some(t) = world.get_mut::<Transform>(entity)
            {
                t.position = pos;
                t.rotation_deg = rot;
                t.scale = scale;
            }
            self.gizmo_drag = None;
            return;
        }
        if input.left_button_down {
            self.follow_gizmo_drag(input, vp, world);
            return;
        }
        self.commit_gizmo(world);
    }

    fn follow_gizmo_drag(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let Some(drag) = &mut self.gizmo_drag else {
            return;
        };
        let mouse = [input.mouse_x, input.mouse_y];
        let axis_i = drag.axis;
        let axis = gizmo::AXES[axis_i];
        let mode = drag.mode;
        match mode {
            GizmoMode::Translate | GizmoMode::Scale => {
                let start_pos = drag.start_position;
                let grab_t = drag.grab_t;
                let Some(ray) = pick::camera_ray(world, vp, mouse) else {
                    return;
                };
                // A parallel-degenerate frame keeps the last value.
                let Some(t) = gizmo::axis_drag_t(start_pos, axis, &ray) else {
                    return;
                };
                let start_scale = drag.start_scale;
                if let Some((_, entity)) = self.gizmo_target(world, mode)
                    && let Some(tr) = world.get_mut::<Transform>(entity)
                {
                    if mode == GizmoMode::Translate {
                        let delta = t - grab_t;
                        tr.position = [
                            start_pos[0] + axis[0] * delta,
                            start_pos[1] + axis[1] * delta,
                            start_pos[2] + axis[2] * delta,
                        ];
                    } else {
                        let factor = (t / grab_t).clamp(SCALE_FACTOR_RANGE.0, SCALE_FACTOR_RANGE.1);
                        let mut scale = start_scale;
                        scale[axis_i] = start_scale[axis_i] * factor;
                        tr.scale = scale;
                    }
                }
            }
            GizmoMode::Rotate => {
                // The turn is measured around the gizmo's projected origin.
                let Some(layout) = self.gizmo_layout(world, vp) else {
                    return;
                };
                let Some(drag) = &mut self.gizmo_drag else {
                    return;
                };
                let (dx, dy) = (mouse[0] - layout.origin[0], mouse[1] - layout.origin[1]);
                let cur = dy.atan2(dx);
                let step = gizmo::wrap_deg((cur - drag.last_angle).to_degrees());
                drag.last_angle = cur;
                drag.accum_deg += step;
                // Screen angles grow clockwise (y is down). Clockwise about an
                // axis pointing away from the viewer is a positive right-hand
                // rotation; toward the viewer it is negative.
                let sign = {
                    let cam = world.query::<Camera3D>().next();
                    let fwd = cam
                        .map(|c| {
                            [
                                -c.view_matrix[0][2],
                                -c.view_matrix[1][2],
                                -c.view_matrix[2][2],
                            ]
                        })
                        .unwrap_or([0.0, 0.0, -1.0]);
                    let d = axis[0] * fwd[0] + axis[1] * fwd[1] + axis[2] * fwd[2];
                    if d >= 0.0 { 1.0 } else { -1.0 }
                };
                let start_rot = drag.start_rotation;
                let accum = drag.accum_deg;
                if let Some((_, entity)) = self.gizmo_target(world, mode)
                    && let Some(tr) = world.get_mut::<Transform>(entity)
                {
                    let mut rot = start_rot;
                    rot[axis_i] = start_rot[axis_i] + sign * accum;
                    tr.rotation_deg = rot;
                }
            }
        }
    }

    // Write the dragged value into the authored entry and mark the edit. A
    // drag that ends where it started leaves the entries (and the undo stack)
    // untouched.
    fn commit_gizmo(&mut self, world: &mut World) {
        let Some(drag) = self.gizmo_drag.take() else {
            return;
        };
        let Some((idx, entity)) = self.gizmo_target(world, drag.mode) else {
            return;
        };
        let Some(tr) = world.get::<Transform>(entity).cloned() else {
            return;
        };
        let (value, start) = match drag.mode {
            GizmoMode::Translate => (tr.position.map(round3), drag.start_position.map(round3)),
            GizmoMode::Rotate => (tr.rotation_deg.map(round1), drag.start_rotation.map(round1)),
            GizmoMode::Scale => (tr.scale.map(round3), drag.start_scale.map(round3)),
        };
        if value == start {
            return;
        }
        if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
            let args = obj
                .entry("args".to_string())
                .or_insert_with(|| serde_json::Value::Object(Default::default()));
            if let Some(a) = args.as_object_mut() {
                a.insert(drag.mode.arg_key().to_string(), serde_json::json!(value));
            }
        }
        self.mark_changed();
        // An open form on this entry still shows the pre-drag text; re-derive
        // it from the committed args. (Dragging and typing cannot overlap, so
        // no in-progress field edit is lost.)
        if self.editing == Some(idx)
            && let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from)
        {
            self.open_form(world, ty, Some(idx));
        }
    }

    // Show or hide the gizmo sprites for this frame.
    pub(super) fn drive_gizmo_draw(&self, world: &mut World, vp: [f32; 2], shown: bool) {
        let layout = if shown && !self.world_capture {
            self.gizmo_layout(world, vp)
        } else {
            None
        };
        match layout {
            Some(l) => gizmo::place(world, &l, self.gizmo_mode),
            None => gizmo::hide(world),
        }
    }
}
