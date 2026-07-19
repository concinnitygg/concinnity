// src/editor/hook/gizmo_drag.rs
//
// EditorHook: the translate-gizmo drive. A press on an axis tip handle starts
// a drag; while the button is held the selected entity's live `Transform`
// moves along that world axis (the renderer, pick index, and selection ring
// all follow the transform, so the whole scene previews the move without a
// rebuild); releasing commits the position to the authored entry as ONE
// undo step (`mark_changed` snapshots the pre-drag entry list); Escape
// cancels and restores the start position.

use super::*;
use crate::assets::{Camera3D, GlobalTransform, Transform};

// Committed positions are rounded so world.jsonl stays readable.
fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

pub(super) struct GizmoDrag {
    axis: usize,
    // `Transform.position` at the press: the drag base and the cancel target.
    start: [f32; 3],
    // Axis parameter under the cursor at the press, so the object keeps its
    // grab offset instead of snapping to the cursor.
    grab_t: f32,
}

impl EditorHook {
    // The selected asset's movable target, resolved fresh every use (rebuilds
    // re-mint entities): the authored entry index and its live entity. `None`
    // when the gizmo must not show: a generated asset (no entry to write
    // back), a type without a `position` arg, a missing/parented entity (the
    // drag works in world axes; a rotated parent would skew it), or no
    // Transform to move.
    fn gizmo_target(&self, world: &World) -> Option<(usize, crate::ecs::Entity)> {
        let name = self.selected.as_deref()?;
        let idx = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))?;
        let ty = entry_type(self.entries.get(idx)?)?;
        let merged = form::working_args(ty, Some(&self.entry_args(idx)));
        if !merged.get("position").is_some_and(|p| p.is_array()) {
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

    // The gizmo's screen layout this frame, when the selection is movable.
    pub(super) fn gizmo_layout(&self, world: &World, vp: [f32; 2]) -> Option<gizmo::Layout> {
        let (_, entity) = self.gizmo_target(world)?;
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
        let Some(layout) = self.gizmo_layout(world, vp) else {
            return false;
        };
        let Some(axis) = gizmo::hit_axis(&layout, mouse) else {
            return false;
        };
        let Some((_, entity)) = self.gizmo_target(world) else {
            return false;
        };
        let Some(start) = world.get::<Transform>(entity).map(|t| t.position) else {
            return false;
        };
        let Some(ray) = pick::camera_ray(world, vp, mouse) else {
            return false;
        };
        // The drag line stays anchored at the press-time position, so each
        // frame's parameter is measured against a stable line.
        let Some(grab_t) = gizmo::axis_drag_t(start, gizmo::AXES[axis], &ray) else {
            return false;
        };
        self.gizmo_drag = Some(GizmoDrag {
            axis,
            start,
            grab_t,
        });
        true
    }

    // Per-frame drag drive: follow the cursor while the button is held,
    // cancel on Escape, commit on release.
    pub(super) fn drive_gizmo_drag(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let Some(drag) = &self.gizmo_drag else {
            return;
        };
        let (axis_i, start, grab_t) = (drag.axis, drag.start, drag.grab_t);
        if input.escape {
            if let Some((_, entity)) = self.gizmo_target(world)
                && let Some(t) = world.get_mut::<Transform>(entity)
            {
                t.position = start;
            }
            self.gizmo_drag = None;
            return;
        }
        if input.left_button_down {
            let mouse = [input.mouse_x, input.mouse_y];
            let axis = gizmo::AXES[axis_i];
            let Some(ray) = pick::camera_ray(world, vp, mouse) else {
                return;
            };
            // A parallel-degenerate frame keeps the last position.
            let Some(t) = gizmo::axis_drag_t(start, axis, &ray) else {
                return;
            };
            let delta = t - grab_t;
            if let Some((_, entity)) = self.gizmo_target(world)
                && let Some(tr) = world.get_mut::<Transform>(entity)
            {
                tr.position = [
                    start[0] + axis[0] * delta,
                    start[1] + axis[1] * delta,
                    start[2] + axis[2] * delta,
                ];
            }
            return;
        }
        self.commit_gizmo(world);
    }

    // Write the dragged position into the authored entry and mark the edit.
    // A drag that ends where it started leaves the entries (and the undo
    // stack) untouched.
    fn commit_gizmo(&mut self, world: &mut World) {
        let Some(drag) = self.gizmo_drag.take() else {
            return;
        };
        let Some((idx, entity)) = self.gizmo_target(world) else {
            return;
        };
        let Some(pos) = world.get::<Transform>(entity).map(|t| t.position) else {
            return;
        };
        let rounded = pos.map(round3);
        if rounded == drag.start.map(round3) {
            return;
        }
        if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
            let args = obj
                .entry("args".to_string())
                .or_insert_with(|| serde_json::Value::Object(Default::default()));
            if let Some(a) = args.as_object_mut() {
                a.insert("position".to_string(), serde_json::json!(rounded));
            }
        }
        self.mark_changed();
        // An open form on this entry still shows the pre-drag position text;
        // re-derive it from the committed args. (Dragging and typing cannot
        // overlap, so no in-progress field edit is lost.)
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
            Some(l) => gizmo::place(world, &l),
            None => gizmo::hide(world),
        }
    }
}
