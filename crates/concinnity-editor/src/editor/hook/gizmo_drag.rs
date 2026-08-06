// src/editor/hook/gizmo_drag.rs
//
// EditorHook: the gizmo drive over the whole selection. One handle skeleton,
// three modes (T/R/S keys), anchored at the selection centroid: translate
// applies one shared world-space delta to every member; rotate orbits member
// positions about the centroid while spinning each member the same amount
// (the standard group rotate); scale stretches each member and pushes its
// position away from the centroid along the dragged axis. While the button is
// held every member's live `Transform` changes (the renderer, pick index, and
// selection rings all follow, so the scene previews the edit without a
// rebuild); releasing commits the changed args of ALL members to their
// authored entries as ONE undo step (`mark_changed` snapshots the pre-drag
// entry list once); Escape cancels and restores the start state.

use super::*;
use crate::assets::{Camera3D, GlobalTransform, Transform};
use gizmo::GizmoMode;

// Committed values are rounded so world.jsonl stays readable: positions and
// scales to 3 decimals, angles to 1.
pub(super) fn round3(v: f32) -> f32 {
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

// One selection member the gizmo can edit this frame, resolved fresh every
// use (rebuilds re-mint entities): the authored entry index and its live
// entity. Members without the mode's editable arg, generated assets (no entry
// to write back), parented entities (the drag works in world axes; a rotated
// parent would skew it), and entities without a Transform are skipped.
pub(super) struct GizmoTarget {
    pub(super) idx: usize,
    pub(super) entity: crate::ecs::Entity,
    // Whether the entry takes a written-back `position`: rotate and scale
    // move members about the pivot, and a type without the arg must only
    // spin / stretch in place.
    pub(super) has_position: bool,
}

// A member's `Transform` fields at the press: the drag base and the cancel
// target, keyed by name so each frame re-resolves the live entity.
struct MemberStart {
    name: String,
    position: [f32; 3],
    rotation: [f32; 3],
    scale: [f32; 3],
    has_position: bool,
}

pub(super) struct GizmoDrag {
    axis: usize,
    // The mode at the press; a mode key mid-drag must not morph the drag.
    mode: GizmoMode,
    // The selection centroid at the press: the drag anchor and the rotate /
    // scale pivot.
    pivot: [f32; 3],
    starts: Vec<MemberStart>,
    // Translate / Scale: the axis parameter under the cursor at the press, so
    // the selection keeps its grab offset instead of snapping to the cursor.
    grab_t: f32,
    // Rotate: the previous frame's screen angle about the gizmo origin and
    // the rotation accumulated so far. Accumulating per-frame wrapped steps
    // keeps a long drag continuous across the atan2 seam.
    last_angle: f32,
    accum_deg: f32,
}

impl EditorHook {
    pub(super) fn member_target(
        &self,
        world: &World,
        mode: GizmoMode,
        name: &str,
    ) -> Option<GizmoTarget> {
        let idx = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))?;
        let ty = entry_type(self.entries.get(idx)?)?;
        let merged = form::working_args(ty, Some(&self.entry_args(idx)));
        if !merged.get(mode.arg_key()).is_some_and(|p| p.is_array()) {
            return None;
        }
        let has_position = merged.get("position").is_some_and(|p| p.is_array());
        let id = crate::ecs::asset_id::lookup(name)?;
        let entity = world
            .resource::<concinnity_core::ecs::EntityByName>()?
            .get(id)?;
        if world.get::<crate::assets::Parent>(entity).is_some() {
            return None;
        }
        world.get::<Transform>(entity)?;
        Some(GizmoTarget {
            idx,
            entity,
            has_position,
        })
    }

    // Every selection member editable in `mode`, in selection order.
    fn gizmo_targets(&self, world: &World, mode: GizmoMode) -> Vec<GizmoTarget> {
        self.selection
            .iter()
            .filter_map(|name| self.member_target(world, mode, name))
            .collect()
    }

    // The entity's world origin: the propagated transform when the renderer
    // has produced one, else the raw component (a headless test world).
    fn gizmo_origin(world: &World, entity: crate::ecs::Entity) -> Option<[f32; 3]> {
        world
            .get::<GlobalTransform>(entity)
            .map(|g| [g.0[3][0], g.0[3][1], g.0[3][2]])
            .or_else(|| world.get::<Transform>(entity).map(|t| t.position))
    }

    // The gizmo anchor: the centroid of the editable members' world origins.
    fn gizmo_anchor(&self, world: &World, mode: GizmoMode) -> Option<[f32; 3]> {
        let origins: Vec<[f32; 3]> = self
            .gizmo_targets(world, mode)
            .iter()
            .filter_map(|t| Self::gizmo_origin(world, t.entity))
            .collect();
        (!origins.is_empty()).then(|| group_transform::centroid(&origins))
    }

    // The gizmo's screen layout this frame, when the selection has at least
    // one member editable in the current mode.
    pub(super) fn gizmo_layout(&self, world: &World, vp: [f32; 2]) -> Option<gizmo::Layout> {
        let anchor = self.gizmo_anchor(world, self.gizmo_mode)?;
        let cam = world.query::<Camera3D>().next()?;
        gizmo::layout(&cam.view_matrix, cam.fov_y_degrees.to_radians(), vp, anchor)
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
        let Some(pivot) = self.gizmo_anchor(world, mode) else {
            return false;
        };
        let starts: Vec<MemberStart> = self
            .gizmo_targets(world, mode)
            .into_iter()
            .filter_map(|t| {
                let tr = world.get::<Transform>(t.entity)?;
                Some(MemberStart {
                    name: entry_name(self.entries.get(t.idx)?)?.to_string(),
                    position: tr.position,
                    rotation: tr.rotation_deg,
                    scale: tr.scale,
                    has_position: t.has_position,
                })
            })
            .collect();
        if starts.is_empty() {
            return false;
        }
        let mut drag = GizmoDrag {
            axis,
            mode,
            pivot,
            starts,
            grab_t: 0.0,
            last_angle: 0.0,
            accum_deg: 0.0,
        };
        match mode {
            GizmoMode::Translate | GizmoMode::Scale => {
                let Some(ray) = pick::camera_ray(world, vp, mouse) else {
                    return false;
                };
                // The drag line stays anchored at the press-time pivot, so
                // each frame's parameter is measured against a stable line.
                let Some(t) = gizmo::axis_drag_t(pivot, gizmo::AXES[axis], &ray) else {
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
        if input.escape {
            let Some(drag) = self.gizmo_drag.take() else {
                return;
            };
            for s in &drag.starts {
                if let Some(t) = self.member_target(world, drag.mode, &s.name)
                    && let Some(tr) = world.get_mut::<Transform>(t.entity)
                {
                    tr.position = s.position;
                    tr.rotation_deg = s.rotation;
                    tr.scale = s.scale;
                }
            }
            return;
        }
        if self.gizmo_drag.is_none() {
            return;
        }
        if input.left_button_down {
            self.follow_gizmo_drag(input, vp, world);
            return;
        }
        self.commit_gizmo(world);
    }

    fn follow_gizmo_drag(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let mouse = [input.mouse_x, input.mouse_y];
        // Held Ctrl inverts each family's snap toggle for this drag.
        let translate_snap = self.snap.translate.active_step(input.ctrl);
        let rotate_snap = self.snap.rotate.active_step(input.ctrl);
        let mode = match &self.gizmo_drag {
            Some(d) => d.mode,
            None => return,
        };
        // Rotate needs this frame's angle step folded into the drag state
        // before the members move.
        let angle = match mode {
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
                    let a = gizmo::AXES[drag.axis];
                    let d = a[0] * fwd[0] + a[1] * fwd[1] + a[2] * fwd[2];
                    if d >= 0.0 { 1.0 } else { -1.0 }
                };
                // The accumulator stays continuous; only the applied angle
                // snaps, so toggling snap mid-drag cannot drift the base.
                let applied = sign * drag.accum_deg;
                match rotate_snap {
                    Some(step) => snap::snap_step(applied, step),
                    None => applied,
                }
            }
            _ => 0.0,
        };
        // The members are written with the drag moved out, so the resolves
        // below can borrow `self` freely.
        let Some(drag) = self.gizmo_drag.take() else {
            return;
        };
        let axis_i = drag.axis;
        let axis = gizmo::AXES[axis_i];
        match mode {
            GizmoMode::Translate | GizmoMode::Scale => {
                let Some(ray) = pick::camera_ray(world, vp, mouse) else {
                    self.gizmo_drag = Some(drag);
                    return;
                };
                // A parallel-degenerate frame keeps the last value.
                let Some(t) = gizmo::axis_drag_t(drag.pivot, axis, &ray) else {
                    self.gizmo_drag = Some(drag);
                    return;
                };
                // Snapping the shared delta (not each member's absolute
                // position) keeps the grab offset and the group's relative
                // offsets intact.
                let delta = match translate_snap {
                    Some(step) => snap::snap_step(t - drag.grab_t, step),
                    None => t - drag.grab_t,
                };
                for s in &drag.starts {
                    let Some(target) = self.member_target(world, mode, &s.name) else {
                        continue;
                    };
                    let Some(tr) = world.get_mut::<Transform>(target.entity) else {
                        continue;
                    };
                    if mode == GizmoMode::Translate {
                        tr.position = group_transform::translate(s.position, axis, delta);
                    } else {
                        let factor =
                            (t / drag.grab_t).clamp(SCALE_FACTOR_RANGE.0, SCALE_FACTOR_RANGE.1);
                        let mut scale = s.scale;
                        scale[axis_i] = s.scale[axis_i] * factor;
                        tr.scale = scale;
                        if s.has_position {
                            tr.position =
                                group_transform::scale_from(s.position, drag.pivot, axis_i, factor);
                        }
                    }
                }
            }
            GizmoMode::Rotate => {
                for s in &drag.starts {
                    let Some(target) = self.member_target(world, mode, &s.name) else {
                        continue;
                    };
                    let Some(tr) = world.get_mut::<Transform>(target.entity) else {
                        continue;
                    };
                    let mut rot = s.rotation;
                    rot[axis_i] = s.rotation[axis_i] + angle;
                    tr.rotation_deg = rot;
                    if s.has_position {
                        tr.position = group_transform::orbit(s.position, drag.pivot, axis_i, angle);
                    }
                }
            }
        }
        self.gizmo_drag = Some(drag);
    }

    // Write every member's dragged values into its authored entry, then mark
    // the whole edit as ONE change (a single history snapshot, so undo
    // restores all members together). A drag that ends where it started
    // leaves the entries (and the undo stack) untouched.
    fn commit_gizmo(&mut self, world: &mut World) {
        let Some(drag) = self.gizmo_drag.take() else {
            return;
        };
        let mut changed = Vec::new();
        for s in &drag.starts {
            let Some(target) = self.member_target(world, drag.mode, &s.name) else {
                continue;
            };
            let Some(tr) = world.get::<Transform>(target.entity).cloned() else {
                continue;
            };
            let (value, start) = match drag.mode {
                GizmoMode::Translate => (tr.position.map(round3), s.position.map(round3)),
                GizmoMode::Rotate => (tr.rotation_deg.map(round1), s.rotation.map(round1)),
                GizmoMode::Scale => (tr.scale.map(round3), s.scale.map(round3)),
            };
            let mut wrote = false;
            if value != start {
                Self::write_arg(&mut self.entries, target.idx, drag.mode.arg_key(), value);
                wrote = true;
            }
            // Rotate and scale also move members about the pivot: persist the
            // carried position alongside the mode's own arg.
            if drag.mode != GizmoMode::Translate && s.has_position {
                let pos = tr.position.map(round3);
                if pos != s.position.map(round3) {
                    Self::write_arg(&mut self.entries, target.idx, "position", pos);
                    wrote = true;
                }
            }
            if wrote {
                changed.push(target.idx);
            }
        }
        if changed.is_empty() {
            return;
        }
        self.mark_changed();
        // An open form on a moved entry still shows the pre-drag text;
        // re-derive it from the committed args. (Dragging and typing cannot
        // overlap, so no in-progress field edit is lost.) Through the by-name
        // path so a template-derived asset keeps its override state.
        if let Some(idx) = self.form_target.entry()
            && changed.contains(&idx)
            && let Some(name) = self.entries.get(idx).and_then(entry_name).map(String::from)
        {
            self.open_asset_form(&name, world);
        }
    }

    pub(super) fn write_arg(
        entries: &mut [serde_json::Value],
        idx: usize,
        key: &str,
        value: [f32; 3],
    ) {
        if let Some(obj) = entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
            let args = obj
                .entry("args".to_string())
                .or_insert_with(|| serde_json::Value::Object(Default::default()));
            if let Some(a) = args.as_object_mut() {
                a.insert(key.to_string(), serde_json::json!(value));
            }
        }
    }

    // Show or hide the gizmo sprites for this frame.
    pub(super) fn drive_gizmo_draw(&self, world: &mut World, vp: [f32; 2], shown: bool) {
        let layout = if shown && !self.sim.playing() {
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
