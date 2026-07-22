// src/editor/hook/marquee_drag.rs
//
// EditorHook: the marquee (box-select) drive. A press over empty viewport
// space -- already past the top bar, the panels, and the gizmo handles by the
// time it reaches the pick -- arms a marquee; while the button is held the
// rect follows the cursor once it clears the start slop. Releasing selects
// every asset whose projected AABB intersects the rect (shift adds to the
// selection instead of replacing it); a release inside the slop is the plain
// empty-space click, which clears. Escape cancels. The fly camera and play
// mode capture the cursor, so `left_click` never arms a marquee there.

use super::*;

pub(super) struct MarqueeDrag {
    anchor: [f32; 2],
    current: [f32; 2],
    // Shift at the press: the release adds to the selection instead of
    // replacing it (and a still release leaves the selection alone).
    additive: bool,
    // Latched once the cursor clears the start slop; a still release is an
    // empty-space click, not a box select.
    moved: bool,
}

impl EditorHook {
    pub(super) fn begin_marquee(&mut self, anchor: [f32; 2], additive: bool) {
        self.marquee = Some(MarqueeDrag {
            anchor,
            current: anchor,
            additive,
            moved: false,
        });
    }

    // Per-frame marquee drive: follow the cursor while the button is held,
    // cancel on Escape, select on release.
    pub(super) fn drive_marquee(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let Some(m) = &mut self.marquee else {
            return;
        };
        if input.escape {
            self.marquee = None;
            return;
        }
        m.current = [input.mouse_x, input.mouse_y];
        let dx = (m.current[0] - m.anchor[0]).abs();
        let dy = (m.current[1] - m.anchor[1]).abs();
        if dx.max(dy) >= marquee::DRAG_START_PX {
            m.moved = true;
        }
        if input.left_button_down {
            return;
        }
        let Some(m) = self.marquee.take() else {
            return;
        };
        if !m.moved {
            // The plain empty-space click; with shift the selection is kept
            // (a missed shift-click must not throw the set away).
            if !m.additive {
                self.selection.clear();
                self.pick_last = None;
            }
            return;
        }
        let rect = marquee::rect_from_corners(m.anchor, m.current);
        let names = marquee_hits(world, vp, rect);
        if m.additive {
            self.selection.extend(names);
        } else {
            self.selection.set(names);
        }
        self.pick_last = None;
        self.follow_active(world);
    }

    // Show or hide the marquee rect for this frame.
    pub(super) fn drive_marquee_draw(&self, world: &mut World, shown: bool) {
        match &self.marquee {
            Some(m) if shown && m.moved => {
                marquee::place(world, marquee::rect_from_corners(m.anchor, m.current));
            }
            _ => marquee::hide(world),
        }
    }
}

// The names of every PickIndex entry whose projected AABB intersects `rect`,
// in index order. Entries whose projection fails (behind the camera, off
// screen) or whose id no longer resolves are skipped.
fn marquee_hits(world: &World, vp: [f32; 2], rect: [f32; 4]) -> Vec<String> {
    let Some(index) = world.resource::<crate::ecs::PickIndex>() else {
        return Vec::new();
    };
    let Some(cam) = world.query::<crate::assets::Camera3D>().next() else {
        return Vec::new();
    };
    let fov = cam.fov_y_degrees.to_radians();
    index
        .entries
        .iter()
        .filter_map(|e| {
            let r = highlight::screen_rect(&cam.view_matrix, fov, vp, e.bb_min, e.bb_max)?;
            if !marquee::rects_intersect(rect, r) {
                return None;
            }
            pick::resolve_name(e.asset_id)
        })
        .collect()
}
