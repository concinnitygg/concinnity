// src/editor/hook/pick.rs
//
// EditorHook: viewport click-to-select. A press that missed the top bar and
// every floating panel is offered to the 3D view: the mouse ray (built from
// the live Camera3D) is tested against the engine-published PickIndex, and the
// nearest hit resolves through the interner's name table into the selection
// set (`editor/selection.rs`). A plain click replaces the selection; a
// shift-click toggles the hit's membership; a press over empty space arms the
// marquee (`hook/marquee_drag.rs`), whose still release clears. A repeat plain
// click on the same spot cycles through overlapping hits near-to-far, which is
// the only way to reach an occluded object without gizmos.

use super::*;
use concinnity_core::gfx::pick::{PickRay, ray_aabb, screen_ray};

// A repeat click within this many pixels of the last one cycles the hit list
// instead of restarting it.
const CYCLE_SLOP_PX: f32 = 4.0;

// The last viewport pick: where it was pressed, the hit list near-to-far, and
// which of those hits is currently selected.
pub(super) struct PickLast {
    pos: [f32; 2],
    hits: Vec<AssetId>,
    index: usize,
}

// The interned name behind a pick hit, if the id still resolves.
pub(super) fn resolve_name(id: AssetId) -> Option<String> {
    crate::ecs::asset_id::name_table()
        .get(id.0 as usize)
        .cloned()
}

impl EditorHook {
    // Resolve an unclaimed press as a viewport pick. Play mode never reaches
    // this: `left_click` stays false while the world holds the cursor.
    pub(super) fn click_world(&mut self, input: &FrameInput, world: &mut World) {
        let mouse = [input.mouse_x, input.mouse_y];
        let Some(ray) = camera_ray(world, input.viewport, mouse) else {
            return;
        };
        let hits = ray_hits(world, &ray);
        if hits.is_empty() {
            // Empty space arms the marquee; its release decides between a box
            // select (moved) and a clearing click (still).
            self.begin_marquee(mouse, input.shift);
            return;
        }

        if input.shift {
            // Shift-click toggles the nearest hit's membership. No cycling:
            // that is the plain click's repeat behavior.
            self.pick_last = None;
            let Some(name) = resolve_name(hits[0]) else {
                return;
            };
            if self.selection.toggle(name.clone()) {
                self.focus_ui_on(&name, world);
            } else {
                self.follow_active(world);
            }
            return;
        }

        // A repeat click over the same hit list advances the cycle; anything
        // else selects the nearest hit and restarts it.
        let index = match &self.pick_last {
            Some(last)
                if last.hits == hits
                    && (last.pos[0] - mouse[0]).abs() <= CYCLE_SLOP_PX
                    && (last.pos[1] - mouse[1]).abs() <= CYCLE_SLOP_PX =>
            {
                (last.index + 1) % hits.len()
            }
            _ => 0,
        };
        let picked = hits[index];
        self.pick_last = Some(PickLast {
            pos: mouse,
            hits,
            index,
        });
        let Some(name) = resolve_name(picked) else {
            self.selection.clear();
            return;
        };
        self.selection.replace(name.clone());
        self.focus_ui_on(&name, world);
    }

    // Open the picked asset for editing: an authored entry in the edit form
    // (with the assets UI up so the form is visible and its browse row
    // highlighted), a build-generated asset on the Expanded tab, where its "+"
    // offers the copy-to-config promotion.
    fn focus_ui_on(&mut self, name: &str, world: &mut World) {
        self.panel_open = true;
        self.combo = Combo::Closed;
        self.row_menu = None;
        if let Some(idx) = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))
        {
            self.tab = Tab::Config;
            if let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from) {
                self.open_form(world, ty, Some(idx));
            }
        } else {
            self.tab = Tab::Expanded;
            self.expanded_stale = true;
            self.refresh_expanded_if_needed();
            self.reveal_expanded(name);
        }
    }

    // Retarget an already-open edit form at the active member (the form
    // follows the active member; a closed form stays closed, so a marquee or
    // a toggle-off never forces panels open).
    pub(super) fn follow_active(&mut self, world: &mut World) {
        if self.editing.is_none() {
            return;
        }
        let Some(idx) = self.selection.active().and_then(|name| {
            self.entries
                .iter()
                .position(|e| entry_name(e) == Some(name))
        }) else {
            return;
        };
        if let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from) {
            self.tab = Tab::Config;
            self.open_form(world, ty, Some(idx));
        }
    }

    // Drive the selection rings: while the HUD is up in edit mode, project
    // each selected asset's current AABB (from the PickIndex GraphicsSystem
    // published last frame; the world is frozen in edit mode, so the one-frame
    // lag is invisible) and place a border sprite over it, the active member
    // in the full accent. Anything unresolvable -- a renamed or deleted asset,
    // the camera inside the box -- goes ringless.
    pub(super) fn drive_highlight(&self, world: &mut World, vp: [f32; 2], shown: bool) {
        if !shown || self.world_capture {
            highlight::hide(world);
            return;
        }
        let active = self.selection.active();
        let rects: Vec<([f32; 4], bool)> = self
            .selection
            .iter()
            .filter_map(|name| {
                Self::member_rect(world, vp, name).map(|r| (r, Some(name) == active))
            })
            .take(highlight::MAX_RINGS)
            .collect();
        highlight::place_all(world, &rects);
    }

    // A selection member's projected screen rect, if it resolves this frame.
    pub(super) fn member_rect(world: &World, vp: [f32; 2], name: &str) -> Option<[f32; 4]> {
        let id = crate::ecs::asset_id::name_table()
            .iter()
            .position(|n| n == name)?;
        let index = world.resource::<crate::ecs::PickIndex>()?;
        let entry = index
            .entries
            .iter()
            .find(|e| e.asset_id == AssetId(id as u32))?;
        let cam = world.query::<crate::assets::Camera3D>().next()?;
        highlight::screen_rect(
            &cam.view_matrix,
            cam.fov_y_degrees.to_radians(),
            vp,
            entry.bb_min,
            entry.bb_max,
        )
    }

    // Unfold the group holding generated asset `name` and scroll its row into
    // the Expanded tab's window. A name the expansion does not know (a cook
    // failure, a stale index entry) leaves the tab as-is.
    fn reveal_expanded(&mut self, name: &str) {
        let Some(group) = self
            .expanded_groups
            .iter()
            .position(|g| g.assets.iter().any(|a| a.name == name))
        else {
            return;
        };
        if !self.expanded_open.contains(&group) {
            self.expanded_open.push(group);
        }
        let rows = self.expanded_rows();
        let Some(row) = rows
            .iter()
            .position(|r| matches!(r, ExpandedRow::Asset { name: n, .. } if n == name))
        else {
            return;
        };
        // Scroll only when the row is outside the visible window, keeping its
        // group header in view when it sits directly above.
        if row < self.expanded_scroll || row >= self.expanded_scroll + panel::MAX_ROWS {
            let max = rows.len().saturating_sub(panel::MAX_ROWS);
            self.expanded_scroll = row.saturating_sub(1).min(max);
        }
    }
}

// The mouse ray from the world's live camera, or `None` in a camera-less
// world (nothing 3D to pick). Shared with the gizmo drag drive.
pub(super) fn camera_ray(world: &World, viewport: [f32; 2], mouse: [f32; 2]) -> Option<PickRay> {
    let cam = world.query::<crate::assets::Camera3D>().next()?;
    screen_ray(
        &cam.view_matrix,
        cam.position,
        cam.fov_y_degrees.to_radians(),
        viewport,
        mouse,
    )
}

// Every PickIndex entry the ray strikes, nearest first.
fn ray_hits(world: &World, ray: &PickRay) -> Vec<AssetId> {
    let Some(index) = world.resource::<crate::ecs::PickIndex>() else {
        return Vec::new();
    };
    let mut hits: Vec<(f32, AssetId)> = index
        .entries
        .iter()
        .filter_map(|e| ray_aabb(ray, e.bb_min, e.bb_max).map(|t| (t, e.asset_id)))
        .collect();
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    hits.into_iter().map(|(_, id)| id).collect()
}
