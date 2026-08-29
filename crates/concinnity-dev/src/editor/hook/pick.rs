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
    crate::ecs::asset_id::name_of(id)
}

impl EditorHook {
    // Resolve an unclaimed press as a viewport pick. Play mode never reaches
    // this: `left_click` stays false while the world holds the cursor.
    pub(super) fn click_world(&mut self, input: &FrameInput, world: &mut World) {
        let mouse = [input.mouse_x, input.mouse_y];
        let Some(ray) = camera_ray(world, input.viewport, mouse) else {
            return;
        };
        let hits = ray_hits(world, &ray, &self.locked_assets);
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
                self.select_in_viewport(&name, world);
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
        self.select_in_viewport(&name, world);
    }

    // Bring the UI along with a viewport selection without opening anything:
    // an already-open edit form retargets to the pick and an open Assets tree
    // reveals its row, but a closed form stays closed so clicking around the
    // scene never spawns panels.
    pub(super) fn select_in_viewport(&mut self, name: &str, world: &mut World) {
        self.follow_active(world);
        self.reveal_in_tree(name, world);
    }

    // Open the named asset for editing, with the assets UI up so the form is
    // visible and the tree row reveals itself (a deliberate "open" from the
    // palette, not a viewport click). A build-generated asset opens the same
    // form seeded from what the expansion produced, so confirming it
    // promotes the asset to an authored line.
    pub(super) fn focus_ui_on(&mut self, name: &str, world: &mut World) {
        self.panel_open = true;
        self.picker_open = false;
        self.row_menu = None;
        self.refresh_tree_if_needed();
        self.open_asset_form(name, world);
        self.reveal_in_tree(name, world);
    }

    // Retarget an already-open edit form at the active member (the form
    // follows the active member; a closed form stays closed, so a marquee or
    // a toggle-off never forces panels open).
    pub(super) fn follow_active(&mut self, world: &mut World) {
        if !self.form_open() {
            return;
        }
        let Some(name) = self.selection.active().map(String::from) else {
            return;
        };
        // Through the by-name path so a template-derived asset (an authored
        // patch line included) opens with its override state.
        self.open_asset_form(&name, world);
    }

    // Drive the selection rings: while the HUD is up in edit mode, project
    // each selected asset's current AABB (from the PickIndex GraphicsSystem
    // published last frame; the world is frozen in edit mode, so the one-frame
    // lag is invisible) and place a border sprite over it, the active member
    // in the full accent. Anything unresolvable -- a renamed or deleted asset,
    // the camera inside the box -- goes ringless.
    pub(super) fn drive_highlight(&self, world: &mut World, vp: [f32; 2], shown: bool) {
        if !shown || self.sim.playing() {
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
        let id = crate::ecs::asset_id::lookup(name)?;
        let index = world.resource::<crate::ecs::PickIndex>()?;
        let entry = index.entries.iter().find(|e| e.asset_id == id)?;
        let cam = world.query::<crate::components::Camera3D>().next()?;
        highlight::screen_rect(
            &cam.view_matrix,
            cam.fov_y_degrees.to_radians(),
            vp,
            entry.bb_min,
            entry.bb_max,
        )
    }
}

// The mouse ray from the world's live camera, or `None` in a camera-less
// world (nothing 3D to pick). Shared with the gizmo drag drive.
pub(super) fn camera_ray(world: &World, viewport: [f32; 2], mouse: [f32; 2]) -> Option<PickRay> {
    let cam = world.query::<crate::components::Camera3D>().next()?;
    screen_ray(
        &cam.view_matrix,
        cam.position,
        cam.fov_y_degrees.to_radians(),
        viewport,
        mouse,
    )
}

// Every PickIndex entry the ray strikes, nearest first. Locked assets (the
// tree's pick lock) are skipped, so a click passes through to whatever
// sits behind them.
fn ray_hits(
    world: &World,
    ray: &PickRay,
    locked: &std::collections::BTreeSet<String>,
) -> Vec<AssetId> {
    let Some(index) = world.resource::<crate::ecs::PickIndex>() else {
        return Vec::new();
    };
    let mut hits: Vec<(f32, AssetId)> = index
        .entries
        .iter()
        .filter(|e| !resolve_name(e.asset_id).is_some_and(|n| locked.contains(&n)))
        .filter_map(|e| ray_aabb(ray, e.bb_min, e.bb_max).map(|t| (t, e.asset_id)))
        .collect();
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    hits.into_iter().map(|(_, id)| id).collect()
}
