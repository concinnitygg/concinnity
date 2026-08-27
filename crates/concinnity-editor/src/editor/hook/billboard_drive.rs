// src/editor/hook/billboard_drive.rs
//
// EditorHook: the billboard drive. Assets with a world position but no
// rendered geometry (lights, trigger volumes, probes, cameras) have no mesh
// AABB, so the PickIndex never sees them; each frame this drive seeds a
// Transform onto every eligible entity (the gizmo's edit surface -- these
// types otherwise never get one), projects each position to a screen-space
// icon, and offers the icons to the click router ahead of the mesh pick.
// Selecting an icon goes through the same name-keyed selection the mesh pick
// uses, so the form, tree, and gizmo all follow for free.

use super::*;
use crate::components::{Camera3D, Transform};
use concinnity_core::gfx::pick::ray_aabb;

// One drawable / pickable billboard this frame: the authored entry it stands
// for, its projected center, and its straight-line camera distance (the
// arbitration metric against mesh ray hits).
pub(super) struct BillboardSpot {
    entry: usize,
    screen: [f32; 2],
    dist: f32,
}

// name -> interned id -> live entity, the same resolve the gizmo uses. The
// index is built at load, so entries despawned by the start-time drains
// (Window, GraphicsConfig, Scene, ...) are filtered by liveness.
pub(super) fn entity_by_name(world: &World, name: &str) -> Option<crate::ecs::Entity> {
    let id = crate::ecs::asset_id::lookup(name)?;
    world
        .resource::<concinnity_core::ecs::EntityByName>()?
        .get(id)
        .filter(|&e| world.is_alive(e))
}

impl EditorHook {
    // Seed a Transform onto every billboard-eligible entity that lacks one,
    // from its authored args. Runs every tick: entities are re-minted on each
    // preview rebuild, and an existing Transform (including one mid-gizmo-
    // drag) is left alone.
    pub(super) fn seed_billboard_transforms(&self, world: &mut World) {
        for e in &self.entries {
            let (Some(name), Some(ty)) = (entry_name(e), entry_type(e)) else {
                continue;
            };
            if !billboards::eligible(ty) {
                continue;
            }
            let Some(entity) = entity_by_name(world, name) else {
                continue;
            };
            if world.get::<Transform>(entity).is_some() {
                continue;
            }
            let merged = form::working_args(ty, e.get("args").and_then(|v| v.as_object()));
            let Some(position) = billboards::position_of(&merged) else {
                continue;
            };
            world.insert(
                entity,
                Transform {
                    position,
                    rotation_deg: billboards::vec3(&merged, "rotation_deg").unwrap_or([0.0; 3]),
                    scale: [1.0; 3],
                },
            );
        }
    }

    // Every billboard drawable this frame, in entry order: eligible, not
    // editor-hidden, resolving to a live seeded entity in front of the
    // camera.
    fn billboard_spots(&self, world: &World, vp: [f32; 2]) -> Vec<BillboardSpot> {
        let Some(cam) = world.query::<Camera3D>().next() else {
            return Vec::new();
        };
        let (view, fov, cam_pos) = (
            cam.view_matrix,
            cam.fov_y_degrees.to_radians(),
            cam.position,
        );
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| entry_type(e).is_some_and(billboards::eligible))
            .filter(|(_, e)| !entry_name(e).is_some_and(|n| self.name_hidden(n)))
            .filter_map(|(i, e)| {
                let entity = entity_by_name(world, entry_name(e)?)?;
                let p = world.get::<Transform>(entity)?.position;
                let (screen, _) = billboards::project(&view, fov, vp, p)?;
                let d = [p[0] - cam_pos[0], p[1] - cam_pos[1], p[2] - cam_pos[2]];
                Some(BillboardSpot {
                    entry: i,
                    screen,
                    dist: (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(),
                })
            })
            .take(billboards::MAX_BILLBOARDS)
            .collect()
    }

    // Place the billboard icons (and the active trigger volume's outline) for
    // this frame, or hide them while the HUD is down or the world holds the
    // cursor (play mode).
    pub(super) fn drive_billboards(&self, world: &mut World, vp: [f32; 2], shown: bool) {
        if !shown || self.sim.playing() {
            billboards::hide(world);
            return;
        }
        let icons: Vec<billboards::Icon> = self
            .billboard_spots(world, vp)
            .into_iter()
            .filter_map(|s| {
                let e = self.entries.get(s.entry)?;
                let (name, ty) = (entry_name(e)?, entry_type(e)?);
                Some(billboards::Icon {
                    screen: s.screen,
                    tint: billboards::tint(ty),
                    glyph: billboards::glyph(ty),
                    selected: self.selection.contains(name),
                    active: self.selection.active() == Some(name),
                })
            })
            .collect();
        billboards::place_icons(world, &icons);
        // The dotted-sprite outline pool has one tenant now: the drag-out
        // ghost. Extent wireframes draw through the renderer's line pass
        // (`hook/outline_drive.rs`).
        billboards::hide_outline(world);
        if self.content_ghost_pose().is_some() {
            self.drive_content_ghost(world, vp);
        }
    }

    // Offer an unclaimed viewport press to the billboards: `true` when an
    // icon takes it. Runs between the gizmo and the mesh pick; a mesh AABB
    // hit nearer than the icon's anchor keeps the press (the nearer thing
    // under the cursor wins), and locked assets are pick-through like the
    // mesh path.
    pub(super) fn try_billboard_press(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        // Hidden icons (the Display menu's billboard toggle) take no clicks.
        if !self.show_billboards {
            return false;
        }
        let mouse = [input.mouse_x, input.mouse_y];
        let spots: Vec<BillboardSpot> = self
            .billboard_spots(world, vp)
            .into_iter()
            .filter(|s| {
                !self
                    .entries
                    .get(s.entry)
                    .and_then(|e| entry_name(e))
                    .is_some_and(|n| self.locked_assets.contains(n))
            })
            .collect();
        let xy: Vec<([f32; 2], f32)> = spots.iter().map(|s| (s.screen, s.dist)).collect();
        let Some(i) = billboards::hit(&xy, mouse) else {
            return false;
        };
        if !billboards::beats_mesh(spots[i].dist, self.nearest_mesh_t(world, vp, mouse)) {
            return false;
        }
        let Some(name) = self
            .entries
            .get(spots[i].entry)
            .and_then(|e| entry_name(e))
            .map(String::from)
        else {
            return false;
        };
        // Same selection semantics as the mesh pick (no cycling: icons have
        // no occlusion stack of their own).
        self.pick_last = None;
        if input.shift {
            if self.selection.toggle(name.clone()) {
                self.select_in_viewport(&name, world);
            } else {
                self.follow_active(world);
            }
        } else {
            self.selection.replace(name.clone());
            self.select_in_viewport(&name, world);
        }
        true
    }

    // The nearest mesh AABB hit under the cursor as a camera distance, for
    // the billboard-vs-mesh arbitration. Locked assets are skipped, matching
    // the mesh pick's pass-through.
    fn nearest_mesh_t(&self, world: &World, vp: [f32; 2], mouse: [f32; 2]) -> Option<f32> {
        let ray = pick::camera_ray(world, vp, mouse)?;
        let index = world.resource::<crate::ecs::PickIndex>()?;
        index
            .entries
            .iter()
            .filter(|e| {
                !pick::resolve_name(e.asset_id).is_some_and(|n| self.locked_assets.contains(&n))
            })
            .filter_map(|e| ray_aabb(&ray, e.bb_min, e.bb_max))
            .min_by(f32::total_cmp)
    }
}
