// src/editor/hook/content_drag.rs
//
// EditorHook: drag-out placement from the Content panel. A press on a grid
// cell arms a drag; once the cursor travels past the slop and leaves the
// panel, a dotted ghost box follows the surface under the cursor (nearest
// pick-index hit, else the ground plane); release commits one authored entry
// through the same path every other creation uses -- one undo step -- and
// selects it. Dragging a Material instead assigns it to the Prop under the
// cursor. Escape cancels; a release back over the panel is just the click
// that already selected the cell.

use super::*;
use crate::components::{Camera3D, Transform};

// Movement below this is a click, not a drag (the marquee's convention).
const DRAG_START_PX: f32 = 4.0;

// Where a ray with no surface below lands: this far along the ray, so a drop
// aimed at the sky still places within reach.
const FREE_DROP_DISTANCE: f32 = 10.0;

// The ghost's half extents: placement fidelity comes from the committed
// entry's real geometry after the rebuild; the ghost marks the landing point.
const GHOST_HALF: [f32; 3] = [0.5, 0.5, 0.5];

pub(super) struct ContentDrag {
    pub name: String,
    pub asset_type: String,
    anchor: [f32; 2],
    moved: bool,
    // The landing pose while the cursor is over the viewport.
    pose: Option<DropPose>,
}

// A drop's landing pose: the surface point, the outward normal of the struck
// AABB face ([0,1,0] for the ground fallback), and the rotation the placed
// entry carries when align-to-surface is on (identity otherwise).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DropPose {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub rotation_deg: [f32; 3],
}

// The rotation carrying local +Y onto an axis-aligned face normal, in the
// engine's YXZ Euler convention ([pitch, yaw, roll] degrees).
pub(super) fn align_rotation(axis: usize, sign: f32) -> [f32; 3] {
    match (axis, sign > 0.0) {
        (0, true) => [0.0, 0.0, -90.0],
        (0, false) => [0.0, 0.0, 90.0],
        (1, true) => [0.0, 0.0, 0.0],
        (1, false) => [180.0, 0.0, 0.0],
        (2, true) => [90.0, 0.0, 0.0],
        _ => [-90.0, 0.0, 0.0],
    }
}

// The world entry a dropped asset creates at `pos`, `None` for a type that
// does not place (Material assigns instead; the rest only browse). Shared with
// the create menu's Prefab-instance rows.
pub(super) fn placement_args(
    asset_type: &str,
    name: &str,
    pos: [f32; 3],
) -> Option<serde_json::Value> {
    let field = match asset_type {
        "Mesh" | "ProceduralMesh" => "mesh",
        "Model" => "model",
        "Prefab" => "prefab",
        _ => return None,
    };
    Some(serde_json::json!({ field: name, "position": pos }))
}

// Whether dragging this type out of the browser does anything on release.
pub(super) fn drag_has_effect(asset_type: &str) -> bool {
    placement_args(asset_type, "", [0.0; 3]).is_some() || asset_type == "Material"
}

impl EditorHook {
    // Arm a drag from a grid cell press (the press also selected the cell).
    pub(super) fn arm_content_drag(&mut self, name: String, asset_type: String, at: [f32; 2]) {
        if !drag_has_effect(&asset_type) {
            return;
        }
        self.content_drag = Some(ContentDrag {
            name,
            asset_type,
            anchor: at,
            moved: false,
            pose: None,
        });
    }

    pub(super) fn drive_content_drag(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) {
        if input.escape {
            self.content_drag = None;
            return;
        }
        let mouse = [input.mouse_x, input.mouse_y];
        if input.left_button_down {
            let over_panel = self.cursor_over_content_panel(mouse, vp);
            let moved = match &mut self.content_drag {
                Some(drag) => {
                    let (dx, dy) = (mouse[0] - drag.anchor[0], mouse[1] - drag.anchor[1]);
                    drag.moved |= (dx * dx + dy * dy).sqrt() >= DRAG_START_PX;
                    drag.moved
                }
                None => return,
            };
            let pose = (moved && !over_panel)
                .then(|| self.drop_pose(world, vp, mouse, input.ctrl))
                .flatten();
            if let Some(drag) = &mut self.content_drag {
                drag.pose = pose;
            }
            return;
        }
        let Some(drag) = self.content_drag.take() else {
            return;
        };
        let Some(pose) = drag.pose else {
            return;
        };
        if drag.asset_type == "Material" {
            self.assign_material_under_cursor(world, vp, mouse, &drag.name);
            return;
        }
        let Some(mut args) = placement_args(
            &drag.asset_type,
            &drag.name,
            pose.position.map(gizmo_drag::round3),
        ) else {
            return;
        };
        if pose.rotation_deg != [0.0; 3]
            && let Some(obj) = args.as_object_mut()
        {
            obj.insert(
                "rotation_deg".to_string(),
                serde_json::json!(pose.rotation_deg),
            );
        }
        let name = self.unique_from(&drag.name);
        self.entries.push(serde_json::json!({
            "name": name, "type": "Prop", "args": args
        }));
        self.mark_changed();
        self.selection.replace(name);
    }

    // The landing position under the cursor. Shared with the create menu,
    // which captures it at open time (and never orients to the surface).
    pub(super) fn drop_point(
        &self,
        world: &World,
        vp: [f32; 2],
        mouse: [f32; 2],
        ctrl: bool,
    ) -> Option<[f32; 3]> {
        self.drop_pose(world, vp, mouse, ctrl).map(|p| p.position)
    }

    // The landing pose under the cursor: the nearest pick-index surface (its
    // entered face is the landing normal), else the ground plane (else a
    // fixed distance along the ray). Grid snapped per axis when move snapping
    // applies; the rotation follows the face only when align-to-surface is on.
    fn drop_pose(
        &self,
        world: &World,
        vp: [f32; 2],
        mouse: [f32; 2],
        ctrl: bool,
    ) -> Option<DropPose> {
        let ray = pick::camera_ray(world, vp, mouse)?;
        let hit = world
            .resource::<crate::ecs::PickIndex>()
            .into_iter()
            .flat_map(|index| index.entries.iter())
            .filter_map(|e| concinnity_core::gfx::pick::ray_aabb_face(&ray, e.bb_min, e.bb_max))
            .min_by(|a, b| a.t.total_cmp(&b.t));
        let t = hit.map(|f| f.t).unwrap_or_else(|| {
            // The ground plane y = 0, when the ray descends toward it.
            if ray.dir[1] < -1e-4 && ray.origin[1] > 0.0 {
                -ray.origin[1] / ray.dir[1]
            } else {
                FREE_DROP_DISTANCE
            }
        });
        let mut position = [
            ray.origin[0] + ray.dir[0] * t,
            ray.origin[1] + ray.dir[1] * t,
            ray.origin[2] + ray.dir[2] * t,
        ];
        if let Some(step) = self.snap.translate.active_step(ctrl) {
            position = position.map(|v| snap::snap_step(v, step));
        }
        let mut normal = [0.0, 1.0, 0.0];
        let mut rotation_deg = [0.0; 3];
        // A zero sign is a degenerate face (the ray started inside a box):
        // no landing normal to align to.
        if let Some(face) = hit.filter(|f| f.sign != 0.0) {
            normal = [0.0; 3];
            normal[face.axis] = face.sign;
            if self.align_to_surface {
                rotation_deg = align_rotation(face.axis, face.sign);
            }
        }
        Some(DropPose {
            position,
            normal,
            rotation_deg,
        })
    }

    // Assign the dragged Material to the Prop entry under the cursor.
    fn assign_material_under_cursor(
        &mut self,
        world: &mut World,
        vp: [f32; 2],
        mouse: [f32; 2],
        material: &str,
    ) {
        let Some(ray) = pick::camera_ray(world, vp, mouse) else {
            return;
        };
        let hit = self
            .ray_hit_names(world, &ray)
            .into_iter()
            .find_map(|name| {
                let idx = self
                    .entries
                    .iter()
                    .position(|e| entry_name(e) == Some(&name))?;
                (entry_type(&self.entries[idx]) == Some("Prop")).then_some(idx)
            });
        let Some(idx) = hit else {
            return;
        };
        if let Some(args) = self.entries[idx]
            .as_object_mut()
            .and_then(|o| o.get_mut("args"))
            .and_then(|a| a.as_object_mut())
        {
            args.insert(
                "material".to_string(),
                serde_json::Value::String(material.to_string()),
            );
            self.mark_changed();
        }
    }

    // The names of the pick-index entries the ray strikes, nearest first.
    fn ray_hit_names(
        &self,
        world: &World,
        ray: &concinnity_core::gfx::pick::PickRay,
    ) -> Vec<String> {
        let Some(index) = world.resource::<crate::ecs::PickIndex>() else {
            return Vec::new();
        };
        let mut hits: Vec<(f32, String)> = index
            .entries
            .iter()
            .filter_map(|e| {
                let t = concinnity_core::gfx::pick::ray_aabb(ray, e.bb_min, e.bb_max)?;
                Some((t, pick::resolve_name(e.asset_id)?))
            })
            .collect();
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits.into_iter().map(|(_, n)| n).collect()
    }

    fn cursor_over_content_panel(&self, mouse: [f32; 2], vp: [f32; 2]) -> bool {
        let o = self.origin(PanelKey::Content, vp);
        widget::point_in(
            mouse[0],
            mouse[1],
            super::super::content_panel::panel_rect(o),
        )
    }

    // Whether a ghost is showing this frame (the drag left the panel and has
    // a landing pose); the trigger outline stands down while it does.
    pub(super) fn content_ghost_pose(&self) -> Option<DropPose> {
        self.content_drag.as_ref().and_then(|d| d.pose)
    }

    // Draw the ghost through the shared outline pool.
    pub(super) fn drive_content_ghost(&self, world: &mut World, vp: [f32; 2]) {
        let Some(pose) = self.content_ghost_pose() else {
            return;
        };
        let Some(cam) = world.query::<Camera3D>().next() else {
            return;
        };
        let (view, fov) = (cam.view_matrix, cam.fov_y_degrees.to_radians());
        // Rest the ghost box on the landing point along its normal rather
        // than centering it, and orient it like the commit would.
        let transform = Transform {
            position: [
                pose.position[0] + pose.normal[0] * GHOST_HALF[0],
                pose.position[1] + pose.normal[1] * GHOST_HALF[1],
                pose.position[2] + pose.normal[2] * GHOST_HALF[2],
            ],
            rotation_deg: pose.rotation_deg,
            scale: [1.0; 3],
        };
        let tint = super::super::theme::ACCENT_TINT;
        if let Some(centers) =
            billboards::box_outline(&view, fov, vp, &transform.model_matrix(), GHOST_HALF)
        {
            billboards::place_box_outline(world, &centers, tint);
        }
    }
}
