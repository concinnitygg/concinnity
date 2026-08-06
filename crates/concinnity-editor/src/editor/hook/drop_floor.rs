// src/editor/hook/drop_floor.rs
//
// EditorHook: drop the selection to the floor (Ctrl+Down, /floor, Preview
// panel row). Each eligible member casts a ray straight down from its bounds'
// bottom-center (its position, for types without indexed bounds) against the
// pick index -- excluding the selection itself, so stacked members never rest
// on each other mid-drop -- and lands its bottom on the nearest surface, or on
// the world ground plane y=0 when nothing is below. The whole batch commits as
// ONE undo step.

use super::*;
use crate::assets::Transform;
use crate::ecs::PickIndex;
use concinnity_core::gfx::pick::{PickRay, ray_aabb};
use gizmo::GizmoMode;

// A member already resting within this distance of the floor is left alone
// (and a no-op drop records no undo step).
const REST_EPSILON: f32 = 1e-3;

impl EditorHook {
    // Drop every eligible selection member; the number of members moved.
    // Eligibility matches the translate gizmo: an authored entry with a
    // `position` arg, a live unparented Transform.
    pub(super) fn drop_selection_to_floor(&mut self, world: &mut World) -> usize {
        let names: Vec<String> = self.selection.iter().map(String::from).collect();
        let selected_ids: std::collections::BTreeSet<AssetId> = names
            .iter()
            .filter_map(|n| crate::ecs::asset_id::lookup(n))
            .collect();
        let mut changed = Vec::new();
        for name in &names {
            let Some(target) = self.member_target(world, GizmoMode::Translate, name) else {
                continue;
            };
            let Some(position) = world.get::<Transform>(target.entity).map(|t| t.position) else {
                continue;
            };
            let id = crate::ecs::asset_id::lookup(name);
            let bounds = id.and_then(|id| {
                world
                    .resource::<PickIndex>()?
                    .entries
                    .iter()
                    .find(|e| e.asset_id == id)
                    .map(|e| (e.bb_min, e.bb_max))
            });
            // Bounds when indexed (props), the position itself otherwise
            // (lights, trigger volumes, and other billboard-only types rest
            // their origin on the floor).
            let foot = match bounds {
                Some((mn, mx)) => [(mn[0] + mx[0]) * 0.5, mn[1], (mn[2] + mx[2]) * 0.5],
                None => position,
            };
            let floor_y = nearest_floor(world, foot, &selected_ids).unwrap_or(0.0);
            let delta = floor_y - foot[1];
            if delta.abs() <= REST_EPSILON {
                continue;
            }
            let landed = [position[0], position[1] + delta, position[2]];
            if let Some(tr) = world.get_mut::<Transform>(target.entity) {
                tr.position = landed;
            }
            Self::write_arg(
                &mut self.entries,
                target.idx,
                "position",
                landed.map(gizmo_drag::round3),
            );
            changed.push(target.idx);
        }
        if changed.is_empty() {
            return 0;
        }
        self.mark_changed();
        changed.len()
    }
}

// The y of the nearest pick-index surface straight below `from`, skipping the
// excluded ids (the selection) and boxes whose top is above the start point
// (a surface the foot is inside or under cannot catch it).
fn nearest_floor(
    world: &World,
    from: [f32; 3],
    exclude: &std::collections::BTreeSet<AssetId>,
) -> Option<f32> {
    let index = world.resource::<PickIndex>()?;
    let ray = PickRay {
        origin: from,
        dir: [0.0, -1.0, 0.0],
    };
    index
        .entries
        .iter()
        .filter(|e| !exclude.contains(&e.asset_id))
        .filter(|e| e.bb_max[1] <= from[1])
        .filter_map(|e| ray_aabb(&ray, e.bb_min, e.bb_max))
        .map(|t| from[1] - t)
        .max_by(f32::total_cmp)
}
