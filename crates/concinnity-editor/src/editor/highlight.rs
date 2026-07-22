// src/editor/highlight.rs
//
// The viewport selection highlight: a pool of border-only sprites (transparent
// fill, accent ring), one placed over each selected asset's projected world
// AABB every frame. The active (most recently picked) member's ring draws in
// the full accent; the rest a dimmed step. Window-space like the rest of the
// editor HUD, injected hidden alongside it, and drawn under the floating
// panels (the ids are absent from the HudLayers map, so they sit at layer 0
// below every focused panel).

use super::registry::ID_BASE;
use super::theme;
use crate::assets::Sprite;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id family: the next free block after the Health panel's 0xB00.
const OUTLINE_BASE: u32 = ID_BASE + 0xC00;
// Ring pool size, bounding the per-frame sprite cost; a selection larger than
// this keeps every member (gizmo and commit included) but only rings the
// first `MAX_RINGS`.
pub(crate) const MAX_RINGS: usize = 32;

const BORDER_W: f32 = 2.0;
// Breathing room so the ring does not sit flush on the object's silhouette.
const PAD_PX: f32 = 3.0;

// Non-active members ring in a dimmed accent so the active one reads at a
// glance.
const MEMBER_TINT: [f32; 4] = [0.20, 0.30, 0.45, 1.0];

fn ring_id(i: usize) -> AssetId {
    AssetId(OUTLINE_BASE + i as u32)
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    (0..MAX_RINGS).map(ring_id).collect()
}

// The injected pool: invisible until the tick places them, transparent fill,
// accent border rings.
pub(crate) fn outline_sprites() -> Vec<Sprite> {
    all_sprite_ids()
        .into_iter()
        .map(|id| Sprite {
            asset_id: id,
            tint: [0.0, 0.0, 0.0, 0.0],
            border_width: BORDER_W,
            border_color: theme::ACCENT_TINT,
            visible: false,
            ..Default::default()
        })
        .collect()
}

// Project a world AABB to its enclosing screen rect `[x, y, w, h]` for the
// `view_matrix`-convention camera, padded and clamped to the viewport. `None`
// when there is nothing to draw: a non-finite box, a degenerate viewport, any
// corner at or behind the camera plane (only possible with the camera inside
// or against the box), or a projection entirely off screen.
pub(crate) fn screen_rect(
    view: &[[f32; 4]; 4],
    fov_y_radians: f32,
    viewport: [f32; 2],
    bb_min: [f32; 3],
    bb_max: [f32; 3],
) -> Option<[f32; 4]> {
    let (vw, vh) = (viewport[0], viewport[1]);
    if vw <= 0.0 || vh <= 0.0 {
        return None;
    }
    for i in 0..3 {
        if !bb_min[i].is_finite() || !bb_max[i].is_finite() {
            return None;
        }
    }
    let tan_half = (fov_y_radians * 0.5).tan();
    if tan_half <= 0.0 || !tan_half.is_finite() {
        return None;
    }
    let aspect = vw / vh;

    let (mut x0, mut y0) = (f32::INFINITY, f32::INFINITY);
    let (mut x1, mut y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for corner in 0..8 {
        let c = [
            if corner & 1 == 0 {
                bb_min[0]
            } else {
                bb_max[0]
            },
            if corner & 2 == 0 {
                bb_min[1]
            } else {
                bb_max[1]
            },
            if corner & 4 == 0 {
                bb_min[2]
            } else {
                bb_max[2]
            },
        ];
        // World to view (column-major view[c][r]); view-space forward is -Z.
        let v = [
            view[0][0] * c[0] + view[1][0] * c[1] + view[2][0] * c[2] + view[3][0],
            view[0][1] * c[0] + view[1][1] * c[1] + view[2][1] * c[2] + view[3][1],
            view[0][2] * c[0] + view[1][2] * c[1] + view[2][2] * c[2] + view[3][2],
        ];
        let depth = -v[2];
        if depth <= 1e-4 {
            return None;
        }
        let px = (v[0] / (depth * tan_half * aspect) + 1.0) * 0.5 * vw;
        let py = (1.0 - v[1] / (depth * tan_half)) * 0.5 * vh;
        x0 = x0.min(px);
        y0 = y0.min(py);
        x1 = x1.max(px);
        y1 = y1.max(py);
    }
    // Pad, then clamp to the viewport so a large or edge-crossing box shows a
    // partial ring instead of geometry far off screen.
    let (x0, y0) = ((x0 - PAD_PX).max(0.0), (y0 - PAD_PX).max(0.0));
    let (x1, y1) = ((x1 + PAD_PX).min(vw), (y1 + PAD_PX).min(vh));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some([x0, y0, x1 - x0, y1 - y0])
}

// Show one ring per `(rect, is_active)` entry (rects past the pool are
// dropped) and hide the rest of the pool.
pub(crate) fn place_all(world: &mut World, rects: &[([f32; 4], bool)]) {
    for (i, id) in all_sprite_ids().into_iter().enumerate() {
        let Some(s) = sprite_mut(world, id) else {
            continue;
        };
        match rects.get(i) {
            Some((rect, active)) => {
                s.x = rect[0];
                s.y = rect[1];
                s.width = rect[2];
                s.height = rect[3];
                s.border_color = if *active {
                    theme::ACCENT_TINT
                } else {
                    MEMBER_TINT
                };
                s.visible = true;
            }
            None => s.visible = false,
        }
    }
}

pub(crate) fn hide(world: &mut World) {
    place_all(world, &[]);
}

fn sprite_mut(world: &mut World, id: AssetId) -> Option<&mut Sprite> {
    world.query_mut::<Sprite>().find(|s| s.asset_id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::camera::view_matrix;

    const VP: [f32; 2] = [1280.0, 720.0];
    const FOV: f32 = core::f32::consts::FRAC_PI_2;

    #[test]
    fn centered_box_projects_to_a_centered_rect() {
        // Camera at origin facing -Z; a unit box straight ahead at depth 5.
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        let r = screen_rect(&view, FOV, VP, [-1.0, -1.0, -6.0], [1.0, 1.0, -4.0]).unwrap();
        let (cx, cy) = (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5);
        assert!((cx - 640.0).abs() < 1.0, "{cx}");
        assert!((cy - 360.0).abs() < 1.0, "{cy}");
        // The near face (depth 4) bounds the projection: half-extent 1 at
        // depth 4 with tan_half 1 spans 1/4 of the half-height -> 90 px, plus
        // the pad.
        assert!((r[3] * 0.5 - (90.0 + PAD_PX)).abs() < 1.0, "{}", r[3]);
    }

    #[test]
    fn offscreen_and_behind_boxes_yield_no_rect() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        // Entirely behind the camera.
        assert_eq!(
            screen_rect(&view, FOV, VP, [-1.0, -1.0, 4.0], [1.0, 1.0, 6.0]),
            None
        );
        // In front but projected far outside the viewport.
        assert_eq!(
            screen_rect(&view, FOV, VP, [50.0, -1.0, -6.0], [52.0, 1.0, -4.0]),
            None
        );
        // The renderer's non-finite sentinel.
        assert_eq!(
            screen_rect(&view, FOV, VP, [f32::NAN; 3], [f32::NAN; 3]),
            None
        );
        // Degenerate viewport.
        assert_eq!(
            screen_rect(&view, FOV, [0.0, 720.0], [-1.0; 3], [1.0; 3]),
            None
        );
    }

    #[test]
    fn edge_crossing_box_clamps_to_the_viewport() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        // A wide box crossing the left edge: the rect starts at 0, not below.
        let r = screen_rect(&view, FOV, VP, [-50.0, -1.0, -6.0], [0.0, 1.0, -4.0]).unwrap();
        assert_eq!(r[0], 0.0);
        assert!(r[2] <= VP[0]);
    }
}
