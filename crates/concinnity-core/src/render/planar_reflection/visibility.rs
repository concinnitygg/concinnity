//! The part of each mirror render a frame actually samples.
//!
//! A reflector reads its mirror at its own screen UV (plus a small lookup
//! offset), and the mirror render shares the main camera's projection, so the
//! reflector's screen footprint is the only part of the render anyone reads. A
//! plane whose reflectors cover no pixels needs no render at all; a visible one
//! needs only the rectangle its reflectors cover.

use super::MAX_PLANAR_PLANES;
use super::mirror::mat_vec;
use crate::math::{ceil, floor};
use crate::transform::Mat4;

type Vec4 = [f32; 4];

// A clip-space point closer to the camera plane than this is treated as behind
// it. Anything just in front projects far off screen and is clamped away.
const MIN_CLIP_W: f32 = 1e-5;

// The corner pairs joined by an edge in a 4-corner quad (corners in winding
// order) and an 8-corner box (bit 0 = +x, bit 1 = +y, bit 2 = +z).
const QUAD_EDGES: &[(u8, u8)] = &[(0, 1), (1, 2), (2, 3), (3, 0)];
const BOX_EDGES: &[(u8, u8)] = &[
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// A conservative convex bound on everything a reflector rasterizes, plus how
/// far past its own pixels (in screen UV) it reads its mirror.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectorHull {
    corners: [[f32; 3]; 8],
    corner_count: usize,
    edges: &'static [(u8, u8)],
    lookup_margin: f32,
}

impl ReflectorHull {
    /// A flat quad: four world-space corners in winding order.
    pub fn quad(corners: [[f32; 3]; 4], lookup_margin: f32) -> Self {
        let mut all = [[0.0; 3]; 8];
        all[..4].copy_from_slice(&corners);
        Self {
            corners: all,
            corner_count: 4,
            edges: QUAD_EDGES,
            lookup_margin,
        }
    }

    /// An axis-aligned box around `center` with the given half extents.
    pub fn aabb(center: [f32; 3], half_extent: [f32; 3], lookup_margin: f32) -> Self {
        let mut corners = [[0.0; 3]; 8];
        for (i, c) in corners.iter_mut().enumerate() {
            for (axis, v) in c.iter_mut().enumerate() {
                let sign = if i & (1 << axis) != 0 { 1.0 } else { -1.0 };
                *v = center[axis] + sign * half_extent[axis];
            }
        }
        Self {
            corners,
            corner_count: 8,
            edges: BOX_EDGES,
            lookup_margin,
        }
    }

    /// The screen UV rectangle the reflector covers under `view_proj`, widened
    /// by its lookup margin and clamped to the screen. `None` when it is off
    /// screen or entirely behind the camera.
    pub fn uv_rect(&self, view_proj: Mat4) -> Option<UvRect> {
        let mut clip = [[0.0f32; 4]; 8];
        for (out, c) in clip.iter_mut().zip(&self.corners[..self.corner_count]) {
            *out = mat_vec(view_proj, [c[0], c[1], c[2], 1.0]);
        }
        let clip = &clip[..self.corner_count];

        // The hull clipped to the camera's front half-space is bounded by its
        // corners in front plus the points where its edges cross into it.
        let mut bound: Option<UvRect> = None;
        let mut include = |p: Vec4| {
            let uv = [p[0] / p[3] * 0.5 + 0.5, 0.5 - p[1] / p[3] * 0.5];
            let r = UvRect { min: uv, max: uv };
            bound = Some(bound.map_or(r, |b| b.union(r)));
        };
        for &p in clip {
            if p[3] > MIN_CLIP_W {
                include(p);
            }
        }
        for &(a, b) in self.edges {
            let (pa, pb) = (clip[a as usize], clip[b as usize]);
            if (pa[3] > MIN_CLIP_W) != (pb[3] > MIN_CLIP_W) {
                let t = (MIN_CLIP_W - pa[3]) / (pb[3] - pa[3]);
                include(core::array::from_fn(|i| pa[i] + (pb[i] - pa[i]) * t));
            }
        }

        let m = self.lookup_margin;
        let r = bound?;
        UvRect {
            min: [r.min[0] - m, r.min[1] - m],
            max: [r.max[0] + m, r.max[1] + m],
        }
        .clamped()
    }
}

/// An axis-aligned rectangle in screen UV, top-left origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvRect {
    /// Top-left corner.
    pub min: [f32; 2],
    /// Bottom-right corner.
    pub max: [f32; 2],
}

impl UvRect {
    /// The whole screen.
    pub const FULL: UvRect = UvRect {
        min: [0.0, 0.0],
        max: [1.0, 1.0],
    };

    /// The smallest rectangle covering both.
    pub fn union(self, other: UvRect) -> UvRect {
        UvRect {
            min: [self.min[0].min(other.min[0]), self.min[1].min(other.min[1])],
            max: [self.max[0].max(other.max[0]), self.max[1].max(other.max[1])],
        }
    }

    // Intersected with the screen; `None` when it misses the screen (or holds a
    // NaN). A zero-width rectangle on screen is kept, so a reflector seen
    // edge-on still renders the few texels its rasterized edge could touch.
    fn clamped(self) -> Option<UvRect> {
        let min = [self.min[0].max(0.0), self.min[1].max(0.0)];
        let max = [self.max[0].min(1.0), self.max[1].min(1.0)];
        (max[0] >= min[0] && max[1] >= min[1]).then_some(UvRect { min, max })
    }

    /// The texels of a `width` x `height` target this rectangle touches, grown
    /// by `margin` texels on every side for the filter footprint and clamped to
    /// the target. `None` for an empty target.
    pub fn to_pixels(self, width: u32, height: u32, margin: u32) -> Option<PixelRect> {
        let span = |lo: f32, hi: f32, size: u32| {
            let a = (floor(lo * size as f32) as i64 - i64::from(margin)).max(0);
            let b = (ceil(hi * size as f32) as i64 + i64::from(margin)).min(i64::from(size));
            (b > a).then_some((a as u32, (b - a) as u32))
        };
        let (x, w) = span(self.min[0], self.max[0], width)?;
        let (y, h) = span(self.min[1], self.max[1], height)?;
        Some(PixelRect {
            x,
            y,
            width: w,
            height: h,
        })
    }
}

/// A texel rectangle of a render target, top-left origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

impl PixelRect {
    /// `view_proj` narrowed to this rectangle of a `width` x `height` target:
    /// the rectangle's edges become the clip-space side planes, so a frustum
    /// extracted from the result rejects geometry that cannot reach the
    /// rectangle. Depth is untouched.
    pub fn crop_view_projection(self, view_proj: Mat4, width: u32, height: u32) -> Mat4 {
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        let left = 2.0 * self.x as f32 / w - 1.0;
        let right = 2.0 * (self.x + self.width) as f32 / w - 1.0;
        let top = 1.0 - 2.0 * self.y as f32 / h;
        let bottom = 1.0 - 2.0 * (self.y + self.height) as f32 / h;
        // Map [left, right] and [bottom, top] onto [-1, 1] in clip space.
        let sx = 2.0 / (right - left).max(f32::EPSILON);
        let ox = -(right + left) / (right - left).max(f32::EPSILON);
        let sy = 2.0 / (top - bottom).max(f32::EPSILON);
        let oy = -(top + bottom) / (top - bottom).max(f32::EPSILON);
        let mut out = view_proj;
        for col in out.iter_mut() {
            col[0] = sx * col[0] + ox * col[3];
            col[1] = sy * col[1] + oy * col[3];
        }
        out
    }
}

/// A reflector that holds a mirror slot, with the bound it is drawn within.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarReflector {
    /// The mirror slot it samples.
    pub slot: usize,
    /// Everything it can rasterize.
    pub hull: ReflectorHull,
}

/// This frame's mirror work: per slot, the screen rectangle its reflectors
/// cover, or nothing when none of them is on screen. A slot with no rectangle
/// renders no mirror, and its reflectors keep the probe fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlanarFramePlan {
    rects: [Option<UvRect>; MAX_PLANAR_PLANES],
}

impl PlanarFramePlan {
    /// Plan the frame seen through `view_proj`, the exact (jittered) matrix the
    /// reflectors are rasterized with.
    pub fn build(reflectors: &[PlanarReflector], view_proj: Mat4) -> Self {
        let mut rects = [None; MAX_PLANAR_PLANES];
        for r in reflectors {
            let (Some(slot), Some(rect)) = (rects.get_mut(r.slot), r.hull.uv_rect(view_proj))
            else {
                continue;
            };
            *slot = Some(slot.map_or(rect, |s: UvRect| s.union(rect)));
        }
        Self { rects }
    }

    /// The rectangle slot `slot` renders, `None` when it renders nothing.
    pub fn rect(&self, slot: usize) -> Option<UvRect> {
        self.rects.get(slot).copied().flatten()
    }

    /// Whether a reflector assigned `slot` samples a mirror rendered this frame.
    pub fn samples_mirror(&self, slot: Option<usize>) -> bool {
        slot.is_some_and(|s| self.rect(s).is_some())
    }

    /// Whether any slot renders this frame.
    pub fn any(&self) -> bool {
        self.rects.iter().any(Option::is_some)
    }

    /// The mirror renders this frame runs among the first `planes` slots: each
    /// kept slot with the texels of its `width` x `height` target it covers,
    /// grown by `margin` texels.
    pub fn crops(&self, planes: usize, width: u32, height: u32, margin: u32) -> PlanarCrops {
        let mut crops = PlanarCrops::default();
        for slot in 0..planes.min(MAX_PLANAR_PLANES) {
            let Some(px) = self
                .rect(slot)
                .and_then(|r| r.to_pixels(width, height, margin))
            else {
                continue;
            };
            if let Some(item) = crops.items.get_mut(crops.len) {
                *item = (slot, px);
                crops.len += 1;
            }
        }
        crops
    }
}

/// The mirror renders a frame runs, in slot order: each kept slot and the
/// texels of its target it renders. Fixed capacity, so planning a frame
/// allocates nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlanarCrops {
    items: [(usize, PixelRect); MAX_PLANAR_PLANES],
    len: usize,
}

impl PlanarCrops {
    /// The kept slots and their crops.
    pub fn as_slice(&self) -> &[(usize, PixelRect)] {
        &self.items[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::frustum::Frustum;
    use crate::gfx::projection::perspective_rh;
    use crate::transform::mat4_mul;
    use alloc::vec::Vec;

    // A camera at `eye` looking down -z (identity rotation).
    fn camera_at(eye: [f32; 3]) -> Mat4 {
        let view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-eye[0], -eye[1], -eye[2], 1.0],
        ];
        mat4_mul(perspective_rh(1.2, 1.0, 0.1, 100.0), view)
    }

    // A 2x2 pane facing +z at depth `z`, centered on x = `x`.
    fn pane(x: f32, z: f32) -> ReflectorHull {
        ReflectorHull::quad(
            [
                [x - 1.0, -1.0, z],
                [x + 1.0, -1.0, z],
                [x + 1.0, 1.0, z],
                [x - 1.0, 1.0, z],
            ],
            0.0,
        )
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn a_centered_pane_projects_to_a_centered_rect() {
        let r = pane(0.0, -5.0)
            .uv_rect(camera_at([0.0; 3]))
            .expect("visible");
        assert!(approx(r.min[0], 1.0 - r.max[0]), "{r:?}");
        assert!(approx(r.min[1], 1.0 - r.max[1]), "{r:?}");
        assert!(r.min[0] > 0.0 && r.max[0] < 1.0, "{r:?}");
    }

    #[test]
    fn up_in_the_world_is_up_on_screen() {
        // A pane entirely above the camera's eye line lands in the top half.
        let high = ReflectorHull::quad(
            [
                [-1.0, 1.0, -5.0],
                [1.0, 1.0, -5.0],
                [1.0, 2.0, -5.0],
                [-1.0, 2.0, -5.0],
            ],
            0.0,
        );
        let r = high.uv_rect(camera_at([0.0; 3])).expect("visible");
        assert!(r.max[1] < 0.5, "{r:?}");
    }

    #[test]
    fn a_pane_off_to_the_side_or_behind_covers_nothing() {
        let vp = camera_at([0.0; 3]);
        assert_eq!(pane(40.0, -5.0).uv_rect(vp), None, "far off to the right");
        assert_eq!(pane(0.0, 5.0).uv_rect(vp), None, "behind the camera");
    }

    #[test]
    fn a_hull_straddling_the_camera_is_clipped_not_dropped() {
        // A floor running from behind the camera to far ahead: its corners
        // behind the eye must not flip the bound, and it covers the bottom of
        // the screen out to both edges.
        let floor = ReflectorHull::quad(
            [
                [-50.0, -1.0, 10.0],
                [50.0, -1.0, 10.0],
                [50.0, -1.0, -50.0],
                [-50.0, -1.0, -50.0],
            ],
            0.0,
        );
        let r = floor.uv_rect(camera_at([0.0; 3])).expect("visible");
        assert_eq!((r.min[0], r.max[0]), (0.0, 1.0), "{r:?}");
        assert_eq!(r.max[1], 1.0, "{r:?}");
        assert!(r.min[1] > 0.5, "stays below the horizon: {r:?}");
    }

    #[test]
    fn an_edge_on_pane_keeps_a_sliver() {
        // A pane in the plane x = 0 seen from a camera on that plane rasterizes
        // to (at most) a line, which still gets a rectangle to render.
        let edge_on = ReflectorHull::quad(
            [
                [0.0, -1.0, -4.0],
                [0.0, -1.0, -6.0],
                [0.0, 1.0, -6.0],
                [0.0, 1.0, -4.0],
            ],
            0.0,
        );
        let r = edge_on.uv_rect(camera_at([0.0; 3])).expect("on screen");
        assert_eq!(r.min[0], r.max[0]);
        let px = r.to_pixels(100, 100, 2).expect("a sliver of texels");
        assert_eq!((px.x, px.width), (48, 4));
    }

    #[test]
    fn the_lookup_margin_widens_the_rect() {
        let vp = camera_at([0.0; 3]);
        let tight = pane(0.0, -5.0).uv_rect(vp).unwrap();
        let mut wide = pane(0.0, -5.0);
        wide.lookup_margin = 0.05;
        let wide = wide.uv_rect(vp).unwrap();
        assert!(approx(wide.min[0], tight.min[0] - 0.05));
        assert!(approx(wide.max[1], tight.max[1] + 0.05));
    }

    #[test]
    fn a_box_bounds_its_whole_volume() {
        // A thin slab whose top face alone would project lower on screen.
        let slab = ReflectorHull::aabb([0.0, -1.0, -6.0], [2.0, 0.5, 2.0], 0.0);
        let r = slab.uv_rect(camera_at([0.0; 3])).unwrap();
        let top = ReflectorHull::quad(
            [
                [-2.0, -0.5, -4.0],
                [2.0, -0.5, -4.0],
                [2.0, -0.5, -8.0],
                [-2.0, -0.5, -8.0],
            ],
            0.0,
        )
        .uv_rect(camera_at([0.0; 3]))
        .unwrap();
        assert!(r.max[1] > top.max[1], "{r:?} vs {top:?}");
    }

    #[test]
    fn pixels_round_outward_and_clamp() {
        let r = UvRect {
            min: [0.125, 0.0],
            max: [0.375, 0.999],
        };
        assert_eq!(
            r.to_pixels(100, 50, 2),
            Some(PixelRect {
                x: 10,
                y: 0,
                width: 30,
                height: 50
            })
        );
        assert_eq!(UvRect::FULL.to_pixels(0, 10, 2), None);
    }

    #[test]
    fn the_crop_keeps_what_the_rect_sees_and_rejects_the_rest() {
        let vp = camera_at([0.0; 3]);
        // The left half of the screen.
        let rect = PixelRect {
            x: 0,
            y: 0,
            width: 50,
            height: 100,
        };
        let frustum = Frustum::from_view_projection(rect.crop_view_projection(vp, 100, 100));
        let left = ([-3.0, -0.5, -10.5], [-2.0, 0.5, -9.5]);
        let right = ([2.0, -0.5, -10.5], [3.0, 0.5, -9.5]);
        assert!(frustum.intersects_aabb(left.0, left.1));
        assert!(!frustum.intersects_aabb(right.0, right.1));
        // The full screen crops to the matrix itself.
        let full = PixelRect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let same = full.crop_view_projection(vp, 100, 100);
        for c in 0..4 {
            for r in 0..4 {
                assert!(approx(same[c][r], vp[c][r]), "[{c}][{r}]");
            }
        }
    }

    #[test]
    fn a_cropped_point_lands_where_the_rect_maps_it() {
        // A point projecting to the rect's top-left corner maps to clip (-1, 1).
        let vp = camera_at([0.0; 3]);
        let rect = PixelRect {
            x: 25,
            y: 10,
            width: 50,
            height: 40,
        };
        let cropped = rect.crop_view_projection(vp, 100, 100);
        // Find the world point at uv (0.25, 0.1) on the z = -5 plane.
        let inv = crate::transform::mat4_inverse(vp);
        let ndc = [-0.5, 0.8];
        let near = mat_vec(inv, [ndc[0], ndc[1], 0.5, 1.0]);
        let p = [near[0] / near[3], near[1] / near[3], near[2] / near[3], 1.0];
        let c = mat_vec(cropped, p);
        assert!(approx(c[0] / c[3], -1.0), "{c:?}");
        assert!(approx(c[1] / c[3], 1.0), "{c:?}");
    }

    #[test]
    fn the_plan_unions_a_slot_and_skips_unseen_ones() {
        let vp = camera_at([0.0; 3]);
        let reflectors = [
            PlanarReflector {
                slot: 0,
                hull: pane(-2.0, -5.0),
            },
            PlanarReflector {
                slot: 0,
                hull: pane(2.0, -5.0),
            },
            PlanarReflector {
                slot: 1,
                hull: pane(0.0, 5.0),
            },
        ];
        let plan = PlanarFramePlan::build(&reflectors, vp);
        let a = pane(-2.0, -5.0).uv_rect(vp).unwrap();
        let b = pane(2.0, -5.0).uv_rect(vp).unwrap();
        assert_eq!(plan.rect(0), Some(a.union(b)));
        assert_eq!(plan.rect(1), None, "only reflector is behind the camera");
        assert!(plan.samples_mirror(Some(0)));
        assert!(!plan.samples_mirror(Some(1)));
        assert!(!plan.samples_mirror(None));
        assert!(plan.any());
        assert!(!PlanarFramePlan::build(&reflectors[2..], vp).any());
    }

    #[test]
    fn crops_list_the_rendered_slots_in_order() {
        let vp = camera_at([0.0; 3]);
        let reflectors = [
            PlanarReflector {
                slot: 2,
                hull: pane(2.0, -5.0),
            },
            PlanarReflector {
                slot: 1,
                hull: pane(0.0, 5.0),
            },
            PlanarReflector {
                slot: 0,
                hull: pane(-2.0, -5.0),
            },
        ];
        let plan = PlanarFramePlan::build(&reflectors, vp);
        let crops = plan.crops(3, 100, 100, 2);
        let slots: Vec<usize> = crops.as_slice().iter().map(|&(s, _)| s).collect();
        assert_eq!(slots, [0, 2], "slot 1 is behind the camera");
        let expected = plan.rect(2).unwrap().to_pixels(100, 100, 2).unwrap();
        assert_eq!(crops.as_slice()[1].1, expected);
        // Only the planes the world has are considered.
        assert_eq!(plan.crops(1, 100, 100, 2).as_slice().len(), 1);
        // An empty target renders nothing.
        assert!(plan.crops(3, 0, 100, 2).as_slice().is_empty());
    }

    #[test]
    fn a_slot_past_capacity_is_ignored() {
        let reflectors = [PlanarReflector {
            slot: MAX_PLANAR_PLANES,
            hull: pane(0.0, -5.0),
        }];
        let plan = PlanarFramePlan::build(&reflectors, camera_at([0.0; 3]));
        assert!(!plan.any());
    }
}
