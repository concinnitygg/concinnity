//! Where the shadow casters that moved since the cascades were last scheduled
//! went: a few boxes kept as they are and the rest merged, so recording a
//! frame's moves allocates nothing however many casters move.

use crate::environment::planet::Rebase;
use crate::gfx::frustum::{Aabb, transform_aabb};
use crate::render::shadow::csm::LightBasis;
use concinnity_core::math::vec3::{dot, scale, sub};

// How many moved boxes are kept apart before the rest merge into one.
const KEPT_BOXES: usize = 16;

/// The world-space boxes casters moved through, in the frame the shadow plan
/// is in.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MovedCasters {
    boxes: [Aabb; KEPT_BOXES],
    len: usize,
    merged: Option<Aabb>,
    // The greatest height of any caster merged into `merged`.
    merged_height: f32,
    anywhere: bool,
}

impl MovedCasters {
    /// Note a caster that moved through `bounds`, or anywhere for `None`.
    pub(crate) fn push(&mut self, bounds: Option<Aabb>) {
        let Some(b) = bounds else {
            self.anywhere = true;
            return;
        };
        if let Some(slot) = self.boxes.get_mut(self.len) {
            *slot = b;
            self.len += 1;
        } else {
            self.merged = Some(self.merged.map_or(b, |m| union(m, b)));
            self.merged_height = self.merged_height.max(height(b));
        }
    }

    /// Whether a caster moved anywhere, with no box to say where.
    pub(crate) fn anywhere(&self) -> bool {
        self.anywhere
    }

    /// The boxes casters moved through, each with the height of the tallest
    /// caster in it: a merged box spans many casters but none taller than
    /// that.
    pub(crate) fn boxes(&self) -> impl Iterator<Item = (Aabb, f32)> + '_ {
        let kept = self.boxes[..self.len].iter().map(|b| (*b, height(*b)));
        kept.chain(self.merged.map(|b| (b, self.merged_height)))
    }

    /// Carry every box into the frame `rebase` moves the world to.
    pub(crate) fn rebase(&mut self, rebase: &Rebase) {
        let carry = |(lo, hi): Aabb| transform_aabb(lo, hi, rebase.matrix());
        for b in &mut self.boxes[..self.len] {
            *b = carry(*b);
        }
        self.merged = self.merged.map(carry);
    }

    /// Forget every move.
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

// A box's height along the frame's up, +Y.
fn height((lo, hi): Aabb) -> f32 {
    hi[1] - lo[1]
}

// The least sine of the light's elevation a shadow's length is measured at:
// a light lower than this lays shadows twenty times a caster's height long.
const MIN_ELEVATION_SINE: f32 = 0.05;

/// The space a moved caster's shadow can fall in: its box swept away from the
/// light by as far as the shadow of something its height standing on the
/// ground reaches.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShadowSweep {
    /// The swept box's corners: the box's, then the same moved away from the
    /// light.
    pub corners: [[f32; 3]; 16],
    /// The light's right, up and toward-the-light axes.
    pub axes: [[f32; 3]; 3],
    /// The sweep's range along each of `axes`.
    pub ranges: [[f32; 2]; 3],
}

impl ShadowSweep {
    /// The sweep of the box `bounds` holding casters up to `height` tall, lit
    /// along the unit vector `to_light`, with the frame's +Y as the ground's
    /// up.
    pub(crate) fn new((lo, hi): Aabb, height: f32, to_light: [f32; 3]) -> Self {
        let length = height.max(0.0) / to_light[1].max(MIN_ELEVATION_SINE);
        let corner = |i: usize| -> [f32; 3] {
            core::array::from_fn(|a| if i & (1 << a) == 0 { lo[a] } else { hi[a] })
        };
        let corners = core::array::from_fn(|i| {
            let c = corner(i % 8);
            if i < 8 {
                c
            } else {
                sub(c, scale(to_light, length))
            }
        });
        let basis = LightBasis::toward(to_light);
        let axes = [basis.right, basis.up, to_light];
        let ranges = axes.map(|axis| span(&corners, axis));
        Self {
            corners,
            axes,
            ranges,
        }
    }
}

/// The range of `dot(axis, p)` over `points`.
pub(crate) fn span(points: &[[f32; 3]], axis: [f32; 3]) -> [f32; 2] {
    points
        .iter()
        .fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], p| {
            let d = dot(axis, *p);
            [lo.min(d), hi.max(d)]
        })
}

fn union((a_lo, a_hi): Aabb, (b_lo, b_hi): Aabb) -> Aabb {
    (
        core::array::from_fn(|i| a_lo[i].min(b_lo[i])),
        core::array::from_fn(|i| a_hi[i].max(b_hi[i])),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::frustum::Frustum;
    use crate::render::depth::shadow_ortho;

    // An orthographic box over [-1, 1] on every axis.
    fn unit_box() -> Frustum {
        Frustum::from_shadow(shadow_ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    }

    fn at(x: f32) -> Aabb {
        ([x, 0.0, 0.0], [x + 0.1, 0.1, 0.1])
    }

    fn reaches(moved: &MovedCasters) -> bool {
        moved.anywhere()
            || moved
                .boxes()
                .any(|((lo, hi), _)| unit_box().intersects_aabb(lo, hi))
    }

    #[test]
    fn moves_are_kept_until_cleared() {
        let mut moved = MovedCasters::default();
        assert_eq!(moved.boxes().count(), 0);
        moved.push(Some(at(50.0)));
        assert!(!reaches(&moved));
        moved.push(Some(at(0.2)));
        assert!(reaches(&moved));
        moved.clear();
        assert_eq!(moved.boxes().count(), 0);
        moved.push(None);
        assert!(moved.anywhere(), "a move with no box is anywhere");
    }

    // Past the boxes kept apart the rest merge, and a move among them still
    // reaches the frustum it lies in.
    #[test]
    fn moves_past_the_kept_boxes_merge() {
        let mut moved = MovedCasters::default();
        for _ in 0..KEPT_BOXES {
            moved.push(Some(at(50.0)));
        }
        moved.push(Some(at(0.2)));
        moved.push(Some(at(60.0)));
        assert_eq!(moved.boxes().count(), KEPT_BOXES + 1);
        assert!(reaches(&moved));
    }

    // A merged box sweeps by its tallest caster, not by its own span: casters
    // spread over a kilometer of flat ground still sweep a short way.
    #[test]
    fn a_merged_box_keeps_its_tallest_caster_s_height() {
        let mut moved = MovedCasters::default();
        for i in 0..(KEPT_BOXES + 4) {
            let x = i as f32 * 60.0;
            moved.push(Some((
                [x, 0.0, 0.0],
                [x + 1.0, if i == KEPT_BOXES + 2 { 3.0 } else { 1.0 }, 1.0],
            )));
        }
        let (merged, height) = moved.boxes().last().unwrap();
        assert!(merged.1[0] - merged.0[0] > 150.0);
        assert_eq!(height, 3.0);
    }

    #[test]
    fn a_rebase_carries_every_box() {
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 1.0, 0.0], 0.0),
            translation: [-50.0, 0.0, 0.0],
        };
        let mut moved = MovedCasters::default();
        moved.push(Some(at(50.0)));
        assert!(!reaches(&moved));
        moved.rebase(&rebase);
        assert!(reaches(&moved));
    }

    // The shadow of a meter-tall box reaches a meter past it with the light
    // overhead and twenty under a light at the horizon, away from the light.
    #[test]
    fn a_sweep_reaches_as_far_as_the_caster_s_shadow() {
        let bounds = ([0.0; 3], [1.0; 3]);
        let overhead = ShadowSweep::new(bounds, 1.0, [0.0, 1.0, 0.0]);
        assert_eq!(span(&overhead.corners, [0.0, 1.0, 0.0]), [-1.0, 1.0]);
        let low = ShadowSweep::new(bounds, 1.0, [1.0, 0.0, 0.0]);
        assert_eq!(span(&low.corners, [1.0, 0.0, 0.0]), [-20.0, 1.0]);
        let toward = span(&low.corners, low.axes[2]);
        assert_eq!(toward, low.ranges[2]);
    }
}
