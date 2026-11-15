//! The things that moved in the world since the last frame, kept until the
//! next frame's view measures how far they moved against the camera. A few
//! movers are kept with their travel; the rest, and changes of shape, merge
//! into one box whose motion is not known. Recording allocates nothing however
//! many things move.

use crate::environment::planet::Rebase;
use crate::gfx::frustum::{Aabb, Frustum, transform_aabb};
use concinnity_core::math::vec3::{add, length, scale, sub};

// How many movers keep their travel before the rest merge.
const KEPT: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
struct Mover {
    // Where it is now.
    bounds: Aabb,
    // Where it was and where it is: what it uncovered or covered.
    swept: Aabb,
    // How far its center traveled to get there.
    travel: [f32; 3],
}

/// What moved since the last frame, in the frame the view plan is in.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ViewMovers {
    kept: [Mover; KEPT],
    len: usize,
    // Movers whose travel is not known: the overflow, and changes of shape.
    unplaced: Option<Aabb>,
    // Something moved with no box to say where.
    anywhere: bool,
}

/// The camera a frame's movers are measured against.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoverView<'a> {
    /// What the camera sees.
    pub frustum: &'a Frustum,
    /// The camera's position.
    pub eye: [f32; 3],
    /// How far the camera traveled since the last frame.
    pub travel: [f32; 3],
    /// Pixels one radian spans at the center of the view.
    pub focal_px: f32,
}

impl ViewMovers {
    /// Note a thing that moved from `before` to `after`; `None` for either
    /// when it has no bounds.
    pub(crate) fn moved(&mut self, before: Option<Aabb>, after: Option<Aabb>) {
        let (Some(before), Some(after)) = (before, after) else {
            self.anywhere = true;
            return;
        };
        let travel = sub(center(after), center(before));
        if let Some(slot) = self.kept.get_mut(self.len) {
            *slot = Mover {
                bounds: after,
                swept: union(before, after),
                travel,
            };
            self.len += 1;
        } else {
            self.unplace(union(before, after));
        }
    }

    /// Note a thing that changed shape within `bounds` without moving.
    pub(crate) fn reshaped(&mut self, bounds: Aabb) {
        self.unplace(bounds);
    }

    /// Carry everything noted into the frame `rebase` moves the world to.
    pub(crate) fn rebase(&mut self, rebase: &Rebase) {
        let carry = |(lo, hi): Aabb| transform_aabb(lo, hi, rebase.matrix());
        for m in &mut self.kept[..self.len] {
            m.bounds = carry(m.bounds);
            m.swept = carry(m.swept);
            m.travel = rebase.apply_vector(m.travel);
        }
        self.unplaced = self.unplaced.map(carry);
    }

    /// Forget everything noted.
    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.unplaced = None;
        self.anywhere = false;
    }

    /// How far, in pixels at the view's center, the fastest mover `view`
    /// sees moved against its camera: 0 when none was or is in view, infinite
    /// when one in view moved in a way that cannot be placed. A mover counts
    /// where it was as well as where it is, since leaving the view uncovers
    /// what it hid.
    pub(crate) fn fastest(&self, view: &MoverView<'_>) -> f32 {
        let seen = |(lo, hi): Aabb| view.frustum.intersects_aabb(lo, hi);
        if self.anywhere || self.unplaced.is_some_and(seen) {
            return f32::INFINITY;
        }
        self.kept[..self.len]
            .iter()
            .filter(|m| seen(m.swept))
            .map(|m| {
                let (lo, hi) = m.bounds;
                let reach = length(sub(center(m.bounds), view.eye)).max(0.5 * length(sub(hi, lo)));
                length(sub(m.travel, view.travel)) * view.focal_px / reach.max(f32::MIN_POSITIVE)
            })
            .fold(0.0, f32::max)
    }

    fn unplace(&mut self, bounds: Aabb) {
        self.unplaced = Some(self.unplaced.map_or(bounds, |u| union(u, bounds)));
    }
}

fn center((lo, hi): Aabb) -> [f32; 3] {
    scale(add(lo, hi), 0.5)
}

fn union(a: Aabb, b: Aabb) -> Aabb {
    (
        core::array::from_fn(|k| a.0[k].min(b.0[k])),
        core::array::from_fn(|k| a.1[k].max(b.1[k])),
    )
}
