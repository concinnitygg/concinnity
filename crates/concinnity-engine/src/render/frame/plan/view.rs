//! The view's facts for each planned frame, and the depth samples that
//! measure them.

use crate::environment::planet::Rebase;
use crate::gfx::frustum::{Aabb, Frustum};
use crate::gfx::projection::focal_px;
use crate::gfx::render_types::{ViewSampleCell, ViewSampleParams};
use crate::render::view::{
    MAX_MEASURED_AGE, MeasureView, MoverView, ViewFacts, ViewMeasure, ViewMovers, ViewSubjects,
};
use crate::transform::{Mat4, basis_turn, view_basis};
use concinnity_core::math::vec3::{length, sub};

/// The frame a view's facts are taken for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewInputs {
    /// World-to-view matrix, column-major.
    pub view: Mat4,
    /// The camera's position.
    pub eye: [f32; 3],
    /// What the camera sees.
    pub frustum: Frustum,
    /// The view-projection the camera sees through, without jitter.
    pub vp: Mat4,
    /// Vertical field of view, in radians.
    pub fov_y: f32,
    /// The inverse of the view-projection the depth is drawn through.
    pub depth_inv_vp: Mat4,
    /// The size the scene renders at.
    pub render_size: [u32; 2],
    /// The subjects in the frame.
    pub subjects: ViewSubjects,
    /// How bright the auto-exposure shows the view, `None` without it.
    pub shown_light: Option<f32>,
}

// How many frames' motion is kept: as many as the oldest measurement that
// still stands has drifted over, and more.
const RECENT_FRAMES: usize = MAX_MEASURED_AGE as usize + 2;

// Where the camera stood and looked in the last frame.
#[derive(Clone, Copy, Debug)]
struct LastView {
    eye: [f32; 3],
    basis: [[f32; 3]; 3],
}

/// What the camera sees, carried from frame to frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewPlan {
    measure: ViewMeasure,
    movers: ViewMovers,
    last: Option<LastView>,
    // Each recent frame's motion, the newest at `frame % RECENT_FRAMES`.
    recent: [f32; RECENT_FRAMES],
    frame: usize,
    /// This frame's depth sampling, `None` when it is not sampled.
    pub samples: Option<ViewSampleParams>,
    facts: ViewFacts,
}

impl ViewPlan {
    /// The facts of the frame planned last.
    pub fn facts(&self) -> &ViewFacts {
        &self.facts
    }

    /// Take depth samples the GPU finished, which stand for what the camera
    /// sees in the frames after them.
    pub fn measured(&mut self, cells: &[ViewSampleCell]) {
        self.measure.accept(cells);
    }

    /// Forget what the camera saw, for a view that does not continue the
    /// last.
    pub fn reset(&mut self) {
        self.measure.reset();
        self.movers.clear();
        self.last = None;
        self.recent = [0.0; RECENT_FRAMES];
    }

    /// Note a thing in the world that moved from `before` to `after` since
    /// the last frame; `None` for bounds it does not have.
    pub fn moved(&mut self, before: Option<Aabb>, after: Option<Aabb>) {
        self.movers.moved(before, after);
    }

    /// Note a thing in the world that changed shape within `bounds` since the
    /// last frame.
    pub fn reshaped(&mut self, bounds: Aabb) {
        self.movers.reshaped(bounds);
    }

    /// Carry where the camera stood, and what moved, into the frame the world
    /// moved to.
    pub fn rebase(&mut self, rebase: &Rebase) {
        if let Some(last) = &mut self.last {
            last.eye = rebase.apply_point(last.eye);
            last.basis = last.basis.map(|a| rebase.apply_vector(a));
        }
        self.movers.rebase(rebase);
    }

    /// Take the facts of `frame`.
    pub(crate) fn refresh(&mut self, frame: &ViewInputs) {
        let focal_px = focal_px(frame.render_size[1] as f32, frame.fov_y);
        let mut subjects = frame
            .subjects
            .subjects(frame.eye, &frame.frustum, &frame.vp);
        if let Some((_, measured)) = self.measure.latest() {
            for (subject, seen) in subjects.iter_mut().zip(measured.subjects) {
                if let Some(s) = subject {
                    s.seen = seen;
                }
            }
        }
        let planet = frame.subjects.planet.map(|p| p.ground);
        let mut facts = ViewFacts {
            focal_px,
            render_size: frame.render_size,
            ground_height: planet.map(|g| length(sub(frame.eye, g.center)) - g.radius),
            sea_height: frame.subjects.sea_height(),
            motion_px: 0.0,
            parallax_px: 0.0,
            mover_px: 0.0,
            shown_light: frame.shown_light,
            seen: self.measure.seen(),
            subjects,
        };
        let basis = view_basis(&frame.view);
        if let Some(last) = self.last {
            let turn = basis_turn(&last.basis, &basis) * focal_px;
            let travel = sub(frame.eye, last.eye);
            let shift = facts
                .nearest_still()
                .map_or(0.0, |n| facts.pixels(length(travel), n));
            facts.motion_px = turn + shift;
            facts.parallax_px = shift;
            facts.mover_px = self.movers.fastest(&MoverView {
                frustum: &frame.frustum,
                eye: frame.eye,
                travel,
                focal_px,
            });
        }
        self.movers.clear();
        self.frame = self.frame.wrapping_add(1);
        self.recent[self.frame % RECENT_FRAMES] = facts.motion_px;
        if let Some(seen) = &mut facts.seen {
            seen.drift_px = self.drift(seen.age);
        }
        self.facts = facts;
        self.last = Some(LastView {
            eye: frame.eye,
            basis,
        });
    }

    // How far the view moved over the last `frames` frames, this one
    // included; every kept frame's when older.
    fn drift(&self, frames: u32) -> f32 {
        let frames = (frames as usize).min(RECENT_FRAMES);
        (0..frames)
            .map(|back| self.recent[self.frame.wrapping_sub(back) % RECENT_FRAMES])
            .sum()
    }

    /// Stamp the depth samples of `frame`, which the GPU takes when
    /// `sampled`.
    pub(crate) fn stamp(&mut self, frame: &ViewInputs, sampled: bool) {
        let params = self.measure.begin(&MeasureView {
            depth_inv_vp: frame.depth_inv_vp,
            eye: frame.eye,
            size: frame.render_size,
            subjects: frame.subjects,
        });
        self.samples = sampled.then_some(params);
    }
}

#[cfg(test)]
mod tests;
