//! The view's facts for each planned frame, and the depth samples that
//! measure them.

use crate::environment::planet::Rebase;
use crate::gfx::color::luminance;
use crate::gfx::render_types::{ViewSampleCell, ViewSampleParams};
use crate::render::light::state::Sun;
use crate::render::view::{MeasureView, ViewFacts, ViewMeasure, ViewSubjects};
use crate::transform::{Mat4, basis_turn, view_basis};
use concinnity_core::math::tan;
use concinnity_core::math::vec3::{dot, length, normalize_or, sub};

/// The frame a view's facts are taken for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewInputs {
    /// World-to-view matrix, column-major.
    pub view: Mat4,
    /// The camera's position.
    pub eye: [f32; 3],
    /// Vertical field of view, in radians.
    pub fov_y: f32,
    /// The inverse of the view-projection the depth is drawn through.
    pub depth_inv_vp: Mat4,
    /// The size the scene renders at.
    pub render_size: [u32; 2],
    /// The subjects in the frame.
    pub subjects: ViewSubjects,
    /// The first directional light.
    pub sun: Sun,
    /// Whether the frame's depth is sampled.
    pub sampled: bool,
}

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
    last: Option<LastView>,
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
        self.last = None;
    }

    /// Carry where the camera stood into the frame the world moved to.
    pub fn rebase(&mut self, rebase: &Rebase) {
        if let Some(last) = &mut self.last {
            last.eye = rebase.apply_point(last.eye);
            last.basis = last.basis.map(|a| rebase.apply_vector(a));
        }
    }

    /// Take the facts of `frame`, and stamp its depth samples.
    pub(crate) fn refresh(&mut self, frame: &ViewInputs) {
        let [_, height] = frame.render_size;
        let focal_px = height.max(1) as f32 / (2.0 * tan(frame.fov_y * 0.5)).max(1e-6);
        let mut subjects = frame.subjects.subjects(frame.eye);
        if let Some((_, measured)) = self.measure.latest() {
            for (subject, seen) in subjects.iter_mut().zip(measured.subjects) {
                if let Some(s) = subject {
                    s.seen = seen;
                }
            }
        }
        let planet = frame.subjects.planet.map(|p| p.ground);
        let up = planet.map_or([0.0, 1.0, 0.0], |g| {
            normalize_or(sub(frame.eye, g.center), 1e-6, [0.0, 1.0, 0.0])
        });
        let mut facts = ViewFacts {
            focal_px,
            render_size: frame.render_size,
            ground_height: planet.map(|g| length(sub(frame.eye, g.center)) - g.radius),
            sea_height: frame.subjects.sea_height(),
            motion_px: 0.0,
            direct_light: luminance(frame.sun.color) * dot(up, frame.sun.direction).max(0.0),
            seen: self.measure.seen(),
            subjects,
        };
        let basis = view_basis(&frame.view);
        if let Some(last) = self.last {
            let turn = basis_turn(&last.basis, &basis) * focal_px;
            let moved = length(sub(frame.eye, last.eye));
            let shift = facts
                .nearest_still()
                .map_or(0.0, |n| facts.pixels(moved, n));
            facts.motion_px = turn + shift;
        }
        self.facts = facts;
        self.last = Some(LastView {
            eye: frame.eye,
            basis,
        });
        let params = self.measure.begin(&MeasureView {
            depth_inv_vp: frame.depth_inv_vp,
            eye: frame.eye,
            size: frame.render_size,
            subjects: frame.subjects,
        });
        self.samples = frame.sampled.then_some(params);
    }
}

#[cfg(test)]
mod tests;
