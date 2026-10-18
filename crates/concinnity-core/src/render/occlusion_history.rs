//! The occlusion test's half of the temporal history: whether the depth
//! pyramid holds the previous frame's depth, and the view-projection that
//! depth was rendered through, which the draw cull and the grass tile test
//! reproject through.

use crate::planet::Rebase;
use crate::render::grass::GrassHiz;
use crate::transform::{IDENTITY, Mat4};

/// Whether the pyramid holds last frame's depth, and the view it was rendered
/// through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionHistory {
    prev_view_proj: Mat4,
    valid: bool,
}

impl Default for OcclusionHistory {
    fn default() -> Self {
        Self {
            prev_view_proj: IDENTITY,
            valid: false,
        }
    }
}

impl OcclusionHistory {
    /// The view-projection the pyramid's depth was rendered through.
    pub fn prev_view_proj(&self) -> Mat4 {
        self.prev_view_proj
    }

    /// Whether the pyramid holds the previous frame's depth, so the occlusion
    /// test can read it.
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// Forget the pyramid's depth: a cut, a resize, or a rebuilt pyramid.
    pub fn reset(&mut self) {
        self.valid = false;
    }

    /// Carry the view into the frame the world moved to.
    pub fn rebase(&mut self, rebase: &Rebase) {
        self.prev_view_proj = rebase.reproject(self.prev_view_proj);
    }

    /// End a frame seen through `view_proj`, which `pyramid` says built the
    /// depth pyramid. The view is captured every frame, so it is current
    /// whenever a pyramid appears; it is only read once one has.
    pub fn advance(&mut self, view_proj: Mat4, pyramid: bool) {
        self.valid |= pyramid;
        self.prev_view_proj = view_proj;
    }

    /// The pyramid as the grass tile test reads it, when it holds the previous
    /// frame's depth: its base `size` in texels and its `mip_count`.
    pub fn grass_hiz(&self, size: [f32; 2], mip_count: u32) -> Option<GrassHiz> {
        self.valid.then_some(GrassHiz {
            prev_vp: self.prev_view_proj,
            size,
            mip_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vp(x: f32) -> Mat4 {
        let mut m = IDENTITY;
        m[3][0] = x;
        m
    }

    #[test]
    fn a_new_history_has_no_pyramid() {
        let h = OcclusionHistory::default();
        assert!(!h.valid());
        assert_eq!(h.prev_view_proj(), IDENTITY);
        assert_eq!(h.grass_hiz([4.0, 4.0], 3), None);
    }

    #[test]
    fn a_built_pyramid_validates_and_captures_the_view() {
        let mut h = OcclusionHistory::default();
        h.advance(vp(2.0), true);
        assert!(h.valid());
        let hiz = h.grass_hiz([8.0, 4.0], 4).expect("valid");
        assert_eq!(
            (hiz.prev_vp, hiz.size, hiz.mip_count),
            (vp(2.0), [8.0, 4.0], 4)
        );
        h.reset();
        assert!(!h.valid());
        assert_eq!(h.prev_view_proj(), vp(2.0), "a reset keeps the view");
    }

    #[test]
    fn the_view_is_captured_before_any_pyramid() {
        let mut h = OcclusionHistory::default();
        h.advance(vp(1.0), false);
        assert_eq!(h.prev_view_proj(), vp(1.0));
        assert!(!h.valid());
        assert_eq!(h.grass_hiz([4.0, 4.0], 3), None);
    }

    #[test]
    fn a_rebase_reprojects_the_view() {
        let rebase = Rebase {
            rotation: crate::math::quat_from_axis_angle([0.0, 1.0, 0.0], 0.3),
            translation: [5.0, 0.0, -2.0],
        };
        let mut h = OcclusionHistory::default();
        h.advance(vp(3.0), true);
        h.rebase(&rebase);
        assert_eq!(h.prev_view_proj(), rebase.reproject(vp(3.0)));
        assert!(h.valid());
    }
}
