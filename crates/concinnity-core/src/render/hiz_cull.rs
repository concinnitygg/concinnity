//! The draw cull's Hi-Z occlusion test: which mip and texel rect of the pyramid
//! an object's screen rect is tested against, and how that rect is reduced to
//! one occluder depth. Mirrors `hiz_occluded` in `cull.hlsl`.

use crate::math::{ceil, floor, log2};
use crate::render::depth;

/// Widest footprint per axis, in texels, the cull gathers at the deepest mip of
/// a pyramid shorter than the full chain: the last level of a 16384x16384
/// target. A wider footprint keeps the object.
pub const MAX_GATHER_SPAN: u32 = 16;

/// The texels of one pyramid mip a screen rect covers, inclusive.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Pyramid mip the rect is tested at.
    pub mip: u32,
    /// Top-left texel.
    pub lo: [u32; 2],
    /// Bottom-right texel.
    pub hi: [u32; 2],
}

/// How the cull reduces a [`Footprint`] to an occluder depth.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// The rect fits a 2x2 footprint: its four corners cover it.
    FourTap,
    /// The rect is wider than 2x2 at the deepest mip: every texel is read.
    Gather,
    /// The rect exceeds [`MAX_GATHER_SPAN`]: the object is kept untested.
    Keep,
}

impl Footprint {
    /// The footprint of the UV rect `uv_min..uv_max` (already clipped to
    /// `[0, 1]`) in a `mip_count`-deep pyramid over a `hiz_size` base. The mip
    /// is the one whose texels are about the rect's size, clamped to the
    /// deepest the pyramid has.
    pub fn new(uv_min: [f32; 2], uv_max: [f32; 2], hiz_size: [f32; 2], mip_count: u32) -> Self {
        let max_dim = ((uv_max[0] - uv_min[0]) * hiz_size[0])
            .max((uv_max[1] - uv_min[1]) * hiz_size[1])
            .max(1.0);
        let mip = (ceil(log2(max_dim)) as i32).clamp(0, mip_count.max(1) as i32 - 1) as u32;
        let scale = (1u32 << mip) as f32;
        let texel = |axis: usize, uv: f32| {
            let dim = (hiz_size[axis] / scale).max(1.0);
            (floor(uv * dim) as i32).clamp(0, dim as i32 - 1) as u32
        };
        Self {
            mip,
            lo: [texel(0, uv_min[0]), texel(1, uv_min[1])],
            hi: [texel(0, uv_max[0]), texel(1, uv_max[1])],
        }
    }

    /// Texels covered per axis.
    pub fn span(&self) -> [u32; 2] {
        [self.hi[0] - self.lo[0] + 1, self.hi[1] - self.lo[1] + 1]
    }

    /// How the cull reads this footprint.
    pub fn reduction(&self) -> Reduction {
        let span = self.span();
        if span[0] <= 2 && span[1] <= 2 {
            Reduction::FourTap
        } else if span[0] > MAX_GATHER_SPAN || span[1] > MAX_GATHER_SPAN {
            Reduction::Keep
        } else {
            Reduction::Gather
        }
    }

    /// The farthest depth `load(x, y, mip)` holds over the footprint, or `None`
    /// when the footprint is too wide to test.
    pub fn occluder_depth(&self, load: impl Fn(u32, u32, u32) -> f32) -> Option<f32> {
        let (lo, hi) = (self.lo, self.hi);
        let at = |x: u32, y: u32| load(x, y, self.mip);
        match self.reduction() {
            Reduction::FourTap => Some(depth::farther(
                depth::farther(at(lo[0], lo[1]), at(hi[0], lo[1])),
                depth::farther(at(lo[0], hi[1]), at(hi[0], hi[1])),
            )),
            Reduction::Gather => Some(
                (lo[1]..=hi[1])
                    .flat_map(|y| (lo[0]..=hi[0]).map(move |x| (x, y)))
                    .fold(depth::DEPTH_NEAR, |d, (x, y)| depth::farther(d, at(x, y))),
            ),
            Reduction::Keep => None,
        }
    }

    /// Whether an object whose nearest device depth is `nearest_depth` is
    /// hidden behind everything `load` holds over the footprint.
    pub fn occludes(&self, nearest_depth: f32, load: impl Fn(u32, u32, u32) -> f32) -> bool {
        self.occluder_depth(load)
            .is_some_and(|occluder| depth::behind(nearest_depth, occluder))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::hiz_spd::{full_mip_count, pyramid_mip_count};
    use crate::render::{shader_consts, shaders};

    const TARGETS: [(u32, u32); 8] = [
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3440, 1440),
        (3840, 2160),
        (5120, 2880),
        (7680, 4320),
        (16384, 16384),
    ];

    const WALL: f32 = 0.5;
    const BEHIND_WALL: f32 = 0.2;

    fn size(w: u32, h: u32) -> [f32; 2] {
        [w as f32, h as f32]
    }

    fn whole_screen(w: u32, h: u32, mip_count: u32) -> Footprint {
        Footprint::new([0.0, 0.0], [1.0, 1.0], size(w, h), mip_count)
    }

    #[test]
    fn the_gather_bound_matches_the_shader() {
        assert_eq!(
            shader_consts::uint(shaders::CULL, "HIZ_GATHER_MAX_SPAN"),
            MAX_GATHER_SPAN as usize
        );
    }

    #[test]
    fn a_small_rect_takes_the_four_tap() {
        let fp = Footprint::new([0.40, 0.40], [0.45, 0.47], size(2560, 1440), 10);
        assert_eq!(fp.reduction(), Reduction::FourTap);
        assert!(fp.occludes(BEHIND_WALL, |_, _, _| WALL));
        assert!(!fp.occludes(BEHIND_WALL, |x, y, _| {
            if [x, y] == fp.hi {
                depth::DEPTH_FAR
            } else {
                WALL
            }
        }));
    }

    // Below the deepest mip the selection always lands on a 2x2 footprint, so
    // only a clamped mip ever gathers.
    #[test]
    fn an_unclamped_mip_always_fits_two_by_two() {
        let hiz = size(1920, 1080);
        let mips = full_mip_count(1920, 1080);
        for step in 1..=64u32 {
            let extent = step as f32 / 64.0;
            for offset in [0.0, 0.013, 0.37, 1.0 - extent] {
                let fp = Footprint::new(
                    [offset, offset * 0.5],
                    [offset + extent, offset * 0.5 + extent * 0.5],
                    hiz,
                    mips,
                );
                assert_eq!(
                    fp.reduction(),
                    Reduction::FourTap,
                    "{extent} at {offset}: {fp:?}"
                );
            }
        }
    }

    // The full chain Metal builds ends at one texel in its larger dimension,
    // so even a rect covering the screen never gathers there.
    #[test]
    fn a_full_chain_never_gathers() {
        for (w, h) in TARGETS {
            let fp = whole_screen(w, h, full_mip_count(w, h));
            assert_eq!(fp.reduction(), Reduction::FourTap, "{w}x{h}: {fp:?}");
        }
    }

    #[test]
    fn a_wide_rect_at_the_deepest_mip_gathers_every_texel() {
        let mips = pyramid_mip_count(3840, 2160);
        let fp = whole_screen(3840, 2160, mips);
        assert_eq!(fp.mip, mips - 1);
        assert_eq!(fp.span(), [7, 4]);
        assert_eq!(fp.reduction(), Reduction::Gather);
        assert!(fp.occludes(BEHIND_WALL, |_, _, _| WALL));
    }

    // A hole in the interior, which the four corners never see, keeps the
    // object visible.
    #[test]
    fn a_gather_sees_an_interior_hole() {
        let fp = whole_screen(3840, 2160, pyramid_mip_count(3840, 2160));
        let hole = |x: u32, y: u32, _| {
            if [x, y] == [3, 2] {
                depth::DEPTH_FAR
            } else {
                WALL
            }
        };
        let corners = [fp.lo, [fp.hi[0], fp.lo[1]], [fp.lo[0], fp.hi[1]], fp.hi];
        assert!(corners.iter().all(|&[x, y]| hole(x, y, fp.mip) == WALL));
        assert!(!fp.occludes(BEHIND_WALL, hole));
    }

    #[test]
    fn every_listed_target_stays_within_the_gather_bound() {
        for (w, h) in TARGETS {
            let fp = whole_screen(w, h, pyramid_mip_count(w, h));
            assert_ne!(fp.reduction(), Reduction::Keep, "{w}x{h}: {fp:?}");
        }
    }

    // 15872x16384 loses a tail level to group collisions and ends on a 31x32
    // mip, past the bound, so a rect covering it is kept untested.
    #[test]
    fn a_rect_past_the_bound_is_kept() {
        let mips = pyramid_mip_count(15872, 16384);
        let fp = whole_screen(15872, 16384, mips);
        assert_eq!(fp.span(), [31, 32]);
        assert_eq!(fp.reduction(), Reduction::Keep);
        assert_eq!(fp.occluder_depth(|_, _, _| WALL), None);
        assert!(!fp.occludes(BEHIND_WALL, |_, _, _| WALL));
    }
}
