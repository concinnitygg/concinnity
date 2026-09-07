//! Dispatch geometry for the single-pass Hi-Z downsampler.
//!
//! One workgroup of the `hiz_spd_*` kernels reduces a [`TILE`]x[`TILE`] tile of
//! its base level through [`LEVELS`] levels, so a pyramid is built in two
//! dispatches instead of one per mip: phase 1 reads the main depth and writes
//! mips 0..5, and the tail continues from mip 5 to write mips 6..10. Vulkan and
//! DirectX share this plan; Metal still runs the per-mip chain.

use crate::render::uniforms::HizSpdParams;

/// Levels one SPD dispatch produces, counting its base level.
pub const LEVELS: u32 = 6;

/// Base-level texels one workgroup reduces, per axis.
pub const TILE: u32 = 32;

/// Deepest pyramid two dispatches can produce: phase 1 covers levels 0..5 and
/// the tail adds another five on top of the mip 5 it starts from.
pub const MAX_MIPS: u32 = LEVELS + LEVELS - 1;

/// One dispatch of the plan.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// Pyramid mip this dispatch's base level is, and the first of the
    /// [`LEVELS`] mips its descriptors bind.
    pub base_mip: u32,
    /// Push / root constants for the kernel.
    pub params: HizSpdParams,
    /// Workgroups to dispatch, X and Y.
    pub groups: (u32, u32),
}

/// The dispatches that build a `mip_count`-deep pyramid over a `width` x
/// `height` depth source.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Depth source to mips 0..6. Always present.
    pub phase1: Dispatch,
    /// Mip 6 to mips 7..12. Absent when the pyramid stops at mip 6 or shallower.
    pub tail: Option<Dispatch>,
}

/// Size of mip `level` of a `base`-sized image, floored at 1.
pub fn level_size(base: (u32, u32), level: u32) -> (u32, u32) {
    ((base.0 >> level).max(1), (base.1 >> level).max(1))
}

impl Plan {
    /// Plan the two dispatches. `mip_count` is clamped to [`MAX_MIPS`]; a
    /// shallower pyramid than requested only costs the cull a coarser level to
    /// pick from, which makes it more permissive rather than wrong.
    pub fn new(width: u32, height: u32, mip_count: u32, sample_count: u32) -> Self {
        let base = (width.max(1), height.max(1));
        let mips = mip_count.clamp(1, MAX_MIPS);
        let phase1 = Dispatch {
            base_mip: 0,
            params: HizSpdParams {
                base_width: base.0,
                base_height: base.1,
                level_count: mips.min(LEVELS),
                sample_count,
            },
            groups: (base.0.div_ceil(TILE), base.1.div_ceil(TILE)),
        };
        // The tail's own base is mip `LEVELS - 1`, which it reads and does not
        // rewrite, so it earns its dispatch only once a deeper mip exists.
        let tail = (mips > LEVELS).then(|| {
            let tail_base = level_size(base, LEVELS - 1);
            let groups = (tail_base.0.div_ceil(TILE), tail_base.1.div_ceil(TILE));
            Dispatch {
                base_mip: LEVELS - 1,
                params: HizSpdParams {
                    base_width: tail_base.0,
                    base_height: tail_base.1,
                    level_count: tail_level_count(tail_base, groups, mips - (LEVELS - 1)),
                    sample_count,
                },
                groups,
            }
        });
        Self { phase1, tail }
    }

    /// Pyramid depth this plan actually writes. The image and the cull's mip
    /// count must both use it: a mip the plan skipped is never written, and
    /// sampling one would feed the cull uninitialised memory.
    pub fn mip_count(&self) -> u32 {
        match self.tail {
            Some(t) => t.base_mip + t.params.level_count,
            None => self.phase1.params.level_count,
        }
    }

    /// Mips the dispatch starting at `base_mip` binds, clamped to the pyramid's
    /// actual depth. Descriptors past the end repeat the last live mip so the
    /// array is fully populated; the kernel's `level_count` keeps it from
    /// writing them.
    pub fn bound_mips(base_mip: u32, mip_count: u32) -> impl Iterator<Item = u32> {
        (0..LEVELS).map(move |i| (base_mip + i).min(mip_count.saturating_sub(1)))
    }
}

// Levels the tail may write before its groups start colliding. A level with
// fewer texts per axis than there are groups would have several groups target
// the same texel, so the plan stops one level short instead; the pyramid ends
// shallower, which only makes the cull more permissive.
fn tail_level_count(base: (u32, u32), groups: (u32, u32), requested: u32) -> u32 {
    let mut count = 1;
    for level in 1..requested.min(LEVELS) {
        let size = level_size(base, level);
        if size.0 < groups.0 || size.1 < groups.1 {
            break;
        }
        count = level + 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1024x768 is 11 mips, exactly what two dispatches reach: phase 1 takes
    // 0..5 over a 32x24 grid, the tail takes 6..10 from the 32x24 mip 5 in a
    // single workgroup.
    #[test]
    fn plan_at_1024x768() {
        let p = Plan::new(1024, 768, 11, 1);
        assert_eq!(p.phase1.groups, (32, 24));
        assert_eq!(p.phase1.params.level_count, LEVELS);
        let tail = p.tail.expect("11 mips needs a tail");
        assert_eq!(tail.base_mip, LEVELS - 1);
        assert_eq!((tail.params.base_width, tail.params.base_height), (32, 24));
        assert_eq!(tail.params.level_count, 6);
        assert_eq!(tail.groups, (1, 1));
        assert_eq!(p.mip_count(), 11);
    }

    // A pyramid that stops inside phase 1's reach spends one dispatch.
    #[test]
    fn shallow_pyramid_has_no_tail() {
        let p = Plan::new(32, 32, LEVELS, 1);
        assert!(p.tail.is_none());
        assert_eq!(p.phase1.groups, (1, 1));
        assert_eq!(p.phase1.params.level_count, LEVELS);
    }

    #[test]
    fn one_mip_writes_only_the_base() {
        let p = Plan::new(1920, 1080, 1, 1);
        assert_eq!(p.phase1.params.level_count, 1);
        assert!(p.tail.is_none());
    }

    // Every level the tail writes is owned by exactly one workgroup: its
    // coarsest level has at least as many texels as there are groups, so no two
    // groups target the same texel.
    #[test]
    fn tail_levels_are_never_shared_between_groups() {
        for (w, h) in [
            (1024u32, 768u32),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
            (7680, 4320),
        ] {
            let mips = 32 - w.max(h).leading_zeros();
            let Some(tail) = Plan::new(w, h, mips, 1).tail else {
                continue;
            };
            let base = (tail.params.base_width, tail.params.base_height);
            let coarsest = tail.params.level_count - 1;
            let size = level_size(base, coarsest);
            assert!(
                size.0 >= tail.groups.0 && size.1 >= tail.groups.1,
                "{w}x{h}: level {coarsest} is {size:?} for {:?} groups",
                tail.groups
            );
        }
    }

    // Beyond MAX_MIPS the plan clamps rather than leaving levels unwritten.
    // 8192 divides into tiles exactly, so every level still has a sole owner
    // and the tail keeps its full reach.
    #[test]
    fn deep_pyramid_clamps_to_what_two_dispatches_reach() {
        assert_eq!(MAX_MIPS, 11);
        let p = Plan::new(8192, 8192, 14, 1);
        let tail = p.tail.expect("deep pyramid needs a tail");
        assert_eq!(tail.params.level_count, LEVELS);
        assert_eq!(p.mip_count(), MAX_MIPS);
    }

    // 7680 does not: mip 5 is 240, which needs eight tiles across but falls to
    // seven texels by the tail's last level, so the plan gives that level up.
    #[test]
    fn tail_gives_up_a_level_rather_than_let_groups_collide() {
        let p = Plan::new(7680, 4320, 13, 1);
        let tail = p.tail.expect("deep pyramid needs a tail");
        assert_eq!(tail.groups, (8, 5));
        assert_eq!(tail.params.level_count, 5);
        assert_eq!(p.mip_count(), 10);
    }

    // The pyramid depth a backend allocates and tells the cull about is the
    // depth the plan writes, never the depth that was asked for.
    #[test]
    fn mip_count_reports_what_is_written() {
        assert_eq!(Plan::new(1024, 768, 11, 1).mip_count(), 11);
        assert_eq!(Plan::new(32, 32, LEVELS, 1).mip_count(), LEVELS);
        assert_eq!(Plan::new(1920, 1080, 1, 1).mip_count(), 1);
        assert!(Plan::new(3840, 2160, 12, 1).mip_count() <= 12);
    }

    #[test]
    fn bound_mips_repeats_the_last_live_mip() {
        assert!(Plan::bound_mips(5, 9).eq([5, 6, 7, 8, 8, 8]));
        assert!(Plan::bound_mips(0, 11).eq([0, 1, 2, 3, 4, 5]));
    }
}
