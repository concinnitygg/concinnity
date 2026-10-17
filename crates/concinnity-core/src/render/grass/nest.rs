//! How the field's cells nest, so the kernel visits about as many candidates
//! as distance thinning keeps blades, at any reach.
//!
//! Cells come in levels: a cell of level `j` covers `2^j` by `2^j` cells of
//! level 0, the cells blades grow in. Each cell above level 0 picks one of its
//! four children by a hash of itself, so each cell of any level has one
//! representative blade, found by walking down its picks. A blade's rank is
//! the coarsest level it represents.
//!
//! Distance thinning keeps a blade while its keep fraction exceeds its hash
//! (see [`lod`](super::lod)). The hash it thins by is [`nested_hash`] of its
//! rank: uniform over the field like any hash, but below `4^-j` exactly for
//! the representatives of level `j`. So wherever the keep fraction, over the
//! shrink band's span, is at most `4^-j`, every blade that survives is a
//! representative of level `j`, and a tile there need only visit its level `j`
//! cells, a `4^j`-th of its blades. Which blades survive is the same
//! whichever level visits them, so a tile changing level moves nothing.
//!
//! [`representative`], [`rank`] and [`nested_hash`] are the kernel's
//! `grass_nest_representative`, `grass_nest_rank` and `grass_nested_hash`.

use super::planet::density::grass_hash;

/// The coarsest level cells nest to: blades of this rank or more share the
/// lowest stretch of hashes.
pub const GRASS_NEST_LEVELS: u32 = 5;

// The kernel's `grass_hash2`.
fn hash2(c: [i32; 2]) -> u32 {
    grass_hash((c[0] as u32).wrapping_add(grass_hash(c[1] as u32)))
}

// The child, as x in bit 0 and y in bit 1, that cell `cell` of level `level`
// picks, for hashes salted by `salt`.
fn pick(cell: [i32; 2], level: u32, salt: u32) -> u32 {
    grass_hash(hash2(cell) ^ salt ^ grass_hash(level.wrapping_add(0x6a09_e667))) & 3
}

/// The level-0 cell of the blade cell `cell` of level `level` represents.
pub fn representative(cell: [i32; 2], level: u32, salt: u32) -> [i32; 2] {
    let mut c = cell;
    for l in (1..=level).rev() {
        let b = pick(c, l, salt) as i32;
        c = [
            c[0].wrapping_mul(2).wrapping_add(b & 1),
            c[1].wrapping_mul(2).wrapping_add(b >> 1),
        ];
    }
    c
}

/// The rank of the blade of level-0 cell `cell`: the coarsest level, at most
/// [`GRASS_NEST_LEVELS`], whose cell over it it represents.
pub fn rank(cell: [i32; 2], salt: u32) -> u32 {
    let mut c = cell;
    let mut rank = 0;
    while rank < GRASS_NEST_LEVELS {
        let parent = [c[0] >> 1, c[1] >> 1];
        let b = pick(parent, rank + 1, salt) as i32;
        if (c[0] & 1) != (b & 1) || (c[1] & 1) != (b >> 1) {
            break;
        }
        c = parent;
        rank += 1;
    }
    rank
}

// 4^-level.
fn quarter_pow(level: u32) -> f32 {
    1.0 / (1u64 << (2 * level)) as f32
}

/// The hash a blade of rank `rank` thins by, from its own unit hash `u`:
/// within `[4^-(rank + 1), 4^-rank)`, or `[0, 4^-rank)` at the coarsest rank.
pub fn nested_hash(rank: u32, u: f32) -> f32 {
    if rank >= GRASS_NEST_LEVELS {
        quarter_pow(GRASS_NEST_LEVELS) * u
    } else {
        quarter_pow(rank + 1) * (1.0 + 3.0 * u)
    }
}

/// The coarsest level, at most `cap`, whose cells hold every blade that
/// survives a keep fraction of at most `keep` where thinning drops blades
/// whose hash times `span` reaches it.
pub fn nest_level(keep: f32, span: f32, cap: u32) -> u32 {
    let mut level = 0;
    while level < cap.min(GRASS_NEST_LEVELS) && keep <= span * quarter_pow(level + 1) {
        level += 1;
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::grass::lod::survives;
    use crate::render::shaders;

    const SALT: u32 = 0x1234_5678;

    #[test]
    fn a_representative_lies_in_its_cell_and_ranks_at_least_its_level() {
        for level in 0..=GRASS_NEST_LEVELS {
            for (x, y) in [(0, 0), (3, -7), (-1, -1), (1000, 25), (-4096, 77)] {
                let fine = representative([x, y], level, SALT);
                assert_eq!([fine[0] >> level, fine[1] >> level], [x, y]);
                assert!(rank(fine, SALT) >= level);
            }
        }
    }

    // A blade's thinning hash falls under 4^-j exactly when it represents
    // its level-j cell, so the representatives are every blade a keep
    // fraction that low can keep.
    #[test]
    fn the_low_hashes_are_the_representatives() {
        for y in -40..40 {
            for x in -40..40 {
                let r = rank([x, y], SALT);
                for level in 0..=GRASS_NEST_LEVELS {
                    let cell = [x >> level, y >> level];
                    let is_rep = representative(cell, level, SALT) == [x, y];
                    assert_eq!(is_rep, r >= level, "{x} {y} level {level}");
                    for u in [0.0, 0.37, 0.999_99] {
                        let low = nested_hash(r, u) < quarter_pow(level);
                        assert_eq!(low, is_rep, "{x} {y} level {level} u {u}");
                    }
                }
            }
        }
    }

    // Over the field the nested hash is spread like any unit hash, so the
    // keep fraction keeps the same share of blades.
    #[test]
    fn the_nested_hash_is_uniform() {
        let n = 256;
        let mut counts = [0u32; 16];
        for y in 0..n {
            for x in 0..n {
                let h = grass_hash(hash2([x, y]));
                let u = (h >> 8) as f32 / 16_777_216.0;
                let v = nested_hash(rank([x, y], SALT), u);
                counts[(v * 16.0) as usize] += 1;
            }
        }
        let mean = (n * n) as f32 / 16.0;
        for c in counts {
            assert!((c as f32 - mean).abs() < 0.05 * mean, "{counts:?}");
        }
        // Each rank holds its share: a quarter of the blades rank 1 or more.
        let ranked = (0..n)
            .flat_map(|y| (0..n).map(move |x| rank([x, y], SALT)))
            .filter(|&r| r >= 1)
            .count();
        assert_eq!(ranked, (n * n / 4) as usize);
    }

    // At the level picked for a keep fraction, every survivor is one of the
    // level's representatives.
    #[test]
    fn the_picked_level_holds_every_survivor() {
        let span = 0.7;
        for keep in [1.0, 0.5, 0.17, 0.04, 0.011, 0.002, 1e-5] {
            let level = nest_level(keep, span, GRASS_NEST_LEVELS);
            assert!(keep <= span * quarter_pow(level) || level == 0);
            if level < GRASS_NEST_LEVELS {
                assert!(keep > span * quarter_pow(level + 1));
            }
            for y in 0..64 {
                for x in 0..64 {
                    let u = (grass_hash(hash2([x, y])) >> 8) as f32 / 16_777_216.0;
                    let h = nested_hash(rank([x, y], SALT), u);
                    if survives(keep, h) {
                        let cell = [x >> level, y >> level];
                        assert_eq!(representative(cell, level, SALT), [x, y]);
                    }
                }
            }
        }
        assert_eq!(nest_level(0.001, span, 1), 1);
    }

    #[test]
    fn the_kernel_nests_as_deep() {
        assert_eq!(
            crate::render::shader_consts::uint(shaders::GRASS, "GRASS_NEST_LEVELS"),
            GRASS_NEST_LEVELS as usize
        );
    }
}
