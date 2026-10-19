//! The integer hash every procedural piece of the night sky is drawn from.
//! Mirrors `night_hash` / `night_unit` in `night_sky.hlsl` bit for bit, so the
//! CPU sees the same stars and galaxies the GPU draws.

/// The PCG output permutation over one step of its generator: a well mixed
/// 32-bit hash of `v`.
pub fn night_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// A hash's top 24 bits as a float in `[0, 1)`.
pub fn night_unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `seed` folded into the 24 bits a shader receives exactly as a float.
pub fn seed_lane(seed: u32) -> f32 {
    (night_hash(seed) >> 8) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_spreads_neighbors_apart_and_units_stay_in_range() {
        let a = night_hash(1);
        let b = night_hash(2);
        assert_ne!(a, b);
        assert!((a ^ b).count_ones() > 8, "{a:08x} {b:08x}");
        assert_eq!(night_unit(0), 0.0);
        assert!(night_unit(u32::MAX) < 1.0);
        let mean = (0..4096).map(|i| night_unit(night_hash(i))).sum::<f32>() / 4096.0;
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
    }

    // The seed lane round-trips through a float exactly.
    #[test]
    fn the_seed_lane_is_an_exact_integer() {
        for seed in [0, 1, 7, u32::MAX] {
            let lane = seed_lane(seed);
            assert_eq!(lane as u32 as f32, lane);
            assert!(lane < 16_777_216.0);
        }
    }
}
