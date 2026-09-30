//! Deterministic integer hashing for procedural content that must come out
//! identical on every build and every platform.

/// One step of a 32-bit linear congruential generator, finished with an
/// xor-shift that folds the high bits into the low ones.
pub fn lcg_hash(v: u32) -> u32 {
    let v = v.wrapping_mul(1664525).wrapping_add(1013904223);
    v ^ (v >> 16)
}

/// Hash of the 2D integer lattice point `(x, y)`.
pub fn lattice_hash(x: u32, y: u32) -> u32 {
    lcg_hash(x.wrapping_mul(1619).wrapping_add(y.wrapping_mul(31337)))
}

/// The low byte of [`lattice_hash`] as a value in `[0, 1]`.
pub fn lattice_value(x: u32, y: u32) -> f32 {
    (lattice_hash(x, y) & 0xFF) as f32 / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_hash_is_the_documented_generator_step() {
        assert_eq!(lcg_hash(0), 1013904223 ^ (1013904223 >> 16));
        assert_ne!(lcg_hash(0), lcg_hash(1));
    }

    #[test]
    fn lattice_values_are_normalized_and_position_dependent() {
        for x in 0..16u32 {
            for y in 0..16u32 {
                let v = lattice_value(x, y);
                assert!((0.0..=1.0).contains(&v), "lattice_value({x},{y}) = {v}");
            }
        }
        assert_ne!(lattice_value(0, 0), lattice_value(1, 0));
        assert_ne!(lattice_hash(0, 1), lattice_hash(1, 0));
    }
}
