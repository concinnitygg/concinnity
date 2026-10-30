//! The hash the clouds' noise is drawn from, and the size of the tiling noise
//! volume every layer is carved from. The GPU builds the volume once
//! (`cloud_noise_kernel` in `clouds.hlsl`); `volume` is its CPU mirror.

#[cfg(test)]
mod volume;

/// Texels along each edge of the noise volume. `CLOUD_NOISE_SIZE` in
/// `cloud_types.hlsl`.
pub const NOISE_SIZE: u32 = 128;

// The PCG hash over three lanes (Jarzynski and Olano 2020). `cloud_hash3`.
pub(super) fn hash3(v: [u32; 3]) -> [u32; 3] {
    let [mut x, mut y, mut z] = v.map(|c| c.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
    x = x.wrapping_add(y.wrapping_mul(z));
    y = y.wrapping_add(z.wrapping_mul(x));
    z = z.wrapping_add(x.wrapping_mul(y));
    x ^= x >> 16;
    y ^= y >> 16;
    z ^= z >> 16;
    x = x.wrapping_add(y.wrapping_mul(z));
    y = y.wrapping_add(z.wrapping_mul(x));
    z = z.wrapping_add(x.wrapping_mul(y));
    [x, y, z]
}

// `hash3` as three floats in [0, 1). `cloud_hash3f`.
pub(super) fn hash3f(v: [u32; 3]) -> [f32; 3] {
    hash3(v).map(|h| (h >> 8) as f32 * (1.0 / 16_777_216.0))
}

/// Three floats in [0, 1) drawn from `seed`.
pub(super) fn hash_unit3(seed: u32) -> [f32; 3] {
    hash3f([seed, seed ^ 0x9e37_79b9, seed.wrapping_add(0x7f4a_7c15)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;

    #[test]
    fn the_hash_is_deterministic_and_spread() {
        assert_eq!(hash3([1, 2, 3]), hash3([1, 2, 3]));
        assert_ne!(hash3([1, 2, 3]), hash3([1, 2, 4]));
        let mean = (0..1000u32).map(|i| hash3f([i, 7, 9])[0]).sum::<f32>() / 1000.0;
        assert!(close(mean, 0.5, 0.05), "{mean}");
    }
}
