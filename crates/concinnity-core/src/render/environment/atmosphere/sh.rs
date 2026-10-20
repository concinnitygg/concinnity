//! Second-order spherical harmonics: the sky's radiance projected onto nine
//! coefficients per color, and the irradiance a surface facing any way
//! receives from it. The CPU mirror of the irradiance kernel in
//! atmosphere.hlsl.

use crate::math::{cos, sin, sqrt};

/// Directions the irradiance kernel samples the sky in. `IRRADIANCE_SAMPLES`
/// in `atmosphere_types.hlsl`.
#[cfg(test)]
const IRRADIANCE_SAMPLES: u32 = 512;

// The cosine lobe convolved band by band (Ramamoorthi and Hanrahan): what
// turns projected radiance into irradiance.
const BAND_LOBE: [f32; 3] = [
    core::f32::consts::PI,
    2.0 * core::f32::consts::PI / 3.0,
    core::f32::consts::PI / 4.0,
];

/// The nine real spherical-harmonic basis functions at unit direction `d`,
/// ordered by band. `sh9_basis` in `atmosphere_common.hlsl`.
pub fn sh9_basis(d: [f32; 3]) -> [f32; 9] {
    let [x, y, z] = d;
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

// The band each of the nine coefficients belongs to.
const BAND: [usize; 9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];

/// Sample `i` of `n` spread evenly over the sphere on a Fibonacci spiral.
/// `fibonacci_direction` in `atmosphere_common.hlsl`.
pub fn fibonacci_direction(i: u32, n: u32) -> [f32; 3] {
    const GOLDEN_ANGLE: f32 = 2.399_963;
    let z = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
    let ring = sqrt((1.0 - z * z).max(0.0));
    let phi = GOLDEN_ANGLE * i as f32;
    [ring * cos(phi), ring * sin(phi), z]
}

/// Radiance projected onto the basis, per color.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Sh9 {
    /// Nine RGB coefficients, ordered as [`sh9_basis`].
    pub coefficients: [[f32; 3]; 9],
}

impl Sh9 {
    /// Project `radiance` over the sphere, sampled in `samples` evenly spread
    /// directions each weighted by its share of the sphere's solid angle.
    pub fn project(samples: u32, radiance: impl Fn([f32; 3]) -> [f32; 3]) -> Self {
        let weight = 4.0 * core::f32::consts::PI / samples as f32;
        let mut coefficients = [[0.0_f32; 3]; 9];
        for i in 0..samples {
            let d = fibonacci_direction(i, samples);
            let l = radiance(d);
            for (c, y) in coefficients.iter_mut().zip(sh9_basis(d)) {
                for k in 0..3 {
                    c[k] += l[k] * y * weight;
                }
            }
        }
        Self { coefficients }
    }

    /// Irradiance on a surface whose normal is the unit `n`, never negative.
    /// `sh9_irradiance` in `atmosphere_common.hlsl`.
    pub fn irradiance(&self, n: [f32; 3]) -> [f32; 3] {
        let basis = sh9_basis(n);
        let mut e = [0.0_f32; 3];
        for (j, (c, y)) in self.coefficients.iter().zip(basis).enumerate() {
            let lobe = BAND_LOBE[BAND[j]];
            for k in 0..3 {
                e[k] += c[k] * lobe * y;
            }
        }
        e.map(|v| v.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::dot;

    #[test]
    fn the_sample_count_matches_the_shader() {
        let src = crate::render::shader::embedded::ATMOSPHERE_TYPES;
        assert_eq!(
            crate::render::shader::consts::uint(src, "IRRADIANCE_SAMPLES"),
            IRRADIANCE_SAMPLES as usize
        );
    }

    // The spiral's samples are unit length and balanced: their mean is the
    // sphere's center.
    #[test]
    fn the_spiral_covers_the_sphere_evenly() {
        let n = IRRADIANCE_SAMPLES;
        let mut mean = [0.0_f32; 3];
        for i in 0..n {
            let d = fibonacci_direction(i, n);
            assert!((dot(d, d) - 1.0).abs() < 1e-5);
            for k in 0..3 {
                mean[k] += d[k] / n as f32;
            }
        }
        assert!(mean.iter().all(|m| m.abs() < 1e-3), "{mean:?}");
    }

    // A uniform sky of radiance L gives irradiance pi * L whichever way a
    // surface faces.
    #[test]
    fn a_uniform_sky_gives_pi_times_its_radiance() {
        let sh = Sh9::project(IRRADIANCE_SAMPLES, |_| [1.0, 0.5, 0.25]);
        for n in [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]] {
            let e = sh.irradiance(n);
            let pi = core::f32::consts::PI;
            assert!((e[0] - pi).abs() < 0.01 * pi, "{e:?}");
            assert!((e[2] - 0.25 * pi).abs() < 0.01 * pi, "{e:?}");
        }
    }

    // A sky lit only above the horizon: a surface facing up sees all of it
    // (pi), one facing down next to none, and a wall half (pi / 2). Second
    // order rings a little, so the bounds are the band-limited ones.
    #[test]
    fn an_upper_hemisphere_lights_the_top_not_the_bottom() {
        let sh = Sh9::project(IRRADIANCE_SAMPLES, |d| {
            if d[1] > 0.0 { [1.0; 3] } else { [0.0; 3] }
        });
        let pi = core::f32::consts::PI;
        let up = sh.irradiance([0.0, 1.0, 0.0])[0];
        let down = sh.irradiance([0.0, -1.0, 0.0])[0];
        let side = sh.irradiance([1.0, 0.0, 0.0])[0];
        assert!((up - pi).abs() < 0.05 * pi, "{up}");
        assert!(down < 0.05 * pi, "{down}");
        assert!((side - 0.5 * pi).abs() < 0.03 * pi, "{side}");
    }
}
