//! Distant galaxies: a few small spiral and elliptical smudges at hashed
//! places on the celestial sphere. Placed here, on the CPU, and handed to the
//! sky's shaders in the atmosphere's constant block; their light profiles are
//! mirrored from `night_sky.hlsl`.

use super::hash::{night_hash, night_unit};
use crate::math::vec3::{cross, dot, normalize_or, scale};
use crate::math::{atan2, cos, exp, ln, powf, sin, smoothstep, sqrt};

/// The most galaxies a sky draws.
pub const MAX_GALAXIES: usize = 16;
/// Angular radius of the smallest and largest galaxy, in degrees.
pub const GALAXY_RADIUS_DEG: [f32; 2] = [0.25, 2.0];
/// Integrated magnitude of the brightest and faintest galaxy.
pub const GALAXY_MAGNITUDE: [f32; 2] = [2.5, 5.5];
/// The integral of [`galaxy_profile`] over a spiral's unit disc.
pub const SPIRAL_NORM: f32 = 0.1945;
/// The integral of [`galaxy_profile`] over an elliptical's unit disc.
pub const ELLIPTICAL_NORM: f32 = 0.1952;

/// One galaxy, in the celestial sphere's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Galaxy {
    /// Unit direction to its center.
    pub direction: [f32; 3],
    /// Unit direction of its long axis across the sky, square to
    /// `direction`.
    pub major: [f32; 3],
    /// Angular radius of its long axis, in radians.
    pub radius: f32,
    /// Its short axis over its long one: how far it is tilted from face-on.
    pub axis_ratio: f32,
    /// Spiral (a disc with arms around a bulge) or elliptical (a smooth
    /// glow).
    pub spiral: bool,
    /// Integrated apparent magnitude.
    pub magnitude: f32,
}

/// Galaxy `index` of the sky with 24-bit seed lane `seed`.
pub fn galaxy(seed: u32, index: u32) -> Galaxy {
    let mut h = night_hash(index.wrapping_add(night_hash(seed ^ 0x6a09_e667)));
    let mut next = || {
        h = night_hash(h);
        night_unit(h)
    };
    let z = 2.0 * next() - 1.0;
    let phi = 2.0 * core::f32::consts::PI * next();
    let ring = sqrt((1.0 - z * z).max(0.0));
    let direction = [ring * cos(phi), z, ring * sin(phi)];
    let size = powf(next(), 3.0);
    let radius =
        (GALAXY_RADIUS_DEG[0] + (GALAXY_RADIUS_DEG[1] - GALAXY_RADIUS_DEG[0]) * size).to_radians();
    let spiral = next() < 0.6;
    let tilt = next();
    let axis_ratio = if spiral {
        tilt.max(0.2)
    } else {
        0.5 + 0.5 * tilt
    };
    let reference = if direction[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t1 = normalize_or(cross(direction, reference), 1e-6, [1.0, 0.0, 0.0]);
    let t2 = cross(direction, t1);
    let angle = 2.0 * core::f32::consts::PI * next();
    let major = normalize_or(
        [0, 1, 2].map(|i| t1[i] * cos(angle) + t2[i] * sin(angle)),
        1e-6,
        t1,
    );
    let magnitude = GALAXY_MAGNITUDE[0] + (GALAXY_MAGNITUDE[1] - GALAXY_MAGNITUDE[0]) * next();
    Galaxy {
        direction,
        major,
        radius,
        axis_ratio,
        spiral,
        magnitude,
    }
}

/// A galaxy's relative brightness at `(x, y)`, in units of its radius along
/// its long axis and, face-on, along its short one: zero past the unit
/// circle. A spiral is an exponential disc wound with two logarithmic arms
/// around a bright bulge; an elliptical a smooth exponential glow.
pub fn galaxy_profile(spiral: bool, x: f32, y: f32) -> f32 {
    let r = sqrt(x * x + y * y);
    if r >= 1.0 {
        return 0.0;
    }
    let edge = 1.0 - smoothstep(0.8, 1.0, r);
    if spiral {
        let arms = 0.5 + 0.5 * cos(2.0 * atan2(y, x) - 6.0 * ln(r + 0.05));
        let disc = exp(-r / 0.22) * (0.3 + 0.7 * arms * arms);
        let bulge = 3.0 * exp(-r / 0.045);
        (disc + bulge) * edge
    } else {
        exp(-r / 0.18) * edge
    }
}

/// The two `float4` lanes a galaxy takes in the atmosphere's constant block:
/// its direction and the radiance at its profile's 1 (`illuminance` is its
/// whole light spread over its area), then its long axis over its radius and
/// its axis ratio, negative for an elliptical.
pub fn galaxy_lanes(g: &Galaxy, illuminance: f32) -> [[f32; 4]; 2] {
    let norm = if g.spiral {
        SPIRAL_NORM
    } else {
        ELLIPTICAL_NORM
    };
    let surface = illuminance / (norm * g.radius * g.radius * g.axis_ratio);
    let major = scale(g.major, 1.0 / g.radius);
    let ratio = if g.spiral {
        g.axis_ratio
    } else {
        -g.axis_ratio
    };
    [
        [g.direction[0], g.direction[1], g.direction[2], surface],
        [major[0], major[1], major[2], ratio],
    ]
}

/// Whether a direction `c` falls on galaxy `g`, and its coordinates there.
/// The shader's test, mirrored for the tests.
pub fn galaxy_coordinates(g: &Galaxy, c: [f32; 3]) -> Option<(f32, f32)> {
    if dot(c, g.direction) <= 0.0 {
        return None;
    }
    let minor = cross(g.direction, g.major);
    let x = dot(c, g.major) / g.radius;
    let y = dot(c, minor) / (g.radius * g.axis_ratio);
    (x * x + y * y < 1.0).then_some((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::length;

    // Galaxies are scattered over the whole sky, a fixed seed places them
    // the same way every time, and their axes lie across the sky.
    #[test]
    fn galaxies_are_placed_over_the_sky_by_their_seed() {
        let all: alloc::vec::Vec<Galaxy> = (0..400).map(|i| galaxy(17, i)).collect();
        let mut mean = [0.0f32; 3];
        for g in &all {
            assert!((length(g.direction) - 1.0).abs() < 1e-4);
            assert!((length(g.major) - 1.0).abs() < 1e-4);
            assert!(dot(g.direction, g.major).abs() < 1e-4);
            let deg = g.radius.to_degrees();
            assert!((GALAXY_RADIUS_DEG[0] - 1e-4..=GALAXY_RADIUS_DEG[1] + 1e-4).contains(&deg));
            assert!((0.2..=1.0).contains(&g.axis_ratio));
            for (m, d) in mean.iter_mut().zip(g.direction) {
                *m += d / all.len() as f32;
            }
        }
        assert!(length(mean) < 0.12, "{mean:?}");
        let spirals = all.iter().filter(|g| g.spiral).count();
        assert!((200..280).contains(&spirals), "{spirals}");
        // Most are small: big ones are rare.
        let big = all.iter().filter(|g| g.radius.to_degrees() > 1.0).count();
        assert!(big < 130, "{big}");
        assert_eq!(galaxy(17, 3), all[3]);
        assert_ne!(galaxy(18, 3), all[3]);
    }

    // The profile's norms make a galaxy's whole light its illuminance, so the
    // lane's radiance spread over its area adds back up to it.
    #[test]
    fn a_galaxy_spreads_its_light_over_its_area() {
        for spiral in [true, false] {
            let n = 400;
            let h = 2.0 / n as f32;
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let x = -1.0 + (i as f32 + 0.5) * h;
                    let y = -1.0 + (j as f32 + 0.5) * h;
                    sum += galaxy_profile(spiral, x, y) * h * h;
                }
            }
            let norm = if spiral { SPIRAL_NORM } else { ELLIPTICAL_NORM };
            assert!((sum / norm - 1.0).abs() < 0.01, "{spiral}: {sum}");
        }
        let g = Galaxy {
            direction: [0.0, 0.0, 1.0],
            major: [1.0, 0.0, 0.0],
            radius: 0.01,
            axis_ratio: 0.5,
            spiral: true,
            magnitude: 4.0,
        };
        let lanes = galaxy_lanes(&g, 2.0);
        let area = g.radius * g.radius * g.axis_ratio;
        assert!((lanes[0][3] * SPIRAL_NORM * area - 2.0).abs() < 1e-4);
        assert!((length([lanes[1][0], lanes[1][1], lanes[1][2]]) - 100.0).abs() < 1e-2);
        assert!(galaxy_coordinates(&g, [0.004, 0.0, 1.0]).is_some());
        assert!(galaxy_coordinates(&g, [0.0, 0.006, 1.0]).is_none());
        assert!(galaxy_coordinates(&g, [0.0, 0.0, -1.0]).is_none());
    }
}
