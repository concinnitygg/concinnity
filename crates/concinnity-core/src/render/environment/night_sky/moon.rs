//! The moon: its path around the celestial sphere, the phase its angle from
//! the sun gives it, and how its face is lit. Its face is mirrored from
//! `moon_face` in `night_sky.hlsl`.

use super::celestial::CelestialBasis;
use crate::math::narrow_f32;
#[cfg(test)]
use crate::math::sqrt;
use crate::math::vec3::{cross, dot, normalize_or, scale, sub};
use crate::math::{acos, cos, sin};
use core::f32::consts::PI;

/// The moon's path: where it starts, how fast it goes round, and how far its
/// circle is tilted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonPath {
    /// Its angle east of the sun at sky angle 0, in degrees.
    pub start_deg: f32,
    /// Full turns of the sky per lap of its path; 0 holds it still.
    pub period_days: f32,
    /// Tilt of its circle about the line to the sun, in radians.
    pub inclination: f32,
}

impl MoonPath {
    /// The moon's angle east of the sun, in radians, once the sky has made
    /// `turns` full turns and stands at `angle_deg`.
    pub fn elongation(&self, turns: i32, angle_deg: f32) -> f32 {
        if self.period_days <= 0.0 {
            return self.start_deg.to_radians();
        }
        let days = f64::from(turns) + f64::from(angle_deg) / 360.0;
        let deg = (f64::from(self.start_deg) + days * 360.0 / f64::from(self.period_days)) % 360.0;
        let deg = if deg < 0.0 { deg + 360.0 } else { deg };
        narrow_f32(deg).to_radians()
    }

    /// The moon's direction on the celestial sphere at `elongation`: on the
    /// great circle leaving the sun due east, turned about the sun by the
    /// inclination.
    pub fn direction(&self, basis: &CelestialBasis, elongation: f32) -> [f32; 3] {
        let side = cross(basis.sun, basis.east);
        let path: [f32; 3] = core::array::from_fn(|i| {
            cos(self.inclination) * basis.east[i] + sin(self.inclination) * side[i]
        });
        let d =
            core::array::from_fn(|i| cos(elongation) * basis.sun[i] + sin(elongation) * path[i]);
        normalize_or(d, 1e-6, basis.sun)
    }
}

/// The share of the full moon's light a moon at `cos_elongation` from the sun
/// sends: a matte sphere's phase law, 1 when full, `1/pi` at a quarter, 0
/// when new.
pub(crate) fn lambert_phase(cos_elongation: f32) -> f32 {
    let alpha = PI - acos(cos_elongation.clamp(-1.0, 1.0));
    (sin(alpha) + (PI - alpha) * cos(alpha)) / PI
}

/// The moon's up across its face: the celestial pole's side, square to the
/// line of sight, so its face keeps the same way up as the sky turns.
pub(crate) fn face_up(pole: [f32; 3], moon: [f32; 3]) -> [f32; 3] {
    let up = sub(pole, scale(moon, dot(moon, pole)));
    let fallback = if moon[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    normalize_or(
        up,
        1e-4,
        normalize_or(cross(cross(moon, fallback), moon), 1e-6, fallback),
    )
}

/// How brightly the point `(x, y)` of the moon's unit disc (`x` right, `y`
/// up) is lit by a sun toward `sun`, as a share of the full moon's mean face:
/// a matte sphere, so a face lit head-on averages 1 over the disc.
#[cfg(test)]
pub(crate) fn face_light(moon: [f32; 3], up: [f32; 3], sun: [f32; 3], x: f32, y: f32) -> f32 {
    let r2 = x * x + y * y;
    if r2 >= 1.0 {
        return 0.0;
    }
    let right = cross(moon, up);
    let z = sqrt(1.0 - r2);
    let n: [f32; 3] = core::array::from_fn(|i| right[i] * x + up[i] * y - moon[i] * z);
    1.5 * dot(n, sun).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sky::SkyOrientation;

    fn basis() -> CelestialBasis {
        CelestialBasis::new([0.0, 0.0, -1.0], [1.0, 0.0, 0.35])
    }

    fn path(start_deg: f32) -> MoonPath {
        MoonPath {
            start_deg,
            period_days: 29.53,
            inclination: 5.1_f32.to_radians(),
        }
    }

    // The moon sits its elongation from the sun, opposite it when full.
    #[test]
    fn the_moon_sits_its_elongation_from_the_sun() {
        let b = basis();
        for deg in [0.0_f32, 45.0, 90.0, 180.0, 300.0] {
            let m = path(deg).direction(&b, deg.to_radians());
            assert!((crate::math::vec3::length(m) - 1.0).abs() < 1e-5);
            let angle = acos(dot(m, b.sun).clamp(-1.0, 1.0)).to_degrees();
            let expected = if deg > 180.0 { 360.0 - deg } else { deg };
            assert!((angle - expected).abs() < 0.05, "{deg}: {angle}");
        }
        // The tilt carries it off the untilted circle by up to its
        // inclination.
        let quarter = path(90.0).direction(&b, core::f32::consts::FRAC_PI_2);
        assert!((dot(quarter, b.east) - cos(5.1_f32.to_radians())).abs() < 1e-5);
    }

    // A day later the moon is about 12 degrees further east, and the clock
    // runs on across the sky's wrap rather than jumping back.
    #[test]
    fn the_moon_drifts_east_a_little_each_day() {
        let p = path(100.0);
        let today = p.elongation(0, 350.0);
        let tomorrow = p.elongation(1, 350.0);
        let step = (tomorrow - today).to_degrees();
        assert!((step - 360.0 / 29.53).abs() < 1e-3, "{step}");
        let before = p.elongation(0, 359.9);
        let after = p.elongation(1, 0.1);
        assert!((after - before).to_degrees().abs() < 0.01);
        let still = MoonPath {
            period_days: 0.0,
            ..p
        };
        assert_eq!(still.elongation(5, 10.0), 100.0_f32.to_radians());
        assert!(p.elongation(-3, 10.0) >= 0.0);
    }

    // A first-quarter moon is high in the sky as the sun sets: it trails the
    // sun across the sky by a quarter turn.
    #[test]
    fn a_first_quarter_moon_is_up_at_sunset() {
        let b = basis();
        let moon = path(90.0).direction(&b, core::f32::consts::FRAC_PI_2);
        let mut angle = 0.0;
        // The sky angle at which the sun sets after noon.
        while angle < 360.0 {
            let sky = SkyOrientation::new(b.pole, angle);
            let next = SkyOrientation::new(b.pole, angle + 0.5);
            if sky.rotate(b.sun)[1] > 0.0 && next.rotate(b.sun)[1] <= 0.0 {
                break;
            }
            angle += 0.5;
        }
        assert!(angle < 360.0);
        let up = SkyOrientation::new(b.pole, angle).rotate(moon)[1];
        assert!(up > 0.5, "the moon at sunset: {up}");
    }

    // The face's light averaged over the disc follows the phase law the
    // moon's light is scaled by, so the disc and the light it casts agree.
    #[test]
    fn the_lit_face_matches_the_phase_law() {
        let moon = [0.0, 0.0, 1.0];
        let up = face_up([0.0, 1.0, 0.0], moon);
        assert!((up[1] - 1.0).abs() < 1e-6);
        for elong_deg in [180.0_f32, 135.0, 90.0, 45.0, 10.0] {
            let e = elong_deg.to_radians();
            // The sun `e` away from the moon, as seen from the viewer.
            let sun = [sin(e), 0.0, cos(e)];
            let n = 300;
            let h = 2.0 / n as f32;
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let x = -1.0 + (i as f32 + 0.5) * h;
                    let y = -1.0 + (j as f32 + 0.5) * h;
                    sum += face_light(moon, up, sun, x, y) * h * h;
                }
            }
            let mean = sum / PI;
            let phase = lambert_phase(cos(e));
            assert!(
                (mean - phase).abs() < 0.01,
                "{elong_deg}: {mean} vs {phase}"
            );
        }
        assert!((lambert_phase(-1.0) - 1.0).abs() < 1e-6);
        assert!((lambert_phase(0.0) - 1.0 / PI).abs() < 1e-5);
        assert!(lambert_phase(1.0).abs() < 1e-6);
    }
}
