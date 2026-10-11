//! The sky of a small planet seen as if over a bigger one.
//!
//! Over a small planet the air a grazing ray crosses is short, so the sky is
//! a flat deep blue and the low sun stays white. An apparent radius builds the
//! sky's tables over a planet of that radius while the camera is near the
//! ground, fading to the true radius as it climbs, so the ground sees an
//! Earth-like sky and the limb seen from high up is the planet's own.
//!
//! The two planets put the horizon at different depths below the level, so
//! the sky is looked up with the horizon warped: a view ray's angle from the
//! zenith is stretched so the true horizon lands on the table's, and the sky
//! meets the drawn ground at its true limb. The sun's place in the table and
//! its color are warped the same way.

use super::lut::horizon_angles;
use crate::math::{exp, ln};

/// An apparent ground radius, and the heights above the true ground over
/// which the sky fades from it to the true radius. Kilometers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApparentGround {
    /// The radius the sky is drawn over near the ground.
    pub radius: f32,
    /// Height at which the fade to the true radius starts.
    pub fade_start: f32,
    /// Height from which the true radius is used.
    pub fade_end: f32,
}

impl ApparentGround {
    /// The ground radius the sky's tables are built over for a camera
    /// `height` above the ground of a planet of `true_radius`: the apparent
    /// radius up to the fade's start, the true one from its end, and between
    /// them a smoothstep in the logarithm of the radius, so each step of the
    /// climb changes the curvature by a like factor.
    pub fn ground_radius(&self, true_radius: f32, height: f32) -> f32 {
        if height <= self.fade_start {
            return self.radius;
        }
        if height >= self.fade_end {
            return true_radius;
        }
        let x = (height - self.fade_start) / (self.fade_end - self.fade_start).max(1e-6);
        let t = x * x * (3.0 - 2.0 * x);
        exp(ln(self.radius) + (ln(true_radius) - ln(self.radius)) * t)
    }
}

/// The angle from the zenith, in radians, at which the sky table holds a
/// view ray `view_angle` from the zenith: the zenith, the horizon and the
/// nadir map onto their own, and the angles between them stretch linearly.
/// `view_horizon` and `table_horizon` are the zenith-to-horizon angles of the
/// view's planet and the table's. Mirrors `atmo_table_view_angle` in
/// atmosphere_common.hlsl.
pub fn table_view_angle(view_angle: f32, view_horizon: f32, table_horizon: f32) -> f32 {
    use core::f32::consts::PI;
    if view_horizon == table_horizon {
        return view_angle;
    }
    if view_angle <= view_horizon {
        view_angle * table_horizon / view_horizon.max(1e-6)
    } else {
        table_horizon
            + (view_angle - view_horizon) * (PI - table_horizon) / (PI - view_horizon).max(1e-6)
    }
}

/// The zenith-to-horizon angle seen from radius `r` over ground `bottom`.
pub fn zenith_horizon(bottom: f32, r: f32) -> f32 {
    horizon_angles(bottom, r).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, PI};

    const SMALL: ApparentGround = ApparentGround {
        radius: 6360.0,
        fade_start: 1.0,
        fade_end: 20.0,
    };

    // Below the fade the apparent radius, above it the true one, and between
    // them every radius in order.
    #[test]
    fn the_ground_radius_fades_with_height() {
        assert_eq!(SMALL.ground_radius(50.0, 0.0), 6360.0);
        assert_eq!(SMALL.ground_radius(50.0, 1.0), 6360.0);
        assert_eq!(SMALL.ground_radius(50.0, 20.0), 50.0);
        assert_eq!(SMALL.ground_radius(50.0, 400.0), 50.0);
        let mut last = 6360.0;
        for h in 1..38 {
            let r = SMALL.ground_radius(50.0, 1.0 + h as f32 * 0.5);
            assert!(r < last && r >= 50.0, "{h}: {r}");
            last = r;
        }
        // Halfway through the fade the radius is the geometric mean.
        let mid = SMALL.ground_radius(50.0, 10.5);
        assert!((mid - (6360.0_f32 * 50.0).sqrt()).abs() < 1.0, "{mid}");
    }

    // The ends of the fade join the flat parts without a jump.
    #[test]
    fn the_fade_is_continuous_at_its_ends() {
        let below = SMALL.ground_radius(50.0, 1.0 + 1e-3);
        assert!((below - 6360.0).abs() < 0.1, "{below}");
        let above = SMALL.ground_radius(50.0, 20.0 - 1e-3);
        assert!((above - 50.0).abs() < 0.01, "{above}");
    }

    #[test]
    fn the_warp_keeps_the_zenith_horizon_and_nadir() {
        let view = zenith_horizon(50.0, 50.4);
        let table = zenith_horizon(6360.0, 6360.4);
        // From 400 m up a 50 km planet's horizon is 7 degrees down, Earth's
        // under one.
        let dip = (view - FRAC_PI_2).to_degrees();
        assert!((dip - 7.2).abs() < 0.2, "{dip}");
        assert!((table - FRAC_PI_2).to_degrees() < 1.0);
        assert_eq!(table_view_angle(0.0, view, table), 0.0);
        assert!((table_view_angle(view, view, table) - table).abs() < 1e-6);
        assert!((table_view_angle(PI, view, table) - PI).abs() < 1e-5);
        let mut last = -1.0;
        for i in 0..=64 {
            let a = table_view_angle(PI * i as f32 / 64.0, view, table);
            assert!(a > last, "{i}");
            last = a;
        }
    }

    // Looking a ray up against the true horizon reads the table, built over
    // the apparent ground, at the warped angle: the true horizon on the
    // table's.
    #[test]
    fn the_lookup_reads_the_table_at_the_warped_angle() {
        use super::super::lut::{sky_view_direction, sky_view_uv};
        let (view_b, view_r) = (50.0, 50.4);
        let (table_b, table_r) = (6360.0, 6360.4);
        let view = zenith_horizon(view_b, view_r);
        let table = zenith_horizon(table_b, table_r);
        for i in 1..32 {
            let angle = PI * i as f32 / 32.0;
            let uv = sky_view_uv(view_b, view_r, crate::math::cos(angle), 0.5);
            let (table_cos, _) = sky_view_direction(table_b, table_r, uv);
            let expected = table_view_angle(angle, view, table);
            let got = crate::math::acos(table_cos);
            assert!(
                (got - expected).abs() < 2e-3,
                "{angle}: {got} vs {expected}"
            );
        }
    }

    // One planet for both is no warp at all, bit for bit.

    #[test]
    fn the_same_horizon_is_no_warp() {
        let h = zenith_horizon(50.0, 50.002);
        for a in [0.0, 0.3, h, 2.0, PI] {
            assert_eq!(table_view_angle(a, h, h), a);
        }
    }
}
