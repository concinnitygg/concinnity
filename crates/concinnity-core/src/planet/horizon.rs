// The planet as an occluder: the sphere of its lowest ground, seen from one
// eye, hides everything wholly behind its horizon.

use super::dvec::DVec3;
use crate::math::dvec3;

/// The planet's lowest ground seen from one eye: a sphere that hides whatever
/// lies wholly inside the cone it subtends past the circle where the eye's
/// sight lines graze it.
///
/// The test is conservative: a sphere it hides cannot be seen from the eye,
/// since every sight line to it enters the planet first. Anything within
/// `keep_within` of the eye is never hidden, so a caster below the horizon
/// still shadows the ground in view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Horizon {
    eye: DVec3,
    // Unit vector from the eye toward the sphere's center.
    axis: DVec3,
    // Distance along `axis` from the eye to the plane of the grazing circle.
    plane: f64,
    sin: f64,
    cos: f64,
    keep_within: f64,
}

impl Horizon {
    /// The horizon of the sphere of `radius` around `center` seen from `eye`,
    /// or `None` when the eye is on or inside the sphere, where it hides
    /// nothing.
    pub fn new(eye: DVec3, center: DVec3, radius: f64, keep_within: f64) -> Option<Self> {
        let to_center = dvec3::sub(center, eye);
        let d = dvec3::length(to_center);
        if !(radius > 0.0 && d > radius) {
            return None;
        }
        let sin = radius / d;
        Some(Self {
            eye,
            axis: dvec3::scale(to_center, 1.0 / d),
            // (d^2 - r^2) / d, factored so a far eye keeps its precision.
            plane: (d - radius) * (d + radius) / d,
            sin,
            cos: libm::sqrt((1.0 - sin * sin).max(0.0)),
            keep_within,
        })
    }

    /// Whether the sphere of `radius` around `center` lies wholly behind the
    /// horizon.
    pub fn hides_sphere(&self, center: DVec3, radius: f64) -> bool {
        let v = dvec3::sub(center, self.eye);
        let along = dvec3::dot(v, self.axis);
        let dist2 = dvec3::dot(v, v);
        if libm::sqrt(dist2) - radius <= self.keep_within || along - radius <= self.plane {
            return false;
        }
        let across = libm::sqrt((dist2 - along * along).max(0.0));
        // Signed distance from the cone's surface, negative inside it.
        across * self.cos - along * self.sin < -radius
    }

    /// Whether the box `lo..hi` lies wholly behind the horizon.
    pub fn hides_box(&self, lo: [f32; 3], hi: [f32; 3]) -> bool {
        let center = core::array::from_fn(|i| 0.5 * (f64::from(lo[i]) + f64::from(hi[i])));
        let half: DVec3 = core::array::from_fn(|i| 0.5 * (f64::from(hi[i]) - f64::from(lo[i])));
        self.hides_sphere(center, dvec3::length(half))
    }
}

/// A planet's lowest ground placed in the simulated frame: the sphere a
/// renderer hides what lies behind, from wherever its camera is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonOccluder {
    /// The planet's center, in the simulated frame.
    pub center: [f32; 3],
    /// The radius of its lowest ground, in meters.
    pub radius: f32,
}

impl HorizonOccluder {
    /// The horizon seen from local point `eye`, keeping everything within
    /// [`HORIZON_SHADOW_REACH`](super::HORIZON_SHADOW_REACH) of it.
    pub fn seen_from(&self, eye: [f32; 3]) -> Option<Horizon> {
        Horizon::new(
            dvec3::from_f32(eye),
            dvec3::from_f32(self.center),
            f64::from(self.radius),
            super::HORIZON_SHADOW_REACH,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 50_000.0;
    const CENTER: DVec3 = [0.0, -R, 0.0];

    // A point `angle` radians round the sphere from its top toward -Z, lifted
    // `height` above it.
    fn round(angle: f64, height: f64) -> DVec3 {
        let r = R + height;
        [0.0, -R + r * libm::cos(angle), -r * libm::sin(angle)]
    }

    fn standing(height: f64) -> Horizon {
        Horizon::new([0.0, height, 0.0], CENTER, R, 0.0).expect("above the ground")
    }

    #[test]
    fn an_eye_on_or_inside_the_sphere_hides_nothing() {
        assert!(Horizon::new([0.0, 0.0, 0.0], CENTER, R, 0.0).is_none());
        assert!(Horizon::new([0.0, -10.0, 0.0], CENTER, R, 0.0).is_none());
        assert!(Horizon::new([0.0, 1.0, 0.0], CENTER, 0.0, 0.0).is_none());
    }

    // The far side of the planet is hidden from anywhere above it; the ground
    // short of the horizon, and anything tall enough to rise over it, is not.
    #[test]
    fn the_far_side_hides_and_the_near_side_shows() {
        let eye = standing(2.0);
        assert!(
            eye.hides_sphere([0.0, -2.0 * R, 0.0], 100.0),
            "the antipode"
        );
        assert!(eye.hides_sphere(round(1.0, 0.0), 500.0), "57 degrees round");
        assert!(!eye.hides_sphere(round(0.004, 0.0), 1.0), "200 m ahead");
        // The horizon from 2 m up lies sqrt(2 R h) = 447 m out; 1 km out the
        // sight line grazing it passes ~3 m over the ground, so a meter-wide
        // stone there is hidden and a tower is not.
        let past = 1_000.0 / R;
        assert!(eye.hides_sphere(round(past, 0.0), 1.0));
        assert!(!eye.hides_sphere(round(past, 30.0), 1.0));
    }

    // Around the grazing point the test follows the sight line: ground a
    // little past it is hidden, ground a little short of it is not.
    #[test]
    fn the_edge_follows_the_grazing_line() {
        let h = 400.0;
        let eye = standing(h);
        let grazing = libm::acos(R / (R + h));
        assert!(eye.hides_sphere(round(grazing * 1.05, 0.0), 0.5));
        assert!(!eye.hides_sphere(round(grazing * 0.95, 0.0), 0.5));
        // Twice as far round, ground hides, but a 1 km peak, whose own
        // horizon reaches another ~10 km, rises over it.
        assert!(eye.hides_sphere(round(grazing * 2.0, 0.0), 1.0));
        assert!(!eye.hides_sphere(round(grazing * 2.0, 1_000.0), 1.0));
    }

    #[test]
    fn nothing_within_the_kept_reach_is_hidden() {
        let far = Horizon::new([0.0, 2.0, 0.0], CENTER, R, 0.0).unwrap();
        let near = Horizon::new([0.0, 2.0, 0.0], CENTER, R, 1_200.0).unwrap();
        let past = round(1_000.0 / R, 0.0);
        assert!(far.hides_sphere(past, 1.0));
        assert!(!near.hides_sphere(past, 1.0));
        assert!(near.hides_sphere(round(1_500.0 / R, 0.0), 1.0));
    }

    #[test]
    fn an_occluder_keeps_the_shadow_reach() {
        let occluder = HorizonOccluder {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
        };
        let eye = occluder.seen_from([0.0, 2.0, 0.0]).unwrap();
        let reach = super::super::HORIZON_SHADOW_REACH;
        assert!(!eye.hides_sphere(round((reach - 10.0) / R, 0.0), 1.0));
        assert!(eye.hides_sphere(round((reach + 900.0) / R, 0.0), 1.0));
        assert!(occluder.seen_from([0.0, -1.0, 0.0]).is_none());
    }

    #[test]
    fn a_box_is_hidden_as_its_bounding_sphere() {
        let eye = standing(2.0);
        let c = round(0.5, 0.0);
        let lo = c.map(|v| (v - 3.0) as f32);
        let hi = c.map(|v| (v + 3.0) as f32);
        assert!(eye.hides_box(lo, hi));
        assert!(!eye.hides_box([-1.0, 0.0, -21.0], [1.0, 2.0, -19.0]));
    }
}
