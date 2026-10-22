use super::*;
use crate::components::Planet;
use crate::render::environment::water::ocean::OceanQuality;
use alloc::vec::Vec;

pub(super) fn coast_planet() -> PlanetShape {
    coast_planet_of(50_000.0)
}

// The coast planet's continents and coasts on a planet `radius` meters round.
pub(super) fn coast_planet_of(radius: f32) -> PlanetShape {
    Planet {
        radius,
        amplitude: 40.0,
        feature_size: 2_000.0,
        octaves: 8,
        seed: 95,
        land_share: 0.4,
        continent_size: 20_000.0,
        ocean_depth: 60.0,
        beach_width: 60.0,
        ..Planet::default()
    }
    .shape()
}

// A 7 m/s sea over 50 km of fetch: Hs about a meter, peaking near 41 m.
pub(super) fn sea_shape() -> SeaShape {
    SeaShape::new(7.0, 50_000.0).unwrap()
}

// No variation: the crests on the contours, the swell at its mean.
pub(super) fn calm(phase: f64) -> ShoreNoise {
    ShoreNoise {
        phase,
        gradient: [0.0; 3],
        swell: 1.0,
    }
}

pub(super) fn shore() -> ShoreWaves {
    let cascades = Cascades::new(sea_shape().peak_wavelength(), OceanQuality::FULL);
    ShoreWaves::new(&coast_planet(), 0.0, &sea_shape(), Some(&cascades)).unwrap()
}

// A direction on the planet's coast and the unit direction along the
// ground toward the land there, found walking out from the top (or, where
// that walk stalls on a ridge of the field, from the next of a few starts).
pub(super) fn coast_point(planet: &PlanetShape) -> (DVec3, DVec3) {
    let starts = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ];
    for start in starts {
        let mut dir = start;
        for _ in 0..1000 {
            let (inland, gradient) = planet.inland(dir).unwrap();
            if inland.abs() < 0.01 {
                return (dir, dvec3::normalize_or(gradient, [1.0, 0.0, 0.0]));
            }
            // A Newton step along the gradient toward the coast.
            let g = dvec3::length(gradient);
            let step = (-inland / g.max(0.05)).clamp(-200.0, 200.0);
            let along = dvec3::scale(gradient, step / (g.max(1e-12) * planet.radius));
            dir = dvec3::normalize_or(dvec3::add(dir, along), dir);
        }
    }
    panic!("no coast found");
}

// The coast's profile is the ground the planet's tiles lay down where no
// hills stand, measured down from the sea.
#[test]
fn the_profile_is_the_planets_coast() {
    let planet = coast_planet();
    let s = shore();
    for inland in [-3000.0, -400.0, -30.0, -0.5, 0.0, 0.5, 12.0, 59.0, 200.0] {
        let want = (-planet.coast_height(inland)).max(0.0);
        assert!((s.coast().depth(inland) - want).abs() < 1e-9, "{inland}");
    }
    assert_eq!(s.coast().waterline, 0.0);
}

// A higher sea meets the beach farther inland, where the beach stands at
// its level; one over the beach has no shore waves.
#[test]
fn a_higher_sea_meets_the_beach_higher_up() {
    let planet = coast_planet();
    let raised = ShoreWaves::new(&planet, 1.2, &sea_shape(), None).unwrap();
    let w = raised.coast().waterline;
    assert!(
        (planet.coast_height(w) - 1.2).abs() < 1e-9 && w > 0.0,
        "{w}"
    );
    assert!(raised.coast().depth(w + 0.1) == 0.0 && raised.coast().depth(w - 0.1) > 0.0);
    assert!(ShoreWaves::new(&planet, 2.5, &sea_shape(), None).is_none());
    let all_land = Planet::default().shape();
    assert!(ShoreWaves::new(&all_land, 0.0, &sea_shape(), None).is_none());
    let dir = [0.0, 1.0, 0.0];
    assert_eq!(shore().at(&all_land, dir, 3.0), ShoreSurface::default());
}

// The trains carry the energy of the sea around its peak between them, at
// frequencies about it.
#[test]
fn the_trains_carry_the_seas_energy() {
    let s = shore();
    let shape = sea_shape();
    let peak = f64::from(shape.peak_omega);
    let variance: f64 = s
        .trains()
        .iter()
        .map(|t| 0.5 * t.amplitude * t.amplitude)
        .sum();
    let want = f64::from(shape.variance_between(narrow(0.6 * peak), narrow(1.6 * peak)));
    assert!((variance - want).abs() < 1e-6 * want, "{variance} {want}");
    let hs = 4.0 * libm::sqrt(variance);
    assert!((0.6..1.4).contains(&hs), "{hs}");
    assert_eq!(s.trains()[0].omega, peak);
    assert!(s.trains().iter().all(|t| t.phase >= 0.0 && t.phase < TAU));
}

// Wherever the coast's distance field runs fast or slow, the crests are
// spaced in meters as the dispersion relation says for the depth under
// them, and the trains' strength depends on that depth alone.
#[test]
fn the_wavelength_follows_the_depth_however_the_field_runs() {
    let s = shore();
    for inland in [-8.0, -35.0, -90.0, -160.0] {
        let depth = s.coast().depth(inland);
        let mut weights = Vec::new();
        for g in [0.5, 1.0, 2.0] {
            let gradient = [g, 0.0, 0.0];
            let a = s.point(inland, gradient);
            // One meter along the sea moves the distance inland `g` meters.
            let b = s.point(inland + g * 0.05, gradient);
            for (i, t) in s.trains().iter().enumerate() {
                let per_meter = (a.phases[i] - b.phases[i]) / 0.05;
                let k = wavenumber(t.omega, depth);
                assert!(
                    (per_meter - k).abs() / k < 0.02,
                    "{inland} {g} {i}: {per_meter} {k}"
                );
            }
            assert_eq!(a.landward, [1.0, 0.0, 0.0]);
            weights.push(a.weight);
        }
        assert!(
            weights.iter().all(|w| (w - weights[0]).abs() < 1e-12),
            "{weights:?}"
        );
    }
}

// Where the distance inland stops changing the direction to the land is
// unknown: the trains fade out there instead of reading noise, and nothing
// turns NaN.
#[test]
fn a_flat_distance_field_fades_the_trains() {
    let s = shore();
    for g in [0.0, 1e-12, 0.1, 0.2] {
        let p = s.point(-40.0, [g, 0.0, 0.0]);
        let surface = s.surface(&p, &calm(0.7), [1.0, 2.0, 3.0]);
        assert!(p.phases.iter().all(|x| x.is_finite()), "{g}");
        assert!(
            surface.height.is_finite() && surface.foam.is_finite(),
            "{g}"
        );
        if g < GRADIENT_FADE[0] {
            assert_eq!(p.weight, 0.0, "{g}");
            assert_eq!(surface, ShoreSurface::default(), "{g}");
        } else {
            assert!(p.weight > 0.0 && p.weight < 1.0, "{g}");
        }
    }
    // The flattest place the planet's field crosses near its coasts reads as
    // calmly.
    let planet = coast_planet();
    let (saddle, g) = (0..20_000)
        .map(|i| {
            let t = f64::from(i) * 0.000_05;
            let dir = dvec3::normalize_or(
                [libm::sin(t * 37.0) * t, 1.0, libm::cos(t * 23.0) * t],
                [0.0, 1.0, 0.0],
            );
            let (inland, gradient) = planet.inland(dir).unwrap();
            (dir, inland, dvec3::length(gradient))
        })
        .filter(|&(_, inland, _)| (-200.0..0.0).contains(&inland))
        .map(|(dir, _, g)| (dir, g))
        .fold(([0.0; 3], f64::MAX), |a, b| if b.1 < a.1 { b } else { a });
    assert!(g < GRADIENT_FADE[1], "{g}");
    let surface = s.at(&planet, saddle, 12.0);
    assert!(surface.height.is_finite() && surface.slope.iter().all(|x| x.is_finite()));
}

// Waves grow as they shoal, never stand taller than the water allows, and
// break into foam in the surf but not out at sea.
#[test]
fn waves_shoal_and_break_in_the_surf() {
    let s = shore();
    let landward = [1.0, 0.0, 0.0];
    let tallest = |inland: f64| {
        let p = s.point(inland, landward);
        let mut most = (0.0_f64, 0.0_f64);
        for step in 0..400 {
            let t = f64::from(step) * 0.25;
            let w = s.surface(&p, &calm(0.0), s.phases_at(t));
            assert!(
                w.height.abs() <= 0.5 * BREAKER_INDEX * p.depth + 1e-9,
                "{inland}"
            );
            most = (most.0.max(w.height), most.1.max(w.foam));
        }
        most
    };
    let (out_height, out_foam) = tallest(-260.0);
    let (mid_height, _) = tallest(-120.0);
    let (_, surf_foam) = tallest(-6.0);
    assert!(out_foam == 0.0, "{out_foam}");
    assert!(mid_height > out_height, "{out_height} {mid_height}");
    assert!(surf_foam > 0.5, "{surf_foam}");
    // Out at sea the deep-water fade leaves the open sea alone.
    let deep = s.point(s.quiet_inland() - 1.0, landward);
    assert_eq!(deep.weight, 0.0);
    assert_eq!(s.cascade_damping(deep.depth), [1.0; MAX_OCEAN_CASCADES]);
}

// A crest pitches forward as it breaks: its face toward the land is steeper
// than its back.
#[test]
fn a_breaking_crest_leans_toward_the_land() {
    let s = shore();
    let p = s.point(-4.0, [1.0, 0.0, 0.0]);
    let (mut front, mut back) = (0.0_f64, 0.0_f64);
    for step in 0..2000 {
        let w = s.surface(&p, &calm(0.0), s.phases_at(f64::from(step) * 0.01));
        front = front.max(-w.slope[0]);
        back = back.max(w.slope[0]);
    }
    assert!(front > 1.3 * back, "{front} {back}");
}

// The cascades fade out in water shallower than half their wavelength,
// the longest first, as the trains take its place.
#[test]
fn the_cascades_fade_in_the_shallows() {
    let s = shore();
    let lengths = s.cascade_wavelengths();
    assert_eq!(lengths[0], s.peak_wavelength());
    assert!(lengths[1] > 0.0 && lengths[1] < lengths[0] && lengths[3] == 0.0);
    assert_eq!(s.cascade_damping(0.0), [0.0, 0.0, 0.0, 1.0]);
    let half = 0.5 * s.peak_wavelength();
    let d = s.cascade_damping(0.75 * half);
    assert!(d[0] > 0.0 && d[0] < 1.0 && d[1] == 1.0, "{d:?}");
    // Where cascade 0 fades, the trains rise in its place.
    let inland = -offshore_where(s.coast(), 0.75 * half);
    let p = s.point(inland, [1.0, 0.0, 0.0]);
    assert!(
        (p.weight + d[0] - 1.0).abs() < 1e-6,
        "{} {}",
        p.weight,
        d[0]
    );
}

// The phase noise is a few radians either way, its gradient matches its
// change along the sea, and it turns the crests only a few degrees.
#[test]
fn the_phase_noise_turns_the_crests_a_little() {
    let s = shore();
    let planet = coast_planet();
    let (dir, landward) = coast_point(&planet);
    let mut steepest = 0.0_f64;
    for i in 0..200 {
        let along = f64::from(i) * 13.0 / s.radius();
        let side = dvec3::cross(dir, landward);
        let d = dvec3::normalize_or(dvec3::add(dir, dvec3::scale(side, along)), dir);
        let ShoreNoise {
            phase: n, gradient, ..
        } = s.noise_at(d);
        assert!(n.abs() <= PHASE_NOISE_RADIANS + 1e-9);
        let step = 0.5;
        let ahead = dvec3::normalize_or(dvec3::add(d, dvec3::scale(side, step / s.radius())), d);
        let fd = (s.noise_at(ahead).phase - n) / step;
        let want = dvec3::dot(gradient, side);
        assert!((fd - want).abs() < 2e-3, "{fd} {want}");
        steepest = steepest.max(dvec3::length(gradient));
    }
    let k = wavenumber(s.trains()[0].omega, 2.0);
    let degrees = libm::atan(steepest / k).to_degrees();
    assert!(degrees > 0.5 && degrees < 15.0, "{degrees}");
}

// The CPU query: at the coast's points the shore waves move with time
// and stay bounded, and a point found on the planet reads what its inland
// distance says.
#[test]
fn the_planet_query_reads_the_shore() {
    let planet = coast_planet();
    let s = shore();
    let (dir, landward) = coast_point(&planet);
    let out = dvec3::normalize_or(
        dvec3::sub(dir, dvec3::scale(landward, 15.0 / planet.radius)),
        dir,
    );
    let a = s.at(&planet, out, 10.0);
    let b = s.at(&planet, out, 11.3);
    assert_ne!(a, b);
    let depth = s.coast().depth(planet.inland(out).unwrap().0);
    assert!(
        depth > 0.3 && a.height.abs() <= 0.5 * BREAKER_INDEX * depth + 1e-9,
        "{depth}"
    );
}

// The swell runs taller and lower along a shore over a few hundred meters,
// within its variation, and a taller stretch breaks farther out.
#[test]
fn the_swell_varies_along_the_shore() {
    let s = shore();
    let planet = coast_planet();
    let (dir, landward) = coast_point(&planet);
    let side = dvec3::cross(dir, landward);
    let swells: Vec<f64> = (0..60)
        .map(|i| {
            let along = f64::from(i) * 50.0 / s.radius();
            s.noise_at(dvec3::normalize_or(
                dvec3::add(dir, dvec3::scale(side, along)),
                dir,
            ))
            .swell
        })
        .collect();
    let (lo, hi) = swells
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    assert!(
        lo >= 1.0 - SWELL_VARIATION && hi <= 1.0 + SWELL_VARIATION,
        "{lo} {hi}"
    );
    assert!(hi - lo > 0.2, "{lo} {hi}");
    let neighbors = swells
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f64::max);
    assert!(neighbors < 0.15, "{neighbors}");
    // The farthest out any wave breaks, for a stretch of swell `swell`.
    let breaks_from = |swell: f64| {
        (0..400)
            .map(|i| -0.5 * f64::from(i))
            .filter(|&inland| {
                let p = s.point(inland, [1.0, 0.0, 0.0]);
                let noise = ShoreNoise { swell, ..calm(0.0) };
                (0..300).any(|t| s.surface(&p, &noise, s.phases_at(f64::from(t) * 0.2)).foam > 0.5)
            })
            .fold(0.0, f64::min)
    };
    assert!(
        breaks_from(1.3) < breaks_from(0.7) - 2.0,
        "{} {}",
        breaks_from(1.3),
        breaks_from(0.7)
    );
}

// The surf's foam rides in with the bore at the share of its speed, `c T`
// per period, then lags and slips back, and is where it formed again as the
// next crest breaks, so the lace runs on unbroken across it.
#[test]
fn the_foam_rides_in_and_slips_back() {
    let k = TAU / 12.0;
    let speed = (foam_drift(1e-6, k) - foam_drift(0.0, k)) / 1e-6;
    assert!((speed - FOAM_RIDE * 12.0).abs() < 1e-3, "{speed}");
    assert_eq!(foam_drift(0.0, k), 0.0);
    assert!(foam_drift(1.0, k).abs() < 1e-12);
    assert!((foam_drift(0.999, k) - foam_drift(0.001, k)).abs() < 0.01);
    let ages: Vec<f64> = (1..100).map(|i| f64::from(i) / 100.0).collect();
    for pair in ages.windows(2) {
        let (a, b) = (foam_drift(pair[0], k), foam_drift(pair[1], k));
        assert!(a > 0.0);
        if pair[1] <= 1.0 / 3.0 {
            assert!(b > a, "{pair:?}");
        } else if pair[0] >= 1.0 / 3.0 {
            assert!(b < a, "{pair:?}");
        }
    }
    // Shorter waves carry it less far.
    assert!(foam_drift(0.3, 2.0 * k) < foam_drift(0.3, k));
}
