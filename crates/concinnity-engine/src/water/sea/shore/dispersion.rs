//! Linear wave theory over a flat bed: how fast a wave of one frequency
//! travels and carries its energy at a depth, and how its height grows as
//! the water shallows under it.

use crate::water::spectrum::GRAVITY;

const G: f64 = GRAVITY as f64;

/// The wavenumber, in radians per meter, of a wave of angular frequency
/// `omega` in water `depth` meters deep: the root of `omega^2 = g k tanh(k
/// depth)`. Deep water's `omega^2 / g` past half a wavelength of depth.
pub fn wavenumber(omega: f64, depth: f64) -> f64 {
    let deep = omega * omega / G;
    if depth <= 0.0 {
        return f64::INFINITY;
    }
    let x0 = deep * depth;
    if x0 > 20.0 {
        return deep;
    }
    // Newton on f(x) = x tanh(x) - x0 for x = k depth, from Eckart's
    // estimate, which is within a few percent everywhere.
    let mut x = x0 / libm::sqrt(libm::tanh(x0));
    for _ in 0..8 {
        let t = libm::tanh(x);
        let f = x * t - x0;
        let df = t + x * (1.0 - t * t);
        let step = f / df;
        x -= step;
        if step.abs() < 1e-13 * x {
            break;
        }
    }
    x / depth
}

/// The speed, in meters per second, a wave of angular frequency `omega` and
/// wavenumber `k` carries its energy at in water `depth` meters deep.
pub fn group_speed(omega: f64, k: f64, depth: f64) -> f64 {
    let kh = k * depth;
    let n = if kh > 20.0 {
        0.5
    } else {
        0.5 * (1.0 + 2.0 * kh / libm::sinh(2.0 * kh))
    };
    n * omega / k
}

/// How many times taller a wave of angular frequency `omega` stands in water
/// `depth` meters deep than in deep water, its energy flux unchanged:
/// `sqrt(cg_deep / cg)`.
pub fn shoaling(omega: f64, depth: f64) -> f64 {
    shoaling_at(omega, wavenumber(omega, depth), depth)
}

/// [`shoaling`] for a wave whose wavenumber `k` at that depth is known.
pub fn shoaling_at(omega: f64, k: f64, depth: f64) -> f64 {
    libm::sqrt(0.5 * G / omega / group_speed(omega, k, depth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    // The wavenumber solves the dispersion relation, and the textbook case
    // (a 10 s wave in 10 m of water is 92.3 m long) comes out.
    #[test]
    fn the_wavenumber_solves_the_dispersion_relation() {
        for &(period, depth) in &[
            (10.0, 10.0),
            (5.0, 1.0),
            (8.0, 40.0),
            (3.0, 0.2),
            (12.0, 200.0),
        ] {
            let omega = TAU / period;
            let k = wavenumber(omega, depth);
            let residual = omega * omega - G * k * libm::tanh(k * depth);
            assert!(residual.abs() < 1e-9, "{period} s {depth} m: {residual}");
        }
        let length = TAU / wavenumber(TAU / 10.0, 10.0);
        assert!((length - 92.3).abs() < 0.2, "{length}");
    }

    // Shallow water carries every wave at sqrt(g h); deep water at its deep
    // speed, half of it for the energy.
    #[test]
    fn the_limits_match_shallow_and_deep_water() {
        let omega = TAU / 8.0;
        let h = 0.05;
        let k = wavenumber(omega, h);
        assert!((omega / k - libm::sqrt(G * h)).abs() / libm::sqrt(G * h) < 1e-3);
        assert!((group_speed(omega, k, h) - libm::sqrt(G * h)).abs() / libm::sqrt(G * h) < 3e-3);
        let deep = omega * omega / G;
        assert_eq!(wavenumber(omega, 5_000.0), deep);
        assert!((group_speed(omega, deep, 5_000.0) - 0.5 * omega / deep).abs() < 1e-9);
        assert!((shoaling(omega, 5_000.0) - 1.0).abs() < 1e-9);
    }

    // The shoaling coefficient dips to its textbook minimum of 0.913 near
    // k h = 1.2 before it climbs, and in shallow water follows Green's law,
    // a quarter power of the depth.
    #[test]
    fn shoaling_matches_known_values() {
        let omega = TAU / 6.0;
        let (lowest, at) = (1..4000)
            .map(|i| f64::from(i) * 0.01)
            .map(|h| (shoaling(omega, h), h))
            .fold((f64::MAX, 0.0), |a, b| if b.0 < a.0 { b } else { a });
        assert!((lowest - 0.9131).abs() < 1e-3, "{lowest}");
        let kh = wavenumber(omega, at) * at;
        assert!((kh - 1.2).abs() < 0.1, "{kh}");
        let ratio = shoaling(omega, 0.01) / shoaling(omega, 0.16);
        assert!((ratio - 2.0).abs() < 0.01, "{ratio}");
    }

    #[test]
    fn dry_ground_has_no_finite_wavenumber() {
        assert!(wavenumber(1.0, 0.0).is_infinite());
    }
}
