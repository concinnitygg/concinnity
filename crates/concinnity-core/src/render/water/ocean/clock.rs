//! Time and place for a simulated ocean, wrapped on the host in f64 so the GPU
//! only sees small numbers. Every wave turns a whole number of times per
//! [`OCEAN_REPEAT_SECONDS`], so the sea at `t` is the sea at `t mod T`, and
//! each cascade tile repeats every `length` meters, so a lookup only needs the
//! anchor's position within one tile.

use core::f64::consts::TAU;

/// The period, in seconds, the simulated sea repeats over: each wave's
/// angular frequency is rounded to a whole number of turns in it.
pub const OCEAN_REPEAT_SECONDS: f64 = 256.0;

/// Whole turns per [`OCEAN_REPEAT_SECONDS`] of a wave of angular frequency
/// `omega` (radians per second): the nearest, and at least one, so no wave
/// stands still.
pub fn omega_steps(omega: f32) -> f32 {
    let turns = f64::from(omega) * OCEAN_REPEAT_SECONDS / TAU;
    libm::round(turns).max(1.0) as f32
}

/// The angular frequency, in radians per second, a wave of `steps` turns
/// per repeat actually runs at.
pub fn stepped_omega(steps: f32) -> f64 {
    f64::from(steps) * TAU / OCEAN_REPEAT_SECONDS
}

/// How far into the repeat period `time` seconds is, in `[0, 1)`. A wave of
/// `steps` turns is then at phase `tau * fract(steps * fraction)`.
pub fn time_fraction(time: f64) -> f32 {
    let f = libm::fmod(time, OCEAN_REPEAT_SECONDS) / OCEAN_REPEAT_SECONDS;
    let f = if f < 0.0 { f + 1.0 } else { f };
    let f = f as f32;
    if f >= 1.0 { 0.0 } else { f }
}

/// Where world `xz` falls within a tile `length` meters across, as texture
/// coordinates in `[0, 1)`.
pub fn tile_offset(xz: [f64; 2], length: f32) -> [f32; 2] {
    xz.map(|v| {
        let t = v / f64::from(length);
        let f = (t - libm::floor(t)) as f32;
        if f >= 1.0 { 0.0 } else { f }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The phase the shader computes from the wrapped fraction is the angle
    // the stepped frequency reaches unwrapped, just after the start and after
    // twelve days.
    #[test]
    fn wrapped_time_gives_the_unwrapped_phase() {
        for omega in [0.33_f32, 1.0, 4.2, 30.0] {
            let steps = omega_steps(omega);
            for time in [0.0, 0.125, 77.7, 1.0e6 + 0.375] {
                let f = time_fraction(time);
                assert!((0.0..1.0).contains(&f));
                let x = steps * f;
                let shader = core::f32::consts::TAU * (x - x.floor());
                let exact = libm::fmod(stepped_omega(steps) * time, TAU);
                let d = (f64::from(shader) - exact).rem_euclid(TAU);
                let err = d.min(TAU - d);
                assert!(err < 2e-3, "{omega} {time}: {err}");
            }
        }
    }

    // Every wave turns a whole number of times per period, the nearest to
    // its own frequency, so the sea repeats exactly.
    #[test]
    fn the_sea_repeats_exactly() {
        for omega in [0.02_f32, 0.33, 1.0, 9.0] {
            let steps = omega_steps(omega);
            assert_eq!(steps, steps.round());
            assert!(steps >= 1.0);
            let quantum = TAU / OCEAN_REPEAT_SECONDS;
            assert!((stepped_omega(steps) - f64::from(omega)).abs() <= quantum / 2.0);
        }
        assert_eq!(
            time_fraction(12.5),
            time_fraction(12.5 + 7.0 * OCEAN_REPEAT_SECONDS)
        );
        assert_eq!(
            time_fraction(-1.0),
            time_fraction(OCEAN_REPEAT_SECONDS - 1.0)
        );
    }

    // A point a few meters from an anchor a thousand kilometers out lands on
    // the tile coordinate its f64 position does.
    #[test]
    fn tile_offsets_stay_precise_far_from_the_origin() {
        let anchor = [1.0e6 + 0.375, -7.5e5 - 0.125];
        let length = 560.0_f32;
        let off = tile_offset(anchor, length);
        for p in [[0.0_f32, 0.0], [3.25, -12.5], [-250.0, 400.0]] {
            for axis in 0..2 {
                let shader = f64::from(p[axis] / length + off[axis]);
                let exact = (anchor[axis] + f64::from(p[axis])) / f64::from(length);
                let d = (shader - exact).rem_euclid(1.0);
                let err = d.min(1.0 - d) * f64::from(length);
                assert!(err < 1e-3, "{p:?} {axis}: {err} m");
            }
        }
        assert!(off.iter().all(|o| (0.0..1.0).contains(o)));
    }
}
