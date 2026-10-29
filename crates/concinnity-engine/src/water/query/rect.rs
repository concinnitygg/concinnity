//! A rectangle of water read on the CPU the way `water.hlsl` draws it: its
//! displacing waves summed, each a vertical sinusoid with its horizontal
//! pinch, plus a simulated sea's longest waves where it has one. Read in the
//! simulated frame, as it is drawn.

use super::{PatchShift, WaterPatch, WaterSample, unit};
use crate::water::ocean::band::OceanBand;
use crate::water::{BodyWave, WaterBody};
use concinnity_core::math::narrow_f32;

// How many times the rest point under a place is refined against the
// horizontal displacement.
const REST_STEPS: usize = 1;

// The waves over one rest point.
#[derive(Debug, Clone, Copy, Default)]
struct Waves {
    height: f64,
    shift: [f64; 2],
    slope: [f64; 2],
    velocity: [f64; 3],
}

// The rectangle at point `at`, and the horizontal displacement of the water
// that moved there; `None` outside it or under its floor.
pub(super) fn sample(
    body: &WaterBody,
    band: Option<&OceanBand>,
    at: [f32; 3],
    time: f64,
) -> Option<(WaterSample, [f64; 2])> {
    let s = &body.surface;
    if !inside(body, at) {
        return None;
    }
    let place = [f64::from(at[0]), f64::from(at[2])];
    let waves = &body.waves[..body.displacing];
    let mut rest = place;
    for _ in 0..REST_STEPS {
        let w = read(waves, band, rest, time, false);
        rest = [place[0] - w.shift[0], place[1] - w.shift[1]];
    }
    let w = read(waves, band, rest, time, true);
    let up = [0.0, 1.0, 0.0];
    let sample = WaterSample {
        depth: narrow_f32(f64::from(s.center[1]) + w.height - f64::from(at[1])),
        up,
        normal: unit([narrow_f32(-w.slope[0]), 1.0, narrow_f32(-w.slope[1])], up),
        velocity: w.velocity.map(narrow_f32),
        density: s.density,
    };
    Some((sample, w.shift))
}

// How far the rectangle's surface lies above point `at`, the water there
// moved by `patch`'s displacement; `None` outside it or under its floor.
pub(super) fn depth(
    body: &WaterBody,
    band: Option<&OceanBand>,
    patch: &WaterPatch,
    at: [f32; 3],
    time: f64,
) -> Option<f32> {
    let PatchShift::Flat(shift) = patch.shift else {
        return None;
    };
    if !inside(body, at) {
        return None;
    }
    let rest = [f64::from(at[0]) - shift[0], f64::from(at[2]) - shift[1]];
    let w = read(&body.waves[..body.displacing], band, rest, time, false);
    Some(narrow_f32(
        f64::from(body.surface.center[1]) + w.height - f64::from(at[1]),
    ))
}

// Whether `at` lies over the rectangle and above its floor.
fn inside(body: &WaterBody, at: [f32; 3]) -> bool {
    let s = &body.surface;
    at[1] >= s.center[1] - s.depth_meters
        && (0..2).all(|a| (at[2 * a] - s.center[2 * a]).abs() <= s.extent[a])
}

// The waves over rest point `xz`; `full` adds the slope and the velocity.
fn read(
    waves: &[BodyWave],
    band: Option<&OceanBand>,
    xz: [f64; 2],
    time: f64,
    full: bool,
) -> Waves {
    let mut out = Waves::default();
    for w in waves {
        let wave = &w.wave;
        let k = f64::from(wave.wavenumber());
        let omega = f64::from(wave.angular_speed());
        let d = [f64::from(wave.direction[0]), f64::from(wave.direction[1])];
        let phase = k * (d[0] * xz[0] + d[1] * xz[1]) + f64::from(wave.phase) - omega * time;
        let (s, c) = (libm::sin(phase), libm::cos(phase));
        let amp = f64::from(wave.amplitude);
        let sway = f64::from(w.pinch) * amp;
        out.height += amp * s;
        out.shift[0] += d[0] * sway * c;
        out.shift[1] += d[1] * sway * c;
        if full {
            out.slope[0] += amp * k * c * d[0];
            out.slope[1] += amp * k * c * d[1];
            out.velocity[0] += d[0] * sway * omega * s;
            out.velocity[1] -= amp * omega * c;
            out.velocity[2] += d[1] * sway * omega * s;
        }
    }
    if let Some(band) = band {
        out.height += f64::from(band.height(time, xz));
        if full {
            let b = band.sample(time, xz);
            out.shift[0] += f64::from(b.shift[0]);
            out.shift[1] += f64::from(b.shift[1]);
            out.slope[0] += f64::from(b.slope[0]);
            out.slope[1] += f64::from(b.slope[1]);
            out.velocity[0] += f64::from(b.shift_rate[0]);
            out.velocity[1] += f64::from(b.rise_rate);
            out.velocity[2] += f64::from(b.shift_rate[1]);
        } else {
            let shift = band.shift(time, xz);
            out.shift[0] += f64::from(shift[0]);
            out.shift[1] += f64::from(shift[1]);
        }
    }
    out
}
