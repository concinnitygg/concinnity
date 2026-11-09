// The wake's math on the CPU, mirroring `water_wake.hlsl` function by
// function, and a small reference simulation built from it that the tests
// run to check the model makes the waves it claims: volume kept, splashes
// that spread, and a Kelvin wake.

use super::{STIR_SPREAD, WAKE_CELL_SIZE};
use crate::render::uniforms::wake::WaterWakeStirGpu;
use alloc::vec;
use alloc::vec::Vec;
use concinnity_core::math::{cos, exp, sin, sqrt};
use core::f32::consts::PI;

/// The share of a stir's volume per square meter at squared distance `d2`
/// from a piece of `radius`: a raised cosine across `STIR_SPREAD` radii, at
/// least two cells wide, whose integral over the plane is 1. Mirrors
/// `wake_disc`.
pub(super) fn disc(d2: f32, radius: f32, cell: f32) -> f32 {
    let r = (STIR_SPREAD * radius).max(2.0 * cell);
    let d = sqrt(d2);
    if d >= r {
        return 0.0;
    }
    (0.5 + 0.5 * cos(PI * d / r)) / (r * r * (0.5 * PI - 2.0 / PI))
}

/// The height `pair` pushes in at `xz`: this frame's volume spread out,
/// less last frame's. Mirrors `wake_stir`.
pub(super) fn stir(pair: &WaterWakeStirGpu, xz: [f32; 2], cell: f32) -> f32 {
    let at = |l: [f32; 4]| {
        let d = [xz[0] - l[0], xz[1] - l[1]];
        l[3] * disc(d[0] * d[0] + d[1] * d[1], l[2], cell)
    };
    at(pair.now) - at(pair.prev)
}

/// One wave of the packed spectrum after `dt` seconds: `z` is the transform
/// of `height + i * rise` at wavenumber `k` and `mirror` at `-k`; returns the
/// new value at `k`. Each wave turns at `w = sqrt(g k)` and loses
/// `damping[0] + damping[1] k^2` of itself per second. Mirrors
/// `wake_propagate`.
pub(super) fn propagate(
    z: [f32; 2],
    mirror: [f32; 2],
    k: f32,
    dt: f32,
    gravity: f32,
    damping: [f32; 2],
) -> [f32; 2] {
    // The height's and the rise's own transforms, from the conjugate pair.
    let h = [0.5 * (z[0] + mirror[0]), 0.5 * (z[1] - mirror[1])];
    let v = [0.5 * (z[1] + mirror[1]), -0.5 * (z[0] - mirror[0])];
    let w = sqrt(gravity * k);
    let (c, s) = (cos(w * dt), sin(w * dt));
    // sin(w dt) / w, which tends to dt as w does.
    let sw = if w > 1e-4 { s / w } else { dt };
    let keep = exp(-(damping[0] + damping[1] * k * k) * dt);
    let h2 = [(h[0] * c + v[0] * sw) * keep, (h[1] * c + v[1] * sw) * keep];
    let v2 = [
        (v[0] * c - h[0] * w * s) * keep,
        (v[1] * c - h[1] * w * s) * keep,
    ];
    [h2[0] - v2[1], h2[1] + v2[0]]
}

/// The share of the field a cell keeps over `dt` seconds `inside` meters in
/// from the window's edge: all of it past the sponge, less and less toward
/// the edge. Mirrors `wake_sponge`.
pub(super) fn sponge(inside: f32, width: f32, rate: f32, dt: f32) -> f32 {
    let t = (1.0 - inside / width).clamp(0.0, 1.0);
    exp(-rate * t * t * dt)
}

/// A periodic patch of `n` cells a side run by the model, on the CPU.
pub(super) struct Reference {
    pub(super) n: usize,
    pub(super) cell: f32,
    pub(super) height: Vec<f32>,
    rise: Vec<f32>,
    work: Vec<[f32; 2]>,
}

impl Reference {
    pub(super) fn new(n: usize, cell: f32) -> Self {
        Self {
            n,
            cell,
            height: vec![0.0; n * n],
            rise: vec![0.0; n * n],
            work: vec![[0.0; 2]; n * n],
        }
    }

    // A cell's center, with the patch centered on the origin.
    pub(super) fn center(&self, x: usize, z: usize) -> [f32; 2] {
        let half = 0.5 * self.n as f32;
        [
            (x as f32 + 0.5 - half) * self.cell,
            (z as f32 + 0.5 - half) * self.cell,
        ]
    }

    /// Push in `pairs`, then run `dt` seconds with `damping`.
    pub(super) fn step(&mut self, pairs: &[WaterWakeStirGpu], dt: f32, damping: [f32; 2]) {
        let n = self.n;
        for z in 0..n {
            for x in 0..n {
                let xz = self.center(x, z);
                let i = x + z * n;
                let push: f32 = pairs.iter().map(|p| stir(p, xz, self.cell)).sum();
                self.height[i] += push;
                self.work[i] = [self.height[i], self.rise[i]];
            }
        }
        fft2(&mut self.work, n, false);
        let src = self.work.clone();
        let dk = 2.0 * PI / (n as f32 * self.cell);
        let signed = |i: usize| {
            if i < n / 2 {
                i as f32
            } else {
                i as f32 - n as f32
            }
        };
        for kz in 0..n {
            for kx in 0..n {
                let k = dk * sqrt(signed(kx) * signed(kx) + signed(kz) * signed(kz));
                let m = (n - kx) % n + ((n - kz) % n) * n;
                self.work[kx + kz * n] = propagate(src[kx + kz * n], src[m], k, dt, 9.81, damping);
            }
        }
        fft2(&mut self.work, n, true);
        for (i, w) in self.work.iter().enumerate() {
            self.height[i] = w[0];
            self.rise[i] = w[1];
        }
    }
}

// An in-place 2D transform of the `n` x `n` row-major `data`, scaled by
// `1 / n^2` on the way back.
fn fft2(data: &mut [[f32; 2]], n: usize, inverse: bool) {
    let mut line = vec![[0.0f32; 2]; n];
    for row in 0..n {
        line.copy_from_slice(&data[row * n..(row + 1) * n]);
        fft(&mut line, inverse);
        data[row * n..(row + 1) * n].copy_from_slice(&line);
    }
    for col in 0..n {
        for (i, l) in line.iter_mut().enumerate() {
            *l = data[col + i * n];
        }
        fft(&mut line, inverse);
        for (i, l) in line.iter().enumerate() {
            data[col + i * n] = *l;
        }
    }
    if inverse {
        let s = 1.0 / (n * n) as f32;
        for d in data.iter_mut() {
            *d = [d[0] * s, d[1] * s];
        }
    }
}

// A radix-2 transform of `a`, whose length is a power of two.
fn fft(a: &mut [[f32; 2]], inverse: bool) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * 2.0 * PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = [cos(angle * k as f32), sin(angle * k as f32)];
                let u = a[start + k];
                let t = a[start + k + len / 2];
                let v = [t[0] * w[0] - t[1] * w[1], t[0] * w[1] + t[1] * w[0]];
                a[start + k] = [u[0] + v[0], u[1] + v[1]];
                a[start + k + len / 2] = [u[0] - v[0], u[1] - v[1]];
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::atan2;

    // The disc spreads exactly the volume it is given, wherever it sits on
    // the cells.
    #[test]
    fn the_disc_spreads_its_whole_volume() {
        for (radius, offset) in [(0.6, 0.0), (0.6, 0.11), (0.1, 0.07), (1.5, 0.2)] {
            let mut sum = 0.0;
            for z in -40..40 {
                for x in -40..40 {
                    let c = [
                        (x as f32 + 0.5) * WAKE_CELL_SIZE - offset,
                        (z as f32 + 0.5) * WAKE_CELL_SIZE,
                    ];
                    sum += disc(c[0] * c[0] + c[1] * c[1], radius, WAKE_CELL_SIZE);
                }
            }
            let area = sum * WAKE_CELL_SIZE * WAKE_CELL_SIZE;
            assert!((area - 1.0).abs() < 0.02, "{radius} {offset}: {area}");
        }
    }

    // A single wave turns at its own frequency, keeps its energy without
    // damping, and loses it as asked with.
    #[test]
    fn a_wave_turns_at_its_own_frequency() {
        let k = 2.0;
        let w = sqrt(9.81 * k);
        let dt = 0.3;
        // A real height wave: its transform at -k is its conjugate.
        let z = [1.0, 0.0];
        let out = propagate(z, z, k, dt, 9.81, [0.0, 0.0]);
        // Height cos(w t), rise -w sin(w t), packed as h + i v.
        assert!((out[0] - cos(w * dt)).abs() < 1e-5);
        assert!((out[1] + w * sin(w * dt)).abs() < 1e-5);
        let damped = propagate(z, z, k, dt, 9.81, [0.1, 0.05]);
        let keep = exp(-(0.1 + 0.05 * k * k) * dt);
        assert!((damped[0] - cos(w * dt) * keep).abs() < 1e-5);
        // Still water at k = 0 rises at its own rate.
        let flat = propagate([0.0, 2.0], [0.0, 2.0], 0.0, 0.5, 9.81, [0.0, 0.0]);
        assert!((flat[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_sponge_eats_only_its_band() {
        assert_eq!(sponge(7.0, 6.0, 6.0, 0.1), 1.0);
        assert!(sponge(3.0, 6.0, 6.0, 0.1) < 1.0);
        assert!(sponge(0.0, 6.0, 6.0, 0.1) < sponge(3.0, 6.0, 6.0, 0.1));
        assert_eq!(sponge(0.0, 6.0, 6.0, 0.0), 1.0);
    }

    fn pair(now: [f32; 4], prev: [f32; 4]) -> WaterWakeStirGpu {
        WaterWakeStirGpu { now, prev }
    }

    // A body dropped in pushes up its volume, which spreads out as rings
    // without the water gaining or losing any.
    #[test]
    fn a_splash_keeps_its_volume_as_it_spreads() {
        let mut sim = Reference::new(128, 0.25);
        let drop = pair([0.0, 0.0, 0.6, 0.3], [0.0, 0.0, 0.6, 0.0]);
        sim.step(&[drop], 1.0 / 30.0, [0.0, 0.0]);
        for _ in 0..90 {
            sim.step(&[], 1.0 / 30.0, [0.0, 0.0]);
        }
        let area = sim.cell * sim.cell;
        let volume: f32 = sim.height.iter().sum::<f32>() * area;
        assert!((volume - 0.3).abs() < 0.01, "{volume}");
        // After three seconds most of the splash has left the middle as rings.
        let (mut near, mut all) = (0.0, 0.0);
        for z in 0..sim.n {
            for x in 0..sim.n {
                let c = sim.center(x, z);
                let e = sim.height[x + z * sim.n].powi(2);
                all += e;
                if c[0] * c[0] + c[1] * c[1] < 1.5 * 1.5 {
                    near += e;
                }
            }
        }
        assert!(near / all < 0.3, "{}", near / all);
    }

    // A body moving through still water draws a Kelvin wake: its waves stay
    // within a narrow wedge behind it whatever its speed, about 19.5 degrees
    // either side of its track.
    #[test]
    fn a_moving_body_draws_a_kelvin_wake() {
        let (n, cell) = (128, 0.25);
        let mut sim = Reference::new(n, cell);
        let speed = 2.5;
        let dt = 1.0 / 30.0;
        let steps = 96;
        let start = -6.0;
        let at = |i: usize| [start + speed * dt * i as f32, 0.0, 0.5, 0.2];
        // Lowered in still, then towed.
        sim.step(&[pair(at(0), [start, 0.0, 0.5, 0.0])], dt, [0.0, 0.0]);
        for _ in 0..60 {
            sim.step(&[], dt, [0.6, 0.0]);
        }
        for i in 1..=steps {
            sim.step(&[pair(at(i), at(i - 1))], dt, [0.0, 0.0]);
        }
        let body = at(steps);
        let mut inside = 0.0;
        let mut outside = 0.0;
        for z in 0..n {
            for x in 0..n {
                let c = sim.center(x, z);
                let behind = body[0] - c[0];
                let d = sqrt(behind * behind + c[1] * c[1]);
                if behind <= 1.5 || d > 9.0 {
                    continue;
                }
                let angle = atan2(c[1].abs(), behind).to_degrees();
                let e = sim.height[x + z * n] * sim.height[x + z * n];
                if angle < 26.0 {
                    inside += e;
                } else {
                    outside += e;
                }
            }
        }
        let share = inside / (inside + outside);
        assert!(share > 0.8, "{share}");
    }
}
