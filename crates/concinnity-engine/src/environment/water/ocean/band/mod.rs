//! A simulated sea's longest waves on the CPU, for whatever floats on it.
//!
//! The band is the first cascade's waves under [`BAND_REACH`] of its
//! wavenumber steps: the same initial spectrum the GPU evolves, at the same
//! whole turns per repeat, so at any time its waves are the drawn ones. Every
//! simulation size from 128 holds all of them, so the band does not depend on
//! the quality preset. It is inverse transformed on a [`BAND_SIZE`] grid into
//! snapshots [`BAND_SNAPSHOT_SECONDS`] apart, each carrying the height, its
//! gradient, the horizontal displacement and their rates of change; a read
//! between two snapshots blends them.

mod fft;
mod sample;

#[cfg(test)]
mod tests;

pub(crate) use sample::BandSample;

use super::OCEAN_CHOPPINESS;
use super::OceanSea;
use super::cascades::Cascades;
use super::clock::OCEAN_REPEAT_SECONDS;
use super::initial::SpectrumBand;
use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, narrow_f32, sin_cos};
use core::f64::consts::TAU;
use fft::{Ifft2, Plane};

/// A complex value, `[re, im]`.
type Complex = [f32; 2];

/// Points along each side of the band's grid.
pub(crate) const BAND_SIZE: usize = 128;

/// The band holds the first cascade's waves whose wavenumber is under this
/// many of its steps.
pub(crate) const BAND_REACH: i32 = 32;

/// Seconds between two snapshots of the band.
pub(crate) const BAND_SNAPSHOT_SECONDS: f64 = 0.1;

// Real fields a snapshot holds per point.
const FIELDS: usize = 8;

/// The fields of one grid point: the height, its gradient along the tile's
/// two axes, the horizontal displacement, and the rates of change of the
/// height and the displacement.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct BandPoint {
    pub(super) height: f32,
    pub(super) gradient: [f32; 2],
    pub(super) shift: [f32; 2],
    pub(super) rise_rate: f32,
    pub(super) shift_rate: [f32; 2],
}

impl BandPoint {
    fn from_fields(f: [f32; FIELDS]) -> Self {
        Self {
            height: f[0],
            gradient: [f[1], f[2]],
            shift: [f[3], f[4]],
            rise_rate: f[5],
            shift_rate: [f[6], f[7]],
        }
    }
}

// One wave of the band: where it lands on the grid, its amplitude and its
// mirror's, and how it turns.
#[derive(Debug, Clone, Copy)]
struct BandWave {
    slot: u32,
    h0: Complex,
    mirror: Complex,
    steps: f32,
    k: [f32; 2],
}

/// The band's grid at one instant.
#[derive(Debug, Clone)]
struct Snapshot {
    index: i64,
    points: Vec<BandPoint>,
}

// The steps a snapshot is built in: the spectra, each grid's columns, each
// grid's rows, then the points.
const BUILD_STAGES: u8 = 10;

// Build stages run per `prepare`: a snapshot interval at the fixed tick
// holds six, so the next snapshot is ready before it is needed.
const STAGES_PER_PREPARE: u8 = 2;

/// A simulated sea's longest waves, and the snapshots around the last time
/// it was made ready for: the two a read blends and the next, built a few
/// stages per [`OceanBand::prepare`] so no one call pays for a whole one.
#[derive(Debug, Clone)]
pub(crate) struct OceanBand {
    length: f32,
    waves: Vec<BandWave>,
    ifft: Ifft2,
    grids: [Plane; 4],
    snapshots: [Snapshot; 3],
    // The slots of the snapshots a read blends, earlier first.
    pair: [usize; 2],
    // The snapshot the grids are partway through, and its next stage.
    building: Option<(i64, u8)>,
}

impl OceanBand {
    /// The band of `sea`'s first cascade among `cascades`; `None` when the
    /// wind raises no sea.
    pub(crate) fn new(sea: &OceanSea, cascades: &Cascades) -> Option<Self> {
        let spectrum = SpectrumBand::new(sea, cascades, 0)?;
        let n = i32::saturating_from(BAND_SIZE);
        let dk = spectrum.dk();
        let mut waves = Vec::new();
        for fy in 1 - BAND_REACH..BAND_REACH {
            for fx in 1 - BAND_REACH..BAND_REACH {
                if fx * fx + fy * fy >= BAND_REACH * BAND_REACH {
                    continue;
                }
                let h0 = spectrum.texel(fx, fy);
                let mirror = spectrum.texel(-fx, -fy);
                if h0[0] == 0.0 && h0[1] == 0.0 && mirror[0] == 0.0 && mirror[1] == 0.0 {
                    continue;
                }
                let slot = fy.rem_euclid(n) * n + fx.rem_euclid(n);
                waves.push(BandWave {
                    slot: u32::saturating_from(slot),
                    h0: [h0[0], h0[1]],
                    mirror: [mirror[0], mirror[1]],
                    steps: h0[2].max(mirror[2]),
                    k: [fx as f32 * dk, fy as f32 * dk],
                });
            }
        }
        let empty = || Snapshot {
            index: i64::MIN,
            points: alloc::vec![BandPoint::default(); BAND_SIZE * BAND_SIZE],
        };
        Some(Self {
            length: cascades.lengths()[0],
            waves,
            ifft: Ifft2::new(BAND_SIZE),
            grids: core::array::from_fn(|_| Plane::new(BAND_SIZE)),
            snapshots: [empty(), empty(), empty()],
            pair: [0, 1],
            building: None,
        })
    }

    /// Make the snapshots around `time` seconds into the world ready, and
    /// build a few more stages of the one after them. A snapshot not ready
    /// in time (the first call, a jump in time) is built whole here.
    pub(crate) fn prepare(&mut self, time: f64) {
        let index = snapshot_index(time);
        let early = self.ready(index);
        let late = self.ready(index + 1);
        self.pair = [early, late];
        let spare = 3 - early - late;
        let next = index + 2;
        if self.snapshots[spare].index == next {
            return;
        }
        if self.building.is_none_or(|(i, _)| i != next) {
            self.building = Some((next, 0));
        }
        for _ in 0..STAGES_PER_PREPARE {
            if self.advance(spare) {
                break;
            }
        }
    }

    /// The band's waves at `time` seconds, at `at` meters along the tile's
    /// two axes. Reads between the snapshots [`OceanBand::prepare`] made;
    /// a time outside them reads the nearer one.
    pub(crate) fn sample(&self, time: f64, at: [f64; 2]) -> BandSample {
        let (a, b, w) = self.blend(time);
        let texel = self.texel();
        let p = at.map(|x| x / texel);
        let sa = sample::read(&a.points, BAND_SIZE, p, texel);
        let sb = sample::read(&b.points, BAND_SIZE, p, texel);
        sa.lerp(&sb, w)
    }

    /// Only the height of [`OceanBand::sample`].
    pub(crate) fn height(&self, time: f64, at: [f64; 2]) -> f32 {
        let (a, b, w) = self.blend(time);
        let texel = self.texel();
        let p = at.map(|x| x / texel);
        let ha = sample::height(&a.points, BAND_SIZE, p, texel);
        let hb = sample::height(&b.points, BAND_SIZE, p, texel);
        ha + (hb - ha) * w
    }

    /// Only the horizontal displacement of [`OceanBand::sample`], which
    /// finding the rest point under a place needs.
    pub(crate) fn shift(&self, time: f64, at: [f64; 2]) -> [f32; 2] {
        let (a, b, w) = self.blend(time);
        let p = at.map(|x| x / self.texel());
        let sa = sample::shift(&a.points, BAND_SIZE, p);
        let sb = sample::shift(&b.points, BAND_SIZE, p);
        [sa[0] + (sb[0] - sa[0]) * w, sa[1] + (sb[1] - sa[1]) * w]
    }

    fn texel(&self) -> f64 {
        f64::from(self.length) / BAND_SIZE as f64
    }

    // The two snapshots a read at `time` blends and the later one's share.
    fn blend(&self, time: f64) -> (&Snapshot, &Snapshot, f32) {
        let a = &self.snapshots[self.pair[0]];
        let b = &self.snapshots[self.pair[1]];
        let t = time / BAND_SNAPSHOT_SECONDS - a.index as f64;
        (a, b, narrow_f32(t.clamp(0.0, 1.0)))
    }

    // The slot holding snapshot `index`, built whole into a free one if no
    // slot does.
    fn ready(&mut self, index: i64) -> usize {
        if let Some(slot) = self.snapshots.iter().position(|s| s.index == index) {
            return slot;
        }
        // Not one the other wanted: slots hold consecutive snapshots, so the
        // one furthest from `index` is free.
        let slot = (0..3)
            .max_by_key(|&s| self.snapshots[s].index.abs_diff(index))
            .unwrap_or(0);
        self.building = Some((index, 0));
        while !self.advance(slot) {}
        slot
    }

    // Run the next stage of the snapshot being built, writing it to `slot`
    // when done; true once it is.
    fn advance(&mut self, slot: usize) -> bool {
        let Some((index, stage)) = self.building else {
            return true;
        };
        let n = BAND_SIZE;
        match stage {
            0 => self.evolve(index),
            1..=4 => {
                let reach = usize::saturating_from(BAND_REACH);
                self.ifft
                    .inverse_columns(&mut self.grids[usize::from(stage - 1)], reach);
            }
            5..=8 => {
                let grid = &mut self.grids[usize::from(stage - 5)];
                self.ifft.transpose(grid);
                self.ifft.inverse_columns(grid, n);
            }
            _ => {
                let [g0, g1, g2, g3] = &self.grids;
                let snapshot = &mut self.snapshots[slot];
                snapshot.index = index;
                // The grids end transposed, and the points keep their order:
                // point (x, y) sits at x * n + y.
                for (i, point) in snapshot.points.iter_mut().enumerate() {
                    *point = BandPoint::from_fields([
                        g0.re[i], g1.re[i], g1.im[i], g2.re[i], g2.im[i], g0.im[i], g3.re[i],
                        g3.im[i],
                    ]);
                }
                self.building = None;
                return true;
            }
        }
        self.building = Some((index, stage + 1));
        stage + 1 >= BUILD_STAGES
    }

    // Snapshot `index`'s spectra into the grids.
    fn evolve(&mut self, index: i64) {
        let time = index as f64 * BAND_SNAPSHOT_SECONDS;
        for grid in &mut self.grids {
            grid.clear();
        }
        let fraction = libm::fmod(time, OCEAN_REPEAT_SECONDS) / OCEAN_REPEAT_SECONDS;
        let fraction = if fraction < 0.0 {
            fraction + 1.0
        } else {
            fraction
        };
        let [g0, g1, g2, g3] = &mut self.grids;
        for w in &self.waves {
            let x = f64::from(w.steps) * fraction;
            let phase = TAU * (x - libm::floor(x));
            let (sin, cos) = sin_cos(narrow_f32(phase));
            let e = [cos, sin];
            let omega = narrow_f32(f64::from(w.steps) * TAU / OCEAN_REPEAT_SECONDS);
            let a = mul(w.h0, [e[0], -e[1]]);
            let b = mul([w.mirror[0], -w.mirror[1]], e);
            let h = [a[0] + b[0], a[1] + b[1]];
            // d/dt of `a` is `-i w a`, of `b` is `i w b`.
            let ht = scale(times_i([b[0] - a[0], b[1] - a[1]]), omega);
            let kl = libm::hypotf(w.k[0], w.k[1]);
            let kn = [w.k[0] / kl, w.k[1] / kl];
            let ih = times_i(h);
            let iht = times_i(ht);
            let s = w.slot as usize;
            let chop = OCEAN_CHOPPINESS;
            put(g0, s, pack(h, ht));
            put(g1, s, pack(scale(ih, w.k[0]), scale(ih, w.k[1])));
            put(
                g2,
                s,
                pack(scale(ih, chop * kn[0]), scale(ih, chop * kn[1])),
            );
            put(
                g3,
                s,
                pack(scale(iht, chop * kn[0]), scale(iht, chop * kn[1])),
            );
        }
    }
}

// The snapshot at or before `time`.
fn snapshot_index(time: f64) -> i64 {
    i64::saturating_from(libm::floor(time / BAND_SNAPSHOT_SECONDS))
}

fn mul(a: Complex, b: Complex) -> Complex {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn put(grid: &mut Plane, at: usize, v: Complex) {
    grid.re[at] += v[0];
    grid.im[at] += v[1];
}

fn scale(a: Complex, s: f32) -> Complex {
    [a[0] * s, a[1] * s]
}

fn times_i(z: Complex) -> Complex {
    [-z[1], z[0]]
}

// Two real fields' spectra as one complex transform's: `a + i b`.
fn pack(a: Complex, b: Complex) -> Complex {
    [a[0] - b[1], a[1] + b[0]]
}
