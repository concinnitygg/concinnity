//! The wash of each wave up the beach, and the wet sand it leaves. As a bore
//! reaches the waterline its water runs on up the sand, slowing to a stop and
//! draining back over about one wave period; the bigger waves of a set run
//! the farthest. Where the water has just been the sand keeps a glossy film
//! for a few seconds and stays dark for longer, drying back down the beach
//! between sets. Run-up heights follow Stockdon et al. (2006) for the beach's
//! slope at the waterline and the sea's height and length.

use super::{SHORE_TRAINS, SWELL_VARIATION, ShoreNoise, ShoreWaves, smoothstep, wrap_phase};
use crate::terrain::GROUND_RULE_WANDER;
use crate::water::spectrum::GRAVITY;
use concinnity_core::math::SaturatingFrom;
use core::f64::consts::TAU;

/// How long, in seconds, a point of the beach remembers the water that
/// reached it: the wetness the last wash left fades out over its last quarter.
pub const SWASH_MEMORY: f64 = 3.0 * DRY_BACK;

/// Most wave cycles back a point looks for that water, so a short-period sea
/// remembers fewer seconds.
pub const SWASH_MAX_CYCLES: u32 = 16;

/// How long the glossy film the wash leaves takes to drain by e, in seconds.
pub const FILM_DRAIN: f64 = 2.5;

/// How long the sand takes to dry back by e, in seconds.
pub const DRY_BACK: f64 = 12.0;

/// How tall the foam line behind the running front stands, in meters of rise
/// up the beach.
pub const SWASH_EDGE: f64 = 0.03;

/// How far, in meters of rise, the thinning tip of each wash wets the sand
/// softly above where it stops.
pub const SWASH_SOFT: f64 = 0.05;

/// How far the sand's small hollows and humps, in meters, shift where the
/// wash stops along a beach.
pub const SWASH_RELIEF: f64 = 0.06;

/// How much farther or shorter the wash runs in its lobes along a beach, as a
/// share of the run-up.
pub const SWASH_LOBES: f64 = 0.4;

/// The share of the ground rules' edge-noise range at which the sand's
/// relief reaches about two thirds of its height (the noise mostly stays
/// within two fifths of its range).
pub const RELIEF_SPREAD: f64 = 0.3;

/// How deep, in meters, the sea floor below the waterline still reads wet.
pub const SEABED_WET: f64 = 1.0;

/// How wet the sand at the waterline stays between waves, fading to dry at
/// the highest run-up.
pub const SATURATED: f64 = 0.5;

/// How much less wet the wash leaves the sand at the top of its run-up than
/// at the water, so the wet sand lightens up the beach.
pub const RUNUP_LIGHTER: f64 = 0.4;

/// The wash at one point of the beach and instant.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Swash {
    /// How much of a running or draining film of water covers the sand, in
    /// `[0, 1]`.
    pub film: f64,
    /// How wet, and so dark, the sand is, in `[0, 1]`.
    pub wet: f64,
    /// How much of the foam line at the running front lies here, in `[0, 1]`.
    pub edge: f64,
}

/// The vertical height, in meters, two waves in a hundred run up a beach of
/// slope `slope` from a sea of significant height `height` and deep-water
/// wavelength `wavelength` (Stockdon et al. 2006: setup plus half the swash).
pub fn stockdon_runup(slope: f64, height: f64, wavelength: f64) -> f64 {
    let hl = (height * wavelength).max(0.0);
    let setup = 0.35 * slope * libm::sqrt(hl);
    let swash = libm::sqrt(hl * (0.563 * slope * slope + 0.004));
    1.1 * (setup + 0.5 * swash)
}

/// The sand's relief in `(-1, 1)` from the ground rules' edge noise `edge`
/// there: it saturates smoothly, never flattening at a bound, so the line the
/// wash stops at stays a smooth curve.
pub fn sand_relief(edge: f64) -> f64 {
    let r = edge / (RELIEF_SPREAD * f64::from(GROUND_RULE_WANDER));
    r / libm::sqrt(1.0 + r * r)
}

impl ShoreWaves {
    /// The beach's slope at the waterline.
    pub fn beach_slope(&self) -> f64 {
        let c = self.coast();
        let t = (c.waterline / c.beach_width).min(1.0);
        2.0 * c.beach_rise / c.beach_width * (1.0 - t)
    }

    /// The vertical height the biggest waves of a set run up the beach to, in
    /// meters above the sea.
    pub fn runup(&self) -> f64 {
        let variance: f64 = self
            .trains()
            .iter()
            .map(|t| 0.5 * t.amplitude * t.amplitude)
            .sum();
        let height = 4.0 * libm::sqrt(variance);
        let omega = self.trains()[0].omega;
        stockdon_runup(
            self.beach_slope(),
            height,
            TAU * f64::from(GRAVITY) / (omega * omega),
        )
    }

    /// How many wave cycles of period `period` a point of the beach looks
    /// back over, and how many seconds that covers.
    pub fn swash_window(period: f64) -> (u32, f64) {
        let cycles =
            u32::saturating_from(libm::ceil(SWASH_MEMORY / period)).clamp(1, SWASH_MAX_CYCLES);
        (cycles, SWASH_MEMORY.min(f64::from(cycles) * period))
    }

    /// The highest rise, in meters above the sea, the wash can wet anywhere
    /// along the shore, before the sand's relief.
    pub fn swash_reach(&self) -> f64 {
        self.runup() * libm::sqrt(1.0 + SWELL_VARIATION) * (1.0 + SWASH_LOBES) + SWASH_SOFT
    }

    /// The wash `height` meters above the sea's rest up the beach, the
    /// shore's slow variation there `noise` (its phase times the waves, its
    /// swell scales how far they run), the trains at waterline phases
    /// `phases`; the sand's `relief` there, in `[-1, 1]`, raises it a little
    /// and draws the wash out into lobes, so the line it stops at wanders.
    pub fn swash_at(
        &self,
        height: f64,
        noise: &ShoreNoise,
        phases: [f64; SHORE_TRAINS],
        relief: f64,
    ) -> Swash {
        let height = height - SWASH_RELIEF * relief;
        if height <= -SEABED_WET {
            return Swash::default();
        }
        if height <= 0.0 {
            // The sea floor just off the waterline, still wet from the troughs.
            let wet = 1.0 - smoothstep(0.0, SEABED_WET, -height);
            return Swash {
                film: wet,
                wet,
                edge: 0.0,
            };
        }
        let trains = self.trains();
        let omega = trains[0].omega;
        let period = TAU / omega;
        let total: f64 = trains.iter().map(|t| t.amplitude).sum();
        // Run-up grows with the square root of the waves' height.
        let runup = self.runup() * libm::sqrt(noise.swell) * (1.0 + SWASH_LOBES * relief);
        let noise = noise.phase;
        if total <= 0.0 || height >= runup + SWASH_SOFT {
            return Swash::default();
        }
        let (cycles, window) = Self::swash_window(period);
        // Seconds since the peak train's last crest reached the waterline.
        let since = wrap_phase(phases[0] + noise) / omega;
        // The sand low on the beach stays damp between waves, and the wash
        // soaks it more than the sand it barely reaches.
        let up = smoothstep(0.0, runup, height);
        let soak = 1.0 - RUNUP_LIGHTER * up;
        let mut out = Swash {
            wet: SATURATED * (1.0 - up),
            ..Swash::default()
        };
        for j in 0..cycles {
            let ago = since + f64::from(j) * period;
            let (mut re, mut im) = (0.0, 0.0);
            for (t, &p) in trains.iter().zip(&phases) {
                let at = p + noise - t.omega * ago;
                re += t.amplitude * libm::cos(at);
                im += t.amplitude * libm::sin(at);
            }
            let reach = runup * libm::hypot(re, im) / total;
            if reach + SWASH_SOFT <= height {
                continue;
            }
            // The front climbs `reach 4 s (1 - s)` over the cycle's period;
            // above its top the thin tip touches the sand at the turn.
            let q = libm::sqrt((1.0 - height / reach).max(0.0));
            let (wetted, left) = (0.5 * (1.0 - q) * period, 0.5 * (1.0 + q) * period);
            if ago < wetted {
                continue;
            }
            let tip = smoothstep(-SWASH_SOFT, SWASH_SOFT, reach - height);
            let kept = tip * soak * (1.0 - smoothstep(0.75 * window, window, ago));
            if ago <= left {
                out.film = out.film.max(tip);
                out.wet = out.wet.max(kept);
            } else {
                let dry = ago - left;
                out.film = out.film.max(tip * libm::exp(-dry / FILM_DRAIN));
                out.wet = out.wet.max(kept * libm::exp(-dry / DRY_BACK));
            }
            if j == 0 {
                // Foam rides just behind the front while it climbs.
                let s = ago / period;
                let front = reach * 4.0 * s * (1.0 - s);
                let behind = front - height;
                if behind >= 0.0 {
                    out.edge = tip
                        * (1.0 - smoothstep(0.35, 0.5, s))
                        * (1.0 - smoothstep(0.0, SWASH_EDGE, behind));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{calm, coast_planet, shore};
    use super::super::{PHASE_NOISE_OCTAVES, SeaShape};
    use super::*;
    use crate::render::uniforms::canopy::ShoreSwashGpu;
    use alloc::format;
    use concinnity_core::math::narrow_f32;

    // The textbook beach: a 1 m sea 41 m long on a 0.08 beach runs up half
    // a meter, and a steeper beach runs it farther.
    #[test]
    fn the_runup_matches_stockdon() {
        let r = stockdon_runup(0.08, 1.0, 41.0);
        let setup = 0.35 * 0.08 * libm::sqrt(41.0);
        let swash = libm::sqrt(41.0 * (0.563 * 0.0064 + 0.004));
        assert!((r - 1.1 * (setup + 0.5 * swash)).abs() < 1e-12);
        assert!((r - 0.504).abs() < 0.005, "{r}");
        assert!(stockdon_runup(0.15, 1.0, 41.0) > r);
        assert_eq!(stockdon_runup(0.08, 0.0, 41.0), 0.0);
        let s = shore();
        assert!((s.beach_slope() - 0.08).abs() < 1e-12);
        assert!((0.3..0.8).contains(&s.runup()), "{}", s.runup());
    }

    // The relief follows the edge noise smoothly over its whole range: it
    // keeps rising, never sitting flat at a bound where the wash's line
    // would run straight and turn a corner.
    #[test]
    fn the_relief_never_flattens() {
        let wander = f64::from(GROUND_RULE_WANDER);
        let mut last = sand_relief(-wander);
        assert!(last > -1.0);
        for step in 1..=400 {
            let edge = -wander + 2.0 * wander * f64::from(step) / 400.0;
            let r = sand_relief(edge);
            let slope = (r - last) / (2.0 * wander / 400.0);
            assert!(slope > 0.03, "{edge}: {slope}");
            assert!(r.abs() < 1.0);
            last = r;
        }
        assert_eq!(sand_relief(0.0), 0.0);
        assert!(sand_relief(0.4 * wander) > 0.75);
    }

    // The sand's relief moves where the wash stops: a hollow is reached
    // where a hump beside it stays dry.
    #[test]
    fn the_relief_draws_the_wash_into_lobes() {
        let s = shore();
        let height = 0.97 * s.runup();
        let reached = |relief: f64| {
            (0..3000).any(|step| {
                s.swash_at(
                    height,
                    &calm(0.0),
                    s.phases_at(f64::from(step) * 0.1),
                    relief,
                )
                .film
                    == 1.0
            })
        };
        assert!(reached(1.0) && !reached(-1.0));
    }

    // Below the sea the sand is always wet, above the highest run-up never,
    // and between, the wash comes and goes: every point it reaches is
    // covered some of the time, stays wet at least as long as glossy, and
    // dries back between the waves.
    #[test]
    fn the_wash_wets_the_beach_and_dries_back() {
        let s = shore();
        let runup = s.runup();
        assert!(s.swash_at(-0.2, &calm(0.0), [0.0; SHORE_TRAINS], 0.0).wet > 0.85);
        let above = s.swash_at(runup + 0.01, &calm(0.0), s.phases_at(3.0), 0.0);
        assert_eq!(above, Swash::default());
        for share in [0.1, 0.4, 0.7] {
            let height = share * runup;
            let (mut covered, mut damp, mut dry) = (0, 0, 0);
            for step in 0..3000 {
                let w = s.swash_at(height, &calm(0.4), s.phases_at(f64::from(step) * 0.1), 0.0);
                assert!(w.wet <= 1.0 && w.film <= 1.0 && w.edge <= w.film + 1e-12);
                if w.film == 1.0 {
                    covered += 1;
                } else if w.wet > 0.3 {
                    damp += 1;
                } else {
                    dry += 1;
                }
            }
            assert!(covered > 0 && damp > 0, "{share}: {covered} {damp} {dry}");
            if share < 0.2 {
                assert!(covered > dry, "{share}: {covered} {dry}");
            }
        }
    }

    // The wash runs up and back once per wave: at a fixed height on the
    // beach the film arrives with each run-up, and the sand darkens as
    // smoothly as it dries, never jumping but when water arrives, however
    // short the sea's period.
    #[test]
    fn the_sand_dries_smoothly() {
        let short = ShoreWaves::new(
            &coast_planet(),
            0.0,
            &SeaShape::new(3.0, 50_000.0).unwrap(),
            None,
        )
        .unwrap();
        for s in [shore(), short] {
            let period = TAU / s.trains()[0].omega;
            let height = 0.3 * s.runup();
            let mut last = s.swash_at(height, &calm(0.0), s.phases_at(0.0), 0.0);
            let mut arrivals = 0;
            for step in 1..6000 {
                let w = s.swash_at(height, &calm(0.0), s.phases_at(f64::from(step) * 0.02), 0.0);
                if w.film > last.film + 0.05 {
                    arrivals += 1;
                } else {
                    assert!(
                        w.wet >= last.wet - 0.02,
                        "{period} {step}: {} -> {}",
                        last.wet,
                        w.wet
                    );
                }
                last = w;
            }
            let waves = 120.0 / period;
            assert!(
                f64::from(arrivals) > 0.3 * waves && f64::from(arrivals) < 1.5 * waves,
                "{period}: {arrivals} {waves}"
            );
        }
        let (cycles, window) = ShoreWaves::swash_window(1.0);
        assert_eq!(
            (cycles, window),
            (SWASH_MAX_CYCLES, f64::from(SWASH_MAX_CYCLES))
        );
        assert_eq!(ShoreWaves::swash_window(10.0), (4, SWASH_MEMORY));
    }

    // The sand is wettest at the water and dries up the beach: averaged over
    // several sets, wetness falls with height, and the sea floor reads wet
    // only just off the waterline.
    #[test]
    fn the_wet_sand_fades_up_the_beach() {
        let s = shore();
        let runup = s.runup();
        let mean = |height: f64| {
            (0..2000)
                .map(|step| {
                    s.swash_at(height, &calm(0.0), s.phases_at(f64::from(step) * 0.17), 0.0)
                        .wet
                })
                .sum::<f64>()
                / 2000.0
        };
        let heights = [0.02, 0.2, 0.4, 0.6, 0.8, 0.95].map(|share| share * runup);
        let wet = heights.map(mean);
        for pair in wet.windows(2) {
            assert!(pair[1] < pair[0], "{wet:?}");
        }
        assert!(wet[0] > 0.8 && wet[5] < 0.3, "{wet:?}");
        let under = |h: f64| s.swash_at(h, &calm(0.0), [0.0; SHORE_TRAINS], 0.0);
        assert_eq!(under(-0.0001).wet, under(-0.0001).film);
        assert!(under(-0.0001).wet > 0.99 && under(-0.5).wet > 0.3);
        assert_eq!(under(-SEABED_WET - 0.01), Swash::default());
        assert_eq!(under(-30.0), Swash::default());
    }

    // The front's foam rides only on water: wherever it shows, the sand is
    // covered, and it shows only while the front climbs.
    #[test]
    fn the_foam_line_leads_the_wash() {
        let s = shore();
        let period = TAU / s.trains()[0].omega;
        let mut seen = 0;
        for share in [0.1, 0.3, 0.6] {
            for step in 0..4000 {
                let time = f64::from(step) * 0.05;
                let w = s.swash_at(share * s.runup(), &calm(0.0), s.phases_at(time), 0.0);
                if w.edge > 0.0 {
                    seen += 1;
                    assert!(w.film >= w.edge - 1e-12, "{share} {time}: {w:?}");
                    let since = wrap_phase(s.phases_at(time)[0]) / s.trains()[0].omega;
                    assert!(since < 0.5 * period, "{share} {time}");
                }
            }
        }
        assert!(seen > 0);
    }

    // `ground_swash_at` and `ground_swash` in canopy.hlsl, past the noise, in
    // f32 from the block the ground reads: the early-out first, then the wash.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the shader's (uint)ceil(x)"
    )]
    fn shader_swash(b: &ShoreSwashGpu, ground: f32, relief: f32, noise: [f32; 2]) -> [f32; 3] {
        let none = [0.0; 3];
        let mut height = ground - b.sea_level;
        let r = narrow_f32(SWASH_RELIEF);
        if b.runup <= 0.0 || height - r >= b.reach || height + r <= -narrow_f32(SEABED_WET) {
            return none;
        }
        let [noise, swell] = noise;
        let smooth = |a: f32, b: f32, x: f32| {
            let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        height -= r * relief;
        if height <= -narrow_f32(SEABED_WET) {
            return none;
        }
        if height <= 0.0 {
            let wet = 1.0 - smooth(0.0, narrow_f32(SEABED_WET), -height);
            return [wet, wet, 0.0];
        }
        let tau = core::f32::consts::TAU;
        let omega = b.omegas[0];
        let period = tau / omega;
        let total = b.amplitudes[0] + b.amplitudes[1] + b.amplitudes[2];
        let runup = b.runup * swell.sqrt() * (1.0 + narrow_f32(SWASH_LOBES) * relief);
        let soft = narrow_f32(SWASH_SOFT);
        if total <= 0.0 || height >= runup + soft {
            return none;
        }
        let memory = narrow_f32(SWASH_MEMORY);
        let cycles = ((memory / period).ceil() as u32).clamp(1, SWASH_MAX_CYCLES);
        let window = memory.min(cycles as f32 * period);
        let mut since = b.phases[0] + noise;
        since = (since - tau * (since / tau).floor()) / omega;
        let up = smooth(0.0, runup, height);
        let soak = 1.0 - narrow_f32(RUNUP_LIGHTER) * up;
        let mut w = [0.0, narrow_f32(SATURATED) * (1.0 - up), 0.0];
        for j in 0..cycles {
            let ago = since + j as f32 * period;
            let (mut re, mut im) = (0.0_f32, 0.0_f32);
            for i in 0..3 {
                let a = b.phases[i] + noise - b.omegas[i] * ago;
                re += b.amplitudes[i] * a.cos();
                im += b.amplitudes[i] * a.sin();
            }
            let reach = runup * (re * re + im * im).sqrt() / total;
            if reach + soft <= height {
                continue;
            }
            let q = (1.0 - height / reach).max(0.0).sqrt();
            let (wetted, left) = (0.5 * (1.0 - q) * period, 0.5 * (1.0 + q) * period);
            if ago < wetted {
                continue;
            }
            let tip = smooth(-soft, soft, reach - height);
            let kept = tip * soak * (1.0 - smooth(0.75 * window, window, ago));
            if ago <= left {
                w[0] = w[0].max(tip);
                w[1] = w[1].max(kept);
            } else {
                let dry = ago - left;
                w[0] = w[0].max(tip * (-dry / narrow_f32(FILM_DRAIN)).exp());
                w[1] = w[1].max(kept * (-dry / narrow_f32(DRY_BACK)).exp());
            }
            if j == 0 {
                let s = ago / period;
                let behind = reach * 4.0 * s * (1.0 - s) - height;
                if behind >= 0.0 {
                    w[2] = tip
                        * (1.0 - smooth(0.35, 0.5, s))
                        * (1.0 - smooth(0.0, narrow_f32(SWASH_EDGE), behind));
                }
            }
        }
        w
    }

    // The ground's shader reads the host's wash from the canopy block: at
    // heights from the sea floor to above the highest run-up and moments
    // through several sets, film, wetness and the front's foam match, its
    // early-out keeps every point the wash reaches, and its constants are
    // the host's.
    #[test]
    fn the_ground_reads_the_hosts_wash() {
        let s = shore();
        let src = crate::render::shader::embedded::CANOPY;
        for line in [
            format!("static const float GROUND_SWASH_MEMORY = {SWASH_MEMORY:.1};"),
            format!("static const uint GROUND_SWASH_MAX_CYCLES = {SWASH_MAX_CYCLES}u;"),
            format!("static const float GROUND_FILM_DRAIN = {FILM_DRAIN};"),
            format!("static const float GROUND_DRY_BACK = {DRY_BACK:.1};"),
            format!("static const float GROUND_SWASH_EDGE = {SWASH_EDGE};"),
            format!("static const float GROUND_SWASH_SOFT = {SWASH_SOFT};"),
            format!("static const float GROUND_SWASH_RELIEF = {SWASH_RELIEF};"),
            format!("static const float GROUND_SWASH_LOBES = {SWASH_LOBES};"),
            format!("static const float GROUND_SEABED_WET = {SEABED_WET:.1};"),
            format!("static const float GROUND_SATURATED = {SATURATED};"),
            format!("static const float GROUND_RUNUP_LIGHTER = {RUNUP_LIGHTER};"),
            format!("static const float GROUND_RELIEF_SPREAD = {RELIEF_SPREAD};"),
            format!(
                "static const float GROUND_PHASE_NOISE_WEIGHT = {};",
                crate::planet::octave_weight_sum(PHASE_NOISE_OCTAVES)
            ),
        ] {
            assert!(src.contains(&line), "{line}");
        }
        let mut mismatched = 0;
        let mut checked = 0;
        for step in 0..400 {
            let time = 1_000.0 + f64::from(step) * 0.37;
            let block = ShoreSwashGpu::new(&s, time);
            assert_eq!(block.runup, narrow_f32(s.runup()));
            assert_eq!(block.reach, narrow_f32(s.swash_reach()));
            for share in [
                -2.0, -1.1, -0.9, -0.1, 0.05, 0.2, 0.45, 0.8, 1.1, 1.5, 1.9, 3.0,
            ] {
                let height = share * s.runup();
                let relief = 1.0 - 0.005 * f64::from(step);
                for swell in [1.0 - SWELL_VARIATION, 1.0, 1.0 + SWELL_VARIATION] {
                    let noise = ShoreNoise { swell, ..calm(1.3) };
                    let want = s.swash_at(height, &noise, s.phases_at(time), relief);
                    let got = shader_swash(
                        &block,
                        narrow_f32(height + s.coast().sea_level),
                        narrow_f32(relief),
                        [narrow_f32(noise.phase), narrow_f32(noise.swell)],
                    );
                    checked += 1;
                    let close = (f64::from(got[0]) - want.film).abs() < 0.02
                        && (f64::from(got[1]) - want.wet).abs() < 0.02
                        && (f64::from(got[2]) - want.edge).abs() < 0.05;
                    // Right at an arrival f32 and f64 can fall either side of it.
                    mismatched += usize::from(!close);
                }
            }
        }
        assert!(mismatched * 100 < checked, "{mismatched} of {checked}");
    }
}
