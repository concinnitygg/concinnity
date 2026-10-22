use super::*;
use crate::render::environment::water::ocean::OceanQuality;
use crate::render::environment::water::ocean::initial::{
    SpectrumTexel, frequency_index, height_variance, initial_spectrum,
};
use concinnity_core::math::narrow_f32;

// The example's sea: a 4 m/s breeze at 1.75 response over 50 km.
fn sea() -> OceanSea {
    OceanSea {
        wind_speed: 7.0,
        direction: [0.94, 0.33],
        fetch: 5.0e4,
    }
}

fn cascades(quality: OceanQuality) -> Cascades {
    Cascades::new(sea().shape().unwrap().peak_wavelength(), quality)
}

// Every texel of a spectrum, as the GPU's transform evaluates it: the
// height at `p` meters on the tile and its gradient, `time` seconds in.
fn direct(spectrum: &[SpectrumTexel], size: u32, length: f32, p: [f64; 2], time: f64) -> [f64; 3] {
    let dk = TAU / f64::from(length);
    let fraction = libm::fmod(time, OCEAN_REPEAT_SECONDS) / OCEAN_REPEAT_SECONDS;
    let mut out = [0.0; 3];
    for (i, t) in spectrum.iter().enumerate() {
        let i = u32::saturating_from(i);
        let fx = frequency_index(i % size, size);
        let fy = frequency_index(i / size, size);
        let m = spectrum[(((size - i / size) % size) * size + (size - i % size) % size) as usize];
        if t[0] == 0.0 && t[1] == 0.0 && m[0] == 0.0 && m[1] == 0.0 {
            continue;
        }
        let x = f64::from(t[2].max(m[2])) * fraction;
        let phase = TAU * (x - libm::floor(x));
        let (s, c) = (libm::sin(phase), libm::cos(phase));
        // h0 e^(-i phase) + conj(m) e^(i phase).
        let re =
            f64::from(t[0]) * c + f64::from(t[1]) * s + f64::from(m[0]) * c + f64::from(m[1]) * s;
        let im =
            f64::from(t[1]) * c - f64::from(t[0]) * s + f64::from(m[0]) * s - f64::from(m[1]) * c;
        let k = [f64::from(fx) * dk, f64::from(fy) * dk];
        let a = k[0] * p[0] + k[1] * p[1];
        let (sa, ca) = (libm::sin(a), libm::cos(a));
        out[0] += re * ca - im * sa;
        // d/dx of Re(H e^(i k x)) = Re(i k H e^(i k x)).
        out[1] += -k[0] * (re * sa + im * ca);
        out[2] += -k[1] * (re * sa + im * ca);
    }
    out
}

// The band's own waves, summed directly where the grid is read.
fn band_spectrum(c: &Cascades) -> (Vec<SpectrumTexel>, u32) {
    let size = c.size();
    let full = initial_spectrum(&sea(), c, 0);
    let band = full
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let i = u32::saturating_from(i);
            let fx = frequency_index(i % size, size);
            let fy = frequency_index(i / size, size);
            if fx * fx + fy * fy < BAND_REACH * BAND_REACH {
                *t
            } else {
                [0.0; 4]
            }
        })
        .collect();
    (band, size)
}

fn points() -> impl Iterator<Item = [f64; 2]> {
    (0..24).map(|i| {
        let i = f64::from(i);
        [37.1 * i + 0.3, -11.7 * i * i + 5.0]
    })
}

// The grid reads its own waves as their direct sum would: the height to a
// few millimeters, the slope to a small share of its spread.
#[test]
fn the_band_reads_its_waves() {
    let c = cascades(OceanQuality::FULL);
    let mut band = OceanBand::new(&sea(), &c).unwrap();
    let (spectrum, size) = band_spectrum(&c);
    let sigma = libm::sqrt(f64::from(height_variance(&spectrum)));
    let mut worst = [0.0f64; 2];
    let mut slope_spread = 0.0;
    for time in [0.0, 3.05, 77.73] {
        band.prepare(time);
        for p in points() {
            let want = direct(&spectrum, size, c.lengths()[0], p, time);
            let got = band.sample(time, p);
            worst[0] = worst[0].max((f64::from(got.height) - want[0]).abs());
            for a in 0..2 {
                worst[1] = worst[1].max((f64::from(got.slope[a]) - want[1 + a]).abs());
            }
            slope_spread = f64::max(slope_spread, libm::hypot(want[1], want[2]));
        }
    }
    assert!(
        worst[0] < 0.03 * sigma,
        "height {} of sigma {sigma}",
        worst[0]
    );
    assert!(
        worst[1] < 0.1 * slope_spread,
        "slope {} of {slope_spread}",
        worst[1]
    );
}

// What the band leaves out of the drawn sea: the rest of the first cascade
// and the shorter cascades. Against the whole first cascade summed directly,
// as the GPU's transform evaluates it, plus the other cascades' variance.
#[test]
fn the_band_holds_most_of_the_drawn_sea() {
    let c = cascades(OceanQuality::FULL);
    let mut band = OceanBand::new(&sea(), &c).unwrap();
    let full = initial_spectrum(&sea(), &c, 0);
    let first = f64::from(height_variance(&full));
    let rest: f64 = (1..c.count())
        .map(|i| f64::from(height_variance(&initial_spectrum(&sea(), &c, i))))
        .sum();
    let mut sq = 0.0;
    let mut n = 0.0;
    for time in [1.0, 40.0] {
        band.prepare(time);
        for p in points() {
            let want = direct(&full, c.size(), c.lengths()[0], p, time)[0];
            let got = f64::from(band.sample(time, p).height);
            sq += (got - want) * (got - want);
            n += 1.0;
        }
    }
    let error = libm::sqrt(sq / n + rest);
    let sigma = libm::sqrt(first + rest);
    let hs = 4.0 * sigma;
    // The example's sea: Hs about a meter, the band within a third of
    // sigma, under 10 cm rms.
    assert!((0.6..1.6).contains(&hs), "{hs}");
    assert!(error < 0.35 * sigma && error < 0.1, "{error} of {sigma}");
}

// Moving on by one snapshot keeps the later one; the read is continuous
// across a snapshot boundary.
#[test]
fn the_read_is_continuous_across_snapshots() {
    let c = cascades(OceanQuality::FULL);
    let mut band = OceanBand::new(&sea(), &c).unwrap();
    let at = [12.0, -40.0];
    let edge = 3.0 * BAND_SNAPSHOT_SECONDS;
    band.prepare(edge - 1e-6);
    let before = band.sample(edge - 1e-6, at);
    band.prepare(edge + 1e-6);
    let after = band.sample(edge + 1e-6, at);
    assert!((before.height - after.height).abs() < 1e-4);
    assert_eq!(band.snapshots[band.pair[0]].index, 3);
    // The kept snapshot is the one a fresh band computes.
    let mut fresh = OceanBand::new(&sea(), &c).unwrap();
    fresh.prepare(edge + 1e-6);
    assert_eq!(fresh.sample(edge + 1e-6, at), after);
}

// The same sea gives the same band at every quality, and the same reads.
#[test]
fn the_band_is_deterministic_and_preset_free() {
    let low = OceanQuality {
        size: 128,
        cascades: 2,
    };
    let mut a = OceanBand::new(&sea(), &cascades(OceanQuality::FULL)).unwrap();
    let mut b = OceanBand::new(&sea(), &cascades(low)).unwrap();
    a.prepare(9.87);
    b.prepare(9.87);
    for p in points() {
        assert_eq!(a.sample(9.87, p), b.sample(9.87, p));
    }
}

// The rates are the time derivatives of what they rate: over one snapshot
// step the mean rate is the change. The rise rate is read bilinearly and the
// height by Hermite, so they agree only to the bilinear read's error.
#[test]
fn the_rates_follow_the_waves() {
    let c = cascades(OceanQuality::FULL);
    let mut band = OceanBand::new(&sea(), &c).unwrap();
    let (t0, t1) = (5.0, 5.0 + BAND_SNAPSHOT_SECONDS);
    band.prepare(t0);
    let mut worst = [0.0f32; 2];
    let mut spread = [0.0f32; 2];
    for p in points() {
        let (a, b) = (band.sample(t0, p), band.sample(t1, p));
        let rate = |x: f32, y: f32| 0.5 * (x + y);
        let pairs = [
            (rate(a.rise_rate, b.rise_rate), b.height - a.height),
            (
                rate(a.shift_rate[0], b.shift_rate[0]),
                b.shift[0] - a.shift[0],
            ),
        ];
        for (i, (r, d)) in pairs.into_iter().enumerate() {
            let fd = d / narrow_f32(BAND_SNAPSHOT_SECONDS);
            worst[i] = worst[i].max((r - fd).abs());
            spread[i] = spread[i].max(r.abs());
        }
    }
    let tolerance = [0.15, 0.05];
    for i in 0..2 {
        assert!(spread[i] > 0.05, "{spread:?}");
        assert!(
            worst[i] < tolerance[i] * spread[i],
            "{worst:?} of {spread:?}"
        );
    }
}

// At every size a simulation runs at, the first cascade draws every wave the
// band holds, so the band is never missing what a later cascade drew.
#[test]
fn the_first_cascade_draws_the_whole_band_at_every_size() {
    for size in [32, 64, 128, 256, 512] {
        let quality = OceanQuality { size, cascades: 2 }.clamped();
        assert!(quality.size >= OceanQuality::MIN_SIZE);
        let c = cascades(quality);
        let dk = core::f32::consts::TAU / c.lengths()[0];
        let reach = BAND_REACH as f32 * dk;
        assert!(
            c.band(0)[1] >= reach * 0.999_999,
            "{size}: {} < {reach}",
            c.band(0)[1]
        );
    }
}
