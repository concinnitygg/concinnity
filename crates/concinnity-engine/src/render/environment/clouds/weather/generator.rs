//! The weather map's generator on the CPU, line for line with
//! `cloud_weather_texel` in `atmosphere.hlsl` and the map's face layout in
//! `cloud_types.hlsl`: the settings measure its mean lean with it so a layer
//! keeps its coverage, and the tests read its shape here.

use super::super::noise::hash3f;
use super::super::noise::perlin;
use super::WeatherSettings;
use concinnity_core::math::{acos, cos, exp, floor, sin, sqrt};

// Each constant below is the shader's of the same name with `CLOUD_WEATHER_`
// before it, and the tests lock the two together.

// A lattice wide enough that no sphere's noise repeats across it.
pub(super) const PERIOD: i32 = 4096;
// How far a storm reaches round its center, in lattice cells; at most one, so
// every storm a point feels lies in the cells about it.
pub(super) const STORM_REACH: f32 = 0.75;
// How far a storm turns the cloud at its center, in radians at full swirl.
pub(super) const STORM_TWIST: f32 = 3.0;
// How much faster the noise runs north to south than east to west, less one,
// so features draw out along the latitudes.
pub(super) const LATITUDE_STRETCH: f32 = 1.5;
// Octaves of the systems' noise and of the convection's and thickness's at
// most, each half the size of the last; the map drops those finer than it
// holds.
pub(super) const OCTAVES: u32 = 6;
#[cfg(test)]
pub(super) const DETAIL_OCTAVES: u32 = 4;
// The finest noise the map holds, in cells per radian: two texels a cell at
// a face's middle, where its texels are widest.
pub(super) const NYQUIST: f32 = super::WEATHER_FACE as f32 / 4.0;
// The most weather systems across a radius the map can draw: the systems'
// noise, at its fastest along the poles, no finer than the map holds.
pub(super) const MAX_SYSTEMS_PER_RADIUS: f32 = NYQUIST / (2.0 * (1.0 + LATITUDE_STRETCH));
// Over what distance from the sphere's center, in lattice cells, a storm
// fades in; nearer it has no direction to turn about.
pub(super) const CENTER_FADE_START: f32 = 0.25;
pub(super) const CENTER_FADE_END: f32 = 0.5;
// How far along the lattice a storm's activity hash and the convection's and
// thickness's noise lie from the storms' and the systems'.
pub(super) const STORM_ACTIVITY: i32 = 1009;
#[cfg(test)]
pub(super) const HEAPS_SHIFT: f32 = 37.0;
#[cfg(test)]
pub(super) const THICKNESS_SHIFT: f32 = 71.0;

/// The climate's and the channels' weights, by their shader names less
/// `CLOUD_WEATHER_`: latitudes and widths in degrees, the rest shares.
#[cfg(test)]
pub(super) const WEIGHTS: [(&str, f32); 26] = [
    ("ITCZ_LEAN", ITCZ_LEAN),
    ("ITCZ_WIDTH", ITCZ_WIDTH),
    ("SUBTROPICS_LEAN", SUBTROPICS_LEAN),
    ("SUBTROPICS_LAT", SUBTROPICS_LAT),
    ("SUBTROPICS_WIDTH", SUBTROPICS_WIDTH),
    ("STORM_TRACK_LEAN", STORM_TRACK_LEAN),
    ("STORM_TRACK_LAT", STORM_TRACK_LAT),
    ("STORM_TRACK_WIDTH", STORM_TRACK_WIDTH),
    ("CLIMATE_BASE", CLIMATE_BASE),
    ("TROPICS_WIDTH", TROPICS_WIDTH),
    ("SYSTEMS_LEAN", SYSTEMS_LEAN),
    ("STORM_LEAN", STORM_LEAN),
    ("CONVECTION_BASE", CONVECTION_BASE),
    ("CONVECTION_TROPICS", CONVECTION_TROPICS),
    ("CONVECTION_SUBTROPICS", CONVECTION_SUBTROPICS),
    ("CONVECTION_HEAPS", CONVECTION_HEAPS),
    ("CONVECTION_LEAN", CONVECTION_LEAN),
    ("DENSITY_BASE", DENSITY_BASE),
    ("DENSITY_LEAN", DENSITY_LEAN),
    ("DENSITY_THICKNESS", DENSITY_THICKNESS),
    ("SHEET_STORM", SHEET_STORM),
    ("SHEET_TRACK", SHEET_TRACK),
    ("SHEET_TRACK_LEAN", SHEET_TRACK_LEAN),
    ("SHEET_TROPICS", SHEET_TROPICS),
    ("STORM_REACH", STORM_REACH),
    ("STORM_TWIST", STORM_TWIST),
];

// The climate's lean by latitude: a wet equatorial belt, dry subtropics and
// stormy middle latitudes over a base, and the tropics convection favors.
const ITCZ_LEAN: f32 = 0.3;
const ITCZ_WIDTH: f32 = 8.0;
const SUBTROPICS_LEAN: f32 = 0.8;
const SUBTROPICS_LAT: f32 = 24.0;
const SUBTROPICS_WIDTH: f32 = 9.0;
const STORM_TRACK_LEAN: f32 = 0.55;
const STORM_TRACK_LAT: f32 = 52.0;
const STORM_TRACK_WIDTH: f32 = 13.0;
const CLIMATE_BASE: f32 = 0.1;
#[cfg(test)]
const TROPICS_WIDTH: f32 = 14.0;
// The lean the systems' noise and the storms add.
const SYSTEMS_LEAN: f32 = 2.2;
const STORM_LEAN: f32 = 0.6;
// The convection's base and what the tropics, the subtropics, its own noise
// and the lean add.
#[cfg(test)]
const CONVECTION_BASE: f32 = 0.45;
#[cfg(test)]
const CONVECTION_TROPICS: f32 = 0.4;
#[cfg(test)]
const CONVECTION_SUBTROPICS: f32 = 0.25;
#[cfg(test)]
const CONVECTION_HEAPS: f32 = 1.5;
#[cfg(test)]
const CONVECTION_LEAN: f32 = 0.15;
// The density channel's base, the authored density at 0.5, and what the
// lean and its own noise add.
#[cfg(test)]
const DENSITY_BASE: f32 = 0.5;
#[cfg(test)]
const DENSITY_LEAN: f32 = 0.4;
#[cfg(test)]
const DENSITY_THICKNESS: f32 = 1.8;
// The sheetness storms, the storm tracks under cloud and the tropics give.
#[cfg(test)]
const SHEET_STORM: f32 = 1.3;
#[cfg(test)]
const SHEET_TRACK: f32 = 0.6;
#[cfg(test)]
const SHEET_TRACK_LEAN: f32 = 0.3;
#[cfg(test)]
const SHEET_TROPICS: f32 = 0.4;

type V3 = [f32; 3];

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(v: V3, s: f32) -> V3 {
    v.map(|c| c * s)
}

fn add(a: V3, b: V3) -> V3 {
    core::array::from_fn(|i| a[i] + b[i])
}

fn sub(a: V3, b: V3) -> V3 {
    core::array::from_fn(|i| a[i] - b[i])
}

fn length(v: V3) -> f32 {
    sqrt(dot(v, v))
}

fn normalize(v: V3) -> V3 {
    scale(v, 1.0 / length(v))
}

fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn smoothstep(lo: f32, hi: f32, v: f32) -> f32 {
    let t = saturate((v - lo) / (hi - lo));
    t * t * (3.0 - 2.0 * t)
}

// `v` turned `angle` radians about the unit `axis` (Rodrigues).
// `cloud_weather_turn`.
pub(super) fn turn(v: V3, axis: V3, angle: f32) -> V3 {
    let (s, c) = (sin(angle), cos(angle));
    add(
        add(scale(v, c), scale(cross(axis, v), s)),
        scale(axis, dot(axis, v) * (1.0 - c)),
    )
}

fn cell(v: f32) -> i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "floored and within the lattice"
    )]
    let c = floor(v) as i32;
    c
}

fn wrapped(c: [i32; 3]) -> [u32; 3] {
    c.map(|v| concinnity_core::math::SaturatingFrom::saturating_from(v.rem_euclid(PERIOD)))
}

// How much of each of up to `octaves` octaves the map holds for noise whose
// first octave runs at `cells` cells per radian: all of one up to the map's
// limit, fading out by twice it. `cloud_weather_octave`.
pub(super) fn octave_share(cells: f32, octave: u32) -> f32 {
    let size = (1u32 << octave) as f32;
    saturate(2.0 - cells * size / NYQUIST)
}

// Up to `octaves` octaves of gradient noise at `p`, its first running at
// `cells` cells per radian, as many as the map holds, in about [-0.5, 0.5].
// `cloud_weather_fbm`.
fn fbm(p: V3, octaves: u32, cells: f32) -> f32 {
    let (mut sum, mut total, mut size) = (0.0, 0.0, 1.0);
    for octave in 0..octaves {
        let w = octave_share(cells, octave) / size;
        if w <= 0.0 {
            break;
        }
        sum += w * perlin(scale(p, size), PERIOD);
        total += w;
        size *= 2.0;
    }
    if total > 0.0 { sum / total } else { 0.0 }
}

// Cells per radian the noise's first octave runs at, along the poles where
// the latitude stretch makes it fastest.
fn noise_cells(s: &WeatherSettings) -> f32 {
    s.systems_per_radius * 2.0 * (1.0 + LATITUDE_STRETCH)
}

// How far the storm centered at `e` turns the lookup at depth `k` in it:
// against the way the cloud is drawn turning, which is the planet's own spin
// about the pole north of the equator, the other way south of it, tapering
// to none on it. `cloud_weather_storm_turn`.
pub(super) fn storm_turn(s: &WeatherSettings, e: V3, k: f32) -> f32 {
    -dot(e, s.pole) * s.swirl * STORM_TWIST * k * k
}

// A bump of width `width` about `center`, both in degrees.
fn bump(deg: f32, center: f32, width: f32) -> f32 {
    let x = (deg - center) / width;
    exp(-x * x)
}

// The direction `d` turned by every storm near it, and how deep in a storm
// it lies, in [0, 1]. Storms are one jittered point per lattice cell, active
// by storminess, and wind the cloud round them as the planet spins north of
// the equator and the other way south of it. `cloud_weather_storms`.
fn storms(s: &WeatherSettings, d: V3) -> (V3, f32) {
    let p = add(scale(d, s.systems_per_radius), s.seed_offset);
    let base = p.map(cell);
    let mut q = d;
    let mut storm = 0.0f32;
    for n in 0..27 {
        let offset = [n % 3 - 1, n / 3 % 3 - 1, n / 9 - 1];
        let c: [i32; 3] = core::array::from_fn(|i| base[i] + offset[i]);
        let active = hash3f(wrapped(c.map(|v| v + STORM_ACTIVITY)))[0];
        if active >= s.storminess {
            continue;
        }
        let jitter = hash3f(wrapped(c));
        let point: V3 = core::array::from_fn(|i| c[i] as f32 + jitter[i]);
        let center = sub(point, s.seed_offset);
        // A storm near the sphere's center has no direction to turn about.
        let k = saturate(1.0 - length(sub(p, point)) / STORM_REACH)
            * smoothstep(CENTER_FADE_START, CENTER_FADE_END, length(center));
        if k <= 0.0 {
            continue;
        }
        let e = normalize(center);
        storm = storm.max(k);
        q = turn(q, e, storm_turn(s, e, k));
    }
    (q, storm)
}

// Degrees from the equator of the unit direction `d`.
fn latitude(s: &WeatherSettings, d: V3) -> f32 {
    (90.0 - acos(dot(d, s.pole).clamp(-1.0, 1.0)).to_degrees()).abs()
}

// Where the turned direction `q` reads the noise, `shift` cells along it:
// drawn out along the latitudes, a system's size to a cell.
fn noise_point(s: &WeatherSettings, q: V3, shift: f32) -> V3 {
    let r = add(q, scale(s.pole, dot(q, s.pole) * LATITUDE_STRETCH));
    add(
        scale(r, s.systems_per_radius * 2.0),
        s.seed_offset.map(|o| o + shift),
    )
}

// The lean at `d` before it is evened and clamped, and the storm depth and
// the turned direction it was read along.
fn raw_lean(s: &WeatherSettings, d: V3) -> (f32, f32, V3) {
    let (q, storm) = storms(s, d);
    let deg = latitude(s, d);
    let climate = ITCZ_LEAN * bump(deg, 0.0, ITCZ_WIDTH)
        - SUBTROPICS_LEAN * bump(deg, SUBTROPICS_LAT, SUBTROPICS_WIDTH)
        + STORM_TRACK_LEAN * bump(deg, STORM_TRACK_LAT, STORM_TRACK_WIDTH)
        + CLIMATE_BASE;
    let systems = fbm(noise_point(s, q, 0.0), OCTAVES, noise_cells(s));
    (
        s.belts * climate + SYSTEMS_LEAN * systems + STORM_LEAN * storm,
        storm,
        q,
    )
}

/// The weather at the unit direction `d` (the planet's fixed axes): its
/// cover lean (0.5 even), convection, density (0.5 the authored density) and
/// sheetness, each in `[0, 1]`. `cloud_weather_texel`.
#[cfg(test)]
pub(super) fn weather_at(s: &WeatherSettings, d: V3) -> [f32; 4] {
    let (raw, storm, q) = raw_lean(s, d);
    let lean = (raw - s.lean_offset).clamp(-1.0, 1.0);
    let deg = latitude(s, d);
    let tropics = bump(deg, 0.0, TROPICS_WIDTH);
    let subtropics = bump(deg, SUBTROPICS_LAT, SUBTROPICS_WIDTH);
    let storm_track = bump(deg, STORM_TRACK_LAT, STORM_TRACK_WIDTH);
    let heaps = fbm(
        noise_point(s, q, HEAPS_SHIFT),
        DETAIL_OCTAVES,
        noise_cells(s),
    );
    let thickness = fbm(
        noise_point(s, q, THICKNESS_SHIFT),
        DETAIL_OCTAVES,
        noise_cells(s),
    );
    let convection = saturate(
        CONVECTION_BASE
            + s.belts * (CONVECTION_TROPICS * tropics - CONVECTION_SUBTROPICS * subtropics)
            + CONVECTION_HEAPS * heaps
            + CONVECTION_LEAN * lean,
    );
    let density = saturate(DENSITY_BASE + DENSITY_LEAN * lean + DENSITY_THICKNESS * thickness);
    let sheet = saturate(
        SHEET_STORM * storm * storm
            + s.belts
                * (SHEET_TRACK * storm_track * saturate(lean + SHEET_TRACK_LEAN)
                    - SHEET_TROPICS * tropics),
    );
    [lean * 0.5 + 0.5, convection, density, sheet]
}

/// The mean of the weather's lean over the sphere before it is evened, which
/// the map takes off so a layer keeps its own coverage on average.
pub(super) fn mean_lean(s: &WeatherSettings) -> f32 {
    const SAMPLES: u32 = 1024;
    let golden = core::f32::consts::PI * (3.0 - sqrt(5.0));
    let total: f32 = (0..SAMPLES)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / SAMPLES as f32;
            let r = sqrt(1.0 - y * y);
            let a = golden * i as f32;
            raw_lean(s, [r * cos(a), y, r * sin(a)]).0
        })
        .sum();
    total / SAMPLES as f32
}

/// The direction through face `face`'s point `uv` (each about `[-1, 1]`, a
/// little past it in the gutter). `cloud_weather_face_dir`.
#[cfg(test)]
pub(super) fn face_dir(face: u32, uv: [f32; 2]) -> V3 {
    let [u, v] = uv;
    normalize(match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        _ => [-u, -v, -1.0],
    })
}

/// The face direction `d` falls on and its point there. `cloud_weather_face`.
#[cfg(test)]
pub(super) fn dir_face(d: V3) -> (u32, [f32; 2]) {
    let a = d.map(f32::abs);
    if a[0] >= a[1] && a[0] >= a[2] {
        if d[0] > 0.0 {
            (0, [-d[2] / a[0], -d[1] / a[0]])
        } else {
            (1, [d[2] / a[0], -d[1] / a[0]])
        }
    } else if a[1] >= a[2] {
        if d[1] > 0.0 {
            (2, [d[0] / a[1], d[2] / a[1]])
        } else {
            (3, [d[0] / a[1], -d[2] / a[1]])
        }
    } else if d[2] > 0.0 {
        (4, [d[0] / a[2], -d[1] / a[2]])
    } else {
        (5, [-d[0] / a[2], -d[1] / a[2]])
    }
}

/// The direction the map's texel `(x, y)` holds, gutter included.
/// `cloud_weather_kernel`.
#[cfg(test)]
pub(super) fn texel_dir(x: u32, y: u32) -> V3 {
    use super::{WEATHER_FACE, WEATHER_STRIDE};
    let face = x / WEATHER_STRIDE + 3 * (y / WEATHER_STRIDE);
    let local = [x % WEATHER_STRIDE, y % WEATHER_STRIDE];
    let uv = local.map(|l| (l as f32 - 0.5) / WEATHER_FACE as f32 * 2.0 - 1.0);
    face_dir(face, uv)
}

/// Where in the map, in texels, the unit direction `d` reads: within its
/// face, the gutter taking the bilinear tap past the edge. `cloud_weather_at`.
#[cfg(test)]
pub(super) fn map_point(d: V3) -> [f32; 2] {
    use super::{WEATHER_FACE, WEATHER_STRIDE};
    let (face, uv) = dir_face(d);
    let size = WEATHER_FACE as f32;
    let corner = [face % 3, face / 3].map(|c| (c * WEATHER_STRIDE) as f32);
    core::array::from_fn(|i| corner[i] + ((uv[i] * 0.5 + 0.5) * size + 1.0).clamp(1.0, size + 1.0))
}

#[cfg(test)]
mod tests {
    use super::super::Weather;
    use super::*;
    use concinnity_core::approx::close;

    fn settings(weather: &Weather) -> WeatherSettings {
        WeatherSettings::resolve(weather, 50.0, [0.0, 1.0, 0.0])
    }

    // Directions spread evenly over the sphere (a Fibonacci lattice).
    fn sphere(count: u32) -> impl Iterator<Item = V3> {
        let golden = core::f32::consts::PI * (3.0 - sqrt(5.0));
        (0..count).map(move |i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let r = sqrt(1.0 - y * y);
            let a = golden * i as f32;
            [r * cos(a), y, r * sin(a)]
        })
    }

    #[test]
    fn a_face_point_comes_back_from_its_direction() {
        for face in 0..6 {
            for uv in [[0.0, 0.0], [0.5, -0.25], [-0.9, 0.8]] {
                let (f, back) = dir_face(face_dir(face, uv));
                assert_eq!(f, face);
                assert!(close(back, uv, 1e-5), "{face} {uv:?} {back:?}");
            }
        }
    }

    #[test]
    fn turning_keeps_the_axis_and_the_length() {
        let axis = normalize([1.0, 2.0, -0.5]);
        assert!(close(turn(axis, axis, 1.3), axis, 1e-6));
        let v = turn([0.0, 0.0, 3.0], axis, 2.0);
        assert!(close(dot(v, v), 9.0, 1e-4));
        assert!(close(
            turn(
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                core::f32::consts::FRAC_PI_2
            ),
            [0.0, 0.0, -1.0],
            1e-6
        ));
    }

    #[test]
    fn every_channel_stays_in_range_and_the_seed_moves_the_systems() {
        let a = settings(&Weather::default());
        let b = settings(&Weather {
            seed: 9,
            ..Weather::default()
        });
        let mut differ = 0;
        for d in sphere(400) {
            let w = weather_at(&a, d);
            assert!(w.iter().all(|c| (0.0..=1.0).contains(c)), "{w:?}");
            assert_eq!(w, weather_at(&a, d));
            if !close(w, weather_at(&b, d), 0.05) {
                differ += 1;
            }
        }
        assert!(differ > 300, "{differ}");
    }

    // The planet example's weather.
    fn example() -> WeatherSettings {
        WeatherSettings::resolve(
            &Weather {
                system_size: 12_000.0,
                storminess: 0.5,
                swirl: 0.7,
                belts: 0.6,
                seed: 13,
                ..Weather::default()
            },
            50.0,
            [0.0, 0.0, -1.0],
        )
    }

    // The largest change of any channel between neighboring points along
    // `circles` great circles walked in `steps` steps each.
    fn largest_step(circles: u32, steps: u32, read: &impl Fn(V3) -> [f32; 4]) -> f32 {
        let mut largest = 0.0f32;
        for (i, axis) in sphere(circles).enumerate() {
            let side = normalize(cross(
                axis,
                if i % 2 == 0 {
                    [0.3, 1.0, 0.1]
                } else {
                    [1.0, 0.2, 0.4]
                },
            ));
            let mut last = read(side);
            for n in 1..=steps {
                let angle = core::f32::consts::TAU * n as f32 / steps as f32;
                let now = read(turn(side, axis, angle));
                largest = (0..4)
                    .map(|c| (now[c] - last[c]).abs())
                    .fold(largest, f32::max);
                last = now;
            }
        }
        largest
    }

    // Walking great circles on the example's planet, the largest change in
    // one step shrinks with the step, as it only does where nothing jumps: no
    // storm drops in or out partway through its reach, and no lattice wrap
    // shows. Steps of about 50 m, then a quarter of that.
    #[test]
    fn the_weather_is_continuous() {
        let s = example();
        let coarse = largest_step(8, 6000, &|d| weather_at(&s, d));
        let fine = largest_step(8, 24_000, &|d| weather_at(&s, d));
        assert!(fine < coarse * 0.4, "{coarse} {fine}");
        assert!(fine < 0.04, "{fine}");
    }

    // The map, read bilinearly as the GPU reads it, steps no further across
    // a cube face's edge than inside a face: the gutter carries each face's
    // neighbor.
    #[test]
    fn the_map_has_no_seam_at_its_face_edges() {
        let s = example();
        let texel = |x: i32, y: i32| {
            weather_at(
                &s,
                texel_dir(x.max(0).cast_unsigned(), y.max(0).cast_unsigned()),
            )
        };
        let read = |d: V3| {
            let [px, py] = map_point(d).map(|v| v - 0.5);
            let (x, y) = (cell(px), cell(py));
            let (fx, fy) = (px - x as f32, py - y as f32);
            let (a, b, c, e) = (
                texel(x, y),
                texel(x + 1, y),
                texel(x, y + 1),
                texel(x + 1, y + 1),
            );
            core::array::from_fn(|i| {
                let top = a[i] + (b[i] - a[i]) * fx;
                let bottom = c[i] + (e[i] - c[i]) * fx;
                top + (bottom - top) * fy
            })
        };
        // A bilinear map is continuous inside a face, so with a gutter its
        // largest step shrinks with the step across the edges too.
        let coarse = largest_step(6, 4000, &read);
        let fine = largest_step(6, 16_000, &read);
        assert!(fine < coarse * 0.4, "{coarse} {fine}");
    }

    #[test]
    fn a_texel_center_reads_back_where_it_lies() {
        use super::super::WEATHER_STRIDE;
        for (x, y) in [
            (1, 1),
            (40, 200),
            (WEATHER_STRIDE + 128, WEATHER_STRIDE + 7),
        ] {
            let got = map_point(texel_dir(x, y));
            assert!(
                close(got, [x as f32 + 0.5, y as f32 + 0.5], 1e-3),
                "{x} {y} {got:?}"
            );
        }
    }

    // A storm north of the equator draws the cloud turned the way the world
    // spins about its center (its up), one south the other way, and one on
    // the equator not at all.
    #[test]
    fn storms_wind_with_the_spin_in_each_hemisphere() {
        let s = example();
        let drawn_sense = |e: V3| {
            let g = normalize(add(e, scale(normalize(cross(e, [0.3, 0.5, 0.8])), 0.05)));
            let drawn = turn(g, e, -storm_turn(&s, e, 0.8));
            dot(cross(g, drawn), e)
        };
        let north = normalize(add(s.pole, [0.6, 0.2, 0.0]));
        let south = normalize(add(scale(s.pole, -1.0), [0.6, 0.2, 0.0]));
        assert!(drawn_sense(north) > 0.0);
        assert!(drawn_sense(south) < 0.0);
        let equator = normalize(cross(s.pole, [1.0, 0.0, 0.0]));
        assert!(close(storm_turn(&s, equator, 1.0), 0.0, 1e-6));
    }

    // The map holds every octave it is given up to its limit and drops
    // those past twice it, for the example and for the finest systems.
    #[test]
    fn no_octave_is_finer_than_the_map_holds() {
        for cells in [noise_cells(&example()), MAX_SYSTEMS_PER_RADIUS * 5.0] {
            assert!(close(octave_share(cells, 0), 1.0, 0.0), "{cells}");
            for octave in 0..OCTAVES {
                let size = (1u32 << octave) as f32;
                if cells * size >= 2.0 * NYQUIST {
                    assert_eq!(octave_share(cells, octave), 0.0, "{cells} {octave}");
                }
                if cells * size <= NYQUIST {
                    assert_eq!(octave_share(cells, octave), 1.0, "{cells} {octave}");
                }
            }
        }
    }

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shader::embedded::ATMOSPHERE;
        let name = |n: &str| alloc::format!("CLOUD_WEATHER_{n}");
        for (n, value) in WEIGHTS {
            assert_eq!(
                crate::render::shader::consts::float(src, &name(n)),
                value,
                "{n}"
            );
        }
        let float = |n: &str| crate::render::shader::consts::float(src, &name(n));
        let uint = |n: &str| crate::render::shader::consts::uint(src, &name(n));
        assert_eq!(float("LATITUDE_STRETCH"), LATITUDE_STRETCH);
        assert_eq!(float("HEAPS_SHIFT"), HEAPS_SHIFT);
        assert_eq!(float("THICKNESS_SHIFT"), THICKNESS_SHIFT);
        assert_eq!(float("NYQUIST"), NYQUIST);
        assert_eq!(float("CENTER_FADE_START"), CENTER_FADE_START);
        assert_eq!(float("CENTER_FADE_END"), CENTER_FADE_END);
        assert_eq!(uint("PERIOD"), PERIOD.cast_unsigned() as usize);
        assert_eq!(uint("OCTAVES"), OCTAVES as usize);
        assert_eq!(uint("DETAIL_OCTAVES"), DETAIL_OCTAVES as usize);
        assert_eq!(
            uint("STORM_ACTIVITY"),
            STORM_ACTIVITY.cast_unsigned() as usize
        );
    }

    // Over the whole sphere the layers keep about their own coverage.
    #[test]
    fn the_lean_averages_near_even() {
        let s = settings(&Weather {
            system_size: 15_000.0,
            ..Weather::default()
        });
        let n = 2000;
        let mean = sphere(n).map(|d| weather_at(&s, d)[0]).sum::<f32>() / n as f32;
        assert!(close(mean, 0.5, 0.04), "{mean}");
    }

    // Belts make the equator and the storm tracks cloudier than the
    // subtropics; without them latitude makes no difference.
    #[test]
    fn belts_wet_the_equator_and_dry_the_subtropics() {
        let band = |s: &WeatherSettings, deg: f32| {
            let y = sin(deg.to_radians());
            let r = sqrt(1.0 - y * y);
            (0..360)
                .map(|i| {
                    let a = (i as f32).to_radians();
                    weather_at(s, [r * cos(a), y, r * sin(a)])[0]
                })
                .sum::<f32>()
                / 360.0
        };
        let belted = settings(&Weather {
            system_size: 15_000.0,
            belts: 1.0,
            ..Weather::default()
        });
        assert!(band(&belted, 0.0) > band(&belted, 24.0) + 0.2);
        assert!(band(&belted, 52.0) > band(&belted, 24.0) + 0.2);
        let even = settings(&Weather {
            system_size: 15_000.0,
            belts: 0.0,
            storminess: 0.0,
            ..Weather::default()
        });
        assert!(close(band(&even, 0.0), band(&even, 24.0), 0.12));
    }

    // With no storms the swirl turns nothing.
    #[test]
    fn no_storms_no_swirl() {
        let calm = |swirl| {
            settings(&Weather {
                storminess: 0.0,
                swirl,
                ..Weather::default()
            })
        };
        for d in sphere(50) {
            assert_eq!(weather_at(&calm(0.0), d), weather_at(&calm(1.0), d));
        }
    }
}
