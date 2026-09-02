//! The `stars` equirectangular generator: a night sky of scattered point
//! stars over a near-black background that darkens below the horizon.
//!
//! Every value comes from an integer hash of the grid cell a star sits in, so
//! the same field is generated on every host without a random source. The
//! brightest stars carry radiance well above 1.0 so bloom catches them, while
//! the background sits low enough that the map lights a scene only as a cold
//! fill.

use alloc::vec::Vec;
use core::f32::consts::PI;

use super::source::HdrImage;
use crate::math::{cos, exp, floor, powf, powi, sin, sin_cos};

// Four texels of width for every texel of a 1024-pixel cube face, so a star
// survives the resample into the cube without landing between texels. Below
// that face size the points are drawn large enough to read as blobs.
const WIDTH: u32 = 4096;
const HEIGHT: u32 = 2048;

// One star candidate per CELL x CELL block of texels.
const CELL: u32 = 16;
const CELLS_X: u32 = WIDTH / CELL;
const CELLS_Y: u32 = HEIGHT / CELL;

// Chance a cell holds a star, before the weighting that spreads the field
// evenly over the sphere instead of piling it up at the poles.
const STAR_CHANCE: f32 = 0.22;

// Angular radius of a star's gaussian core, and how far out it is drawn. A
// little under a texel: smaller and a star's brightness would depend on where
// in its texel it happened to fall.
const STAR_SIGMA: f32 = 0.0011;
const STAR_REACH: f32 = 3.0 * STAR_SIGMA;
// Where a star's tail falls below this fraction of its peak it stops being
// written.
const STAR_CUTOFF: f32 = 1e-3;

// Peak radiance of the faintest star, and the ratio between it and the
// brightest. The curve between them is magnitude-like: most stars sit near
// the faint end and roughly one in six clears 1.0.
const STAR_FAINTEST: f32 = 0.10;
const STAR_RANGE: f32 = 90.0;
// How far a tinted star is pushed off white, warm on one side, cold on the
// other.
const STAR_TINT: [f32; 2] = [0.25, 0.30];

// The unlit sky: near black above the horizon, darker still below it. No glow
// at the horizon itself: a band there would draw a line at eye level that a
// finite water surface, whose far edge sits below eye level, cannot meet.
const ZENITH: [f32; 3] = [0.0016, 0.0021, 0.0038];
const GROUND: [f32; 3] = [0.0004, 0.0005, 0.0008];
// Height over which the background darkens to the ground colour.
const GROUND_DEPTH: f32 = 0.25;

/// Synthetic equirectangular HDR for the `generator: "stars"` source.
pub fn generate_stars_equirect() -> HdrImage {
    let mut pixels = background();
    scatter_stars(&mut pixels);
    HdrImage {
        width: WIDTH,
        height: HEIGHT,
        pixels,
    }
}

// The starless sky, one colour per row.
fn background() -> Vec<[f32; 3]> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for row in 0..HEIGHT {
        let up = cos(row_theta(row));
        let s = (-up / GROUND_DEPTH).clamp(0.0, 1.0);
        let colour = [
            lerp(ZENITH[0], GROUND[0], s),
            lerp(ZENITH[1], GROUND[1], s),
            lerp(ZENITH[2], GROUND[2], s),
        ];
        for _ in 0..WIDTH {
            pixels.push(colour);
        }
    }
    pixels
}

// Add each cell's star, if it has one, to the texels its core reaches.
fn scatter_stars(pixels: &mut [[f32; 3]]) {
    for cy in 0..CELLS_Y {
        for cx in 0..CELLS_X {
            let seed = cell_seed(cx, cy);
            let u = (cx as f32 + unit(seed, 1)) / CELLS_X as f32;
            let v = (cy as f32 + unit(seed, 2)) / CELLS_Y as f32;
            let (sin_theta, cos_theta) = sin_cos(v * PI);
            // A row of cells near a pole covers far less of the sphere than a
            // row at the equator, and holds proportionally fewer stars.
            if unit(seed, 3) > STAR_CHANCE * sin_theta {
                continue;
            }
            let dir = direction(u, sin_theta, cos_theta);
            splat(pixels, dir, star_colour(seed), u, v, sin_theta);
        }
    }
}

// The star's peak radiance per channel: a magnitude-like brightness curve,
// tinted warm or cold by a little.
fn star_colour(seed: u32) -> [f32; 3] {
    let peak = STAR_FAINTEST * powf(STAR_RANGE, powi(unit(seed, 4), 4));
    let tint = powi(unit(seed, 5) * 2.0 - 1.0, 3);
    [
        peak * (1.0 + STAR_TINT[0] * tint),
        peak,
        peak * (1.0 - STAR_TINT[1] * tint),
    ]
}

// Draw one star's gaussian core. The box it covers is wider in texels the
// closer it sits to a pole, since a row of texels there spans less sky.
fn splat(pixels: &mut [[f32; 3]], dir: [f32; 3], colour: [f32; 3], u: f32, v: f32, sin_theta: f32) {
    let half_y = (STAR_REACH / PI * HEIGHT as f32) as i32 + 1;
    let spread = STAR_REACH / (2.0 * PI * sin_theta.max(1e-4)) * WIDTH as f32;
    let half_x = spread.min(WIDTH as f32 * 0.5) as i32 + 1;
    let centre_x = floor(u * WIDTH as f32 - 0.5) as i32;
    let centre_y = floor(v * HEIGHT as f32 - 0.5) as i32;
    let inv_sigma2 = 1.0 / (2.0 * STAR_SIGMA * STAR_SIGMA);
    for row in (centre_y - half_y)..=(centre_y + half_y) {
        if row < 0 || row >= HEIGHT as i32 {
            continue;
        }
        let (st, ct) = sin_cos(row_theta(row as u32));
        for col in (centre_x - half_x)..=(centre_x + half_x) {
            let d = direction((col as f32 + 0.5) / WIDTH as f32, st, ct);
            let dot = d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2];
            let falloff = exp(-2.0 * (1.0 - dot).max(0.0) * inv_sigma2);
            if falloff < STAR_CUTOFF {
                continue;
            }
            let texel = &mut pixels[(row as u32 * WIDTH + wrap_col(col)) as usize];
            for k in 0..3 {
                texel[k] += colour[k] * falloff;
            }
        }
    }
}

// Latitude of a texel row's centre, matching the equirect sampler's mapping.
fn row_theta(row: u32) -> f32 {
    PI * (row as f32 + 0.5) / HEIGHT as f32
}

// Unit direction for a longitude given as an equirect u, with the latitude
// already resolved into its sine and cosine.
fn direction(u: f32, sin_theta: f32, cos_theta: f32) -> [f32; 3] {
    let phi = (u - 0.5) * 2.0 * PI;
    [cos(phi) * sin_theta, cos_theta, sin(phi) * sin_theta]
}

fn wrap_col(col: i32) -> u32 {
    col.rem_euclid(WIDTH as i32) as u32
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// A cell's seed, and the independent [0, 1) draws taken from it.
fn cell_seed(cx: u32, cy: u32) -> u32 {
    mix(cx.wrapping_mul(0x9e37_79b9) ^ mix(cy))
}

fn unit(seed: u32, stream: u32) -> f32 {
    (mix(seed ^ stream.wrapping_mul(0x27d4_eb2f)) >> 8) as f32 / 16_777_216.0
}

fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

#[cfg(test)]
mod tests {
    use super::super::deserialise;
    use super::super::schedule::Serial;
    use super::super::source::bake_payload;
    use super::*;

    // Peak radiance of the faintest star, as the threshold that separates a
    // star's texels from the background beneath them.
    const LIT: f32 = STAR_FAINTEST * 0.5;

    fn brightest(texel: [f32; 3]) -> f32 {
        texel[0].max(texel[1]).max(texel[2])
    }

    #[test]
    fn stars_generator_bakes_into_a_full_payload_serially() {
        let hdr = generate_stars_equirect();
        assert_eq!((hdr.width, hdr.height), (WIDTH, HEIGHT));
        let payload = bake_payload(&hdr, 16, 8, 32, 12.0, &Serial);
        let view = deserialise(&payload).expect("deserialise");
        assert_eq!(view.irradiance_face, 8);
        assert_eq!(view.prefilter_face, 16);
        // Prefilter mips for face_size 16: 16, 8, 4 -> 3 levels.
        assert_eq!(view.prefilter_mip_bytes.len(), 3);
    }

    #[test]
    fn the_field_is_near_black_apart_from_a_scatter_of_hdr_points() {
        let hdr = generate_stars_equirect();
        let texels = hdr.pixels.len();
        let mean = hdr.pixels.iter().map(|p| brightest(*p)).sum::<f32>() / texels as f32;
        assert!(mean < 0.02, "the sky averages near black, not {mean}");

        let hdr_points = hdr.pixels.iter().filter(|p| brightest(**p) > 1.0).count();
        assert!(
            (200..texels / 100).contains(&hdr_points),
            "a scatter of texels clears 1.0, not {hdr_points} of {texels}"
        );
        // Bright enough for a bloom threshold well above 1.0 to catch some.
        let peak = hdr.pixels.iter().map(|p| brightest(*p)).fold(0.0, f32::max);
        assert!(peak > 4.0, "the brightest star reaches {peak}");
    }

    #[test]
    fn stars_range_in_brightness_and_a_few_carry_a_tint() {
        let hdr = generate_stars_equirect();
        let cores: Vec<[f32; 3]> = hdr
            .pixels
            .iter()
            .copied()
            .filter(|p| brightest(*p) > LIT)
            .collect();
        assert!(cores.len() > 1000, "only {} lit texels", cores.len());
        // Most of the field is faint, a minority bright.
        let bright = cores.iter().filter(|p| brightest(**p) > 1.0).count();
        assert!(
            bright * 3 < cores.len(),
            "{bright} of {} lit texels are bright",
            cores.len()
        );
        // Tint is the red/blue split, near zero for a white star.
        let tinted = cores
            .iter()
            .filter(|p| (p[0] - p[2]).abs() > 0.25 * brightest(**p))
            .count();
        assert!(tinted > 0, "no star carries a tint");
        assert!(
            tinted * 2 < cores.len(),
            "{tinted} of {} lit texels are tinted, which is not a few",
            cores.len()
        );
    }

    #[test]
    fn the_same_field_is_generated_every_time() {
        let a = generate_stars_equirect();
        let b = generate_stars_equirect();
        assert_eq!(a.pixels, b.pixels);
    }

    // The sky is one faint shade down to the horizon, with nothing drawn at
    // the horizon itself, and the ground below it is darker still.
    #[test]
    fn the_sky_is_flat_to_the_horizon_and_the_ground_below_it_is_darkest() {
        let hdr = generate_stars_equirect();
        // A row's background, taken as its dimmest texel so no star counts.
        let floor_of = |row: u32| {
            let start = (row * WIDTH) as usize;
            hdr.pixels[start..start + WIDTH as usize]
                .iter()
                .map(|p| brightest(*p))
                .fold(f32::INFINITY, f32::min)
        };
        let zenith = floor_of(0);
        let horizon = floor_of(HEIGHT / 2 - 1);
        let ground = floor_of(HEIGHT - 1);
        assert!(
            (horizon - zenith).abs() < 1e-6,
            "the horizon ({horizon}) is the zenith's shade ({zenith})"
        );
        assert!(
            ground < zenith,
            "the ground ({ground}) is darker than the zenith ({zenith})"
        );
        assert!(zenith < 0.01, "the sky stays faint, not {zenith}");
    }

    #[test]
    fn star_density_follows_solid_angle_rather_than_texel_count() {
        let hdr = generate_stars_equirect();
        // The share of a band of rows the stars cover. Texels are counted by
        // the sky they span, not one apiece: an equirect row near a pole holds
        // as many texels as one at the equator over far less sky, so a star
        // there spreads over more of them without being any larger.
        let density = |first: u32, count: u32| {
            let mut lit = 0.0f32;
            let mut sky = 0.0f32;
            for row in first..first + count {
                let start = (row * WIDTH) as usize;
                let span = sin(row_theta(row));
                lit += span
                    * hdr.pixels[start..start + WIDTH as usize]
                        .iter()
                        .filter(|p| brightest(**p) > LIT)
                        .count() as f32;
                sky += span * WIDTH as f32;
            }
            lit / sky
        };
        let pole = density(0, HEIGHT / 20);
        let equator = density(HEIGHT / 2 - HEIGHT / 40, HEIGHT / 20);
        assert!(pole > 0.0, "the pole band holds no stars at all");
        let ratio = pole / equator;
        assert!(
            (0.5..2.0).contains(&ratio),
            "pole density {pole} against equator density {equator}"
        );
    }
}
