// Baking the daylight curves: the transmittance and multiple-scattering
// tables built as their kernels in atmosphere.hlsl build them, then the sky
// marched up from the ground as the sky-view kernel marches it.

use super::super::lut::{
    MULTISCATTER_SIZE, TRANSMITTANCE_HEIGHT, TRANSMITTANCE_WIDTH, texels_to_unit,
    transmittance_r_mu, transmittance_uv, unit_to_texels,
};
use super::super::optics::{ray_exit, ray_hits_ground, sun_transmittance, transmittance};
use super::super::settings::AtmosphereSettings;
use super::super::sh::fibonacci_direction;
use super::{DAYLIGHT_KNOTS, Daylight, knot_mu};
use alloc::vec;
use alloc::vec::Vec;
use concinnity_core::fanout::FanOut;
use concinnity_core::math::vec3::{self, dot};
use concinnity_core::math::{SaturatingFrom, cos, exp, sin, sqrt};

use core::f32::consts::PI;

// Directions each multiple-scattering texel averages over, and the steps
// along each. `MULTISCATTER_DIRECTIONS` and `MULTISCATTER_STEPS` in the shaders.
const MULTISCATTER_DIRECTIONS: u32 = 64;
const MULTISCATTER_STEPS: u32 = 20;
// Steps along each sky ray. `SKY_VIEW_STEPS` in `atmosphere_types.hlsl`.
const SKY_STEPS: u32 = 30;
// The sky's hemisphere for the irradiance: rings even in the cosine to the
// up, each split around the up. The sky is the same either side of the sun,
// so only the half round one side is marched.
const RINGS: usize = 8;
const AROUND: usize = 16;
// The sine of the elevation the horizon's radiance is taken at.
const HORIZON_SIN: f32 = 0.05;
// How far above the ground the point sits, in kilometers: off the ground
// sphere, so no ray starts inside it.
const LIFT_KM: f32 = 1e-3;

/// The knots for the ground of `s`, the tables' rows and the knots spread
/// over `rows`.
pub(super) fn knots<S: FanOut>(s: &AtmosphereSettings, rows: &S) -> [Daylight; DAYLIGHT_KNOTS] {
    let tables = Tables::new(s, rows);
    let r = s.bottom_radius + LIFT_KM;
    let mut knots: [(usize, Daylight); DAYLIGHT_KNOTS] =
        core::array::from_fn(|i| (i, Daylight::default()));
    rows.spawn_each(&mut knots, |(i, knot)| {
        let mu = knot_mu(*i);
        let sun = [sqrt((1.0 - mu * mu).max(0.0)), mu, 0.0];
        *knot = tables.daylight(r, mu, sun);
    });
    knots.map(|(_, knot)| knot)
}

struct Tables<'a> {
    s: &'a AtmosphereSettings,
    transmittance: Vec<[f32; 3]>,
    multiscatter: Vec<[f32; 3]>,
}

impl<'a> Tables<'a> {
    fn new<S: FanOut>(s: &'a AtmosphereSettings, rows: &S) -> Self {
        let (w, h) = (TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT);
        let mut table = Self {
            s,
            transmittance: vec![[0.0; 3]; (w * h) as usize],
            multiscatter: vec![[0.0; 3]; (MULTISCATTER_SIZE * MULTISCATTER_SIZE) as usize],
        };
        let mut texels = by_row(&mut table.transmittance, w);
        rows.for_each(&mut texels, |(y, row)| {
            for (x, texel) in row.iter_mut().enumerate() {
                let uv = [(x as f32 + 0.5) / w as f32, (*y as f32 + 0.5) / h as f32];
                let (r, mu) = transmittance_r_mu(s.bottom_radius, s.top_radius, uv);
                *texel = transmittance(s, r, mu);
            }
        });
        let mut multiscatter = core::mem::take(&mut table.multiscatter);
        let mut texels = by_row(&mut multiscatter, MULTISCATTER_SIZE);
        rows.spawn_each(&mut texels, |(y, row)| {
            for (x, texel) in (0..).zip(row.iter_mut()) {
                *texel = table.multiscatter_texel(x, *y);
            }
        });
        table.multiscatter = multiscatter;
        table
    }

    // Light surviving the path out of the air, none into the ground: the
    // table read as `atmo_transmittance` reads it.
    fn to_top(&self, r: f32, mu: f32) -> [f32; 3] {
        if ray_hits_ground(r, mu, self.s.bottom_radius) {
            return [0.0; 3];
        }
        let uv = transmittance_uv(self.s.bottom_radius, self.s.top_radius, r, mu);
        bilinear(
            &self.transmittance,
            TRANSMITTANCE_WIDTH,
            TRANSMITTANCE_HEIGHT,
            uv,
        )
    }

    // `atmosphere_multiscatter_kernel` for texel (`x`, `y`).
    fn multiscatter_texel(&self, x: u32, y: u32) -> [f32; 3] {
        let s = self.s;
        let size = MULTISCATTER_SIZE;
        let mu_s = texels_to_unit((x as f32 + 0.5) / size as f32, size) * 2.0 - 1.0;
        let t = texels_to_unit((y as f32 + 0.5) / size as f32, size);
        let r = (s.bottom_radius + 1e-3) + (s.top_radius - 1e-3 - s.bottom_radius - 1e-3) * t;
        let sun = [sqrt((1.0 - mu_s * mu_s).max(0.0)), mu_s, 0.0];
        let mut second = [0.0_f32; 3];
        let mut again = [0.0_f32; 3];
        for lane in 0..MULTISCATTER_DIRECTIONS {
            let fib = fibonacci_direction(lane, MULTISCATTER_DIRECTIONS);
            let dir = [fib[0], fib[2], fib[1]];
            let ground = ray_ground(r, dir[1], s.bottom_radius);
            let length = if ground >= 0.0 {
                ground
            } else {
                ray_exit(r, dir[1], s.top_radius)
            };
            let dt = length / MULTISCATTER_STEPS as f32;
            let mut throughput = [1.0_f32; 3];
            for i in 0..MULTISCATTER_STEPS {
                let pos = along([0.0, r, 0.0], dir, (i as f32 + 0.5) * dt);
                let rp = vec3::length(pos);
                let air = s.air.medium_at(rp - s.bottom_radius);
                let to_sun = self.to_top(rp, dot(pos, sun) / rp);
                for c in 0..3 {
                    let scattering = air.rayleigh[c] + air.mie[c];
                    let step_t = exp(-air.extinction[c] * dt);
                    let inv = 1.0 / air.extinction[c].max(1e-7);
                    let src = scattering * to_sun[c] / (4.0 * PI);
                    second[c] += throughput[c] * (src - src * step_t) * inv;
                    again[c] += throughput[c] * (scattering - scattering * step_t) * inv;
                    throughput[c] *= step_t;
                }
            }
            if ground >= 0.0 {
                let hit = along([0.0, r, 0.0], dir, ground);
                let mu_g = dot(hit, sun) / vec3::length(hit);
                let lit = self.to_top(s.bottom_radius, mu_g);
                for c in 0..3 {
                    second[c] += throughput[c] * lit[c] * mu_g.max(0.0) * s.ground_albedo[c] / PI;
                }
            }
        }
        let n = MULTISCATTER_DIRECTIONS as f32;
        core::array::from_fn(|c| (second[c] / n) / (1.0 - again[c] / n).max(1e-3))
    }

    // The multiple-scattering table read as `atmo_multiscatter` reads it.
    fn multiscatter(&self, r: f32, mu_s: f32) -> [f32; 3] {
        let s = self.s;
        let size = MULTISCATTER_SIZE;
        let uv = [
            unit_to_texels((mu_s * 0.5 + 0.5).clamp(0.0, 1.0), size),
            unit_to_texels(
                ((r - s.bottom_radius) / (s.top_radius - s.bottom_radius)).clamp(0.0, 1.0),
                size,
            ),
        ];
        bilinear(&self.multiscatter, size, size, uv)
    }

    // The sky's radiance toward the point at radius `r` along `-dir`, per unit
    // sunlight from `sun`, marched as the sky-view kernel marches a ray from
    // inside the air.
    fn sky(&self, r: f32, dir: [f32; 3], sun: [f32; 3]) -> [f32; 3] {
        let s = self.s;
        let length = ray_exit(r, dir[1], s.top_radius);
        let dt = length / SKY_STEPS as f32;
        let g = s.mie_anisotropy;
        let nu = dot(dir, sun);
        let (ray_phase, mie_phase) = (rayleigh_phase(nu), mie_phase(nu, g));
        let mut radiance = [0.0_f32; 3];
        let mut throughput = [1.0_f32; 3];
        for i in 0..SKY_STEPS {
            let pos = along([0.0, r, 0.0], dir, (i as f32 + 0.5) * dt);
            let rp = vec3::length(pos);
            let mu_s = dot(pos, sun) / rp;
            let air = s.air.medium_at(rp - s.bottom_radius);
            let to_sun = self.to_top(rp, mu_s);
            let ms = self.multiscatter(rp, mu_s);
            for c in 0..3 {
                let src = (air.rayleigh[c] * ray_phase + air.mie[c] * mie_phase) * to_sun[c]
                    + (air.rayleigh[c] + air.mie[c]) * ms[c];
                let step_t = exp(-air.extinction[c] * dt);
                radiance[c] += throughput[c] * (src - src * step_t) / air.extinction[c].max(1e-7);
                throughput[c] *= step_t;
            }
        }
        radiance
    }

    // What the point at radius `r` receives from a sun at cosine `mu`, along
    // `sun`.
    fn daylight(&self, r: f32, mu: f32, sun: [f32; 3]) -> Daylight {
        let mut irradiance = [0.0_f32; 3];
        let mut horizon = [0.0_f32; 3];
        let solid = (1.0 / RINGS as f32) * (2.0 * PI / AROUND as f32);
        for j in 0..AROUND / 2 {
            let phi = 2.0 * PI * (j as f32 + 0.5) / AROUND as f32;
            for k in 0..RINGS {
                let up = (k as f32 + 0.5) / RINGS as f32;
                let l = self.sky(r, ring_dir(up, phi), sun);
                for c in 0..3 {
                    irradiance[c] += 2.0 * l[c] * up * solid;
                }
            }
            let l = self.sky(r, ring_dir(HORIZON_SIN, phi), sun);
            for c in 0..3 {
                horizon[c] += 2.0 * l[c] / AROUND as f32;
            }
        }
        Daylight {
            transmittance: sun_transmittance(self.s, r, mu),
            irradiance,
            zenith: self.sky(r, [0.0, 1.0, 0.0], sun),
            horizon,
        }
    }
}

// `table`, `w` texels wide, as its rows, each with its index.
fn by_row(table: &mut [[f32; 3]], w: u32) -> Vec<(u32, &mut [[f32; 3]])> {
    (0..).zip(table.chunks_mut(w as usize)).collect()
}

// The direction `up` (cosine) above the horizon, `phi` round from the sun's
// side.
fn ring_dir(up: f32, phi: f32) -> [f32; 3] {
    let across = sqrt((1.0 - up * up).max(0.0));
    [across * cos(phi), up, across * sin(phi)]
}

fn rayleigh_phase(nu: f32) -> f32 {
    3.0 / (16.0 * PI) * (1.0 + nu * nu)
}

// Cornette-Shanks, as `mie_phase` in `atmosphere_common.hlsl`.
fn mie_phase(nu: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * nu).max(1e-4);
    3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + nu * nu)) / ((2.0 + g2) * denom * sqrt(denom))
}

// Distance to the ground along a ray from radius `r` with cosine `mu`, or -1
// when it misses. `ray_ground` in `atmosphere_common.hlsl`.
fn ray_ground(r: f32, mu: f32, bottom: f32) -> f32 {
    let disc = r * r * (mu * mu - 1.0) + bottom * bottom;
    if mu >= 0.0 || disc < 0.0 {
        return -1.0;
    }
    (-r * mu - sqrt(disc)).max(0.0)
}

// A table of `w` by `h` texels read at `uv` with linear filtering and the
// edges clamped, as the GPU samples it.
fn bilinear(table: &[[f32; 3]], w: u32, h: u32, uv: [f32; 2]) -> [f32; 3] {
    let fx = (uv[0] * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (uv[1] * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (u32::saturating_from(fx), u32::saturating_from(fy));
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: u32, y: u32| table[(y * w + x) as usize];
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    core::array::from_fn(|i| {
        let top = a[i] + (b[i] - a[i]) * tx;
        let bottom = c[i] + (d[i] - c[i]) * tx;
        top + (bottom - top) * ty
    })
}

fn along(origin: [f32; 3], dir: [f32; 3], t: f32) -> [f32; 3] {
    core::array::from_fn(|i| origin[i] + dir[i] * t)
}
