//! A CPU mirror of how `water.hlsl` reads a planet sea's shore waves, line
//! for line in f32 from the block the shader binds. Tests check here that the
//! shader's waves are the host's, and that they stay where they are on the
//! planet when the simulated frame moves.

use crate::render::uniforms::{WaterParams, WaterShoreGpu};
use concinnity_core::math::narrow_f32;
use core::f32::consts::TAU;

use crate::planet::{CONTINENT_OCTAVES, octave_weight_sum};
use crate::water::sea::shore::{FOAM_RIDE, PHASE_NOISE_OCTAVES, SHORE_TRAINS};
const SHORE_NODES: u32 = 32;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    a.map(|x| x * k)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// `ground_lcg` / `ground_lattice` / `ground_noise` in planet_noise.hlsl.
fn lcg(v: u32) -> u32 {
    let v = v.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    v ^ (v >> 16)
}

fn lattice(c: [i32; 3], salt: u32) -> f32 {
    let h = lcg(lcg(lcg(c[0].cast_unsigned() ^ salt)
        .wrapping_add(c[1].cast_unsigned().wrapping_mul(0x85eb_ca6b)))
    .wrapping_add(c[2].cast_unsigned().wrapping_mul(0xc2b2_ae35)));
    (h & 0xffff) as f32 / 65535.0
}

fn noise(c: [i32; 3], t: [f32; 3], salt: u32) -> [f32; 4] {
    let s = t.map(|t| t * t * (3.0 - 2.0 * t));
    let ds = t.map(|t| 6.0 * t * (1.0 - t));
    let at = |x: i32, y: i32, z: i32| lattice([c[0] + x, c[1] + y, c[2] + z], salt);
    let (a, b, d0, d1) = (at(0, 0, 0), at(1, 0, 0), at(0, 1, 0), at(1, 1, 0));
    let (e, f, g, h) = (at(0, 0, 1), at(1, 0, 1), at(0, 1, 1), at(1, 1, 1));
    let k1 = b - a;
    let k2 = d0 - a;
    let k3 = e - a;
    let k4 = a - b - d0 + d1;
    let k5 = a - d0 - e + g;
    let k6 = a - b - e + f;
    let k7 = -a + b + d0 - d1 + e - f - g + h;
    let value = a
        + k1 * s[0]
        + k2 * s[1]
        + k3 * s[2]
        + k4 * s[0] * s[1]
        + k5 * s[1] * s[2]
        + k6 * s[2] * s[0]
        + k7 * s[0] * s[1] * s[2];
    [
        ds[0] * (k1 + k4 * s[1] + k6 * s[2] + k7 * s[1] * s[2]),
        ds[1] * (k2 + k5 * s[2] + k4 * s[0] + k7 * s[2] * s[0]),
        ds[2] * (k3 + k6 * s[0] + k5 * s[1] + k7 * s[0] * s[1]),
        value,
    ]
}

type Mat = [[f32; 3]; 3];

// `WATER_OCTAVE_TURN`, row by row.
fn turn() -> Mat {
    let t = crate::planet::OCTAVE_TURN;
    core::array::from_fn(|r| core::array::from_fn(|c| narrow_f32(t[c][r])))
}

fn mul(m: &Mat, v: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|r| dot(m[r], v))
}

fn mul_t(v: [f32; 3], m: &Mat) -> [f32; 3] {
    core::array::from_fn(|c| v[0] * m[0][c] + v[1] * m[1][c] + v[2] * m[2][c])
}

fn matmul(a: &Mat, b: &Mat) -> Mat {
    core::array::from_fn(|r| {
        core::array::from_fn(|c| a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c])
    })
}

/// `water_shore_octaves`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the shader's int3(floor(q))"
)]
fn octaves(
    sh: &WaterShoreGpu,
    away: [f32; 3],
    first: usize,
    frequency: f32,
    count: u32,
    salts: [[u32; 4]; 2],
    keep: [f32; 2],
) -> [f32; 4] {
    let turn = turn();
    let mut to_lattice = turn.map(|r| r.map(|x| x * frequency));
    let mut weight = 1.0_f32;
    let mut sum = [0.0_f32; 4];
    for o in 0..count {
        if o >= 2 {
            let rest = 2.0 * weight - (1.0 - count as f32).exp2();
            if sum[3] + rest < keep[0] {
                sum[3] += rest;
                break;
            }
            if sum[3] > keep[1] {
                break;
            }
        }
        let at = first + o as usize;
        let from = sh.lattice_places[at];
        let step = mul(&to_lattice, away);
        let q: [f32; 3] = core::array::from_fn(|i| from[i] + step[i]);
        let whole = q.map(f32::floor);
        let cell: [i32; 3] = core::array::from_fn(|i| sh.lattice_cells[at][i] + whole[i] as i32);
        let t = core::array::from_fn(|i| q[i] - whole[i]);
        let n = noise(cell, t, salts[(o >> 2) as usize][(o & 3) as usize]);
        let g = mul_t([n[0], n[1], n[2]], &to_lattice);
        for i in 0..3 {
            sum[i] += g[i] * weight;
        }
        sum[3] += n[3] * weight;
        to_lattice = matmul(&turn, &to_lattice).map(|r| r.map(|x| x * 2.0));
        weight *= 0.5;
    }
    sum
}

/// `water_shore_point`, from the coast `water_shore_field` found.
#[derive(Debug, Clone, Copy)]
pub(super) struct Point {
    pub(super) depth: f32,
    pub(super) weight: f32,
    pub(super) phases: [f32; 3],
    pub(super) k: [f32; 3],
    pub(super) landward: [f32; 3],
    pub(super) damp: [f32; 4],
}

fn depth(sh: &WaterShoreGpu, inland: f32) -> f32 {
    let ground = if inland >= 0.0 {
        let t = (inland / sh.beach[0]).min(1.0);
        sh.beach[1] * t * (2.0 - t)
    } else {
        -sh.coast[2] * (1.0 - (inland / sh.coast[3]).exp())
    };
    (sh.coast[0] - ground).max(0.0)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the shader's uint(x)"
)]
fn table(sh: &WaterShoreGpu, offshore: f32) -> ([f32; 3], [f32; 3]) {
    let reach = sh.beach[2];
    let u = offshore.max(0.0);
    let last = SHORE_NODES - 1;
    let row = |i: u32| {
        let r = sh.table[i as usize];
        [r[0], r[1], r[2]]
    };
    if u >= reach {
        let k = row(2 * last + 1);
        let p = row(2 * last);
        return (core::array::from_fn(|i| p[i] + k[i] * (u - reach)), k);
    }
    let j = ((last as f32 * (u / reach).sqrt()) as u32).min(last - 1);
    let fa = j as f32 / last as f32;
    let fb = (j + 1) as f32 / last as f32;
    let a = reach * fa * fa;
    let span = reach * fb * fb - a;
    let s = (u - a) / span;
    let (s2, s3) = (s * s, s * s * s);
    let (p0, k0, p1, k1) = (row(2 * j), row(2 * j + 1), row(2 * j + 2), row(2 * j + 3));
    let phase = core::array::from_fn(|i| {
        (2.0 * s3 - 3.0 * s2 + 1.0) * p0[i]
            + (s3 - 2.0 * s2 + s) * span * k0[i]
            + (-2.0 * s3 + 3.0 * s2) * p1[i]
            + (s3 - s2) * span * k1[i]
    });
    (phase, core::array::from_fn(|i| k0[i] + s * (k1[i] - k0[i])))
}

/// `water_shore_place`: a rest point's direction and that less the anchor's.
#[derive(Debug, Clone, Copy)]
pub(super) struct Place {
    pub(super) dir: [f32; 3],
    pub(super) away: [f32; 3],
}

pub(super) fn shore_place(p: &WaterParams, rest: [f32; 3]) -> Place {
    let sh = &p.sea.shore;
    let rows = sh.planet_rows;
    let a = [rows[0][3], rows[1][3], rows[2][3]];
    let l: [f32; 3] =
        core::array::from_fn(|i| dot([rows[i][0], rows[i][1], rows[i][2]], rest) * sh.noise[3]);
    let al = dot(a, l);
    let ll = dot(l, l);
    let n = (1.0 + 2.0 * al + ll).sqrt();
    let m = (2.0 * al + ll) / (n + 1.0);
    let away = core::array::from_fn(|i| (l[i] - a[i] * m) / n);
    Place {
        dir: core::array::from_fn(|i| a[i] + away[i]),
        away,
    }
}

/// `water_shore_field`: `[inland, gradient]`.
pub(super) fn field(p: &WaterParams, at: Place) -> [f32; 4] {
    let sh = &p.sea.shore;
    if sh.continent_octaves == 0 {
        return [0.0; 4];
    }
    let (scale_k, offset) = (sh.continent_scale, sh.continent_offset);
    let keep = [
        (sh.bounds[0] + offset) / scale_k,
        (sh.coast[1] + offset) / scale_k,
    ];
    let f = octaves(
        sh,
        at.away,
        0,
        sh.continent_frequency,
        sh.continent_octaves.min(CONTINENT_OCTAVES),
        sh.continent_salts,
        keep,
    );
    let dir = at.dir;
    let grad = [f[0], f[1], f[2]].map(|x| x * scale_k);
    let g = scale(sub(grad, scale(dir, dot(dir, grad))), sh.noise[2]);
    [f[3] * scale_k - offset, g[0], g[1], g[2]]
}

/// `water_shore_point`.
pub(super) fn point(p: &WaterParams, f: [f32; 4]) -> Point {
    let sh = &p.sea.shore;
    let mut out = Point {
        depth: 0.0,
        weight: 0.0,
        phases: [0.0; 3],
        k: [0.0; 3],
        landward: [0.0; 3],
        damp: [1.0; 4],
    };
    if sh.continent_octaves == 0 || f[0] <= sh.bounds[0] {
        return out;
    }
    let inland = f[0];
    let g = [f[1], f[2], f[3]];
    let gl = length(g);
    out.depth = depth(sh, inland);
    for c in 0..4 {
        let l = sh.cascade_wavelengths[c];
        out.damp[c] = if l > 0.0 {
            smoothstep(0.25 * l, 0.5 * l, out.depth)
        } else {
            1.0
        };
    }
    let offshore = sh.coast[1] - inland;
    let (phases, k) = table(sh, offshore);
    let (peak, reach) = (sh.beach[3], sh.beach[2]);
    out.weight = if out.depth > 0.0 {
        (1.0 - smoothstep(0.25 * peak, 0.5 * peak, out.depth))
            * (1.0 - smoothstep(0.8 * reach, reach, offshore))
            * smoothstep(0.15, 0.3, gl)
    } else {
        0.0
    };
    out.phases = phases.map(|x| x / gl.max(0.15));
    out.k = k;
    out.landward = if gl > 1e-9 {
        scale(g, 1.0 / gl)
    } else {
        [0.0; 3]
    };
    out
}

/// `water_shore_swell`.
pub(super) fn swell(p: &WaterParams, at: Place) -> f32 {
    let sh = &p.sea.shore;
    let salts = [[sh.noise_salts[2], 0, 0, 0], [0; 4]];
    let first = (CONTINENT_OCTAVES + PHASE_NOISE_OCTAVES) as usize;
    let n = octaves(sh, at.away, first, sh.swell[0], 1, salts, [-1.0e30, 1.0e30]);
    1.0 + sh.swell[1] * (2.0 * n[3] - 1.0)
}

/// Where the surf's lace is read at a rest point: meters in the planet's
/// axes, within one lace period.
pub(super) fn lace_point(p: &WaterParams, rest: [f32; 3]) -> [f32; 3] {
    let sh = &p.sea.shore;
    core::array::from_fn(|i| {
        let r = sh.planet_rows[i];
        sh.lace[i] + dot([r[0], r[1], r[2]], rest)
    })
}

/// `water_shore_noise`: `[gradient, radians]`.
pub(super) fn phase_noise(p: &WaterParams, at: Place) -> [f32; 4] {
    let sh = &p.sea.shore;
    let n = octaves(
        sh,
        at.away,
        CONTINENT_OCTAVES as usize,
        sh.noise[0],
        PHASE_NOISE_OCTAVES,
        [sh.noise_salts, [0; 4]],
        [-1.0e30, 1.0e30],
    );
    let k = 2.0 * sh.noise[1] / narrow_f32(octave_weight_sum(PHASE_NOISE_OCTAVES));
    let g = [n[0], n[1], n[2]];
    let dir = at.dir;
    let along = scale(sub(g, scale(dir, dot(dir, g))), k * sh.noise[3]);
    [along[0], along[1], along[2], k * n[3] - sh.noise[1]]
}

/// `water_surf_drift`.
pub(super) fn surf_drift(age: f32, k: f32) -> f32 {
    let a = age.clamp(0.0, 1.0);
    narrow_f32(FOAM_RIDE) * TAU / k.max(1e-4) * a * (1.0 - a) * (1.0 - a)
}

/// `water_shore_surface`: height, shift and slope in the planet's axes, foam.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Surface {
    pub(super) height: f32,
    pub(super) shift: [f32; 3],
    pub(super) slope: [f32; 3],
    pub(super) foam: f32,
}

fn shoaling(omega: f32, k: f32, depth: f32) -> f32 {
    let kh = k * depth;
    let n = if kh > 20.0 {
        0.5
    } else {
        0.5 * (1.0 + 2.0 * kh / (2.0 * kh).sinh())
    };
    (0.5 * 9.81 / omega / (n * omega / k)).sqrt()
}

pub(super) fn surface(p: &WaterParams, pt: &Point, n: [f32; 4], swell: f32) -> Surface {
    let sh = &p.sea.shore;
    if pt.weight <= 0.0 {
        return Surface::default();
    }
    let shallow = pt.depth.max(0.1);
    let (mut re, mut im, mut total, mut energy, mut k_sum) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for i in 0..SHORE_TRAINS {
        let a = pt.weight * swell * sh.amplitudes[i] * shoaling(sh.omegas[i], pt.k[i], shallow);
        let ph = -(sh.phases[i] + pt.phases[i] + n[3]);
        re += a * ph.cos();
        im += a * ph.sin();
        total += a;
        energy += a * a;
        k_sum += a * pt.k[i];
    }
    let envelope = (re * re + im * im).sqrt();
    let carrier = if envelope > 0.0 { im.atan2(re) } else { 0.0 };
    let travel = sub(
        scale(pt.landward, k_sum / total.max(1e-12)),
        [n[0], n[1], n[2]],
    );
    let k = length(travel);
    if k <= 1e-9 {
        return Surface::default();
    }
    let toward = scale(travel, 1.0 / k);
    let cap = (0.78 * pt.depth).max(1e-6);
    let ratio = 2.0 * envelope / cap;
    let held = envelope.min(0.5 * cap);
    let lean = 0.75 * smoothstep(0.5, 1.0, ratio);
    let u = carrier + lean * (1.0 - carrier.cos());
    let du = 1.0 + lean * carrier.sin();
    let pinch = 0.9 * smoothstep(0.4, 1.0, ratio) / (1.0 + lean);
    let (su, cu) = (u.sin(), u.cos());
    let rise = -held * su * du * k;
    let spread = -pinch * cu * du;
    let mut since = -carrier;
    since -= TAU * (since / TAU).floor();
    let trail = (-since / (TAU * 0.45)).exp();
    let lead = smoothstep(TAU - 0.5, TAU, since);
    let crest = smoothstep(0.85, 1.15, ratio) * trail.max(lead);
    let rms = 2.0 * energy.sqrt() / cap;
    Surface {
        height: held * cu,
        shift: scale(toward, -pinch * su / k),
        slope: scale(toward, rise / (1.0 + spread)),
        foam: crest.max(0.2 * smoothstep(0.8, 1.5, rms)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, Wind};
    use crate::planet::PlanetShape;
    use crate::planet::{DVec3, LocalFrame};
    use crate::render::environment::water::frame::WaterMotion;
    use crate::water::WaterBody;
    use crate::water::ocean::OceanQuality;
    use crate::water::sea::shore::tests::{coast_planet, coast_planet_of, coast_point};
    use crate::water::sea::shore::{
        BREAKER_INDEX, FOAM_LEAD, FOAM_RIDE, FOAM_TRAIL, GRADIENT_FADE, MAX_LEAN, MAX_PINCH,
        RESIDUAL_FOAM, SHALLOWEST, SHORE_OCTAVES,
    };
    use crate::water::spectrum::GRAVITY;
    use crate::wind::WindField;
    use alloc::format;
    use concinnity_core::math::dvec3;

    fn sea() -> WaterBody {
        sea_on(&coast_planet())
    }

    fn sea_on(planet: &PlanetShape) -> WaterBody {
        let surface = WaterSurface {
            fetch_meters: 50_000.0,
            wind_response: 1.75,
            ..Default::default()
        };
        let wind = WindField::new(Some(&Wind {
            strength: 4.0,
            ..Wind::default()
        }));
        let sea = crate::water::PlanetWater {
            planet: planet.clone(),
            level: 0.0,
            surface: &surface,
        };
        let (mut bodies, _) = WaterBody::resolve_all([], &wind, OceanQuality::FULL, Some(sea));
        bodies.remove(0)
    }

    // Points on the sea `meters` out from the coast at `dir` (toward the sea,
    // against `landward`), and as many along the coast.
    fn around(dir: DVec3, landward: DVec3, r: f64) -> impl Iterator<Item = DVec3> {
        let side = dvec3::cross(dir, landward);
        [-2.0, 3.0, 12.0, 40.0, 95.0, 180.0, 290.0, 900.0]
            .into_iter()
            .flat_map(move |out| {
                [-170.0, 0.0, 60.0].into_iter().map(move |along| {
                    let t = dvec3::add(dvec3::scale(landward, -out), dvec3::scale(side, along));
                    dvec3::normalize_or(dvec3::add(dir, dvec3::scale(t, 1.0 / r)), dir)
                })
            })
    }

    fn rest_of(p: &WaterParams, frame: &LocalFrame, center: DVec3, r: f64, dir: DVec3) -> [f32; 3] {
        let q = frame.to_local(dvec3::add(center, dvec3::scale(dir, r)));
        core::array::from_fn(|i| q[i] - p.anchor[i])
    }

    // The shader's constants are the host's.
    #[test]
    fn the_shader_constants_match() {
        let src = crate::render::shader::embedded::WATER;
        for (name, value) in [
            ("WATER_BREAKER_INDEX", BREAKER_INDEX),
            ("WATER_SHALLOWEST", SHALLOWEST),
            ("WATER_MAX_LEAN", MAX_LEAN),
            ("WATER_MAX_PINCH", MAX_PINCH),
            ("WATER_FOAM_TRAIL", FOAM_TRAIL),
            ("WATER_FOAM_LEAD", FOAM_LEAD),
            ("WATER_RESIDUAL_FOAM", RESIDUAL_FOAM),
            ("WATER_FOAM_RIDE", FOAM_RIDE),
        ] {
            let line = format!("static const float {name} = {value};");
            assert!(src.contains(&line), "{line}");
        }
        let fade = format!(
            "static const float2 WATER_GRADIENT_FADE = float2({}, {});",
            GRADIENT_FADE[0], GRADIENT_FADE[1]
        );
        assert!(src.contains(&fade), "{fade}");
        for line in [
            format!("static const uint WATER_SHORE_TRAINS = {SHORE_TRAINS}u;"),
            format!("static const uint WATER_CONTINENT_OCTAVES = {CONTINENT_OCTAVES}u;"),
            format!("static const uint WATER_SHORE_OCTAVES = {SHORE_OCTAVES}u;"),
            format!("static const uint WATER_PHASE_NOISE_OCTAVES = {PHASE_NOISE_OCTAVES}u;"),
            format!(
                "static const float WATER_PHASE_NOISE_WEIGHT = {};",
                octave_weight_sum(PHASE_NOISE_OCTAVES)
            ),
            format!("static const float WATER_GRAVITY = {GRAVITY};"),
        ] {
            assert!(src.contains(&line), "{line}");
        }
        for column in crate::planet::OCTAVE_TURN {
            for x in column {
                assert!(src.contains(&format!("{x:.12}")), "{x}");
            }
        }
    }

    // The shader reads the host's shore: at points across the surf and out
    // to sea, its depth, phases, height, slope and foam are the host's.
    #[test]
    fn the_shader_reads_the_hosts_shore() {
        let body = sea();
        let shore = body.sea.as_ref().unwrap().shore.as_ref().unwrap();
        let planet = coast_planet();
        let (dir, landward) = coast_point(&planet);
        let sphere = body.sea.as_ref().unwrap().sphere;
        let r = sphere.radius;
        let frame = LocalFrame::AUTHORED;
        let camera = frame.to_local(dvec3::add(sphere.center, dvec3::scale(dir, r + 3.0)));
        let time = 1_234.5;
        let p = WaterMotion::new(&body, false).params_at(time, camera);
        let mut surf = 0;
        for d in around(dir, landward, r) {
            let rest = rest_of(&p, &frame, sphere.center, r, d);
            let at = shore_place(&p, rest);
            let got = point(&p, field(&p, at));
            let (inland, gradient) = planet.inland(d).unwrap();
            let want = shore.point(inland, gradient);
            // Past the shore's reach the shader keeps only a bound on the
            // distance inland, which reads the same.
            if inland > shore.quiet_inland() && inland < shore.coast().waterline {
                let depth_tol = 0.02 + 1e-3 * want.depth;
                assert!(
                    (f64::from(got.depth) - want.depth).abs() < depth_tol,
                    "{} {}",
                    got.depth,
                    want.depth
                );
            }
            let want_damp = shore.cascade_damping(want.depth);
            for (got, want) in got.damp.iter().zip(want_damp) {
                assert!((f64::from(*got) - want).abs() < 0.02, "{got} {want}");
            }
            assert!(
                (f64::from(got.weight) - want.weight).abs() < 0.02,
                "{} {}",
                got.weight,
                want.weight
            );
            if want.weight == 0.0 {
                continue;
            }
            for i in 0..3 {
                let tol = 0.02 + 2e-4 * want.phases[i].abs();
                assert!(
                    (f64::from(got.phases[i]) - want.phases[i]).abs() < tol,
                    "{i}"
                );
            }
            let n = phase_noise(&p, at);
            let wn = shore.noise_at(d);
            assert!((f64::from(n[3]) - wn.phase).abs() < 1e-3);
            let sw = swell(&p, at);
            assert!((f64::from(sw) - wn.swell).abs() < 1e-4, "{sw} {}", wn.swell);
            let s = surface(&p, &got, n, sw);
            let w = shore.surface(&want, &shore.noise_at(d), shore.phases_at(time));
            let h = 0.05 * (0.5 * BREAKER_INDEX * want.depth).max(0.05);
            assert!(
                (f64::from(s.height) - w.height).abs() < h,
                "{} {}",
                s.height,
                w.height
            );
            assert!(
                (f64::from(s.foam) - w.foam).abs() < 0.1,
                "{} {}",
                s.foam,
                w.foam
            );
            for a in 0..3 {
                let slope = 0.05 + 0.1 * w.slope[a].abs();
                assert!(
                    (f64::from(s.slope[a]) - w.slope[a]).abs() < slope,
                    "{:?} {:?}",
                    s.slope,
                    w.slope
                );
                assert!(
                    (f64::from(s.shift[a]) - w.shift[a]).abs() < 0.05,
                    "{:?} {:?}",
                    s.shift,
                    w.shift
                );
            }
            surf += usize::from(w.foam > 0.0);
        }
        assert!(surf > 0);
    }

    // A move of the simulated frame leaves the shore waves where they were
    // on the planet: every point reads the same depth, phases and waves
    // before the move and after it.
    #[test]
    fn a_frame_move_leaves_the_shore_where_it_was() {
        let body = sea();
        let planet = coast_planet();
        let (dir, landward) = coast_point(&planet);
        let sphere = body.sea.as_ref().unwrap().sphere;
        let r = sphere.radius;
        let frame = LocalFrame::AUTHORED;
        let camera = frame.to_local(dvec3::add(sphere.center, dvec3::scale(dir, r + 3.0)));
        let up = dvec3::normalize_or(
            dvec3::sub(frame.to_world(camera), sphere.center),
            [0.0, 1.0, 0.0],
        );
        let (_, rebase) = frame.recentered(camera, up);
        let mut motion = WaterMotion::new(&body, false);
        let time = 4_321.5;
        let before = motion.params_at(time, camera);
        motion.rebase(&rebase);
        let moved = frame.followed(&rebase);
        let after = motion.params_at(time, rebase.apply_point(camera));
        let read = |p: &WaterParams, f: &LocalFrame, d: DVec3| {
            let rest = rest_of(p, f, sphere.center, r, d);
            let at = shore_place(p, rest);
            let surf = surface(p, &point(p, field(p, at)), phase_noise(p, at), swell(p, at));
            (at.dir, point(p, field(p, at)), surf, lace_point(p, rest))
        };
        let period = before.sea.shore.lace[3];
        assert!(
            period > 0.0
                && before.sea.shore.lace[..3]
                    .iter()
                    .all(|x| x.abs() <= 0.5 * period)
        );
        for d in around(dir, landward, r) {
            let (da, a, sa, la) = read(&before, &frame, d);
            let (db, b, sb, lb) = read(&after, &moved, d);
            assert!(length(sub(da, db)) < 1e-6);
            // The surf's lace lies at the same place in its period.
            for i in 0..3 {
                let gap = (la[i] - lb[i]).rem_euclid(period);
                assert!(gap.min(period - gap) < 2e-3, "{la:?} {lb:?}");
            }
            assert!(
                (a.depth - b.depth).abs() < 1e-3 + 1e-4 * a.depth,
                "{} {}",
                a.depth,
                b.depth
            );
            for i in 0..3 {
                assert!((a.phases[i] - b.phases[i]).abs() < 5e-3, "{i}");
            }
            assert!(
                (sa.height - sb.height).abs() < 2e-3,
                "{} {}",
                sa.height,
                sb.height
            );
            assert!((sa.foam - sb.foam).abs() < 1e-2);
        }
        // Without the move carried, the frame's planet rows go stale and the
        // points read the shore somewhere else.
        let stale = WaterMotion::new(&body, false).params_at(time, rebase.apply_point(camera));
        let d = around(dir, landward, r).nth(9).unwrap();
        let (da, ..) = read(&before, &frame, d);
        let (db, ..) = read(&stale, &moved, d);
        assert!(length(sub(da, db)) > 1e-5);
    }

    // On an Earth-sized planet the shader still reads the host's shore, and
    // a snap of the grid's anchor leaves every point where it was: each
    // point's direction and lattice steps are taken from the anchor's, not
    // rounded with it.
    #[test]
    fn the_shore_holds_still_on_an_earth_sized_planet() {
        let planet = coast_planet_of(6_371_000.0);
        let body = sea_on(&planet);
        let shore = body.sea.as_ref().unwrap().shore.as_ref().unwrap();
        let (dir, landward) = coast_point(&planet);
        let sphere = body.sea.as_ref().unwrap().sphere;
        let r = sphere.radius;
        let above = |d: DVec3| dvec3::add(sphere.center, dvec3::scale(d, r + 3.0));
        let frame = LocalFrame::AUTHORED;
        let camera = frame.to_local(above(dir));
        let (_, rebase) = frame.recentered(camera, dir);
        let near = frame.followed(&rebase);
        let mut motion = WaterMotion::new(&body, false);
        motion.rebase(&rebase);
        let time = 2_024.25;
        let side = dvec3::normalize_or(dvec3::cross(dir, landward), [1.0, 0.0, 0.0]);
        let away = dvec3::normalize_or(dvec3::add(dir, dvec3::scale(side, 230.0 / r)), dir);
        let a = motion.params_at(time, near.to_local(above(dir)));
        let b = motion.params_at(time, near.to_local(above(away)));
        assert_ne!(a.anchor, b.anchor);
        let read = |p: &WaterParams, d: DVec3| {
            let at = shore_place(p, rest_of(p, &near, sphere.center, r, d));
            (at, field(p, at), point(p, field(p, at)))
        };
        let mut checked = 0;
        for d in around(dir, landward, r) {
            let (at_a, field_a, pa) = read(&a, d);
            let (_, field_b, pb) = read(&b, d);
            let (inland, gradient) = planet.inland(d).unwrap();
            if inland < shore.quiet_inland() || inland > shore.coast().waterline {
                continue;
            }
            checked += 1;
            // Within a centimeter of the host, and of itself across the snap.
            let host = f64::from(field_a[0]) - inland;
            assert!(host.abs() < 0.01, "{} {inland}", field_a[0]);
            assert!(
                (field_a[0] - field_b[0]).abs() < 0.01,
                "{field_a:?} {field_b:?}"
            );
            let want = shore.point(inland, gradient);
            for i in 0..SHORE_TRAINS {
                assert!(
                    (f64::from(pa.phases[i]) - want.phases[i]).abs() < 0.02,
                    "{i}"
                );
                assert!((pa.phases[i] - pb.phases[i]).abs() < 0.01, "{i}");
            }
            let wn = shore.noise_at(d);
            assert!((f64::from(phase_noise(&a, at_a)[3]) - wn.phase).abs() < 1e-3);
            assert!((f64::from(swell(&a, at_a)) - wn.swell).abs() < 1e-4);
        }
        assert!(checked > 6, "{checked}");
    }

    // The shader's surf foam drifts as the host's: at every age and on
    // short and long waves the lace is read from as far back along the
    // travel, so it moves shoreward with age and runs on unbroken as the
    // next crest breaks.
    #[test]
    fn the_shaders_foam_rides_with_the_bore() {
        for k in [0.05_f32, 0.2, 0.6] {
            for step in 0..=200 {
                let age = step as f32 / 200.0;
                let want = crate::water::sea::shore::foam_drift(f64::from(age), f64::from(k));
                assert!((f64::from(surf_drift(age, k)) - want).abs() < 1e-4 * (1.0 + want));
            }
            let toward = [0.6, 0.0, 0.8];
            let read = |age: f32| sub([3.0, 1.0, -2.0], scale(toward, surf_drift(age, k)));
            assert!(dot(sub(read(0.0), read(0.2)), toward) > 0.0);
            assert!(length(sub(read(0.9999), read(0.0))) < 1e-2);
        }
    }
}
