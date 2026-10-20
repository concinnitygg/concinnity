//! A CPU mirror of how `water.hlsl` reads a planet sea, line for line in f32:
//! where a rest point falls on each face, each face's share of it, and the
//! coordinates its cascades and ripples are looked up at. Tests check here
//! that the layout the shader sees does not move when the frame does, and
//! does not tear where two faces meet.

use crate::render::uniforms::{WaterParams, WaterSeaFaceGpu};

/// Where a rest point falls on one face (`water_sea_point`).
#[derive(Debug, Clone, Copy)]
pub(super) struct SeaPoint {
    /// Face meters from the anchor's place on the face.
    pub(super) s: [f32; 2],
    /// The point's direction from the center, in the face's axes.
    pub(super) dir: [f32; 3],
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rest offset `rest` from the anchor, in the simulated frame, on `face`.
pub(super) fn sea_point(face: &WaterSeaFaceGpu, inv_radius: f32, rest: [f32; 3]) -> SeaPoint {
    let [r0, r1, r2] = face.rows;
    let a = [r0[3], r1[3], r2[3]];
    let dl = [
        dot([r0[0], r0[1], r0[2]], rest),
        dot([r1[0], r1[1], r1[2]], rest),
        dot([r2[0], r2[1], r2[2]], rest),
    ];
    let dir = core::array::from_fn(|i| a[i] + dl[i] * inv_radius);
    let denom_a = 1.0 + a[1];
    let denom_p = 1.0 + dir[1];
    let k = 2.0 / (denom_p * denom_a);
    let s = [
        k * (dl[0] * denom_a - a[0] * dl[1]),
        k * (dl[2] * denom_a - a[2] * dl[1]),
    ];
    SeaPoint { s, dir }
}

/// Each face's share of a rest point, its squares summing to one
/// (`water_sea_weights`).
pub(super) fn sea_weights(params: &WaterParams, rest: [f32; 3]) -> [f32; 3] {
    let sea = &params.sea;
    let w: [f32; 3] = core::array::from_fn(|f| {
        let d = sea_point(&sea.faces[f], sea.shape[0], rest).dir;
        let margin = d[1] - d[0].abs().max(d[2].abs());
        (0.5 + margin * sea.shape[1]).clamp(0.0, 1.0)
    });
    let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).max(1e-12).sqrt();
    w.map(|x| x / n)
}

/// The texture coordinate a face's cascade `c` is read at for face meters
/// `s` (`water_ocean_uv`), without the half-texel shift.
pub(super) fn cascade_uv(params: &WaterParams, face: usize, c: usize, s: [f32; 2]) -> [f32; 2] {
    let pair = params.sea.faces[face].ocean_offsets[c >> 1];
    let offset = if c & 1 == 0 {
        [pair[0], pair[1]]
    } else {
        [pair[2], pair[3]]
    };
    let inv = params.ocean.inv_lengths[c];
    [s[0] * inv + offset[0], s[1] * inv + offset[1]]
}

/// The noise coordinate ripple octave `r` is read at for face meters `s`,
/// before the domain warp (`water_shading`).
pub(super) fn ripple_cell(params: &WaterParams, face: usize, r: usize, s: [f32; 2]) -> [f32; 2] {
    let shape = params.ripples[r].shape;
    let (dir, wavelength) = ([shape[0], shape[1]], shape[2]);
    let across = [-dir[1], dir[0]];
    let row = params.sea.faces[face].ripple_offsets[r >> 1];
    let offset = if r & 1 == 0 {
        [row[0], row[1]]
    } else {
        [row[2], row[3]]
    };
    let crest = super::super::ripples::RIPPLE_CREST;
    [
        (s[0] * dir[0] + s[1] * dir[1]) / wavelength + offset[0],
        (s[0] * across[0] + s[1] * across[1]) * crest / wavelength + offset[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Planet, WaterShape, WaterSurface, Wind};
    use crate::planet::{DVec3, LocalFrame};
    use crate::render::environment::water::WaterBody;
    use crate::render::environment::water::frame::WaterMotion;
    use crate::render::environment::water::ocean::OceanQuality;
    use crate::render::environment::water::ripples::NOISE_PERIOD;
    use crate::render::environment::wind::WindField;
    use concinnity_core::math::dvec3;
    use concinnity_core::math::{SaturatingFrom, narrow_f32};

    fn sea() -> WaterBody {
        let surface = WaterSurface {
            shape: WaterShape::PlanetSea,
            sea_level: 12.0,
            fetch_meters: 1.0e5,
            ..Default::default()
        };
        let wind = WindField::new(Some(&Wind {
            strength: 9.0,
            ..Wind::default()
        }));
        let planet = Planet::default().shape();
        let (mut bodies, _) =
            WaterBody::resolve_all([&surface], &wind, OceanQuality::FULL, Some(&planet));
        bodies.remove(0)
    }

    fn sphere(body: &WaterBody) -> (DVec3, f64) {
        let s = body.sea.as_ref().unwrap().sphere;
        (s.center, s.radius)
    }

    // The authored point on the sea `meters` along the surface from the top
    // toward `heading` (radians from +X toward +Z).
    fn on_sea(body: &WaterBody, top: DVec3, heading: f64, meters: f64) -> DVec3 {
        let (center, r) = sphere(body);
        let up = dvec3::normalize_or(dvec3::sub(top, center), [0.0, 1.0, 0.0]);
        let side = dvec3::normalize_or(
            dvec3::cross(up, [libm::sin(heading), 0.0, -libm::cos(heading)]),
            [1.0, 0.0, 0.0],
        );
        let toward = dvec3::cross(side, up);
        let a = meters / r;
        let dir = dvec3::add(
            dvec3::scale(up, libm::cos(a)),
            dvec3::scale(toward, libm::sin(a)),
        );
        dvec3::add(center, dvec3::scale(dir, r))
    }

    // How far apart two coordinates on a lattice of `period` are, wrapped.
    fn wrapped(a: f32, b: f32, period: f32) -> f32 {
        let d = (a - b).rem_euclid(period);
        d.min(period - d)
    }

    // Every pattern the shader reads at authored sea point `q`, for the
    // params `p` drawn in `frame`: per live face, its share, the cascades'
    // coordinates and the ripples' cells.
    struct Reading {
        weights: [f32; 3],
        uvs: [[[f32; 2]; 3]; 3],
        cells: [[[f32; 2]; 4]; 3],
    }

    fn read(p: &WaterParams, frame: &LocalFrame, q: DVec3) -> Reading {
        let local = frame.to_local(q);
        let rest = core::array::from_fn(|i| local[i] - p.anchor[i]);
        let weights = sea_weights(p, rest);
        let s: [[f32; 2]; 3] =
            core::array::from_fn(|f| sea_point(&p.sea.faces[f], p.sea.shape[0], rest).s);
        let uvs = core::array::from_fn(|f| core::array::from_fn(|c| cascade_uv(p, f, c, s[f])));
        let cells = core::array::from_fn(|f| core::array::from_fn(|r| ripple_cell(p, f, r, s[f])));
        Reading {
            weights,
            uvs,
            cells,
        }
    }

    // A move of the simulated frame leaves every pattern where it was on the
    // planet: at points beside the camera and kilometers off, each face's
    // share, its cascade lookups and its ripple cells read the same before
    // the move and after it. The moved camera reads from the same anchor.
    #[test]
    fn a_frame_move_leaves_the_waves_where_they_were() {
        let body = sea();
        let mut motion = WaterMotion::new(&body, false);
        let frame = LocalFrame::AUTHORED;
        let camera = [3.0, 30.0, -995.0];
        let (center, _) = sphere(&body);
        let up = dvec3::normalize_or(dvec3::sub(frame.to_world(camera), center), [0.0, 1.0, 0.0]);
        let (_, rebase) = frame.recentered(camera, up);
        let time = 4_321.5;
        let before = motion.params_at(time, camera);
        motion.rebase(&rebase);
        let moved_frame = frame.followed(&rebase);
        let after = motion.params_at(time, rebase.apply_point(camera));
        let anchor =
            |p: &WaterParams, f: &LocalFrame| f.to_world([p.anchor[0], p.anchor[1], p.anchor[2]]);
        let drift = dvec3::length(dvec3::sub(
            anchor(&before, &frame),
            anchor(&after, &moved_frame),
        ));
        assert!(drift < 2e-3, "anchor {drift} m");

        let top = frame.to_world([camera[0], 0.0, camera[2]]);
        for (meters, tolerance) in [(0.5, 1e-4), (40.0, 2e-4), (900.0, 2e-3), (6_000.0, 1e-2)] {
            for heading in [0.3, 2.1, 4.4] {
                let q = on_sea(&body, top, heading, meters);
                let a = read(&before, &frame, q);
                let b = read(&after, &moved_frame, q);
                for f in 0..3 {
                    assert!((a.weights[f] - b.weights[f]).abs() < 1e-5, "weight {f}");
                    if a.weights[f] == 0.0 {
                        continue;
                    }
                    for c in 0..3 {
                        let length = 1.0 / before.ocean.inv_lengths[c];
                        for axis in 0..2 {
                            let d = wrapped(a.uvs[f][c][axis], b.uvs[f][c][axis], 1.0) * length;
                            assert!(d < tolerance, "{meters} m: cascade {c} moved {d} m");
                        }
                    }
                    for r in 0..4 {
                        let wavelength = before.ripples[r].shape[2];
                        for axis in 0..2 {
                            let d = wrapped(a.cells[f][r][axis], b.cells[f][r][axis], NOISE_PERIOD)
                                * wavelength;
                            assert!(d < tolerance, "{meters} m: ripple {r} moved {d} m");
                        }
                    }
                }
            }
        }
    }

    // Walking from the +Y face onto the +X face the shares hand over
    // smoothly across a band at the edge, their squares always summing to
    // one, and a sea laid on each face blends without a step: the blended
    // value moves no more per step than the steepest wave allows.
    #[test]
    fn the_faces_blend_without_a_seam() {
        let body = sea();
        let (center, r) = sphere(&body);
        let edge_angle = core::f64::consts::FRAC_PI_4;
        let start = on_sea(&body, [0.0, 0.0, 0.0], core::f64::consts::FRAC_PI_2, 0.0);
        let at_edge = on_sea(&body, start, core::f64::consts::FRAC_PI_2, edge_angle * r);
        let up = dvec3::normalize_or(dvec3::sub(at_edge, center), [0.0, 1.0, 0.0]);
        let eye = LocalFrame::AUTHORED.to_local(dvec3::add(at_edge, dvec3::scale(up, 30.0)));
        let (_, rebase) = LocalFrame::AUTHORED.recentered(eye, up);
        let frame = LocalFrame::AUTHORED.followed(&rebase);
        let mut motion = WaterMotion::new(&body, false);
        motion.rebase(&rebase);
        let p = motion.params_at(10.0, [0.0, 0.0, 0.0]);

        let length = 1.0 / p.ocean.inv_lengths[0];
        let wave = |uv: [f32; 2]| {
            (core::f32::consts::TAU * uv[0]).sin() + (core::f32::consts::TAU * uv[1]).cos()
        };
        // The steepest the blended value can climb per meter: two unit waves
        // a cascade tile long on each of two faces, plus the shares' turn.
        let step = 0.5;
        let half = sphere_half_band(&body);
        let bound = step * (6.0 * core::f32::consts::TAU / length + 4.0 / narrow_f32(half));

        let mut last: Option<(f32, [f32; 3])> = None;
        let mut saw_blend = false;
        let n = i32::saturating_from(2.0 * 1500.0 / step);
        for i in 0..=n {
            let meters = -1500.0 + f64::from(i) * f64::from(step);
            let q = on_sea(&body, at_edge, core::f64::consts::FRAC_PI_2, meters);
            let reading = read(&p, &frame, q);
            let w = reading.weights;
            let sum: f32 = w.iter().map(|x| x * x).sum();
            assert!((sum - 1.0).abs() < 1e-4, "{meters}: {w:?}");
            if w[0] > 0.1 && w[1] > 0.1 {
                saw_blend = true;
            }
            let value: f32 = (0..3).map(|f| w[f] * wave(reading.uvs[f][0])).sum();
            if let Some((prev, prev_w)) = last {
                assert!((value - prev).abs() < bound, "{meters}: {prev} -> {value}");
                for f in 0..3 {
                    assert!(
                        (w[f] - prev_w[f]).abs() < 0.05,
                        "{meters}: {prev_w:?} -> {w:?}"
                    );
                }
            }
            last = Some((value, w));
        }
        assert!(saw_blend);
        // Well inside each face only that face is read.
        let deep = |m: f64| {
            read(
                &p,
                &frame,
                on_sea(&body, at_edge, core::f64::consts::FRAC_PI_2, m),
            )
        };
        assert_eq!(deep(-1200.0).weights, [0.0, 1.0, 0.0]);
        assert_eq!(deep(1200.0).weights, [1.0, 0.0, 0.0]);
    }

    fn sphere_half_band(body: &WaterBody) -> f64 {
        body.sea.as_ref().unwrap().sphere.blend_half_width()
    }
}
