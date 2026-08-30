// Extrude a 2D profile in the XZ plane along Y.
//
// Authored by the macOS Mesh Editor: users sketch a polygon in the top-down
// view and pick an extrude height plus an optional uniform corner radius.
// Output is a closed mesh with a flat top, flat bottom, and one flat-shaded
// quad per profile edge.
//
// Concave (reflex) corners and corners where the radius would exceed the
// available edge length are passed through as-is rather than rounded.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::Vert;
use crate::math::{acos, atan2, cos, sin, sqrt, tan};

/// Extrude `profile` (an `[x, z]` polygon of at least 3 points, either
/// winding) by `height` along Y, optionally rounding convex corners with
/// `corner_radius` sampled at `corner_segments` arc points per corner.
pub fn build_extrude(
    profile: &[[f32; 2]],
    height: f32,
    corner_radius: f32,
    corner_segments: u32,
) -> Result<(Vec<Vert>, Vec<u16>), String> {
    let mut profile: Vec<[f32; 2]> = profile.to_vec();
    if profile.len() < 3 {
        return Err(format!(
            "extrude profile must have at least 3 points, got {}",
            profile.len()
        ));
    }

    if !height.is_finite() || height <= 0.0 {
        return Err(format!(
            "extrude height must be a positive number, got {height}"
        ));
    }

    if !corner_radius.is_finite() || corner_radius < 0.0 {
        return Err(format!(
            "extrude corner_radius must be non-negative, got {corner_radius}"
        ));
    }
    let corner_segments = corner_segments.max(1) as usize;

    // Normalise to CCW-math (positive shoelace area in (x, z)). Ear clipping
    // assumes this orientation; the top-cap triangle indices are emitted in
    // reversed winding so the geometric normal still resolves to +Y.
    if signed_area(&profile) < 0.0 {
        profile.reverse();
    }

    if corner_radius > 0.0 {
        profile = round_corners(&profile, corner_radius, corner_segments);
        if profile.len() < 3 {
            return Err("extrude profile collapsed below 3 points after rounding".into());
        }
    }

    let n = profile.len();
    let total_verts = n * 6; // top n + bottom n + 4 per side wall * n walls
    if total_verts > 65536 {
        return Err(format!(
            "extrude profile of {n} points produces {total_verts} vertices, exceeding the u16 limit"
        ));
    }

    let half_h = height / 2.0;
    let top_color = [0.78f32, 0.76, 0.74];
    let bot_color = [0.66f32, 0.64, 0.62];
    let side_color = [0.72f32, 0.70, 0.68];

    let mut verts: Vec<Vert> = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    // Top cap (y = +half_h, normal +Y). Planar UV uses XZ directly.
    let top_base = verts.len() as u16;
    for &[x, z] in &profile {
        verts.push(([x, half_h, z], [0.0, 1.0, 0.0], top_color, [x, z]));
    }
    let top_tris = ear_clip(&profile)?;
    for &[a, b, c] in &top_tris {
        // Reversed winding so the face normal matches the per-vertex +Y.
        idxs.extend_from_slice(&[
            top_base + c as u16,
            top_base + b as u16,
            top_base + a as u16,
        ]);
    }

    // Bottom cap (y = -half_h, normal -Y). Original winding gives -Y face normal.
    let bot_base = verts.len() as u16;
    for &[x, z] in &profile {
        verts.push(([x, -half_h, z], [0.0, -1.0, 0.0], bot_color, [x, z]));
    }
    for &[a, b, c] in &top_tris {
        idxs.extend_from_slice(&[
            bot_base + a as u16,
            bot_base + b as u16,
            bot_base + c as u16,
        ]);
    }

    // Side walls. One flat-shaded quad per profile edge with its own normal.
    for i in 0..n {
        let p0 = profile[i];
        let p1 = profile[(i + 1) % n];
        let dx = p1[0] - p0[0];
        let dz = p1[1] - p0[1];
        let len = sqrt(dx * dx + dz * dz).max(1e-6);
        // Outward normal for CCW-math polygon: rotate edge direction -90° in XZ.
        let normal = [dz / len, 0.0, -dx / len];

        let base = verts.len() as u16;
        verts.push(([p0[0], -half_h, p0[1]], normal, side_color, [0.0, 0.0]));
        verts.push(([p0[0], half_h, p0[1]], normal, side_color, [0.0, height]));
        verts.push(([p1[0], half_h, p1[1]], normal, side_color, [len, height]));
        verts.push(([p1[0], -half_h, p1[1]], normal, side_color, [len, 0.0]));
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    Ok((verts, idxs))
}

fn signed_area(profile: &[[f32; 2]]) -> f32 {
    let mut a = 0.0f32;
    for i in 0..profile.len() {
        let p = profile[i];
        let q = profile[(i + 1) % profile.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * a
}

// Round each convex corner of a CCW-math polygon with the given radius.
//
// Concave (reflex) corners and corners where the tangent distance would
// exceed half the adjacent edge length are passed through unchanged.
fn round_corners(profile: &[[f32; 2]], radius: f32, segments: usize) -> Vec<[f32; 2]> {
    let n = profile.len();
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(n * (segments + 1));
    for i in 0..n {
        let prev = profile[(i + n - 1) % n];
        let curr = profile[i];
        let next = profile[(i + 1) % n];
        let in_dx = curr[0] - prev[0];
        let in_dz = curr[1] - prev[1];
        let out_dx = next[0] - curr[0];
        let out_dz = next[1] - curr[1];
        let in_len = sqrt(in_dx * in_dx + in_dz * in_dz);
        let out_len = sqrt(out_dx * out_dx + out_dz * out_dz);
        if in_len < 1e-6 || out_len < 1e-6 {
            out.push(curr);
            continue;
        }
        let in_ux = in_dx / in_len;
        let in_uz = in_dz / in_len;
        let out_ux = out_dx / out_len;
        let out_uz = out_dz / out_len;
        let cross = in_ux * out_uz - in_uz * out_ux;
        let dot = in_ux * out_ux + in_uz * out_uz;
        if cross < 1e-6 {
            // Straight or right turn (concave): no rounding for this corner.
            out.push(curr);
            continue;
        }
        let phi = acos(dot.clamp(-1.0, 1.0));
        let half_phi = phi / 2.0;
        let tan_half = tan(half_phi);
        if tan_half < 1e-6 {
            out.push(curr);
            continue;
        }
        let t = radius * tan_half;
        let max_t = in_len.min(out_len) * 0.5;
        if t > max_t {
            out.push(curr);
            continue;
        }
        let tin = [curr[0] - t * in_ux, curr[1] - t * in_uz];
        let tout = [curr[0] + t * out_ux, curr[1] + t * out_uz];
        // Arc center sits perpendicular-left of the incoming edge at radius r.
        let cx = tin[0] + radius * (-in_uz);
        let cz = tin[1] + radius * in_ux;
        let start = atan2(tin[1] - cz, tin[0] - cx);
        let mut delta = atan2(tout[1] - cz, tout[0] - cx) - start;
        while delta > core::f32::consts::PI {
            delta -= core::f32::consts::TAU;
        }
        while delta < -core::f32::consts::PI {
            delta += core::f32::consts::TAU;
        }
        for s in 0..=segments {
            let theta = start + delta * (s as f32 / segments as f32);
            out.push([cx + radius * cos(theta), cz + radius * sin(theta)]);
        }
    }
    out
}

// Ear clipping triangulation for a simple CCW-math polygon.
//
// Falls back to a fan triangulation if no ear is found within a generous
// guard; better to deliver a slightly degenerate mesh than to fail the
// build on unusual user input.
fn ear_clip(profile: &[[f32; 2]]) -> Result<Vec<[usize; 3]>, String> {
    let n = profile.len();
    if n < 3 {
        return Err("ear_clip needs at least 3 vertices".into());
    }
    let mut indices: Vec<usize> = (0..n).collect();
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(n.saturating_sub(2));
    let mut guard = 0usize;
    while indices.len() > 3 {
        let m = indices.len();
        let mut clipped = false;
        for i in 0..m {
            let i0 = indices[(i + m - 1) % m];
            let i1 = indices[i];
            let i2 = indices[(i + 1) % m];
            let a = profile[i0];
            let b = profile[i1];
            let c = profile[i2];
            let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
            if cross <= 0.0 {
                continue;
            }
            let mut contains = false;
            for &j in indices.iter() {
                if j == i0 || j == i1 || j == i2 {
                    continue;
                }
                if point_in_triangle(profile[j], a, b, c) {
                    contains = true;
                    break;
                }
            }
            if contains {
                continue;
            }
            tris.push([i0, i1, i2]);
            indices.remove(i);
            clipped = true;
            break;
        }
        guard += 1;
        if !clipped || guard > n * n {
            tris.clear();
            for k in 1..n - 1 {
                tris.push([0, k, k + 1]);
            }
            return Ok(tris);
        }
    }
    if indices.len() == 3 {
        tris.push([indices[0], indices[1], indices[2]]);
    }
    Ok(tris)
}

fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let d1 = side_sign(p, a, b);
    let d2 = side_sign(p, b, c);
    let d3 = side_sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn side_sign(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrude_square_produces_caps_and_walls() {
        let profile = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let (verts, idxs) = build_extrude(&profile, 2.0, 0.0, 8).unwrap();
        // Square: top + bottom (4 each) + 4 side walls (4 verts each) = 24 verts.
        assert_eq!(verts.len(), 24);
        assert_eq!(idxs.len() % 3, 0);
    }

    #[test]
    fn extrude_rejects_too_few_points_and_bad_scalars() {
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert!(build_extrude(&tri[..2], 1.0, 0.0, 8).is_err());
        assert!(build_extrude(&tri, 0.0, 0.0, 8).is_err());
        assert!(build_extrude(&tri, f32::NAN, 0.0, 8).is_err());
        assert!(build_extrude(&tri, 1.0, -0.5, 8).is_err());
    }

    #[test]
    fn corner_rounding_adds_arc_points_on_convex_corners() {
        let square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let sharp = build_extrude(&square, 1.0, 0.0, 4).unwrap().0.len();
        let rounded = build_extrude(&square, 1.0, 0.2, 4).unwrap().0.len();
        assert!(rounded > sharp, "rounded {rounded} <= sharp {sharp}");
    }

    // Either winding is accepted: a clockwise profile is normalised to the
    // orientation ear clipping assumes, so it extrudes to the same mesh as the
    // counter-clockwise one.
    #[test]
    fn a_clockwise_profile_extrudes_the_same_as_its_reverse() {
        let ccw = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let mut cw = ccw;
        cw.reverse();
        let (a, ai) = build_extrude(&ccw, 2.0, 0.0, 8).expect("ccw extrudes");
        let (b, bi) = build_extrude(&cw, 2.0, 0.0, 8).expect("cw extrudes");
        assert_eq!(a.len(), b.len());
        assert_eq!(ai.len(), bi.len());
    }

    // Indices are u16, so a profile whose extrusion would need more than
    // 65536 vertices is refused rather than silently wrapping.
    #[test]
    fn a_profile_too_large_for_u16_indices_is_refused() {
        let profile: Vec<[f32; 2]> = (0..11_000)
            .map(|i| {
                let a = i as f32 * core::f32::consts::TAU / 11_000.0;
                [cos(a), sin(a)]
            })
            .collect();
        let err = build_extrude(&profile, 1.0, 0.0, 4).expect_err("too many vertices");
        assert!(err.contains("u16 limit"), "{err}");
    }

    // Concave corners and corners where the radius does not fit are passed
    // through unrounded rather than failing the build or cutting the shape.
    #[test]
    fn rounding_skips_the_corners_it_cannot_round() {
        // An L: the reflex corner cannot be rounded, the convex ones can.
        let l_shape = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let sharp = build_extrude(&l_shape, 1.0, 0.0, 4).expect("sharp").0.len();
        let rounded = build_extrude(&l_shape, 1.0, 0.1, 4)
            .expect("rounded")
            .0
            .len();
        assert!(rounded > sharp, "the convex corners still round");

        // A radius far larger than any edge: every corner is left alone, so
        // the mesh matches the unrounded one.
        let huge = build_extrude(&l_shape, 1.0, 50.0, 4)
            .expect("an unroundable radius is not an error")
            .0
            .len();
        assert_eq!(huge, sharp, "no corner had room for the arc");
    }

    // A repeated point is a zero-length edge with no direction to round
    // against, so that corner is passed through.
    #[test]
    fn a_repeated_point_is_passed_through_rather_than_rounded() {
        let profile = [[0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let (verts, idxs) =
            build_extrude(&profile, 1.0, 0.2, 4).expect("a degenerate edge is not an error");
        assert!(!verts.is_empty());
        assert_eq!(idxs.len() % 3, 0);
    }

    // A concave profile makes the clipper skip reflex vertices and reject
    // candidate ears that contain another vertex, so it exercises both
    // rejections rather than clipping the first corner it looks at.
    #[test]
    fn ear_clipping_triangulates_a_concave_profile() {
        let l_shape = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let tris = ear_clip(&l_shape).expect("a concave profile triangulates");
        // A simple polygon of n points triangulates into n - 2 triangles.
        assert_eq!(tris.len(), l_shape.len() - 2);
        for t in &tris {
            assert!(t.iter().all(|&i| i < l_shape.len()));
        }
    }

    #[test]
    fn ear_clipping_needs_a_polygon() {
        let err = ear_clip(&[[0.0, 0.0], [1.0, 0.0]]).expect_err("two points are not a polygon");
        assert!(err.contains("at least 3"), "{err}");
    }
}
