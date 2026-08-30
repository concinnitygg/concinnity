// Box, cylinder, plane, and sphere generators.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::Vert;
use crate::math::{cos, sin, sin_cos};

// A box face: four corner positions, the outward normal, then the face width
// and height (used to scale UV tiling).
type BoxFace = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 3], f32, f32);

/// Build an axis-aligned box from half-extents `[x, y, z]`.
///
/// All six faces are included, wound CCW from the outside. Each face carries
/// its outward-facing normal. UV coordinates tile once across each face,
/// scaling with the face dimensions (one repeat per metre).
pub fn build_box(half_extents: [f32; 3]) -> (Vec<Vert>, Vec<u16>) {
    let [hx, hy, hz] = half_extents;

    let color = [0.75f32, 0.74, 0.72];

    let mut verts: Vec<Vert> = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    let faces: &[BoxFace] = &[
        // +Y (top)
        (
            [-hx, hy, hz],
            [hx, hy, hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [0.0, 1.0, 0.0],
            hx * 2.0,
            hz * 2.0,
        ),
        // -Y (bottom)
        (
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
            [0.0, -1.0, 0.0],
            hx * 2.0,
            hz * 2.0,
        ),
        // +Z (front)
        (
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
            [0.0, 0.0, 1.0],
            hx * 2.0,
            hy * 2.0,
        ),
        // -Z (back)
        (
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, -hz],
            [0.0, 0.0, -1.0],
            hx * 2.0,
            hy * 2.0,
        ),
        // +X (right)
        (
            [hx, -hy, hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [hx, hy, hz],
            [1.0, 0.0, 0.0],
            hz * 2.0,
            hy * 2.0,
        ),
        // -X (left)
        (
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
            [-1.0, 0.0, 0.0],
            hz * 2.0,
            hy * 2.0,
        ),
    ];

    for (a, b, c, d, normal, u_max, v_max) in faces {
        let base = verts.len() as u16;
        verts.extend_from_slice(&[
            (*a, *normal, color, [0.0, 0.0]),
            (*b, *normal, color, [*u_max, 0.0]),
            (*c, *normal, color, [*u_max, *v_max]),
            (*d, *normal, color, [0.0, *v_max]),
        ]);
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    (verts, idxs)
}

/// Build an upright cylinder from radius, height, and segment count.
///
/// Centred on the origin: bottom cap at y = -height/2, top cap at y = +height/2.
/// Sides use cylindrical UV projection; caps use planar UV. `segments` is
/// clamped to at least 3.
pub fn build_cylinder(radius: f32, height: f32, segments: u32) -> (Vec<Vert>, Vec<u16>) {
    let segments = segments.max(3) as usize;

    let half_h = height / 2.0;
    let side_color = [0.70f32, 0.68, 0.66];
    let cap_color = [0.60f32, 0.58, 0.56];

    let mut verts: Vec<Vert> = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    // sides: two rings (bottom then top)
    let side_base = verts.len() as u16;
    for ring in 0..=1 {
        let y = if ring == 0 { -half_h } else { half_h };
        for i in 0..segments {
            let t = i as f32 / segments as f32;
            let angle = t * core::f32::consts::TAU;
            let nx = cos(angle);
            let nz = sin(angle);
            let u = t * core::f32::consts::TAU * radius;
            let v = if ring == 0 { height } else { 0.0 };
            verts.push((
                [nx * radius, y, nz * radius],
                [nx, 0.0, nz],
                side_color,
                [u, v],
            ));
        }
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        let b = side_base + i as u16;
        let bn = side_base + next as u16;
        let t = b + segments as u16;
        let tn = bn + segments as u16;
        idxs.extend_from_slice(&[b, bn, tn, tn, t, b]);
    }

    // top cap; normal = +Y
    let top_base = verts.len() as u16;
    verts.push(([0.0, half_h, 0.0], [0.0, 1.0, 0.0], cap_color, [0.5, 0.5]));
    for i in 0..segments {
        let t = i as f32 / segments as f32;
        let angle = t * core::f32::consts::TAU;
        let x = cos(angle) * radius;
        let z = sin(angle) * radius;
        verts.push((
            [x, half_h, z],
            [0.0, 1.0, 0.0],
            cap_color,
            [0.5 + x / (radius * 2.0), 0.5 + z / (radius * 2.0)],
        ));
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        idxs.extend_from_slice(&[
            top_base,
            top_base + 1 + next as u16,
            top_base + 1 + i as u16,
        ]);
    }

    // bottom cap; normal = -Y
    let bot_base = verts.len() as u16;
    verts.push(([0.0, -half_h, 0.0], [0.0, -1.0, 0.0], cap_color, [0.5, 0.5]));
    for i in 0..segments {
        let t = i as f32 / segments as f32;
        let angle = t * core::f32::consts::TAU;
        let x = cos(angle) * radius;
        let z = sin(angle) * radius;
        verts.push((
            [x, -half_h, z],
            [0.0, -1.0, 0.0],
            cap_color,
            [0.5 + x / (radius * 2.0), 0.5 + z / (radius * 2.0)],
        ));
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        idxs.extend_from_slice(&[
            bot_base,
            bot_base + 1 + i as u16,
            bot_base + 1 + next as u16,
        ]);
    }

    (verts, idxs)
}

/// Build a flat horizontal plane from half-width and half-depth.
///
/// Lies in the XZ plane at Y = 0, facing up. UV tiles at one repeat per metre.
pub fn build_plane(half_width: f32, half_depth: f32) -> (Vec<Vert>, Vec<u16>) {
    let normal = [0.0f32, 1.0, 0.0];
    let color = [0.80f32, 0.79, 0.78];
    let w = half_width * 2.0;
    let d = half_depth * 2.0;

    let verts = alloc::vec![
        ([-half_width, 0.0, -half_depth], normal, color, [0.0, 0.0]),
        ([half_width, 0.0, -half_depth], normal, color, [w, 0.0]),
        ([half_width, 0.0, half_depth], normal, color, [w, d]),
        ([-half_width, 0.0, half_depth], normal, color, [0.0, d]),
    ];
    let idxs = alloc::vec![0u16, 1, 2, 2, 3, 0];

    (verts, idxs)
}

/// Build a UV sphere from radius, ring count, and segment count.
///
/// Centred on the origin; poles at Y = ±radius. UV mapping is spherical, and
/// the normal at every point equals `normalise(pos)`. `rings` is clamped to at
/// least 2 and `segments` to at least 3; a tessellation past the u16 index
/// limit is refused.
pub fn build_sphere(
    radius: f32,
    rings: u32,
    segments: u32,
) -> Result<(Vec<Vert>, Vec<u16>), String> {
    let rings = rings.max(2) as usize;
    let segments = segments.max(3) as usize;

    let vert_count = (rings + 1) * (segments + 1) + 2;
    if vert_count > 65536 {
        return Err(format!(
            "sphere rings={} segments={} produces {} vertices, exceeding the u16 limit",
            rings, segments, vert_count
        ));
    }

    let color = [0.82f32, 0.80, 0.78];
    let mut verts: Vec<Vert> = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    // theta depends only on the segment, so the ring loop would otherwise
    // recompute the same sin/cos pair once per ring.
    let ring_angles: Vec<(f32, f32)> = (0..=segments)
        .map(|seg| {
            let theta = core::f32::consts::TAU * seg as f32 / segments as f32;
            sin_cos(theta)
        })
        .collect();

    for ring in 0..=rings {
        let phi = core::f32::consts::PI * (ring as f32 + 1.0) / (rings as f32 + 1.0);
        let (sin_phi, cos_phi) = sin_cos(phi);
        for (seg, &(sin_theta, cos_theta)) in ring_angles.iter().enumerate() {
            let nx = sin_phi * cos_theta;
            let ny = cos_phi;
            let nz = sin_phi * sin_theta;
            let u = seg as f32 / segments as f32;
            let v = (ring as f32 + 1.0) / (rings as f32 + 1.0);
            verts.push((
                [nx * radius, ny * radius, nz * radius],
                [nx, ny, nz],
                color,
                [u, v],
            ));
        }
    }

    // north pole cap
    let pole_n = verts.len() as u16;
    verts.push(([0.0, radius, 0.0], [0.0, 1.0, 0.0], color, [0.5, 0.0]));
    for seg in 0..segments {
        idxs.extend_from_slice(&[pole_n, seg as u16, (seg + 1) as u16]);
    }

    // south pole cap
    let pole_s = verts.len() as u16;
    verts.push(([0.0, -radius, 0.0], [0.0, -1.0, 0.0], color, [0.5, 1.0]));
    let last_ring_start = ((rings - 1) * (segments + 1)) as u16;
    for seg in 0..segments {
        idxs.extend_from_slice(&[
            pole_s,
            last_ring_start + (seg + 1) as u16,
            last_ring_start + seg as u16,
        ]);
    }

    // quads between adjacent rings
    for ring in 0..rings - 1 {
        let row0 = (ring * (segments + 1)) as u16;
        let row1 = row0 + (segments + 1) as u16;
        for seg in 0..segments {
            let tl = row0 + seg as u16;
            let tr = row0 + (seg + 1) as u16;
            let bl = row1 + seg as u16;
            let br = row1 + (seg + 1) as u16;
            idxs.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    Ok((verts, idxs))
}

// No winding-direction unit tests live here. A previous iteration added an
// "outward winding" check and silently flipped every primitive's index order
// to make it pass, which produced a renderer-visible regression because
// the rasterizer pipeline empirically expects the *opposite* winding for
// the sphere / cylinder-side / plane triangles. The mesh as a whole is not
// even uniformly wound (cylinder caps and the box wind outward; everything
// else winds inward), and the project's Metal pipeline doesn't enable
// back-face culling, so any per-primitive winding test is misleading. Keep
// the index orders in this file untouched unless you have re-validated the
// full render with the showcase world.
