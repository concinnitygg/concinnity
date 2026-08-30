// Subdivided terrain grid driven by a grayscale heightmap image.
//
// Sibling of terrain.rs. Same XZ grid + smooth-normal pass; the only difference
// is the height function: instead of three octaves of LCG-hash noise, this
// generator samples pre-decoded heightmap pixels and maps the red channel
// through the configured elevation range. Image decoding is the caller's
// problem (the cook crate's, in practice) -- this crate links no image decoders.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::Vert;
use crate::math::floor;
use crate::math::vec3::{vec3_add, vec3_face_normal, vec3_normalise};

/// The field a heightmap image displaces: grid extents, resolution, and the
/// elevation range the red channel maps into.
pub struct HeightfieldField {
    /// Half the terrain extent along X, in world units.
    pub half_width: f32,
    /// Half the terrain extent along Z, in world units.
    pub half_depth: f32,
    /// Grid resolution per axis, clamped to 4..=255.
    pub subdivisions: u32,
    /// World Y a red-channel value of 0 maps to.
    pub elevation_min: f32,
    /// World Y a red-channel value of 255 maps to.
    pub elevation_max: f32,
}

/// Build a heightfield grid from decoded RGBA pixels: bilinear-sample the
/// image's red channel across the field's grid and map it through the
/// elevation range.
pub fn build_heightfield_from_pixels(
    field: &HeightfieldField,
    img_w: u32,
    img_h: u32,
    rgba: &[u8],
) -> Result<(Vec<Vert>, Vec<u16>), String> {
    let HeightfieldField {
        half_width,
        half_depth,
        subdivisions,
        elevation_min,
        elevation_max,
    } = *field;
    let subdivisions = subdivisions.clamp(4, 255) as usize;

    if img_w == 0 || img_h == 0 {
        return Err("heightfield source image has zero extent".into());
    }

    let needed = (img_w as usize) * (img_h as usize) * 4;
    if rgba.len() < needed {
        return Err(format!(
            "heightfield source image buffer too small: have {}, need {} for {}x{}",
            rgba.len(),
            needed,
            img_w,
            img_h
        ));
    }

    let cols = subdivisions + 1;
    let rows = subdivisions + 1;

    if cols * rows > 65536 {
        return Err(format!(
            "heightfield subdivisions {} produces {} vertices, exceeding the u16 limit; use subdivisions ≤ 255",
            subdivisions,
            cols * rows
        ));
    }

    let color = [0.55f32, 0.62, 0.42];

    // Pre-sample the heightmap to per-vertex Y. Bilinear filter so the mesh
    // doesn't inherit the heightmap's pixel grid when subdivisions and image
    // resolution differ.
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let s = col as f32 / subdivisions as f32;
            let t = row as f32 / subdivisions as f32;
            let x = -half_width + s * half_width * 2.0;
            let z = -half_depth + t * half_depth * 2.0;
            let y = sample_height_bilinear(rgba, img_w, img_h, s, t, elevation_min, elevation_max);
            positions.push([x, y, z]);
        }
    }

    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; cols * rows];
    for row in 0..subdivisions {
        for col in 0..subdivisions {
            let tl = row * cols + col;
            let tr = tl + 1;
            let bl = tl + cols;
            let br = bl + 1;
            let n1 = vec3_face_normal(positions[tl], positions[bl], positions[tr]);
            vec3_add(&mut normals[tl], n1);
            vec3_add(&mut normals[bl], n1);
            vec3_add(&mut normals[tr], n1);
            let n2 = vec3_face_normal(positions[tr], positions[bl], positions[br]);
            vec3_add(&mut normals[tr], n2);
            vec3_add(&mut normals[bl], n2);
            vec3_add(&mut normals[br], n2);
        }
    }

    let mut idxs: Vec<u16> = Vec::with_capacity(subdivisions * subdivisions * 6);
    let mut verts: Vec<Vert> = Vec::with_capacity(cols * rows);

    for i in 0..cols * rows {
        let [x, y, z] = positions[i];
        let normal = vec3_normalise(normals[i]);
        verts.push(([x, y, z], normal, color, [x, z]));
    }

    for row in 0..subdivisions {
        for col in 0..subdivisions {
            let tl = (row * cols + col) as u16;
            let tr = tl + 1;
            let bl = tl + cols as u16;
            let br = bl + 1;
            idxs.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    Ok((verts, idxs))
}

// Bilinear-sample the heightmap's red channel at normalised UV (s, t) in [0,1]
// and map [0, 255] to [elevation_min, elevation_max].
fn sample_height_bilinear(
    rgba: &[u8],
    img_w: u32,
    img_h: u32,
    s: f32,
    t: f32,
    elevation_min: f32,
    elevation_max: f32,
) -> f32 {
    let fx = s.clamp(0.0, 1.0) * (img_w - 1) as f32;
    let fy = t.clamp(0.0, 1.0) * (img_h - 1) as f32;
    let x0 = floor(fx) as u32;
    let y0 = floor(fy) as u32;
    let x1 = (x0 + 1).min(img_w - 1);
    let y1 = (y0 + 1).min(img_h - 1);
    let sx = fx - x0 as f32;
    let sy = fy - y0 as f32;

    let r = |x: u32, y: u32| -> f32 {
        let idx = (y * img_w + x) as usize * 4;
        rgba[idx] as f32 / 255.0
    };
    let top = r(x0, y0) + (r(x1, y0) - r(x0, y0)) * sx;
    let bot = r(x0, y1) + (r(x1, y1) - r(x0, y1)) * sx;
    let h = top + (bot - top) * sy;
    elevation_min + h * (elevation_max - elevation_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `w`x`h` grayscale-RGBA buffer whose red channel ramps 0..255 across X
    // so the generated mesh has real elevation variation to sample.
    fn ramp_rgba(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..h {
            for x in 0..w {
                let v = if w > 1 { (x * 255 / (w - 1)) as u8 } else { 0 };
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        out
    }

    #[test]
    fn bilinear_sample_recovers_corner_values() {
        let rgba = ramp_rgba(4, 4);
        let h_min = sample_height_bilinear(&rgba, 4, 4, 0.0, 0.0, -1.0, 1.0);
        let h_max = sample_height_bilinear(&rgba, 4, 4, 1.0, 0.0, -1.0, 1.0);
        assert!((h_min - -1.0).abs() < 1e-5, "h_min = {}", h_min);
        assert!((h_max - 1.0).abs() < 1e-5, "h_max = {}", h_max);
    }

    fn field(half: f32, subdivisions: u32, elevation_max: f32) -> HeightfieldField {
        HeightfieldField {
            half_width: half,
            half_depth: half,
            subdivisions,
            elevation_min: 0.0,
            elevation_max,
        }
    }

    #[test]
    fn rejects_zero_extent_and_short_pixel_buffers() {
        let err = build_heightfield_from_pixels(&field(64.0, 3, 1.0), 0, 0, &[]).unwrap_err();
        assert!(err.contains("zero extent"), "got: {}", err);
        // 8x8 RGBA needs 256 bytes; hand it 100.
        let err =
            build_heightfield_from_pixels(&field(64.0, 4, 1.0), 8, 8, &[0u8; 100]).unwrap_err();
        assert!(err.contains("have 100, need 256"), "got: {err}");
    }

    #[test]
    fn vertex_and_index_counts_match_grid() {
        // subdivisions=4 -> 5x5 = 25 verts, 4*4*2 = 32 tris -> 96 indices.
        let rgba = ramp_rgba(8, 8);
        let (verts, idxs) =
            build_heightfield_from_pixels(&field(5.0, 4, 10.0), 8, 8, &rgba).expect("builds");
        assert_eq!(verts.len(), 5 * 5);
        assert_eq!(idxs.len(), 4 * 4 * 6);

        // The ramp gives real elevation variation bracketed by the range.
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for v in &verts {
            min_y = min_y.min(v.0[1]);
            max_y = max_y.max(v.0[1]);
        }
        assert!(min_y >= 0.0);
        assert!(max_y <= 10.0);
        assert!(
            max_y > min_y,
            "expected variation but got flat at {}",
            max_y
        );
    }
}
