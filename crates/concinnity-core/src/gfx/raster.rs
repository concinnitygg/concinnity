// src/gfx/raster.rs
//
// Pure CPU rasterizer for small offline previews (asset thumbnails): an
// orthographic three-quarter view of a triangle mesh with a z-buffer and
// simple key + ambient shading. No GPU, no backend, no ECS; deterministic
// across platforms, so a baked image is identical everywhere.

use super::mesh_payload::Vertex;

// The fixed camera direction (toward the subject) and key light, chosen so a
// box shows three faces at distinct brightnesses.
const VIEW_DIR: [f32; 3] = [-0.577, -0.577, -0.577];
const LIGHT_DIR: [f32; 3] = [0.408, 0.816, 0.408];
const AMBIENT: f32 = 0.30;
const DIFFUSE: f32 = 0.70;
// Fraction of the image the fitted subject spans (the rest is margin).
const FIT: f32 = 0.9;

/// An RGBA8 image buffer with a transparent background, the rasterizer's
/// render target.
pub struct RasterImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 0.0 || !len.is_finite() {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// The orthographic camera basis for the fixed view: right / up in the image
// plane, forward toward the subject.
fn camera_basis() -> ([f32; 3], [f32; 3], [f32; 3]) {
    let fwd = normalize(VIEW_DIR);
    let right = normalize(cross([0.0, 1.0, 0.0], fwd));
    let up = cross(fwd, right);
    (right, up, fwd)
}

/// One shaded piece of a multi-part render: a triangle mesh and the color it
/// shades with. Parts share one camera framing and one z-buffer, so they
/// occlude each other like a composed model.
pub struct MeshPart<'a> {
    pub verts: &'a [Vertex],
    pub indices: &'a [u16],
    pub color: [f32; 3],
}

/// Shade `verts`/`indices` into a `size` x `size` RGBA8 image: orthographic
/// three-quarter view auto-framed to the mesh bounds, z-buffered, smooth
/// N·L + ambient shading of `base_color`, transparent background. An empty or
/// degenerate mesh returns a fully transparent image.
pub fn shade_mesh(
    verts: &[Vertex],
    indices: &[u16],
    size: u32,
    base_color: [f32; 3],
) -> RasterImage {
    shade_parts(
        &[MeshPart {
            verts,
            indices,
            color: base_color,
        }],
        size,
    )
}

/// [shade_mesh](#method.shade_mesh) over several parts at once, framed to
/// their combined bounds (a Model's sub-meshes, each with its material color).
pub fn shade_parts(parts: &[MeshPart], size: u32) -> RasterImage {
    let mut img = RasterImage {
        width: size,
        height: size,
        rgba: vec![0u8; (size * size * 4) as usize],
    };
    let mut positions = parts.iter().flat_map(|p| p.verts.iter().map(|v| v.pos));
    let Some(first) = positions.next() else {
        return img;
    };
    if size == 0 || parts.iter().all(|p| p.indices.len() < 3) {
        return img;
    }
    let (right, up, fwd) = camera_basis();

    // Frame the combined bounds, then project each part into the shared
    // camera basis about that center.
    let mut min = first;
    let mut max = first;
    for pos in positions {
        for a in 0..3 {
            min[a] = min[a].min(pos[a]);
            max[a] = max[a].max(pos[a]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let project = |v: &Vertex| {
        let p = [
            v.pos[0] - center[0],
            v.pos[1] - center[1],
            v.pos[2] - center[2],
        ];
        [dot(p, right), dot(p, up), dot(p, fwd)]
    };
    let extent = parts
        .iter()
        .flat_map(|p| p.verts.iter())
        .map(|v| {
            let p = project(v);
            p[0].abs().max(p[1].abs())
        })
        .fold(0.0f32, f32::max);
    if extent <= 0.0 || !extent.is_finite() {
        return img;
    }
    let half = size as f32 * 0.5;
    let scale = half * FIT / extent;
    let light = normalize(LIGHT_DIR);
    let mut depth = vec![f32::NEG_INFINITY; (size * size) as usize];
    for part in parts {
        // Image coordinates: x right, y down.
        let screen: Vec<[f32; 3]> = part
            .verts
            .iter()
            .map(|v| {
                let p = project(v);
                [half + p[0] * scale, half - p[1] * scale, p[2]]
            })
            .collect();
        for tri in part.indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= screen.len() || i1 >= screen.len() || i2 >= screen.len() {
                continue;
            }
            fill_triangle(
                &mut img,
                &mut depth,
                [screen[i0], screen[i1], screen[i2]],
                [
                    part.verts[i0].normal,
                    part.verts[i1].normal,
                    part.verts[i2].normal,
                ],
                light,
                part.color,
            );
        }
    }
    img
}

// Rasterize one triangle with barycentric depth + normal interpolation.
fn fill_triangle(
    img: &mut RasterImage,
    depth: &mut [f32],
    p: [[f32; 3]; 3],
    n: [[f32; 3]; 3],
    light: [f32; 3],
    base_color: [f32; 3],
) {
    let area =
        (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
    if area.abs() <= f32::EPSILON || !area.is_finite() {
        return;
    }
    let min_x = p.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min).floor();
    let max_x = p
        .iter()
        .map(|v| v[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil();
    let min_y = p.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min).floor();
    let max_y = p
        .iter()
        .map(|v| v[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil();
    let x0 = (min_x.max(0.0)) as u32;
    let x1 = (max_x.min(img.width as f32 - 1.0)).max(0.0) as u32;
    let y0 = (min_y.max(0.0)) as u32;
    let y1 = (max_y.min(img.height as f32 - 1.0)).max(0.0) as u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            // Barycentric weights via edge functions; either winding accepted.
            let w0 = ((p[1][0] - px) * (p[2][1] - py) - (p[1][1] - py) * (p[2][0] - px)) / area;
            let w1 = ((p[2][0] - px) * (p[0][1] - py) - (p[2][1] - py) * (p[0][0] - px)) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let z = w0 * p[0][2] + w1 * p[1][2] + w2 * p[2][2];
            let idx = (y * img.width + x) as usize;
            // Camera forward points at the subject, so nearer surfaces have
            // smaller forward depth: keep the largest -z.
            if -z <= depth[idx] {
                continue;
            }
            depth[idx] = -z;
            let normal = normalize([
                w0 * n[0][0] + w1 * n[1][0] + w2 * n[2][0],
                w0 * n[0][1] + w1 * n[1][1] + w2 * n[2][1],
                w0 * n[0][2] + w1 * n[1][2] + w2 * n[2][2],
            ]);
            // Two-sided: a flipped normal shades like its front face.
            let diff = dot(normal, light).abs().clamp(0.0, 1.0);
            let shade = AMBIENT + DIFFUSE * diff;
            let o = idx * 4;
            for (c, &b) in base_color.iter().enumerate() {
                img.rgba[o + c] = ((b * shade).clamp(0.0, 1.0) * 255.0) as u8;
            }
            img.rgba[o + 3] = 255;
        }
    }
}

/// Shade a lit sphere swatch into a `size` x `size` RGBA8 image: `albedo`
/// diffuse with a specular highlight whose width follows `roughness` and whose
/// tint follows `metallic`. Transparent outside the sphere.
pub fn shade_sphere(size: u32, albedo: [f32; 3], roughness: f32, metallic: f32) -> RasterImage {
    let mut img = RasterImage {
        width: size,
        height: size,
        rgba: vec![0u8; (size * size * 4) as usize],
    };
    if size == 0 {
        return img;
    }
    let light = normalize([0.45, 0.65, 0.6]);
    let view = [0.0, 0.0, 1.0];
    let h = normalize([light[0] + view[0], light[1] + view[1], light[2] + view[2]]);
    let rough = roughness.clamp(0.05, 1.0);
    let metal = metallic.clamp(0.0, 1.0);
    let shininess = 2.0 / (rough * rough) - 1.0;
    let radius = size as f32 * 0.5 * FIT;
    let half = size as f32 * 0.5;
    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 + 0.5 - half) / radius;
            let dy = (half - (y as f32 + 0.5)) / radius;
            let d2 = dx * dx + dy * dy;
            if d2 > 1.0 {
                continue;
            }
            let normal = [dx, dy, (1.0 - d2).sqrt()];
            let diff = dot(normal, light).max(0.0);
            let spec = dot(normal, h).max(0.0).powf(shininess) * (1.0 - rough * 0.6);
            // Metals tint the highlight with the albedo and lose diffuse.
            let spec_color = [
                1.0 - metal + metal * albedo[0],
                1.0 - metal + metal * albedo[1],
                1.0 - metal + metal * albedo[2],
            ];
            let o = ((y * size + x) * 4) as usize;
            for c in 0..3 {
                let v = albedo[c] * (AMBIENT + DIFFUSE * diff) * (1.0 - metal * 0.7)
                    + spec_color[c] * spec;
                img.rgba[o + c] = (v.clamp(0.0, 1.0) * 255.0) as u8;
            }
            img.rgba[o + 3] = 255;
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(normal: [f32; 3]) -> (Vec<Vertex>, Vec<u16>) {
        let v = |pos: [f32; 3]| Vertex {
            pos,
            normal,
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0; 2],
        };
        (
            vec![
                v([-1.0, -1.0, 0.0]),
                v([1.0, -1.0, 0.0]),
                v([1.0, 1.0, 0.0]),
                v([-1.0, 1.0, 0.0]),
            ],
            vec![0, 1, 2, 0, 2, 3],
        )
    }

    fn coverage(img: &RasterImage) -> usize {
        img.rgba.chunks_exact(4).filter(|p| p[3] > 0).count()
    }

    #[test]
    fn a_mesh_fills_pixels_and_the_background_stays_transparent() {
        let (verts, indices) = quad([0.0, 0.0, 1.0]);
        let img = shade_mesh(&verts, &indices, 64, [0.8, 0.8, 0.8]);
        let covered = coverage(&img);
        assert!(covered > 500, "the quad covers a real area: {covered}");
        assert!(
            covered < (64 * 64),
            "the margin stays transparent: {covered}"
        );
        // Corner pixel is background.
        assert_eq!(img.rgba[3], 0);
    }

    #[test]
    fn empty_or_degenerate_input_renders_transparent() {
        assert_eq!(coverage(&shade_mesh(&[], &[], 32, [1.0; 3])), 0);
        let (verts, _) = quad([0.0, 0.0, 1.0]);
        assert_eq!(coverage(&shade_mesh(&verts, &[0, 1], 32, [1.0; 3])), 0);
        // All vertices coincident: zero extent.
        let point = vec![verts[0], verts[0], verts[0]];
        assert_eq!(coverage(&shade_mesh(&point, &[0, 1, 2], 32, [1.0; 3])), 0);
    }

    #[test]
    fn nearer_geometry_wins_the_depth_test() {
        // Two stacked quads along the view direction; the nearer (toward the
        // camera) is red, the farther green. The image center must be red.
        let (mut near, mut idx) = quad([0.0, 0.0, 1.0]);
        let (far, far_idx) = quad([0.0, 0.0, 1.0]);
        // Push the "near" quad toward the camera (opposite VIEW_DIR).
        for v in &mut near {
            for (p, d) in v.pos.iter_mut().zip(VIEW_DIR) {
                *p -= d * 2.0;
            }
        }
        let base = near.len() as u16;
        near.extend(far);
        idx.extend(far_idx.iter().map(|i| i + base));
        // Shade in one pass with a single color, then re-shade each quad alone
        // to know which shade the near quad produces at the center.
        let both = shade_mesh(&near, &idx, 64, [1.0, 1.0, 1.0]);
        let alone = shade_mesh(&near[..4], &[0, 1, 2, 0, 2, 3], 64, [1.0, 1.0, 1.0]);
        let center = ((32 * 64 + 32) * 4) as usize;
        assert_eq!(
            both.rgba[center..center + 3],
            alone.rgba[center..center + 3],
            "the nearer quad's shade wins at the center"
        );
    }

    #[test]
    fn deterministic_output() {
        let (verts, indices) = quad([0.0, 0.0, 1.0]);
        let a = shade_mesh(&verts, &indices, 48, [0.5, 0.6, 0.7]);
        let b = shade_mesh(&verts, &indices, 48, [0.5, 0.6, 0.7]);
        assert_eq!(a.rgba, b.rgba);
    }

    #[test]
    fn parts_share_one_frame_and_depth_buffer() {
        // A big far green quad behind a small near red quad, fully overlapped
        // in the image. Both colors must show (the shared framing covers
        // both), and swapping the part order must not change a pixel: the
        // shared z-buffer decides the overlap, not paint order.
        let (mut far, far_idx) = quad([0.0, 0.0, 1.0]);
        let (mut near, near_idx) = quad([0.0, 0.0, 1.0]);
        for v in &mut far {
            v.pos[0] *= 4.0;
            v.pos[1] *= 4.0;
        }
        for v in &mut near {
            v.pos[0] *= 0.5;
            v.pos[1] *= 0.5;
            for (p, d) in v.pos.iter_mut().zip(VIEW_DIR) {
                *p -= d * 2.0;
            }
        }
        let parts = |a: bool| {
            let far_part = MeshPart {
                verts: &far,
                indices: &far_idx,
                color: [0.0, 1.0, 0.0],
            };
            let near_part = MeshPart {
                verts: &near,
                indices: &near_idx,
                color: [1.0, 0.0, 0.0],
            };
            if a {
                [far_part, near_part]
            } else {
                [near_part, far_part]
            }
        };
        let img = shade_parts(&parts(true), 64);
        let count = |channel: usize| {
            img.rgba
                .chunks_exact(4)
                .filter(|p| p[3] > 0 && p[channel] > p[(channel + 1) % 3].max(p[(channel + 2) % 3]))
                .count()
        };
        assert!(count(0) > 0, "the near red part shows");
        assert!(count(1) > 0, "the far green part shows");
        // The near quad projects inside the far quad's footprint, so if paint
        // order (not depth) decided, swapping the parts would repaint the
        // overlap green.
        let swapped = shade_parts(&parts(false), 64);
        assert_eq!(img.rgba, swapped.rgba, "depth decides, not paint order");
    }

    #[test]
    fn sphere_swatch_is_round_lit_and_material_sensitive() {
        let rough = shade_sphere(64, [0.8, 0.2, 0.2], 1.0, 0.0);
        let covered = coverage(&rough);
        let full = 64 * 64;
        assert!(covered > full / 2 && covered < full, "a disc: {covered}");
        assert_eq!(rough.rgba[3], 0, "corners transparent");
        let shiny = shade_sphere(64, [0.8, 0.2, 0.2], 0.1, 0.0);
        assert_ne!(rough.rgba, shiny.rgba, "roughness changes the highlight");
        let metal = shade_sphere(64, [0.8, 0.2, 0.2], 0.1, 1.0);
        assert_ne!(shiny.rgba, metal.rgba, "metallic changes the shading");
    }
}
