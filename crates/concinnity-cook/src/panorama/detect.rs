// Predicate recognising a panorama sphere in a parsed glTF document.
//
// Every criterion below has to hold. They are checked in declaration order and
// the first miss is returned as the reason, so `cn add` can explain why a file
// the author expected to become a sky imported as geometry instead.

use crate::gltf_source::GltfDoc;

// A vertex may sit this far off the mean radius, as a fraction of it, and the
// mesh still counts as a sphere. An exported UV sphere is exact; the slack
// covers float noise and lightly deformed spheres.
const RADIUS_TOLERANCE: f32 = 0.05;

// How close the UV bounds have to come to the unit square's edges. A panorama
// wraps the whole image around the sphere exactly once, so its UVs span very
// nearly [0, 1] on both axes and never tile past them.
const UV_EDGE_TOLERANCE: f32 = 0.02;

// Below this a mesh is too coarse to be carrying a panorama; it excludes the
// boxes, quads, and low-poly props that would otherwise pass the radius test
// by accident.
const MIN_VERTICES: usize = 64;

// A base colour at or under this counts as black. The panorama has to be
// unlit: an emissive image over a black base is what makes it immune to scene
// lighting, and it is how every one of these files is packaged.
const BLACK_EPSILON: f32 = 1.0 / 255.0;

// Bounds on the source image's aspect ratio. An equirectangular image covers
// 360 degrees horizontally against 180 vertically, so it is always wider than
// tall -- 2:1 in the ideal case, and often cropped to a display ratio like
// 16:9. Square and portrait images are ordinary textures.
const MIN_IMAGE_ASPECT: f32 = 1.5;
const MAX_IMAGE_ASPECT: f32 = 4.0;

/// A `.glb` recognised as a panorama sphere: an environment image wrapped on a
/// sphere rather than scene geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanoramaSphere {
    /// Index of the glTF image carrying the equirectangular panorama.
    pub image_index: u32,
}

/// Recognise a panorama sphere, or report the first criterion the document
/// missed. Callers that import geometry treat any error as "ordinary scene
/// file" rather than a failure.
pub fn detect(doc: &GltfDoc) -> Result<PanoramaSphere, String> {
    let document = &doc.doc.document;

    let mesh_count = document.meshes().count();
    if mesh_count != 1 {
        return Err(format!(
            "a panorama sphere has exactly one mesh, this has {}",
            mesh_count
        ));
    }
    let mesh = document.meshes().next().expect("one mesh");
    let primitive_count = mesh.primitives().count();
    if primitive_count != 1 {
        return Err(format!(
            "a panorama sphere has exactly one primitive, this has {}",
            primitive_count
        ));
    }
    let primitive = mesh.primitives().next().expect("one primitive");
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(format!(
            "a panorama sphere is a triangle mesh, this uses {:?}",
            primitive.mode()
        ));
    }
    if primitive.morph_targets().count() != 0 || document.skins().count() != 0 {
        return Err("a panorama sphere is not skinned or morphed".to_string());
    }

    let image_count = document.images().count();
    if image_count != 1 {
        return Err(format!(
            "a panorama sphere carries exactly one image, this has {}",
            image_count
        ));
    }
    let material_count = document.materials().count();
    if material_count != 1 {
        return Err(format!(
            "a panorama sphere has exactly one material, this has {}",
            material_count
        ));
    }
    let material = document.materials().next().expect("one material");
    check_emissive_only(&material)?;

    let reader = primitive.reader(|b| doc.buffer_bytes(b));
    let positions: Vec<[f32; 3]> = match reader.read_positions() {
        Some(p) => p.collect(),
        None => return Err("mesh has no POSITION data".to_string()),
    };
    check_spherical(&positions)?;

    let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(t) => t.into_f32().collect(),
        None => return Err("mesh has no TEXCOORD_0 data".to_string()),
    };
    check_equirect_uvs(&uvs, positions.len())?;

    let (width, height) = super::equirect::source_dimensions(doc, 0)?;
    check_panorama_aspect(width, height)?;

    Ok(PanoramaSphere { image_index: 0 })
}

// The image has to arrive entirely on the emissive channel over a black base
// colour, with nothing else bound. That combination renders the panorama at
// full brightness regardless of scene lighting, which is the whole point of
// the packaging, and it is what separates these files from a textured ball.
fn check_emissive_only(material: &gltf::Material<'_>) -> Result<(), String> {
    let pbr = material.pbr_metallic_roughness();

    let Some(emissive) = material.emissive_texture() else {
        return Err("material has no emissive texture".to_string());
    };
    if emissive.texture().source().index() != 0 {
        return Err("material's emissive texture is not the document's image".to_string());
    }
    if material.emissive_factor().iter().all(|c| *c <= 0.0) {
        return Err("material's emissive factor is zero, so the panorama would not show".into());
    }
    if pbr.base_color_texture().is_some() {
        return Err("material has a base colour texture, so it is lit geometry".to_string());
    }
    let base = pbr.base_color_factor();
    if base[..3].iter().any(|c| *c > BLACK_EPSILON) {
        return Err(format!(
            "material's base colour {:?} is not black, so it is lit geometry",
            &base[..3]
        ));
    }
    if pbr.metallic_roughness_texture().is_some()
        || material.normal_texture().is_some()
        || material.occlusion_texture().is_some()
    {
        return Err("material binds surface maps, so it is lit geometry".to_string());
    }
    Ok(())
}

// Every vertex sits the same distance from the mesh's centre. Comparing against
// the mean radius rather than a fixed one keeps the test independent of the
// sphere's authored scale.
fn check_spherical(positions: &[[f32; 3]]) -> Result<(), String> {
    if positions.len() < MIN_VERTICES {
        return Err(format!(
            "mesh has {} vertices, too coarse for a panorama sphere (needs {})",
            positions.len(),
            MIN_VERTICES
        ));
    }
    let n = positions.len() as f32;
    let mut centre = [0.0f32; 3];
    for p in positions {
        for axis in 0..3 {
            centre[axis] += p[axis] / n;
        }
    }
    let radii: Vec<f32> = positions
        .iter()
        .map(|p| {
            let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .collect();
    let mean = radii.iter().sum::<f32>() / n;
    if mean <= 0.0 || !mean.is_finite() {
        return Err("mesh has no extent".to_string());
    }
    if let Some(off) = radii
        .iter()
        .find(|r| ((*r - mean) / mean).abs() > RADIUS_TOLERANCE)
    {
        return Err(format!(
            "mesh is not a sphere: a vertex sits at radius {:.4} against a mean of {:.4}",
            off, mean
        ));
    }
    Ok(())
}

// The panorama covers the sphere exactly once: UVs reach both edges of the
// unit square and never run past them.
fn check_equirect_uvs(uvs: &[[f32; 2]], vertex_count: usize) -> Result<(), String> {
    if uvs.len() != vertex_count {
        return Err(format!(
            "mesh has {} UVs against {} vertices",
            uvs.len(),
            vertex_count
        ));
    }
    for axis in 0..2 {
        let min = uvs.iter().map(|uv| uv[axis]).fold(f32::MAX, f32::min);
        let max = uvs.iter().map(|uv| uv[axis]).fold(f32::MIN, f32::max);
        let name = if axis == 0 { "U" } else { "V" };
        if min > UV_EDGE_TOLERANCE || max < 1.0 - UV_EDGE_TOLERANCE {
            return Err(format!(
                "{} spans {:.3}..{:.3}, not the full image a panorama wraps",
                name, min, max
            ));
        }
        if min < -UV_EDGE_TOLERANCE || max > 1.0 + UV_EDGE_TOLERANCE {
            return Err(format!(
                "{} spans {:.3}..{:.3}, so the image tiles rather than wrapping once",
                name, min, max
            ));
        }
    }
    Ok(())
}

fn check_panorama_aspect(width: u32, height: u32) -> Result<(), String> {
    if height == 0 {
        return Err("image has zero height".to_string());
    }
    let aspect = width as f32 / height as f32;
    if !(MIN_IMAGE_ASPECT..=MAX_IMAGE_ASPECT).contains(&aspect) {
        return Err(format!(
            "image is {}x{} ({:.2}:1), not the landscape shape of an equirectangular panorama",
            width, height, aspect
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use crate::glb::test_fixtures::{f32s, make_glb, u16s};

    // A UV sphere with `segments` columns and `rings` rows of quads, unit
    // radius, UVs spanning the full unit square: the shape an exporter writes
    // for a panorama sphere.
    pub(crate) fn uv_sphere(segments: usize, rings: usize) -> (Vec<[f32; 3]>, Vec<[f32; 2]>) {
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let theta = v * std::f32::consts::PI;
            for seg in 0..=segments {
                let u = seg as f32 / segments as f32;
                let phi = u * std::f32::consts::TAU;
                positions.push([
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                ]);
                uvs.push([u, v]);
            }
        }
        (positions, uvs)
    }

    // A 4x2 RGB PNG, the smallest thing that reads as a landscape panorama.
    // `value` fills every channel so a decode is checkable against one number.
    pub(crate) fn panorama_png(width: u32, height: u32, value: u8) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("png header");
            let pixels = vec![value; (width * height * 3) as usize];
            writer.write_image_data(&pixels).expect("png data");
        }
        out
    }

    // A 16-bit RGBA PNG, the packaging the galaxy panorama actually ships in.
    pub(crate) fn panorama_png16(width: u32, height: u32, value: u16) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Sixteen);
            let mut writer = encoder.write_header().expect("png header");
            let mut pixels = Vec::new();
            for i in 0..(width * height * 4) {
                let channel = if i % 4 == 3 { u16::MAX } else { value };
                pixels.extend_from_slice(&channel.to_be_bytes());
            }
            writer.write_image_data(&pixels).expect("png data");
        }
        out
    }

    // Knobs the negative tests flip one at a time, so each rejection reason is
    // reachable from an otherwise-valid document.
    pub(crate) struct PanoramaShape {
        pub segments: usize,
        pub rings: usize,
        pub(crate) base_color: [f32; 4],
        pub(crate) emissive_factor: [f32; 3],
        pub(crate) uv_scale: f32,
        pub png: Vec<u8>,
    }

    impl Default for PanoramaShape {
        fn default() -> Self {
            Self {
                segments: 12,
                rings: 8,
                base_color: [0.0, 0.0, 0.0, 1.0],
                emissive_factor: [1.0, 1.0, 1.0],
                uv_scale: 1.0,
                png: panorama_png(4, 2, 128),
            }
        }
    }

    // Assemble a panorama-sphere `.glb` from `shape`. Buffer layout, in order:
    // positions, UVs, indices, then the PNG.
    pub(crate) fn panorama_glb_with(shape: PanoramaShape) -> Vec<u8> {
        let (positions, uvs) = uv_sphere(shape.segments, shape.rings);
        let scaled: Vec<[f32; 2]> = uvs
            .iter()
            .map(|uv| [uv[0] * shape.uv_scale, uv[1] * shape.uv_scale])
            .collect();
        let indices: Vec<u16> = (0..positions.len() as u16).collect();

        let pos_bytes = f32s(&positions.iter().flatten().copied().collect::<Vec<f32>>());
        let uv_bytes = f32s(&scaled.iter().flatten().copied().collect::<Vec<f32>>());
        let idx_bytes = u16s(&indices);

        let mut bin = Vec::new();
        let pos_off = 0;
        bin.extend_from_slice(&pos_bytes);
        let uv_off = bin.len();
        bin.extend_from_slice(&uv_bytes);
        let idx_off = bin.len();
        bin.extend_from_slice(&idx_bytes);
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let png_off = bin.len();
        bin.extend_from_slice(&shape.png);

        // The glTF validator requires bounds on a POSITION accessor.
        let mut pos_min = [f32::MAX; 3];
        let mut pos_max = [f32::MIN; 3];
        for p in &positions {
            for axis in 0..3 {
                pos_min[axis] = pos_min[axis].min(p[axis]);
                pos_max[axis] = pos_max[axis].max(p[axis]);
            }
        }

        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": bin.len()}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": pos_off, "byteLength": pos_bytes.len()},
                {"buffer": 0, "byteOffset": uv_off, "byteLength": uv_bytes.len()},
                {"buffer": 0, "byteOffset": idx_off, "byteLength": idx_bytes.len()},
                {"buffer": 0, "byteOffset": png_off, "byteLength": shape.png.len()}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": positions.len(), "type": "VEC3",
                 "min": pos_min, "max": pos_max},
                {"bufferView": 1, "componentType": 5126, "count": positions.len(), "type": "VEC2"},
                {"bufferView": 2, "componentType": 5123, "count": indices.len(), "type": "SCALAR"}
            ],
            "images": [{"bufferView": 3, "mimeType": "image/png"}],
            "textures": [{"source": 0}],
            "materials": [{
                "doubleSided": true,
                "emissiveFactor": shape.emissive_factor,
                "emissiveTexture": {"index": 0},
                "pbrMetallicRoughness": {"baseColorFactor": shape.base_color}
            }],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                "indices": 2,
                "material": 0
            }]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0
        });
        make_glb(&json, Some(&bin))
    }

    pub(crate) fn panorama_glb() -> Vec<u8> {
        panorama_glb_with(PanoramaShape::default())
    }

    // An ordinary two-mesh scene: what must keep importing as geometry.
    pub(crate) fn ordinary_scene_glb() -> Vec<u8> {
        let mut json = crate::glb::test_fixtures::static_triangle_json();
        json["meshes"] = serde_json::json!([
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]},
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}
        ]);
        json["nodes"] = serde_json::json!([{"mesh": 0}, {"mesh": 1}]);
        json["scenes"] = serde_json::json!([{"nodes": [0, 1]}]);
        make_glb(
            &json,
            Some(&crate::glb::test_fixtures::static_triangle_bin()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::*;
    use super::*;

    fn doc(bytes: &[u8]) -> GltfDoc {
        GltfDoc::from_slice(bytes, None, "test.glb").expect("parse")
    }

    fn reject(shape: PanoramaShape) -> String {
        detect(&doc(&panorama_glb_with(shape))).unwrap_err()
    }

    #[test]
    fn a_panorama_sphere_is_recognised() {
        let found = detect(&doc(&panorama_glb())).expect("panorama");
        assert_eq!(found, PanoramaSphere { image_index: 0 });
    }

    #[test]
    fn a_sixteen_bit_panorama_is_recognised() {
        // The packaging the galaxy file ships in: 16-bit RGBA rather than 8-bit.
        let shape = PanoramaShape {
            png: panorama_png16(4, 2, 30000),
            ..Default::default()
        };
        detect(&doc(&panorama_glb_with(shape))).expect("panorama");
    }

    #[test]
    fn an_ordinary_scene_is_rejected_for_its_mesh_count() {
        let err = detect(&doc(&ordinary_scene_glb())).unwrap_err();
        assert!(err.contains("exactly one mesh"), "got: {err}");
    }

    #[test]
    fn a_single_mesh_scene_without_a_material_is_rejected() {
        let glb = crate::glb::test_fixtures::static_triangle_glb();
        let err = detect(&doc(&glb)).unwrap_err();
        assert!(err.contains("exactly one image"), "got: {err}");
    }

    #[test]
    fn a_lit_sphere_is_rejected_for_its_base_colour() {
        let err = reject(PanoramaShape {
            base_color: [0.8, 0.8, 0.8, 1.0],
            ..Default::default()
        });
        assert!(err.contains("not black"), "got: {err}");
    }

    #[test]
    fn a_sphere_with_no_emissive_output_is_rejected() {
        let err = reject(PanoramaShape {
            emissive_factor: [0.0, 0.0, 0.0],
            ..Default::default()
        });
        assert!(err.contains("emissive factor is zero"), "got: {err}");
    }

    #[test]
    fn a_coarse_mesh_is_rejected() {
        let err = reject(PanoramaShape {
            segments: 4,
            rings: 3,
            ..Default::default()
        });
        assert!(err.contains("too coarse"), "got: {err}");
    }

    #[test]
    fn tiled_uvs_are_rejected() {
        let err = reject(PanoramaShape {
            uv_scale: 4.0,
            ..Default::default()
        });
        assert!(err.contains("tiles rather than wrapping"), "got: {err}");
    }

    #[test]
    fn a_square_texture_is_rejected() {
        let err = reject(PanoramaShape {
            png: panorama_png(4, 4, 128),
            ..Default::default()
        });
        assert!(err.contains("not the landscape shape"), "got: {err}");
    }

    // The geometry / UV / aspect predicates on their own, so each boundary is
    // pinned without routing a whole document through it.

    #[test]
    fn check_spherical_accepts_a_sphere_anywhere_in_space() {
        let (positions, _) = uv_sphere(12, 8);
        let shifted: Vec<[f32; 3]> = positions
            .iter()
            .map(|p| [p[0] * 50.0 + 7.0, p[1] * 50.0 - 3.0, p[2] * 50.0])
            .collect();
        check_spherical(&shifted).expect("scale and offset must not matter");
    }

    #[test]
    fn check_spherical_rejects_a_dented_sphere() {
        let (mut positions, _) = uv_sphere(12, 8);
        positions[10] = [0.5, 0.0, 0.0];
        let err = check_spherical(&positions).unwrap_err();
        assert!(err.contains("not a sphere"), "got: {err}");
    }

    #[test]
    fn check_spherical_rejects_a_degenerate_mesh() {
        let err = check_spherical(&[[1.0, 2.0, 3.0]; MIN_VERTICES]).unwrap_err();
        assert_eq!(err, "mesh has no extent");
    }

    #[test]
    fn check_equirect_uvs_rejects_a_partial_span() {
        let uvs = vec![[0.25, 0.25], [0.75, 0.75]];
        let err = check_equirect_uvs(&uvs, 2).unwrap_err();
        assert!(err.contains("not the full image"), "got: {err}");
    }

    #[test]
    fn check_equirect_uvs_rejects_a_uv_count_mismatch() {
        let err = check_equirect_uvs(&[[0.0, 0.0]], 2).unwrap_err();
        assert!(err.contains("1 UVs against 2 vertices"), "got: {err}");
    }

    #[test]
    fn check_panorama_aspect_brackets_the_landscape_range() {
        check_panorama_aspect(4096, 2048).expect("2:1 is the ideal equirect");
        check_panorama_aspect(3840, 2160).expect("16:9 is how many ship");
        assert!(check_panorama_aspect(1024, 1024).is_err());
        assert!(check_panorama_aspect(1024, 2048).is_err());
        assert!(check_panorama_aspect(8192, 1024).is_err());
        assert!(check_panorama_aspect(1024, 0).is_err());
    }
}
