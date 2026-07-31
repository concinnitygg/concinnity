// src/thumbnail.rs
//
// Baked asset thumbnails: small PNG previews of a compiled world's visual
// resources (textures, materials, meshes, environment maps), rendered on the
// CPU from the compiled payloads and written content-addressed to the
// thumbnails directory. Runs in the disk-build tail (write_build_outputs), so
// the editor's per-edit preview rebuilds never pay for it; content addressing
// makes a re-bake of unchanged content a file-existence check.
//
// Layout: `<thumbnails dir>/<sha256>.png` plus an `index.json` written each
// bake mapping asset name -> key, so a browser can resolve a name without
// recomputing hashes.

use crate::pipeline::PipelineResult;
use concinnity_blob::ResourceKind;
use concinnity_core::build::texture as texture_payload;
use concinnity_core::build::texture::{TextureFormat, downscale_rgba};
use concinnity_core::gfx::raster;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

// Folded into every key: bump when the rendering itself changes so stale
// images re-bake. Version history:
//   1: initial bake (texture / material / mesh / environment map).
const THUMB_FORMAT_VERSION: u32 = 1;

// Thumbnail edge length. Textures keep their aspect within this budget; mesh
// and material renders are square.
const THUMB_SIZE: u32 = 128;

// The neutral surface color mesh previews shade with.
const MESH_COLOR: [f32; 3] = [0.72, 0.72, 0.75];

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ThumbReport {
    pub baked: usize,
    pub reused: usize,
    // Resources with no bakeable preview (unsupported compressed format,
    // undecodable payload). The browser falls back to a typed icon.
    pub skipped: usize,
}

// Bake into the project's thumbnails directory.
pub fn bake_thumbnails(result: &PipelineResult) -> std::io::Result<ThumbReport> {
    bake_thumbnails_in(&concinnity_core::paths::thumbnails_dir(), result)
}

// Bake into `dir` (tests point this at a scratch directory).
pub fn bake_thumbnails_in(dir: &Path, result: &PipelineResult) -> std::io::Result<ThumbReport> {
    std::fs::create_dir_all(dir)?;
    let mut report = ThumbReport::default();
    let mut index: Vec<(String, String)> = Vec::new();
    // Average texture colors by handle, feeding material sphere albedo.
    let mut texture_tints: HashMap<u32, [f32; 3]> = HashMap::new();

    // Textures first: material swatches sample their averages.
    for (record, name) in records_of(result, ResourceKind::Texture) {
        let Some(bytes) = payload_of(result, record) else {
            report.skipped += 1;
            continue;
        };
        let Some((w, h, rgba)) = texture_preview(bytes) else {
            report.skipped += 1;
            continue;
        };
        texture_tints.insert(record.handle, average_color(&rgba));
        let key = key_of(&[b"texture", &hash_bytes(bytes)]);
        write_png(dir, &key, w, h, &rgba, &mut report)?;
        index.push((name.to_string(), key));
    }

    for (record, name) in records_of(result, ResourceKind::Material) {
        let Ok(mat) = postcard::from_bytes::<crate::assets::Material>(&record.data_bytes) else {
            report.skipped += 1;
            continue;
        };
        let albedo = mat
            .albedo
            .and_then(|h| texture_tints.get(&h.0).copied())
            .unwrap_or([0.8, 0.8, 0.8]);
        let tinted = [
            albedo[0] * mat.tint[0],
            albedo[1] * mat.tint[1],
            albedo[2] * mat.tint[2],
        ];
        let img = raster::shade_sphere(THUMB_SIZE, tinted, mat.roughness, mat.metallic);
        // The visual inputs are the key: the baked record plus the sampled
        // albedo average, so retexturing re-bakes.
        let albedo_bytes: Vec<u8> = tinted.iter().flat_map(|c| c.to_le_bytes()).collect();
        let key = key_of(&[b"material", &record.data_bytes, &albedo_bytes]);
        write_png(dir, &key, img.width, img.height, &img.rgba, &mut report)?;
        index.push((name.to_string(), key));
    }

    for (record, name) in records_of(result, ResourceKind::Mesh) {
        let Some(bytes) = payload_of(result, record) else {
            report.skipped += 1;
            continue;
        };
        let Ok((verts, indices, _)) =
            concinnity_core::gfx::mesh_payload::deserialise_with_lods(bytes)
        else {
            report.skipped += 1;
            continue;
        };
        let img = raster::shade_mesh(&verts, &indices, THUMB_SIZE, MESH_COLOR);
        let key = key_of(&[b"mesh", &hash_bytes(bytes)]);
        write_png(dir, &key, img.width, img.height, &img.rgba, &mut report)?;
        index.push((name.to_string(), key));
    }

    for (record, name) in records_of(result, ResourceKind::EnvironmentMap) {
        let Some(bytes) = payload_of(result, record) else {
            report.skipped += 1;
            continue;
        };
        let Some((w, h, rgba)) = envmap_preview(bytes) else {
            report.skipped += 1;
            continue;
        };
        let key = key_of(&[b"envmap", &hash_bytes(bytes)]);
        write_png(dir, &key, w, h, &rgba, &mut report)?;
        index.push((name.to_string(), key));
    }

    write_index(dir, &index)?;
    Ok(report)
}

// The records of one kind, paired with their lock names (resource_locks is
// index-aligned with resources).
fn records_of(
    result: &PipelineResult,
    kind: ResourceKind,
) -> impl Iterator<Item = (&concinnity_blob::ResourceRecord, &str)> {
    result
        .resources
        .iter()
        .zip(result.resource_locks.iter())
        .filter(move |(r, _)| r.resource_kind == kind as u8)
        .map(|(r, l)| (r, l.name.as_str()))
}

// The payload bytes a record's locator points at, sliced out of the in-memory
// blob payload sections.
fn payload_of<'a>(
    result: &'a PipelineResult,
    record: &concinnity_blob::ResourceRecord,
) -> Option<&'a [u8]> {
    let loc = record.payload.as_ref()?;
    let blob = result.payloads.get(loc.blob_index as usize)?;
    let start = usize::try_from(loc.offset).ok()?;
    let end = start.checked_add(usize::try_from(loc.len).ok()?)?;
    blob.get(start..end)
}

// An RGBA8 preview of a texture payload: the mip nearest the thumbnail size
// (block formats decode just that mip), box-filtered into the size budget.
fn texture_preview(payload: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let image = texture_payload::deserialise(payload).ok()?;
    let mip = image
        .mips
        .iter()
        .rfind(|m| m.width >= THUMB_SIZE || m.height >= THUMB_SIZE)
        .or_else(|| image.mips.first())?;
    let rgba = match image.format {
        TextureFormat::Rgba8 => mip.data.clone(),
        TextureFormat::Bc1 => crate::bcn::decode_bc1(&mip.data, mip.width, mip.height).ok()?,
        TextureFormat::Bc3 => crate::bcn::decode_bc3(&mip.data, mip.width, mip.height).ok()?,
        TextureFormat::Bc5 => crate::bcn::decode_bc5(&mip.data, mip.width, mip.height).ok()?,
        // No CPU decoder; the browser shows a typed icon instead.
        TextureFormat::Bc7 => return None,
    };
    Some(downscale_rgba(mip.width, mip.height, rgba, THUMB_SIZE))
}

// A tonemapped face of an environment map's prefilter chain: the mip whose
// face size is nearest the thumbnail budget.
fn envmap_preview(payload: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let view = concinnity_core::build::environment_map::deserialise(payload).ok()?;
    let (mip_index, face) = view
        .prefilter_mip_bytes
        .iter()
        .enumerate()
        .map(|(i, _)| (i, view.prefilter_face >> i))
        .filter(|&(_, s)| s > 0)
        .min_by_key(|&(_, s)| s.abs_diff(THUMB_SIZE))?;
    let bytes = view.prefilter_mip_bytes.get(mip_index)?;
    let face_px = (face * face) as usize;
    // Face layout: 6 consecutive faces of face*face RGBA32F texels. Face 0
    // (+X) is representative enough for a preview.
    let face_bytes = bytes.get(0..face_px * 16)?;
    let mut rgba = Vec::with_capacity(face_px * 4);
    for texel in face_bytes.chunks_exact(16) {
        for c in 0..3 {
            let v = f32::from_le_bytes(texel[c * 4..c * 4 + 4].try_into().ok()?);
            // Reinhard + gamma, clamped: an HDR sky stays readable.
            let mapped = (v.max(0.0) / (1.0 + v.max(0.0))).powf(1.0 / 2.2);
            rgba.push((mapped * 255.0) as u8);
        }
        rgba.push(255);
    }
    Some((face, face, rgba))
}

// The average color of an RGBA8 image (alpha-weighted texels only).
fn average_color(rgba: &[u8]) -> [f32; 3] {
    let mut acc = [0.0f64; 3];
    let mut n = 0u64;
    for p in rgba.chunks_exact(4) {
        if p[3] == 0 {
            continue;
        }
        for c in 0..3 {
            acc[c] += p[c] as f64;
        }
        n += 1;
    }
    if n == 0 {
        return [0.8, 0.8, 0.8];
    }
    [
        (acc[0] / n as f64 / 255.0) as f32,
        (acc[1] / n as f64 / 255.0) as f32,
        (acc[2] / n as f64 / 255.0) as f32,
    ]
}

fn hash_bytes(bytes: &[u8]) -> Vec<u8> {
    Sha256::digest(bytes).to_vec()
}

// The content key: version + kind tag + each input, length-prefixed so
// adjacent inputs cannot alias.
fn key_of(inputs: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(THUMB_FORMAT_VERSION.to_le_bytes());
    for input in inputs {
        hasher.update((input.len() as u64).to_le_bytes());
        hasher.update(input);
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// Write `<key>.png` unless it already exists (the content-addressed reuse).
fn write_png(
    dir: &Path,
    key: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
    report: &mut ThumbReport,
) -> std::io::Result<()> {
    let path = dir.join(format!("{key}.png"));
    if path.exists() {
        report.reused += 1;
        return Ok(());
    }
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    writer
        .write_image_data(rgba)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    report.baked += 1;
    Ok(())
}

// Rewrite the name -> key index for the whole bake (it is fully derived).
fn write_index(dir: &Path, index: &[(String, String)]) -> std::io::Result<()> {
    let map: serde_json::Map<String, serde_json::Value> = index
        .iter()
        .map(|(name, key)| (name.clone(), serde_json::Value::String(key.clone())))
        .collect();
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(map))?;
    std::fs::write(dir.join("index.json"), text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_blob::{PayloadLocator, ResourceRecord};

    fn record(kind: ResourceKind, handle: u32, blob: u32, offset: u64, len: u64) -> ResourceRecord {
        ResourceRecord {
            resource_kind: kind as u8,
            handle,
            payload: Some(PayloadLocator {
                blob_index: blob,
                offset,
                len,
            }),
            data_bytes: Vec::new(),
        }
    }

    fn lock(name: &str, kind: &str) -> crate::blob::LockedResource {
        crate::blob::LockedResource {
            name: name.to_string(),
            id: None,
            kind: kind.to_string(),
            handle: 0,
            args_hash: String::new(),
            payload_blob: Some(0),
        }
    }

    fn empty_result() -> PipelineResult {
        PipelineResult {
            defs: Vec::new(),
            names: Vec::new(),
            resources: Vec::new(),
            scene_groups: Vec::new(),
            mesh_bounds: Vec::new(),
            payloads: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
            texture_sources: Vec::new(),
            mesh_sources: Vec::new(),
            resource_locks: Vec::new(),
        }
    }

    // A tiny RGBA8 texture payload, a box mesh payload, and their records.
    fn textured_meshed_result() -> PipelineResult {
        let mut result = empty_result();
        let tex =
            texture_payload::serialise(&concinnity_core::build::texture::TextureImage::rgba8(
                2,
                2,
                vec![
                    255, 0, 0, 255, 255, 0, 0, 255, //
                    255, 0, 0, 255, 255, 0, 0, 255,
                ],
            ));
        let mesh =
            crate::geometry::compile_mesh_payload(&serde_json::json!({ "generator": "box" }))
                .unwrap();
        let mut payload = tex.clone();
        let mesh_off = payload.len() as u64;
        payload.extend_from_slice(&mesh);
        result.payloads = vec![payload];
        result.resources = vec![
            record(ResourceKind::Texture, 0, 0, 0, tex.len() as u64),
            record(ResourceKind::Mesh, 0, 0, mesh_off, mesh.len() as u64),
        ];
        result.resource_locks = vec![lock("red_tex", "Texture"), lock("box_mesh", "Mesh")];
        result
    }

    #[test]
    fn bakes_pngs_and_an_index_then_reuses_on_rebake() {
        let dir = tempfile::tempdir().unwrap();
        let result = textured_meshed_result();
        let report = bake_thumbnails_in(dir.path(), &result).unwrap();
        assert_eq!(report.baked, 2, "texture + mesh");
        assert_eq!(report.reused, 0);

        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("index.json")).unwrap())
                .unwrap();
        for name in ["red_tex", "box_mesh"] {
            let key = index[name].as_str().unwrap();
            let path = dir.path().join(format!("{key}.png"));
            assert!(path.exists(), "{name} thumbnail written");
            let decoder =
                png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            assert!(info.width <= THUMB_SIZE && info.height <= THUMB_SIZE);
        }

        let again = bake_thumbnails_in(dir.path(), &result).unwrap();
        assert_eq!(again.baked, 0, "unchanged content re-bakes nothing");
        assert_eq!(again.reused, 2);
    }

    #[test]
    fn material_swatch_keys_on_record_and_albedo_average() {
        let dir = tempfile::tempdir().unwrap();
        let mut result = textured_meshed_result();
        let mat = crate::assets::Material {
            roughness: 0.4,
            metallic: 0.1,
            ..Default::default()
        };
        result.resources.push(ResourceRecord {
            resource_kind: ResourceKind::Material as u8,
            handle: 0,
            payload: None,
            data_bytes: postcard::to_allocvec(&mat).unwrap(),
        });
        result.resource_locks.push(lock("plaster", "Material"));
        let report = bake_thumbnails_in(dir.path(), &result).unwrap();
        assert_eq!(report.baked, 3);
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("index.json")).unwrap())
                .unwrap();
        assert!(index["plaster"].is_string());
    }

    #[test]
    fn undecodable_payloads_are_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let mut result = empty_result();
        result.payloads = vec![vec![0u8; 16]];
        result.resources = vec![record(ResourceKind::Texture, 0, 0, 0, 16)];
        result.resource_locks = vec![lock("broken", "Texture")];
        let report = bake_thumbnails_in(dir.path(), &result).unwrap();
        assert_eq!(report.baked, 0);
        assert_eq!(report.skipped, 1);
    }

    #[test]
    fn out_of_range_locators_are_skipped() {
        let mut result = empty_result();
        result.payloads = vec![vec![0u8; 4]];
        result.resources = vec![record(ResourceKind::Mesh, 0, 0, 2, 100)];
        result.resource_locks = vec![lock("truncated", "Mesh")];
        let dir = tempfile::tempdir().unwrap();
        let report = bake_thumbnails_in(dir.path(), &result).unwrap();
        assert_eq!(
            report,
            ThumbReport {
                baked: 0,
                reused: 0,
                skipped: 1
            }
        );
    }
}
