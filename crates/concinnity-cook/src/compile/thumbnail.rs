//! Baked asset thumbnails: small PNG previews of a compiled world's visual
//! resources (textures, materials, meshes, environment maps), rendered on the
//! CPU from the compiled payloads and held in the build cache segment. Runs in
//! the disk-build tail (write_build_outputs), so the editor's per-edit preview
//! rebuilds never pay for it; content addressing makes a re-bake of unchanged
//! content an index lookup.
//!
//! The rendering here is pure -- `collect` decides what a bake produces and
//! answers reuse through a caller-supplied lookup -- and `bake_thumbnails` is
//! the thin half that puts the result into `cache/1`
//! (`crate::cache::thumbnails`).

use crate::pipeline::PipelineResult;
use concinnity_core::bake::texture as texture_payload;
use concinnity_core::bake::texture::{TextureFormat, downscale_rgba};
use concinnity_core::blob::{BlobAssetDef, PayloadLocator, ResourceKind, ResourceRecord};
use concinnity_core::gfx::raster;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

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
pub(crate) struct ThumbReport {
    pub baked: usize,
    pub reused: usize,
    // Resources with no bakeable preview (unsupported compressed format,
    // undecodable payload). The browser falls back to a typed icon.
    pub skipped: usize,
}

// What one bake produced: the images the segment does not already hold, the
// whole set's name -> key map, and the counts.
#[derive(Debug, Default)]
pub(crate) struct ThumbBake {
    pub report: ThumbReport,
    pub images: Vec<(String, Vec<u8>)>,
    pub names: Vec<(String, String)>,
}

// Bake into the build cache segment, holding what it produced for the flush
// that follows. Every step is best effort: with no state root nothing is
// stored, which costs the editor its previews and never a build.
pub(crate) fn bake_thumbnails(result: &PipelineResult) -> ThumbReport {
    let bake = collect(result, &crate::cache::thumbnails::holds);
    crate::cache::thumbnails::hold(&bake.images, &bake.names);
    crate::cache::flush();
    bake.report
}

// Render every previewable asset, asking `held` whether the segment already
// carries the key each one hashes to. No I/O: the caller decides what to do
// with the images.
pub(crate) fn collect(result: &PipelineResult, held: &dyn Fn(&str) -> bool) -> ThumbBake {
    let mut out = Collector {
        held,
        seen: HashSet::new(),
        bake: ThumbBake::default(),
    };
    // Average texture colors by handle, feeding material sphere albedo.
    let mut texture_tints: HashMap<u32, [f32; 3]> = HashMap::new();

    // Textures first: material swatches sample their averages.
    for (record, name) in records_of(result, ResourceKind::Texture) {
        let Some(bytes) = payload_of(result, record) else {
            out.skip();
            continue;
        };
        let Some((w, h, rgba)) = texture_preview(bytes) else {
            out.skip();
            continue;
        };
        texture_tints.insert(record.handle, average_color(&rgba));
        out.add(name, key_of(&[b"texture", &hash_bytes(bytes)]), w, h, &rgba);
    }

    // Surface colors by material handle, feeding model sub-mesh shading.
    let mut material_colors: HashMap<u32, [f32; 3]> = HashMap::new();
    for (record, name) in records_of(result, ResourceKind::Material) {
        let Ok(mat) = postcard::from_bytes::<crate::components::Material>(&record.data_bytes)
        else {
            out.skip();
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
        material_colors.insert(record.handle, tinted);
        let img = raster::shade_sphere(THUMB_SIZE, tinted, mat.roughness, mat.metallic);
        // The visual inputs are the key: the baked record plus the sampled
        // albedo average, so retexturing re-bakes.
        let albedo_bytes: Vec<u8> = tinted.iter().flat_map(|c| c.to_le_bytes()).collect();
        let key = key_of(&[b"material", &record.data_bytes, &albedo_bytes]);
        out.add(name, key, img.width, img.height, &img.rgba);
    }

    for (record, name) in records_of(result, ResourceKind::Mesh) {
        let Some(bytes) = payload_of(result, record) else {
            out.skip();
            continue;
        };
        let Ok((verts, indices, _)) =
            concinnity_core::gfx::mesh_payload::deserialise_with_lods(bytes)
        else {
            out.skip();
            continue;
        };
        let img = raster::shade_mesh(&verts, &indices, THUMB_SIZE, MESH_COLOR);
        let key = key_of(&[b"mesh", &hash_bytes(bytes)]);
        out.add(name, key, img.width, img.height, &img.rgba);
    }

    collect_models(result, &material_colors, &mut out);

    for (record, name) in records_of(result, ResourceKind::EnvironmentMap) {
        let Some(bytes) = payload_of(result, record) else {
            out.skip();
            continue;
        };
        let Some((w, h, rgba)) = envmap_preview(bytes) else {
            out.skip();
            continue;
        };
        out.add(name, key_of(&[b"envmap", &hash_bytes(bytes)]), w, h, &rgba);
    }

    out.bake
}

// Accumulates a bake: renders reach it as encoded images, keys it has already
// answered for reach it as reuse.
struct Collector<'a> {
    held: &'a dyn Fn(&str) -> bool,
    seen: HashSet<String>,
    bake: ThumbBake,
}

impl Collector<'_> {
    // Index `name` against `key`, encoding the image unless the segment (or an
    // earlier asset of this same bake) already carries it.
    fn add(&mut self, name: &str, key: String, width: u32, height: u32, rgba: &[u8]) {
        if self.seen.contains(&key) || (self.held)(&key) {
            self.bake.report.reused += 1;
        } else {
            let Some(png) = encode_png(width, height, rgba) else {
                self.skip();
                return;
            };
            self.bake.images.push((key.clone(), png));
            self.bake.report.baked += 1;
        }
        self.seen.insert(key.clone());
        self.bake.names.push((name.to_string(), key));
    }

    // An asset with no bakeable preview: it is counted, and left out of the
    // map so a consumer falls back to a typed icon rather than to a key that
    // addresses nothing.
    fn skip(&mut self) {
        self.bake.report.skipped += 1;
    }
}

// Compose each Model def's sub-meshes into one render, each part shaded with
// its material's surface color. Sub-mesh payloads resolve by unified
// mesh-source handle: resource-stream Mesh records lead the space; component
// mesh sources (ProceduralMesh) resolve through `mesh_component_names` to
// their def's payload.
fn collect_models(
    result: &PipelineResult,
    material_colors: &HashMap<u32, [f32; 3]>,
    out: &mut Collector,
) {
    let mut mesh_payloads: HashMap<u32, &[u8]> = HashMap::new();
    for (record, _) in records_of(result, ResourceKind::Mesh) {
        if let Some(bytes) = payload_of(result, record) {
            mesh_payloads.insert(record.handle, bytes);
        }
    }
    let def_index: HashMap<&str, usize> = result
        .names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    for (handle, name) in &result.mesh_component_names {
        let bytes = def_index
            .get(name.as_str())
            .and_then(|&i| def_payload_of(result, &result.defs[i]));
        if let Some(bytes) = bytes {
            mesh_payloads.insert(*handle, bytes);
        }
    }
    let Some(model_disc) =
        crate::registry::RegisteredType::parse("Model").and_then(|t| t.discriminant())
    else {
        return;
    };
    for (def, name) in result.defs.iter().zip(result.names.iter()) {
        if def.discriminant != model_disc {
            continue;
        }
        let Ok(model) = postcard::from_bytes::<crate::components::Model>(&def.args_bytes) else {
            out.skip();
            continue;
        };
        let mut key_inputs: Vec<Vec<u8>> = vec![b"model".to_vec()];
        let mut decoded = Vec::new();
        for sub in &model.meshes {
            let Some(bytes) = sub.mesh.and_then(|h| mesh_payloads.get(&h.0).copied()) else {
                continue;
            };
            let Ok((verts, indices, _)) =
                concinnity_core::gfx::mesh_payload::deserialise_with_lods(bytes)
            else {
                continue;
            };
            let color = sub
                .material
                .and_then(|m| material_colors.get(&m.0).copied())
                .unwrap_or(MESH_COLOR);
            key_inputs.push(hash_bytes(bytes));
            key_inputs.push(color.iter().flat_map(|c| c.to_le_bytes()).collect());
            decoded.push((verts, indices, color));
        }
        if decoded.is_empty() {
            out.skip();
            continue;
        }
        let parts: Vec<raster::MeshPart> = decoded
            .iter()
            .map(|(verts, indices, color)| raster::MeshPart {
                verts,
                indices,
                color: *color,
            })
            .collect();
        let img = raster::shade_parts(&parts, THUMB_SIZE);
        let inputs: Vec<&[u8]> = key_inputs.iter().map(|v| v.as_slice()).collect();
        out.add(name, key_of(&inputs), img.width, img.height, &img.rgba);
    }
}

// The payload bytes a component def's locator points at.
fn def_payload_of<'a>(result: &'a PipelineResult, def: &BlobAssetDef) -> Option<&'a [u8]> {
    slice_payload(&result.payloads, def.payload.as_ref()?)
}

// The records of one kind, paired with their lock names (resource_locks is
// index-aligned with resources).
fn records_of(
    result: &PipelineResult,
    kind: ResourceKind,
) -> impl Iterator<Item = (&ResourceRecord, &str)> {
    result
        .resources
        .iter()
        .zip(result.resource_locks.iter())
        .filter(move |(r, _)| r.resource_kind == kind as u8)
        .map(|(r, l)| (r, l.name.as_str()))
}

// The payload bytes a record's locator points at, sliced out of the in-memory
// blob payload sections.
fn payload_of<'a>(result: &'a PipelineResult, record: &ResourceRecord) -> Option<&'a [u8]> {
    slice_payload(&result.payloads, record.payload.as_ref()?)
}

fn slice_payload<'a>(payloads: &'a [Vec<u8>], loc: &PayloadLocator) -> Option<&'a [u8]> {
    let blob = payloads.get(loc.blob_index as usize)?;
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
        // No CPU decoder for BC7; the browser shows a typed icon instead.
        TextureFormat::Bc7 => return None,
        other => crate::codec::bcn::decode(other, &mip.data, mip.width, mip.height).ok()?,
    };
    Some(downscale_rgba(mip.width, mip.height, rgba, THUMB_SIZE))
}

// A tonemapped face of an environment map's prefilter chain: the mip whose
// face size is nearest the thumbnail budget.
fn envmap_preview(payload: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let view = concinnity_core::bake::environment_map::deserialise(payload).ok()?;
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

// Encode one RGBA8 image as a PNG. `None` when the encoder refuses the image,
// which costs that asset its preview and nothing else.
fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().ok()?.write_image_data(rgba).ok()?;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::blob::AssetKind;

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
            kind: kind.to_string(),
            payload_blob: Some(0),
            ..Default::default()
        }
    }

    fn empty_result() -> PipelineResult {
        PipelineResult {
            defs: Vec::new(),
            names: Vec::new(),
            resources: Vec::new(),
            scene_groups: Vec::new(),
            mesh_bounds: Vec::new(),
            physics_budget: None,
            mesh_component_names: Vec::new(),
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
        let tex = texture_payload::serialise(&concinnity_core::bake::texture::TextureImage::rgba8(
            2,
            2,
            vec![
                255, 0, 0, 255, 255, 0, 0, 255, //
                255, 0, 0, 255, 255, 0, 0, 255,
            ],
        ));
        let mesh = crate::compile::geometry::compile_mesh_payload(
            &serde_json::json!({ "generator": "box" }),
        )
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

    // Every key a bake produced, the reuse lookup a later bake asks against.
    fn held_by(bake: &ThumbBake) -> HashSet<String> {
        bake.images.iter().map(|(key, _)| key.clone()).collect()
    }

    fn key_for(bake: &ThumbBake, name: &str) -> String {
        bake.names
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, key)| key.clone())
            .unwrap_or_else(|| panic!("{name} indexed"))
    }

    fn image_for(bake: &ThumbBake, name: &str) -> Vec<u8> {
        let key = key_for(bake, name);
        bake.images
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, png)| png.clone())
            .expect("an image under the indexed key")
    }

    #[test]
    fn bakes_pngs_and_an_index_then_reuses_on_rebake() {
        let result = textured_meshed_result();
        let bake = collect(&result, &|_| false);
        assert_eq!(bake.report.baked, 2, "texture + mesh");
        assert_eq!(bake.report.reused, 0);

        for name in ["red_tex", "box_mesh"] {
            let png = image_for(&bake, name);
            let decoder = png::Decoder::new(std::io::Cursor::new(png));
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            assert!(info.width <= THUMB_SIZE && info.height <= THUMB_SIZE);
        }

        let held = held_by(&bake);
        let again = collect(&result, &|key| held.contains(key));
        assert_eq!(again.report.baked, 0, "unchanged content re-bakes nothing");
        assert_eq!(again.report.reused, 2);
        assert_eq!(
            again.names, bake.names,
            "the map is the whole set either way"
        );
    }

    // Two assets whose previews hash alike index one image, which is what lets
    // the map stay a separate entry rather than being folded into the keys.
    #[test]
    fn two_assets_sharing_a_preview_bake_one_image() {
        let mut result = textured_meshed_result();
        let tex = result.resources[0].clone();
        result.resources.push(tex);
        result.resource_locks.push(lock("red_tex_copy", "Texture"));

        let bake = collect(&result, &|_| false);
        assert_eq!(bake.report.baked, 2, "texture + mesh");
        assert_eq!(bake.report.reused, 1, "the second texture reuses the first");
        assert_eq!(bake.names.len(), 3);
        assert_eq!(key_for(&bake, "red_tex"), key_for(&bake, "red_tex_copy"));
    }

    #[test]
    fn material_swatch_keys_on_record_and_albedo_average() {
        let mut result = textured_meshed_result();
        let mat = crate::components::Material {
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
        let bake = collect(&result, &|_| false);
        assert_eq!(bake.report.baked, 3);
        assert!(!key_for(&bake, "plaster").is_empty());
    }

    #[test]
    fn model_thumbnails_compose_resource_and_component_sub_meshes() {
        // A resource-stream mesh (handle 0) and a component-compiled mesh
        // (ProceduralMesh, unified handle 1) both feed the model.
        let mut result = textured_meshed_result();
        let ball = crate::compile::geometry::compile_mesh_payload(
            &serde_json::json!({ "generator": "sphere" }),
        )
        .unwrap();
        let ball_off = result.payloads[0].len() as u64;
        let ball_len = ball.len() as u64;
        result.payloads[0].extend_from_slice(&ball);
        let mat = crate::components::Material {
            tint: [0.2, 0.9, 0.2],
            ..Default::default()
        };
        result.resources.push(ResourceRecord {
            resource_kind: ResourceKind::Material as u8,
            handle: 0,
            payload: None,
            data_bytes: postcard::to_allocvec(&mat).unwrap(),
        });
        result.resource_locks.push(lock("green", "Material"));

        let model_disc = crate::registry::RegisteredType::parse("Model")
            .unwrap()
            .discriminant()
            .unwrap();
        let model = crate::components::Model {
            meshes: vec![
                crate::components::SubMeshRef {
                    mesh: Some(crate::ecs::MeshHandle(0)),
                    material: Some(crate::ecs::MaterialHandle(0)),
                },
                crate::components::SubMeshRef {
                    mesh: Some(crate::ecs::MeshHandle(1)),
                    material: None,
                },
            ],
            ..Default::default()
        };
        let def = |disc: u8, args_bytes: Vec<u8>, payload| BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            discriminant: disc,
            args_bytes,
            payload,
        };
        result.defs = vec![
            def(model_disc, postcard::to_allocvec(&model).unwrap(), None),
            // The ProceduralMesh def carrying unified handle 1's payload.
            def(
                model_disc.wrapping_add(1),
                Vec::new(),
                Some(PayloadLocator {
                    blob_index: 0,
                    offset: ball_off,
                    len: ball_len,
                }),
            ),
        ];
        result.names = vec!["crate_model".to_string(), "proc_ball".to_string()];
        result.mesh_component_names = vec![(1, "proc_ball".to_string())];

        let bake = collect(&result, &|_| false);
        // Texture + mesh + material + model.
        assert_eq!(bake.report.baked, 4, "{:?}", bake.report);
        assert!(!image_for(&bake, "crate_model").is_empty());

        // A model whose sub-meshes all fail to resolve is skipped, not fatal.
        let mut broken = result;
        let orphan = crate::components::Model {
            meshes: vec![crate::components::SubMeshRef {
                mesh: Some(crate::ecs::MeshHandle(99)),
                material: None,
            }],
            ..Default::default()
        };
        broken.defs[0].args_bytes = postcard::to_allocvec(&orphan).unwrap();
        let bake = collect(&broken, &|_| false);
        assert_eq!(bake.report.baked, 3);
        assert_eq!(bake.report.skipped, 1);
    }

    #[test]
    fn undecodable_payloads_are_skipped_not_fatal() {
        let mut result = empty_result();
        result.payloads = vec![vec![0u8; 16]];
        result.resources = vec![record(ResourceKind::Texture, 0, 0, 0, 16)];
        result.resource_locks = vec![lock("broken", "Texture")];
        let bake = collect(&result, &|_| false);
        assert_eq!(bake.report.baked, 0);
        assert_eq!(bake.report.skipped, 1);
        assert!(bake.names.is_empty(), "a skipped asset is not indexed");
    }

    #[test]
    fn out_of_range_locators_are_skipped() {
        let mut result = empty_result();
        result.payloads = vec![vec![0u8; 4]];
        result.resources = vec![record(ResourceKind::Mesh, 0, 0, 2, 100)];
        result.resource_locks = vec![lock("truncated", "Mesh")];
        let bake = collect(&result, &|_| false);
        assert_eq!(
            bake.report,
            ThumbReport {
                baked: 0,
                reused: 0,
                skipped: 1
            }
        );
    }
}
