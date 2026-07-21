// src/ktx2.rs
//
// Compiles a `.ktx2` texture container into the tagged multi-mip payload the
// runtime uploads. Build-only, like the other source decoders in this crate.
//
// KTX2 arrives in two shapes:
//   - A real block-compressed format (`vkFormat` set to a BCn block format).
//     The blocks pass straight through into the payload, one mip level at a
//     time, inflating any zstd supercompression first.
//   - Basis Universal (`vkFormat` = VK_FORMAT_UNDEFINED). ETC1S transcodes to
//     BC1 (BC3 when the texture carries alpha); UASTC LDR transcodes to BC7.
//     Every container mip level is transcoded.
//
// A compressed source without a mip chain (a single level) decodes or
// transcodes to RGBA8 instead, so the runtime mip generator restores full
// minification quality. There is no cook-time BCn encoder, so RGBA8 sources
// (PNG / JPEG) are never compressed here.
//
// Cubemaps, texture arrays, and 3D textures are rejected: the payload and the
// runtime upload path handle single-layer 2D textures only.

use basisu::{DecodeFlags, SourceFormat, TargetFormat, Transcoder};
use concinnity_core::build::texture::{TextureFormat, TextureImage, TextureMip};
use ktx2::{Format, Reader, SupercompressionScheme};

// VkFormat values for the BCn block formats this compiler accepts, matched
// against `ktx2::Format::value()`. SRGB variants are treated as linear data
// (the engine has no sRGB sampling distinction).
const VK_BC1_RGB_UNORM: u32 = 131;
const VK_BC1_RGB_SRGB: u32 = 132;
const VK_BC1_RGBA_UNORM: u32 = 133;
const VK_BC1_RGBA_SRGB: u32 = 134;
const VK_BC3_UNORM: u32 = 137;
const VK_BC3_SRGB: u32 = 138;
const VK_BC5_UNORM: u32 = 141;
const VK_BC7_UNORM: u32 = 145;
const VK_BC7_SRGB: u32 = 146;

// Compile a `.ktx2` byte buffer into a tagged texture image.
pub fn compile_ktx2(bytes: &[u8]) -> Result<TextureImage, String> {
    let reader = Reader::new(bytes).map_err(|e| format!("not a valid KTX2 container: {:?}", e))?;
    let header = reader.header();

    if header.face_count > 1 {
        return Err(format!(
            "KTX2 cubemaps are not supported ({} faces); only single-layer 2D textures are",
            header.face_count
        ));
    }
    if header.layer_count > 1 {
        return Err(format!(
            "KTX2 texture arrays are not supported ({} layers); only single-layer 2D textures are",
            header.layer_count
        ));
    }
    if header.pixel_depth > 1 {
        return Err(format!(
            "KTX2 3D textures are not supported (depth {}); only 2D textures are",
            header.pixel_depth
        ));
    }
    let width = header.pixel_width;
    let height = header.pixel_height;
    if width == 0 || height == 0 {
        return Err(format!("KTX2 has a zero dimension {}x{}", width, height));
    }

    match header.format {
        Some(format) => compile_block_format(&reader, format, width, height),
        None => compile_basis(bytes),
    }
}

// A real BCn `vkFormat`: pass the blocks through. Every level is copied (zstd
// supercompression inflated) into a compressed mip. A single-level source with
// no chain falls back to RGBA8 so runtime mip generation applies.
fn compile_block_format(
    reader: &Reader<&[u8]>,
    format: Format,
    width: u32,
    height: u32,
) -> Result<TextureImage, String> {
    let tex_format = match format.value() {
        VK_BC1_RGB_UNORM | VK_BC1_RGB_SRGB | VK_BC1_RGBA_UNORM | VK_BC1_RGBA_SRGB => {
            TextureFormat::Bc1
        }
        VK_BC3_UNORM | VK_BC3_SRGB => TextureFormat::Bc3,
        VK_BC5_UNORM => TextureFormat::Bc5,
        VK_BC7_UNORM | VK_BC7_SRGB => TextureFormat::Bc7,
        other => {
            return Err(format!(
                "unsupported KTX2 vkFormat {:?} ({}); supported: BC1, BC3, BC5, BC7 \
                 (unorm/srgb), or Basis Universal (ETC1S / UASTC LDR)",
                format, other
            ));
        }
    };

    let scheme = reader.header().supercompression_scheme;
    let levels: Vec<&[u8]> = reader.levels().map(|l| l.data).collect();
    if levels.is_empty() {
        return Err("KTX2 declares no mip levels".to_string());
    }

    // Inflate every stored level up front so the no-chain fallback and the
    // passthrough path both work from raw block data.
    let mut level_blocks: Vec<Vec<u8>> = Vec::with_capacity(levels.len());
    for (level, data) in levels.iter().enumerate() {
        let blocks =
            decompress_level(data, scheme).map_err(|e| format!("KTX2 level {}: {}", level, e))?;
        level_blocks.push(blocks);
    }

    // No mip chain: recover RGBA8 so distance minification stays sharp. BC1/3/5
    // have CPU decoders; BC7 has none, so it stays a single compressed level.
    if level_blocks.len() == 1 && tex_format != TextureFormat::Bc7 {
        let rgba = decode_blocks_to_rgba8(tex_format, &level_blocks[0], width, height)?;
        tracing::info!(
            "KTX2 {:?} texture has no mip chain; decoded to RGBA8 for runtime mip generation",
            tex_format
        );
        return Ok(TextureImage::rgba8(width, height, rgba));
    }

    let mut mips = Vec::with_capacity(level_blocks.len());
    for (level, blocks) in level_blocks.into_iter().enumerate() {
        let (mw, mh) = mip_dims(width, height, level as u32);
        let expected = tex_format.mip_byte_len(mw, mh);
        if blocks.len() < expected {
            return Err(format!(
                "KTX2 level {} ({}x{} {:?}) is {} bytes, needs {}",
                level,
                mw,
                mh,
                tex_format,
                blocks.len(),
                expected
            ));
        }
        mips.push(TextureMip {
            width: mw,
            height: mh,
            // A level's stored length can exceed the tight block size when the
            // container pads; keep only the blocks the format defines.
            data: blocks[..expected].to_vec(),
        });
    }
    Ok(TextureImage {
        format: tex_format,
        mips,
    })
}

// Basis Universal (`vkFormat` = UNDEFINED): transcode ETC1S -> BC1/BC3 and
// UASTC LDR -> BC7. A single-level source transcodes to RGBA8 instead.
fn compile_basis(bytes: &[u8]) -> Result<TextureImage, String> {
    let transcoder =
        Transcoder::new(bytes).map_err(|e| format!("KTX2 Basis parse failed: {:?}", e))?;

    let (target, tex_format, flags) = match transcoder.source_format() {
        SourceFormat::Etc1s => {
            if transcoder.has_alpha() {
                (TargetFormat::Bc3Rgba, TextureFormat::Bc3, DecodeFlags::NONE)
            } else {
                (TargetFormat::Bc1Rgb, TextureFormat::Bc1, DecodeFlags::NONE)
            }
        }
        SourceFormat::UastcLdr => (
            TargetFormat::Bc7Rgba,
            TextureFormat::Bc7,
            DecodeFlags::HIGH_QUALITY,
        ),
        other => {
            return Err(format!(
                "unsupported Basis source format {:?}; only ETC1S and UASTC LDR are handled",
                other
            ));
        }
    };

    let level_count = transcoder.level_count();
    let base = transcoder
        .image_level_info(0)
        .map_err(|e| format!("KTX2 Basis level 0 info failed: {:?}", e))?;

    // No mip chain: transcode the base level to RGBA8 for runtime mip
    // generation rather than shipping a single compressed level.
    if level_count <= 1 {
        let rgba = transcoder
            .transcode(0, TargetFormat::Rgba32, DecodeFlags::NONE)
            .map_err(|e| format!("KTX2 Basis RGBA transcode failed: {:?}", e))?;
        tracing::info!(
            "KTX2 Basis texture has no mip chain; transcoded to RGBA8 for runtime mip generation"
        );
        return Ok(TextureImage::rgba8(base.width, base.height, rgba));
    }

    let mut mips = Vec::with_capacity(level_count as usize);
    for level in 0..level_count {
        let info = transcoder
            .image_level_info(level)
            .map_err(|e| format!("KTX2 Basis level {} info failed: {:?}", level, e))?;
        let data = transcoder.transcode(level, target, flags).map_err(|e| {
            format!(
                "KTX2 Basis level {} transcode to {:?} failed: {:?}",
                level, target, e
            )
        })?;
        let expected = tex_format.mip_byte_len(info.width, info.height);
        if data.len() != expected {
            return Err(format!(
                "KTX2 Basis level {} transcoded to {} bytes, {:?} needs {}",
                level,
                data.len(),
                tex_format,
                expected
            ));
        }
        mips.push(TextureMip {
            width: info.width,
            height: info.height,
            data,
        });
    }
    Ok(TextureImage {
        format: tex_format,
        mips,
    })
}

// Decode a KTX2 to RGBA8 for the `cn debug` hot-reload live preview, which
// uploads RGBA8 rather than a compressed texture. Reuses `compile_ktx2`, then
// decodes the base level: Basis and BC1/3/5 have decoders here; BC7 does not.
pub fn decode_ktx2_rgba8(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let image = compile_ktx2(bytes)?;
    let (w, h) = (image.width(), image.height());
    match image.format {
        TextureFormat::Rgba8 => image.into_rgba8(),
        TextureFormat::Bc1 => Ok((w, h, crate::bcn::decode_bc1(&image.mips[0].data, w, h)?)),
        TextureFormat::Bc3 => Ok((w, h, crate::bcn::decode_bc3(&image.mips[0].data, w, h)?)),
        TextureFormat::Bc5 => Ok((w, h, crate::bcn::decode_bc5(&image.mips[0].data, w, h)?)),
        TextureFormat::Bc7 => Err("hot-reload preview of a BC7 KTX2 is not supported".to_string()),
    }
}

// Inflate one stored level given the container's supercompression scheme. Only
// raw and zstd levels reach this path (BasisLZ levels go through the transcoder,
// never here).
fn decompress_level(
    data: &[u8],
    scheme: Option<SupercompressionScheme>,
) -> Result<Vec<u8>, String> {
    match scheme {
        None => Ok(data.to_vec()),
        Some(SupercompressionScheme::Zstandard) => inflate_zstd(data),
        Some(other) => Err(format!(
            "unsupported KTX2 supercompression {:?} for a block format; use none or zstd",
            other
        )),
    }
}

// Decompress a zstd frame into its raw bytes.
fn inflate_zstd(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(data)
        .map_err(|e| format!("zstd frame header invalid: {}", e))?;
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| format!("zstd inflate failed: {}", e))?;
    Ok(out)
}

// Decode BC1/BC3/BC5 blocks back to RGBA8 (the no-mip-chain fallback). BC7 has
// no CPU decoder, so callers never route it here.
fn decode_blocks_to_rgba8(
    format: TextureFormat,
    blocks: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    match format {
        TextureFormat::Bc1 => crate::bcn::decode_bc1(blocks, width, height),
        TextureFormat::Bc3 => crate::bcn::decode_bc3(blocks, width, height),
        TextureFormat::Bc5 => crate::bcn::decode_bc5(blocks, width, height),
        other => Err(format!("no CPU decoder for {:?}", other)),
    }
}

// Mip dimensions for level `level`: each axis halves, floored, minimum 1 (the
// GPU mip convention). Matches how block-compressed containers store their
// chains.
fn mip_dims(width: u32, height: u32, level: u32) -> (u32, u32) {
    ((width >> level).max(1), (height >> level).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Assemble a minimal single-DFD KTX2 container around already-laid-out
    // level data. `levels` are (width, height, bytes), level 0 first.
    fn build_ktx2(
        vk_format: u32,
        supercompression: u32,
        base_w: u32,
        base_h: u32,
        levels: &[(u32, u32, Vec<u8>)],
    ) -> Vec<u8> {
        const HEADER_LEN: usize = 80;
        const LEVEL_INDEX_LEN: usize = 24;
        let level_index_bytes = levels.len() * LEVEL_INDEX_LEN;
        let dfd_offset = HEADER_LEN + level_index_bytes;
        let dfd_len = 4u32; // just the DFD total-length field, no blocks
        let mut data_offset = dfd_offset + dfd_len as usize;

        // Precompute each level's absolute byte offset.
        let mut level_offsets = Vec::with_capacity(levels.len());
        for (_, _, bytes) in levels {
            level_offsets.push(data_offset);
            data_offset += bytes.len();
        }

        let mut out = vec![0u8; data_offset];
        out[0..12].copy_from_slice(&ktx2::MAGIC);
        let put =
            |out: &mut [u8], at: usize, v: u32| out[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let put64 =
            |out: &mut [u8], at: usize, v: u64| out[at..at + 8].copy_from_slice(&v.to_le_bytes());
        put(&mut out, 12, vk_format);
        put(&mut out, 16, 1); // typeSize
        put(&mut out, 20, base_w);
        put(&mut out, 24, base_h);
        put(&mut out, 28, 0); // pixelDepth
        put(&mut out, 32, 0); // layerCount
        put(&mut out, 36, 1); // faceCount
        put(&mut out, 40, levels.len() as u32); // levelCount
        put(&mut out, 44, supercompression);
        // Index section
        put(&mut out, 48, dfd_offset as u32);
        put(&mut out, 52, dfd_len);
        put(&mut out, 56, 0); // kvdByteOffset
        put(&mut out, 60, 0); // kvdByteLength
        put64(&mut out, 64, 0); // sgdByteOffset
        put64(&mut out, 72, 0); // sgdByteLength
        // Level index
        for (i, (_, _, bytes)) in levels.iter().enumerate() {
            let base = HEADER_LEN + i * LEVEL_INDEX_LEN;
            put64(&mut out, base, level_offsets[i] as u64);
            put64(&mut out, base + 8, bytes.len() as u64);
            put64(&mut out, base + 16, bytes.len() as u64);
        }
        // DFD total length field
        put(&mut out, dfd_offset, dfd_len);
        // Level data
        for (i, (_, _, bytes)) in levels.iter().enumerate() {
            let off = level_offsets[i];
            out[off..off + bytes.len()].copy_from_slice(bytes);
        }
        out
    }

    // A solid-red BC1 block (color0 = 565 red, indices 0).
    fn bc1_red_block() -> Vec<u8> {
        vec![0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    }

    #[test]
    fn passthrough_bc1_with_mip_chain() {
        // Two levels: 4x4 and 2x2, one BC1 block each (8 bytes).
        let levels = vec![(4, 4, bc1_red_block()), (2, 2, bc1_red_block())];
        let ktx = build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels);
        let image = compile_ktx2(&ktx).expect("compile");
        assert_eq!(image.format, TextureFormat::Bc1);
        assert_eq!(image.mips.len(), 2);
        assert_eq!((image.width(), image.height()), (4, 4));
        assert_eq!(image.mips[0].data, bc1_red_block());
        assert_eq!((image.mips[1].width, image.mips[1].height), (2, 2));
    }

    #[test]
    fn zstd_supercompressed_bc1_passes_through() {
        let block = bc1_red_block();
        let compressed = ruzstd::encoding::compress_to_vec(
            &block[..],
            ruzstd::encoding::CompressionLevel::Uncompressed,
        );
        // Two levels so the passthrough path (not the RGBA8 fallback) is taken.
        let levels = vec![(4, 4, compressed.clone()), (2, 2, compressed)];
        let ktx = build_ktx2(VK_BC1_RGBA_UNORM, 2, 4, 4, &levels);
        let image = compile_ktx2(&ktx).expect("compile");
        assert_eq!(image.format, TextureFormat::Bc1);
        assert_eq!(image.mips[0].data, bc1_red_block());
    }

    #[test]
    fn single_mip_bc1_falls_back_to_rgba8() {
        let levels = vec![(4, 4, bc1_red_block())];
        let ktx = build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels);
        let image = compile_ktx2(&ktx).expect("compile");
        assert_eq!(image.format, TextureFormat::Rgba8);
        assert_eq!(image.mips.len(), 1);
        // Solid red block -> first texel is opaque red.
        assert_eq!(&image.mips[0].data[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn rejects_cubemap() {
        // faceCount 6 at header offset 36.
        let levels = vec![(4, 4, bc1_red_block())];
        let mut ktx = build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels);
        ktx[36..40].copy_from_slice(&6u32.to_le_bytes());
        let err = compile_ktx2(&ktx).unwrap_err();
        assert!(err.contains("cubemap"), "got: {err}");
        assert!(
            err.contains("2D"),
            "error should name what is supported: {err}"
        );
    }

    #[test]
    fn rejects_texture_array() {
        let levels = vec![(4, 4, bc1_red_block())];
        let mut ktx = build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels);
        ktx[32..36].copy_from_slice(&4u32.to_le_bytes()); // layerCount
        let err = compile_ktx2(&ktx).unwrap_err();
        assert!(err.contains("array"), "got: {err}");
    }

    #[test]
    fn rejects_unsupported_vkformat() {
        // BC2 (135) is not in the supported set.
        let levels = vec![(4, 4, vec![0u8; 16])];
        let ktx = build_ktx2(135, 0, 4, 4, &levels);
        let err = compile_ktx2(&ktx).unwrap_err();
        assert!(err.contains("unsupported KTX2 vkFormat"), "got: {err}");
        assert!(
            err.contains("BC7"),
            "error should name the supported set: {err}"
        );
    }

    #[test]
    fn rejects_garbage() {
        let err = compile_ktx2(b"not a ktx2 file at all").unwrap_err();
        assert!(err.contains("KTX2"), "got: {err}");
    }

    #[test]
    fn mip_dims_halve_and_floor() {
        assert_eq!(mip_dims(256, 256, 0), (256, 256));
        assert_eq!(mip_dims(256, 256, 8), (1, 1));
        assert_eq!(mip_dims(256, 256, 20), (1, 1));
        assert_eq!(mip_dims(4, 2, 1), (2, 1));
    }

    // Drives the Basis transcoder against real fixtures (generated with the
    // `basisu` CLI from private/assets/images/ktx2_test/). Ignored by default
    // because it needs the checked-in fixtures; run with:
    //   cargo test -p concinnity-cook ktx2_fixture -- --ignored
    fn fixture(name: &str) -> Vec<u8> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../private/assets/images/ktx2_test/"
        );
        std::fs::read(std::path::Path::new(path).join(name))
            .unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
    }

    // Peak signal-to-noise ratio (dB) between two equally sized RGBA8 buffers,
    // averaged over the RGB channels. Higher is closer to the reference.
    fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut sum_sq = 0.0f64;
        let mut count = 0.0f64;
        for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            for c in 0..3 {
                let d = pa[c] as f64 - pb[c] as f64;
                sum_sq += d * d;
                count += 1.0;
            }
        }
        let mse = sum_sq / count;
        if mse <= f64::EPSILON {
            return f64::INFINITY;
        }
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    // Tripwire for `basisu` output drift: decodes the checked-in ETC1S fixture
    // through the shipping path (transcode to BC1, then CPU-decode the base mip)
    // and holds the reconstruction quality against the source PNG above a floor.
    // A `basisu` bump that regresses ETC1S quality trips this. Run with:
    //   cargo test -p concinnity-cook ktx2_psnr -- --ignored
    #[test]
    #[ignore = "needs the local KTX2 fixtures under private/assets/images/ktx2_test"]
    fn ktx2_psnr_etc1s_stays_above_quality_floor() {
        let png = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../private/assets/images/ktx2_test/wood_256.png"
        );
        let (sw, sh, source) = crate::texture::decode_source(png, 0).expect("decode source png");
        assert_eq!((sw, sh), (256, 256));

        let image = compile_ktx2(&fixture("wood_256_etc1s.ktx2")).expect("etc1s compile");
        assert_eq!(image.format, TextureFormat::Bc1);
        let decoded = crate::bcn::decode_bc1(&image.mips[0].data, 256, 256).expect("bc1 decode");

        let psnr = psnr_rgb(&source, &decoded);
        assert!(
            psnr >= 24.0,
            "ETC1S->BC1 reconstruction PSNR {psnr:.2} dB dropped below the 24 dB floor; \
             a basisu bump may have regressed quality"
        );
    }

    // Tripwire for `basisu` UASTC->BC7 output drift. BC7 has no CPU decoder here,
    // so this pins the transcoded base-mip bytes: the pinned `basisu = =0.1.0`
    // makes the output deterministic, and a version bump that shifts the bytes
    // trips the recorded hash, prompting a review before rebaselining.
    #[test]
    #[ignore = "needs the local KTX2 fixtures under private/assets/images/ktx2_test"]
    fn ktx2_uastc_bc7_bytes_are_pinned() {
        let image = compile_ktx2(&fixture("wood_256_uastc.ktx2")).expect("uastc compile");
        assert_eq!(image.format, TextureFormat::Bc7);
        // Deterministic across runs.
        let again = compile_ktx2(&fixture("wood_256_uastc.ktx2")).expect("uastc compile 2");
        assert_eq!(image.mips[0].data, again.mips[0].data);
        assert_eq!(
            fnv1a(&image.mips[0].data),
            1395555890415016196,
            "UASTC->BC7 base-mip bytes changed; a basisu bump likely shifted the transcode output"
        );
    }

    #[test]
    #[ignore = "needs the local KTX2 fixtures under private/assets/images/ktx2_test"]
    fn ktx2_fixture_etc1s_transcodes_to_bc1_with_full_chain() {
        let image = compile_ktx2(&fixture("wood_256_etc1s.ktx2")).expect("etc1s compile");
        assert_eq!(image.format, TextureFormat::Bc1);
        assert_eq!((image.width(), image.height()), (256, 256));
        // 256 -> 1 is 9 mip levels.
        assert_eq!(image.mips.len(), 9);
        for (level, mip) in image.mips.iter().enumerate() {
            let (mw, mh) = mip_dims(256, 256, level as u32);
            assert_eq!((mip.width, mip.height), (mw, mh));
            assert_eq!(mip.data.len(), TextureFormat::Bc1.mip_byte_len(mw, mh));
        }
    }

    #[test]
    #[ignore = "needs the local KTX2 fixtures under private/assets/images/ktx2_test"]
    fn ktx2_fixture_uastc_transcodes_to_bc7_with_full_chain() {
        let image = compile_ktx2(&fixture("wood_256_uastc.ktx2")).expect("uastc compile");
        assert_eq!(image.format, TextureFormat::Bc7);
        assert_eq!((image.width(), image.height()), (256, 256));
        assert_eq!(image.mips.len(), 9);
        assert_eq!(
            image.mips[0].data.len(),
            TextureFormat::Bc7.mip_byte_len(256, 256)
        );
    }
}
