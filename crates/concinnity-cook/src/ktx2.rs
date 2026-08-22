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
use concinnity_cpu::build::texture::{TextureFormat, TextureImage, TextureMip};
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
pub(crate) fn compile_ktx2(bytes: &[u8]) -> Result<TextureImage, String> {
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

    // A chainless source needs runtime mip generation to stay sharp at
    // distance, so it decodes to RGBA8. BC7 has no CPU decoder, so it stays a
    // single compressed level.
    let decodes = level_blocks.len() == 1 && tex_format != TextureFormat::Bc7;
    if decodes {
        let rgba = crate::bcn::decode(tex_format, &level_blocks[0], width, height)?;
        tracing::info!(
            "KTX2 {:?} texture decoded to RGBA8 ({} stored level(s))",
            tex_format,
            level_blocks.len()
        );
        return Ok(TextureImage::rgba8(width, height, rgba));
    }

    let mut mips = Vec::with_capacity(level_blocks.len());
    for (level, blocks) in level_blocks.into_iter().enumerate() {
        let (mw, mh) = mip_dims(width, height, level as u32);
        let expected = tex_format.mip_byte_len(mw, mh)?;
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
        let expected = tex_format.mip_byte_len(info.width, info.height)?;
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
pub(crate) fn decode_ktx2_rgba8(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let image = compile_ktx2(bytes)?;
    let (w, h) = (image.width(), image.height());
    match image.format {
        TextureFormat::Rgba8 => image.into_rgba8(),
        TextureFormat::Bc7 => Err("hot-reload preview of a BC7 KTX2 is not supported".to_string()),
        other => Ok((w, h, crate::bcn::decode(other, &image.mips[0].data, w, h)?)),
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

// Mip dimensions for level `level`: each axis halves, floored, minimum 1 (the
// GPU mip convention). Matches how block-compressed containers store their
// chains.
fn mip_dims(width: u32, height: u32, level: u32) -> (u32, u32) {
    ((width >> level).max(1), (height >> level).max(1))
}

// Synthetic KTX2 containers for this crate's tests: `texture.rs` builds them to
// exercise the KTX2 branch of the file-backed texture compiler.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::VK_BC1_RGBA_UNORM;

    // Assemble a KTX2 container around already-laid-out level data. `levels`
    // are (width, height, bytes), level 0 first; `dfd` is the whole Data Format
    // Descriptor, starting with its own total-size field.
    pub(crate) fn build_ktx2_with_dfd(
        vk_format: u32,
        supercompression: u32,
        base_w: u32,
        base_h: u32,
        levels: &[(u32, u32, Vec<u8>)],
        dfd: &[u8],
    ) -> Vec<u8> {
        const HEADER_LEN: usize = 80;
        const LEVEL_INDEX_LEN: usize = 24;
        let level_index_bytes = levels.len() * LEVEL_INDEX_LEN;
        let dfd_offset = HEADER_LEN + level_index_bytes;
        let dfd_len = dfd.len() as u32;
        let mut data_offset = dfd_offset + dfd.len();

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
        out[dfd_offset..dfd_offset + dfd.len()].copy_from_slice(dfd);
        // Level data
        for (i, (_, _, bytes)) in levels.iter().enumerate() {
            let off = level_offsets[i];
            out[off..off + bytes.len()].copy_from_slice(bytes);
        }
        out
    }

    // A container whose DFD is nothing but its total-size field: enough for a
    // real `vkFormat`, which never consults the descriptor.
    pub(crate) fn build_ktx2(
        vk_format: u32,
        supercompression: u32,
        base_w: u32,
        base_h: u32,
        levels: &[(u32, u32, Vec<u8>)],
    ) -> Vec<u8> {
        build_ktx2_with_dfd(
            vk_format,
            supercompression,
            base_w,
            base_h,
            levels,
            &4u32.to_le_bytes(),
        )
    }

    // The 44-byte Data Format Descriptor a Basis Universal container carries:
    // `color_model` names the source codec and `channel0` the sample's channel
    // id, which is what marks the payload as carrying alpha.
    pub(crate) fn basis_dfd(
        color_model: u32,
        block_w: u32,
        block_h: u32,
        channel0: u32,
    ) -> Vec<u8> {
        let mut dfd = vec![0u8; 44];
        let put = |d: &mut [u8], at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut dfd, 0, 44); // dfdTotalSize
        put(&mut dfd, 8, 2 | (40 << 16)); // versionNumber | descriptorBlockSize
        put(&mut dfd, 12, color_model | (1 << 16)); // colorModel | linear transfer
        put(&mut dfd, 16, (block_w - 1) | ((block_h - 1) << 8));
        put(&mut dfd, 28, channel0 << 24);
        dfd
    }

    // A solid-red BC1 block (color0 = 565 red, indices 0).
    pub(crate) fn bc1_red_block() -> Vec<u8> {
        vec![0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    }

    // A 4x4 BC1 container with a 2x2 second level, so the compiler takes the
    // block-passthrough path rather than the RGBA8 fallback.
    pub(crate) fn bc1_mip_chain_ktx2() -> Vec<u8> {
        let levels = vec![(4, 4, bc1_red_block()), (2, 2, bc1_red_block())];
        build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels)
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{
        basis_dfd, bc1_mip_chain_ktx2, bc1_red_block, build_ktx2, build_ktx2_with_dfd,
    };
    use super::*;

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

    #[test]
    fn rejects_a_3d_texture() {
        let levels = vec![(4, 4, bc1_red_block())];
        let mut ktx = build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels);
        ktx[28..32].copy_from_slice(&4u32.to_le_bytes()); // pixelDepth
        let err = compile_ktx2(&ktx).unwrap_err();
        assert_eq!(
            err,
            "KTX2 3D textures are not supported (depth 4); only 2D textures are"
        );
    }

    #[test]
    fn rejects_a_zero_dimension() {
        let levels = vec![(4, 4, bc1_red_block())];
        let mut ktx = build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels);
        ktx[24..28].copy_from_slice(&0u32.to_le_bytes()); // pixelHeight
        let err = compile_ktx2(&ktx).unwrap_err();
        assert_eq!(err, "KTX2 has a zero dimension 4x0");
    }

    // Every non-BC1 block format the compiler accepts, checked through a two
    // level chain so the blocks pass through rather than decoding to RGBA8.
    #[test]
    fn maps_the_supported_bcn_vkformats() {
        for (vk, want, block_bytes) in [
            (VK_BC1_RGB_UNORM, TextureFormat::Bc1, 8),
            (VK_BC1_RGB_SRGB, TextureFormat::Bc1, 8),
            (VK_BC1_RGBA_SRGB, TextureFormat::Bc1, 8),
            (VK_BC3_UNORM, TextureFormat::Bc3, 16),
            (VK_BC3_SRGB, TextureFormat::Bc3, 16),
            (VK_BC7_UNORM, TextureFormat::Bc7, 16),
            (VK_BC7_SRGB, TextureFormat::Bc7, 16),
        ] {
            let levels = vec![
                (4, 4, vec![0xAAu8; block_bytes]),
                (2, 2, vec![0xBBu8; block_bytes]),
            ];
            let image = compile_ktx2(&build_ktx2(vk, 0, 4, 4, &levels)).expect("compile");
            assert_eq!(image.format, want, "vkFormat {vk}");
            assert_eq!(image.mips.len(), 2, "vkFormat {vk}");
            assert_eq!(image.mips[0].data, vec![0xAAu8; block_bytes]);
            assert_eq!((image.mips[1].width, image.mips[1].height), (2, 2));
        }
    }

    // BC5 carries no blue channel, but the shaders reconstruct Z from X and Y,
    // so a two-channel chain uploads as blocks at a quarter of the RGBA8 cost.
    #[test]
    fn a_bc5_chain_passes_through_as_blocks() {
        let block = [128u8, 128, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 0, 0];
        let levels = vec![(4, 4, block.to_vec()), (2, 2, block.to_vec())];
        let image = compile_ktx2(&build_ktx2(VK_BC5_UNORM, 0, 4, 4, &levels)).expect("compile");

        assert_eq!(image.format, TextureFormat::Bc5);
        assert_eq!(image.mips.len(), 2);
        assert_eq!(image.mips[0].data, block.to_vec());
        assert_eq!((image.mips[1].width, image.mips[1].height), (2, 2));
    }

    #[test]
    fn a_single_level_bc7_stays_compressed() {
        // BC7 has no CPU decoder, so the no-mip-chain fallback cannot apply.
        let levels = vec![(4, 4, vec![0u8; 16])];
        let image = compile_ktx2(&build_ktx2(VK_BC7_UNORM, 0, 4, 4, &levels)).expect("compile");
        assert_eq!(image.format, TextureFormat::Bc7);
        assert_eq!(image.mips.len(), 1);
    }

    #[test]
    fn a_single_level_bc3_falls_back_to_rgba8() {
        // Alpha endpoints 200/10 with index 0, then an opaque white colour block.
        let mut block = vec![0u8; 16];
        block[0] = 200;
        block[1] = 10;
        block[8] = 0xFF;
        block[9] = 0xFF;
        let image =
            compile_ktx2(&build_ktx2(VK_BC3_UNORM, 0, 4, 4, &[(4, 4, block)])).expect("compile");
        assert_eq!(image.format, TextureFormat::Rgba8);
        assert!(
            image.mips[0]
                .data
                .chunks_exact(4)
                .all(|c| c == [255, 255, 255, 200])
        );
    }

    #[test]
    fn a_single_level_bc5_falls_back_to_rgba8() {
        // Red and green both mid-grey: the flat normal.
        let mut block = vec![0u8; 16];
        block[0] = 128;
        block[1] = 128;
        block[8] = 128;
        block[9] = 128;
        let image =
            compile_ktx2(&build_ktx2(VK_BC5_UNORM, 0, 4, 4, &[(4, 4, block)])).expect("compile");
        assert_eq!(image.format, TextureFormat::Rgba8);
        assert_eq!(&image.mips[0].data[0..2], &[128, 128]);
        assert!(
            image.mips[0].data[2] >= 253,
            "expected near-255 blue, got {}",
            image.mips[0].data[2]
        );
    }

    #[test]
    fn rejects_a_single_level_too_short_for_the_rgba8_fallback() {
        // 8x8 BC1 needs four blocks; the CPU decoder refuses one.
        let err = compile_ktx2(&build_ktx2(
            VK_BC1_RGBA_UNORM,
            0,
            8,
            8,
            &[(8, 8, bc1_red_block())],
        ))
        .unwrap_err();
        assert!(
            err.starts_with("block-compressed data too short"),
            "got: {err}"
        );
    }

    #[test]
    fn decode_rgba8_surfaces_a_compile_error() {
        let err = decode_ktx2_rgba8(b"not a ktx2 file at all").unwrap_err();
        assert!(err.starts_with("not a valid KTX2 container"), "got: {err}");
    }

    #[test]
    fn rejects_a_level_shorter_than_its_block_size() {
        // 8x8 BC1 needs four blocks (32 bytes); only one is supplied.
        let levels = vec![(8, 8, bc1_red_block()), (4, 4, bc1_red_block())];
        let err = compile_ktx2(&build_ktx2(VK_BC1_RGBA_UNORM, 0, 8, 8, &levels)).unwrap_err();
        assert_eq!(err, "KTX2 level 0 (8x8 Bc1) is 8 bytes, needs 32");
    }

    #[test]
    fn trims_a_level_padded_beyond_its_block_size() {
        // Level 0 carries eight surplus bytes of container padding.
        let mut padded = bc1_red_block().repeat(4);
        padded.extend_from_slice(&[0xFF; 8]);
        let levels = vec![(8, 8, padded), (4, 4, bc1_red_block())];
        let image =
            compile_ktx2(&build_ktx2(VK_BC1_RGBA_UNORM, 0, 8, 8, &levels)).expect("compile");
        assert_eq!(image.mips[0].data, bc1_red_block().repeat(4));
    }

    #[test]
    fn rejects_an_unsupported_supercompression_scheme() {
        // Scheme 1 is BasisLZ, which only reaches a container with no vkFormat.
        let levels = vec![(4, 4, bc1_red_block()), (2, 2, bc1_red_block())];
        let err = compile_ktx2(&build_ktx2(VK_BC1_RGBA_UNORM, 1, 4, 4, &levels)).unwrap_err();
        assert!(
            err.starts_with("KTX2 level 0: unsupported KTX2 supercompression"),
            "got: {err}"
        );
        assert!(err.contains("use none or zstd"), "got: {err}");
    }

    #[test]
    fn rejects_a_zstd_level_whose_body_is_cut_short() {
        let mut compressed = ruzstd::encoding::compress_to_vec(
            &bc1_red_block()[..],
            ruzstd::encoding::CompressionLevel::Uncompressed,
        );
        // The frame header survives; the compressed body does not.
        compressed.truncate(compressed.len() - 6);
        let levels = vec![(4, 4, compressed.clone()), (2, 2, compressed)];
        let err = compile_ktx2(&build_ktx2(VK_BC1_RGBA_UNORM, 2, 4, 4, &levels)).unwrap_err();
        assert!(
            err.starts_with("KTX2 level 0: zstd inflate failed"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_a_corrupt_zstd_level() {
        let levels = vec![(4, 4, vec![0xFFu8; 16]), (2, 2, vec![0xFFu8; 16])];
        let err = compile_ktx2(&build_ktx2(VK_BC1_RGBA_UNORM, 2, 4, 4, &levels)).unwrap_err();
        assert!(
            err.starts_with("KTX2 level 0: zstd frame header invalid"),
            "got: {err}"
        );
    }

    // Hot-reload preview decode

    #[test]
    fn decode_rgba8_returns_the_fallback_pixels_verbatim() {
        // A single-level BC1 already decodes to RGBA8 during compile.
        let levels = vec![(4, 4, bc1_red_block())];
        let ktx = build_ktx2(VK_BC1_RGBA_UNORM, 0, 4, 4, &levels);
        let (w, h, px) = decode_ktx2_rgba8(&ktx).expect("decode");
        assert_eq!((w, h), (4, 4));
        assert_eq!(px.len(), 4 * 4 * 4);
        assert!(px.chunks_exact(4).all(|c| c == [255, 0, 0, 255]));
    }

    #[test]
    fn decode_rgba8_expands_a_bc1_mip_chain_base_level() {
        let (w, h, px) = decode_ktx2_rgba8(&bc1_mip_chain_ktx2()).expect("decode");
        assert_eq!((w, h), (4, 4));
        assert!(px.chunks_exact(4).all(|c| c == [255, 0, 0, 255]));
    }

    #[test]
    fn decode_rgba8_expands_a_bc3_mip_chain_base_level() {
        // Alpha endpoints 200/10 with index 0, then an opaque white colour block.
        let mut block = vec![0u8; 16];
        block[0] = 200;
        block[1] = 10;
        block[8] = 0xFF;
        block[9] = 0xFF;
        let levels = vec![(4, 4, block), (2, 2, vec![0u8; 16])];
        let ktx = build_ktx2(VK_BC3_UNORM, 0, 4, 4, &levels);
        let (_, _, px) = decode_ktx2_rgba8(&ktx).expect("decode");
        assert!(px.chunks_exact(4).all(|c| c == [255, 255, 255, 200]));
    }

    #[test]
    fn decode_rgba8_expands_a_bc5_mip_chain_base_level() {
        // Red and green both mid-grey: the flat normal.
        let mut block = vec![0u8; 16];
        block[0] = 128;
        block[1] = 128;
        block[8] = 128;
        block[9] = 128;
        let levels = vec![(4, 4, block), (2, 2, vec![0u8; 16])];
        let ktx = build_ktx2(VK_BC5_UNORM, 0, 4, 4, &levels);
        let (_, _, px) = decode_ktx2_rgba8(&ktx).expect("decode");
        assert_eq!(&px[0..2], &[128, 128]);
        assert!(px[2] >= 253, "expected near-255 blue, got {}", px[2]);
    }

    #[test]
    fn decode_rgba8_refuses_bc7() {
        let levels = vec![(4, 4, vec![0u8; 16])];
        let ktx = build_ktx2(VK_BC7_UNORM, 0, 4, 4, &levels);
        let err = decode_ktx2_rgba8(&ktx).unwrap_err();
        assert_eq!(err, "hot-reload preview of a BC7 KTX2 is not supported");
    }

    #[test]
    fn decode_blocks_to_rgba8_has_no_bc7_decoder() {
        let err = crate::bcn::decode(TextureFormat::Bc7, &[0u8; 16], 4, 4).unwrap_err();
        assert_eq!(err, "no CPU decoder for Bc7");
    }

    // Basis Universal
    //
    // Real ETC1S fixtures need the BasisLZ codebooks, which cannot be
    // synthesised here; UASTC LDR blocks can, so the transcode path is driven
    // through a hand-built container.

    // DFD color model 166 (UASTC LDR) with an RGBA sample channel.
    const DF_MODEL_UASTC: u32 = 166;
    const DF_CHANNEL_UASTC_RGBA: u32 = 3;

    // Write `bits` low bits of `value` into `block` at `bit_ofs`, LSB first.
    fn put_bits(block: &mut [u8; 16], bit_ofs: &mut usize, value: u32, bits: usize) {
        for i in 0..bits {
            if (value >> i) & 1 != 0 {
                block[(*bit_ofs + i) / 8] |= 1 << ((*bit_ofs + i) % 8);
            }
        }
        *bit_ofs += bits;
    }

    // A UASTC LDR solid-colour block: mode 8 (Huffman code 0x17, five bits)
    // followed by the eight-bit R, G, B and A channels.
    fn uastc_solid_block(rgba: [u8; 4]) -> Vec<u8> {
        let mut block = [0u8; 16];
        let mut bit_ofs = 0usize;
        put_bits(&mut block, &mut bit_ofs, 0x17, 5);
        for channel in rgba {
            put_bits(&mut block, &mut bit_ofs, channel as u32, 8);
        }
        block.to_vec()
    }

    fn uastc_ktx2(base_w: u32, base_h: u32, levels: &[(u32, u32, Vec<u8>)]) -> Vec<u8> {
        build_ktx2_with_dfd(
            0,
            0,
            base_w,
            base_h,
            levels,
            &basis_dfd(DF_MODEL_UASTC, 4, 4, DF_CHANNEL_UASTC_RGBA),
        )
    }

    #[test]
    fn rejects_an_unparseable_basis_payload() {
        // vkFormat UNDEFINED routes to the Basis path, but the stub DFD is not
        // one the transcoder recognises.
        let levels = vec![(4, 4, vec![0u8; 16])];
        let err = compile_ktx2(&build_ktx2(0, 0, 4, 4, &levels)).unwrap_err();
        assert!(err.starts_with("KTX2 Basis parse failed"), "got: {err}");
    }

    // Point a one-level container at a 12-byte supercompression global-data
    // block: the per-image slice descriptor the intermediate codecs carry, and
    // which they refuse to open without.
    fn with_intermediate_global_data(mut ktx: Vec<u8>) -> Vec<u8> {
        const LEVEL_INDEX: usize = 80;
        // The intermediate schemes record no per-level uncompressed size.
        ktx[LEVEL_INDEX + 16..LEVEL_INDEX + 24].copy_from_slice(&0u64.to_le_bytes());
        let sgd = ktx.len() as u64;
        ktx[64..72].copy_from_slice(&sgd.to_le_bytes());
        ktx[72..80].copy_from_slice(&12u64.to_le_bytes());
        ktx.extend_from_slice(&0u32.to_le_bytes()); // slice offset
        ktx.extend_from_slice(&16u32.to_le_bytes()); // slice length
        ktx.extend_from_slice(&0u32.to_le_bytes()); // profile
        ktx.push(0); // the reader wants a byte past the end of every section
        ktx
    }

    #[test]
    fn rejects_a_basis_codec_the_compiler_does_not_transcode() {
        // DFD color model 168 under supercompression 4 is the UASTC HDR 6x6
        // intermediate stream, which has no LDR transcode target here.
        const DF_MODEL_UASTC_HDR_6X6: u32 = 168;
        const SS_UASTC_HDR_6X6: u32 = 4;
        let ktx = build_ktx2_with_dfd(
            0,
            SS_UASTC_HDR_6X6,
            4,
            4,
            &[(4, 4, vec![0u8; 16])],
            &basis_dfd(DF_MODEL_UASTC_HDR_6X6, 6, 6, DF_CHANNEL_UASTC_RGBA),
        );
        let err = compile_ktx2(&with_intermediate_global_data(ktx)).unwrap_err();
        assert!(
            err.contains("unsupported Basis source format"),
            "got: {err}"
        );
        assert!(err.contains("only ETC1S and UASTC LDR"), "got: {err}");
    }

    #[test]
    fn transcodes_a_uastc_mip_chain_to_bc7() {
        let levels = vec![
            (4, 4, uastc_solid_block([200, 60, 30, 255])),
            (2, 2, uastc_solid_block([200, 60, 30, 255])),
        ];
        let image = compile_ktx2(&uastc_ktx2(4, 4, &levels)).expect("compile");
        assert_eq!(image.format, TextureFormat::Bc7);
        assert_eq!(image.mips.len(), 2);
        assert_eq!((image.width(), image.height()), (4, 4));
        assert_eq!(
            image.mips[0].data.len(),
            TextureFormat::Bc7.mip_byte_len(4, 4).unwrap()
        );
        assert_eq!((image.mips[1].width, image.mips[1].height), (2, 2));
        assert_eq!(
            image.mips[1].data.len(),
            TextureFormat::Bc7.mip_byte_len(2, 2).unwrap()
        );
    }

    // A block whose first byte decodes to mode 19, one past the last real
    // UASTC mode, so it fails to unpack and every transcode target rejects it.
    fn corrupt_uastc_block() -> Vec<u8> {
        let mut block = vec![0u8; 16];
        block[0] = 69;
        block
    }

    #[test]
    fn reports_a_uastc_block_the_bc7_transcode_rejects() {
        let levels = vec![
            (4, 4, corrupt_uastc_block()),
            (2, 2, uastc_solid_block([1, 2, 3, 255])),
        ];
        let err = compile_ktx2(&uastc_ktx2(4, 4, &levels)).unwrap_err();
        assert!(
            err.starts_with("KTX2 Basis level 0 transcode to Bc7Rgba failed"),
            "got: {err}"
        );
    }

    #[test]
    fn reports_a_uastc_block_the_rgba_transcode_rejects() {
        let levels = vec![(4, 4, corrupt_uastc_block())];
        let err = compile_ktx2(&uastc_ktx2(4, 4, &levels)).unwrap_err();
        assert!(
            err.starts_with("KTX2 Basis RGBA transcode failed"),
            "got: {err}"
        );
    }

    #[test]
    fn a_single_level_uastc_transcodes_to_rgba8() {
        let colour = [200u8, 60, 30, 255];
        let levels = vec![(4, 4, uastc_solid_block(colour))];
        let image = compile_ktx2(&uastc_ktx2(4, 4, &levels)).expect("compile");
        assert_eq!(image.format, TextureFormat::Rgba8);
        assert_eq!((image.width(), image.height()), (4, 4));
        assert!(
            image.mips[0].data.chunks_exact(4).all(|c| c == colour),
            "solid UASTC block should transcode to a solid RGBA8 image"
        );
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
            assert_eq!(
                mip.data.len(),
                TextureFormat::Bc1.mip_byte_len(mw, mh).unwrap()
            );
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
            TextureFormat::Bc7.mip_byte_len(256, 256).unwrap()
        );
    }
}
