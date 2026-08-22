// src/build/dds.rs
//
// Decodes a DDS container. Only the legacy (non-DX10) header and the three
// fourCC formats the asset set uses are handled: DXT1 (BC1), DXT5 (BC3), and
// ATI2 (BC5). A DX10 extended header is rejected with a clear message rather
// than silently misread.
//
// `decode_dds` returns the top mip decoded to RGBA8 (the legacy path).
// `decode_dds_blocks` reads the container's block-compressed mip chain for
// passthrough into the compressed texture payload.

use concinnity_cpu::build::texture::{MAX_MIP_LEVELS, TextureFormat, TextureMip};
use concinnity_cpu::decode::ByteReader;

const MAGIC: &[u8; 4] = b"DDS ";
const HEADER_LEN: usize = 124;
const PIXELDATA_OFFSET: usize = 4 + HEADER_LEN;

// Block-compressed mip chain read straight from a DDS file: the fourCC-derived
// format plus one entry per stored mip level, level 0 first.
pub(crate) struct DdsBlocks {
    pub format: TextureFormat,
    pub(crate) mips: Vec<TextureMip>,
}

// Read a legacy DDS's block data and mip chain without decoding. Returns the
// BCn format and every stored mip level's blocks. A DX10 header or an
// unsupported fourCC is rejected the same way `decode_dds` rejects them.
pub(crate) fn decode_dds_blocks(bytes: &[u8]) -> Result<DdsBlocks, String> {
    let DdsHeader {
        width,
        height,
        mip_count,
        format,
    } = parse_dds_header(bytes)?;

    let mut r = ByteReader::new(bytes, "DDS");
    r.seek(PIXELDATA_OFFSET)?;
    let mut mips = Vec::with_capacity(mip_count as usize);
    for level in 0..mip_count {
        let mw = (width >> level).max(1);
        let mh = (height >> level).max(1);
        let len = format.mip_byte_len(mw, mh)?;
        let at = r.position();
        let data = r.take(len).map_err(|_| {
            format!(
                "DDS mip {} ({}x{}) needs {} bytes at offset {}, file has {}",
                level,
                mw,
                mh,
                len,
                at,
                bytes.len()
            )
        })?;
        mips.push(TextureMip {
            width: mw,
            height: mh,
            data: data.to_vec(),
        });
    }
    Ok(DdsBlocks { format, mips })
}

// Decode the top mip of a DDS file into (width, height, RGBA8 pixels).
pub(crate) fn decode_dds(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let DdsHeader {
        width,
        height,
        format,
        ..
    } = parse_dds_header(bytes)?;
    let pixels = crate::bcn::decode(format, &bytes[PIXELDATA_OFFSET..], width, height)?;
    Ok((width, height, pixels))
}

// The fixed preamble of a legacy DDS: dimensions, stored mip count, and the
// block format its fourCC selects.
struct DdsHeader {
    width: u32,
    height: u32,
    mip_count: u32,
    format: TextureFormat,
}

fn parse_dds_header(bytes: &[u8]) -> Result<DdsHeader, String> {
    if bytes.len() < PIXELDATA_OFFSET {
        return Err(format!("DDS too short: {} bytes", bytes.len()));
    }
    if &bytes[0..4] != MAGIC {
        return Err("not a DDS file (bad magic)".to_string());
    }
    let height = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let width = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    // The count is file-supplied and drives a reservation, so it is bounded
    // here rather than trusted. Zero is common and means a single level.
    let mip_count = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]).max(1);
    if mip_count as usize > MAX_MIP_LEVELS {
        return Err(format!(
            "DDS declares {} mip levels (expected at most {})",
            mip_count, MAX_MIP_LEVELS
        ));
    }
    if width == 0 || height == 0 {
        return Err(format!("DDS has zero dimension {}x{}", width, height));
    }
    let format = match &bytes[84..88] {
        b"DXT1" => TextureFormat::Bc1,
        b"DXT5" => TextureFormat::Bc3,
        b"ATI2" => TextureFormat::Bc5,
        b"DX10" => {
            return Err("DDS uses a DX10 extended header, which is not supported; \
                 re-export as DXT1/DXT5/ATI2"
                .to_string());
        }
        other => {
            return Err(format!(
                "unsupported DDS fourCC {:?}; only DXT1, DXT5, and ATI2 are handled",
                String::from_utf8_lossy(other)
            ));
        }
    };
    Ok(DdsHeader {
        width,
        height,
        mip_count,
        format,
    })
}

// Synthetic legacy DDS containers for this crate's tests: `texture.rs` builds
// them to exercise the DDS branch of the file-backed texture compiler.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::{MAGIC, PIXELDATA_OFFSET};

    // Build a minimal legacy DDS around concatenated block levels.
    pub(crate) fn wrap_dds_mips(
        fourcc: &[u8; 4],
        width: u32,
        height: u32,
        mip_count: u32,
        data: &[u8],
    ) -> Vec<u8> {
        let mut v = vec![0u8; PIXELDATA_OFFSET];
        v[0..4].copy_from_slice(MAGIC);
        v[12..16].copy_from_slice(&height.to_le_bytes());
        v[16..20].copy_from_slice(&width.to_le_bytes());
        v[28..32].copy_from_slice(&mip_count.to_le_bytes());
        v[84..88].copy_from_slice(fourcc);
        v.extend_from_slice(data);
        v
    }

    // A solid-red BC1 block (color0 = 565 red, indices 0).
    pub(crate) fn bc1_red_block() -> [u8; 8] {
        [0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{bc1_red_block, wrap_dds_mips};
    use super::*;

    // A single-level DDS: mip_count 0 is clamped to one stored level.
    fn wrap_dds(fourcc: &[u8; 4], width: u32, height: u32, block: &[u8]) -> Vec<u8> {
        wrap_dds_mips(fourcc, width, height, 0, block)
    }

    // `DdsBlocks` is not `Debug`, so unwrap the error side by hand.
    fn blocks_err(bytes: &[u8]) -> String {
        decode_dds_blocks(bytes)
            .err()
            .expect("expected decode_dds_blocks to fail")
    }

    // A mip count near u32::MAX must be rejected on the declared value rather
    // than reserving one `TextureMip` per claimed level.
    #[test]
    fn rejects_an_absurd_mip_count() {
        let bytes = wrap_dds_mips(b"DXT1", 4, 4, u32::MAX, &bc1_red_block());
        let err = blocks_err(&bytes);
        assert!(err.contains("mip levels"), "{}", err);
    }

    #[test]
    fn rejects_a_mip_count_past_the_dimension_limit() {
        let bytes = wrap_dds_mips(b"DXT1", 4, 4, 33, &bc1_red_block());
        assert!(blocks_err(&bytes).contains("mip levels"));
    }

    // At maximal dimensions a 16-byte block format needs 2^64 bytes exactly,
    // one past what `usize` holds. The footprint has to be reported rather
    // than wrapped to zero and compared against the file length.
    #[test]
    fn rejects_dimensions_that_overflow_the_block_footprint() {
        let bytes = wrap_dds_mips(b"DXT5", u32::MAX, u32::MAX, 1, &[0u8; 16]);
        let err = blocks_err(&bytes);
        assert!(err.contains("overflow"), "{}", err);
    }

    // The same dimensions with 8-byte blocks stay inside `usize`, so this one
    // is a plain shortfall and must name the size it wanted.
    #[test]
    fn reports_the_required_size_when_the_footprint_still_fits() {
        let bytes = wrap_dds_mips(b"DXT1", u32::MAX, u32::MAX, 1, &bc1_red_block());
        let err = blocks_err(&bytes);
        assert!(err.contains("needs"), "{}", err);
    }

    #[test]
    fn rejects_a_mip_chain_truncated_partway() {
        // Declares four levels of an 8x8 BC1 image but stores only level 0.
        let bytes = wrap_dds_mips(b"DXT1", 8, 8, 4, &[0u8; 32]);
        let err = blocks_err(&bytes);
        assert!(err.contains("needs"), "{}", err);
    }

    #[test]
    fn decodes_dxt1() {
        let block = [0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // solid red
        let dds = wrap_dds(b"DXT1", 4, 4, &block);
        let (w, h, px) = decode_dds(&dds).unwrap();
        assert_eq!((w, h), (4, 4));
        assert_eq!(&px[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn rejects_dx10_header() {
        let dds = wrap_dds(b"DX10", 4, 4, &[0u8; 16]);
        let err = decode_dds(&dds).unwrap_err();
        assert!(err.contains("DX10"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_fourcc() {
        let dds = wrap_dds(b"DXT3", 4, 4, &[0u8; 16]);
        assert!(decode_dds(&dds).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut dds = wrap_dds(b"DXT1", 4, 4, &[0u8; 8]);
        dds[0] = b'X';
        assert!(decode_dds(&dds).is_err());
    }

    #[test]
    fn reads_bc1_mip_chain_blocks() {
        // 8x8 BC1: mip0 = 4 blocks (32B), mip1 (4x4) = 1 block (8B),
        // mip2 (2x2) = 1 block, mip3 (1x1) = 1 block.
        let block = bc1_red_block();
        let mut data = Vec::new();
        data.extend_from_slice(&block.repeat(4)); // mip0
        data.extend_from_slice(&block); // mip1
        data.extend_from_slice(&block); // mip2
        data.extend_from_slice(&block); // mip3
        let dds = wrap_dds_mips(b"DXT1", 8, 8, 4, &data);
        let blocks = decode_dds_blocks(&dds).expect("blocks");
        assert_eq!(blocks.format, TextureFormat::Bc1);
        assert_eq!(blocks.mips.len(), 4);
        assert_eq!((blocks.mips[0].width, blocks.mips[0].height), (8, 8));
        assert_eq!(blocks.mips[0].data.len(), 32);
        assert_eq!((blocks.mips[3].width, blocks.mips[3].height), (1, 1));
        assert_eq!(blocks.mips[3].data.len(), 8);
    }

    #[test]
    fn dds_blocks_rejects_truncated_chain() {
        // Claims 4 mips but only supplies mip0's blocks.
        let block = [0u8; 8];
        let dds = wrap_dds_mips(b"DXT1", 8, 8, 4, &block.repeat(4));
        let err = blocks_err(&dds);
        assert_eq!(
            err,
            "DDS mip 1 (4x4) needs 8 bytes at offset 160, file has 160"
        );
    }

    #[test]
    fn dds_blocks_rejects_a_file_shorter_than_the_header() {
        let err = blocks_err(&[0u8; 16]);
        assert_eq!(err, "DDS too short: 16 bytes");
    }

    #[test]
    fn dds_blocks_rejects_bad_magic() {
        let mut dds = wrap_dds(b"DXT1", 4, 4, &bc1_red_block());
        dds[0] = b'X';
        let err = blocks_err(&dds);
        assert_eq!(err, "not a DDS file (bad magic)");
    }

    #[test]
    fn dds_blocks_rejects_a_zero_dimension() {
        let dds = wrap_dds(b"DXT1", 0, 4, &bc1_red_block());
        let err = blocks_err(&dds);
        assert_eq!(err, "DDS has zero dimension 0x4");
    }

    #[test]
    fn dds_blocks_maps_dxt5_and_ati2_to_bc3_and_bc5() {
        let dds = wrap_dds(b"DXT5", 4, 4, &[0u8; 16]);
        let blocks = decode_dds_blocks(&dds).expect("dxt5");
        assert_eq!(blocks.format, TextureFormat::Bc3);
        assert_eq!(blocks.mips[0].data.len(), 16);

        let dds = wrap_dds(b"ATI2", 4, 4, &[0u8; 16]);
        let blocks = decode_dds_blocks(&dds).expect("ati2");
        assert_eq!(blocks.format, TextureFormat::Bc5);
        assert_eq!(blocks.mips[0].data.len(), 16);
    }

    #[test]
    fn dds_blocks_rejects_dx10_and_unknown_fourccs() {
        let err = blocks_err(&wrap_dds(b"DX10", 4, 4, &[0u8; 16]));
        assert!(err.contains("DX10 extended header"), "got: {err}");
        let err = blocks_err(&wrap_dds(b"DXT3", 4, 4, &[0u8; 16]));
        assert!(
            err.contains("unsupported DDS fourCC \"DXT3\""),
            "got: {err}"
        );
    }

    #[test]
    fn decode_dds_rejects_a_zero_dimension() {
        let dds = wrap_dds(b"DXT1", 4, 0, &bc1_red_block());
        let err = decode_dds(&dds).unwrap_err();
        assert_eq!(err, "DDS has zero dimension 4x0");
    }

    #[test]
    fn decodes_dxt5_colour_and_alpha() {
        // Alpha endpoints 200/10 with all indices 0, then an opaque white
        // BC1 colour block.
        let mut block = [0u8; 16];
        block[0] = 200;
        block[1] = 10;
        block[8] = 0xFF;
        block[9] = 0xFF;
        let dds = wrap_dds(b"DXT5", 4, 4, &block);
        let (w, h, px) = decode_dds(&dds).unwrap();
        assert_eq!((w, h), (4, 4));
        for chunk in px.chunks(4) {
            assert_eq!(chunk, &[255, 255, 255, 200]);
        }
    }

    #[test]
    fn decode_dds_rejects_pixel_data_shorter_than_one_block() {
        for fourcc in [b"DXT1", b"DXT5", b"ATI2"] {
            let dds = wrap_dds(fourcc, 4, 4, &[0u8; 4]);
            let err = decode_dds(&dds).unwrap_err();
            assert!(
                err.starts_with("block-compressed data too short"),
                "{}: {err}",
                String::from_utf8_lossy(fourcc)
            );
        }
    }

    #[test]
    fn decodes_ati2_into_a_reconstructed_normal() {
        // Red and green both mid-grey: the flat normal, whose Z reconstructs
        // to near-max blue.
        let mut block = [0u8; 16];
        block[0] = 128;
        block[1] = 128;
        block[8] = 128;
        block[9] = 128;
        let dds = wrap_dds(b"ATI2", 4, 4, &block);
        let (_, _, px) = decode_dds(&dds).unwrap();
        assert_eq!(&px[0..2], &[128, 128]);
        assert!(px[2] >= 253, "expected near-255 blue, got {}", px[2]);
        assert_eq!(px[3], 255);
    }
}
