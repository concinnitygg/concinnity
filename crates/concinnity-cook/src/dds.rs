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

use concinnity_core::build::texture::{TextureFormat, TextureMip};

const MAGIC: &[u8; 4] = b"DDS ";
const HEADER_LEN: usize = 124;
const PIXELDATA_OFFSET: usize = 4 + HEADER_LEN;

// Block-compressed mip chain read straight from a DDS file: the fourCC-derived
// format plus one entry per stored mip level, level 0 first.
pub struct DdsBlocks {
    pub format: TextureFormat,
    pub mips: Vec<TextureMip>,
}

// Read a legacy DDS's block data and mip chain without decoding. Returns the
// BCn format and every stored mip level's blocks. A DX10 header or an
// unsupported fourCC is rejected the same way `decode_dds` rejects them.
pub fn decode_dds_blocks(bytes: &[u8]) -> Result<DdsBlocks, String> {
    if bytes.len() < PIXELDATA_OFFSET {
        return Err(format!("DDS too short: {} bytes", bytes.len()));
    }
    if &bytes[0..4] != MAGIC {
        return Err("not a DDS file (bad magic)".to_string());
    }
    let height = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let width = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let mip_count = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]).max(1);
    let fourcc = &bytes[84..88];
    if width == 0 || height == 0 {
        return Err(format!("DDS has zero dimension {}x{}", width, height));
    }

    let format = match fourcc {
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

    let mut mips = Vec::with_capacity(mip_count as usize);
    let mut cursor = PIXELDATA_OFFSET;
    for level in 0..mip_count {
        let mw = (width >> level).max(1);
        let mh = (height >> level).max(1);
        let len = format.mip_byte_len(mw, mh);
        if cursor + len > bytes.len() {
            return Err(format!(
                "DDS mip {} ({}x{}) needs {} bytes at offset {}, file has {}",
                level,
                mw,
                mh,
                len,
                cursor,
                bytes.len()
            ));
        }
        mips.push(TextureMip {
            width: mw,
            height: mh,
            data: bytes[cursor..cursor + len].to_vec(),
        });
        cursor += len;
    }
    Ok(DdsBlocks { format, mips })
}

// Decode the top mip of a DDS file into (width, height, RGBA8 pixels).
pub fn decode_dds(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    if bytes.len() < PIXELDATA_OFFSET {
        return Err(format!("DDS too short: {} bytes", bytes.len()));
    }
    if &bytes[0..4] != MAGIC {
        return Err("not a DDS file (bad magic)".to_string());
    }

    let height = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let width = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let fourcc = &bytes[84..88];
    let data = &bytes[PIXELDATA_OFFSET..];

    if width == 0 || height == 0 {
        return Err(format!("DDS has zero dimension {}x{}", width, height));
    }

    let pixels = match fourcc {
        b"DXT1" => crate::bcn::decode_bc1(data, width, height)?,
        b"DXT5" => crate::bcn::decode_bc3(data, width, height)?,
        b"ATI2" => crate::bcn::decode_bc5(data, width, height)?,
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

    Ok((width, height, pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal legacy DDS around a single block payload.
    fn wrap_dds(fourcc: &[u8; 4], width: u32, height: u32, block: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; PIXELDATA_OFFSET];
        v[0..4].copy_from_slice(MAGIC);
        v[12..16].copy_from_slice(&height.to_le_bytes());
        v[16..20].copy_from_slice(&width.to_le_bytes());
        v[84..88].copy_from_slice(fourcc);
        v.extend_from_slice(block);
        v
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

    // Build a DDS with an explicit mip count and concatenated block levels.
    fn wrap_dds_mips(
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

    #[test]
    fn reads_bc1_mip_chain_blocks() {
        // 8x8 BC1: mip0 = 4 blocks (32B), mip1 (4x4) = 1 block (8B),
        // mip2 (2x2) = 1 block, mip3 (1x1) = 1 block.
        let block = [0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
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
        assert!(decode_dds_blocks(&dds).is_err());
    }
}
