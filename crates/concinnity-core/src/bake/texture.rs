//! Texture payload format helpers shared between the runtime and the build
//! crate. The file -> pixels decoders (PNG / JPEG / DDS / TGA / KTX2 /
//! glb-embedded images) live in `concinnity_cook::compile::texture`; this module keeps
//! only what a running engine needs with no image-decode dependencies: turning a
//! compiled payload back into a [`TextureImage`] (`deserialise`) and the
//! box-filter `downscale_rgba` the build pipeline uses to cap oversized source
//! maps.
//!
//! One tagged format serves every 2D texture (little-endian):
//!   u32  magic      = b"TEX2"
//!   u32  format_id  (0 RGBA8, 1 BC1, 2 BC3, 3 BC5, 4 BC7)
//!   u32  mip_count  (>= 1)
//!   per mip, level 0 first (largest):
//!     u32  width
//!     u32  height
//!     u32  byte_len
//!     byte_len bytes of level data
//!
//! RGBA8 sources (PNG / JPEG / procedural generators) carry a single mip; the
//! backend upload generates the minification chain (`crate::render::mipmap`).
//! Block-compressed sources (KTX2 / DDS) carry the container's full mip chain and
//! upload it verbatim, since no runtime BCn encoder exists.

// Magic tagging every compiled 2D texture payload.
use crate::decode::{ByteReader, checked_product};
use crate::math::ceil;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// A u32 dimension bottoms out at 1x1 after at most this many halvings, so a
/// file or payload declaring more levels than this is malformed rather than
/// large. Decoders bound a declared mip count against it before reserving.
pub const MAX_MIP_LEVELS: usize = 32;

pub(crate) const TEXTURE_PAYLOAD_MAGIC: u32 = u32::from_le_bytes(*b"TEX2");
const HEADER_BYTES: usize = 12;

/// GPU pixel format of a compiled texture payload. RGBA8 is the uncompressed
/// path (runtime mip generation); the rest are block-compressed formats uploaded
/// with their container mip chains.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextureFormat {
    /// Uncompressed 8-bit RGBA; mips are generated at upload.
    Rgba8,
    /// BC1 block compression.
    Bc1,
    /// BC3 block compression.
    Bc3,
    /// BC5 block compression, for normal maps.
    Bc5,
    /// BC7 block compression.
    Bc7,
}

impl TextureFormat {
    /// Stable on-disk identifier written into the payload header.
    pub fn id(self) -> u32 {
        match self {
            TextureFormat::Rgba8 => 0,
            TextureFormat::Bc1 => 1,
            TextureFormat::Bc3 => 2,
            TextureFormat::Bc5 => 3,
            TextureFormat::Bc7 => 4,
        }
    }

    pub(crate) fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(TextureFormat::Rgba8),
            1 => Some(TextureFormat::Bc1),
            2 => Some(TextureFormat::Bc3),
            3 => Some(TextureFormat::Bc5),
            4 => Some(TextureFormat::Bc7),
            _ => None,
        }
    }

    /// Bytes per 4x4 block for a compressed format. `None` for RGBA8, which is
    /// sized per pixel rather than per block.
    pub fn block_bytes(self) -> Option<usize> {
        match self {
            TextureFormat::Rgba8 => None,
            TextureFormat::Bc1 => Some(8),
            TextureFormat::Bc3 | TextureFormat::Bc5 | TextureFormat::Bc7 => Some(16),
        }
    }

    /// Byte length one mip of `width` x `height` occupies in this format.
    /// Dimensions are read out of the file being decoded, so a footprint that
    /// overflows `usize` is reported rather than wrapped into a small length
    /// that would pass the caller's bounds check.
    pub fn mip_byte_len(self, width: u32, height: u32) -> Result<usize, String> {
        match self.block_bytes() {
            None => checked_product("texture mip", &[width as usize, height as usize, 4]),
            Some(block) => checked_product(
                "texture mip",
                &[
                    width.div_ceil(4) as usize,
                    height.div_ceil(4) as usize,
                    block,
                ],
            ),
        }
    }
}

/// One mip level: dimensions plus its tightly packed level bytes (RGBA8 pixels or
/// block-compressed data, per the owning image's format).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureMip {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixels or compressed blocks of this level.
    pub data: Vec<u8>,
}

/// A decoded 2D texture: its GPU format plus one or more mip levels, level 0
/// first. The backend uploads `mips` directly for compressed formats and
/// generates the minification chain from `mips[0]` for RGBA8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureImage {
    /// GPU pixel format of every mip.
    pub format: TextureFormat,
    /// Mip levels, level 0 first.
    pub mips: Vec<TextureMip>,
}

impl TextureImage {
    /// Wrap a single RGBA8 level (the PNG / JPEG / procedural path). The upload
    /// path generates the mip chain.
    pub fn rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        TextureImage {
            format: TextureFormat::Rgba8,
            mips: vec![TextureMip {
                width,
                height,
                data: pixels,
            }],
        }
    }

    /// Base (level 0) dimensions.
    pub fn width(&self) -> u32 {
        self.mips.first().map(|m| m.width).unwrap_or(0)
    }

    /// Base (level 0) height in pixels.
    pub fn height(&self) -> u32 {
        self.mips.first().map(|m| m.height).unwrap_or(0)
    }

    /// Total resident bytes across every mip level (streaming budget accounting).
    pub fn byte_len(&self) -> usize {
        self.mips.iter().map(|m| m.data.len()).sum()
    }

    /// Recover the base RGBA8 pixels, or an error if the image is block
    /// compressed. Used by the sprite / glyph-atlas paths, which upload RGBA8
    /// only.
    pub fn into_rgba8(self) -> Result<(u32, u32, Vec<u8>), String> {
        if self.format != TextureFormat::Rgba8 {
            return Err(format!(
                "texture is {:?}, expected RGBA8 for this path",
                self.format
            ));
        }
        let mip = self
            .mips
            .into_iter()
            .next()
            .ok_or("RGBA8 texture has no mip level")?;
        Ok((mip.width, mip.height, mip.data))
    }
}

/// Serialise a [`TextureImage`] into the tagged payload the runtime reads. The
/// build crate writes payloads through this so the reader and writer share one
/// format definition.
pub fn serialise(image: &TextureImage) -> Vec<u8> {
    let total: usize = HEADER_BYTES + image.mips.iter().map(|m| 12 + m.data.len()).sum::<usize>();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&TEXTURE_PAYLOAD_MAGIC.to_le_bytes());
    buf.extend_from_slice(&image.format.id().to_le_bytes());
    buf.extend_from_slice(&(image.mips.len() as u32).to_le_bytes());
    for mip in &image.mips {
        buf.extend_from_slice(&mip.width.to_le_bytes());
        buf.extend_from_slice(&mip.height.to_le_bytes());
        buf.extend_from_slice(&(mip.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&mip.data);
    }
    buf
}

/// Deserialise a tagged payload back into a [`TextureImage`].
///
/// Called by GraphicsSystem at runtime to recover texture format, dimensions,
/// and mip data before uploading to the GPU.
pub fn deserialise(bytes: &[u8]) -> Result<TextureImage, String> {
    let mut r = ByteReader::open_payload(bytes, TEXTURE_PAYLOAD_MAGIC, HEADER_BYTES, "texture")?;
    let format_id = r.u32()?;
    let format = TextureFormat::from_id(format_id)
        .ok_or_else(|| format!("texture payload has unknown format_id {}", format_id))?;
    // Bounded before it reaches `with_capacity`: the count is payload-supplied
    // and reserving for a u32 of them would abort on the allocation alone.
    let mip_count = r.u32()? as usize;
    if mip_count == 0 || mip_count > MAX_MIP_LEVELS {
        return Err(format!(
            "texture payload declares {} mip levels (expected 1..={})",
            mip_count, MAX_MIP_LEVELS
        ));
    }

    r.seek(HEADER_BYTES)?;
    let mut mips = Vec::with_capacity(mip_count);
    for level in 0..mip_count {
        let width = r.u32()?;
        let height = r.u32()?;
        let byte_len = r.u32()? as usize;
        let expected = format.mip_byte_len(width, height)?;
        if byte_len != expected {
            return Err(format!(
                "texture payload mip {} ({}x{} {:?}) declares {} bytes, format needs {}",
                level, width, height, format, byte_len, expected
            ));
        }
        mips.push(TextureMip {
            width,
            height,
            data: r.take(byte_len)?.to_vec(),
        });
    }

    Ok(TextureImage { format, mips })
}

/// Box-filter an RGBA image down so its longest edge is at most `max_size`. A
/// `max_size` of 0 (or an image already within budget) returns the input
/// unchanged. Used to keep oversized source maps (4K+ DDS) from exploding the
/// compiled blob, which stores raw RGBA8.
pub fn downscale_rgba(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    max_size: u32,
) -> (u32, u32, Vec<u8>) {
    if max_size == 0 || (width <= max_size && height <= max_size) {
        return (width, height, pixels);
    }
    let scale = ceil(width.max(height) as f32 / max_size as f32) as u32;
    let scale = scale.max(2);
    let dst_w = (width / scale).max(1);
    let dst_h = (height / scale).max(1);

    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for sy in 0..scale {
                let src_y = dy * scale + sy;
                if src_y >= height {
                    break;
                }
                for sx in 0..scale {
                    let src_x = dx * scale + sx;
                    if src_x >= width {
                        break;
                    }
                    let si = ((src_y * width + src_x) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += pixels[si + c] as u32;
                    }
                    n += 1;
                }
            }
            let di = ((dy * dst_w + dx) * 4) as usize;
            for c in 0..4 {
                out[di + c] = acc[c].checked_div(n).unwrap_or(0) as u8;
            }
        }
    }
    (dst_w, dst_h, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(image: &TextureImage) -> TextureImage {
        let bytes = serialise(image);
        deserialise(&bytes).expect("deserialise")
    }

    #[test]
    fn rgba8_single_mip_round_trips() {
        let image = TextureImage::rgba8(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let back = round_trip(&image);
        assert_eq!(back, image);
        assert_eq!(back.format, TextureFormat::Rgba8);
        assert_eq!((back.width(), back.height()), (2, 1));
    }

    #[test]
    fn compressed_multi_mip_round_trips() {
        // BC1: 8 bytes per 4x4 block. A 4x4 mip is one block; a 2x2 mip clips to
        // one block too (ceil(2/4) == 1).
        let image = TextureImage {
            format: TextureFormat::Bc1,
            mips: vec![
                TextureMip {
                    width: 4,
                    height: 4,
                    data: vec![0xAB; 8],
                },
                TextureMip {
                    width: 2,
                    height: 2,
                    data: vec![0xCD; 8],
                },
            ],
        };
        let back = round_trip(&image);
        assert_eq!(back, image);
        assert_eq!(back.byte_len(), 16);
    }

    #[test]
    fn deserialise_rejects_bad_magic() {
        let mut bytes = serialise(&TextureImage::rgba8(1, 1, vec![0, 0, 0, 0]));
        bytes[0] ^= 0xFF;
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("magic"), "got: {err}");
    }

    #[test]
    fn deserialise_rejects_unknown_format() {
        let mut bytes = serialise(&TextureImage::rgba8(1, 1, vec![0, 0, 0, 0]));
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("unknown format_id"), "got: {err}");
    }

    #[test]
    fn deserialise_rejects_wrong_mip_length() {
        // Declare a BC7 4x4 mip (needs 16 bytes) but supply 8.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&TEXTURE_PAYLOAD_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&TextureFormat::Bc7.id().to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("format needs 16"), "got: {err}");
    }

    // A one-mip header with caller-chosen fields and no mip body behind it.
    fn header(format: TextureFormat, mip_count: u32, width: u32, height: u32, len: u32) -> Vec<u8> {
        let mut bytes = TEXTURE_PAYLOAD_MAGIC.to_le_bytes().to_vec();
        bytes.extend_from_slice(&format.id().to_le_bytes());
        bytes.extend_from_slice(&mip_count.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes
    }

    #[test]
    fn deserialise_rejects_a_payload_shorter_than_the_header() {
        let full = serialise(&TextureImage::rgba8(1, 1, vec![0; 4]));
        for len in 0..HEADER_BYTES {
            assert!(deserialise(&full[..len]).is_err(), "len {} decoded", len);
        }
    }

    #[test]
    fn deserialise_rejects_a_truncated_mip_header() {
        let mut bytes = header(TextureFormat::Rgba8, 1, 2, 2, 16);
        bytes.truncate(HEADER_BYTES + 6);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("unexpected end"), "got: {err}");
    }

    #[test]
    fn deserialise_rejects_truncated_mip_data() {
        let mut bytes = header(TextureFormat::Rgba8, 1, 2, 2, 16);
        bytes.extend_from_slice(&[0u8; 8]);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("unexpected end"), "got: {err}");
    }

    // `width * height * 4` wraps for these dimensions; the declared length
    // must be rejected on the overflow rather than compared against a
    // wrapped-around footprint.
    #[test]
    fn deserialise_rejects_dimensions_that_overflow_the_footprint() {
        let bytes = header(TextureFormat::Rgba8, 1, u32::MAX, u32::MAX, 16);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("overflow"), "got: {err}");
    }

    // A mip count near u32::MAX must be rejected on the declared value, not
    // by attempting to reserve for it.
    #[test]
    fn deserialise_rejects_an_absurd_mip_count() {
        let bytes = header(TextureFormat::Rgba8, u32::MAX, 1, 1, 4);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("mip levels"), "got: {err}");
    }

    #[test]
    fn deserialise_rejects_zero_mips() {
        let bytes = header(TextureFormat::Rgba8, 0, 1, 1, 4);
        assert!(deserialise(&bytes).is_err());
    }

    #[test]
    fn mip_byte_len_reports_overflow_for_max_dimensions() {
        assert!(
            TextureFormat::Rgba8
                .mip_byte_len(u32::MAX, u32::MAX)
                .is_err()
        );
        assert!(TextureFormat::Bc7.mip_byte_len(u32::MAX, u32::MAX).is_err());
        assert_eq!(TextureFormat::Rgba8.mip_byte_len(2, 2).unwrap(), 16);
        assert_eq!(TextureFormat::Bc7.mip_byte_len(4, 4).unwrap(), 16);
    }

    #[test]
    fn into_rgba8_rejects_compressed() {
        let image = TextureImage {
            format: TextureFormat::Bc3,
            mips: vec![TextureMip {
                width: 4,
                height: 4,
                data: vec![0; 16],
            }],
        };
        assert!(image.into_rgba8().is_err());
    }

    #[test]
    fn downscale_rgba_noop_within_budget() {
        let px = vec![1u8; 8 * 8 * 4];
        let (w, h, out) = downscale_rgba(8, 8, px.clone(), 16);
        assert_eq!((w, h), (8, 8));
        assert_eq!(out, px);
    }

    #[test]
    fn downscale_rgba_halves_oversized() {
        let px = vec![128u8; 8 * 8 * 4];
        let (w, h, out) = downscale_rgba(8, 8, px, 4);
        assert_eq!((w, h), (4, 4));
        assert_eq!(out.len(), 4 * 4 * 4);
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn downscale_rgba_zero_max_is_noop() {
        let px = vec![7u8; 4 * 4 * 4];
        let (w, h, out) = downscale_rgba(4, 4, px.clone(), 0);
        assert_eq!((w, h), (4, 4));
        assert_eq!(out, px);
    }
}
