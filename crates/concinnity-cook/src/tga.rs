// src/build/tga.rs
//
// Decodes a Targa (.tga) image into RGBA8. Handles the variants the asset set
// uses: uncompressed and RLE true-colour (24/32-bit, stored BGRA) and 8-bit
// grayscale. Colour-mapped images are not supported. The image-descriptor origin
// bit is honoured so bottom-left-origin files are flipped to top-row-first.

use concinnity_cpu::decode::checked_product;

const HEADER_LEN: usize = 18;

// Decode a TGA byte buffer into (width, height, RGBA8 pixels).
pub fn decode_tga(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    if bytes.len() < HEADER_LEN {
        return Err(format!("TGA too short: {} bytes", bytes.len()));
    }

    let id_len = bytes[0] as usize;
    let color_map_type = bytes[1];
    let image_type = bytes[2];
    let width = u16::from_le_bytes([bytes[12], bytes[13]]) as u32;
    let height = u16::from_le_bytes([bytes[14], bytes[15]]) as u32;
    let bpp = bytes[16];
    let descriptor = bytes[17];
    let top_origin = descriptor & 0x20 != 0;

    if color_map_type != 0 {
        return Err("colour-mapped TGA images are not supported".to_string());
    }
    if width == 0 || height == 0 {
        return Err(format!("TGA has zero dimension {}x{}", width, height));
    }

    let channels = match bpp {
        32 => 4usize,
        24 => 3,
        8 => 1,
        other => return Err(format!("unsupported TGA bit depth {}", other)),
    };

    let pixel_count = checked_product("TGA", &[width as usize, height as usize])?;
    let data_start = HEADER_LEN + id_len;
    if data_start > bytes.len() {
        return Err("TGA id field exceeds file length".to_string());
    }
    let data = &bytes[data_start..];

    // Decode into a flat BGRA-or-grayscale source buffer of `pixel_count` pixels.
    let raw = match image_type {
        2 | 3 => decode_raw(data, pixel_count, channels)?,
        10 | 11 => decode_rle(data, pixel_count, channels)?,
        other => return Err(format!("unsupported TGA image type {}", other)),
    };

    // Convert source channel order to RGBA, then flip rows if the origin is at
    // the bottom-left (the TGA default) so callers always get top-row-first.
    let mut rgba = vec![0u8; checked_product("TGA output", &[pixel_count, 4])?];
    let src = raw.chunks_exact(channels);
    let dst = rgba.chunks_exact_mut(4);
    match channels {
        4 => {
            for (s, d) in src.zip(dst) {
                d.copy_from_slice(&[s[2], s[1], s[0], s[3]]);
            }
        }
        3 => {
            for (s, d) in src.zip(dst) {
                d.copy_from_slice(&[s[2], s[1], s[0], 255]);
            }
        }
        _ => {
            for (s, d) in src.zip(dst) {
                d.copy_from_slice(&[s[0], s[0], s[0], 255]);
            }
        }
    }

    if !top_origin {
        flip_rows(&mut rgba, width as usize, height as usize);
    }

    Ok((width, height, rgba))
}

fn decode_raw(data: &[u8], pixel_count: usize, channels: usize) -> Result<Vec<u8>, String> {
    let needed = checked_product("TGA pixel data", &[pixel_count, channels])?;
    if data.len() < needed {
        return Err(format!(
            "TGA pixel data too short: have {}, need {}",
            data.len(),
            needed
        ));
    }
    Ok(data[..needed].to_vec())
}

fn decode_rle(data: &[u8], pixel_count: usize, channels: usize) -> Result<Vec<u8>, String> {
    let total = checked_product("TGA RLE", &[pixel_count, channels])?;
    // Reserved against the compressed length, not the declared pixel count: a
    // header claiming 65535x65535 would otherwise reserve gigabytes before a
    // single packet is read.
    let mut out = Vec::with_capacity(data.len().min(total));
    let mut p = 0usize;
    while out.len() < total {
        if p >= data.len() {
            return Err("TGA RLE stream ended early".to_string());
        }
        let packet = data[p];
        p += 1;
        let count = (packet & 0x7F) as usize + 1;
        if packet & 0x80 != 0 {
            // Run-length packet: one pixel repeated `count` times.
            if p + channels > data.len() {
                return Err("TGA RLE run packet truncated".to_string());
            }
            let pixel = &data[p..p + channels];
            p += channels;
            for _ in 0..count {
                out.extend_from_slice(pixel);
            }
        } else {
            // Raw packet: `count` literal pixels.
            let bytes = count * channels;
            if p + bytes > data.len() {
                return Err("TGA RLE raw packet truncated".to_string());
            }
            out.extend_from_slice(&data[p..p + bytes]);
            p += bytes;
        }
    }
    out.truncate(pixel_count * channels);
    Ok(out)
}

fn flip_rows(rgba: &mut [u8], width: usize, height: usize) {
    let stride = width * 4;
    for y in 0..height / 2 {
        let (head, tail) = rgba.split_at_mut((height - 1 - y) * stride);
        head[y * stride..y * stride + stride].swap_with_slice(&mut tail[..stride]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(image_type: u8, w: u16, h: u16, bpp: u8, top_origin: bool) -> Vec<u8> {
        let mut v = vec![0u8; HEADER_LEN];
        v[2] = image_type;
        v[12..14].copy_from_slice(&w.to_le_bytes());
        v[14..16].copy_from_slice(&h.to_le_bytes());
        v[16] = bpp;
        v[17] = if top_origin { 0x20 } else { 0 };
        v
    }

    // A maximal declared size in an otherwise empty RLE file. The decoder must
    // report the short stream; reserving for the declared pixel count up front
    // would ask the allocator for gigabytes off an 19-byte input.
    #[test]
    fn rle_with_maximal_dimensions_reports_a_short_stream() {
        let mut v = header(10, u16::MAX, u16::MAX, 32, true);
        v.push(0x80);
        let err = decode_tga(&v).unwrap_err();
        assert!(err.contains("RLE"), "{}", err);
    }

    #[test]
    fn uncompressed_with_maximal_dimensions_reports_short_pixel_data() {
        let v = header(2, u16::MAX, u16::MAX, 32, true);
        let err = decode_tga(&v).unwrap_err();
        assert!(err.contains("too short"), "{}", err);
    }

    #[test]
    fn rejects_an_id_field_past_the_end() {
        let mut v = header(2, 1, 1, 24, true);
        v[0] = 200;
        assert!(decode_tga(&v).is_err());
    }

    #[test]
    fn uncompressed_24bit_bgr_to_rgba() {
        // 1x1, top origin, single BGR pixel (B=10, G=20, R=30) -> RGBA(30,20,10,255).
        let mut v = header(2, 1, 1, 24, true);
        v.extend_from_slice(&[10, 20, 30]);
        let (w, h, px) = decode_tga(&v).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(px, vec![30, 20, 10, 255]);
    }

    #[test]
    fn uncompressed_32bit_keeps_alpha() {
        let mut v = header(2, 1, 1, 32, true);
        v.extend_from_slice(&[10, 20, 30, 128]); // BGRA
        let (_, _, px) = decode_tga(&v).unwrap();
        assert_eq!(px, vec![30, 20, 10, 128]);
    }

    #[test]
    fn rle_run_packet_expands() {
        // 4x1, top origin, one run packet of 4 identical BGR pixels.
        let mut v = header(10, 4, 1, 24, true);
        v.push(0x80 | 3); // run of 4
        v.extend_from_slice(&[1, 2, 3]);
        let (_, _, px) = decode_tga(&v).unwrap();
        assert_eq!(px.len(), 4 * 4);
        for chunk in px.chunks(4) {
            assert_eq!(chunk, &[3, 2, 1, 255]);
        }
    }

    #[test]
    fn bottom_origin_flips_rows() {
        // 1x2, bottom origin: source row0 then row1; output should be flipped.
        let mut v = header(2, 1, 2, 24, false);
        v.extend_from_slice(&[0, 0, 0]); // bottom row (B/G/R = black)
        v.extend_from_slice(&[255, 255, 255]); // top row (white)
        let (_, _, px) = decode_tga(&v).unwrap();
        // After flip, first output row is the white pixel.
        assert_eq!(&px[0..4], &[255, 255, 255, 255]);
        assert_eq!(&px[4..8], &[0, 0, 0, 255]);
    }

    #[test]
    fn grayscale_replicates_to_rgb() {
        let mut v = header(3, 1, 1, 8, true);
        v.push(77);
        let (_, _, px) = decode_tga(&v).unwrap();
        assert_eq!(px, vec![77, 77, 77, 255]);
    }

    #[test]
    fn rejects_a_buffer_shorter_than_the_fixed_header() {
        let err = decode_tga(&[0u8; HEADER_LEN - 1]).unwrap_err();
        assert_eq!(err, "TGA too short: 17 bytes");
    }

    #[test]
    fn rejects_a_colour_mapped_image() {
        let mut v = header(1, 1, 1, 24, true);
        v[1] = 1; // colour-map type
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "colour-mapped TGA images are not supported");
    }

    #[test]
    fn rejects_a_zero_dimension() {
        let v = header(2, 0, 4, 24, true);
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "TGA has zero dimension 0x4");
    }

    #[test]
    fn rejects_an_unsupported_bit_depth() {
        let v = header(2, 1, 1, 16, true);
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "unsupported TGA bit depth 16");
    }

    #[test]
    fn rejects_an_unsupported_image_type() {
        // Type 1 is colour-mapped image data, which this decoder does not read.
        let v = header(1, 1, 1, 24, true);
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "unsupported TGA image type 1");
    }

    #[test]
    fn rejects_an_id_field_that_runs_past_the_end() {
        let mut v = header(2, 1, 1, 24, true);
        v[0] = 200; // id length far beyond the file
        v.extend_from_slice(&[10, 20, 30]);
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "TGA id field exceeds file length");
    }

    #[test]
    fn skips_the_id_field_before_the_pixel_data() {
        let mut v = header(2, 1, 1, 24, true);
        v[0] = 3;
        v.extend_from_slice(b"abc"); // id field, not pixels
        v.extend_from_slice(&[10, 20, 30]);
        let (_, _, px) = decode_tga(&v).unwrap();
        assert_eq!(px, vec![30, 20, 10, 255]);
    }

    #[test]
    fn rejects_truncated_uncompressed_pixel_data() {
        let mut v = header(2, 2, 1, 24, true);
        v.extend_from_slice(&[1, 2, 3]); // one pixel, two needed
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "TGA pixel data too short: have 3, need 6");
    }

    #[test]
    fn rle_raw_packet_copies_literal_pixels() {
        // 2x1, one raw packet holding two literal BGR pixels.
        let mut v = header(10, 2, 1, 24, true);
        v.push(1); // raw packet, count = 2
        v.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let (_, _, px) = decode_tga(&v).unwrap();
        assert_eq!(px, vec![3, 2, 1, 255, 6, 5, 4, 255]);
    }

    #[test]
    fn rle_run_overshooting_the_image_is_truncated() {
        // 3x1 image but a run of 4 pixels: the surplus pixel is dropped.
        let mut v = header(10, 3, 1, 24, true);
        v.push(0x80 | 3);
        v.extend_from_slice(&[1, 2, 3]);
        let (_, _, px) = decode_tga(&v).unwrap();
        assert_eq!(px.len(), 3 * 4);
    }

    #[test]
    fn rle_grayscale_run_replicates_to_rgb() {
        let mut v = header(11, 2, 1, 8, true);
        v.push(0x80 | 1); // run of 2
        v.push(90);
        let (_, _, px) = decode_tga(&v).unwrap();
        assert_eq!(px, vec![90, 90, 90, 255, 90, 90, 90, 255]);
    }

    #[test]
    fn rejects_an_rle_stream_that_ends_before_the_image_is_filled() {
        let v = header(10, 4, 1, 24, true); // no packets at all
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "TGA RLE stream ended early");
    }

    #[test]
    fn rejects_a_truncated_rle_run_packet() {
        let mut v = header(10, 4, 1, 24, true);
        v.push(0x80 | 3);
        v.extend_from_slice(&[1, 2]); // the repeated pixel needs three bytes
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "TGA RLE run packet truncated");
    }

    #[test]
    fn rejects_a_truncated_rle_raw_packet() {
        let mut v = header(10, 4, 1, 24, true);
        v.push(1); // raw packet of two pixels
        v.extend_from_slice(&[1, 2, 3, 4]); // four of the six bytes
        let err = decode_tga(&v).unwrap_err();
        assert_eq!(err, "TGA RLE raw packet truncated");
    }
}
