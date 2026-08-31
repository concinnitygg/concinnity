//! PNG bytes.

/// Encode `pixels` as an 8-bit RGBA PNG.
///
/// `pixels` is `width * height * 4` bytes, row-major.
///
/// # Panics
///
/// If `pixels` is not the length the dimensions call for.
pub fn rgba(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    encode(width, height, png::ColorType::Rgba, pixels)
}

/// Encode `pixels` as an 8-bit grayscale PNG.
///
/// `pixels` is `width * height` bytes, row-major.
///
/// # Panics
///
/// If `pixels` is not the length the dimensions call for.
pub fn gray(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    encode(width, height, png::ColorType::Grayscale, pixels)
}

/// An 8-bit RGBA PNG of one colour.
///
/// # Panics
///
/// If the dimensions overflow the pixel buffer.
pub fn solid(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
    let count = usize::try_from(width * height).expect("the image fits in memory");
    let pixels: Vec<u8> = color.iter().copied().cycle().take(count * 4).collect();
    rgba(width, height, &pixels)
}

/// The smallest PNG a decoder will accept: one opaque RGBA pixel.
pub fn one_pixel() -> Vec<u8> {
    solid(1, 1, [10, 20, 30, 255])
}

/// Encode at an explicit colour type, for the decode paths that branch on one.
///
/// # Panics
///
/// If `pixels` is not the length the dimensions and colour type call for.
pub fn encode(width: u32, height: u32, color: png::ColorType, pixels: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("the PNG header is written");
    writer
        .write_image_data(pixels)
        .expect("the pixels match the declared dimensions");
    writer.finish().expect("the PNG is finished");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().expect("the fixture is a readable PNG");
        let mut buf = vec![0; reader.output_buffer_size().expect("a bounded buffer")];
        let info = reader.next_frame(&mut buf).expect("one frame decodes");
        buf.truncate(info.buffer_size());
        (info.width, info.height, buf)
    }

    #[test]
    fn an_rgba_fixture_decodes_to_the_pixels_it_was_given() {
        let pixels = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let (w, h, out) = decode(&rgba(2, 1, &pixels));

        assert_eq!((w, h), (2, 1));
        assert_eq!(out, pixels);
    }

    #[test]
    fn a_grayscale_fixture_keeps_its_colour_type() {
        let (w, h, out) = decode(&gray(2, 2, &[0, 64, 128, 255]));

        assert_eq!((w, h), (2, 2));
        assert_eq!(out, [0, 64, 128, 255]);
    }

    #[test]
    fn a_solid_fixture_repeats_one_colour() {
        let (w, h, out) = decode(&solid(2, 2, [9, 8, 7, 255]));

        assert_eq!((w, h), (2, 2));
        assert_eq!(out.len(), 2 * 2 * 4);
        assert!(out.chunks_exact(4).all(|p| p == [9, 8, 7, 255]));
    }

    #[test]
    fn the_one_pixel_fixture_is_a_single_opaque_texel() {
        let (w, h, out) = decode(&one_pixel());

        assert_eq!((w, h), (1, 1));
        assert_eq!(out, [10, 20, 30, 255]);
    }
}
