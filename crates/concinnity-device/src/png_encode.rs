//! The PNG writer behind every backend's screenshot readback.

use concinnity_core::render::error::{RenderError, RenderResult};

/// Write RGBA8 pixel data to a PNG file.
pub(crate) fn encode_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> RenderResult<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| RenderError::Other(format!("screenshot: create {path}: {e}")))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| RenderError::Other(format!("screenshot: png header: {e}")))?;
    writer
        .write_image_data(rgba)
        .map_err(|e| RenderError::Other(format!("screenshot: png data: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_rgba8_that_decodes_back_unchanged() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("shot.png");
        let rgba = [255, 0, 0, 255, 0, 128, 255, 64];
        encode_png(path.to_str().unwrap(), 2, 1, &rgba).unwrap();

        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let frame = reader.next_frame(&mut pixels).unwrap();
        assert_eq!((frame.width, frame.height), (2, 1));
        assert_eq!(frame.color_type, png::ColorType::Rgba);
        assert_eq!(frame.bit_depth, png::BitDepth::Eight);
        assert_eq!(&pixels[..frame.buffer_size()], &rgba);
    }

    #[test]
    fn a_missing_directory_is_an_error() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("absent").join("shot.png");
        let err = encode_png(path.to_str().unwrap(), 1, 1, &[0; 4]).unwrap_err();
        assert!(matches!(err, RenderError::Other(_)));
    }
}
