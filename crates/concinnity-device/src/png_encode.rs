//! The PNG writer behind every backend's screenshot readback.

use concinnity_engine::render::backend::error::{RenderError, RenderResult};

use std::io::Write;

/// Write an RGBA8 image to a PNG file one row at a time: `row(y, out)` appends
/// row `y`'s `width * 4` bytes to `out`, which is cleared before each row. Only
/// a row and the compressor's window are held, never the whole image.
pub(crate) fn encode_png_rows(
    path: &str,
    width: u32,
    height: u32,
    mut row: impl FnMut(usize, &mut Vec<u8>) -> RenderResult<()>,
) -> RenderResult<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| RenderError::Other(format!("screenshot: create {path}: {e}")))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut stream = encoder
        .write_header()
        .and_then(png::Writer::into_stream_writer)
        .map_err(|e| RenderError::Other(format!("screenshot: png header: {e}")))?;
    let mut bytes = Vec::with_capacity(width as usize * 4);
    for y in 0..height as usize {
        bytes.clear();
        row(y, &mut bytes)?;
        stream
            .write_all(&bytes)
            .map_err(|e| RenderError::Other(format!("screenshot: png data: {e}")))?;
    }
    stream
        .finish()
        .map_err(|e| RenderError::Other(format!("screenshot: png data: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_rgba8_that_decodes_back_unchanged() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("shot.png");
        let rgba = [255, 0, 0, 255, 0, 128, 255, 64, 1, 2, 3, 4, 5, 6, 7, 8];
        encode_png_rows(path.to_str().unwrap(), 2, 2, |y, out| {
            out.extend_from_slice(&rgba[y * 8..(y + 1) * 8]);
            Ok(())
        })
        .unwrap();

        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let frame = reader.next_frame(&mut pixels).unwrap();
        assert_eq!((frame.width, frame.height), (2, 2));
        assert_eq!(frame.color_type, png::ColorType::Rgba);
        assert_eq!(frame.bit_depth, png::BitDepth::Eight);
        assert_eq!(&pixels[..frame.buffer_size()], &rgba);
    }

    #[test]
    fn a_missing_directory_is_an_error() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("absent").join("shot.png");
        let err = encode_png_rows(path.to_str().unwrap(), 1, 1, |_, out| {
            out.extend_from_slice(&[0; 4]);
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(err, RenderError::Other(_)));
    }

    #[test]
    fn a_row_that_fails_stops_the_write() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("shot.png");
        let err = encode_png_rows(path.to_str().unwrap(), 1, 2, |y, out| {
            if y == 1 {
                return Err(RenderError::Other("short read-back".into()));
            }
            out.extend_from_slice(&[0; 4]);
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(err, RenderError::Other(m) if m == "short read-back"));
    }
}
