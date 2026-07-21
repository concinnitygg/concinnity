use super::helpers::file_extension;

// Read an image file's pixel dimensions from its header, without decoding
// the pixels.
pub(super) fn probe_image_dims(path: &str) -> Result<(u32, u32), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot read '{}': {}", path, e))?;
    if file_extension(path) == "png" {
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let reader = decoder
            .read_info()
            .map_err(|e| format!("'{}': {}", path, e))?;
        let info = reader.info();
        Ok((info.width, info.height))
    } else {
        let mut decoder = jpeg_decoder::Decoder::new(std::io::BufReader::new(file));
        decoder
            .read_info()
            .map_err(|e| format!("'{}': {}", path, e))?;
        let info = decoder
            .info()
            .ok_or_else(|| format!("'{}': no image info", path))?;
        Ok((info.width as u32, info.height as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_png_dimensions_from_the_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.png");
        // A 6x9 grey PNG, encoded in memory to a temp file.
        let file = std::fs::File::create(&path).unwrap();
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 6, 9);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[0u8; 6 * 9]).unwrap();
        drop(writer);

        let (w, h) = probe_image_dims(path.to_str().unwrap()).expect("probe png");
        assert_eq!((w, h), (6, 9));
    }

    #[test]
    fn missing_file_surfaces_a_read_error() {
        let err = probe_image_dims("/no/such/portrait.png").unwrap_err();
        assert!(err.contains("cannot read"), "got: {err}");
    }

    #[test]
    fn a_corrupt_png_surfaces_a_decode_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.png");
        std::fs::write(&path, b"not a png at all").unwrap();
        let err = probe_image_dims(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("portrait.png"), "got: {err}");
        assert!(!err.contains("cannot read"), "got: {err}");
    }

    // SOI plus a baseline 6x9 single-component frame header: everything the
    // probe reads, since it stops as soon as the frame is known.
    const JPEG_6X9_HEADER: [u8; 15] = [
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x09, 0x00, 0x06, 0x01, 0x01, 0x11, 0x00,
    ];

    #[test]
    fn reads_jpeg_dimensions_from_the_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.jpg");
        std::fs::write(&path, JPEG_6X9_HEADER).unwrap();
        let (w, h) = probe_image_dims(path.to_str().unwrap()).expect("probe jpeg");
        assert_eq!((w, h), (6, 9));
    }

    // Anything that is not a `.png` goes down the JPEG path, extension aside.
    #[test]
    fn a_non_png_extension_is_probed_as_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.jpeg");
        std::fs::write(&path, JPEG_6X9_HEADER).unwrap();
        assert_eq!(probe_image_dims(path.to_str().unwrap()), Ok((6, 9)));
    }

    #[test]
    fn a_corrupt_jpeg_surfaces_a_decode_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.jpg");
        std::fs::write(&path, b"not a jpeg at all").unwrap();
        let err = probe_image_dims(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("portrait.jpg"), "got: {err}");
        assert!(!err.contains("cannot read"), "got: {err}");
    }
}
