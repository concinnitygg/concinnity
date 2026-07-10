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
}
