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
