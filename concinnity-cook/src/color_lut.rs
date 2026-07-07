// src/color_lut.rs
//
// Compiles a ColorLut component's args into the binary payload the renderer
// uploads as a 3D colour-grading LUT. The runtime samples this LUT in the
// composite (post-process) pass with the display-referred sRGB colour as the
// texture coordinate, blending the graded result by `PostProcessConfig`'s
// `lut_strength`.
//
// Two source formats are supported, picked by file extension:
//   - `.cube`  Adobe Cube LUT (plain text): a `LUT_3D_SIZE n` line followed by
//              n*n*n "r g b" float triplets with red varying fastest.
//   - `.png`   A horizontal slice strip: an (n*n)-by-n image of n square
//              slices. Blue selects the slice, red is the column within a
//              slice, green is the row.
//
// Payload format (little-endian):
//   u32  magic     = b"LUT3" = 0x3354554c
//   u32  size      LUT edge length n; the runtime texture is n x n x n
//   u32  format_id = 0  (RGBA8)
//   n*n*n*4 bytes  raw RGBA8, x(red) fastest, then y(green), then z(blue)
//
// The texel order matches both the Metal 3D-texture upload layout and the
// `.cube` data-line order, so the `.cube` path appends triplets verbatim.

// The no-dependency `.cube` parse, the classifier, the (de)serialisers, and the
// size validator stay in concinnity-core; the `.png` slice-strip decode below
// lives here in the build crate alongside the `png` crate.
use concinnity_core::build::color_lut::{
    LutFormat, classify_source, parse_cube, resolve_lut_source, serialise, validate_size,
};

// Validate that args specify a `ColorLut` source with a supported extension.
pub fn validate_color_lut_args(args: &serde_json::Value) -> Result<(), String> {
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    if source.is_empty() {
        return Err("ColorLut requires a `source` path".into());
    }
    classify_source(source)?;
    Ok(())
}

// Compile a `ColorLut` component's JSON args into a packed binary payload.
pub fn compile_color_lut_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    validate_color_lut_args(args)?;
    let source = args.get("source").and_then(|v| v.as_str()).unwrap();
    let format = classify_source(source)?;
    let resolved = resolve_lut_source(source);

    let (size, data) = match format {
        LutFormat::Cube => {
            let text = std::fs::read_to_string(&resolved)
                .map_err(|e| format!("failed to read LUT source '{}': {}", resolved, e))?;
            parse_cube(&text)?
        }
        LutFormat::Png => parse_png_strip(&resolved)?,
    };
    Ok(serialise(size, &data))
}

// Decode a ColorLut source path the same way `compile_color_lut_payload` does
// at build time. Dispatches between `.cube` text parse and PNG-strip parse
// based on the resolved file extension. Exposed for the runtime asset
// hot-reload path (`cn debug` only); production reads the compiled payload via
// `concinnity_core::build::color_lut::deserialise` instead. The `source`
// argument is the raw string from the asset declaration; this function applies
// the same asset-dir resolution the build pipeline uses.
pub fn decode_source(source: &str) -> Result<(u32, Vec<u8>), String> {
    let format = classify_source(source)?;
    let resolved = resolve_lut_source(source);
    match format {
        LutFormat::Cube => {
            let text = std::fs::read_to_string(&resolved)
                .map_err(|e| format!("failed to read LUT source '{}': {}", resolved, e))?;
            parse_cube(&text)
        }
        LutFormat::Png => parse_png_strip(&resolved),
    }
}

// .png slice-strip parsing

// Parse a horizontal LUT slice strip PNG into `(size, RGBA8 data)`. The image
// is `(n*n)` wide by `n` tall: n square slices laid left-to-right where blue
// selects the slice, red is the column within a slice and green is the row.
pub fn parse_png_strip(path: &str) -> Result<(u32, Vec<u8>), String> {
    let (width, height, pixels) = load_png_rgba8(path)?;
    let size = height;
    validate_size(size)?;
    if width != size * size {
        return Err(format!(
            "ColorLut strip '{}' is {}x{}; a size-{} strip must be {}x{}",
            path,
            width,
            height,
            size,
            size * size,
            size
        ));
    }
    let n = size as usize;
    let mut data = Vec::with_capacity(n * n * n * 4);
    // Emit in red-fastest, then green, then blue order to match the payload.
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let px = b * n + r;
                let src = (g * width as usize + px) * 4;
                data.extend_from_slice(&pixels[src..src + 4]);
            }
        }
    }
    Ok((size, data))
}

// Decode a PNG into (width, height, RGBA8 pixels). Only RGB / RGBA colour
// types are accepted: a LUT strip is always full colour.
fn load_png_rgba8(path: &str) -> Result<(u32, u32, Vec<u8>), String> {
    use png::ColorType;
    let file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open LUT source '{}': {}", path, e))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("failed to read PNG info for '{}': {}", path, e))?;
    let mut buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .ok_or("failed to compute PNG output buffer size")?
    ];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("failed to decode PNG frame for '{}': {}", path, e))?;
    let raw = &buf[..info.buffer_size()];
    let pixels = match info.color_type {
        ColorType::Rgba => raw.to_vec(),
        ColorType::Rgb => {
            let mut out = Vec::with_capacity(info.width as usize * info.height as usize * 4);
            for chunk in raw.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            out
        }
        other => {
            return Err(format!(
                "ColorLut strip '{}' has unsupported PNG colour type {:?} (need RGB/RGBA)",
                path, other
            ));
        }
    };
    Ok((info.width, info.height, pixels))
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_args_requires_supported_source() {
        assert!(validate_color_lut_args(&serde_json::json!({})).is_err());
        assert!(validate_color_lut_args(&serde_json::json!({"source": "g.tga"})).is_err());
        validate_color_lut_args(&serde_json::json!({"source": "g.cube"})).expect("ok");
    }

    #[test]
    fn validate_args_rejects_a_non_string_source() {
        let err = validate_color_lut_args(&serde_json::json!({"source": 5})).unwrap_err();
        assert!(err.contains("requires a `source` path"), "got: {err}");
    }

    // Encode an 8-bit PNG in memory and write it to `dir`.
    fn write_png(
        dir: &tempfile::TempDir,
        name: &str,
        width: u32,
        height: u32,
        color: png::ColorType,
        data: &[u8],
    ) -> String {
        let path = dir.path().join(name);
        let file = std::fs::File::create(&path).expect("create png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("png header");
        writer.write_image_data(data).expect("png data");
        writer.finish().expect("png finish");
        path.to_string_lossy().into_owned()
    }

    // A size-2 strip is 4x2: two 2x2 slices side by side. Each pixel encodes
    // its own (x, y) so the reorder into red-fastest texel order is checkable.
    fn strip2_pixels() -> Vec<u8> {
        let mut data = Vec::new();
        for y in 0..2u8 {
            for x in 0..4u8 {
                data.extend_from_slice(&[x * 10, y * 10, 42, 255]);
            }
        }
        data
    }

    #[test]
    fn parse_png_strip_reorders_slices_into_texel_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_png(&dir, "s.png", 4, 2, png::ColorType::Rgba, &strip2_pixels());
        let (size, data) = parse_png_strip(&src).expect("parse strip");
        assert_eq!(size, 2);
        assert_eq!(data.len(), 2 * 2 * 2 * 4);
        // Texel (r, g, b) comes from strip pixel (b*n + r, g).
        // (r=0, g=0, b=0) -> pixel (0, 0).
        assert_eq!(&data[0..4], &[0, 0, 42, 255]);
        // (r=0, g=0, b=1) is the 5th texel -> pixel (2, 0).
        assert_eq!(&data[4 * 4..4 * 4 + 4], &[20, 0, 42, 255]);
        // (r=1, g=1, b=1) is the last texel -> pixel (3, 1).
        assert_eq!(&data[28..32], &[30, 10, 42, 255]);
    }

    #[test]
    fn parse_png_strip_accepts_rgb_and_fills_alpha() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rgb: Vec<u8> = strip2_pixels()
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        let src = write_png(&dir, "s.png", 4, 2, png::ColorType::Rgb, &rgb);
        let (size, data) = parse_png_strip(&src).expect("parse strip");
        assert_eq!(size, 2);
        assert!(data.chunks_exact(4).all(|c| c[3] == 255));
    }

    #[test]
    fn parse_png_strip_rejects_a_wrong_width() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Height 2 demands width 4; give it 3.
        let src = write_png(&dir, "s.png", 3, 2, png::ColorType::Rgba, &[0u8; 3 * 2 * 4]);
        let err = parse_png_strip(&src).unwrap_err();
        assert!(err.contains("must be 4x2"), "got: {err}");
    }

    #[test]
    fn parse_png_strip_rejects_an_out_of_range_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Height 1 means LUT size 1, below the minimum of 2.
        let src = write_png(&dir, "s.png", 1, 1, png::ColorType::Rgba, &[0u8; 4]);
        let err = parse_png_strip(&src).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn parse_png_strip_rejects_grayscale() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_png(&dir, "s.png", 4, 2, png::ColorType::Grayscale, &[0u8; 8]);
        let err = parse_png_strip(&src).unwrap_err();
        assert!(err.contains("unsupported PNG colour type"), "got: {err}");
    }

    #[test]
    fn parse_png_strip_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing.png");
        let err = parse_png_strip(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("failed to open LUT source"), "got: {err}");
    }

    // .cube compile

    const CUBE_2: &str = "\
# identity-ish test cube
TITLE \"test\"
LUT_3D_SIZE 2
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
";

    fn write_cube(dir: &tempfile::TempDir, text: &str) -> String {
        let path = dir.path().join("t.cube");
        std::fs::write(&path, text).expect("write cube");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn compile_color_lut_payload_packs_a_cube_lut() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_cube(&dir, CUBE_2);
        let payload =
            compile_color_lut_payload(&serde_json::json!({"source": src})).expect("compile");
        // Header: magic "LUT3", size, format 0 (RGBA8), then 2^3 texels.
        assert_eq!(&payload[0..4], &0x3354554cu32.to_le_bytes());
        assert_eq!(&payload[4..8], &2u32.to_le_bytes());
        assert_eq!(&payload[8..12], &0u32.to_le_bytes());
        assert_eq!(payload.len(), 12 + 8 * 4);
        assert_eq!(&payload[12..16], &[0, 0, 0, 255]);
        assert_eq!(&payload[40..44], &[255, 255, 255, 255]);
    }

    #[test]
    fn compile_color_lut_payload_reports_a_missing_cube_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing.cube");
        let args = serde_json::json!({"source": path.to_str().unwrap()});
        let err = compile_color_lut_payload(&args).unwrap_err();
        assert!(err.contains("failed to read LUT source"), "got: {err}");
    }

    #[test]
    fn decode_source_dispatches_cube_and_png_by_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cube = write_cube(&dir, CUBE_2);
        let (size, data) = decode_source(&cube).expect("cube decode");
        assert_eq!(size, 2);
        assert_eq!(data.len(), 32);

        let strip = write_png(&dir, "s.png", 4, 2, png::ColorType::Rgba, &strip2_pixels());
        let (size, data) = decode_source(&strip).expect("png decode");
        assert_eq!(size, 2);
        assert_eq!(data.len(), 32);
    }

    #[test]
    fn decode_source_rejects_an_unknown_extension() {
        let err = decode_source("grade.tga").unwrap_err();
        assert!(err.contains("must be a .cube or .png"), "got: {err}");
    }
}
