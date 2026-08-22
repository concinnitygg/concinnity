// Turns the image embedded in a panorama sphere into the linear-light
// equirectangular source the EnvironmentMap bake consumes.
//
// Range. A panorama shipped inside a `.glb` is a display-encoded PNG or JPEG,
// so its brightest value is white and nothing in it exceeds it. This decode
// reads it literally: the sRGB transfer curve is inverted so the values become
// scene-linear, and white lands at 1.0 radiance. Nothing invents range beyond
// that -- no inverse tonemapping, no guessed sun multiplier -- because a made-up
// range bakes an error into the irradiance and reflection cubemaps that no
// later setting can undo.
//
// What that costs. A Radiance `.hdr` carries a sun some thousands of times
// brighter than the sky around it, and the prefilter convolution turns that
// into a bright key light and a hot specular highlight. A panorama capped at
// white cannot: its image-based lighting is a faithful but low-contrast ambient
// whose irradiance never exceeds pi. That makes it an exact backdrop and a soft
// fill light, not a key light. `PostProcessConfig.ambient_intensity` scales the
// ambient term at render time and is the knob for raising it; the bake stays
// honest about what the file actually contains.
//
// Bit depth is preserved: 16-bit-per-channel PNGs, which these panoramas often
// are, keep their precision through the float conversion, so a dark sky
// gradient does not band.

use crate::gltf_source::GltfDoc;
use crate::hdr::HdrImage;

/// Pixel dimensions of a glTF image, read from its header without decoding the
/// pixel data. Detection uses this to check the panorama's aspect ratio, which
/// would otherwise cost a full decode of a multi-megabyte image.
pub(super) fn source_dimensions(doc: &GltfDoc, image_index: u32) -> Result<(u32, u32), String> {
    let (bytes, mime) = doc.image_bytes(image_index)?;
    match mime.as_deref() {
        Some("image/png") | None => png_dimensions(&bytes),
        Some("image/jpeg") => jpeg_dimensions(&bytes),
        Some(other) => Err(format!(
            "image {} has unsupported MIME type '{}'; a panorama must be PNG or JPEG",
            image_index, other
        )),
    }
}

// Decode the panorama image into a linear-light equirectangular image. See
// the module header for the range this assumes.
pub(crate) fn load_equirect(
    doc: &GltfDoc,
    source: &str,
    image_index: u32,
) -> Result<HdrImage, String> {
    let (bytes, mime) = doc
        .image_bytes(image_index)
        .map_err(|e| format!("'{}': {}", source, e))?;
    match mime.as_deref() {
        Some("image/png") | None => decode_png(bytes),
        Some("image/jpeg") => decode_jpeg(bytes),
        Some(other) => Err(format!(
            "image {} has unsupported MIME type '{}'; a panorama must be PNG or JPEG",
            image_index, other
        )),
    }
    .map_err(|e| format!("'{}': {}", source, e))
}

// Width of one decoded sample. A 4K panorama is eight million pixels, so the
// decode reads straight from the container's bytes into the final linear-light
// pixels: an intermediate normalised buffer would be another 130 MB live at
// once, on top of the 100 MB the result already costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleDepth {
    Eight,
    Sixteen,
}

impl SampleDepth {
    fn bytes(self) -> usize {
        match self {
            Self::Eight => 1,
            Self::Sixteen => 2,
        }
    }

    // Normalise one channel of a pixel to 0..1.
    fn read(self, pixel: &[u8], channel: usize) -> f32 {
        match self {
            Self::Eight => pixel[channel] as f32 / u8::MAX as f32,
            Self::Sixteen => {
                let offset = channel * 2;
                u16::from_be_bytes([pixel[offset], pixel[offset + 1]]) as f32 / u16::MAX as f32
            }
        }
    }
}

// The standard sRGB EOTF (IEC 61966-2-1), piecewise so the near-black segment
// stays linear rather than crushing.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn png_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    let reader = png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .map_err(|e| format!("failed to read PNG info: {}", e))?;
    let info = reader.info();
    Ok((info.width, info.height))
}

fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    decoder
        .read_info()
        .map_err(|e| format!("failed to read JPEG info: {}", e))?;
    let info = decoder
        .info()
        .ok_or_else(|| "JPEG has no info after header read".to_string())?;
    Ok((info.width as u32, info.height as u32))
}

// Decode a PNG of any colour type or bit depth. Palette and sub-byte depths
// are expanded; 16-bit stays 16-bit so a smooth sky keeps its precision.
fn decode_png(bytes: Vec<u8>) -> Result<HdrImage, String> {
    use png::{BitDepth, ColorType};

    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("failed to read PNG info: {}", e))?;
    let mut buffer = vec![
        0u8;
        reader
            .output_buffer_size()
            .ok_or("failed to compute PNG output buffer size")?
    ];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|e| format!("failed to decode PNG frame: {}", e))?;

    let channels = match info.color_type {
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        other => {
            return Err(format!(
                "unsupported PNG colour type {:?}; convert the panorama to RGB or RGBA",
                other
            ));
        }
    };
    let depth = match info.bit_depth {
        BitDepth::Sixteen => SampleDepth::Sixteen,
        _ => SampleDepth::Eight,
    };

    Ok(HdrImage {
        width: info.width,
        height: info.height,
        pixels: gather_linear_rgb(&buffer[..info.buffer_size()], channels, depth),
    })
}

fn decode_jpeg(bytes: Vec<u8>) -> Result<HdrImage, String> {
    use jpeg_decoder::PixelFormat;

    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let raw = decoder
        .decode()
        .map_err(|e| format!("failed to decode JPEG: {}", e))?;
    let info = decoder
        .info()
        .ok_or_else(|| "JPEG has no info after decode".to_string())?;
    let channels = match info.pixel_format {
        PixelFormat::RGB24 => 3,
        PixelFormat::L8 => 1,
        other => {
            return Err(format!(
                "unsupported JPEG pixel format {:?}; convert the panorama to RGB",
                other
            ));
        }
    };
    Ok(HdrImage {
        width: info.width as u32,
        height: info.height as u32,
        pixels: gather_linear_rgb(&raw, channels, SampleDepth::Eight),
    })
}

// Pack interleaved samples into linear-light RGB triples: grey broadcasts,
// alpha drops (a panorama is opaque by construction), and the sRGB transfer
// curve inverts so the values become scene radiance. The renderer's output
// pass re-encodes after tonemapping, so linearising here is what makes the
// displayed sky match the source image.
fn gather_linear_rgb(raw: &[u8], channels: usize, depth: SampleDepth) -> Vec<[f32; 3]> {
    raw.chunks_exact(channels * depth.bytes())
        .map(|px| match channels {
            1 | 2 => {
                let v = srgb_to_linear(depth.read(px, 0));
                [v, v, v]
            }
            _ => [
                srgb_to_linear(depth.read(px, 0)),
                srgb_to_linear(depth.read(px, 1)),
                srgb_to_linear(depth.read(px, 2)),
            ],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::detect::test_fixtures::{
        PanoramaShape, panorama_glb_with, panorama_png, panorama_png16,
    };
    use super::*;

    fn doc_with(png: Vec<u8>) -> GltfDoc {
        let glb = panorama_glb_with(PanoramaShape {
            png,
            ..Default::default()
        });
        GltfDoc::from_slice(&glb, None, "test.glb").expect("parse")
    }

    #[test]
    fn source_dimensions_reads_a_png_header_without_decoding() {
        let doc = doc_with(panorama_png(64, 32, 200));
        assert_eq!(source_dimensions(&doc, 0).unwrap(), (64, 32));
    }

    #[test]
    fn source_dimensions_reports_a_missing_image() {
        let doc = doc_with(panorama_png(4, 2, 200));
        let err = source_dimensions(&doc, 7).unwrap_err();
        assert!(err.contains("image_index 7 is out of range"), "got: {err}");
    }

    #[test]
    fn an_eight_bit_panorama_decodes_to_linear_light() {
        // 128/255 is mid-grey on the sRGB curve, which is far darker than 0.5
        // once linearised. Getting this wrong washes the sky out.
        let doc = doc_with(panorama_png(4, 2, 128));
        let image = load_equirect(&doc, "test.glb", 0).expect("decode");
        assert_eq!((image.width, image.height), (4, 2));
        assert_eq!(image.pixels.len(), 8);
        let expected = srgb_to_linear(128.0 / 255.0);
        for px in &image.pixels {
            assert!((px[0] - expected).abs() < 1e-5, "got {:?}", px);
            assert_eq!(px[0], px[2]);
        }
        assert!(expected < 0.25, "sRGB mid-grey linearises well below 0.5");
    }

    #[test]
    fn a_sixteen_bit_panorama_keeps_its_precision() {
        // A value that is not representable in 8 bits: truncating to a byte
        // would land on a visibly different linear value.
        let raw = 30001u16;
        let doc = doc_with(panorama_png16(4, 2, raw));
        let image = load_equirect(&doc, "test.glb", 0).expect("decode");
        let expected = srgb_to_linear(raw as f32 / u16::MAX as f32);
        assert!(
            (image.pixels[0][0] - expected).abs() < 1e-6,
            "got {:?}, want {}",
            image.pixels[0],
            expected
        );
        let truncated = srgb_to_linear((raw >> 8) as f32 / u8::MAX as f32);
        assert!(
            (image.pixels[0][0] - truncated).abs() > 1e-6,
            "a 16-bit source must not decode as if it were 8-bit"
        );
    }

    #[test]
    fn white_decodes_to_one_and_black_to_zero() {
        let doc = doc_with(panorama_png(4, 2, 255));
        let white = load_equirect(&doc, "test.glb", 0).expect("decode");
        assert!((white.pixels[0][0] - 1.0).abs() < 1e-6);

        let doc = doc_with(panorama_png(4, 2, 0));
        let black = load_equirect(&doc, "test.glb", 0).expect("decode");
        assert_eq!(black.pixels[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_corrupt_image_is_reported_against_the_source() {
        let doc = doc_with(b"not a png".to_vec());
        let err = load_equirect(&doc, "sky.glb", 0).unwrap_err();
        assert!(err.starts_with("'sky.glb': "), "got: {err}");
        assert!(err.contains("PNG"), "got: {err}");
    }

    #[test]
    fn srgb_to_linear_matches_the_standard_curve() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        // The piecewise knee: below 0.04045 the curve is a plain divide.
        assert!((srgb_to_linear(0.04) - 0.04 / 12.92).abs() < 1e-9);
        assert!((srgb_to_linear(0.5) - 0.21404).abs() < 1e-4);
    }

    #[test]
    fn gather_linear_rgb_broadcasts_grey_and_drops_alpha() {
        let grey = srgb_to_linear(128.0 / 255.0);
        assert_eq!(
            gather_linear_rgb(&[128, 40], 2, SampleDepth::Eight),
            vec![[grey, grey, grey]],
            "grey broadcasts and alpha drops"
        );
        assert_eq!(
            gather_linear_rgb(&[128], 1, SampleDepth::Eight),
            vec![[grey, grey, grey]]
        );
        assert_eq!(
            gather_linear_rgb(&[0, 128, 255, 40], 4, SampleDepth::Eight),
            vec![[0.0, grey, 1.0]]
        );
    }

    #[test]
    fn sample_depth_reads_both_widths_from_the_same_channel_index() {
        // RGB, second channel: one byte apart at 8-bit, two at 16-bit.
        assert_eq!(SampleDepth::Eight.read(&[0, 255, 0], 1), 1.0);
        assert_eq!(SampleDepth::Sixteen.read(&[0, 0, 255, 255, 0, 0], 1), 1.0);
        assert_eq!(SampleDepth::Eight.bytes(), 1);
        assert_eq!(SampleDepth::Sixteen.bytes(), 2);
    }
}
