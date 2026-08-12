// src/font.rs
//
// Font asset compilation: resolves a `Font`'s arguments to TTF bytes (a file on
// disk, or the bundled default face) and hands them to the shared rasteriser in
// `concinnity-font`, which packs the glyphs into an SDF atlas payload.

use serde::Deserialize;

use concinnity_core::assets::Font;

/// Source filename of the bundled default face. Companion injection derives the
/// auto-injected Font asset's name from it, so a generated default font is named
/// exactly as `cn add` would name the same file.
pub const BUILTIN_FONT_FILE: &str = concinnity_font::BUILTIN_FONT_FILE;

// Compile a Font asset's arguments into the binary blob payload format.
//
// When `path` is empty or absent the engine's bundled default font is used
// instead of reading from disk, so no external file is required.
pub fn compile_font_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let font: Font =
        Deserialize::deserialize(args).map_err(|e| format!("Font: invalid args: {}", e))?;
    let path = font.path.as_str();

    let ttf_bytes: Vec<u8> = if path.is_empty() {
        concinnity_font::BUILTIN_FONT_BYTES.to_vec()
    } else {
        std::fs::read(path).map_err(|e| format!("Font: could not read '{}': {}", path, e))?
    };
    let source = if path.is_empty() { "<built-in>" } else { path };

    concinnity_font::compile(&ttf_bytes, font.size_px, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cpu::build::font::deserialise;

    // An empty or absent `path` compiles the bundled default font at the default
    // 48px size. Before the atlas layout sizes were widened to u32, the high-res
    // glyph stride times the glyph count overflowed u16 and panicked in debug
    // builds at this size.
    #[test]
    fn builtin_font_compiles_at_default_size() {
        for args in [
            serde_json::json!({ "size_px": 48 }),
            serde_json::json!({ "path": "", "size_px": 48 }),
        ] {
            let payload = compile_font_payload(&args).expect("compile bundled font at 48px");
            let (w, h, _supersample, _size_px, rgba, metrics) = deserialise(&payload).unwrap();
            assert!(w > 0 && h > 0, "atlas has non-zero dimensions");
            assert!(!rgba.is_empty());
            assert!(!metrics.is_empty());
        }
    }

    #[test]
    fn font_compiles_from_a_ttf_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bundled.ttf");
        std::fs::write(&path, concinnity_font::BUILTIN_FONT_BYTES).expect("write ttf");

        let args = serde_json::json!({ "path": path.to_str().unwrap(), "size_px": 12 });
        let payload = compile_font_payload(&args).expect("compile font read from disk");
        let (w, h, _supersample, size_px, rgba, metrics) = deserialise(&payload).unwrap();
        assert_eq!(size_px, 12);
        // Every printable ASCII glyph (32..=126) is rasterised.
        assert_eq!(metrics.len(), 95);
        assert_eq!(metrics[0].char_code, ' ' as u32);
        assert_eq!(metrics[94].char_code, '~' as u32);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        // 'A' occupies real atlas texels and advances the pen.
        let a = metrics
            .iter()
            .find(|m| m.char_code == 'A' as u32)
            .expect("'A' is rasterised");
        assert!(a.atlas_w > 0 && a.atlas_h > 0);
        assert!(a.advance_px > 0.0);
    }

    #[test]
    fn font_reports_a_missing_ttf() {
        let args = serde_json::json!({ "path": "/no/such/font.ttf", "size_px": 12 });
        let err = compile_font_payload(&args).unwrap_err();
        assert!(err.contains("could not read"), "got: {err}");
        assert!(err.contains("/no/such/font.ttf"), "got: {err}");
    }

    #[test]
    fn font_rejects_a_file_that_is_not_a_ttf() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bogus.ttf");
        std::fs::write(&path, b"not a font at all").expect("write junk");

        let args = serde_json::json!({ "path": path.to_str().unwrap(), "size_px": 12 });
        let err = compile_font_payload(&args).unwrap_err();
        assert!(err.contains("failed to parse"), "got: {err}");
    }

    #[test]
    fn font_rejects_args_that_are_not_a_font() {
        let err = compile_font_payload(&serde_json::json!({ "size_px": "big" })).unwrap_err();
        assert!(err.contains("Font: invalid args"), "got: {err}");
    }
}
