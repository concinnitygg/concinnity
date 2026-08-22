// 3D colour-grading lookup-table schema.

use crate::{AssetId, PayloadLocator};
use alloc::string::String;

/// A 3D colour-grading lookup table applied as a final post-process step. The
/// build bakes the source into a colour cube; the graded result is blended over
/// the image by [PostProcessConfig](#postprocessconfig)'s `lut_strength`.
///
/// A world declares at most one `ColorLut`; the first wins. When none is
/// present, colour grading is skipped regardless of `lut_strength`.
///
/// Two source formats are accepted, picked by file extension:
///   - `.cube`  Adobe Cube LUT (plain-text interchange format).
///   - `.png`   A horizontal slice strip: `(n*n)` wide by `n` tall.
///
/// ```rust
/// # use concinnity_asset::ColorLut;
/// ColorLut {
///     source: "luts/cinematic_warm.cube".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ColorLut {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source `.cube` or `.png` LUT file.
    pub source: String,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_lut_names_no_source() {
        let l = ColorLut::default();
        assert!(l.source.is_empty());
        assert_eq!(l.asset_id, AssetId::default());
        assert!(l.locator.is_none());
    }

    #[test]
    fn the_source_path_is_the_only_authored_field() {
        let l: ColorLut = serde_json::from_str(r#"{"source":"grade/warm.cube"}"#).unwrap();
        assert_eq!(l.source, "grade/warm.cube");
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"source":"grade/warm.cube"}"#
        );

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: ColorLut = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source, "grade/warm.cube");
        assert!(back.locator.is_none());
    }
}
