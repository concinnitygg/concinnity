// 3D color-grading lookup-table schema.

use crate::ecs::PayloadLocator;
use alloc::string::String;

/// A 3D color-grading lookup table applied as a final post-process step. The
/// build bakes the source into a color cube; the graded result is blended over
/// the image by `strength`.
///
/// A world declares at most one `ColorLut`; the first wins. When none is
/// present, color grading is skipped.
///
/// Two source formats are accepted, picked by file extension:
///   - `.cube`  Adobe Cube LUT (plain-text interchange format).
///   - `.png`   A horizontal slice strip: `(n*n)` wide by `n` tall.
///
/// ```rust
/// # use concinnity_core::components::ColorLut;
/// ColorLut {
///     source: "luts/cinematic_warm.cube".into(),
///     strength: 0.8,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct ColorLut {
    /// Path to the source `.cube` or `.png` LUT file.
    pub source: String,
    /// Blend in `[0, 1]`: mixes the graded color over the ungraded one by this
    /// amount.
    #[asset(default = 1.0, min = 0.0, max = 1.0, apply = live)]
    pub strength: f32,
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
        assert!(l.locator.is_none());
    }

    #[test]
    fn the_source_and_strength_are_the_authored_fields() {
        let l: ColorLut = serde_json::from_str(r#"{"source":"grade/warm.cube"}"#).unwrap();
        assert_eq!(l.source, "grade/warm.cube");
        assert_eq!(l.strength, 1.0);
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"source":"grade/warm.cube","strength":1.0}"#
        );

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: ColorLut = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source, "grade/warm.cube");
        assert!(back.locator.is_none());
    }
}
