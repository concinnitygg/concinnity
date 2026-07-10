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
/// ```jsonl
/// {"name":"grade","type":"ColorLut","args":{"source":"luts/cinematic_warm.cube"}}
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
