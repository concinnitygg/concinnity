// Display-output schema.

/// How the finished frame is encoded for the display: standard dynamic range,
/// extended-range HDR, or PQ (HDR10). The `display` section of
/// [GraphicsConfig](#graphicsconfig); left unset, the output is standard
/// dynamic range.
///
/// ```rust
/// # use concinnity_core::components::{DisplayOverrides, GraphicsConfig};
/// GraphicsConfig {
///     display: DisplayOverrides {
///         hdr: Some(true),
///         ..Default::default()
///     },
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
#[asset(section)]
#[serde(default)]
pub struct Display {
    /// HDR display output. On a capable display, emits extended-range HDR
    /// instead of the standard tonemapped output. Falls back to standard output
    /// when the display or platform doesn't support HDR. Auto-exposure meters
    /// toward perceptual middle gray while it is on.
    #[asset(apply = restart)]
    pub hdr: bool,
    /// PQ (HDR10) output. When true, and `hdr` is on, and the display has HDR
    /// headroom, output is PQ-encoded for HDR10 panels. No effect when `hdr` is
    /// off.
    #[asset(apply = restart)]
    pub pq: bool,
}
