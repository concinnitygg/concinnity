//! The depth-bias raster state every shadow pass binds, on every host.
//!
//! The sample side owns per-cascade growth (`shadow_bias.slang`); the raster
//! side contributes slope alone, and does not vary by cascade or between the
//! cascade and spot passes.
//!
//! Slope scale and clamp are the two bias terms Metal, Vulkan and D3D12 define
//! identically: each multiplies the primitive's maximum depth slope per pixel
//! by the scale and adds it in NDC depth units, then clamps the sum in the
//! same units. The constant term is not portable -- Vulkan and D3D12 scale it
//! by an implementation-defined resolution derived from the primitive's
//! exponent for a float depth format, Metal's documentation gives no such
//! scaling -- so the same literal can mean offsets seven orders of magnitude
//! apart on the D32 float shadow maps all three hosts use. It is zero here
//! rather than converted per host.

/// Constant depth bias. Zero: see the module note.
pub const RASTER_CONSTANT: f32 = 0.0;

/// Slope-scale factor, in NDC depth per unit depth slope.
pub const RASTER_SLOPE: f32 = 2.0;

/// Upper bound on the summed bias, in NDC depth units. Keeps a triangle nearly
/// edge-on to the light, where the slope term diverges, from pushing its
/// casters far enough to detach their contact shadows.
pub const RASTER_CLAMP: f32 = 0.01;
