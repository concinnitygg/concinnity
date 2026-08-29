//! The small parameter blocks the fullscreen post passes and the two compute
//! helpers that feed them take.

/// Input to the TAA resolve fragment shader. Matches `TaaParams` in
/// `shaders/taa.slang`. 4 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct TaaParams {
    /// 0 on the first frame / after a resize, 1.0 otherwise.
    pub history_valid: f32,
}

/// Input to the auto-exposure histogram kernels: the three luminance-mapping
/// scalars then a pad rounding to 16 bytes. Matches `AutoExposureParams` in
/// `shaders/auto_exposure.slang`.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct AutoExposureParams {
    /// Lowest log2(luminance) the histogram covers.
    pub lum_log2_min: f32,
    /// Width of the log2(luminance) span the histogram covers (max - min).
    pub lum_log2_range: f32,
    /// `HISTOGRAM_BINS / lum_log2_range`. The build kernel multiplies the
    /// centred log-luminance by this to derive a bin index.
    pub lum_to_bin_scale: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: f32,
}

/// Per-dispatch params for the Hi-Z build kernels: four tightly-packed uints.
/// Matches `HizParams` in `shaders/hiz_build.slang`. 16 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct HizParams {
    /// Destination width in pixels.
    pub dst_width: u32,
    /// Destination height in pixels.
    pub dst_height: u32,
    /// Source mip level sampled.
    pub src_mip: u32,
    /// MSAA sample count of the source.
    pub sample_count: u32,
}
