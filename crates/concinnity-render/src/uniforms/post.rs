// src/uniforms/post.rs
//
// The small parameter blocks the fullscreen post passes and the two compute
// helpers that feed them take.

// Input to the TAA resolve fragment shader. Matches `TaaParams` in
// `shaders/taa.slang`. 4 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TaaParams {
    // 0 on the first frame / after a resize, 1.0 otherwise.
    pub history_valid: f32,
}

// Input to the auto-exposure histogram kernels: the three luminance-mapping
// scalars then a pad rounding to 16 bytes. Matches `AutoExposureParams` in
// `shaders/auto_exposure.slang`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AutoExposureParams {
    // Lowest log2(luminance) the histogram covers.
    pub lum_log2_min: f32,
    // Width of the log2(luminance) span the histogram covers (max - min).
    pub lum_log2_range: f32,
    // `HISTOGRAM_BINS / lum_log2_range`. The build kernel multiplies the
    // centred log-luminance by this to derive a bin index.
    pub lum_to_bin_scale: f32,
    pub _pad: f32,
}

// Per-dispatch params for the Hi-Z build kernels: four tightly-packed uints.
// Matches `HizParams` in `shaders/hiz_build.slang`. 16 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct HizParams {
    pub dst_width: u32,
    pub dst_height: u32,
    pub src_mip: u32,
    pub sample_count: u32,
}
