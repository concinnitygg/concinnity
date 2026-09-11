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
    /// centered log-luminance by this to derive a bin index.
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

/// Per-dispatch params for the single-pass Hi-Z downsampler.
/// Matches `HizSpdParams` in `shaders/hiz_build.slang`. 16 bytes.
#[derive(Copy, Clone, PartialEq, Eq, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct HizSpdParams {
    /// Width of this dispatch's base level: mip 0 for phase 1, mip 6 for the tail.
    pub base_width: u32,
    /// Height of this dispatch's base level.
    pub base_height: u32,
    /// Levels this dispatch writes, counting the base as one.
    pub level_count: u32,
    /// MSAA sample count of the depth source. Unused by the tail.
    pub sample_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // Both Hi-Z param blocks are four tightly-packed uints, which is what the
    // shader's push-constant / root-constant block expects.
    #[test]
    fn hiz_params_layout_matches_shader() {
        assert_eq!(size_of::<HizParams>(), 16);
        assert_eq!(offset_of!(HizParams, dst_width), 0);
        assert_eq!(offset_of!(HizParams, dst_height), 4);
        assert_eq!(offset_of!(HizParams, src_mip), 8);
        assert_eq!(offset_of!(HizParams, sample_count), 12);
    }

    #[test]
    fn hiz_spd_params_layout_matches_shader() {
        assert_eq!(size_of::<HizSpdParams>(), 16);
        assert_eq!(offset_of!(HizSpdParams, base_width), 0);
        assert_eq!(offset_of!(HizSpdParams, base_height), 4);
        assert_eq!(offset_of!(HizSpdParams, level_count), 8);
        assert_eq!(offset_of!(HizSpdParams, sample_count), 12);
    }
}
