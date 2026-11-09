// What the GPU reports of a frame's luminance for exposure to adapt to.

/// What the auto-exposure average kernel measures of a frame, read back by
/// the CPU. Matches `ExposureMeasurement` in `shaders/auto_exposure.hlsl`.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct ExposureMeasurement {
    /// Mean log2 luminance of the pixels above the histogram floor and within
    /// [`METERED_RANGE_EV`](super::METERED_RANGE_EV) of
    /// the highlight.
    pub mean_log_lum: f32,
    /// Log2 luminance the brightest
    /// [`HIGHLIGHT_FRACTION`](super::HIGHLIGHT_FRACTION)
    /// of all pixels starts at.
    pub highlight_log_lum: f32,
    /// Mean log2 luminance of every pixel, those under the histogram floor
    /// read at the floor: what an eye taking in the whole view adapts to.
    pub view_log_lum: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;

    #[test]
    fn exposure_measurement_layout_matches_shader() {
        assert_eq!(size_of::<ExposureMeasurement>(), 12);
        assert_eq!(offset_of!(ExposureMeasurement, mean_log_lum), 0);
        assert_eq!(offset_of!(ExposureMeasurement, highlight_log_lum), 4);
        assert_eq!(offset_of!(ExposureMeasurement, view_log_lum), 8);
    }
}
