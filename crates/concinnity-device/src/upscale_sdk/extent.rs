//! The render-to-output resolution split every upscaler is created for.

use std::fmt;

/// The render resolution an upscaler reads, the output resolution it
/// reconstructs, and the per-axis ratio between them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct UpscaleExtent {
    pub(crate) render: (u32, u32),
    pub(crate) output: (u32, u32),
    pub(crate) scale: f32,
}

impl UpscaleExtent {
    /// The split for `output` at the requested per-axis `scale`. The temporal
    /// kernels reconstruct at most 3x per axis, so the scale is clamped into
    /// `[1/3, 1]`; a non-positive scale means native resolution, where the
    /// upscaler runs as anti-aliasing only.
    pub(crate) fn resolve(output: (u32, u32), scale: f32) -> Self {
        let scale = if scale > 0.0 {
            scale.clamp(1.0 / 3.0, 1.0)
        } else {
            1.0
        };
        let axis = |size: u32| (((size as f32) * scale).round() as u32).max(1);
        Self {
            render: (axis(output.0), axis(output.1)),
            output,
            scale,
        }
    }
}

impl fmt::Display for UpscaleExtent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ((rw, rh), (ow, oh)) = (self.render, self.output);
        write!(
            f,
            "render {rw}x{rh} -> upscale {ow}x{oh} (scale {:.3})",
            self.scale
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_size_applies_the_quality_scale() {
        let e = UpscaleExtent::resolve((1920, 1080), 2.0 / 3.0);
        assert_eq!(e.render, (1280, 720));
        assert_eq!(e.output, (1920, 1080));
        assert!((e.scale - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(UpscaleExtent::resolve((1920, 1080), 0.5).render, (960, 540));
    }

    #[test]
    fn out_of_range_scales_are_clamped() {
        let e = UpscaleExtent::resolve((800, 600), 2.0);
        assert_eq!((e.render, e.scale), ((800, 600), 1.0));
        let e = UpscaleExtent::resolve((800, 600), 0.0);
        assert_eq!((e.render, e.scale), ((800, 600), 1.0));
        let e = UpscaleExtent::resolve((900, 900), 0.1);
        assert_eq!(e.render, (300, 300));
        assert!((e.scale - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn displays_both_sizes_and_the_scale() {
        let e = UpscaleExtent::resolve((1920, 1080), 0.5);
        assert_eq!(
            e.to_string(),
            "render 960x540 -> upscale 1920x1080 (scale 0.500)"
        );
    }

    #[test]
    fn render_size_is_never_zero() {
        assert_eq!(UpscaleExtent::resolve((1, 1), 1.0 / 3.0).render, (1, 1));
    }
}
