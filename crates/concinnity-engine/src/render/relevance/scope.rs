//! Which part of the screen the reflections show on.

use crate::gfx::render_types::ReflectionScopeGpu;
use crate::render::view::ScreenRect;

/// How many pixels past the prop's part of the screen its reflections fall
/// to nothing over, so the edge between a resolved reflection and the forward
/// probe one never shows a seam.
pub(super) const PROP_RAMP_PIXELS: f32 = 16.0;

/// The share of each pixel the reflections show on: `whole` everywhere, and
/// `prop` within `rect`, the part of the screen the prop can cover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionScope {
    /// The share everywhere on screen.
    pub whole: f32,
    /// The share within `rect`.
    pub prop: f32,
    /// Where on screen the prop can show.
    pub rect: ScreenRect,
}

impl ReflectionScope {
    /// No reflection anywhere.
    pub const NONE: Self = Self {
        whole: 0.0,
        prop: 0.0,
        rect: ScreenRect::EMPTY,
    };

    /// Reflections in full everywhere.
    pub const FULL: Self = Self {
        whole: 1.0,
        ..Self::NONE
    };

    /// The share at `uv`, as the passes reading [`Self::gpu`] take it, for a
    /// screen `size` pixels across.
    #[cfg(test)]
    pub(crate) fn share(&self, uv: [f32; 2], size: [u32; 2]) -> f32 {
        let gpu = self.gpu(size);
        if gpu.prop <= 0.0 {
            return gpu.whole;
        }
        let past = (0..2)
            .map(|k| (gpu.rect[k] - uv[k]).max(uv[k] - gpu.rect[k + 2]).max(0.0) * gpu.ramp[k])
            .fold(0.0_f32, f32::max);
        gpu.whole.max(gpu.prop * (1.0 - past).clamp(0.0, 1.0))
    }

    /// The block the passes read it through, for a screen `size` pixels
    /// across.
    pub fn gpu(&self, size: [u32; 2]) -> ReflectionScopeGpu {
        if self.prop <= 0.0 || self.rect.is_empty() {
            return ReflectionScopeGpu {
                whole: self.whole,
                ..ReflectionScopeGpu::NONE
            };
        }
        ReflectionScopeGpu {
            rect: [
                self.rect.min[0],
                self.rect.min[1],
                self.rect.max[0],
                self.rect.max[1],
            ],
            whole: self.whole,
            prop: self.prop,
            ramp: size.map(|s| s.max(1) as f32 / PROP_RAMP_PIXELS),
        }
    }
}

impl Default for ReflectionScope {
    /// No reflection anywhere.
    fn default() -> Self {
        Self::NONE
    }
}
