// Reflection schema: screen-space and ray-traced.

use crate::components::PassResolution;

/// Reflections on glossy surfaces (water, polished floors): screen-space,
/// hardware ray-traced, or neither. The `reflections` section of
/// [`GraphicsConfig`](#graphicsconfig); a field left unset takes the default
/// below.
///
/// Both methods are on by default, which is the look on capable hardware. The
/// `Auto` graphics quality preset forces screen-space reflections off below
/// the high tier and permits ray tracing only on the top tier; where ray
/// tracing is not permitted or the GPU lacks it, `screen_space` takes over. A
/// ceiling only ever reduces, so turning a method off here holds on every tier.
///
/// ```rust
/// # use concinnity_core::components::{GraphicsConfig, ReflectionsOverrides};
/// GraphicsConfig {
///     reflections: ReflectionsOverrides {
///         ray_traced: Some(false),
///         max_distance: Some(60.0),
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
pub struct Reflections {
    /// Screen-space reflections: mix reflected scene color over glossy
    /// surfaces, from what is already on screen.
    #[asset(default = true, apply = rebuild)]
    pub screen_space: bool,
    /// Hardware ray-traced reflections. When the GPU supports ray tracing,
    /// traces real reflection rays so off-screen geometry still appears.
    /// Takes precedence over `screen_space`, falling back to it where ray
    /// tracing isn't available.
    #[asset(default = true, apply = rebuild)]
    pub ray_traced: bool,
    /// Reflection blend strength. Scales the Fresnel-weighted reflection mixed
    /// over the base shading.
    #[asset(default = 0.7, min = 0.0, max = 1.0, apply = live)]
    pub intensity: f32,
    /// How far a reflection reaches, in world units. Longer reaches catch more
    /// distant reflections, more coarsely.
    #[asset(default = 40.0, min = 1.0, max = 1000.0, apply = live)]
    pub max_distance: f32,
    /// Internal resolution the ray-traced reflection rays are cast at. `half`
    /// (default) traces a quarter of the pixels and upsamples with a depth- and
    /// normal-aware filter; `full` traces every pixel for the sharpest mirrors;
    /// `quarter` is the cheapest. Only matters when `ray_traced` is on.
    #[asset(apply = rebuild)]
    pub ray_traced_resolution: PassResolution,
    /// Whether surfaces seen in a ray-traced reflection are shadowed from the
    /// sun. Each reflected hit then casts a second ray toward the sun, which
    /// roughly doubles the trace cost; off lights every reflected surface as if
    /// the sun reached it. Only matters when `ray_traced` is on.
    #[asset(default = true, apply = rebuild)]
    pub ray_traced_shadows: bool,
    /// Internal resolution of the roughness-aware reflection blur. `half`
    /// (default) blurs at a quarter of the pixels for a large saving and
    /// bilinearly upsamples; `full` blurs at native resolution; `quarter` is
    /// the cheapest. Smooth mirror surfaces stay sharp at any setting (the
    /// composite keeps the sharp reflection for low roughness).
    #[asset(apply = rebuild)]
    pub blur_resolution: PassResolution,
}

impl Reflections {
    /// Per-axis divisor for the roughness-aware reflection blur target, resolved
    /// from `blur_resolution`. Always at least 1.
    pub fn blur_divisor(&self) -> u32 {
        self.blur_resolution.scale_divisor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_divisor_follows_the_resolution() {
        let cfg = Reflections {
            blur_resolution: PassResolution::Quarter,
            ..Default::default()
        };
        assert_eq!(cfg.blur_divisor(), 4);
    }
}
