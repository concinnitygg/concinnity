//! Volumetric clouds: the `Cloud` layers an [`Atmosphere`](crate::components::Atmosphere)
//! names, resolved into the numbers the cloud shaders march, and the CPU
//! halves of what they compute.
//!
//! A cloud layer is a shell of air between two heights over the ground. Its
//! density at a point is a weather field (where clouds are, and how tall they
//! grow) times a height profile, carved by a tiling volume of base-shape noise
//! and eroded at its edges by a finer detail noise. The noise volume is built
//! once on the GPU when the world loads. Each frame the clouds are marched at
//! half resolution against the scene's depth, lit by the sun and the moon
//! through the air with the planet's daylight curves, hazed by the air in
//! front of them, and blended over the scene once its see-through surfaces
//! are drawn. The atmosphere's pass also writes a shadow map of the light
//! surviving them over the ground around the camera, which the shaded passes
//! read through the cascades' shadow, and a panorama of them around the camera
//! the sky's cubes are drawn behind.
//!
//! Positions are kilometers about the planet's center, in the planet's own
//! fixed axes, so a cloud stays over the same ground through every frame
//! move. The shader side is `shaders/clouds_common.hlsl` (density and
//! lighting), the `CLOUD_MARCH` kernel in `shaders/atmosphere.hlsl` and
//! `shaders/clouds.hlsl` (the noise kernel and the composite).

mod frame;
mod noise;
mod params;
mod settings;
mod shadow;
mod shape;

pub use frame::{CloudFrame, CloudLayerFrame, CloudView};
pub use noise::NOISE_SIZE;
pub use params::{CloudLayerBlock, CloudMarch, CloudParams, block_pixel};
pub use settings::{CloudLayer, CloudSettings, MAX_CLOUD_LAYERS};
pub use shadow::{CloudShadowSquare, SHADOW_SIZE};
pub use shape::{CloudShell, coverage_density, height_profile, shell_span, weather_cover};

/// The resolution the clouds are marched at, as a divisor of the scene's.
pub const CLOUD_TARGET_DIVISOR: u32 = 2;

/// Texels across and down the panorama of the clouds around the camera the
/// sky's cubes are drawn over. `CLOUD_PANORAMA_WIDTH` and
/// `CLOUD_PANORAMA_HEIGHT` in `cloud_types.hlsl`.
pub const PANORAMA_SIZE: [u32; 2] = [256, 128];

/// The size of the target the clouds of a `width` x `height` scene are
/// marched into.
pub fn cloud_target_size(width: u32, height: u32) -> [u32; 2] {
    [
        width.div_ceil(CLOUD_TARGET_DIVISOR).max(1),
        height.div_ceil(CLOUD_TARGET_DIVISOR).max(1),
    ]
}

/// A layer of clouds in the sky: fair-weather cumulus, a gray stratus sheet,
/// towering heaps or anything between, drawn as a volume that can be seen
/// from below, flown through and seen from above.
///
/// The layer fills the air between `base` and `base + thickness` meters over
/// the ground (over the sea level of a [`Planet`](#planet)). Where clouds form
/// in it is a weather pattern whose cells are about `feature_size` meters
/// apart: `coverage` is the share of the sky they cover, from clear at 0
/// through scattered and broken clouds to overcast at 1. `towering` is how
/// tall each cloud grows inside the layer: 0 draws flat sheets, around 0.5
/// rounded heaps with flat bases, and 1 heaps that tower to the top of the
/// layer. Their edges are worn into wisps about `detail_size` meters across.
///
/// Clouds are lit by the sun through the air, so they are white at noon,
/// golden and pink at sunset, and stay lit a while after the ground below
/// them has gone dark, since they stand higher. Light passing through a cloud
/// is scattered on through it, which brightens their sunlit edges and gives
/// a silver lining when the sun is behind them, while the cores of thick
/// clouds go gray. The sky lights them from above and the ground from below.
/// By night a [`NightSky`](#nightsky)'s moon lights them. The air between the
/// camera and a cloud hazes it like any distant surface.
///
/// Clouds shade the ground, the grass and the sea beneath them from the sun
/// (or the moon), soft-edged and moving with them, and they stand over the
/// sea as over the land. Shiny surfaces and the water reflect them, and the
/// sky's light on the scene is the light of a sky with them in it, so an
/// overcast sky dims and grays the shade.
///
/// `density` is how much a cloud dims the light per meter at its densest:
/// higher draws harder, brighter edged clouds and darker cores.
///
/// Clouds drift with the [`Wind`](#wind), at `wind_drift` times its speed,
/// since winds blow faster aloft, and slowly change shape, each taking about
/// `evolve_seconds` to become a different cloud. Without a wind they hold
/// still. `stretch` draws them out along the wind: a thin, high layer
/// stretched five times or more becomes streaks of cirrus. On a planet each
/// cloud stays over the same ground, wherever on it the camera roams, so the
/// clouds seen from orbit are the ones seen from the ground. `seed` places
/// them: each seed is a different sky.
///
/// An [`Atmosphere`](#atmosphere) draws the layers its `clouds` names, at most
/// two, one wholly above the other; a layer nothing names is not drawn.
///
/// ```rust
/// # use concinnity_engine::components::Cloud;
/// Cloud {
///     base: 600.0,
///     thickness: 1200.0,
///     coverage: 0.45,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Cloud {
    /// Height of the layer's bottom over the ground, in meters. At least 0.
    #[asset(default = 1500.0)]
    pub base: f32,
    /// Depth of the layer from its bottom to its top, in meters. At least 10.
    #[asset(default = 1500.0)]
    pub thickness: f32,
    /// The share of the sky the clouds cover, in `[0, 1]`: 0 is a clear sky
    /// and 1 overcast.
    #[asset(default = 0.45)]
    pub coverage: f32,
    /// How tall the clouds grow inside the layer, in `[0, 1]`: 0 flat sheets,
    /// 1 towers reaching the top of the layer.
    #[asset(default = 0.5)]
    pub towering: f32,
    /// How much a cloud dims the light per meter at its densest. Greater than
    /// 0.
    #[asset(default = 0.04)]
    pub density: f32,
    /// Typical distance between clouds, in meters. At least 100.
    #[asset(default = 4000.0)]
    pub feature_size: f32,
    /// Size of the wisps the clouds' edges are worn into, in meters. At least
    /// 1, and at most a tenth of `feature_size`.
    #[asset(default = 120.0)]
    pub detail_size: f32,
    /// How fast the clouds drift, as a multiple of the wind's speed. At least
    /// 0.
    #[asset(default = 2.0)]
    pub wind_drift: f32,
    /// Seconds a cloud takes to change into a different one. 0 holds their
    /// shapes still.
    #[asset(default = 600.0)]
    pub evolve_seconds: f32,
    /// Places the clouds: each seed is a different sky.
    pub seed: u32,
    /// How far each cloud is drawn out along the wind, as a multiple of its
    /// width. At least 1: 1 keeps them round, and 5 or more streaks a thin,
    /// high layer into cirrus.
    #[asset(default = 1.0)]
    pub stretch: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::shader::consts::uint;

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shader::embedded::CLOUD_TYPES;
        assert_eq!(uint(src, "CLOUD_NOISE_SIZE"), NOISE_SIZE as usize);
        assert_eq!(uint(src, "CLOUD_MAX_LAYERS"), MAX_CLOUD_LAYERS);
        assert_eq!(
            uint(src, "CLOUD_TARGET_DIVISOR"),
            CLOUD_TARGET_DIVISOR as usize
        );
        assert_eq!(uint(src, "CLOUD_SHADOW_SIZE"), SHADOW_SIZE as usize);
        assert_eq!(uint(src, "CLOUD_PANORAMA_WIDTH"), PANORAMA_SIZE[0] as usize);
        assert_eq!(
            uint(src, "CLOUD_PANORAMA_HEIGHT"),
            PANORAMA_SIZE[1] as usize
        );
    }

    #[test]
    fn the_target_is_half_the_scene_rounded_up() {
        assert_eq!(cloud_target_size(1920, 1080), [960, 540]);
        assert_eq!(cloud_target_size(1921, 1), [961, 1]);
        assert_eq!(cloud_target_size(0, 0), [1, 1]);
    }
}
