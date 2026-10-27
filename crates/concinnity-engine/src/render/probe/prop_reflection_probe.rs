// Prop reflection-probe schema.

use crate::components::Prop;
use crate::ecs::Ref;

/// A reflection probe that rides on a [Prop](#prop) and is captured again
/// all the time, so a mirror-like prop (a chrome ball, a car's paint)
/// reflects what is around it wherever it goes: the ground it rolls over,
/// the shore and the sea, the props it passes.
///
/// The cube is rendered from the center of the prop's bounds, all six faces
/// every frame. The prop itself is left out of its own capture. Its glossy
/// surface reflects the cube at full strength, a surface right beside it
/// blends the cube with any static [`ReflectionProbe`](#reflectionprobe)
/// there. Its influence ends a fifth of the prop's smallest half-size past
/// its bounds, enough to cover the prop's own surface, and beyond that the
/// reflection hands over to the sky within about half the prop's size.
///
/// The capture draws the opaque scene within `capture_distance` of the prop,
/// 200 m by default, and the environment's sky beyond it, which on a planet
/// carries the far ground and sea below the horizon. Grass is drawn as the
/// far field's meadow rather than as blades, and a planet's sea as a calm
/// surface at sea level. It is skipped where ray-traced reflections already
/// reflect the scene.
///
/// ```json
/// ["PropReflectionProbe", { "prop": "marble" }]
/// ```
///
/// Each face is `resolution` texels square. The default suits a prop that
/// covers a few hundred pixels at most; a larger reflector seen up close
/// (the side of a car) may want 256, which shades four times the pixels.
/// A shorter `capture_distance` costs less to capture, a longer one draws
/// more of the far scene itself; `null` draws all of it.
///
/// ```json
/// ["PropReflectionProbe", { "prop": "car", "resolution": 256, "capture_distance": 400.0 }]
/// ```
///
/// A world uses at most one.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct PropReflectionProbe {
    /// The [Prop](#prop) the probe rides on and reflects in.
    pub prop: Option<Ref<Prop>>,
    /// Edge of each cube face, in texels: a power of two from 16 to 512.
    /// Another value is rounded to the nearest power of two in that range.
    #[asset(default = 128)]
    pub resolution: u32,
    /// How far the capture reaches, in meters: an object lying wholly farther
    /// than this from the prop along the axis a cube face looks down is left
    /// out of that face, and the sky shows in its place. `null` captures
    /// without limit.
    #[asset(default = Some(200.0))]
    pub capture_distance: Option<f32>,
}

/// Smallest face edge a prop probe captures at.
pub(crate) const MIN_PROP_PROBE_RESOLUTION: u32 = 16;

/// Largest face edge a prop probe captures at: the cube array's own.
pub(crate) const MAX_PROP_PROBE_RESOLUTION: u32 = 512;

/// `resolution` as the power of two in range nearest to it.
pub(crate) fn prop_probe_resolution(resolution: u32) -> u32 {
    let clamped = resolution.clamp(MIN_PROP_PROBE_RESOLUTION, MAX_PROP_PROBE_RESOLUTION);
    let up = clamped.next_power_of_two();
    let down = up >> 1;
    if up - clamped <= clamped - down {
        up
    } else {
        down
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_capture_small_faces_nearby() {
        let bare: PropReflectionProbe = crate::test_support::from_json("{}");
        assert!(bare.prop.is_none());
        assert_eq!(bare.resolution, 128);
        assert_eq!(bare.capture_distance, Some(200.0));
        let unlimited: PropReflectionProbe =
            crate::test_support::from_json(r#"{"capture_distance": null}"#);
        assert_eq!(unlimited.capture_distance, None);
    }

    #[test]
    fn the_resolution_rounds_to_a_power_of_two_in_range() {
        assert_eq!(prop_probe_resolution(128), 128);
        assert_eq!(prop_probe_resolution(100), 128);
        assert_eq!(prop_probe_resolution(90), 64);
        assert_eq!(prop_probe_resolution(96), 128);
        assert_eq!(prop_probe_resolution(0), MIN_PROP_PROBE_RESOLUTION);
        assert_eq!(prop_probe_resolution(4096), MAX_PROP_PROBE_RESOLUTION);
    }

    #[test]
    fn validation_rounds_the_resolution_and_floors_the_distance() {
        let probe = crate::components::validate::prop_reflection_probe(PropReflectionProbe {
            resolution: 200,
            capture_distance: Some(-3.0),
            ..Default::default()
        });
        assert_eq!(probe.resolution, 256);
        assert_eq!(probe.capture_distance, Some(0.0));
    }
}
