// Prop reflection-probe schema.

use crate::components::Prop;
use crate::ecs::Ref;

/// A reflection probe that rides on a [Prop](#prop) and is captured again
/// all the time, so a mirror-like prop (a chrome ball, a car's paint)
/// reflects what is around it wherever it goes: the ground it rolls over,
/// the shore and the sea, the props it passes.
///
/// The cube is rendered from the center of the prop's bounds,
/// `faces_per_frame` of its six faces each frame, and the cube the prop
/// reflects is renewed whenever all six have been rendered. The prop itself
/// is left out of its own capture. Its glossy
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
/// Rendering fewer faces a frame costs less but renews the reflection less
/// often: at the default 2 the cube is renewed every third frame from faces
/// rendered over those three, so a fast prop's reflection trails it by a few
/// frames. 6 renews it every frame.
///
/// ```json
/// ["PropReflectionProbe", { "prop": "car", "faces_per_frame": 6 }]
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
    /// How many of the cube's six faces are rendered each frame: 1, 2, 3 or
    /// 6. Another value is taken as the nearest of those.
    #[asset(default = 2)]
    pub faces_per_frame: u32,
}

/// Smallest face edge a prop probe captures at.
pub(crate) const MIN_PROP_PROBE_RESOLUTION: u32 = 16;

/// Largest face edge a prop probe captures at: the cube array's own.
pub(crate) const MAX_PROP_PROBE_RESOLUTION: u32 = 512;

/// The face counts a frame can render that divide the cube's six evenly, so
/// every cycle renews the whole cube.
const FACES_PER_FRAME: [u32; 4] = [1, 2, 3, 6];

/// `faces` as the allowed face count nearest to it, the lower on a tie.
pub(crate) fn prop_probe_faces_per_frame(faces: u32) -> u32 {
    FACES_PER_FRAME
        .into_iter()
        .min_by_key(|&n| n.abs_diff(faces))
        .unwrap_or(1)
}

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
        assert_eq!(bare.faces_per_frame, 2);
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
    fn the_faces_per_frame_divide_the_cube() {
        assert_eq!(prop_probe_faces_per_frame(0), 1);
        assert_eq!(prop_probe_faces_per_frame(2), 2);
        assert_eq!(prop_probe_faces_per_frame(4), 3);
        assert_eq!(prop_probe_faces_per_frame(5), 6);
        assert_eq!(prop_probe_faces_per_frame(100), 6);
    }

    #[test]
    fn validation_rounds_the_resolution_and_floors_the_distance() {
        let probe = crate::components::validate::prop_reflection_probe(PropReflectionProbe {
            resolution: 200,
            capture_distance: Some(-3.0),
            faces_per_frame: 4,
            ..Default::default()
        });
        assert_eq!(probe.resolution, 256);
        assert_eq!(probe.capture_distance, Some(0.0));
        assert_eq!(probe.faces_per_frame, 3);
    }
}
