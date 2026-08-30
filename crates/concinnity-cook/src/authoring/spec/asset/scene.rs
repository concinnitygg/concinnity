// Scene asset builders: the lighting and atmosphere pieces the world templates
// layer onto a scene. Standalone assets with no source-file dependency, so a
// template built from them applies cleanly to any fresh world.

use crate::authoring::spec::AssetSpec;

// A directional (sun) light of `color` (RGB) shining along `direction`, scaled by
// `intensity`.
pub(crate) fn directional_light(
    name: impl Into<String>,
    color: [f32; 3],
    direction: [f32; 3],
    intensity: f32,
) -> AssetSpec {
    AssetSpec::new(name, "DirectionalLight")
        .set("color", color)
        .set("direction", direction)
        .set("intensity", intensity)
}

// A spot light of `color` (RGB) at `position` aimed along `direction`, with
// `intensity`, falloff `range`, and a cone that fades from `cone[0]` to `cone[1]`
// (inner and outer half-angles, in degrees).
#[cfg(test)]
pub(crate) fn spot_light(
    name: impl Into<String>,
    color: [f32; 3],
    position: [f32; 3],
    direction: [f32; 3],
    intensity: f32,
    range: f32,
    cone: [f32; 2],
) -> AssetSpec {
    AssetSpec::new(name, "SpotLight")
        .set("color", color)
        .set("position", position)
        .set("direction", direction)
        .set("intensity", intensity)
        .set("range", range)
        .set("inner_angle", cone[0])
        .set("outer_angle", cone[1])
}

// An image-based-lighting environment generated from the procedural sky (no .hdr
// source needed).
pub(crate) fn environment_map_sky(name: impl Into<String>) -> AssetSpec {
    AssetSpec::new(name, "EnvironmentMap").set("generator", "sky")
}

/// A perspective camera at `position` (world units), facing `yaw` / `pitch`
/// (radians; yaw 0 / pitch 0 looks down -Z). The type's default inspector
/// controller keeps the scene navigable out of the box.
pub(crate) fn camera(
    name: impl Into<String>,
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
) -> AssetSpec {
    AssetSpec::new(name, "Camera3D")
        .set("position", position)
        .set("yaw", yaw)
        .set("pitch", pitch)
}

/// A self-contained room (floor, ceiling, four walls) of full extents
/// `size` = [width, depth, height], centred on the origin. Standalone geometry:
/// no mesh source or texture reference needed.
pub(crate) fn room(name: impl Into<String>, size: [f32; 3]) -> AssetSpec {
    AssetSpec::new(name, "Room").set("size", size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::spec::ArgValue;

    #[test]
    fn directional_light_sets_its_fields() {
        let d = directional_light("key", [1.0, 0.95, 0.8], [-0.3, 0.85, 0.4], 1.1);
        assert_eq!(d.asset_type, "DirectionalLight");
        let field = |k: &str| d.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("color"), Some(&ArgValue::floats(&[1.0, 0.95, 0.8])));
        // f32 -> f64 widening is exact only for representable values, so compare
        // against the same widening rather than an f64 literal.
        assert_eq!(field("intensity"), Some(&ArgValue::Float(1.1f32 as f64)));
    }

    #[test]
    fn spot_light_splits_the_cone_into_its_two_angles() {
        let s = spot_light(
            "lamp",
            [1.0, 0.9, 0.7],
            [0.0, 4.0, 0.0],
            [0.0, -1.0, 0.0],
            20.0,
            10.0,
            [18.0, 30.0],
        );
        assert_eq!(s.asset_type, "SpotLight");
        let field = |k: &str| s.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(
            field("direction"),
            Some(&ArgValue::floats(&[0.0, -1.0, 0.0]))
        );
        assert_eq!(field("inner_angle"), Some(&ArgValue::Float(18.0f32 as f64)));
        assert_eq!(field("outer_angle"), Some(&ArgValue::Float(30.0f32 as f64)));
    }

    #[test]
    fn camera_sets_position_and_orientation() {
        let c = camera("cam", [0.0, 2.4, 9.0], 0.0, -0.12);
        assert_eq!(c.asset_type, "Camera3D");
        let field = |k: &str| c.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("position"), Some(&ArgValue::floats(&[0.0, 2.4, 9.0])));
        assert_eq!(field("pitch"), Some(&ArgValue::Float(-0.12f32 as f64)));
    }

    #[test]
    fn room_sets_its_full_extents() {
        let r = room("room", [16.0, 20.0, 5.0]);
        assert_eq!(r.asset_type, "Room");
        assert_eq!(
            r.fields.iter().find(|(k, _)| k == "size").map(|(_, v)| v),
            Some(&ArgValue::floats(&[16.0, 20.0, 5.0]))
        );
    }

    #[test]
    fn environment_map_uses_the_sky_generator() {
        let e = environment_map_sky("env");
        assert_eq!(
            e.fields
                .iter()
                .find(|(k, _)| k == "generator")
                .map(|(_, v)| v),
            Some(&ArgValue::Str("sky".to_string()))
        );
    }
}
