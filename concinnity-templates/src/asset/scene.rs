// Scene asset builders: the lighting and atmosphere pieces the world templates
// layer onto a scene. Standalone assets with no source-file dependency, so a
// template built from them applies cleanly to any fresh world.

use crate::spec::AssetSpec;
use alloc::string::String;

// A directional (sun) light of `color` (RGB) shining along `direction`, scaled by
// `intensity`.
pub fn directional_light(
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

// A point light of `color` (RGB) at `position`, with `intensity` and falloff
// `range`.
pub fn point_light(
    name: impl Into<String>,
    color: [f32; 3],
    position: [f32; 3],
    intensity: f32,
    range: f32,
) -> AssetSpec {
    AssetSpec::new(name, "PointLight")
        .set("color", color)
        .set("position", position)
        .set("intensity", intensity)
        .set("range", range)
}

// An image-based-lighting environment generated from the procedural sky (no .hdr
// source needed).
pub fn environment_map_sky(name: impl Into<String>) -> AssetSpec {
    AssetSpec::new(name, "EnvironmentMap").set("generator", "sky")
}

// A volumetric fog volume: `density`, tint `color` (RGB), vertical `height_falloff`,
// `max_distance`, and the Henyey-Greenstein `phase_g`.
pub fn volumetric_fog(
    name: impl Into<String>,
    density: f32,
    color: [f32; 3],
    height_falloff: f32,
    max_distance: f32,
    phase_g: f32,
) -> AssetSpec {
    AssetSpec::new(name, "VolumetricFog")
        .set("density", density)
        .set("color", color)
        .set("height_falloff", height_falloff)
        .set("max_distance", max_distance)
        .set("phase_g", phase_g)
}

// A post-process configuration: bloom strength / threshold and exposure bias.
pub fn post_process(
    name: impl Into<String>,
    bloom_intensity: f32,
    bloom_threshold: f32,
    exposure_ev: f32,
) -> AssetSpec {
    AssetSpec::new(name, "PostProcessConfig")
        .set("bloom_intensity", bloom_intensity)
        .set("bloom_threshold", bloom_threshold)
        .set("exposure_ev", exposure_ev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ArgValue;
    use alloc::string::ToString;

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
