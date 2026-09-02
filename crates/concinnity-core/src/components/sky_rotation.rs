// Celestial-sphere rotation schema.

use crate::ecs::asset_id::AssetId;

/// Turns the whole celestial sphere: the sky, the image-based lighting it
/// casts, every [DirectionalLight](#directionallight), and any
/// [Prop](#prop) hung on it.
///
/// One per world. The rotation at elapsed time `t` is `angle_deg +
/// degrees_per_second * t` about `axis`, taken in the sense a planet's own
/// spin gives the sky: with the default axis a body rises from `+Z`, passes
/// overhead through `+Y`, and sets toward `-Z`.
///
/// The component's own entity carries that rotation as its transform, so a
/// `Prop` naming this asset as its `parent` orbits with the sky. Reflection
/// probes are baked once and do not turn.
///
/// ```rust
/// # use concinnity_core::components::SkyRotation;
/// SkyRotation {
///     axis: [1.0, 0.0, 0.0],
///     degrees_per_second: 3.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SkyRotation {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The celestial pole in world space: the axis the sphere turns about.
    /// Does not need to be normalised.
    pub axis: [f32; 3],
    /// Turn rate in degrees per second. Negative runs the sky backwards.
    pub degrees_per_second: f32,
    /// The angle the sky starts at, in degrees.
    pub angle_deg: f32,
}

impl Default for SkyRotation {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            axis: [1.0, 0.0, 0.0],
            degrees_per_second: 1.0,
            angle_deg: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_pole_is_horizontal_so_bodies_rise_and_set() {
        // A pole along Y would only spin the sky about the zenith, which no
        // observer on the ground can see; a horizontal pole is what makes a
        // body cross the sky.
        let s = SkyRotation::default();
        assert_eq!(s.axis, [1.0, 0.0, 0.0]);
        assert_eq!(s.degrees_per_second, 1.0);
        assert_eq!(s.angle_deg, 0.0);
    }

    #[test]
    fn an_authored_rotation_parses_and_round_trips_through_postcard() {
        let s: SkyRotation =
            serde_json::from_str(r#"{"axis":[0,0,1],"degrees_per_second":6,"angle_deg":45}"#)
                .unwrap();
        assert_eq!(s.axis, [0.0, 0.0, 1.0]);
        assert_eq!(s.degrees_per_second, 6.0);
        assert_eq!(s.angle_deg, 45.0);

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: SkyRotation = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.axis, [0.0, 0.0, 1.0]);
        assert_eq!(back.degrees_per_second, 6.0);
        assert_eq!(back.angle_deg, 45.0);
    }
}
