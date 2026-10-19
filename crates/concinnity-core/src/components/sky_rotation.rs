// Celestial-sphere rotation schema.

/// Turns the whole celestial sphere: the sky, the image-based lighting it
/// casts, every [`DirectionalLight`](#directionallight), and any
/// [Prop](#prop) hung on it.
///
/// One per world. The rotation at elapsed time `t` is `angle_deg +
/// degrees_per_second * t` about `axis`, taken in the sense a planet's own
/// spin gives the sky: with the default axis a body rises from `+Z`, passes
/// overhead through `+Y`, and sets toward `-Z`.
///
/// The component's own entity carries that rotation as its transform, so a
/// `Prop` naming this asset as its `parent` orbits with the sky. Reflection
/// probes are baked once and do not turn. Under an [Atmosphere](#atmosphere)
/// the air itself stays put and the sun moves through it, so the rotation is
/// the time of day.
///
/// ```rust
/// # use concinnity_core::components::SkyRotation;
/// SkyRotation {
///     axis: [1.0, 0.0, 0.0],
///     degrees_per_second: 3.0,
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
#[serde(default)]
pub struct SkyRotation {
    /// The celestial pole in world space: the axis the sphere turns about.
    /// Does not need to be normalized.
    #[asset(default = [1.0, 0.0, 0.0])]
    pub axis: [f32; 3],
    /// Turn rate in degrees per second. Negative runs the sky backwards.
    #[asset(default = 1.0)]
    pub degrees_per_second: f32,
    /// The angle the sky starts at, in degrees.
    pub angle_deg: f32,
}

impl SkyRotation {
    /// Direction `dir` as the sky has carried it `seconds` after the start:
    /// where a [`DirectionalLight`](#directionallight) authored along `dir`
    /// shines from at that moment.
    pub fn turned(&self, dir: [f32; 3], seconds: f64) -> [f32; 3] {
        let angle = f64::from(self.angle_deg) + f64::from(self.degrees_per_second) * seconds;
        crate::sky::SkyOrientation::new(self.axis, (angle % 360.0) as f32).rotate(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-5)
    }

    // With the default pole a body rises from +Z, is overhead at a quarter
    // turn and sets toward -Z, and the start angle counts toward the turn.
    #[test]
    fn a_body_rises_overhead_and_sets() {
        let sky = SkyRotation {
            degrees_per_second: 9.0,
            ..Default::default()
        };
        assert!(close(sky.turned([0.0, 0.0, 1.0], 0.0), [0.0, 0.0, 1.0]));
        assert!(close(sky.turned([0.0, 0.0, 1.0], 10.0), [0.0, 1.0, 0.0]));
        assert!(close(sky.turned([0.0, 0.0, 1.0], 20.0), [0.0, 0.0, -1.0]));
        let later = SkyRotation {
            angle_deg: 90.0,
            ..sky
        };
        assert!(close(later.turned([0.0, 0.0, 1.0], 0.0), [0.0, 1.0, 0.0]));
        // The pole itself never moves, and a long run lands where a short one
        // a whole number of turns earlier does.
        assert!(close(sky.turned([1.0, 0.0, 0.0], 7.0), [1.0, 0.0, 0.0]));
        assert!(close(
            sky.turned([0.0, 0.6, 0.8], 4_000.0 + 3.0),
            sky.turned([0.0, 0.6, 0.8], 3.0)
        ));
    }
}
