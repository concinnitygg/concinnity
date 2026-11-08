// Buoyancy schema for a dynamic body.

/// Makes a [`PropBody`](#propbody) float in water.
///
/// A body with `buoyancy` floats in every [`WaterSurface`](#watersurface) it
/// reaches, on the waves where they are drawn: it rises and falls with the
/// swell, pitches and rolls with the slope of the water under it, drifts with
/// the water, and in the surf of a planet sea the breaking waves carry it in
/// toward the beach. On calm water it settles and comes to rest.
///
/// How high it floats follows its density, the body's `mass` over the volume
/// of its collider, against the water's `density`: a body half as dense as
/// the water floats half submerged, one denser than the water sinks. Give the
/// body a `mass`: one left at 0 takes the mass of its collider at 1 kg per
/// cubic meter, a thousandth of water's density, so it rides with all but a
/// sliver of it above the water.
///
/// The water slows what moves through it: `linear_drag` and `angular_drag`
/// set how quickly the body comes to move with the water and stops spinning
/// once it is fully under, and less while it is only partly submerged.
///
/// ```json
/// ["PropBody", { "prop_name": "crate_a", "mass": 300.0, "buoyancy": { "linear_drag": 1.5 } }]
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Buoyancy {
    /// How quickly the water brings the body to its own speed, per second,
    /// fully submerged. At least 0.
    #[asset(default = 3.0)]
    pub linear_drag: f32,
    /// How quickly the water stops the body's spin, per second, fully
    /// submerged. At least 0.
    #[asset(default = 3.0)]
    pub angular_drag: f32,
}

impl Buoyancy {
    /// This with both drags finite and not negative: NaN or a negative drag
    /// takes none, an infinite one the most a step can apply.
    #[must_use]
    pub fn validated(self) -> Self {
        let drag = |d: f32| {
            if d.is_nan() {
                0.0
            } else {
                d.clamp(0.0, f32::MAX)
            }
        };
        Self {
            linear_drag: drag(self.linear_drag),
            angular_drag: drag(self.angular_drag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drags_are_held_to_what_a_step_can_apply() {
        let b = Buoyancy {
            linear_drag: f32::NAN,
            angular_drag: -3.0,
        }
        .validated();
        assert_eq!(
            b,
            Buoyancy {
                linear_drag: 0.0,
                angular_drag: 0.0
            }
        );
        let huge = Buoyancy {
            linear_drag: f32::INFINITY,
            ..Buoyancy::default()
        }
        .validated();
        assert_eq!(huge.linear_drag, f32::MAX);
        assert_eq!(Buoyancy::default().validated(), Buoyancy::default());
    }

    #[test]
    fn json_round_trips() {
        let b: Buoyancy = serde_json::from_str(r#"{"angular_drag":2.5}"#).unwrap();
        assert_eq!(b.linear_drag, 3.0);
        assert_eq!(b.angular_drag, 2.5);
        let back: Buoyancy = serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
        assert_eq!(back, b);
    }
}
