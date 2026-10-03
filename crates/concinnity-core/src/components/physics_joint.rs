// Physics-joint constraint schema.

use crate::components::Prop;
use crate::components::{Vocabulary, vocabulary_synonyms};
use crate::ecs::Ref;

/// The constraint shape a `PhysicsJoint` declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Vocabulary)]
pub enum PhysicsJointKind {
    /// All 6 degrees of freedom locked. The bodies move and rotate as one
    /// rigid assembly relative to their anchors. Use to weld two props
    /// together.
    #[default]
    #[vocab("fixed")]
    Fixed,
    /// Single rotational axis. Rotation around `axis` (in each body's local
    /// frame) is free; everything else is locked. The canonical door hinge.
    #[vocab("revolute")]
    Revolute,
    /// Three rotational axes free, all translation locked. Ball-and-socket
    /// joint: the canonical rope link or a hip socket.
    #[vocab("spherical")]
    Spherical,
    /// Single translational axis. Sliding along `axis` is free; rotation and
    /// the other two translational axes are locked. The canonical slider /
    /// piston.
    #[vocab("prismatic")]
    Prismatic,
}

vocabulary_synonyms!(PhysicsJointKind, "a joint kind index");

impl PhysicsJointKind {
    /// Every authored name, canonical and synonym, this accepts. The build
    /// lists these when it rejects an unknown kind.
    pub const ACCEPTED: &'static [&'static str] = &[
        "fixed",
        "weld",
        "revolute",
        "hinge",
        "spherical",
        "ball",
        "socket",
        "prismatic",
        "slider",
        "piston",
    ];

    /// The kind an authored name selects, accepting the common synonyms
    /// (`hinge`, `ball`, `slider`, ...). `None` for an unknown name.
    pub fn from_str_norm(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fixed" | "weld" => Some(Self::Fixed),
            "revolute" | "hinge" => Some(Self::Revolute),
            "spherical" | "ball" | "socket" => Some(Self::Spherical),
            "prismatic" | "slider" | "piston" => Some(Self::Prismatic),
            _ => None,
        }
    }
}

/// A physics constraint connecting two [Prop](#prop)s that own a `collider`.
///
/// The joint pins `anchor_a` on `body_a` to `anchor_b` on `body_b` and locks
/// the relative motion of the two bodies according to its `kind`. Anchors are
/// in each body's local frame: `[0, 0, 0]` is the body's own pivot.
///
/// To anchor a body to "the world" (no second prop), leave `body_b` empty: a
/// hidden static anchor is created at `anchor_b` (interpreted as world space in
/// that case) and the body joints to it. This is the pendulum / lamp / trapeze
/// pattern.
///
/// `axis` only applies to `revolute` and `prismatic`: it is the single free
/// axis (rotation or translation) in each body's local frame. The vector is
/// normalized on load; a zero axis falls back to `[0, 1, 0]`.
///
/// `limits_enabled` clamps the free axis: angle in degrees for revolute,
/// distance in world units for prismatic. `motor_target_velocity` and
/// `motor_max_force` drive the free axis when `motor_max_force > 0`; the
/// velocity is in degrees/sec for revolute, units/sec for prismatic.
///
/// ```rust
/// # use concinnity_core::components::{PhysicsJoint, PhysicsJointKind};
/// PhysicsJoint {
///     kind: PhysicsJointKind::Revolute,
///     anchor_a: [0.0, 2.0, 0.0],
///     anchor_b: [0.0, 5.0, 0.0],
///     axis: [0.0, 0.0, 1.0],
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
pub struct PhysicsJoint {
    /// Constraint shape; defaults to `fixed`. See [PhysicsJointKind].
    pub kind: PhysicsJointKind,
    /// First body: a [Prop](#prop) name. Required.
    pub body_a: Option<Ref<Prop>>,
    /// Second body: a [Prop](#prop) name. Empty means "world anchor", in which
    /// case `anchor_b` is interpreted as a world-space position.
    pub body_b: Option<Ref<Prop>>,
    /// Attach point in `body_a`'s local frame.
    pub anchor_a: [f32; 3],
    /// Attach point in `body_b`'s local frame (or world space if `body_b` is
    /// empty).
    pub anchor_b: [f32; 3],
    /// Free axis for revolute/prismatic, in each body's local frame.
    #[asset(default = [0.0, 1.0, 0.0])]
    pub axis: [f32; 3],
    /// Whether the `limits` clamp is enforced.
    pub limits_enabled: bool,
    /// `[min, max]` clamp on the free axis: degrees for revolute, world units
    /// for prismatic. Ignored unless `limits_enabled` is true.
    pub limits: [f32; 2],
    /// Motor target velocity: degrees/sec for revolute, world units/sec for
    /// prismatic. Ignored unless `motor_max_force > 0`.
    pub motor_target_velocity: f32,
    /// Motor force budget. The motor is inactive when this is 0.
    pub motor_max_force: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_accepts_its_aliases_and_round_trips_through_its_canonical_name() {
        let cases = [
            (PhysicsJointKind::Fixed, "fixed", ["fixed", "weld", "WELD"]),
            (
                PhysicsJointKind::Revolute,
                "revolute",
                ["revolute", "hinge", "Hinge"],
            ),
            (
                PhysicsJointKind::Spherical,
                "spherical",
                ["spherical", "ball", "socket"],
            ),
            (
                PhysicsJointKind::Prismatic,
                "prismatic",
                ["prismatic", "slider", "piston"],
            ),
        ];
        for (kind, canonical, aliases) in cases {
            assert_eq!(kind.as_str(), canonical);
            for alias in aliases {
                assert_eq!(
                    PhysicsJointKind::from_str_norm(alias),
                    Some(kind),
                    "{alias}"
                );
            }
            assert_eq!(PhysicsJointKind::from_str_norm(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn an_unrecognized_kind_has_no_parse() {
        assert_eq!(PhysicsJointKind::from_str_norm("bendy"), None);
        assert_eq!(PhysicsJointKind::from_str_norm(""), None);
    }

    // A typo is no longer a joint: the field is typed, so the load rejects it
    // rather than silently welding the two bodies.
    #[test]
    fn a_typo_in_kind_is_rejected_with_the_kinds_it_expected() {
        let err = serde_json::from_str::<PhysicsJoint>(r#"{"kind":"hindge"}"#)
            .expect_err("an unknown kind does not deserialize");
        let msg = alloc::format!("{err}");
        assert!(msg.contains("hindge"), "{msg}");
        for kind in PhysicsJointKind::NAMES {
            assert!(msg.contains(kind), "{msg}");
        }
    }

    // Every name a world may author, canonical or synonym, still loads.
    #[test]
    fn every_accepted_name_deserializes() {
        for name in PhysicsJointKind::ACCEPTED {
            let json = alloc::format!(r#""{name}""#);
            let kind: PhysicsJointKind = serde_json::from_str(&json).expect(name);
            assert_eq!(Some(kind), PhysicsJointKind::from_str_norm(name));
        }
    }
}
