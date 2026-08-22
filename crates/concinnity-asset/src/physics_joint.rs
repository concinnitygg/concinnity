// Physics-joint constraint schema.

use crate::{AssetId, de_opt_asset_ref};
use alloc::string::{String, ToString};

/// The constraint shape a `PhysicsJoint` declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsJointKind {
    /// All 6 degrees of freedom locked. The bodies move and rotate as one
    /// rigid assembly relative to their anchors. Use to weld two props
    /// together.
    Fixed,
    /// Single rotational axis. Rotation around `axis` (in each body's local
    /// frame) is free; everything else is locked. The canonical door hinge.
    Revolute,
    /// Three rotational axes free, all translation locked. Ball-and-socket
    /// joint: the canonical rope link or a hip socket.
    Spherical,
    /// Single translational axis. Sliding along `axis` is free; rotation and
    /// the other two translational axes are locked. The canonical slider /
    /// piston.
    Prismatic,
}

impl PhysicsJointKind {
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

    /// The kind's canonical authored name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Revolute => "revolute",
            Self::Spherical => "spherical",
            Self::Prismatic => "prismatic",
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
/// normalised on load; a zero axis falls back to `[0, 1, 0]`.
///
/// `limits_enabled` clamps the free axis: angle in degrees for revolute,
/// distance in world units for prismatic. `motor_target_velocity` and
/// `motor_max_force` drive the free axis when `motor_max_force > 0`; the
/// velocity is in degrees/sec for revolute, units/sec for prismatic.
///
/// ```rust
/// # use concinnity_asset::PhysicsJoint;
/// PhysicsJoint {
///     kind: "revolute".into(),
///     anchor_a: [0.0, 2.0, 0.0],
///     anchor_b: [0.0, 5.0, 0.0],
///     axis: [0.0, 0.0, 1.0],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PhysicsJoint {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Constraint shape; defaults to "fixed".
    pub kind: String,
    /// First body: a [Prop](#prop) name. Required.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub body_a: Option<AssetId>,
    /// Second body: a [Prop](#prop) name. Empty means "world anchor", in which
    /// case `anchor_b` is interpreted as a world-space position.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub body_b: Option<AssetId>,
    /// Attach point in `body_a`'s local frame.
    pub anchor_a: [f32; 3],
    /// Attach point in `body_b`'s local frame (or world space if `body_b` is
    /// empty).
    pub anchor_b: [f32; 3],
    /// Free axis for revolute/prismatic, in each body's local frame.
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

impl Default for PhysicsJoint {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            kind: "fixed".to_string(),
            body_a: None,
            body_b: None,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            limits_enabled: false,
            limits: [0.0, 0.0],
            motor_target_velocity: 0.0,
            motor_max_force: 0.0,
        }
    }
}

impl PhysicsJoint {
    /// Parse `kind`; falls back to `Fixed` for unrecognised values so a typo
    /// degrades safely. Cross-reference validation flags bad kinds explicitly.
    pub fn parsed_kind(&self) -> PhysicsJointKind {
        PhysicsJointKind::from_str_norm(&self.kind).unwrap_or(PhysicsJointKind::Fixed)
    }
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
    fn an_unrecognised_kind_has_no_parse() {
        assert_eq!(PhysicsJointKind::from_str_norm("bendy"), None);
        assert_eq!(PhysicsJointKind::from_str_norm(""), None);
    }

    #[test]
    fn a_blank_joint_welds_two_unset_bodies() {
        let j = PhysicsJoint::default();
        assert_eq!(j.kind, "fixed");
        assert_eq!(j.parsed_kind(), PhysicsJointKind::Fixed);
        assert_eq!(j.body_a, None);
        assert_eq!(j.body_b, None);
        assert_eq!(j.axis, [0.0, 1.0, 0.0]);
        assert!(!j.limits_enabled);
    }

    #[test]
    fn a_typo_in_kind_degrades_to_a_weld() {
        // Cross-reference validation reports the bad kind; the accessor must not
        // panic in the meantime.
        let j: PhysicsJoint = serde_json::from_str(r#"{"kind":"hindge"}"#).unwrap();
        assert_eq!(j.parsed_kind(), PhysicsJointKind::Fixed);
    }

    #[test]
    fn an_authored_hinge_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let j: PhysicsJoint = serde_json::from_str(
            r#"{"kind":"hinge","body_a":"door","body_b":"frame","axis":[0,1,0],
                "limits_enabled":true,"limits":[-90,0],"motor_max_force":12.5}"#,
        )
        .unwrap();
        assert_eq!(j.parsed_kind(), PhysicsJointKind::Revolute);
        assert_eq!(j.body_a, Some(crate::AssetId(4)));
        assert_eq!(j.body_b, Some(crate::AssetId(5)));

        let bytes = postcard::to_allocvec(&j).unwrap();
        let back: PhysicsJoint = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.parsed_kind(), PhysicsJointKind::Revolute);
        assert_eq!(back.limits, [-90.0, 0.0]);
        assert_eq!(back.motor_max_force, 12.5);
        // `asset_id` is injected, never authored, so it does not ride the wire.
        assert_eq!(back.asset_id, crate::AssetId::default());
    }
}
