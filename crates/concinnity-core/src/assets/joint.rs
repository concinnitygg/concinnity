// src/assets/joint.rs

use crate::assets::{Joint, JointKind};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for Joint {
    const NAME: &'static str = "Joint";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Normalise the kind string so `to_args` round-trips cleanly.
        if let Some(k) = JointKind::from_str_norm(&args.kind) {
            args.kind = k.as_str().to_string();
        }
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_defaults() {
        let j: Joint = serde_json::from_str("{}").unwrap();
        assert_eq!(j.kind, "fixed");
        assert_eq!(j.anchor_a, [0.0, 0.0, 0.0]);
        assert_eq!(j.axis, [0.0, 1.0, 0.0]);
        assert!(!j.limits_enabled);
        assert_eq!(j.motor_max_force, 0.0);
    }

    #[test]
    fn deserialises_all_fields() {
        crate::ecs::asset_id::reset_interner();
        let json = r#"{
            "kind":"revolute",
            "body_a":"door",
            "body_b":"wall",
            "anchor_a":[0.5,1.0,0.0],
            "anchor_b":[1.0,1.0,0.0],
            "axis":[0,1,0],
            "limits_enabled":true,
            "limits":[-90,90],
            "motor_target_velocity":30.0,
            "motor_max_force":50.0
        }"#;
        let j: Joint = serde_json::from_str(json).unwrap();
        assert_eq!(j.parsed_kind(), JointKind::Revolute);
        assert!(j.body_a.is_some());
        assert!(j.body_b.is_some());
        assert!(j.limits_enabled);
    }

    #[test]
    fn aliases_resolve_to_canonical_kind() {
        assert_eq!(JointKind::from_str_norm("hinge"), Some(JointKind::Revolute));
        assert_eq!(JointKind::from_str_norm("WELD"), Some(JointKind::Fixed));
        assert_eq!(JointKind::from_str_norm("ball"), Some(JointKind::Spherical));
        assert_eq!(
            JointKind::from_str_norm("slider"),
            Some(JointKind::Prismatic)
        );
    }

    #[test]
    fn from_args_normalises_kind_string() {
        let json = r#"{"kind":"HINGE"}"#;
        let parsed: Joint = serde_json::from_str(json).unwrap();
        let normalised = Joint::from_args(parsed);
        assert_eq!(normalised.kind, "revolute");
    }

    #[test]
    fn unknown_kind_falls_back_to_fixed() {
        let j = Joint {
            kind: "frumpus".to_string(),
            ..Default::default()
        };
        assert_eq!(j.parsed_kind(), JointKind::Fixed);
    }
}
