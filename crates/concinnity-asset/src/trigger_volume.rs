// Trigger-volume schema: a spatial sensor region.

use crate::{AssetId, PropCollider};

/// An invisible sensor region that reports when something enters or leaves it.
///
/// A trigger volume senses overlap and never collides: nothing bounces off
/// it and it blocks no movement. [Reaction](#reaction)s listen for its
/// crossings with an `enter` or `exit` source, so "when the player steps into
/// this area, open that door" is two declared assets. `detects` filters what
/// sets it off: the player character, dynamic props, or anything. Volumes
/// sense at their authored position; they do not move at runtime.
///
/// ```jsonl
/// {"name":"vault_zone","type":"TriggerVolume","args":{"position":[4,1,-2],"collider":{"shape":"cuboid","half_extents":[2,1.5,2]}}}
/// {"name":"vault_opens","type":"Reaction","args":{"on":{"enter":"vault_zone"},"actions":[{"despawn":{"target":"vault_door"}}],"once":true}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TriggerVolume {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// World-space position of the volume's center.
    pub position: [f32; 3],
    /// Euler rotation of the volume in degrees.
    pub rotation_deg: [f32; 3],
    /// The sensed region, in the same shape vocabulary as a
    /// [PropCollider](#propcollider): a `cuboid` with `half_extents`, a `ball`
    /// with `radius`, or a `capsule`.
    pub collider: PropCollider,
    /// What sets the volume off: the `player` character, dynamic `props`, or
    /// `any` of them.
    pub detects: TriggerFilter,
}

/// What a [TriggerVolume](#triggervolume) senses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TriggerFilter {
    /// Only the player character (the controlled camera capsule or the
    /// followed character).
    #[default]
    Player,
    /// Only dynamic props (a `Prop` with a `PropBody`).
    Props,
    /// Anything the physics simulation moves.
    Any,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_volume_is_a_unit_cuboid_sensing_the_player() {
        let v: TriggerVolume = serde_json::from_str("{}").unwrap();
        assert_eq!(v.collider.shape, "cuboid");
        assert_eq!(v.collider.half_extents, [0.5, 0.5, 0.5]);
        assert_eq!(v.detects, TriggerFilter::Player);
    }

    #[test]
    fn filter_names_parse() {
        let v: TriggerVolume = serde_json::from_str(r#"{"detects":"props"}"#).unwrap();
        assert_eq!(v.detects, TriggerFilter::Props);
        let v: TriggerVolume = serde_json::from_str(r#"{"detects":"any"}"#).unwrap();
        assert_eq!(v.detects, TriggerFilter::Any);
    }

    #[test]
    fn baked_round_trip_is_postcard_stable() {
        let v: TriggerVolume = serde_json::from_str(
            r#"{"position":[4,1,-2],"collider":{"shape":"ball","radius":2.0},"detects":"any"}"#,
        )
        .unwrap();
        let bytes = postcard::to_allocvec(&v).unwrap();
        let back: TriggerVolume = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.position, [4.0, 1.0, -2.0]);
        assert_eq!(back.collider.shape, "ball");
        assert_eq!(back.detects, TriggerFilter::Any);
    }
}
