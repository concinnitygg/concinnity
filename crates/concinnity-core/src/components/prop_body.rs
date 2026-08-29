// Dynamic physics body schema for a companion Prop.

use crate::ecs::AudioClipHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;
use crate::ecs::de_opt_audio_clip_handle;

/// Makes a companion [Prop](#prop) a dynamic physics body.
///
/// Attach a PropBody to give a [Prop](#prop) real physics: it falls, collides,
/// stacks, tumbles, and (with `pickup: true` on the prop) can be carried and
/// thrown. A Prop with a `collider` but no PropBody is a static, immovable
/// obstacle.
///
/// ```json
/// {
///   "name": "crate_a_body",
///   "type": "PropBody",
///   "args": { "prop_name": "crate_a", "mass": 4.0, "friction": 0.6 }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PropBody {
    /// The [Prop](#prop) this body drives. Must match a Prop declared in the
    /// same world.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub prop_name: Option<AssetId>,
    /// Mass in kilograms. 0 lets the simulation derive mass from the collider
    /// shape and a default density.
    pub mass: f32,
    /// Friction coefficient used for contacts with this body.
    pub friction: f32,
    /// Bounciness in [0, 1]. 0 is fully inelastic.
    pub restitution: f32,
    /// Multiplier applied to world gravity for this body. 1.0 is normal.
    pub gravity_scale: f32,
    /// Linear velocity damping, modelling air drag.
    pub linear_damping: f32,
    /// Optional [AudioClip](#audioclip) played at the contact point when this
    /// body collides hard enough to pass the world's `contact_min_impulse`
    /// (see [PhysicsConfig](#physicsconfig)). Louder impacts play louder.
    #[serde(deserialize_with = "de_opt_audio_clip_handle")]
    pub impact_clip: Option<AudioClipHandle>,
    /// Linear gain applied to the impact clip at full impulse.
    pub impact_volume: f32,
}

impl Default for PropBody {
    fn default() -> Self {
        Self {
            prop_name: None,
            mass: 0.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.05,
            impact_clip: None,
            impact_volume: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_body_falls_under_full_gravity_without_bouncing() {
        let b = PropBody::default();
        assert_eq!(b.gravity_scale, 1.0);
        assert_eq!(b.friction, 0.5);
        assert_eq!(b.restitution, 0.0);
        assert_eq!(b.linear_damping, 0.05);
        // Zero mass means "derive it from the collider", not "massless".
        assert_eq!(b.mass, 0.0);
        assert!(b.prop_name.is_none());
    }

    #[test]
    fn a_bouncy_floating_body_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let b: PropBody = serde_json::from_str(
            r#"{"prop_name":"ball","mass":2.5,"friction":0.1,"restitution":0.9,
                "gravity_scale":0,"linear_damping":0.2}"#,
        )
        .unwrap();
        assert_eq!(b.prop_name, Some(AssetId(4)));
        assert_eq!(b.gravity_scale, 0.0);

        let bytes = postcard::to_allocvec(&b).unwrap();
        let back: PropBody = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.prop_name, Some(AssetId(4)));
        assert_eq!(back.mass, 2.5);
        assert_eq!(back.friction, 0.1);
        assert_eq!(back.restitution, 0.9);
        assert_eq!(back.linear_damping, 0.2);
    }
}
