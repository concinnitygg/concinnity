// src/assets/collider.rs

use crate::assets::PropCollider;

/// Collision volume attached to an entity, in local space scaled by the
/// entity's transform.
///
/// Runtime-only. Carries the same shape description a `Prop` declares through
/// its `collider` field.
#[derive(Debug, Clone, Default)]
pub struct Collider(pub PropCollider);
