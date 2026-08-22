// src/assets/visibility_request.rs

use crate::assets::EntityTarget;

/// Runtime-only event requesting that an entity (and its descendants)
/// be hidden or shown without despawning it. SpawnSystem drains these each
/// step, toggling the subtree's Hidden tags and its draw slots' visibility.
/// World authors never declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct VisibilityRequest {
    /// The entity whose subtree is toggled.
    pub target: EntityTarget,
    /// `true` to show the subtree, `false` to hide it.
    pub visible: bool,
}
