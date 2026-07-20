// src/assets/visibility_request.rs

use crate::ecs::asset_id::AssetId;

// Runtime-only event requesting that a named entity (and its descendants)
// be hidden or shown without despawning it. SpawnSystem drains these each
// step, toggling the subtree's Hidden tags and its draw slots' visibility.
// World authors never declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct VisibilityRequest {
    pub name: AssetId,
    pub visible: bool,
}
