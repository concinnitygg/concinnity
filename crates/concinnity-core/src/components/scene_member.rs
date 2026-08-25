// src/components/scene_member.rs

use crate::ecs::asset_id::AssetId;

/// The `Scene` an entity belongs to, for per-scene show/hide.
///
/// Runtime-only. An entity without this component is visible in every scene.
/// Carries the scene identity a `Prop` resolves into its `scene` field.
#[derive(Debug, Clone, Copy, Default)]
pub struct SceneMember(pub AssetId);
