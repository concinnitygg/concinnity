//! Init-captured state GraphicsSystem parks as world resources beside the
//! backend, for the tooling that edits a running world through it.

use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::volumetric_fog::FogSettings;
use std::collections::HashMap;

/// Texture asset name -> live pool slot, so a runtime decal or emitter spawn can
/// resolve an authored Texture name. Parked only when hot-reload capture is on.
#[derive(Debug, Default)]
pub struct TextureNameSlots(pub HashMap<AssetId, usize>);

/// The fog settings last pushed to the backend, `None` while fog is off. A
/// world reload compares against it and skips an unchanged value.
#[derive(Debug, Default)]
pub struct PushedFogSettings(pub Option<FogSettings>);
