//! Init-captured state GraphicsSystem parks as world resources beside the
//! backend, for the tooling that edits a running world through it.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::volumetric_fog::FogSettings;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

/// Texture asset name -> live pool slot, so a runtime decal or emitter spawn can
/// resolve an authored Texture name. Parked only when hot-reload capture is on.
#[derive(Debug, Default)]
pub struct TextureNameSlots(pub HashMap<AssetId, usize>);

/// The fog settings last pushed to the backend, `None` while fog is off. A
/// world reload compares against it and skips an unchanged value.
#[derive(Debug, Default)]
pub struct PushedFogSettings(pub Option<FogSettings>);

/// Hot-reloaded Shader programs that stand in for the cooked ones, keyed by
/// shader bucket. Clones share one set: hot reload writes it and the streaming
/// pump installs from it when a Shader's scene loads after the edit. Created
/// fresh by each init under hot-reload capture, so an override never outlives
/// the world its bucket numbering belongs to; `cn build` is what persists an
/// edit.
#[derive(Debug, Clone, Default)]
pub struct ShaderOverrides(Arc<Mutex<HashMap<u32, Arc<ShaderPrograms>>>>);

impl ShaderOverrides {
    /// Replace `bucket`'s programs.
    pub fn set(&self, bucket: u32, programs: Arc<ShaderPrograms>) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(bucket, programs);
    }

    /// The programs standing in for `bucket`, if it was hot-reloaded.
    pub fn get(&self, bucket: u32) -> Option<Arc<ShaderPrograms>> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&bucket)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn programs(name: &str) -> Arc<ShaderPrograms> {
        Arc::new(ShaderPrograms {
            name: name.to_string(),
            ..Default::default()
        })
    }

    // A clone reads what the original writes, and a newer edit replaces the
    // older one per bucket.
    #[test]
    fn clones_share_one_set_and_a_newer_edit_replaces_an_older_one() {
        let overrides = ShaderOverrides::default();
        let reader = overrides.clone();
        assert!(reader.get(1).is_none());
        overrides.set(1, programs("first"));
        overrides.set(2, programs("other"));
        overrides.set(1, programs("second"));
        assert_eq!(reader.get(1).unwrap().name, "second");
        assert_eq!(reader.get(2).unwrap().name, "other");
    }

    // A fresh set, as each init creates, carries nothing over.
    #[test]
    fn a_fresh_set_is_empty() {
        let old = ShaderOverrides::default();
        old.set(1, programs("edit"));
        assert!(ShaderOverrides::default().get(1).is_none());
    }
}
