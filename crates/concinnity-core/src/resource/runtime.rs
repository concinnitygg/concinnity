//! Payloads a running world bakes for itself at start.
//!
//! A geometry producer the build compiled carries a [`PayloadLocator`] into a
//! blob section. One the world mints at start has no blob behind it, so its
//! baked bytes live here and the renderer reads them straight out.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::ecs::asset_id::AssetId;

/// Mesh payloads baked at world start, keyed by the asset id of the
/// `ProceduralMesh` that owns each one.
///
/// Their handles sit in the [`MeshBlock::Runtime`](super::MeshBlock::Runtime)
/// tail of the shared mesh-source space, past every handle the build assigned.
#[derive(Debug, Clone, Default)]
pub struct RuntimeMeshPayloads(pub BTreeMap<AssetId, Vec<u8>>);

impl RuntimeMeshPayloads {
    /// The baked payload for one asset, if the world minted its geometry.
    pub fn get(&self, id: AssetId) -> Option<&[u8]> {
        self.0.get(&id).map(|b| &b[..])
    }

    /// Whether nothing was baked at start.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_baked_payload_is_found_by_its_owner_id() {
        let mut payloads = RuntimeMeshPayloads::default();
        assert!(payloads.is_empty());
        payloads.0.insert(AssetId(7), vec![1, 2, 3]);
        assert_eq!(payloads.get(AssetId(7)), Some(&[1u8, 2, 3][..]));
        assert_eq!(payloads.get(AssetId(8)), None);
        assert!(!payloads.is_empty());
    }
}
