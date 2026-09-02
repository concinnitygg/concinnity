//! Payloads a running world bakes for itself at start.
//!
//! A geometry producer the build compiled carries a [`PayloadLocator`] into a
//! blob section. One the world mints at start has no blob behind it, so its
//! baked bytes live here and the renderer reads them straight out.

use alloc::vec::Vec;

use crate::ecs::asset_id::AssetId;

/// Mesh payloads baked at world start, each under the asset id of the value
/// that owns it, in the order they were installed.
///
/// Their handles sit in the [`MeshBlock::Runtime`](super::MeshBlock::Runtime)
/// tail of the shared mesh-source space, past every handle the build assigned,
/// so the position here is the handle's offset into that tail.
#[derive(Debug, Clone, Default)]
pub struct RuntimeMeshPayloads(Vec<(AssetId, Vec<u8>)>);

impl RuntimeMeshPayloads {
    /// Install `payload` as `id`'s geometry, at the next handle in the tail.
    pub fn push(&mut self, id: AssetId, payload: Vec<u8>) {
        self.0.push((id, payload));
    }

    /// The baked payload for one asset, if the world minted its geometry.
    pub fn get(&self, id: AssetId) -> Option<&[u8]> {
        self.0
            .iter()
            .find(|(owner, _)| *owner == id)
            .map(|(_, b)| &b[..])
    }

    /// The payloads in handle order, each with its owner's id.
    pub fn iter(&self) -> impl Iterator<Item = (AssetId, &[u8])> {
        self.0.iter().map(|(id, b)| (*id, &b[..]))
    }

    /// How many were baked at start.
    pub fn len(&self) -> usize {
        self.0.len()
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
        payloads.push(AssetId(7), vec![1, 2, 3]);
        assert_eq!(payloads.get(AssetId(7)), Some(&[1u8, 2, 3][..]));
        assert_eq!(payloads.get(AssetId(8)), None);
        assert!(!payloads.is_empty());
        assert_eq!(payloads.len(), 1);
    }

    #[test]
    fn payloads_keep_the_order_they_were_installed_in() {
        let mut payloads = RuntimeMeshPayloads::default();
        payloads.push(AssetId(9), vec![1]);
        payloads.push(AssetId(4), vec![2]);
        let order: Vec<AssetId> = payloads.iter().map(|(id, _)| id).collect();
        assert_eq!(order, vec![AssetId(9), AssetId(4)]);
    }
}
