// Resource handle assignment.
//
// A resource (mesh, texture, material, ...) is addressed at runtime by a dense
// per-kind handle. Today the renderer assigns those indices itself at init (it
// drains the component column and enumerates it) and resolves every `AssetId`
// reference to one by scanning. This module moves the assignment to build time:
// cook walks the world's assets, gives each resource the next handle in its
// kind's `0..N` space (declaration order), and records `AssetId -> handle` so a
// later pass can resolve a reference to a handle without a runtime scan.
//
// This is the additive first step of the resource-table migration; nothing
// consumes the map yet.

use crate::ecs::asset_id::AssetId;
use crate::registry::ComponentType;
use std::collections::HashMap;

// The kinds of resource the runtime keeps in per-kind tables. One dense handle
// space per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Mesh,
    Texture,
    Material,
    Font,
    AudioClip,
    CubemapTexture,
    EnvironmentMap,
    ColorLut,
    SkinnedMesh,
}

// The resource kind an asset type is, or `None` if it is not (yet) a resource.
pub fn resource_kind(ct: ComponentType) -> Option<ResourceKind> {
    Some(match ct {
        ComponentType::Mesh => ResourceKind::Mesh,
        ComponentType::Texture => ResourceKind::Texture,
        ComponentType::Material => ResourceKind::Material,
        ComponentType::Font => ResourceKind::Font,
        ComponentType::AudioClip => ResourceKind::AudioClip,
        ComponentType::CubemapTexture => ResourceKind::CubemapTexture,
        ComponentType::EnvironmentMap => ResourceKind::EnvironmentMap,
        ComponentType::ColorLut => ResourceKind::ColorLut,
        ComponentType::SkinnedMesh => ResourceKind::SkinnedMesh,
        _ => return None,
    })
}

// Per-kind handles assigned to each resource asset, keyed by its identity.
#[derive(Debug, Default)]
pub struct ResourceHandles {
    // Next unused handle per kind (the count assigned so far).
    next: HashMap<ResourceKind, u32>,
    // The handle each resource asset received.
    map: HashMap<(ResourceKind, AssetId), u32>,
}

impl ResourceHandles {
    // Give one resource the next handle in its kind's space and record it.
    // Declaration order in, dense `0..N` out.
    pub fn assign(&mut self, kind: ResourceKind, id: AssetId) -> u32 {
        let next = self.next.entry(kind).or_insert(0);
        let handle = *next;
        *next += 1;
        self.map.insert((kind, id), handle);
        handle
    }

    // The handle a resource received, if it was assigned one.
    pub fn get(&self, kind: ResourceKind, id: AssetId) -> Option<u32> {
        self.map.get(&(kind, id)).copied()
    }

    // How many handles a kind has assigned (its table length).
    pub fn count(&self, kind: ResourceKind) -> u32 {
        self.next.get(&kind).copied().unwrap_or(0)
    }

    // Assign handles across a world's assets, in the order given. Non-resource
    // assets are skipped; each resource kind counts independently from zero.
    pub fn from_assets(assets: impl IntoIterator<Item = (AssetId, ComponentType)>) -> Self {
        let mut handles = Self::default();
        for (id, ct) in assets {
            if let Some(kind) = resource_kind(ct) {
                handles.assign(kind, id);
            }
        }
        handles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_types_classify_others_do_not() {
        assert_eq!(
            resource_kind(ComponentType::Texture),
            Some(ResourceKind::Texture)
        );
        assert_eq!(resource_kind(ComponentType::Mesh), Some(ResourceKind::Mesh));
        assert_eq!(
            resource_kind(ComponentType::Material),
            Some(ResourceKind::Material)
        );
        // Pure-data components and containers are not resources.
        assert_eq!(resource_kind(ComponentType::PointLight), None);
        assert_eq!(resource_kind(ComponentType::Prop), None);
        assert_eq!(resource_kind(ComponentType::Transform), None);
    }

    #[test]
    fn handles_are_dense_per_kind_in_order() {
        let assets = [
            (AssetId(10), ComponentType::Texture),
            (AssetId(11), ComponentType::Mesh),
            (AssetId(12), ComponentType::Texture),
            (AssetId(13), ComponentType::PointLight), // skipped: not a resource
            (AssetId(14), ComponentType::Texture),
        ];
        let handles = ResourceHandles::from_assets(assets);

        // Each kind counts independently from zero, in declaration order.
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(10)), Some(0));
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(12)), Some(1));
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(14)), Some(2));
        assert_eq!(handles.get(ResourceKind::Mesh, AssetId(11)), Some(0));

        assert_eq!(handles.count(ResourceKind::Texture), 3);
        assert_eq!(handles.count(ResourceKind::Mesh), 1);
        assert_eq!(handles.count(ResourceKind::Material), 0);

        // A non-resource asset and an unassigned id have no handle.
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(13)), None);
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(99)), None);
    }

    #[test]
    fn the_same_id_in_two_kinds_gets_independent_handles() {
        // Handle spaces are per kind, so the same AssetId can hold a Texture 0
        // and a Mesh 0 without collision (they are distinct resources).
        let mut handles = ResourceHandles::default();
        assert_eq!(handles.assign(ResourceKind::Texture, AssetId(1)), 0);
        assert_eq!(handles.assign(ResourceKind::Mesh, AssetId(1)), 0);
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(1)), Some(0));
        assert_eq!(handles.get(ResourceKind::Mesh, AssetId(1)), Some(0));
    }
}
