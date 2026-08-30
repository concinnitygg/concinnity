//! Resource handle assignment: the one place a resource's dense per-kind handle
//! is decided.
//!
//! A resource (a mesh, texture, material, ...) is addressed at runtime by an
//! index into its kind's table, and the table index *is* the handle, so the
//! order handles are handed out in is load-bearing. Both producers of a world
//! -- the cook pipeline over authored JSON, and the typed
//! [`bake`](crate::bake) builder -- assign through this module, so the two can
//! never drift.
//!
//! The rules:
//!
//! - Each [`ResourceKind`] counts independently from zero, in declaration
//!   order.
//! - Geometry draws from one shared `Mesh` space across all four producers,
//!   assigned in [`MeshBlock`] order and in declaration order within a block,
//!   because that is the order the runtime enumerates mesh sources in. The
//!   trailing [`MeshBlock::Runtime`] block belongs to the world itself and is
//!   assigned at load time, past everything a build hands out.
//! - Shaders have a space of their own: a `Shader` is a component rather than a
//!   resource, but a `Material`'s `shader` reference still bakes to a dense
//!   declaration-order index.

use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::ecs::ResourceKind;
use crate::ecs::asset_id::AssetId;

/// Which block of the shared mesh-source handle space an asset belongs to.
///
/// Handles are assigned block by block, so a `.mesh` reference resolves to the
/// same index the runtime reaches that geometry at while decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeshBlock {
    /// A `Mesh` resource, compiled into the blob's resource stream.
    Mesh,
    /// A `ProceduralMesh` component, whose payload rides its own def.
    ProceduralMesh,
    /// A `VoxelChunk` component.
    VoxelChunk,
    /// A mesh-kind `File` component.
    File,
    /// Geometry a running world baked for itself at start, whose payload is a
    /// [`RuntimeMeshPayloads`](super::RuntimeMeshPayloads) entry. Last, so
    /// minting one moves no handle the build handed out. Neither producer of a
    /// world assigns into it: the block exists at load time only.
    Runtime,
}

impl MeshBlock {
    /// The block's position in the assignment order.
    pub fn order(self) -> u8 {
        self as u8
    }
}

/// Per-kind handles assigned to each resource, keyed by its identity.
#[derive(Debug, Default, Clone)]
pub struct ResourceHandles {
    // Next unused handle per kind (the count assigned so far).
    next: HashMap<u8, u32>,
    // The handle each resource received.
    map: HashMap<(u8, AssetId), u32>,
    // Shader handles, a space of their own.
    shader_map: HashMap<AssetId, u32>,
    shader_next: u32,
}

impl ResourceHandles {
    /// Give one resource the next handle in its kind's space and record it.
    /// Declaration order in, dense `0..N` out.
    pub fn assign(&mut self, kind: ResourceKind, id: AssetId) -> u32 {
        let next = self.next.entry(kind as u8).or_insert(0);
        let handle = *next;
        *next += 1;
        self.map.insert((kind as u8, id), handle);
        handle
    }

    /// The handle a resource received, if it was assigned one.
    pub fn get(&self, kind: ResourceKind, id: AssetId) -> Option<u32> {
        self.map.get(&(kind as u8, id)).copied()
    }

    /// How many handles a kind has assigned: its table length.
    pub fn count(&self, kind: ResourceKind) -> u32 {
        self.next.get(&(kind as u8)).copied().unwrap_or(0)
    }

    /// Assign handles across a world's resources, in the order given. The
    /// caller has already classified each asset and passes only the resources;
    /// each kind counts independently from zero.
    pub fn from_assets(assets: impl IntoIterator<Item = (AssetId, ResourceKind)>) -> Self {
        let mut handles = Self::default();
        for (id, kind) in assets {
            handles.assign(kind, id);
        }
        handles
    }

    /// Assign the shared mesh-source handle space over a world's geometry
    /// producers, given in declaration order with the block each belongs to.
    /// Handles go out in block order, declaration order within a block.
    pub fn assign_mesh_sources(&mut self, sources: impl IntoIterator<Item = (AssetId, MeshBlock)>) {
        let mut sources: Vec<(AssetId, MeshBlock)> = sources.into_iter().collect();
        // A stable sort by block keeps declaration order within each one.
        sources.sort_by_key(|(_, block)| block.order());
        for (id, _) in sources {
            self.assign(ResourceKind::Mesh, id);
        }
    }

    /// Give one `Shader` the next handle in the shader space and record it.
    pub fn assign_shader(&mut self, id: AssetId) -> u32 {
        let handle = self.shader_next;
        self.shader_next += 1;
        self.shader_map.insert(id, handle);
        handle
    }

    /// The handle a `Shader` received, if it was assigned one.
    pub fn shader(&self, id: AssetId) -> Option<u32> {
        self.shader_map.get(&id).copied()
    }

    /// How many shader handles were assigned.
    pub fn shader_count(&self) -> u32 {
        self.shader_next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_are_dense_per_kind_in_declaration_order() {
        // An AudioClip lands in its own space, independent of the textures
        // declared before it.
        let handles = ResourceHandles::from_assets([
            (AssetId(10), ResourceKind::Texture),
            (AssetId(11), ResourceKind::Mesh),
            (AssetId(12), ResourceKind::Texture),
            (AssetId(20), ResourceKind::AudioClip),
            (AssetId(14), ResourceKind::Texture),
        ]);

        assert_eq!(handles.get(ResourceKind::Texture, AssetId(10)), Some(0));
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(12)), Some(1));
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(14)), Some(2));
        assert_eq!(handles.get(ResourceKind::Mesh, AssetId(11)), Some(0));
        assert_eq!(handles.get(ResourceKind::AudioClip, AssetId(20)), Some(0));

        assert_eq!(handles.count(ResourceKind::Texture), 3);
        assert_eq!(handles.count(ResourceKind::Mesh), 1);
        assert_eq!(handles.count(ResourceKind::Material), 0);
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(99)), None);
    }

    #[test]
    fn the_same_id_in_two_kinds_gets_independent_handles() {
        let mut handles = ResourceHandles::default();
        assert_eq!(handles.assign(ResourceKind::Texture, AssetId(1)), 0);
        assert_eq!(handles.assign(ResourceKind::Mesh, AssetId(1)), 0);
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(1)), Some(0));
        assert_eq!(handles.get(ResourceKind::Mesh, AssetId(1)), Some(0));
    }

    // The load-bearing invariant of the shared mesh space: block order first,
    // declaration order within a block, whatever order they were declared in.
    #[test]
    fn mesh_sources_are_block_ordered_across_kinds() {
        let mut handles = ResourceHandles::default();
        handles.assign_mesh_sources([
            (AssetId(0), MeshBlock::Mesh),
            (AssetId(1), MeshBlock::ProceduralMesh),
            (AssetId(2), MeshBlock::VoxelChunk),
            (AssetId(3), MeshBlock::Mesh),
            (AssetId(4), MeshBlock::File),
            (AssetId(5), MeshBlock::ProceduralMesh),
        ]);

        let h = |id: u32| handles.get(ResourceKind::Mesh, AssetId(id));
        assert_eq!(h(0), Some(0));
        assert_eq!(h(3), Some(1));
        assert_eq!(h(1), Some(2));
        assert_eq!(h(5), Some(3));
        assert_eq!(h(2), Some(4));
        assert_eq!(h(4), Some(5));
        assert_eq!(handles.count(ResourceKind::Mesh), 6);
    }

    #[test]
    fn mesh_blocks_run_mesh_procedural_voxel_file_then_runtime() {
        assert_eq!(MeshBlock::Mesh.order(), 0);
        assert_eq!(MeshBlock::ProceduralMesh.order(), 1);
        assert_eq!(MeshBlock::VoxelChunk.order(), 2);
        assert_eq!(MeshBlock::File.order(), 3);
        // The world's own block trails every build-assigned one, so minting
        // geometry at start cannot move a handle already baked into a Prop.
        assert_eq!(MeshBlock::Runtime.order(), 4);
    }

    #[test]
    fn shaders_count_in_a_space_of_their_own() {
        let mut handles = ResourceHandles::default();
        handles.assign(ResourceKind::Material, AssetId(7));
        assert_eq!(handles.assign_shader(AssetId(7)), 0);
        assert_eq!(handles.assign_shader(AssetId(8)), 1);
        assert_eq!(handles.shader(AssetId(7)), Some(0));
        assert_eq!(handles.shader(AssetId(8)), Some(1));
        assert_eq!(handles.shader(AssetId(9)), None);
        assert_eq!(handles.shader_count(), 2);
    }
}
