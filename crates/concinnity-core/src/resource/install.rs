//! Installing a baked resource into a running world: the payload or bytes go
//! where the owning system reads them, and the caller gets the dense handle a
//! component references them by. Shared by the world-start defaults pass and
//! by [`World`](crate::ecs::World)'s data-entry methods, so a resource the
//! engine injects and one an application hands over land identically.

use alloc::vec::Vec;

use crate::components::{File, FileKind, Material, ProceduralMesh, VoxelChunk, validate};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{EnvironmentMapHandle, MaterialHandle, MeshHandle, PipelineContext};

use super::{EnvironmentMapTable, MaterialTable, MeshTable, ResourceEntry, RuntimeMeshPayloads};

/// Record `payload` as `id`'s geometry and return the handle it lands on.
///
/// A mesh installed here trails the build's four compiled blocks (the build
/// assigns Mesh, ProceduralMesh, VoxelChunk, and File blocks in that order),
/// so it moves no handle a compiled world baked into a Prop.
pub fn append_mesh(ctx: &mut PipelineContext, id: AssetId, payload: Vec<u8>) -> MeshHandle {
    let handle = build_assigned(ctx) + runtime_count(ctx);
    if ctx.resource::<RuntimeMeshPayloads>().is_none() {
        ctx.insert_resource(RuntimeMeshPayloads::default());
    }
    if let Some(payloads) = ctx.resource_mut::<RuntimeMeshPayloads>() {
        payloads.0.insert(id, payload);
    }
    MeshHandle(handle as u32)
}

/// Install `material` into the world's material table and return its handle.
/// A material is a data resource: its clamped parameters are the whole entry.
pub fn append_material(ctx: &mut PipelineContext, material: Material) -> MaterialHandle {
    let bytes = postcard::to_allocvec(&validate::material(material))
        .expect("a Material is a plain struct; postcard cannot fail on one");
    if ctx.resource::<MaterialTable>().is_none() {
        ctx.insert_resource(MaterialTable::default());
    }
    let table = ctx
        .resource_mut::<MaterialTable>()
        .expect("the table was just ensured");
    MaterialHandle(table.append(ResourceEntry {
        payload: None,
        data_bytes: bytes,
    }))
}

/// Install a baked IBL `payload` into the world's environment-map table and
/// return its handle. The renderer lights with the map at handle 0.
pub fn append_environment_map(ctx: &mut PipelineContext, payload: Vec<u8>) -> EnvironmentMapHandle {
    if ctx.resource::<EnvironmentMapTable>().is_none() {
        ctx.insert_resource(EnvironmentMapTable::default());
    }
    let table = ctx
        .resource_mut::<EnvironmentMapTable>()
        .expect("the table was just ensured");
    EnvironmentMapHandle(table.append(ResourceEntry::baked(payload)))
}

// How many mesh handles the build handed out: the four compiled blocks,
// counted the way the renderer enumerates them.
fn build_assigned(ctx: &PipelineContext) -> usize {
    let meshes = ctx.resource::<MeshTable>().map_or(0, MeshTable::len);
    let procedural = ctx
        .query::<ProceduralMesh>()
        .filter(|m| m.locator.is_some())
        .count();
    let voxels = ctx.query::<VoxelChunk>().count();
    let files = ctx
        .query::<File>()
        .filter(|f| f.kind.as_ref().is_some_and(FileKind::is_mesh))
        .count();
    meshes + procedural + voxels + files
}

// Geometry already installed at runtime, which the trailing block counts
// before it.
fn runtime_count(ctx: &PipelineContext) -> usize {
    ctx.resource::<RuntimeMeshPayloads>()
        .map_or(0, |p| p.0.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;
    use alloc::vec;

    #[test]
    fn installed_resources_take_dense_handles_in_call_order() {
        let mut world = World::default();
        let mut ctx = world.context();
        let first = append_material(&mut ctx, Material::default());
        let second = append_material(&mut ctx, Material::default());
        assert_eq!((first.0, second.0), (0, 1));

        let map = append_environment_map(&mut ctx, vec![1, 2, 3]);
        assert_eq!(map.0, 0);
        let table = ctx
            .resource::<EnvironmentMapTable>()
            .expect("the table exists");
        assert_eq!(table.0[0].baked_bytes(), Some(&[1u8, 2, 3][..]));
    }

    #[test]
    fn an_installed_mesh_trails_the_builds_blocks() {
        let mut world = World::default();
        let mut ctx = world.context();
        let first = append_mesh(&mut ctx, AssetId(7), vec![1]);
        let second = append_mesh(&mut ctx, AssetId(8), vec![2]);
        assert_eq!((first.0, second.0), (0, 1));
        let payloads = ctx
            .resource::<RuntimeMeshPayloads>()
            .expect("the payload store exists");
        assert_eq!(payloads.0.len(), 2);
    }
}
