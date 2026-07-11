// src/assets/procedural_mesh.rs
//
// `ProceduralMesh`'s `Component` impl is generated centrally (see
// `cn_impl_components!`); this module keeps the blob-residency helper
// `PhysicsSystem` relies on.

use crate::assets::ProceduralMesh;

// Blob indices of heightfield-generator ProceduralMeshes. GraphicsSystem's
// init release sweep must spare these blobs: PhysicsSystem inits afterwards and
// reads the baked heightfield collider grid from the payload, mirroring the
// AudioClip / SdfVolume precedent of holding a blob resident for a later system.
pub fn heightfield_blob_indices(
    ctx: &crate::ecs::PipelineContext,
) -> std::collections::HashSet<u32> {
    ctx.query::<ProceduralMesh>()
        .filter(|m| m.generator == "heightfield")
        .filter_map(|m| m.locator.as_ref().map(|l| l.blob_index))
        .collect()
}
