// src/assets/procedural_mesh.rs

use crate::assets::ProceduralMesh;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

impl Component for ProceduralMesh {
    const NAME: &'static str = "ProceduralMesh";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

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
