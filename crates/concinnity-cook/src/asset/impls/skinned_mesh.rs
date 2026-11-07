// A SkinnedMesh compiles its geometry into a payload while its placement,
// material and capsule fields bake into the record's data beside it.

use concinnity_engine::components::{SkeletonJoint, SkinnedMesh};
use concinnity_host::thread::asset_id;
use serde::Deserialize;

use crate::asset::{BuildAsset, BuildCtx, SourceFiles};

impl BuildAsset for SkinnedMesh {
    fn compile_payload(args: &serde_json::Value, _: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        compile_skinned_mesh_payload(args)
    }

    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> SourceFiles {
        super::resources::source_files(args, ctx.assets_dir)
    }

    fn compile_data(name: &str, args: &serde_json::Value) -> std::io::Result<Option<Vec<u8>>> {
        compile_skinned_mesh_data(name, args)
            .map(Some)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// Compile a SkinnedMesh's geometry payload: vertices + indices + the skeleton
// the desugar pass wrote into the args, with the LOD trailer. The placement /
// material fields ride separately in the record's `data_bytes` (see
// `compile_skinned_mesh_data`).
fn compile_skinned_mesh_payload(args: &serde_json::Value) -> std::io::Result<Vec<u8>> {
    let mesh: SkinnedMesh = Deserialize::deserialize(args)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    // `skeleton` is not a field on the schema struct; the desugar pass writes it
    // into the args JSON for glTF-sourced meshes, and inline-authored worlds may
    // carry it directly. Read it straight from the JSON so it can be baked into
    // the compiled payload alongside vertices and indices.
    let skeleton: Vec<SkeletonJoint> = match args.get("skeleton") {
        Some(v) => Deserialize::deserialize(v).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("SkinnedMesh: invalid skeleton args: {e}"),
            )
        })?,
        None => Vec::new(),
    };
    crate::compile::geometry::compile_skinned_mesh_payload_with_lods(
        &mesh.vertices,
        &mesh.indices,
        &skeleton,
        &mesh.morph_target_names,
        &mesh.morph_deltas,
        &crate::compile::geometry::SkinnedLods {
            levels: mesh.lod_levels,
            distances: &mesh.lod_distances,
        },
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// Bake a SkinnedMesh's runtime data (its placement, material/texture handles,
// capsule, and spawn reserve) into the resource record's `data_bytes`, applying
// the clamps that used to run in the retired `from_args`. The geometry is
// dropped from the baked form -- it rides the compiled payload. Serialized as a
// `(name_id, mesh)` JSON tuple: the struct carries no identity, so the interned
// name (which the runtime's spawn-by-name registration still needs) travels
// beside it.
fn compile_skinned_mesh_data(name: &str, args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mut sm: SkinnedMesh =
        Deserialize::deserialize(args).map_err(|e| format!("SkinnedMesh args: {e}"))?;
    // A zero scale would collapse the world matrix; clamp to a sane unit.
    if sm.scale == [0.0, 0.0, 0.0] {
        sm.scale = [1.0, 1.0, 1.0];
    }
    // 0 and 1 are equivalent (no alternates); cap at the 8-level LOD ceiling.
    if sm.lod_levels == 0 {
        sm.lod_levels = 1;
    }
    sm.lod_levels = sm.lod_levels.min(8);
    // A runaway reserve would pre-expand the skinned vertex buffer without
    // bound; cap it. The skinned index buffer is u16, so init applies a further
    // per-mesh limit when the copies would overflow that.
    sm.max_instances = sm.max_instances.min(4096);
    // Geometry rides the compiled payload, not the baked data. The morph
    // fields are geometry too: the desugar pass writes them into the args only
    // on a payload-cache miss, so leaving them here would make the record
    // depend on whether the payload was compiled or replayed.
    sm.vertices = Vec::new();
    sm.indices = Vec::new();
    sm.morph_target_names = Vec::new();
    sm.morph_deltas = Vec::new();
    let name_id = asset_id::intern(name);
    postcard::to_allocvec(&(name_id.0, sm)).map_err(|e| format!("SkinnedMesh serialize: {e}"))
}
