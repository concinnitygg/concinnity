//! Asset hot-reload (`cn debug`) decode helpers: re-import a file-backed Mesh /
//! SkinnedMesh from a pre-parsed glTF document into the runtime Vertex /
//! SkinnedVertex form, mirroring the build pipeline so a hot-reloaded mesh is
//! byte-identical to a fresh `cn build`. The runtime crate links no image / glTF
//! decoders, so these live here in the build crate; the editor's debug server
//! drives them.

use crate::geometry::{
    compile_mesh_payload, compile_skinned_mesh_payload_with_lods, payload_joints_to_defs,
};
use concinnity_cpu::gfx::mesh_payload::{deserialise_skinned, deserialise_with_lods};

// LOD alternates: (switch_distance, index buffer) pairs.
type LodAlternates = Vec<(f32, Vec<u16>)>;

// Imported skinned mesh: runtime vertices, indices, and the bind-pose skeleton.
type SkinnedImport = (
    Vec<concinnity_cpu::gfx::mesh_payload::SkinnedVertex>,
    Vec<u16>,
    Vec<concinnity_core::assets::SkeletonJoint>,
);

/// Decode a file-backed `Mesh` primitive from a pre-parsed glTF document the
/// same way the build pipeline does at compile time, returning the runtime
/// `Vertex` / index form with normals + tangents + optional LOD alternates baked
/// in. Used by the asset hot-reload path (`cn debug` only); production reads the
/// compiled payload from a blob locator and goes through
/// `deserialise_with_lods` instead.
///
/// The caller is responsible for parsing the `.glb` (via [`crate::glb::parse_glb`])
/// so a single reload pass can amortise the parse across every `Mesh` that
/// references the same file: `ABeautifulGame` alone fans 35+ Mesh assets out of
/// one `.glb`.
///
/// `primitive_index` selects which primitive (flattened across glTF meshes) to
/// import; `lod_levels` and `lod_distances` mirror the asset declaration so the
/// reload produces a byte-identical payload to the build pass. The third
/// component of the result is empty for `lod_levels <= 1`.
pub fn decode_mesh_from_parsed_glb(
    doc: &crate::gltf_source::GltfDoc,
    source: &str,
    primitive_index: u32,
    lod_levels: u32,
    lod_distances: &[f32],
) -> Result<
    (
        Vec<concinnity_cpu::gfx::mesh_payload::Vertex>,
        Vec<u16>,
        LodAlternates,
    ),
    String,
> {
    let (vertex_data, indices) =
        crate::glb::import_static_glb_primitive_from_doc(doc, source, primitive_index)?;
    // Rebuild the JSON args the desugar pass would have produced, then run the
    // existing compile + deserialise cycle. This keeps the runtime path
    // byte-identical to the build pass so any difference is a build bug, not a
    // reload-only divergence.
    let args = serde_json::json!({
        "vertices": vertex_data,
        "indices": indices,
        "lod_levels": lod_levels,
        "lod_distances": lod_distances,
    });
    let payload = compile_mesh_payload(&args)?;
    deserialise_with_lods(&payload)
}

/// Decode a file-backed `SkinnedMesh` from a pre-parsed glTF document the same
/// way the build pipeline does at compile time, returning the runtime
/// `SkinnedVertex` / index form (normals + tangents baked in) plus the imported
/// bind-pose skeleton. Used by the asset hot-reload path (`cn debug` only);
/// production reads the compiled payload from a blob locator and goes through
/// `deserialise_skinned` instead.
///
/// The caller is responsible for parsing the `.glb` (via [`crate::glb::parse_glb`])
/// so a single reload pass can amortise the parse across every `Mesh` /
/// `SkinnedMesh` that references the same file. The skeleton is returned in the
/// same `SkeletonJoint` form the `SkinnedMesh` asset args carry; the reload helper
/// checks it against the init-time joint count before pushing to the GPU.
pub fn decode_skinned_from_parsed_glb(
    doc: &crate::gltf_source::GltfDoc,
    source: &str,
    skin_index: u32,
) -> Result<SkinnedImport, String> {
    let imported = crate::glb::import_skinned_from_doc(doc, source, skin_index)?;
    let payload = compile_skinned_mesh_payload_with_lods(
        &imported.vertices,
        &imported.indices,
        &imported.skeleton,
        &imported.morph_target_names,
        &imported.morph_deltas,
        1,
        &[],
    )?;
    let (verts, idxs, payload_joints) = deserialise_skinned(&payload)?;
    let skeleton = payload_joints_to_defs(payload_joints);
    Ok((verts, idxs, skeleton))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glb::test_fixtures::{
        make_glb, parse, skinned_bin, skinned_glb, skinned_json, static_triangle_glb,
    };

    #[test]
    fn decode_static_mesh_round_trips_a_triangle() {
        let doc = parse(&static_triangle_glb());
        let (vertices, indices, lods) =
            decode_mesh_from_parsed_glb(&doc, "t.glb", 0, 1, &[]).expect("decode");
        assert_eq!(vertices.len(), 3);
        assert_eq!(indices.len(), 3);
        // A single LOD level produces no alternates.
        assert!(lods.is_empty());
    }

    #[test]
    fn decode_static_mesh_rejects_out_of_range_primitive() {
        let doc = parse(&static_triangle_glb());
        let err = decode_mesh_from_parsed_glb(&doc, "t.glb", 3, 1, &[]).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn decode_skinned_mesh_round_trips_vertices_and_the_bind_skeleton() {
        let doc = parse(&skinned_glb());
        let (vertices, indices, skeleton) =
            decode_skinned_from_parsed_glb(&doc, "s.glb", 0).expect("decode");
        assert_eq!(vertices.len(), 3);
        assert_eq!(indices, vec![0u16, 1, 2]);

        // The skin authors its joints child-first; the payload round trip keeps
        // the importer's parents-before-children order.
        assert_eq!(skeleton.len(), 2);
        assert_eq!(skeleton[0].name, "root");
        assert_eq!(skeleton[0].parent, -1);
        assert_eq!(skeleton[1].name, "tip");
        assert_eq!(skeleton[1].parent, 0);
        assert_eq!(skeleton[1].translation, [0.0, 0.5, 0.0]);

        // Vertex bindings are remapped into that order, and the compile step
        // baked normals for the triangle's plane.
        assert_eq!(vertices[0].joints, [1, 1, 1, 1]);
        assert_eq!(vertices[0].weights, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(vertices[0].normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn decode_skinned_mesh_surfaces_a_payload_compile_failure() {
        // The glTF importer passes indices through untouched, so a triangle
        // referencing a vertex the primitive does not have is caught by the
        // payload compile rather than reaching the runtime.
        let mut bin = skinned_bin();
        bin[40..42].copy_from_slice(&9u16.to_le_bytes());
        let doc = parse(&make_glb(&skinned_json(true, true, false), Some(&bin)));
        let err = decode_skinned_from_parsed_glb(&doc, "s.glb", 0).unwrap_err();
        assert_eq!(err, "SkinnedMesh index out of range in triangle 0");
    }

    #[test]
    fn decode_skinned_mesh_rejects_a_file_with_no_skinned_node() {
        let doc = parse(&static_triangle_glb());
        let err = decode_skinned_from_parsed_glb(&doc, "t.glb", 0).unwrap_err();
        assert!(
            err.contains("no node with both a mesh and a skin"),
            "got: {err}"
        );
    }
}
