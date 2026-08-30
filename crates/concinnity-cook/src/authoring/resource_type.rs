//! Which resource kind an authored asset compiles into.
//!
//! The resource asset types are one group of the shared registry list, so their
//! naming, parsing and authoring metadata come from `RegisteredType` like every
//! other type's. What lives here is the part a registry entry cannot state: the
//! mesh-source handle space, which four different type groups draw from in a
//! fixed order, and the `File` classifier, which needs an asset's args rather
//! than just its type name. The per-type compile dispatch lives in
//! concinnity-cook, where the compilers are.

// The resource kinds (and their `resource_kind` blob tag) are defined with the
// blob format; re-exported here for the classifiers and the cook-side handle
// assigner so both sides agree on the kind and its tag. `MeshBlock` comes from
// the same crate, with the handle-assignment rules it orders.
pub use concinnity_core::ecs::ResourceKind;
pub(crate) use concinnity_core::resource::MeshBlock;

// The mesh-source handle space is shared across every geometry-producing kind
// (Mesh, ProceduralMesh, VoxelChunk, and mesh-kind File), so it is not assigned
// through the type-name classifier above: File is polymorphic (only a mesh-kind
// File is geometry, which the type name alone cannot tell), and the four kinds
// must draw from one dense space in a fixed order. The assignment itself is
// `concinnity_core::resource::ResourceHandles`, shared with the typed bake
// builder; what lives here is the classifier that reads authored args.

// Normalize an asset type name the same way the cross-reference checker does, so
// both agree on what counts as a mesh source.
fn norm_type(t: &str) -> String {
    t.to_lowercase().replace('_', "")
}

// True when a `File`'s args name a mesh-kind file (the only File that produces
// geometry).
fn file_is_mesh(args: &serde_json::Value) -> bool {
    args.get("kind")
        .and_then(|k| k.as_str())
        .and_then(crate::components::FileKind::from_ext)
        .map(|fk| fk.is_mesh())
        .unwrap_or(false)
}

/// The mesh-source block an asset belongs to, or `None` if it is not a geometry
/// producer. The blocks are the fixed order the runtime enumerates mesh sources
/// in, so a handle assigned in block order equals the runtime's mesh-source
/// index.
pub(crate) fn mesh_source_block(asset_type: &str, args: &serde_json::Value) -> Option<MeshBlock> {
    match norm_type(asset_type).as_str() {
        "mesh" => Some(MeshBlock::Mesh),
        "proceduralmesh" => Some(MeshBlock::ProceduralMesh),
        "voxelchunk" | "chunk" => Some(MeshBlock::VoxelChunk),
        "file" => file_is_mesh(args).then_some(MeshBlock::File),
        _ => None,
    }
}

/// Whether an asset produces geometry addressable by a mesh handle. The single
/// classifier the cross-reference checker and the handle assigner share.
pub(crate) fn is_mesh_source(asset_type: &str, args: &serde_json::Value) -> bool {
    mesh_source_block(asset_type, args).is_some()
}

/// The resource kind a declarable asset type name maps to, or `None` for a
/// non-resource type. The single classifier the build uses to assign handles
/// over the world's assets.
pub(crate) fn asset_resource_kind(asset_type: &str) -> Option<ResourceKind> {
    let ty = crate::authoring::registry::RegisteredType::parse(asset_type)?;
    // Mesh draws from the shared mesh-source handle space (assigned by cook's
    // `assign_mesh_source_handles` in block order across all four geometry
    // producers), so the per-kind declaration-order classifier must not also
    // assign it.
    if ty == crate::authoring::registry::RegisteredType::Mesh {
        return None;
    }
    ty.resource_kind()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::registry::RegisteredType;

    // Every resource asset is a registered type like any other; what marks it as
    // a resource is the handle space it reports, not which registry it is in.
    #[test]
    fn resource_types_report_a_kind_and_others_do_not() {
        for (name, kind) in [
            ("Texture", ResourceKind::Texture),
            ("CubemapTexture", ResourceKind::CubemapTexture),
            ("EnvironmentMap", ResourceKind::EnvironmentMap),
            ("ColorLut", ResourceKind::ColorLut),
            ("Font", ResourceKind::Font),
            ("Material", ResourceKind::Material),
            ("SkinnedMesh", ResourceKind::SkinnedMesh),
            ("AudioClip", ResourceKind::AudioClip),
        ] {
            let ty = RegisteredType::parse(name).expect("a declarable type");
            assert_eq!(ty.resource_kind(), Some(kind), "{name}");
            assert_eq!(asset_resource_kind(name), Some(kind), "{name}");
            assert_eq!(ty.discriminant(), None, "{name} is not stored in a column");
        }

        // A stored component reports no resource kind.
        let prop = RegisteredType::parse("Prop").expect("Prop is registered");
        assert_eq!(prop.resource_kind(), None);
        assert_eq!(asset_resource_kind("Prop"), None);
        assert!(prop.discriminant().is_some());
    }

    // Mesh is a resource, but its handle is not assigned by the type-name
    // classifier: it shares the mesh-source handle space with ProceduralMesh /
    // VoxelChunk / File, assigned in block order by cook's
    // `assign_mesh_source_handles`.
    #[test]
    fn mesh_is_a_resource_but_not_classified_by_name() {
        let mesh = RegisteredType::parse("Mesh").expect("Mesh is registered");
        assert_eq!(mesh.resource_kind(), Some(ResourceKind::Mesh));
        assert_eq!(asset_resource_kind("Mesh"), None);
        assert!(is_mesh_source("Mesh", &serde_json::json!({})));
    }

    // Material's compiled bytes ride inline in its record; every other resource
    // points at a payload section.
    #[test]
    fn only_material_is_a_data_resource() {
        for &ty in RegisteredType::all() {
            let expected = ty.as_str() == "Material";
            assert_eq!(ty.is_data(), expected, "{}", ty.as_str());
        }
    }

    // The typed round-trip fills the schema defaults and rejects what the
    // schema cannot hold, so authoring tools can check a resource's args
    // without paying for its payload compile.
    #[test]
    fn normalized_args_fills_defaults_and_rejects_bad_types() {
        let env = RegisteredType::parse("EnvironmentMap").expect("registered");
        let out = env
            .normalized_args(&serde_json::json!({"source": "studio.hdr"}))
            .expect("a source-only EnvironmentMap round-trips");
        assert_eq!(out["source"], "studio.hdr");
        assert_eq!(out["prefilter_face_size"], 512, "defaults are materialized");

        // A negative value cannot land in a u32 field.
        RegisteredType::parse("Font")
            .expect("registered")
            .normalized_args(&serde_json::json!({"size_px": -5}))
            .expect_err("a negative u32 must be rejected");
    }

    #[test]
    fn mesh_source_blocks_follow_the_fixed_order() {
        let none = serde_json::json!({});
        assert_eq!(mesh_source_block("Mesh", &none), Some(MeshBlock::Mesh));
        assert_eq!(
            mesh_source_block("ProceduralMesh", &none),
            Some(MeshBlock::ProceduralMesh)
        );
        assert_eq!(
            mesh_source_block("VoxelChunk", &none),
            Some(MeshBlock::VoxelChunk)
        );
        // Only a mesh-kind File is a geometry producer.
        assert_eq!(
            mesh_source_block("File", &serde_json::json!({"kind": "obj"})),
            Some(MeshBlock::File)
        );
        assert_eq!(
            mesh_source_block("File", &serde_json::json!({"kind": "png"})),
            None
        );
        assert_eq!(mesh_source_block("PointLight", &none), None);
    }
}
