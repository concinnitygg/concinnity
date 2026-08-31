//! Imports a skinned mesh + skeleton from a glTF file (binary `.glb` or text
//! `.gltf` with external / data-URI buffers) into the inline `SkinnedMesh`
//! asset fields. glTF animations are not imported here: the mesh lands in its
//! bind pose.
//!
//! glTF stores a skin's joints in an arbitrary order; this engine's `SkeletonJoint`
//! list requires parents before children. Joints are therefore topologically
//! reordered and a remap table rewrites both each joint's parent index and
//! every vertex's `JOINTS_0` binding into the new index space.

// The `.glb` container decode lives in `crate::import::glb`; the asset-level
// desugar wrappers here call into it for parsing, shared geometry reads, and
// the Imported* payload types.
use std::path::Path;

use crate::import::glb::{
    ImportedAnimation, ImportedSkinnedMesh, import_glb_animations_from_doc,
    import_skinned_from_doc, parse_glb,
};

// Parse a glTF file into inline `SkinnedMesh` geometry plus a
// parents-before-children skeleton. `skin_index` selects among the file's
// skinned nodes in declaration order; other nodes, materials, cameras, and
// animations are ignored.
pub(crate) fn import_skinned_glb(
    source: &str,
    skin_index: u32,
    assets_dir: Option<&Path>,
) -> Result<ImportedSkinnedMesh, String> {
    let doc = parse_glb(source, assets_dir)?;
    import_skinned_from_doc(&doc, source, skin_index)
}

// Vertex count for the indexed primitive without reading any vertex data,
// used by `cn add` to decide whether a primitive fits Concinnity's u16 index
// limit or needs splitting.
#[cfg(test)]
pub(crate) fn primitive_vertex_count(
    doc: &crate::import::gltf_source::GltfDoc,
    primitive_index: u32,
) -> Option<usize> {
    doc.doc
        .document
        .meshes()
        .flat_map(|m| m.primitives())
        .nth(primitive_index as usize)
        .and_then(|p| p.get(&gltf::Semantic::Positions))
        .map(|a| a.count())
}

// Import every animation in a `.glb` whose channels target joints of the
// `skin_index`-th skinned node. Channels whose target node is not a skin
// joint, or whose interpolation method we cannot honour, are dropped
// silently; per-clip warnings would spam build output for files that mix
// joint and non-joint animations (e.g. character + camera).
pub(crate) fn import_glb_animations(
    source: &str,
    skin_index: u32,
    assets_dir: Option<&Path>,
) -> Result<Vec<ImportedAnimation>, String> {
    let doc = parse_glb(source, assets_dir)?;
    import_glb_animations_from_doc(&doc, source, skin_index)
}

// Import a single animation by its glTF index. Index out of range is a hard
// error; the user authored an animation entry the file does not contain.
pub(crate) fn import_glb_animation(
    source: &str,
    index: usize,
    skin_index: u32,
    assets_dir: Option<&Path>,
) -> Result<ImportedAnimation, String> {
    let mut anims = import_glb_animations(source, skin_index, assets_dir)?;
    if index >= anims.len() {
        return Err(format!(
            "'{}': animation_index {} out of range (file has {} animation{})",
            source,
            index,
            anims.len(),
            if anims.len() == 1 { "" } else { "s" }
        ));
    }
    Ok(anims.swap_remove(index))
}

// Names of every animation in a `.glb`, in file declaration order. Useful
// for the desugar pass when the user authored `animation_name` instead of
// `animation_index` and we need to look up the index.
pub(crate) fn glb_animation_names(
    source: &str,
    assets_dir: Option<&Path>,
) -> Result<Vec<String>, String> {
    Ok(crate::import::glb::parse_glb(source, assets_dir)?
        .doc
        .document
        .animations()
        .map(|a| a.name().unwrap_or("").to_string())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::glb::parse_glb;
    use crate::import::glb::test_fixtures::{skinned_glb, static_triangle_glb};

    fn write_glb(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
        concinnity_testing::utf8(&concinnity_testing::write_into(dir.path(), name, bytes))
    }

    #[test]
    fn import_skinned_glb_reads_mesh_and_skeleton() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_glb(&dir, "s.glb", &skinned_glb());
        let imported = import_skinned_glb(&src, 0, None).expect("skinned import");
        assert_eq!(imported.vertices.len(), 3);
        assert_eq!(imported.skeleton.len(), 2);
    }

    #[test]
    fn import_skinned_glb_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("missing.glb");
        let err = import_skinned_glb(src.to_str().unwrap(), 0, None)
            .err()
            .expect("expected error");
        assert!(err.contains("failed to read"), "got: {err}");
    }

    #[test]
    fn primitive_vertex_count_reads_the_indexed_primitive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_glb(&dir, "t.glb", &static_triangle_glb());
        let doc = parse_glb(&src, None).expect("parse");
        assert_eq!(primitive_vertex_count(&doc, 0), Some(3));
        assert_eq!(primitive_vertex_count(&doc, 1), None);
    }

    #[test]
    fn import_glb_animations_returns_every_clip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_glb(&dir, "s.glb", &skinned_glb());
        let anims = import_glb_animations(&src, 0, None).expect("animations");
        assert_eq!(anims.len(), 1);
        assert_eq!(anims[0].name, "wave");
    }

    #[test]
    fn import_glb_animation_returns_the_indexed_clip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_glb(&dir, "s.glb", &skinned_glb());
        let anim = import_glb_animation(&src, 0, 0, None).expect("clip");
        assert_eq!(anim.name, "wave");
    }

    #[test]
    fn import_glb_animation_rejects_an_out_of_range_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_glb(&dir, "s.glb", &skinned_glb());
        let err = import_glb_animation(&src, 3, 0, None).unwrap_err();
        assert!(err.contains("animation_index 3 out of range"), "got: {err}");
        assert!(err.contains("1 animation"), "got: {err}");
    }

    #[test]
    fn glb_animation_names_lists_clips_in_declaration_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_glb(&dir, "s.glb", &skinned_glb());
        assert_eq!(
            glb_animation_names(&src, None).expect("names"),
            vec!["wave"]
        );
    }

    #[test]
    fn glb_animation_names_is_empty_for_a_file_without_animations() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_glb(&dir, "t.glb", &static_triangle_glb());
        assert!(glb_animation_names(&src, None).expect("names").is_empty());
    }

    #[test]
    fn import_glb_animation_on_a_skinned_file_without_clips_reports_a_plural_count() {
        use crate::import::glb::test_fixtures::{make_glb, skinned_bin, skinned_json};
        let dir = tempfile::tempdir().expect("tempdir");
        let bytes = make_glb(&skinned_json(true, true, false), Some(&skinned_bin()));
        let src = write_glb(&dir, "s.glb", &bytes);
        let err = import_glb_animation(&src, 0, 0, None).unwrap_err();
        assert!(err.contains("animation_index 0 out of range"), "got: {err}");
        assert!(err.contains("0 animations"), "got: {err}");
    }

    #[test]
    fn glb_animation_names_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("missing.glb");
        let err = glb_animation_names(src.to_str().unwrap(), None).unwrap_err();
        assert!(err.contains("failed to read"), "got: {err}");
        assert!(err.contains("missing.glb"), "got: {err}");
    }

    #[test]
    fn glb_animation_names_rejects_invalid_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("junk.glb");
        std::fs::write(&path, b"garbage").expect("write junk");
        let err = glb_animation_names(path.to_str().unwrap(), None).unwrap_err();
        assert!(err.contains("not a valid glTF/GLB file"), "got: {err}");
    }
}
