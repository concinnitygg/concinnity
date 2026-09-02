use crate::components::FileKind;

// Compile a File asset's source into a binary payload.
//
// Only mesh-kind files (e.g. OBJ) produce a blob. Call `kind.is_mesh()` before
// invoking this to skip non-mesh kinds at the call site.
pub(crate) fn compile_file_payload(path: &str, kind: &FileKind) -> Result<Vec<u8>, String> {
    match kind {
        FileKind::Obj => {
            let source = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read '{}': {}", path, e))?;
            let (vertices, indices) = crate::import::wavefront::parse_obj(&source)?;
            crate::compile::geometry::compile_mesh_from_vertex_data(&vertices, &indices)
        }
        other => Err(format!(
            "File kind '{:?}' does not produce a compiled payload",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_mesh_kind_returns_error() {
        assert!(compile_file_payload("irrelevant.png", &FileKind::Png).is_err());
    }

    #[test]
    fn obj_source_compiles_to_a_non_empty_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tri.obj");
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let payload =
            compile_file_payload(path.to_str().unwrap(), &FileKind::Obj).expect("obj compiles");
        assert!(!payload.is_empty());
    }

    #[test]
    fn obj_read_error_surfaces_the_path() {
        let err = compile_file_payload("/no/such/model.obj", &FileKind::Obj).unwrap_err();
        assert!(err.contains("/no/such/model.obj"), "got: {err}");
    }
}
