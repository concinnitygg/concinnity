//! Build-time mesh payload entry point. Every generator is pure geometry, so
//! this delegates straight to `crate::compile::geometry`.

/// Compile a Mesh / ProceduralMesh component's JSON args into a packed binary
/// payload.
pub fn compile_mesh_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    crate::compile::geometry::compile_mesh_payload(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_generator_compiles_through_the_geometry_module() {
        let args = serde_json::json!({ "generator": "sphere", "radius": 1.0 });
        assert_eq!(
            compile_mesh_payload(&args).unwrap(),
            crate::compile::geometry::compile_mesh_payload(&args).unwrap()
        );
    }

    #[test]
    fn a_removed_generator_is_unknown() {
        for name in ["terrain", "heightfield"] {
            let args = serde_json::json!({ "generator": name });
            let err = compile_mesh_payload(&args).unwrap_err();
            assert!(err.contains("unknown mesh generator"), "{name}: {err}");
        }
    }
}
