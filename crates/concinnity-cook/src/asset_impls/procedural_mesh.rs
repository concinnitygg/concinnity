// asset_impls/procedural_mesh.rs

use concinnity_core::components::ProceduralMesh;

impl crate::asset::BuildAsset for ProceduralMesh {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::compile::mesh_compile::compile_mesh_payload(args, ctx.assets_dir)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, BuildCtx};

    fn ctx() -> BuildCtx<'static> {
        BuildCtx {
            name: "prop",
            assets_dir: None,
            artifacts_dir: None,
            all_assets: &[],
        }
    }

    #[test]
    fn a_generator_compiles_to_the_same_payload_as_the_mesh_compiler() {
        let args = serde_json::json!({"generator": "box", "half_extents": [1.0, 2.0, 3.0]});
        let payload = ProceduralMesh::compile_payload(&args, &ctx()).expect("box compiles");
        assert_eq!(
            payload,
            crate::compile::mesh_compile::compile_mesh_payload(&args, None).unwrap()
        );
    }

    #[test]
    fn a_generator_error_is_reported_as_invalid_data() {
        let args = serde_json::json!({"generator": "teapot"});
        let err = ProceduralMesh::compile_payload(&args, &ctx()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("teapot"), "got: {err}");
    }
}
