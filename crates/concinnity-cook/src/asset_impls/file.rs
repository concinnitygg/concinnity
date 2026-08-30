// asset_impls/file.rs

use concinnity_core::components::File;
use concinnity_core::components::FileKind;

impl crate::asset::BuildAsset for File {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let kind_str = args.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        let kind = FileKind::from_ext(kind_str).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': unsupported File kind '{}'", ctx.name, kind_str),
            )
        })?;
        crate::compile::file::compile_file_payload(path, &kind)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, BuildCtx};

    fn ctx() -> BuildCtx<'static> {
        BuildCtx {
            name: "model",
            assets_dir: None,
            artifacts_dir: None,
            all_assets: &[],
        }
    }

    #[test]
    fn obj_source_compiles_to_a_mesh_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tri.obj");
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let args = serde_json::json!({"path": path.to_str().unwrap(), "kind": "obj"});
        let payload = File::compile_payload(&args, &ctx()).expect("obj compiles");
        assert!(!payload.is_empty());
    }

    #[test]
    fn an_unrecognised_kind_names_the_asset() {
        let args = serde_json::json!({"path": "model.xyz", "kind": "xyz"});
        let err = File::compile_payload(&args, &ctx()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("Asset 'model': unsupported File kind 'xyz'"),
            "got: {err}"
        );
    }

    #[test]
    fn a_compile_failure_is_reported_as_invalid_data() {
        let args = serde_json::json!({"path": "/no/such/model.obj", "kind": "obj"});
        let err = File::compile_payload(&args, &ctx()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("/no/such/model.obj"), "got: {err}");
    }
}
