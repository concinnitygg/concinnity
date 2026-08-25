// asset_impls/voxel_chunk.rs

use concinnity_core::components::VoxelChunk;

impl crate::asset::BuildAsset for VoxelChunk {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let palette_lookup = |bt_name: &str| {
            ctx.all_assets
                .iter()
                .find(|a| {
                    let tn = a.asset_type.to_lowercase().replace('_', "");
                    (tn == "blocktype" || tn == "block") && a.name == bt_name
                })
                .map(|a| a.args.clone())
        };
        crate::geometry::compile_voxel_chunk_payload(args, palette_lookup)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, BuildCtx};
    use crate::world::WorldJsonlAsset;

    fn block_type(name: &str, asset_type: &str) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args: serde_json::json!({"solid": true}),
        }
    }

    fn args() -> serde_json::Value {
        serde_json::json!({"dim": [1, 1, 1], "palette": ["stone"], "blocks": [0]})
    }

    #[test]
    fn the_palette_resolves_against_sibling_block_type_assets() {
        // The declared type is matched case- and underscore-insensitively.
        for type_name in ["BlockType", "block_type", "block"] {
            let assets = [block_type("stone", type_name)];
            let ctx = BuildCtx {
                name: "chunk",
                artifacts_dir: None,
                all_assets: &assets,
            };
            let payload = VoxelChunk::compile_payload(&args(), &ctx).expect("chunk compiles");
            let expected = crate::geometry::compile_voxel_chunk_payload(&args(), |_| {
                Some(serde_json::json!({"solid": true}))
            })
            .unwrap();
            assert_eq!(payload, expected, "type '{type_name}' should resolve");
        }
    }

    #[test]
    fn a_block_type_with_another_name_does_not_satisfy_the_palette() {
        let assets = [block_type("dirt", "BlockType")];
        let ctx = BuildCtx {
            name: "chunk",
            artifacts_dir: None,
            all_assets: &assets,
        };
        let err = VoxelChunk::compile_payload(&args(), &ctx).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("palette entry 'stone' has no matching BlockType"),
            "got: {err}"
        );
    }

    #[test]
    fn a_compile_error_is_reported_as_invalid_data() {
        let ctx = BuildCtx {
            name: "chunk",
            artifacts_dir: None,
            all_assets: &[],
        };
        let err = VoxelChunk::compile_payload(&serde_json::json!({}), &ctx).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("dim"), "got: {err}");
    }
}
