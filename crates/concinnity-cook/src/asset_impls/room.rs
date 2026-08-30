// asset_impls/room.rs

use concinnity_core::components::Room;

impl crate::asset::BuildAsset for Room {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::compile::geometry::compile_room_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, BuildCtx};

    fn ctx() -> BuildCtx<'static> {
        BuildCtx {
            name: "lobby",
            assets_dir: None,
            artifacts_dir: None,
            all_assets: &[],
        }
    }

    #[test]
    fn room_args_compile_to_the_same_payload_as_the_geometry_compiler() {
        let args = serde_json::json!({"size": [16.0, 20.0, 3.5]});
        let payload = Room::compile_payload(&args, &ctx()).expect("room compiles");
        assert_eq!(
            payload,
            crate::compile::geometry::compile_room_payload(&args).unwrap()
        );
    }

    #[test]
    fn a_compile_error_is_reported_as_invalid_data() {
        let args = serde_json::json!({"lod_levels": 3, "lod_distances": [10.0]});
        let err = Room::compile_payload(&args, &ctx()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("lod_distances"), "got: {err}");
    }
}
