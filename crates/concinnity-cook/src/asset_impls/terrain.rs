use concinnity_core::components::Terrain;

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// The Texture asset named `name`, among the world's assets.
fn texture<'a>(assets: &'a [WorldJsonlAsset], name: &str) -> Option<&'a serde_json::Value> {
    assets
        .iter()
        .find(|a| a.asset_type == RegisteredType::Texture && a.id == name)
        .map(|a| &a.args)
}

impl crate::asset::BuildAsset for Terrain {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::compile::terrain::compile_terrain_payload(
            ctx.name,
            args,
            |name| texture(ctx.all_assets, name),
            ctx.assets_dir,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    // The textures a terrain decodes are other assets: their args, and the
    // files those name, are inputs to the payload too.
    fn dependency_args(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> Vec<serde_json::Value> {
        crate::compile::terrain::texture_names(args)
            .into_iter()
            .filter_map(|name| texture(ctx.all_assets, name).cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, BuildCtx};

    fn asset(id: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type,
            args,
        }
    }

    fn ctx(assets: &[WorldJsonlAsset]) -> BuildCtx<'_> {
        BuildCtx {
            name: "hills",
            platform: concinnity_core::platform::Platform::Metal,
            assets_dir: None,
            all_assets: assets,
        }
    }

    fn args() -> serde_json::Value {
        serde_json::json!({
            "resolution": 4,
            "layers": [{"grass": "meadow", "density_mask": "patches"}],
        })
    }

    #[test]
    fn a_mask_resolves_against_a_sibling_texture() {
        let assets = [
            asset(
                "patches",
                RegisteredType::Texture,
                serde_json::json!({"generator": "checker", "resolution": 4}),
            ),
            asset("patches", RegisteredType::Material, serde_json::json!({})),
        ];
        let payload = Terrain::compile_payload(&args(), &ctx(&assets)).expect("the terrain cooks");
        assert!(!payload.is_empty());
        assert_eq!(
            Terrain::dependency_args(&args(), &ctx(&assets)),
            [assets[0].args.clone()]
        );
    }

    #[test]
    fn a_mask_naming_no_texture_is_invalid_data() {
        let assets = [asset(
            "patches",
            RegisteredType::Material,
            serde_json::json!({}),
        )];
        let err = Terrain::compile_payload(&args(), &ctx(&assets)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("patches"), "{err}");
        assert!(Terrain::dependency_args(&args(), &ctx(&assets)).is_empty());
    }
}
