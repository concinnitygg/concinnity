use concinnity_engine::components::PlanetGround;
use concinnity_engine::components::cook::PlanetGround as PlanetGroundArgs;
use concinnity_engine::planet::encode_page;

use crate::asset::BuildCtx;

impl crate::asset::BuildAsset for PlanetGround {
    // The square's page, compressed: each square compiles on its own, so
    // sculpting one rebuilds only its page.
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        let invalid = |e: String| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("PlanetGround '{}': {e}", ctx.name),
            )
        };
        let args: PlanetGroundArgs =
            serde_json::from_value(args.clone()).map_err(|e| invalid(e.to_string()))?;
        let (_, page) = args.page().map_err(|e| invalid(e.to_string()))?;
        Ok(encode_page(&page))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::BuildAsset;
    use concinnity_engine::planet::{PAGE_SAMPLES, decode_page};

    fn ctx() -> BuildCtx<'static> {
        BuildCtx {
            name: "ground",
            targets: concinnity_engine::platform::PlatformSet::single(
                concinnity_engine::platform::Platform::Metal,
            ),
            assets_dir: None,
            all_assets: &[],
        }
    }

    #[test]
    fn a_square_compiles_to_its_page() {
        let args = serde_json::json!({
            "tile": [2, 8, 128, 127],
            "height": vec![3.0; PAGE_SAMPLES],
        });
        let payload = PlanetGround::compile_payload(&args, &ctx()).unwrap();
        let page = decode_page(&payload).unwrap();
        assert_eq!(page.e_at(10, 10), 3.0);
    }

    #[test]
    fn a_bad_square_fails_naming_itself_and_why() {
        let args = serde_json::json!({ "tile": [2, 8, 128, 127], "height": [1.0, 2.0] });
        let err = PlanetGround::compile_payload(&args, &ctx()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let message = err.to_string();
        assert!(
            message.contains("PlanetGround 'ground'") && message.contains("2 values"),
            "{message}"
        );
    }
}
