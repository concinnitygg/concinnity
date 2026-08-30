pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    crate::compile::environment_map::validate_environment_map_args(args)
        .map_err(|e| format!("Asset '{}': {}", name, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sky_generator_passes() {
        check("ibl", &serde_json::json!({"generator": "sky"})).expect("should validate");
    }

    #[test]
    fn an_unsupported_source_container_is_reported_against_the_asset_name() {
        let err = check("ibl", &serde_json::json!({"source": "studio.png"})).unwrap_err();
        assert_eq!(
            err,
            "Asset 'ibl': EnvironmentMap source 'studio.png' must be a Radiance .hdr \
             file or a panorama-sphere .glb / .gltf"
        );
    }

    #[test]
    fn a_panorama_glb_source_passes() {
        check("ibl", &serde_json::json!({"source": "galaxy.glb"})).expect("should validate");
    }

    #[test]
    fn an_unknown_generator_is_reported_against_the_asset_name() {
        let err = check("ibl", &serde_json::json!({"generator": "aurora"})).unwrap_err();
        assert_eq!(
            err,
            "Asset 'ibl': unknown EnvironmentMap generator 'aurora'"
        );
    }
}
