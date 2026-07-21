pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    crate::environment_map::validate_environment_map_args(args)
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
    fn a_non_hdr_source_is_reported_against_the_asset_name() {
        let err = check("ibl", &serde_json::json!({"source": "studio.png"})).unwrap_err();
        assert_eq!(
            err,
            "Asset 'ibl': EnvironmentMap source 'studio.png' must be a Radiance .hdr file"
        );
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
