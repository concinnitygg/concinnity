pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    crate::cubemap::validate_cubemap_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_hdr_source_passes() {
        check("sky", &serde_json::json!({"source": "studio.hdr"})).expect("should validate");
    }

    #[test]
    fn a_non_hdr_source_is_reported_against_the_asset_name() {
        let err = check("sky", &serde_json::json!({"source": "studio.png"})).unwrap_err();
        assert_eq!(
            err,
            "Asset 'sky': CubemapTexture source 'studio.png' must be a Radiance .hdr file"
        );
    }

    #[test]
    fn a_bad_face_size_is_reported_against_the_asset_name() {
        let args = serde_json::json!({"source": "studio.hdr", "face_size": 300});
        let err = check("sky", &args).unwrap_err();
        assert_eq!(
            err,
            "Asset 'sky': CubemapTexture face_size 300 must be a power of two"
        );
    }
}
