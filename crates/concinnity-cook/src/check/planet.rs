// Structural validation for Planet args: what a planet's layers can use.

/// Check a `Planet`'s authored args.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let layers = args.get("layers").and_then(|v| v.as_array());
    for (i, layer) in layers.into_iter().flatten().enumerate() {
        if layer.get("density_mask").is_some_and(|m| !m.is_null()) {
            return Err(format!(
                "Asset '{name}': Planet layers[{i}] sets a `density_mask`, which a Planet does not \
                 support; a planet's grass density comes from its procedural meadow, so drop the \
                 mask"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_masked_layer_is_refused() {
        let args = serde_json::json!({
            "layers": [{"grass": "meadow"}, {"grass": "meadow", "density_mask": "mask"}]
        });
        let err = check("world", &args).unwrap_err();
        assert!(
            err.contains("layers[1]") && err.contains("density_mask"),
            "{err}"
        );
    }

    #[test]
    fn unmasked_layers_pass() {
        assert!(check("world", &serde_json::json!({})).is_ok());
        let args = serde_json::json!({"layers": [{"grass": "meadow", "density_mask": null}]});
        assert!(check("world", &args).is_ok());
    }
}
