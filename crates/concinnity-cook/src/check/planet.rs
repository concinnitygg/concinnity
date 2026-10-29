// Structural validation for Planet args: what a planet's layers can use, and
// covers past the most a planet lays.

/// Check a `Planet`'s authored args, warning about what it drops.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    for warning in warnings(name, args) {
        tracing::warn!("{warning}");
    }
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

// What a planet's args set that the planet drops.
fn warnings(name: &str, args: &serde_json::Value) -> Vec<String> {
    let max = concinnity_engine::planet::MAX_GROUND_COVERS;
    let covers = args
        .get("covers")
        .and_then(|v| v.as_array())
        .map_or(0, Vec::len);
    if covers > max {
        return vec![format!(
            "Asset '{name}': Planet sets {covers} covers; a planet lays at most {max}, so the \
             last {} are dropped",
            covers - max
        )];
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_past_the_cap_are_warned_about() {
        let cover = serde_json::json!({"material": "sand"});
        let at_cap = serde_json::json!({"covers": vec![cover.clone(); 4]});
        assert!(warnings("world", &at_cap).is_empty());
        let over = serde_json::json!({"covers": vec![cover; 6]});
        let w = warnings("world", &over);
        assert_eq!(w.len(), 1);
        assert!(
            w[0].contains("6 covers") && w[0].contains("last 2"),
            "{w:?}"
        );
        assert!(check("world", &over).is_ok());
    }

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
