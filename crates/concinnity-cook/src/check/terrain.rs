// Structural validation for Terrain args: what a terrain's layers ignore.

/// Warn about what a `Terrain`'s authored args set that it ignores.
pub(crate) fn warn(name: &str, args: &serde_json::Value) {
    for warning in warnings(name, args) {
        tracing::warn!("{warning}");
    }
}

// What a terrain's args set that the terrain ignores: a layer's `rule`, which
// only a planet reads.
fn warnings(name: &str, args: &serde_json::Value) -> Vec<String> {
    let layers = args.get("layers").and_then(|v| v.as_array());
    layers
        .into_iter()
        .flatten()
        .enumerate()
        .filter(|(_, layer)| {
            layer.get("rule").is_some_and(|rule| {
                !rule.is_null()
                    && serde_json::from_value::<concinnity_engine::components::GroundRule>(
                        rule.clone(),
                    )
                    .map_or(true, |r| {
                        r != concinnity_engine::components::GroundRule::default()
                    })
            })
        })
        .map(|(i, _)| {
            format!(
                "Asset '{name}': Terrain layers[{i}] sets a `rule`, which a Terrain does not \
                 read yet; the layer grows over the whole terrain"
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_rule_on_a_terrain_is_warned_about() {
        let args = serde_json::json!({
            "layers": [
                {"grass": "meadow"},
                {"grass": "meadow", "rule": {}},
                {"grass": "meadow", "rule": {"altitude": [2.0, null]}}
            ]
        });
        let w = warnings("field", &args);
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("layers[2]"), "{w:?}");
    }
}
