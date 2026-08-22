//! Physics authoring checks: collider shape strings, and the collision-layer
//! names a world's PhysicsConfig declares against every reference to them.

use crate::world::WorldJsonlAsset;

const COLLIDER_SHAPES: [&str; 5] = ["aabb", "cuboid", "ball", "sphere", "capsule"];
const BUILTIN_LAYERS: [&str; 4] = ["world", "prop", "character", "trigger"];
// Rapier interaction groups are 32 bits: the built-ins plus at most 28 more.
const MAX_USER_LAYERS: usize = 32 - BUILTIN_LAYERS.len();

// A `collider` object's `shape`, when present, must be a recognized shape
// name. Shared by the Prop and TriggerVolume checks.
pub(crate) fn check_collider_shape(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let Some(shape) = args
        .get("collider")
        .and_then(|c| c.get("shape"))
        .and_then(|s| s.as_str())
    else {
        return Ok(());
    };
    if COLLIDER_SHAPES.contains(&shape) {
        return Ok(());
    }
    Err(format!(
        "Asset '{}': unknown collider shape '{}'; expected one of {}",
        name,
        shape,
        COLLIDER_SHAPES.join(", ")
    ))
}

/// Check a physics asset's authored args.
pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_collider_shape(name, args)
}

// World-shape rule: every layer name a world uses must resolve. Declared
// names must be unique, not shadow a built-in, and fit the 32-bit group
// space; `no_collide` pairs and collider `layer` fields must name a declared
// or built-in layer.
pub(crate) fn check_layers(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let norm = |t: &str| t.to_lowercase().replace('_', "");
    let config = assets
        .iter()
        .find(|a| norm(&a.asset_type) == "physicsconfig");

    let mut known: Vec<String> = BUILTIN_LAYERS.iter().map(|s| s.to_string()).collect();
    if let Some(config) = config {
        let declared: Vec<&str> = config
            .args
            .get("layers")
            .and_then(|v| v.as_array())
            .map(|list| list.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        if declared.len() > MAX_USER_LAYERS {
            errors.push(format!(
                "PhysicsConfig '{}': {} extra layers declared; at most {} fit the 32-bit group space",
                config.name,
                declared.len(),
                MAX_USER_LAYERS
            ));
        }
        for layer in declared {
            if layer.is_empty() {
                errors.push(format!("PhysicsConfig '{}': empty layer name", config.name));
            } else if known.iter().any(|k| k == layer) {
                errors.push(format!(
                    "PhysicsConfig '{}': layer '{}' is already defined",
                    config.name, layer
                ));
            } else {
                known.push(layer.to_string());
            }
        }
        if let Some(pairs) = config.args.get("no_collide").and_then(|v| v.as_array()) {
            for pair in pairs {
                let names = pair
                    .as_array()
                    .map(|p| p.iter().filter_map(|v| v.as_str()).collect::<Vec<&str>>());
                let Some(names) = names.filter(|n| n.len() == 2) else {
                    errors.push(format!(
                        "PhysicsConfig '{}': no_collide entries are [\"layer_a\", \"layer_b\"] pairs",
                        config.name
                    ));
                    continue;
                };
                for layer in names {
                    if !known.iter().any(|k| k == layer) {
                        errors.push(format!(
                            "PhysicsConfig '{}': no_collide names unknown layer '{}'",
                            config.name, layer
                        ));
                    }
                }
            }
        }
        if let Some(min) = config
            .args
            .get("contact_min_impulse")
            .and_then(|v| v.as_f64())
            && min < 0.0
        {
            errors.push(format!(
                "PhysicsConfig '{}': contact_min_impulse must not be negative",
                config.name
            ));
        }
    }

    for asset in assets.iter().filter(|a| norm(&a.asset_type) == "prop") {
        let Some(layer) = asset
            .args
            .get("collider")
            .and_then(|c| c.get("layer"))
            .and_then(|l| l.as_str())
            .filter(|l| !l.is_empty())
        else {
            continue;
        };
        if !known.iter().any(|k| k == layer) {
            errors.push(format!(
                "Prop '{}': collider layer '{}' is not a built-in layer or declared in PhysicsConfig `layers`",
                asset.name, layer
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
        }
    }

    fn layer_errors(assets: &[WorldJsonlAsset]) -> Vec<String> {
        let mut errors = Vec::new();
        check_layers(assets, &mut errors);
        errors
    }

    #[test]
    fn known_shapes_pass_and_unknown_shapes_fail() {
        let ok = serde_json::json!({"collider": {"shape": "capsule"}});
        assert!(check_collider_shape("p", &ok).is_ok());
        // No collider or no shape field: nothing to judge.
        assert!(check_collider_shape("p", &serde_json::json!({})).is_ok());
        let err = check_collider_shape("p", &serde_json::json!({"collider": {"shape": "mesh"}}))
            .unwrap_err();
        assert!(err.contains("unknown collider shape 'mesh'"), "got: {err}");
    }

    #[test]
    fn declared_layers_resolve_everywhere() {
        let assets = [
            asset(
                "physics",
                "PhysicsConfig",
                serde_json::json!({"layers": ["debris"], "no_collide": [["debris", "character"]]}),
            ),
            asset(
                "rock",
                "Prop",
                serde_json::json!({"mesh": "m", "collider": {"shape": "ball", "layer": "debris"}}),
            ),
        ];
        assert!(layer_errors(&assets).is_empty());
    }

    #[test]
    fn unknown_layer_references_fail() {
        let assets = [
            asset(
                "physics",
                "PhysicsConfig",
                serde_json::json!({"no_collide": [["ghost", "world"]]}),
            ),
            asset(
                "rock",
                "Prop",
                serde_json::json!({"mesh": "m", "collider": {"shape": "ball", "layer": "ghost"}}),
            ),
        ];
        let errors = layer_errors(&assets);
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors[0].contains("no_collide names unknown layer 'ghost'"));
        assert!(errors[1].contains("collider layer 'ghost'"));
    }

    #[test]
    fn prop_layers_work_without_a_physics_config() {
        // Built-ins resolve with no config declared; unknown names still fail.
        let ok = [asset(
            "rock",
            "Prop",
            serde_json::json!({"mesh": "m", "collider": {"shape": "ball", "layer": "prop"}}),
        )];
        assert!(layer_errors(&ok).is_empty());
        let bad = [asset(
            "rock",
            "Prop",
            serde_json::json!({"mesh": "m", "collider": {"shape": "ball", "layer": "ghost"}}),
        )];
        assert_eq!(layer_errors(&bad).len(), 1);
    }

    #[test]
    fn duplicate_and_overflowing_layer_declarations_fail() {
        let dup = [asset(
            "physics",
            "PhysicsConfig",
            serde_json::json!({"layers": ["debris", "debris", "world"]}),
        )];
        let errors = layer_errors(&dup);
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors.iter().all(|e| e.contains("already defined")));

        let many: Vec<String> = (0..29).map(|i| format!("layer{i}")).collect();
        let over = [asset(
            "physics",
            "PhysicsConfig",
            serde_json::json!({"layers": many}),
        )];
        let errors = layer_errors(&over);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("32-bit group space"));
    }

    #[test]
    fn malformed_no_collide_and_negative_impulse_fail() {
        let assets = [asset(
            "physics",
            "PhysicsConfig",
            serde_json::json!({"no_collide": [["world"]], "contact_min_impulse": -1.0}),
        )];
        let errors = layer_errors(&assets);
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors[0].contains("pairs"));
        assert!(errors[1].contains("must not be negative"));
    }
}
