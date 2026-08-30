// Build-time expansion: LightRig -> DirectionalLight / PointLight assets.

use std::path::Path;

use super::expand::{asset_name, type_norm};
use super::preset::load_preset_obj;

pub(crate) fn expand_light_rigs(
    asset_values: &mut Vec<serde_json::Value>,
    assets_dir: Option<&Path>,
) {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if type_norm(&value) != "lightrig" {
            result.push(value);
            continue;
        }
        let rig_name = asset_name(&value);
        let args = value.get("args").cloned().unwrap_or(serde_json::json!({}));
        let preset = args.get("preset").and_then(|v| v.as_str()).unwrap_or("");
        if !preset.is_empty() {
            for light in expand_light_rig_preset(&rig_name, preset, assets_dir) {
                result.push(light);
            }
        }
        // lights: Vec<String>; referenced lights are already declared; the rig
        // entry is consumed and those lights pass through untouched.
    }
    *asset_values = result;
}

fn expand_light_rig_preset(
    rig_name: &str,
    preset: &str,
    assets_dir: Option<&Path>,
) -> Vec<serde_json::Value> {
    let defs = {
        let hardcoded = rig_preset_lights(preset);
        if hardcoded.is_empty() {
            load_preset_obj(preset, "light_rigs", assets_dir)
                .get("args")
                .and_then(|a| a.get("lights"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        } else {
            hardcoded
        }
    };

    defs.iter()
        .map(|light| {
            let kind = light.get("kind").and_then(|v| v.as_str()).unwrap_or("directional");
            let lname = light.get("name").and_then(|v| v.as_str()).unwrap_or("light");
            let expanded = format!("{}_{}", rig_name, lname);
            match kind {
                "point" => serde_json::json!({
                    "name": expanded,
                    "type": "PointLight",
                    "args": {
                        "position": light.get("position").cloned().unwrap_or(serde_json::json!([0.0, 2.5, 0.0])),
                        "color":    light.get("color").cloned().unwrap_or(serde_json::json!([1.0, 1.0, 1.0])),
                        "intensity": light.get("intensity").and_then(|v| v.as_f64()).unwrap_or(8.0),
                        "range":     light.get("range").and_then(|v| v.as_f64()).unwrap_or(6.0)
                    }
                }),
                _ => serde_json::json!({
                    "name": expanded,
                    "type": "DirectionalLight",
                    "args": {
                        "direction": light.get("direction").cloned().unwrap_or(serde_json::json!([-0.3, 0.85, 0.4])),
                        "color":     light.get("color").cloned().unwrap_or(serde_json::json!([1.0, 1.0, 1.0])),
                        "intensity": light.get("intensity").and_then(|v| v.as_f64()).unwrap_or(1.0)
                    }
                }),
            }
        })
        .collect()
}

fn rig_preset_lights(preset: &str) -> Vec<serde_json::Value> {
    match preset {
        "rig_outdoor_sun" => vec![
            serde_json::json!({"kind":"directional","name":"sun","direction":[-0.4,0.7,0.3],"color":[1.0,0.95,0.8],"intensity":1.2}),
        ],
        "rig_outdoor_sun_fill" => vec![
            serde_json::json!({"kind":"directional","name":"sun","direction":[-0.4,0.7,0.3],"color":[1.0,0.95,0.8],"intensity":1.2}),
            serde_json::json!({"kind":"directional","name":"fill","direction":[0.3,0.5,-0.5],"color":[0.6,0.8,1.0],"intensity":0.3}),
        ],
        "rig_studio_three_point" => vec![
            serde_json::json!({"kind":"directional","name":"key","direction":[-0.6,0.7,0.4],"color":[1.0,0.95,0.9],"intensity":1.2}),
            serde_json::json!({"kind":"directional","name":"fill","direction":[0.8,0.4,0.3],"color":[0.8,0.9,1.0],"intensity":0.4}),
            serde_json::json!({"kind":"directional","name":"rim","direction":[0.2,0.6,-0.8],"color":[0.9,0.9,1.0],"intensity":0.6}),
        ],
        "rig_interior_candles" => vec![
            serde_json::json!({"kind":"directional","name":"ambient","direction":[0.0,1.0,0.0],"color":[0.8,0.6,0.4],"intensity":0.2}),
            serde_json::json!({"kind":"point","name":"candle_a","position":[3.0,1.5,-3.0],"color":[1.0,0.7,0.3],"intensity":8.0,"range":5.0}),
            serde_json::json!({"kind":"point","name":"candle_b","position":[-3.0,1.5,-3.0],"color":[1.0,0.7,0.3],"intensity":8.0,"range":5.0}),
            serde_json::json!({"kind":"point","name":"candle_c","position":[0.0,1.5,4.0],"color":[1.0,0.7,0.3],"intensity":8.0,"range":5.0}),
        ],
        "rig_night_moon" => vec![
            serde_json::json!({"kind":"directional","name":"moon","direction":[-0.2,0.8,0.3],"color":[0.7,0.8,1.0],"intensity":0.4}),
        ],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_lights_consumed_leaves_lights_intact() {
        let mut assets = vec![
            serde_json::json!({"name":"sun","type":"DirectionalLight","args":{"direction":[-0.4,0.7,0.3]}}),
            serde_json::json!({"name":"torch","type":"PointLight","args":{"position":[3.0,2.0,-5.0]}}),
            serde_json::json!({"name":"rig","type":"LightRig","args":{"lights":["sun","torch"]}}),
        ];
        expand_light_rigs(&mut assets, None);
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "sun");
        assert_eq!(assets[1]["name"], "torch");
    }

    #[test]
    fn preset_sun_fill_expands_to_two_lights() {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": "rig_outdoor_sun_fill"}
        })];
        expand_light_rigs(&mut assets, None);
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "rig_sun");
        assert_eq!(assets[1]["name"], "rig_fill");
        assert_eq!(assets[0]["type"], "DirectionalLight");
    }

    #[test]
    fn preset_interior_candles_includes_point_lights() {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": "rig_interior_candles"}
        })];
        expand_light_rigs(&mut assets, None);
        assert_eq!(assets.len(), 4);
        let point_count = assets.iter().filter(|v| v["type"] == "PointLight").count();
        assert_eq!(point_count, 3);
    }

    #[test]
    fn preset_studio_three_point_expands_to_three() {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": "rig_studio_three_point"}
        })];
        expand_light_rigs(&mut assets, None);
        assert_eq!(assets.len(), 3);
    }

    #[test]
    fn non_rig_assets_pass_through() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        expand_light_rigs(&mut assets, None);
        assert_eq!(assets[0]["type"], "Logger");
    }

    fn expand_preset(preset: &str) -> Vec<serde_json::Value> {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": preset}
        })];
        expand_light_rigs(&mut assets, None);
        assets
    }

    #[test]
    fn preset_outdoor_sun_expands_to_one_warm_directional() {
        let lights = expand_preset("rig_outdoor_sun");
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0]["name"], "rig_sun");
        assert_eq!(lights[0]["type"], "DirectionalLight");
        assert_eq!(
            lights[0]["args"]["direction"],
            serde_json::json!([-0.4, 0.7, 0.3])
        );
        assert_eq!(
            lights[0]["args"]["color"],
            serde_json::json!([1.0, 0.95, 0.8])
        );
        assert_eq!(lights[0]["args"]["intensity"], 1.2);
    }

    #[test]
    fn preset_night_moon_expands_to_one_cool_directional() {
        let lights = expand_preset("rig_night_moon");
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0]["name"], "rig_moon");
        assert_eq!(
            lights[0]["args"]["color"],
            serde_json::json!([0.7, 0.8, 1.0])
        );
        assert_eq!(lights[0]["args"]["intensity"], 0.4);
    }

    // The candle points carry their position, tint, intensity, and range
    // through to the PointLight args.
    #[test]
    fn preset_candle_point_lights_carry_their_placement() {
        let lights = expand_preset("rig_interior_candles");
        let candle = lights
            .iter()
            .find(|v| v["name"] == "rig_candle_a")
            .expect("candle_a light");
        assert_eq!(candle["type"], "PointLight");
        assert_eq!(
            candle["args"]["position"],
            serde_json::json!([3.0, 1.5, -3.0])
        );
        assert_eq!(candle["args"]["color"], serde_json::json!([1.0, 0.7, 0.3]));
        assert_eq!(candle["args"]["intensity"], 8.0);
        assert_eq!(candle["args"]["range"], 5.0);
    }

    // An unknown preset is not a build error: the on-disk preset lookup misses
    // and the rig expands to nothing.
    #[test]
    fn unknown_preset_expands_to_no_lights() {
        assert!(expand_preset("cn_test_no_such_rig").is_empty());
    }

    // A rig with no preset and no lights list is consumed and adds nothing.
    #[test]
    fn rig_without_a_preset_expands_to_nothing() {
        let mut assets = vec![serde_json::json!({"name":"rig","type":"LightRig"})];
        expand_light_rigs(&mut assets, None);
        assert!(assets.is_empty());
    }
}
