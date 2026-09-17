// Build-time expansion: LightRig -> DirectionalLight / PointLight assets.

use std::path::Path;

use crate::authoring::registry::RegisteredType;
use crate::authoring::registry::build_only::LightRig;
use crate::build_only::expand::{asset_name, registered_type, schema_args};
use crate::build_only::preset::load_preset_obj;

pub(crate) fn expand_light_rigs(
    asset_values: &mut Vec<serde_json::Value>,
    assets_dir: Option<&Path>,
) -> Result<(), String> {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if registered_type(&value) != Some(RegisteredType::LightRig) {
            result.push(value);
            continue;
        }
        let rig_name = asset_name(&value);
        let rig: LightRig = schema_args(RegisteredType::LightRig, &rig_name, value.get("args"))?;
        if !rig.preset.is_empty() {
            result.extend(expand_light_rig_preset(&rig_name, &rig.preset, assets_dir));
        }
        // The lights a rig lists are declared on their own lines and pass
        // through untouched; the rig entry itself is consumed.
    }
    *asset_values = result;
    Ok(())
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
        expand_light_rigs(&mut assets, None).unwrap();
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
        expand_light_rigs(&mut assets, None).unwrap();
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
        expand_light_rigs(&mut assets, None).unwrap();
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
        expand_light_rigs(&mut assets, None).unwrap();
        assert_eq!(assets.len(), 3);
    }

    #[test]
    fn non_rig_assets_pass_through() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        expand_light_rigs(&mut assets, None).unwrap();
        assert_eq!(assets[0]["type"], "Logger");
    }

    fn expand_preset(preset: &str) -> Vec<serde_json::Value> {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": preset}
        })];
        expand_light_rigs(&mut assets, None).unwrap();
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
        expand_light_rigs(&mut assets, None).unwrap();
        assert!(assets.is_empty());
    }

    // A preset takes over the rig: a listed light name is not also expanded.
    #[test]
    fn a_preset_rig_ignores_its_light_list() {
        let mut assets = vec![serde_json::json!({
            "name": "rig", "type": "LightRig",
            "args": {"preset": "rig_night_moon", "lights": ["torch"]}
        })];
        expand_light_rigs(&mut assets, None).unwrap();
        let names: Vec<String> = assets.iter().map(asset_name).collect();
        assert_eq!(names, ["rig_moon"]);
    }

    #[test]
    fn malformed_fields_name_the_rig_and_the_field() {
        for (args, field) in [
            (serde_json::json!({"preset": 5}), "`preset`"),
            (serde_json::json!({"lights": "sun"}), "`lights`"),
            (serde_json::json!({"lights": ["sun", 2]}), "`lights[1]`"),
        ] {
            let mut assets =
                vec![serde_json::json!({"name": "rig", "type": "LightRig", "args": args})];
            let err = expand_light_rigs(&mut assets, None).unwrap_err();
            assert!(err.starts_with("LightRig 'rig': invalid args: "), "{err}");
            assert!(err.contains(field), "{err}");
        }
    }
}
