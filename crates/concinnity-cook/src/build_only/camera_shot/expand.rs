// Build-time expansion: CameraShot -> Camera3D.

use std::path::Path;

use crate::authoring::registry::RegisteredType;
use crate::authoring::registry::build_only::CameraShot;
use crate::build_only::expand::{asset_name, registered_type, schema_args};
use crate::build_only::preset::load_preset_obj;

pub(crate) fn expand_camera_shots(
    asset_values: &mut Vec<serde_json::Value>,
    assets_dir: Option<&Path>,
) -> Result<(), String> {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if registered_type(&value) != Some(RegisteredType::CameraShot) {
            result.push(value);
            continue;
        }
        let shot_name = asset_name(&value);
        let shot = resolve_shot(&shot_name, value.get("args"), assets_dir)?;
        result.push(serde_json::json!({
            "name": shot_name,
            "type": "Camera3D",
            "args": {
                "fov_y_degrees": shot.fov_y_degrees,
                "near": shot.near,
                "far": shot.far,
                "position": shot.position,
                "yaw": shot.yaw,
                "pitch": shot.pitch
            }
        }));
    }
    *asset_values = result;
    Ok(())
}

// The shot's inline args laid over its preset's, each validated on its own so
// a failure names the side it came from.
fn resolve_shot(
    name: &str,
    args: Option<&serde_json::Value>,
    assets_dir: Option<&Path>,
) -> Result<CameraShot, String> {
    let ty = RegisteredType::CameraShot;
    let inline: CameraShot = schema_args(ty, name, args)?;
    if inline.preset.is_empty() {
        return Ok(inline);
    }
    let hardcoded = camera_shot_preset(&inline.preset);
    let mut merged = if hardcoded.is_null() {
        load_preset_obj(&inline.preset, "shots", assets_dir)
            .get("args")
            .cloned()
            .unwrap_or(serde_json::json!({}))
    } else {
        hardcoded
    };
    schema_args::<CameraShot>(ty, name, Some(&merged))
        .map_err(|e| format!("{e} (in preset '{}')", inline.preset))?;
    if let (Some(base), Some(over)) = (merged.as_object_mut(), args.and_then(|a| a.as_object())) {
        base.extend(over.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    schema_args(ty, name, Some(&merged))
}

fn camera_shot_preset(preset: &str) -> serde_json::Value {
    match preset {
        "shot_eye_level" => {
            serde_json::json!({"fov_y_degrees":75.0,"position":[0.0,1.75,0.0],"yaw":std::f64::consts::PI,"near":0.05,"far":200.0})
        }
        "shot_overhead" => {
            serde_json::json!({"fov_y_degrees":60.0,"position":[0.0,8.0,0.0],"pitch":-1.3963,"near":0.05,"far":200.0})
        }
        "shot_dramatic_low" => {
            serde_json::json!({"fov_y_degrees":85.0,"position":[0.0,0.4,0.0],"pitch":0.2618,"near":0.05,"far":200.0})
        }
        "shot_outdoor_wide" => {
            serde_json::json!({"fov_y_degrees":80.0,"position":[0.0,1.75,0.0],"near":0.05,"far":500.0})
        }
        _ => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_to_camera3d() {
        let mut assets = vec![serde_json::json!({
            "name": "wide",
            "type": "CameraShot",
            "args": {"fov_y_degrees": 80.0, "position": [0.0, 1.75, 8.0], "yaw": std::f64::consts::PI}
        })];
        expand_camera_shots(&mut assets, None).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["name"], "wide");
        assert_eq!(assets[0]["type"], "Camera3D");
        assert_eq!(assets[0]["args"]["fov_y_degrees"], 80.0);
    }

    #[test]
    fn preset_eye_level_expands() {
        let mut assets = vec![serde_json::json!({
            "name": "cam",
            "type": "CameraShot",
            "args": {"preset": "shot_eye_level"}
        })];
        expand_camera_shots(&mut assets, None).unwrap();
        assert_eq!(assets[0]["type"], "Camera3D");
        let fov = assets[0]["args"]["fov_y_degrees"].as_f64().unwrap();
        assert!((fov - 75.0).abs() < 0.01);
    }

    #[test]
    fn inline_args_override_preset() {
        let mut assets = vec![serde_json::json!({
            "name": "cam",
            "type": "CameraShot",
            "args": {"preset": "shot_eye_level", "fov_y_degrees": 90.0}
        })];
        expand_camera_shots(&mut assets, None).unwrap();
        let fov = assets[0]["args"]["fov_y_degrees"].as_f64().unwrap();
        assert!((fov - 90.0).abs() < 0.01);
    }

    #[test]
    fn non_camera_shot_assets_pass_through() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        expand_camera_shots(&mut assets, None).unwrap();
        assert_eq!(assets[0]["type"], "Logger");
    }

    // Expand a single CameraShot built from `args` and return the Camera3D args.
    fn expand_args(args: serde_json::Value) -> serde_json::Value {
        let mut assets = vec![serde_json::json!({
            "name": "cam",
            "type": "CameraShot",
            "args": args
        })];
        expand_camera_shots(&mut assets, None).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["type"], "Camera3D");
        assets[0]["args"].clone()
    }

    #[test]
    fn preset_overhead_looks_down_from_above() {
        let args = expand_args(serde_json::json!({"preset": "shot_overhead"}));
        assert_eq!(args["fov_y_degrees"], serde_json::json!(60.0f32));
        assert_eq!(
            args["position"],
            serde_json::json!([0.0f32, 8.0f32, 0.0f32])
        );
        assert_eq!(args["pitch"], serde_json::json!(-1.3963f32));
        // The preset sets no yaw, so the type default applies.
        assert_eq!(args["yaw"], serde_json::json!(0.0f32));
    }

    #[test]
    fn preset_dramatic_low_tilts_up_from_near_the_floor() {
        let args = expand_args(serde_json::json!({"preset": "shot_dramatic_low"}));
        assert_eq!(args["fov_y_degrees"], serde_json::json!(85.0f32));
        assert_eq!(
            args["position"],
            serde_json::json!([0.0f32, 0.4f32, 0.0f32])
        );
        assert_eq!(args["pitch"], serde_json::json!(0.2618f32));
    }

    #[test]
    fn preset_outdoor_wide_pushes_the_far_plane_out() {
        let args = expand_args(serde_json::json!({"preset": "shot_outdoor_wide"}));
        assert_eq!(args["fov_y_degrees"], serde_json::json!(80.0f32));
        assert_eq!(args["far"], serde_json::json!(500.0f32));
        assert_eq!(args["near"], serde_json::json!(0.05f32));
    }

    // An unknown preset name is not a build error: it falls through to the
    // on-disk preset lookup, which misses, leaving the type defaults.
    #[test]
    fn unknown_preset_falls_back_to_defaults() {
        let args = expand_args(serde_json::json!({"preset": "cn_test_no_such_shot"}));
        assert_eq!(args["fov_y_degrees"], serde_json::json!(75.0f32));
        assert_eq!(args["near"], serde_json::json!(0.05f32));
        assert_eq!(args["far"], serde_json::json!(200.0f32));
        assert_eq!(
            args["position"],
            serde_json::json!([0.0f32, 0.0f32, 0.0f32])
        );
    }

    // A field of the wrong shape fails the build naming the shot and the field,
    // where it used to fall back to a default.
    #[test]
    fn malformed_fields_name_the_shot_and_the_field() {
        for (args, field) in [
            (serde_json::json!({"position": [1.0, 2.0]}), "`position`"),
            (
                serde_json::json!({"fov_y_degrees": "wide"}),
                "`fov_y_degrees`",
            ),
            (serde_json::json!({"preset": 3}), "`preset`"),
        ] {
            let mut assets =
                vec![serde_json::json!({"name": "cam", "type": "CameraShot", "args": args})];
            let err = expand_camera_shots(&mut assets, None).unwrap_err();
            assert!(err.starts_with("CameraShot 'cam': invalid args: "), "{err}");
            assert!(err.contains(field), "{err}");
        }
    }

    #[test]
    fn defaults_applied_when_no_args() {
        let mut assets = vec![serde_json::json!({
            "name": "cam",
            "type": "CameraShot",
            "args": {}
        })];
        expand_camera_shots(&mut assets, None).unwrap();
        assert_eq!(assets[0]["type"], "Camera3D");
        let fov = assets[0]["args"]["fov_y_degrees"].as_f64().unwrap();
        assert!((fov - 75.0).abs() < 0.01);
        let near = assets[0]["args"]["near"].as_f64().unwrap();
        assert!((near - 0.05).abs() < 0.001);
    }

    // Every inline field reaches the camera, and an inline position replaces
    // the preset's while the preset still fills what the shot leaves unset.
    #[test]
    fn inline_fields_reach_the_camera_over_the_preset() {
        let args = expand_args(serde_json::json!({
            "preset": "shot_overhead", "near": 0.5, "far": 50.0,
            "position": [1.0, 2.0, 3.0], "yaw": 1.5
        }));
        assert_eq!(args["fov_y_degrees"], serde_json::json!(60.0f32));
        assert_eq!(
            (args["near"].as_f64(), args["far"].as_f64()),
            (Some(0.5), Some(50.0))
        );
        assert_eq!(
            args["position"],
            serde_json::json!([1.0f32, 2.0f32, 3.0f32])
        );
        assert_eq!(args["yaw"], serde_json::json!(1.5f32));
        assert_eq!(args["pitch"].as_f64().map(|p| p as f32), Some(-1.3963));
    }
}
