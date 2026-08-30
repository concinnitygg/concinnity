// Build-time expansion: Prefab templates + Prop instances -> concrete assets.
// Every generated asset is recorded against its instantiating Prop, and an
// authored line with a generated asset's name is the user's patch of it: the
// authored fields win and the rest keep the generated values.

use std::path::Path;

use super::expand::{ExpandReport, asset_name, type_norm};
use super::preset::load_preset_obj;

// The Prop instance a prefab is expanded under: the name its generated assets
// are prefixed with, and the placement its entries are composed onto. Nested
// prefabs recurse with the composed placement as the new instance.
struct Instance<'a> {
    name: &'a str,
    position: [f32; 3],
    rotation_deg: [f32; 3],
    scale: [f32; 3],
}

pub(crate) fn expand_prefabs(
    asset_values: &mut Vec<serde_json::Value>,
    authored: &std::collections::HashMap<String, String>,
    report: &mut ExpandReport,
    assets_dir: Option<&Path>,
) -> Result<(), String> {
    // Collect all Prefab definitions.
    let mut prefab_defs: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();
    let mut non_prefab: Vec<serde_json::Value> = Vec::new();

    for value in asset_values.drain(..) {
        if type_norm(&value) == "prefab" {
            let name = asset_name(&value);
            if !name.is_empty() {
                prefab_defs.insert(name, value);
            }
        } else {
            non_prefab.push(value);
        }
    }

    // Names already present before any instance expands. A generated name
    // landing on an authored one is a patch; landing on anything else (an
    // earlier expansion's output or another instance's) is a conflict.
    let mut taken: std::collections::HashSet<String> = non_prefab.iter().map(asset_name).collect();

    // Expand Prop entries that reference a prefab.
    let mut result: Vec<serde_json::Value> = Vec::new();
    // Shadow hits found while expanding: the authored patch line may not be in
    // `result` yet, so the merges apply after the rebuild.
    let mut merges: Vec<(String, serde_json::Value)> = Vec::new();
    for value in non_prefab {
        if type_norm(&value) != "prop" {
            result.push(value);
            continue;
        }
        let prefab_ref = value
            .get("args")
            .and_then(|a| a.get("prefab"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if prefab_ref.is_empty() {
            result.push(value);
            continue;
        }

        let instance_name = asset_name(&value);
        let args = value.get("args").cloned().unwrap_or(serde_json::json!({}));
        let inst_pos = f32_arr3(&args, "position", [0.0, 0.0, 0.0]);
        let inst_rot = f32_arr3(&args, "rotation_deg", [0.0, 0.0, 0.0]);
        let inst_scale = f32_arr3(&args, "scale", [1.0, 1.0, 1.0]);

        let prefab_def = if let Some(def) = prefab_defs.get(prefab_ref) {
            def.clone()
        } else {
            let loaded = load_preset_obj(prefab_ref, "prefabs", assets_dir);
            if loaded.is_null() {
                return Err(format!(
                    "Prop '{}': prefab '{}' not found, declare a Prefab asset with that name",
                    instance_name, prefab_ref
                ));
            }
            loaded
        };

        let mut call_stack: Vec<String> = vec![prefab_ref.to_string()];
        let expanded = expand_prefab_entries(
            &Instance {
                name: &instance_name,
                position: inst_pos,
                rotation_deg: inst_rot,
                scale: inst_scale,
            },
            &prefab_def,
            &prefab_defs,
            &mut call_stack,
            assets_dir,
        )?;
        for entry in expanded {
            if resolve_generated(
                &entry,
                &instance_name,
                authored,
                &mut taken,
                report,
                &mut merges,
            )? {
                result.push(entry);
            }
        }
    }

    for (name, template_args) in &merges {
        super::shadow::merge_into_authored(&mut result, name, template_args);
    }
    *asset_values = result;
    Ok(())
}

// Whether one generated entry should be emitted, recording the outcome so
// every generated asset is accounted for. `false` means the world declares its
// own patch of the entry, which the caller merges the generated args under.
fn resolve_generated(
    entry: &serde_json::Value,
    instance_name: &str,
    authored: &std::collections::HashMap<String, String>,
    taken: &mut std::collections::HashSet<String>,
    report: &mut ExpandReport,
    merges: &mut Vec<(String, serde_json::Value)>,
) -> Result<bool, String> {
    let name = asset_name(entry);
    let entry_type = entry
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("?")
        .to_string();

    if let Some(authored_type) = authored.get(&name) {
        if authored_type.to_lowercase().replace('_', "") != type_norm(entry) {
            return Err(format!(
                "Prop '{}': generated asset '{}' ({}) collides with your {} asset of the same \
                 name; rename that asset or the instance",
                instance_name, name, entry_type, authored_type,
            ));
        }
        let args = entry.get("args").cloned().unwrap_or(serde_json::json!({}));
        report.record_shadowed(&name, authored_type, instance_name, args.clone());
        merges.push((name, args));
        return Ok(false);
    }

    if !taken.insert(name.clone()) {
        return Err(format!(
            "Prop '{}': generated asset name '{}' collides with another asset; rename the \
             instance or the prefab entry",
            instance_name, name
        ));
    }
    report.record_generated(&name, &entry_type, instance_name);
    Ok(true)
}

fn expand_prefab_entries(
    instance: &Instance<'_>,
    prefab_def: &serde_json::Value,
    prefab_defs: &std::collections::HashMap<String, serde_json::Value>,
    call_stack: &mut Vec<String>,
    assets_dir: Option<&Path>,
) -> Result<Vec<serde_json::Value>, String> {
    let (inst_pos, inst_rot, inst_scale) =
        (instance.position, instance.rotation_deg, instance.scale);
    let entries = prefab_def
        .get("args")
        .and_then(|a| a.get("props"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut result: Vec<serde_json::Value> = Vec::new();

    for entry in &entries {
        let kind = entry.get("kind").and_then(|v| v.as_str()).unwrap_or("prop");
        let entry_name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("obj");
        let expanded_name = format!("{}_{}", instance.name, entry_name);

        let local_pos = f32_arr3(entry, "position", [0.0, 0.0, 0.0]);
        let local_rot = f32_arr3(entry, "rotation_deg", [0.0, 0.0, 0.0]);
        let local_scale = f32_arr3(entry, "scale", [1.0, 1.0, 1.0]);

        let rotated = rotate_local(local_pos, inst_rot);
        let world_pos = [
            inst_pos[0] + inst_scale[0] * rotated[0],
            inst_pos[1] + inst_scale[1] * rotated[1],
            inst_pos[2] + inst_scale[2] * rotated[2],
        ];
        // Component-wise rotation composition (accurate for common yaw-only case).
        let world_rot = [
            inst_rot[0] + local_rot[0],
            inst_rot[1] + local_rot[1],
            inst_rot[2] + local_rot[2],
        ];
        let world_scale = [
            inst_scale[0] * local_scale[0],
            inst_scale[1] * local_scale[1],
            inst_scale[2] * local_scale[2],
        ];

        match kind {
            "point_light" => {
                let color = entry
                    .get("light_color")
                    .cloned()
                    .unwrap_or(serde_json::json!([1.0, 1.0, 1.0]));
                let intensity = entry
                    .get("light_intensity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(8.0);
                let range = entry
                    .get("light_range")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(6.0);
                result.push(serde_json::json!({
                    "name": expanded_name,
                    "type": "PointLight",
                    "args": {
                        "position": [world_pos[0], world_pos[1], world_pos[2]],
                        "color": color,
                        "intensity": intensity,
                        "range": range
                    }
                }));
            }
            "prefab" => {
                let nested_ref = entry.get("prefab").and_then(|v| v.as_str()).unwrap_or("");
                if nested_ref.is_empty() {
                    return Err(format!(
                        "Prefab entry '{}': kind=prefab but 'prefab' field is empty",
                        expanded_name
                    ));
                }
                if call_stack.contains(&nested_ref.to_string()) {
                    return Err(format!(
                        "Prefab '{}': cycle detected (via '{}')",
                        call_stack[0], nested_ref
                    ));
                }
                let nested_def = if let Some(def) = prefab_defs.get(nested_ref) {
                    def.clone()
                } else {
                    let loaded = load_preset_obj(nested_ref, "prefabs", assets_dir);
                    if loaded.is_null() {
                        return Err(format!(
                            "Prefab entry '{}': nested prefab '{}' not found",
                            expanded_name, nested_ref
                        ));
                    }
                    loaded
                };
                call_stack.push(nested_ref.to_string());
                let nested = expand_prefab_entries(
                    &Instance {
                        name: &expanded_name,
                        position: world_pos,
                        rotation_deg: world_rot,
                        scale: world_scale,
                    },
                    &nested_def,
                    prefab_defs,
                    call_stack,
                    assets_dir,
                )?;
                call_stack.pop();
                result.extend(nested);
            }
            _ => {
                // "prop"
                let collider = entry.get("collider").cloned();
                let mut prop_args = serde_json::json!({
                    "position":    [world_pos[0], world_pos[1], world_pos[2]],
                    "rotation_deg":[world_rot[0], world_rot[1], world_rot[2]],
                    "scale":       [world_scale[0], world_scale[1], world_scale[2]]
                });
                for field in &[
                    "model",
                    "mesh",
                    "material",
                    "texture",
                    "parent",
                    "interactable",
                    "pickup",
                ] {
                    if let Some(v) = entry.get(*field) {
                        prop_args[field] = v.clone();
                    }
                }
                if let Some(c) = collider {
                    prop_args["collider"] = c;
                }
                result.push(serde_json::json!({
                    "name": expanded_name,
                    "type": "Prop",
                    "args": prop_args
                }));
            }
        }
    }

    Ok(result)
}

// Rotate a 3-D local-space offset by a YXZ Euler rotation (degrees).
// Mirrors the rotation part of Prop::model_matrix().
fn rotate_local(pos: [f32; 3], rotation_deg: [f32; 3]) -> [f32; 3] {
    let [px, py, pz] = pos;
    let [pitch_deg, yaw_deg, roll_deg] = rotation_deg;
    let (sp, cp) = (pitch_deg.to_radians().sin(), pitch_deg.to_radians().cos());
    let (sy, cy) = (yaw_deg.to_radians().sin(), yaw_deg.to_radians().cos());
    let (sr, cr) = (roll_deg.to_radians().sin(), roll_deg.to_radians().cos());
    let rx = (cy * cr + sy * sp * sr) * px + (-cy * sr + sy * sp * cr) * py + (sy * cp) * pz;
    let ry = (cp * sr) * px + (cp * cr) * py + (-sp) * pz;
    let rz = (-sy * cr + cy * sp * sr) * px + (sy * sr + cy * sp * cr) * py + (cy * cp) * pz;
    [rx, ry, rz]
}

fn f32_arr3(v: &serde_json::Value, key: &str, default: [f32; 3]) -> [f32; 3] {
    v.get(key)
        .and_then(|a| a.as_array())
        .and_then(|a| {
            if a.len() == 3 {
                Some([
                    a[0].as_f64().unwrap_or(default[0] as f64) as f32,
                    a[1].as_f64().unwrap_or(default[1] as f64) as f32,
                    a[2].as_f64().unwrap_or(default[2] as f64) as f32,
                ])
            } else {
                None
            }
        })
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_norm_str(v: &serde_json::Value) -> String {
        v.get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_lowercase()
            .replace('_', "")
    }

    // Mirror expand_world: snapshot the authored names before expanding.
    fn expand(assets: &mut Vec<serde_json::Value>) -> Result<ExpandReport, String> {
        let authored: std::collections::HashMap<String, String> = assets
            .iter()
            .map(|v| {
                (
                    asset_name(v),
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("?")
                        .to_string(),
                )
            })
            .filter(|(n, _)| !n.is_empty())
            .collect();
        let mut report = ExpandReport::default();
        expand_prefabs(assets, &authored, &mut report, None)?;
        Ok(report)
    }

    #[test]
    fn single_instance_expands() {
        let mut assets = vec![
            serde_json::json!({"name":"box_mesh","type":"ProceduralMesh","args":{}}),
            serde_json::json!({"name":"table_set","type":"Prefab","args":{"props":[
                {"name":"table","kind":"prop","mesh":"box_mesh","position":[0,0,0]},
                {"name":"chair","kind":"prop","mesh":"box_mesh","position":[0,0,1]}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"table_set","position":[3,0,-5]}}),
        ];
        expand(&mut assets).unwrap();
        let names: Vec<&str> = assets
            .iter()
            .filter(|v| type_norm_str(v) == "prop")
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert!(names.contains(&"inst_table"));
        assert!(names.contains(&"inst_chair"));
        assert!(!assets.iter().any(|v| type_norm_str(v) == "prefab"));
    }

    #[test]
    fn two_instances_with_rotation() {
        let mut assets = vec![
            serde_json::json!({"name":"box","type":"ProceduralMesh","args":{}}),
            serde_json::json!({"name":"pair","type":"Prefab","args":{"props":[
                {"name":"a","kind":"prop","mesh":"box","position":[1,0,0]},
                {"name":"b","kind":"prop","mesh":"box","position":[-1,0,0]}
            ]}}),
            serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"pair","position":[0,0,0]}}),
            serde_json::json!({"name":"i2","type":"Prop","args":{"prefab":"pair","position":[10,0,0],"rotation_deg":[0,90,0]}}),
        ];
        expand(&mut assets).unwrap();
        let props: Vec<_> = assets
            .iter()
            .filter(|v| type_norm_str(v) == "prop")
            .collect();
        assert_eq!(props.len(), 4);
    }

    #[test]
    fn point_light_entry_expands() {
        let mut assets = vec![
            serde_json::json!({"name":"alcove","type":"Prefab","args":{"props":[
                {"name":"lamp","kind":"point_light","position":[0,2,0],
                 "light_color":[1.0,0.9,0.7],"light_intensity":8.0,"light_range":5.0}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"alcove","position":[5,0,-3]}}),
        ];
        expand(&mut assets).unwrap();
        let lights: Vec<_> = assets
            .iter()
            .filter(|v| type_norm_str(v) == "pointlight")
            .collect();
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0]["name"], "inst_lamp");
    }

    #[test]
    fn cycle_is_detected() {
        let mut assets = vec![
            serde_json::json!({"name":"pa","type":"Prefab","args":{"props":[
                {"name":"n","kind":"prefab","prefab":"pb"}
            ]}}),
            serde_json::json!({"name":"pb","type":"Prefab","args":{"props":[
                {"name":"n","kind":"prefab","prefab":"pa"}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"pa","position":[0,0,0]}}),
        ];
        let err = expand(&mut assets).unwrap_err();
        assert!(err.contains("cycle"));
    }

    #[test]
    fn missing_prefab_returns_error() {
        let mut assets = vec![
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"ghost","position":[0,0,0]}}),
        ];
        let err = expand(&mut assets).unwrap_err();
        assert!(err.contains("ghost"));
    }

    // A nested prefab expands under the outer instance's name and inherits its
    // placement: the inner offset is scaled and added to the outer one.
    #[test]
    fn nested_prefab_entries_compose_their_transforms() {
        let mut assets = vec![
            serde_json::json!({"name":"leaf","type":"Prefab","args":{"props":[
                {"name":"cup","kind":"prop","mesh":"box","position":[1,0,0],"scale":[2,2,2]}
            ]}}),
            serde_json::json!({"name":"table","type":"Prefab","args":{"props":[
                {"name":"top","kind":"prop","mesh":"box"},
                {"name":"set","kind":"prefab","prefab":"leaf","position":[0,1,0]}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{
                "prefab":"table","position":[10,0,0],"scale":[3,3,3]
            }}),
        ];
        expand(&mut assets).unwrap();
        let names: Vec<String> = assets.iter().map(asset_name).collect();
        assert_eq!(names, ["inst_top", "inst_set_cup"]);

        let cup = &assets[1]["args"];
        // inst(10,0,0) + 3 * [ set(0,1,0) + 1 * cup(1,0,0) ] -> (13, 3, 0).
        assert_eq!(cup["position"], serde_json::json!([13.0, 3.0, 0.0]));
        // Scales multiply all the way down: 3 * 1 * 2.
        assert_eq!(cup["scale"], serde_json::json!([6.0, 6.0, 6.0]));
    }

    #[test]
    fn nested_prefab_without_a_name_is_an_error() {
        let mut assets = vec![
            serde_json::json!({"name":"pa","type":"Prefab","args":{"props":[
                {"name":"n","kind":"prefab"}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"pa"}}),
        ];
        let err = expand(&mut assets).unwrap_err();
        assert!(err.contains("inst_n"), "{err}");
        assert!(err.contains("'prefab' field is empty"), "{err}");
    }

    #[test]
    fn undeclared_nested_prefab_is_an_error() {
        let mut assets = vec![
            serde_json::json!({"name":"pa","type":"Prefab","args":{"props":[
                {"name":"n","kind":"prefab","prefab":"ghost"}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"pa"}}),
        ];
        let err = expand(&mut assets).unwrap_err();
        assert!(err.contains("nested prefab 'ghost' not found"), "{err}");
    }

    // Prop fields the template carries ride along to the instance, including
    // the collider, which is only written when the entry declares one.
    #[test]
    fn prop_entry_fields_and_collider_carry_through() {
        let mut assets = vec![
            serde_json::json!({"name":"crate_set","type":"Prefab","args":{"props":[
                {"name":"a","kind":"prop","model":"m","material":"mat","texture":"t",
                 "parent":"p","interactable":true,"pickup":true,
                 "collider":{"shape":"box"}},
                {"name":"b","kind":"prop","mesh":"box"}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"crate_set"}}),
        ];
        expand(&mut assets).unwrap();
        let a = &assets[0]["args"];
        assert_eq!(a["model"], "m");
        assert_eq!(a["material"], "mat");
        assert_eq!(a["texture"], "t");
        assert_eq!(a["parent"], "p");
        assert_eq!(a["interactable"], true);
        assert_eq!(a["pickup"], true);
        assert_eq!(a["collider"], serde_json::json!({"shape":"box"}));
        // A entry without a collider does not grow an empty one.
        assert!(assets[1]["args"].get("collider").is_none());
    }

    // A Prop that names no prefab is a plain prop and passes through untouched,
    // and an unnamed Prefab cannot be referenced so it is simply consumed.
    #[test]
    fn plain_props_pass_through_and_unnamed_prefabs_are_dropped() {
        let mut assets = vec![
            serde_json::json!({"type":"Prefab","args":{"props":[{"name":"x","kind":"prop"}]}}),
            serde_json::json!({"name":"lamp","type":"Prop","args":{"mesh":"box"}}),
        ];
        expand(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["name"], "lamp");
    }

    // A malformed vector (not exactly three numbers) falls back to the default
    // rather than being read component-wise.
    #[test]
    fn a_non_triple_vector_falls_back_to_the_default() {
        let v = serde_json::json!({"position": [1.0, 2.0], "scale": "big"});
        assert_eq!(f32_arr3(&v, "position", [7.0, 8.0, 9.0]), [7.0, 8.0, 9.0]);
        assert_eq!(f32_arr3(&v, "scale", [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]);
        // A non-numeric component falls back per component.
        let mixed = serde_json::json!({"position": [1.0, "x", 3.0]});
        assert_eq!(
            f32_arr3(&mixed, "position", [0.0, 5.0, 0.0]),
            [1.0, 5.0, 3.0]
        );
    }

    #[test]
    fn an_authored_patch_overrides_only_the_fields_it_names() {
        let mut assets = vec![
            serde_json::json!({"name":"pair","type":"Prefab","args":{"props":[
                {"name":"a","kind":"prop","mesh":"box","position":[1,0,0]},
            ]}}),
            serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"pair","position":[10,0,0]}}),
            serde_json::json!({"name":"i1_a","type":"Prop","args":{"material":"gold"}}),
        ];
        let report = expand(&mut assets).unwrap();

        let a = assets.iter().find(|v| asset_name(v) == "i1_a").unwrap();
        // The patched field wins; the generated transform survives the merge.
        assert_eq!(a["args"]["material"], "gold");
        assert_eq!(a["args"]["position"], serde_json::json!([11.0, 0.0, 0.0]));
        assert_eq!(a["args"]["mesh"], "box");

        assert_eq!(report.shadowed.len(), 1);
        let shadow = &report.shadowed[0];
        assert_eq!(shadow.name, "i1_a");
        assert_eq!(shadow.generated_by, "i1");
        // The recorded baseline is the pre-merge generated args.
        assert_eq!(shadow.args["position"], serde_json::json!([11.0, 0.0, 0.0]));
        assert!(shadow.args.get("material").is_none());
        assert!(report.generated.is_empty());
    }

    #[test]
    fn generated_assets_are_recorded_against_their_instance() {
        let mut assets = vec![
            serde_json::json!({"name":"pair","type":"Prefab","args":{"props":[
                {"name":"a","kind":"prop","mesh":"box"},
                {"name":"lamp","kind":"point_light","position":[0,2,0]},
            ]}}),
            serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"pair"}}),
            serde_json::json!({"name":"i2","type":"Prop","args":{"prefab":"pair","position":[5,0,0]}}),
        ];
        let report = expand(&mut assets).unwrap();
        let by: Vec<(&str, &str)> = report
            .generated
            .iter()
            .map(|g| (g.name.as_str(), g.generated_by.as_str()))
            .collect();
        assert!(by.contains(&("i1_a", "i1")));
        assert!(by.contains(&("i1_lamp", "i1")));
        assert!(by.contains(&("i2_a", "i2")));
        assert!(by.contains(&("i2_lamp", "i2")));
    }

    #[test]
    fn a_patch_with_the_wrong_type_is_a_hard_error() {
        let mut assets = vec![
            serde_json::json!({"name":"pair","type":"Prefab","args":{"props":[
                {"name":"a","kind":"prop","mesh":"box"},
            ]}}),
            serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"pair"}}),
            serde_json::json!({"name":"i1_a","type":"Sprite","args":{}}),
        ];
        let err = expand(&mut assets).unwrap_err();
        assert!(err.contains("collides"), "{err}");
    }

    #[test]
    fn two_instances_generating_one_name_is_a_hard_error() {
        let mut assets = vec![
            serde_json::json!({"name":"pa","type":"Prefab","args":{"props":[
                {"name":"b_c","kind":"prop","mesh":"box"},
            ]}}),
            serde_json::json!({"name":"pb","type":"Prefab","args":{"props":[
                {"name":"c","kind":"prop","mesh":"box"},
            ]}}),
            serde_json::json!({"name":"a","type":"Prop","args":{"prefab":"pa"}}),
            serde_json::json!({"name":"a_b","type":"Prop","args":{"prefab":"pb"}}),
        ];
        let err = expand(&mut assets).unwrap_err();
        assert!(err.contains("a_b_c"), "{err}");
    }

    #[test]
    fn a_point_light_patch_merges_over_the_generated_light() {
        let mut assets = vec![
            serde_json::json!({"name":"alcove","type":"Prefab","args":{"props":[
                {"name":"lamp","kind":"point_light","position":[0,2,0],
                 "light_color":[1.0,0.9,0.7],"light_intensity":8.0,"light_range":5.0}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"alcove"}}),
            serde_json::json!({"name":"inst_lamp","type":"PointLight","args":{"intensity":2.0}}),
        ];
        expand(&mut assets).unwrap();
        let lamp = assets
            .iter()
            .find(|v| asset_name(v) == "inst_lamp")
            .unwrap();
        assert_eq!(lamp["args"]["intensity"], 2.0);
        assert_eq!(lamp["args"]["color"], serde_json::json!([1.0, 0.9, 0.7]));
        assert_eq!(lamp["args"]["range"], 5.0);
    }

    #[test]
    fn rotate_local_identity_at_zero_rotation() {
        let result = rotate_local([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!(result[1].abs() < 1e-5);
        assert!(result[2].abs() < 1e-5);
    }

    #[test]
    fn rotate_local_yaw_90_rotates_x_to_neg_z() {
        let result = rotate_local([1.0, 0.0, 0.0], [0.0, 90.0, 0.0]);
        assert!(result[0].abs() < 1e-5);
        assert!(result[1].abs() < 1e-5);
        assert!((result[2] - (-1.0)).abs() < 1e-5);
    }
}
