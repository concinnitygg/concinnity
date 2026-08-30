// src/editor/overrides.rs
//
// Per-field override state for template-derived assets. A generated or
// injected asset's authored world.jsonl line is a sparse patch over what the
// expansion produces (cook's patch-merge shadowing); everything here derives
// field-level state from that patch, purely, so nothing persisted can go
// stale: template, patch, and classification are all recomputed from the
// working entries on every cook.

pub(crate) mod prefab_map;

use std::collections::{BTreeMap, HashSet};

use concinnity_cook::build_only::LoadedWorld;
use serde_json::Value;

// How one form field relates to the asset's template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldOrigin {
    // The value comes from the template (no patch entry covers it).
    Inherited,
    // The patch pins it; the template value is shadowed.
    Overridden,
    // The patch pins it and the template has no counterpart (an array element
    // past the template's length, a nested key the template lacks).
    InstanceOnly,
}

// One template-derived asset: what produced it and the args it would have
// without the authored patch.
#[derive(Debug, Clone)]
pub(crate) struct TemplateInfo {
    pub asset_type: String,
    // The args the expansion produced (pre-merge for a patched asset, the
    // asset's own args for a pristine one).
    pub baseline: Value,
    // The authored asset (Prop instance, SceneImport) or injection pass that
    // produced it.
    pub generated_by: String,
}

// Every template-derived asset in the expanded world, by name.
#[derive(Debug, Default)]
pub(crate) struct TemplateIndex {
    map: BTreeMap<String, TemplateInfo>,
}

impl TemplateIndex {
    pub(crate) fn from_loaded(loaded: &LoadedWorld) -> Self {
        let mut map = BTreeMap::new();
        for s in &loaded.shadowed {
            map.insert(
                s.name.clone(),
                TemplateInfo {
                    asset_type: s.asset_type.clone(),
                    baseline: s.args.clone(),
                    generated_by: s.generated_by.clone(),
                },
            );
        }
        let pristine_args = |name: &str| {
            loaded
                .assets
                .iter()
                .find(|a| a.name == name)
                .map(|a| a.args.clone())
        };
        for g in &loaded.generated {
            if map.contains_key(&g.name) {
                continue;
            }
            let Some(args) = pristine_args(&g.name) else {
                continue;
            };
            map.insert(
                g.name.clone(),
                TemplateInfo {
                    asset_type: g.asset_type.clone(),
                    baseline: args,
                    generated_by: g.generated_by.clone(),
                },
            );
        }
        for i in &loaded.injected {
            if map.contains_key(&i.name) {
                continue;
            }
            map.insert(
                i.name.clone(),
                TemplateInfo {
                    asset_type: i.asset_type.clone(),
                    baseline: i.args.clone(),
                    generated_by: i.injected_by.to_string(),
                },
            );
        }
        TemplateIndex { map }
    }

    pub(crate) fn get(&self, name: &str) -> Option<&TemplateInfo> {
        self.map.get(name)
    }
}

// Classify the field at dotted `path` against the committed patch. A path is
// covered as soon as the walk lands on a patch value it cannot descend past:
// arrays replace wholesale, so any element path under a patched array is
// overridden.
pub(crate) fn classify(template: &Value, patch: &Value, path: &str) -> FieldOrigin {
    let mut cur = patch;
    for seg in path.split('.') {
        match cur {
            Value::Object(map) => match map.get(seg) {
                Some(next) => cur = next,
                None => return FieldOrigin::Inherited,
            },
            // A non-object patch value covers everything below it.
            _ => break,
        }
    }
    if value_at_path(template, path).is_none() {
        FieldOrigin::InstanceOnly
    } else {
        FieldOrigin::Overridden
    }
}

// The dotted prefix of `path` the patch is anchored at: the point where the
// walk lands on a non-object (a whole replaced array / scalar / null), or the
// full path when the patch descends all the way. `None` when the patch does
// not cover the path (the field is inherited).
pub(crate) fn covered_root(patch: &Value, path: &str) -> Option<String> {
    let mut cur = patch;
    let mut taken: Vec<&str> = Vec::new();
    for seg in path.split('.') {
        match cur {
            Value::Object(map) => {
                cur = map.get(seg)?;
                taken.push(seg);
            }
            _ => break,
        }
    }
    Some(taken.join("."))
}

// The value at dotted `path`, descending objects by key and arrays by index.
pub(crate) fn value_at_path<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = v;
    for seg in path.split('.') {
        cur = match cur {
            Value::Object(map) => map.get(seg)?,
            Value::Array(arr) => arr.get(seg.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

// Remove the key at dotted `path` from a patch, pruning parent objects the
// removal empties. Returns whether anything was removed. Only object segments
// are walked: array contents are covered by their root key, which is what
// `covered_root` hands in.
pub(crate) fn remove_at_path(patch: &mut Value, path: &str) -> bool {
    let Value::Object(map) = patch else {
        return false;
    };
    match path.split_once('.') {
        None => map.remove(path).is_some(),
        Some((head, rest)) => {
            let Some(child) = map.get_mut(head) else {
                return false;
            };
            let removed = remove_at_path(child, rest);
            if removed && child.as_object().is_some_and(|o| o.is_empty()) {
                map.remove(head);
            }
            removed
        }
    }
}

// Every covered path a patch anchors: the dotted paths to its non-object
// values (objects descend; arrays, scalars, and nulls are wholesale
// replacements and stop the walk).
pub(crate) fn patch_roots(patch: &Value) -> Vec<String> {
    fn walk(v: &Value, prefix: &str, out: &mut Vec<String>) {
        match v {
            Value::Object(map) => {
                for (key, child) in map {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    walk(child, &path, out);
                }
            }
            _ => {
                if !prefix.is_empty() {
                    out.push(prefix.to_string());
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(patch, "", &mut out);
    out
}

// The sparse patch that turns `template` into `full`: only differing keys,
// nested objects recursed so equal siblings drop out, arrays and scalars
// replaced wholesale. `None` when the two are equal (no overrides).
pub(crate) fn minimal_patch(template: &Value, full: &Value) -> Option<Value> {
    if template == full {
        return None;
    }
    match (template, full) {
        (Value::Object(t), Value::Object(f)) => {
            let mut out = serde_json::Map::new();
            for (key, fv) in f {
                match t.get(key) {
                    Some(tv) => {
                        if let Some(diff) = minimal_patch(tv, fv) {
                            out.insert(key.clone(), diff);
                        }
                    }
                    None => {
                        out.insert(key.clone(), fv.clone());
                    }
                }
            }
            (!out.is_empty()).then_some(Value::Object(out))
        }
        _ => Some(full.clone()),
    }
}

// How many Prop instances a change to the Prefab named `def` reaches: every
// Prop whose prefab chain (directly or through nested prefab entries in other
// authored definitions) includes it.
pub(crate) fn instance_count(entries: &[Value], def: &str) -> usize {
    let mut reached: HashSet<String> = HashSet::new();
    reached.insert(def.to_string());
    // Fixed point over authored defs: a def whose entries reference a reached
    // def is itself reached.
    loop {
        let mut grew = false;
        for e in entries {
            if type_norm(e) != "prefab" {
                continue;
            }
            let name = entry_name(e);
            if name.is_empty() || reached.contains(&name) {
                continue;
            }
            if prefab_refs(e).iter().any(|r| reached.contains(r)) {
                reached.insert(name);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    entries
        .iter()
        .filter(|e| type_norm(e) == "prop")
        .filter(|e| {
            e.get("args")
                .and_then(|a| a.get("prefab"))
                .and_then(|v| v.as_str())
                .is_some_and(|r| reached.contains(r))
        })
        .count()
}

// The nested prefab names a Prefab definition's entries reference.
fn prefab_refs(def: &Value) -> Vec<String> {
    def.get("args")
        .and_then(|a| a.get("props"))
        .and_then(|v| v.as_array())
        .map(|props| {
            props
                .iter()
                .filter(|p| p.get("kind").and_then(|k| k.as_str()) == Some("prefab"))
                .filter_map(|p| p.get("prefab").and_then(|v| v.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn type_norm(v: &Value) -> String {
    v.get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_lowercase()
        .replace('_', "")
}

fn entry_name(v: &Value) -> String {
    v.get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_path_absent_from_the_patch_is_inherited() {
        let template = json!({"position": [1, 0, 0], "mesh": "box"});
        let patch = json!({"material": "gold"});
        assert_eq!(
            classify(&template, &patch, "position"),
            FieldOrigin::Inherited
        );
    }

    #[test]
    fn a_patched_scalar_and_its_template_counterpart_classify_overridden() {
        let template = json!({"mesh": "box"});
        let patch = json!({"mesh": "sphere"});
        assert_eq!(classify(&template, &patch, "mesh"), FieldOrigin::Overridden);
    }

    #[test]
    fn nested_leaves_classify_through_object_patches() {
        let template = json!({"collider": {"shape": "box", "radius": 1.0}});
        let patch = json!({"collider": {"shape": "sphere"}});
        assert_eq!(
            classify(&template, &patch, "collider.shape"),
            FieldOrigin::Overridden
        );
        assert_eq!(
            classify(&template, &patch, "collider.radius"),
            FieldOrigin::Inherited
        );
    }

    #[test]
    fn element_paths_under_a_patched_array_are_covered_wholesale() {
        let template = json!({"waves": [{"amplitude": 1.0}]});
        let patch = json!({"waves": [{"amplitude": 2.0}, {"amplitude": 3.0}]});
        assert_eq!(
            classify(&template, &patch, "waves.0.amplitude"),
            FieldOrigin::Overridden
        );
        // The second element has no template counterpart.
        assert_eq!(
            classify(&template, &patch, "waves.1.amplitude"),
            FieldOrigin::InstanceOnly
        );
    }

    #[test]
    fn covered_root_stops_at_a_wholesale_replacement() {
        let patch = json!({"waves": [{"amplitude": 2.0}], "collider": {"shape": "s"}});
        assert_eq!(
            covered_root(&patch, "waves.0.amplitude").as_deref(),
            Some("waves")
        );
        assert_eq!(
            covered_root(&patch, "collider.shape").as_deref(),
            Some("collider.shape")
        );
        assert_eq!(covered_root(&patch, "position"), None);
    }

    #[test]
    fn remove_at_path_prunes_emptied_parents() {
        let mut patch = json!({"collider": {"shape": "s"}, "mesh": "box"});
        assert!(remove_at_path(&mut patch, "collider.shape"));
        assert_eq!(patch, json!({"mesh": "box"}));
        assert!(remove_at_path(&mut patch, "mesh"));
        assert_eq!(patch, json!({}));
        assert!(!remove_at_path(&mut patch, "mesh"));
    }

    #[test]
    fn patch_roots_lists_every_covered_path() {
        let patch = json!({"mesh": "box", "collider": {"shape": "s"}, "waves": [1, 2]});
        let mut roots = patch_roots(&patch);
        roots.sort();
        assert_eq!(roots, vec!["collider.shape", "mesh", "waves"]);
    }

    #[test]
    fn minimal_patch_keeps_only_differing_leaves() {
        let template = json!({"a": 1, "b": {"c": 2, "d": 3}, "e": [1, 2]});
        let full = json!({"a": 1, "b": {"c": 9, "d": 3}, "e": [1, 2]});
        assert_eq!(
            minimal_patch(&template, &full),
            Some(json!({"b": {"c": 9}}))
        );
    }

    #[test]
    fn minimal_patch_of_equal_values_is_none() {
        let v = json!({"a": [1, 2], "b": {"c": null}});
        assert_eq!(minimal_patch(&v, &v.clone()), None);
    }

    #[test]
    fn minimal_patch_replaces_a_differing_array_wholesale() {
        let template = json!({"e": [1, 2]});
        let full = json!({"e": [1, 2, 3]});
        assert_eq!(
            minimal_patch(&template, &full),
            Some(json!({"e": [1, 2, 3]}))
        );
    }

    #[test]
    fn minimal_patch_keeps_keys_the_template_lacks() {
        let template = json!({"a": 1});
        let full = json!({"a": 1, "extra": true});
        assert_eq!(
            minimal_patch(&template, &full),
            Some(json!({"extra": true}))
        );
    }

    #[test]
    fn instance_count_follows_nested_prefab_chains() {
        let entries = vec![
            json!({"name":"leaf","type":"Prefab","args":{"props":[
                {"name":"cup","kind":"prop","mesh":"box"}]}}),
            json!({"name":"table","type":"Prefab","args":{"props":[
                {"name":"set","kind":"prefab","prefab":"leaf"}]}}),
            json!({"name":"i1","type":"Prop","args":{"prefab":"table"}}),
            json!({"name":"i2","type":"Prop","args":{"prefab":"leaf"}}),
            json!({"name":"plain","type":"Prop","args":{"mesh":"box"}}),
        ];
        assert_eq!(instance_count(&entries, "leaf"), 2);
        assert_eq!(instance_count(&entries, "table"), 1);
    }

    #[test]
    fn template_index_prefers_shadow_baselines_and_falls_back_per_kind() {
        use concinnity_cook::authoring::world::WorldJsonlAsset;
        use concinnity_cook::build_only::{
            GeneratedAsset, InjectedAsset, LoadedWorld, ShadowedAsset,
        };
        let loaded = LoadedWorld {
            assets: vec![WorldJsonlAsset {
                name: "i1_a".into(),
                asset_type: "Prop".into(),
                args: json!({"mesh": "box"}),
            }],
            injected: vec![InjectedAsset {
                name: "hud".into(),
                asset_type: "DebugHud".into(),
                args: json!({"enabled": true}),
                injected_by: "debug_hud",
            }],
            generated: vec![GeneratedAsset {
                name: "i1_a".into(),
                asset_type: "Prop".into(),
                generated_by: "i1".into(),
            }],
            shadowed: vec![ShadowedAsset {
                name: "i1_b".into(),
                asset_type: "Prop".into(),
                generated_by: "i1".into(),
                args: json!({"mesh": "sphere"}),
            }],
            authored: vec!["i1_b".into()],
        };
        let index = TemplateIndex::from_loaded(&loaded);
        assert_eq!(
            index.get("i1_b").unwrap().baseline,
            json!({"mesh": "sphere"})
        );
        assert_eq!(index.get("i1_a").unwrap().baseline, json!({"mesh": "box"}));
        assert_eq!(index.get("i1_a").unwrap().generated_by, "i1");
        assert_eq!(index.get("hud").unwrap().generated_by, "debug_hud");
        assert!(index.get("plain").is_none());
    }
}
