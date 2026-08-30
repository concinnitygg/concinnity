// src/editor/select_related.rs
//
// Pure resolution for the /select console command: which working entries share
// an origin group with a name, reference a given asset, or are of a type. The
// origin rule reuses the outliner's grouping (`asset_tree::groups_from`), so
// "same origin" always means exactly what the Assets tree shows.

use super::asset_tree::TreeGroup;

// Every name in the origin group holding `name`, or `None` when no group
// lists it.
pub(crate) fn same_group(groups: &[TreeGroup], name: &str) -> Option<Vec<String>> {
    groups
        .iter()
        .find(|g| g.assets.iter().any(|a| a.name == name))
        .map(|g| g.assets.iter().map(|a| a.name.clone()).collect())
}

// The names of every working entry whose reference set contains `target`.
pub(crate) fn names_using(entries: &[serde_json::Value], target: &str) -> Vec<String> {
    entries
        .iter()
        .filter_map(|e| {
            let asset = concinnity_cook::authoring::world::WorldJsonlAsset::from_value(e);
            if asset.name.is_empty() {
                return None;
            }
            concinnity_cook::authoring::refs::referenced_names(&asset)
                .iter()
                .any(|r| r == target)
                .then_some(asset.name)
        })
        .collect()
}

// The names of every working entry of type `ty` (exact match).
pub(crate) fn names_of_type(entries: &[serde_json::Value], ty: &str) -> Vec<String> {
    entries
        .iter()
        .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some(ty))
        .filter_map(|e| e.get("name").and_then(|v| v.as_str()).map(String::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::asset_tree::{Badge, TreeAsset};
    use super::*;

    fn group(label: &str, names: &[&str]) -> TreeGroup {
        TreeGroup {
            label: label.to_string(),
            assets: names
                .iter()
                .map(|n| TreeAsset {
                    name: n.to_string(),
                    asset_type: "Prop".to_string(),
                    badge: Badge::Authored,
                    promote: None,
                })
                .collect(),
        }
    }

    #[test]
    fn same_group_returns_the_whole_group() {
        let groups = [group("World", &["a", "b"]), group("bistro", &["c", "d"])];
        assert_eq!(same_group(&groups, "c").unwrap(), vec!["c", "d"]);
        assert_eq!(same_group(&groups, "a").unwrap(), vec!["a", "b"]);
        assert_eq!(same_group(&groups, "zzz"), None);
    }

    #[test]
    fn names_using_walks_the_reference_graph() {
        let entries = vec![
            serde_json::json!({"name":"p1","type":"Prop","args":{"mesh":"box","material":"mat"}}),
            serde_json::json!({"name":"p2","type":"Prop","args":{"mesh":"box"}}),
            serde_json::json!({"name":"mat","type":"Material","args":{}}),
        ];
        assert_eq!(names_using(&entries, "mat"), vec!["p1"]);
        assert_eq!(names_using(&entries, "box"), vec!["p1", "p2"]);
        assert!(names_using(&entries, "nothing").is_empty());
    }

    #[test]
    fn names_of_type_matches_exactly() {
        let entries = vec![
            serde_json::json!({"name":"p1","type":"Prop","args":{}}),
            serde_json::json!({"name":"m","type":"Material","args":{}}),
        ];
        assert_eq!(names_of_type(&entries, "Prop"), vec!["p1"]);
        assert!(names_of_type(&entries, "prop").is_empty());
    }
}
