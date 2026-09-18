//! Pure resolution for the /select console command: which working entries share
//! an origin group with a name, reference a given asset, or are of a type. The
//! origin rule reuses the outliner's grouping (`asset_tree::groups_from`), so
//! "same origin" always means exactly what the Assets tree shows.
//!
//! The two rules that read the entry list answer with positions rather than
//! names, so an entry that declares no name is selectable like any other; the
//! caller turns a position into the handle its entry is addressed by. The
//! origin rule reads the cooked tree, whose rows are names either way.

use super::panels::asset_tree::TreeGroup;

// Every name in the origin group holding `name`, or `None` when no group
// lists it.
pub(crate) fn same_group(groups: &[TreeGroup], name: &str) -> Option<Vec<String>> {
    groups
        .iter()
        .find(|g| g.assets.iter().any(|a| a.name == name))
        .map(|g| g.assets.iter().map(|a| a.name.clone()).collect())
}

// The positions of every working entry whose reference set contains `target`.
pub(crate) fn entries_using(entries: &[serde_json::Value], target: &str) -> Vec<usize> {
    entries
        .iter()
        .enumerate()
        .filter_map(|(i, e)| {
            let asset = concinnity_cook::authoring::world::WorldJsonlAsset::from_value(e).ok()?;
            concinnity_cook::authoring::refs::referenced_names(&asset)
                .iter()
                .any(|r| r == target)
                .then_some(i)
        })
        .collect()
}

// The positions of every working entry of type `ty` (exact match).
pub(crate) fn entries_of_type(entries: &[serde_json::Value], ty: &str) -> Vec<usize> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.get("type").and_then(|v| v.as_str()) == Some(ty))
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::panels::asset_tree::{Badge, TreeAsset};
    use super::*;
    use concinnity_cook::authoring::registry::RegisteredType;

    fn group(label: &str, names: &[&str]) -> TreeGroup {
        TreeGroup {
            label: label.to_string(),
            assets: names
                .iter()
                .map(|n| TreeAsset {
                    name: n.to_string(),
                    asset_type: RegisteredType::Prop,
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
    fn entries_using_walks_the_reference_graph() {
        let entries = vec![
            serde_json::json!({"type":"Prop","args":{"$id":"p1","mesh":"box","material":"mat"}}),
            serde_json::json!({"type":"Prop","args":{"$id":"p2","mesh":"box"}}),
            serde_json::json!({"type":"Material","args":{"$id":"mat"}}),
        ];
        assert_eq!(entries_using(&entries, "mat"), vec![0]);
        assert_eq!(entries_using(&entries, "box"), vec![0, 1]);
        assert!(entries_using(&entries, "nothing").is_empty());
    }

    // A referencing entry with no identity of its own is still selectable: it is
    // the caller's handle that addresses it, not its name.
    #[test]
    fn entries_using_keeps_a_referrer_with_no_identity() {
        let entries = vec![
            serde_json::json!({"type":"Prop","args":{"$id":"","mesh":"box"}}),
            serde_json::json!({"type":"ProceduralMesh","args":{"$id":"box"}}),
        ];
        assert_eq!(entries_using(&entries, "box"), vec![0]);
    }

    #[test]
    fn entries_of_type_matches_exactly() {
        let entries = vec![
            serde_json::json!({"type":"Prop","args":{"$id":"p1"}}),
            serde_json::json!({"type":"Prop","args":{"$id":""}}),
            serde_json::json!({"type":"Material","args":{"$id":"m"}}),
        ];
        assert_eq!(entries_of_type(&entries, "Prop"), vec![0, 1]);
        assert!(entries_of_type(&entries, "prop").is_empty());
    }
}
