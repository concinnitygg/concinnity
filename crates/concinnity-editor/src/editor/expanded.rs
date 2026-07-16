// src/editor/expanded.rs
//
// The data half of the Assets panel's "Expanded" tab: everything a build adds to
// the world that has no world.jsonl line of its own -- a scene import's meshes
// and materials, the engine's injected defaults, the primitives a menu or story
// expands to. The tab exists so those assets are visible (they are most of a
// world's content, yet the Config tab never shows them) and so one can be copied
// into world.jsonl to edit.
//
// Everything here is pure: `groups_from` turns a cooked `LoadedWorld` into the
// grouped model, and `rows` flattens it against the collapse state. The panel
// draws the rows, and the hook owns when to re-cook. Grouping is by origin
// rather than by type because a single scene import expands to thousands of
// assets: collapsed, a bistro world's Expanded tab is a handful of headers.

use concinnity_cook::world::LoadedWorld;

// One expansion-produced asset, as the tab lists it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpandedAsset {
    pub name: String,
    pub asset_type: String,
    // The world.jsonl entry that copying this asset would add. `None` when the
    // asset cannot be overridden by a copy (see `Provenance::is_overridable`),
    // in which case the row is informational and offers no "+".
    pub entry: Option<serde_json::Value>,
    // The world already declares this asset: its own line wins, so the row is
    // shown grayed with no "+".
    pub in_config: bool,
}

// The assets one authored asset or injection pass produced.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpandedGroup {
    // What produced these ("bistro" for a scene import, "debug_hud" for an
    // engine default), or `UNATTRIBUTED` for the macro expansions that record
    // no source.
    pub origin: String,
    pub assets: Vec<ExpandedAsset>,
}

impl ExpandedGroup {
    // How many of the group's assets are already in the config.
    pub(crate) fn in_config_count(&self) -> usize {
        self.assets.iter().filter(|a| a.in_config).count()
    }
}

// The group holding assets whose producing pass records no source: menu, story,
// and prefab primitives. They are listed so the tab is a complete account of the
// world, but they cannot be copied out (their passes emit unconditionally, so a
// copy would sit beside the generated asset rather than replace it).
pub(crate) const UNATTRIBUTED: &str = "Other expansions";

// Build the grouped model from a cooked world. `assets` supplies each generated
// asset's full entry (the tab copies it verbatim); the shadow records supply the
// assets that are missing from `assets` precisely because the config overrides
// them, which is what makes an already-copied asset stay listed (grayed) rather
// than vanish from the tab.
pub(crate) fn groups_from(loaded: &LoadedWorld) -> Vec<ExpandedGroup> {
    let mut groups: Vec<ExpandedGroup> = Vec::new();

    for asset in &loaded.assets {
        let prov = loaded.provenance(&asset.name);
        if prov.is_authored() {
            continue;
        }
        let origin = prov.source().unwrap_or(UNATTRIBUTED).to_string();
        let entry = prov.is_overridable().then(|| entry_of(loaded, asset));
        push(
            &mut groups,
            origin,
            ExpandedAsset {
                name: asset.name.clone(),
                asset_type: asset.asset_type.clone(),
                entry,
                in_config: false,
            },
        );
    }

    // The assets the config already overrides. They are not in `loaded.assets`
    // under their generated form, so they come from the shadow records.
    for s in &loaded.shadowed {
        push(
            &mut groups,
            s.generated_by.clone(),
            ExpandedAsset {
                name: s.name.clone(),
                asset_type: s.asset_type.clone(),
                entry: None,
                in_config: true,
            },
        );
    }

    for g in &mut groups {
        g.assets.sort_by(|a, b| a.name.cmp(&b.name));
    }
    // Attributed groups first, alphabetically; the catch-all last, since it is
    // the one group nothing can be done with.
    groups.sort_by(|a, b| {
        let key = |g: &ExpandedGroup| (g.origin == UNATTRIBUTED, g.origin.clone());
        key(a).cmp(&key(b))
    });
    groups
}

// The world.jsonl entry for a generated asset: name, type, and the args the
// expansion gave it, in the same shape `cn explain` prints for copying.
fn entry_of(
    loaded: &LoadedWorld,
    asset: &concinnity_cook::world::WorldJsonlAsset,
) -> serde_json::Value {
    let args = loaded
        .injected
        .iter()
        .find(|i| i.name == asset.name)
        .map(|i| i.args.clone())
        .unwrap_or_else(|| asset.args.clone());
    serde_json::json!({
        "name": asset.name,
        "type": asset.asset_type,
        "args": args,
    })
}

fn push(groups: &mut Vec<ExpandedGroup>, origin: String, asset: ExpandedAsset) {
    match groups.iter_mut().find(|g| g.origin == origin) {
        Some(g) => g.assets.push(asset),
        None => groups.push(ExpandedGroup {
            origin,
            assets: vec![asset],
        }),
    }
}

// One rendered row of the tab: a group header (always shown, click to expand) or
// one of an expanded group's assets.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExpandedRow {
    Header {
        // Index into the groups list, for the collapse toggle.
        group: usize,
        origin: String,
        count: usize,
        in_config: usize,
        open: bool,
    },
    Asset {
        group: usize,
        // Index into the group's assets.
        index: usize,
        name: String,
        asset_type: String,
        // Grayed, no "+": the config already declares it.
        in_config: bool,
        // Whether the row offers a "+" (it can be copied and is not already in
        // the config).
        addable: bool,
    },
}

// Flatten the groups into rows: every header, plus the assets of each group in
// `open`. Groups are collapsed by default -- a scene import expands to thousands
// of assets, so an open-by-default tab would be unusable.
pub(crate) fn rows(groups: &[ExpandedGroup], open: &[usize]) -> Vec<ExpandedRow> {
    let mut out = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let is_open = open.contains(&gi);
        out.push(ExpandedRow::Header {
            group: gi,
            origin: g.origin.clone(),
            count: g.assets.len(),
            in_config: g.in_config_count(),
            open: is_open,
        });
        if !is_open {
            continue;
        }
        for (ai, a) in g.assets.iter().enumerate() {
            out.push(ExpandedRow::Asset {
                group: gi,
                index: ai,
                name: a.name.clone(),
                asset_type: a.asset_type.clone(),
                in_config: a.in_config,
                addable: !a.in_config && a.entry.is_some(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cook::world::{GeneratedAsset, InjectedAsset, ShadowedAsset, WorldJsonlAsset};

    fn asset(name: &str, ty: &str) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args: serde_json::json!({"k": 1}),
        }
    }

    // A world with one scene import (one generated asset plus one the config
    // overrides), one injected default, one macro-expanded primitive, and one
    // plain authored line.
    fn loaded() -> LoadedWorld {
        LoadedWorld {
            assets: vec![
                asset("cam", "Camera3D"),
                asset("fox_mat_0", "Material"),
                asset("fox_mat_glass", "Material"),
                asset("hud_font", "Font"),
                asset("menu_tab_0", "TextLabel"),
            ],
            injected: vec![InjectedAsset {
                name: "hud_font".to_string(),
                asset_type: "Font".to_string(),
                args: serde_json::json!({"size_px": 20}),
                injected_by: "debug_hud",
            }],
            generated: vec![
                GeneratedAsset {
                    name: "fox_mat_0".to_string(),
                    asset_type: "Material".to_string(),
                    generated_by: "fox".to_string(),
                },
                GeneratedAsset {
                    name: "fox_mat_glass".to_string(),
                    asset_type: "Material".to_string(),
                    generated_by: "fox".to_string(),
                },
            ],
            shadowed: vec![ShadowedAsset {
                name: "fox_mat_wood".to_string(),
                asset_type: "Material".to_string(),
                generated_by: "fox".to_string(),
            }],
            authored: vec!["cam".to_string(), "fox_mat_wood".to_string()],
        }
    }

    fn group<'a>(groups: &'a [ExpandedGroup], origin: &str) -> &'a ExpandedGroup {
        groups.iter().find(|g| g.origin == origin).expect(origin)
    }

    // Authored lines belong to the Config tab and never appear here; everything
    // else is grouped under whatever produced it.
    #[test]
    fn groups_exclude_authored_lines_and_key_by_origin() {
        let groups = groups_from(&loaded());
        let origins: Vec<&str> = groups.iter().map(|g| g.origin.as_str()).collect();
        assert_eq!(origins, vec!["debug_hud", "fox", UNATTRIBUTED]);
        assert!(
            !groups
                .iter()
                .any(|g| g.assets.iter().any(|a| a.name == "cam")),
            "an authored line is Config-tab content, not expansion output"
        );
        // The macro-expanded primitive lands in the catch-all, last.
        assert_eq!(group(&groups, UNATTRIBUTED).assets[0].name, "menu_tab_0");
        assert_eq!(groups.last().unwrap().origin, UNATTRIBUTED);
    }

    // An asset the config already overrides stays listed under its origin, so
    // copying one grays its row instead of making it disappear from the tab.
    #[test]
    fn an_overridden_asset_stays_listed_and_is_marked() {
        let groups = groups_from(&loaded());
        let fox = group(&groups, "fox");
        let wood = fox
            .assets
            .iter()
            .find(|a| a.name == "fox_mat_wood")
            .expect("the overridden asset is still listed");
        assert!(wood.in_config);
        assert!(
            wood.entry.is_none(),
            "already in the config: nothing to add"
        );
        assert_eq!(fox.in_config_count(), 1);
    }

    // A generated asset carries the entry the "+" copies, with the args the
    // expansion produced.
    #[test]
    fn a_generated_asset_carries_its_entry() {
        let groups = groups_from(&loaded());
        let mat = group(&groups, "fox")
            .assets
            .iter()
            .find(|a| a.name == "fox_mat_0")
            .unwrap();
        let entry = mat.entry.as_ref().expect("generated assets are copyable");
        assert_eq!(entry["name"], "fox_mat_0");
        assert_eq!(entry["type"], "Material");
        assert_eq!(entry["args"]["k"], 1);
    }

    // An injected default copies the args the injection recorded (what the lock
    // shows for overriding), not the compiled asset's.
    #[test]
    fn an_injected_asset_carries_the_recorded_injection_args() {
        let groups = groups_from(&loaded());
        let font = &group(&groups, "debug_hud").assets[0];
        let entry = font.entry.as_ref().expect("injected assets are copyable");
        assert_eq!(entry["args"]["size_px"], 20);
    }

    // The macro expansions emit unconditionally, so a copy would duplicate the
    // asset rather than override it: those rows are informational only.
    #[test]
    fn unattributed_assets_offer_no_entry() {
        let groups = groups_from(&loaded());
        let row = &group(&groups, UNATTRIBUTED).assets[0];
        assert!(row.entry.is_none());
        assert!(!row.in_config);
    }

    #[test]
    fn rows_collapse_by_default_and_expand_one_group() {
        let groups = groups_from(&loaded());
        let collapsed = rows(&groups, &[]);
        assert_eq!(collapsed.len(), 3, "three headers, no asset rows");
        assert!(
            collapsed
                .iter()
                .all(|r| matches!(r, ExpandedRow::Header { .. }))
        );

        // Open the "fox" group (index 1 in the sorted order).
        let open = rows(&groups, &[1]);
        let ExpandedRow::Header {
            origin,
            count,
            in_config,
            open: is_open,
            ..
        } = &open[1]
        else {
            panic!("expected the fox header at 1");
        };
        assert_eq!(
            (origin.as_str(), *count, *in_config, *is_open),
            ("fox", 3, 1, true)
        );
        // Its three assets follow, alphabetical, and only that group expands.
        let names: Vec<&str> = open
            .iter()
            .filter_map(|r| match r {
                ExpandedRow::Asset { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["fox_mat_0", "fox_mat_glass", "fox_mat_wood"]);
    }

    // Only a copyable asset that is not already in the config offers a "+".
    #[test]
    fn only_addable_assets_are_marked_addable() {
        let groups = groups_from(&loaded());
        let all = rows(&groups, &[0, 1, 2]);
        let addable = |n: &str| {
            all.iter().any(
                |r| matches!(r, ExpandedRow::Asset { name, addable, .. } if name == n && *addable),
            )
        };
        assert!(addable("fox_mat_0"), "generated and not yet copied");
        assert!(addable("hud_font"), "an injected default can be overridden");
        assert!(!addable("fox_mat_wood"), "already in the config");
        assert!(!addable("menu_tab_0"), "a macro primitive cannot be copied");
    }

    #[test]
    fn a_world_with_no_expansion_output_has_no_groups() {
        let world = LoadedWorld {
            assets: vec![asset("cam", "Camera3D")],
            injected: Vec::new(),
            generated: Vec::new(),
            shadowed: Vec::new(),
            authored: vec!["cam".to_string()],
        };
        assert!(groups_from(&world).is_empty());
        assert!(rows(&[], &[]).is_empty());
    }
}
