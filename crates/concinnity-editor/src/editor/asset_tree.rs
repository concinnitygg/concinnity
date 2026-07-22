// src/editor/asset_tree.rs
//
// The data half of the Assets panel: every asset of the expanded world grouped
// by origin -- the world's own lines first, then each scene import's /
// injection pass's output under the line that produced it, and the
// unattributed macro expansions last. Grouping by origin rather than by type
// keeps the tree usable: a single scene import expands to thousands of assets,
// so collapsed it is one header. Each asset carries a provenance badge and,
// when the build generates it, the entry that promoting it would append.
//
// Everything here is pure: `groups_from` builds the grouped model from a cooked
// `LoadedWorld`, and `rows` flattens it against the fold state and the live
// search filter. The panel draws the rows; the hook owns when to re-cook and
// the per-session hide / lock sets.

use concinnity_cook::world::LoadedWorld;

// The group holding the world.jsonl lines themselves.
pub(crate) const WORLD_GROUP: &str = "World";

// The group holding assets whose producing pass records no source: menu, story,
// and prefab primitives. They are listed so the tree is a complete account of
// the world, but they cannot be promoted (their passes emit unconditionally, so
// an authored copy would sit beside the generated asset rather than replace it).
pub(crate) const UNATTRIBUTED: &str = "Other expansions";

// An asset's provenance badge, as the row shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Badge {
    Authored,
    Imported,
    Injected,
}

// One listed asset: its name, type, provenance badge, and how editing it
// reaches world.jsonl.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TreeAsset {
    pub name: String,
    pub asset_type: String,
    pub badge: Badge,
    // The world.jsonl entry that editing this asset would append, promoting it
    // to an authored line that overrides what the build generates. `None` for a
    // line the world already has (edited in place instead) and for the
    // unconditional macro expansions, which cannot be overridden at all.
    pub promote: Option<serde_json::Value>,
}

// The assets one origin produced: the world's own lines (`WORLD_GROUP`), a
// scene import's or injection pass's output, or the unattributed expansions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TreeGroup {
    pub label: String,
    pub assets: Vec<TreeAsset>,
}

// Build the grouped model from a cooked world. Every live asset appears exactly
// once, under whatever produced it; an authored line that shadows a generated
// asset lists as authored under the world group, since the tree shows the live
// world rather than a copy-promotion view.
pub(crate) fn groups_from(loaded: &LoadedWorld) -> Vec<TreeGroup> {
    let mut groups: Vec<TreeGroup> = Vec::new();
    for asset in &loaded.assets {
        let prov = loaded.provenance(&asset.name);
        let (label, badge) = if prov.is_authored() {
            (WORLD_GROUP, Badge::Authored)
        } else if let Some(source) = prov.source() {
            let badge = match prov {
                concinnity_cook::world::Provenance::Injected { .. } => Badge::Injected,
                _ => Badge::Imported,
            };
            (source, badge)
        } else {
            (UNATTRIBUTED, Badge::Imported)
        };
        let entry = TreeAsset {
            name: asset.name.clone(),
            asset_type: asset.asset_type.clone(),
            badge,
            promote: prov.is_overridable().then(|| promote_entry(loaded, asset)),
        };
        match groups.iter_mut().find(|g| g.label == label) {
            Some(g) => g.assets.push(entry),
            None => groups.push(TreeGroup {
                label: label.to_string(),
                assets: vec![entry],
            }),
        }
    }
    // The world's own lines keep their authored order; expansion output sorts
    // by name (its emission order is a cook detail).
    for g in &mut groups {
        if g.label != WORLD_GROUP {
            g.assets.sort_by(|a, b| a.name.cmp(&b.name));
        }
    }
    // World first, attributed origins alphabetically, the catch-all last.
    groups.sort_by(|a, b| {
        let key = |g: &TreeGroup| {
            (
                g.label != WORLD_GROUP,
                g.label == UNATTRIBUTED,
                g.label.clone(),
            )
        };
        key(a).cmp(&key(b))
    });
    groups
}

// The world.jsonl entry promoting a generated asset would append: its name,
// type, and the args the expansion gave it. An injected asset uses the args the
// injection recorded (what the lock shows for overriding) rather than the
// compiled asset's.
fn promote_entry(
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

// Whether an asset passes the search filter: a case-insensitive substring
// match against its name or type. A blank filter matches everything.
pub(crate) fn filter_matches(filter: &str, name: &str, asset_type: &str) -> bool {
    let f = filter.trim().to_ascii_lowercase();
    if f.is_empty() {
        return true;
    }
    name.to_ascii_lowercase().contains(&f) || asset_type.to_ascii_lowercase().contains(&f)
}

// One rendered row of the tree: a group header (click to fold), or one of an
// unfolded group's assets.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TreeRow {
    Header {
        // Index into the groups list, for the fold toggle.
        group: usize,
        label: String,
        count: usize,
        open: bool,
    },
    Asset {
        // Index into the groups list and into that group's assets, so the hook
        // can reach the row's promote entry.
        group: usize,
        index: usize,
        name: String,
        asset_type: String,
        // Colours the row's type caption, and marks an authored line: only one
        // of those has a world.jsonl entry the row menu can delete.
        badge: Badge,
    },
}

// Flatten the groups into rows against the fold state and the filter. With a
// blank filter every header shows and only the groups in `open` unfold; a live
// filter unfolds every group with a match and drops the rest entirely, so a
// match inside a folded group is never invisible. Groups are folded by default
// -- a scene import expands to thousands of assets, so an open-by-default tree
// would be unusable.
pub(crate) fn rows(groups: &[TreeGroup], open: &[usize], filter: &str) -> Vec<TreeRow> {
    let filtering = !filter.trim().is_empty();
    let mut out = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let listed: Vec<(usize, &TreeAsset)> = g
            .assets
            .iter()
            .enumerate()
            .filter(|(_, a)| filter_matches(filter, &a.name, &a.asset_type))
            .collect();
        if filtering && listed.is_empty() {
            continue;
        }
        let is_open = filtering || open.contains(&gi);
        out.push(TreeRow::Header {
            group: gi,
            label: g.label.clone(),
            count: listed.len(),
            open: is_open,
        });
        if !is_open {
            continue;
        }
        for (ai, a) in listed {
            out.push(TreeRow::Asset {
                group: gi,
                index: ai,
                name: a.name.clone(),
                asset_type: a.asset_type.clone(),
                badge: a.badge,
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

    // A world with authored lines (one shadowing a generated asset), a scene
    // import's output, an injected default, and a macro-expanded primitive.
    fn loaded() -> LoadedWorld {
        LoadedWorld {
            assets: vec![
                asset("cam", "Camera3D"),
                asset("fox_mat_wood", "Material"),
                asset("fox_mat_b", "Material"),
                asset("fox_mat_a", "Material"),
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
                    name: "fox_mat_b".to_string(),
                    asset_type: "Material".to_string(),
                    generated_by: "fox".to_string(),
                },
                GeneratedAsset {
                    name: "fox_mat_a".to_string(),
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

    fn group<'a>(groups: &'a [TreeGroup], label: &str) -> &'a TreeGroup {
        groups.iter().find(|g| g.label == label).expect(label)
    }

    #[test]
    fn groups_put_world_first_origins_alphabetical_catch_all_last() {
        let groups = groups_from(&loaded());
        let labels: Vec<&str> = groups.iter().map(|g| g.label.as_str()).collect();
        assert_eq!(labels, vec![WORLD_GROUP, "debug_hud", "fox", UNATTRIBUTED]);
    }

    #[test]
    fn every_live_asset_lists_exactly_once_with_its_badge() {
        let groups = groups_from(&loaded());
        let world = group(&groups, WORLD_GROUP);
        // Authored order kept, shadowing line included as authored.
        let names: Vec<&str> = world.assets.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["cam", "fox_mat_wood"]);
        assert!(world.assets.iter().all(|a| a.badge == Badge::Authored));

        let fox = group(&groups, "fox");
        let names: Vec<&str> = fox.assets.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["fox_mat_a", "fox_mat_b"], "sorted by name");
        assert!(fox.assets.iter().all(|a| a.badge == Badge::Imported));
        assert!(
            !fox.assets.iter().any(|a| a.name == "fox_mat_wood"),
            "the shadowing line lists under World, not twice"
        );

        assert_eq!(group(&groups, "debug_hud").assets[0].badge, Badge::Injected);
        assert_eq!(
            group(&groups, UNATTRIBUTED).assets[0].badge,
            Badge::Imported
        );
    }

    // A generated asset carries the entry a promotion appends, with the args
    // the expansion produced.
    #[test]
    fn a_generated_asset_carries_its_promote_entry() {
        let groups = groups_from(&loaded());
        let mat = group(&groups, "fox")
            .assets
            .iter()
            .find(|a| a.name == "fox_mat_a")
            .unwrap();
        let entry = mat.promote.as_ref().expect("generated assets promote");
        assert_eq!(entry["name"], "fox_mat_a");
        assert_eq!(entry["type"], "Material");
        assert_eq!(entry["args"]["k"], 1);
    }

    // An injected default promotes with the args the injection recorded (what
    // the lock shows for overriding), not the compiled asset's.
    #[test]
    fn an_injected_asset_promotes_with_the_recorded_injection_args() {
        let groups = groups_from(&loaded());
        let font = &group(&groups, "debug_hud").assets[0];
        let entry = font.promote.as_ref().expect("injected assets promote");
        assert_eq!(entry["args"]["size_px"], 20);
    }

    // Only a generated asset an authored line could override carries a promote
    // entry: an authored line edits in place, and an unconditional macro
    // expansion cannot be overridden by a copy at all.
    #[test]
    fn only_overridable_generated_assets_carry_a_promote_entry() {
        let groups = groups_from(&loaded());
        let cam = &group(&groups, WORLD_GROUP).assets[0];
        assert!(cam.promote.is_none(), "an authored line edits in place");
        assert!(group(&groups, "fox").assets[0].promote.is_some());
        assert!(group(&groups, UNATTRIBUTED).assets[0].promote.is_none());
    }

    #[test]
    fn filter_matches_name_or_type_case_insensitively() {
        assert!(filter_matches("", "cam", "Camera3D"));
        assert!(filter_matches("  ", "cam", "Camera3D"));
        assert!(filter_matches("CAM", "cam", "Camera3D"));
        assert!(filter_matches("material", "fox_mat_a", "Material"));
        assert!(!filter_matches("light", "cam", "Camera3D"));
    }

    #[test]
    fn rows_fold_by_default_and_unfold_open_groups() {
        let groups = groups_from(&loaded());
        let folded = rows(&groups, &[], "");
        assert_eq!(folded.len(), 4, "four headers, no asset rows");
        assert!(
            folded
                .iter()
                .all(|r| matches!(r, TreeRow::Header { open: false, .. }))
        );

        let open = rows(&groups, &[0], "");
        assert!(matches!(
            &open[0],
            TreeRow::Header {
                group: 0,
                count: 2,
                open: true,
                ..
            }
        ));
        assert!(matches!(&open[1], TreeRow::Asset { name, .. } if name == "cam"));
        assert!(
            matches!(&open[3], TreeRow::Header { group: 1, .. }),
            "only the opened group unfolds"
        );
    }

    // An asset row carries the indices its promote entry is reached through,
    // and they survive a filter dropping earlier assets from the group.
    #[test]
    fn asset_rows_index_back_into_their_group() {
        let groups = groups_from(&loaded());
        let filtered = rows(&groups, &[], "fox_mat_b");
        let TreeRow::Asset { group, index, .. } = filtered
            .iter()
            .find(|r| matches!(r, TreeRow::Asset { .. }))
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(groups[*group].assets[*index].name, "fox_mat_b");
    }

    #[test]
    fn a_live_filter_unfolds_matches_and_drops_empty_groups() {
        let groups = groups_from(&loaded());
        let filtered = rows(&groups, &[], "fox_mat");
        // World's shadowing line and the fox group match; the rest drop.
        let headers: Vec<&str> = filtered
            .iter()
            .filter_map(|r| match r {
                TreeRow::Header { label, .. } => Some(label.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(headers, vec![WORLD_GROUP, "fox"]);
        assert!(
            filtered
                .iter()
                .filter_map(|r| match r {
                    TreeRow::Header { open, count, .. } => Some((*open, *count)),
                    _ => None,
                })
                .all(|(open, count)| open && count > 0),
            "matching groups auto-unfold and count what they list"
        );
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|r| match r {
                TreeRow::Asset { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["fox_mat_wood", "fox_mat_a", "fox_mat_b"]);

        // A type filter reaches assets by their type string.
        let fonts = rows(&groups, &[], "font");
        assert!(
            fonts
                .iter()
                .any(|r| matches!(r, TreeRow::Asset { name, .. } if name == "hud_font"))
        );
    }

    #[test]
    fn an_empty_world_has_no_groups_or_rows() {
        let world = LoadedWorld {
            assets: Vec::new(),
            injected: Vec::new(),
            generated: Vec::new(),
            shadowed: Vec::new(),
            authored: Vec::new(),
        };
        assert!(groups_from(&world).is_empty());
        assert!(rows(&[], &[], "").is_empty());
    }
}
