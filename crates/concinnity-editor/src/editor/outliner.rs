// src/editor/outliner.rs
//
// The data half of the Outliner panel: every asset of the expanded world,
// grouped by origin -- the world's own lines first, then each scene import's /
// injection pass's output under the line that produced it, and the
// unattributed macro expansions last. Grouping by origin rather than by type
// keeps the tree usable: a single scene import expands to thousands of assets,
// so collapsed it is one header. Each asset carries a provenance badge
// (authored / imported / injected), classified through the same
// `LoadedWorld::provenance` the Expanded tab uses, so the two cannot drift.
//
// Everything here is pure: `groups_from` builds the grouped model from a
// cooked `LoadedWorld`, and `rows` flattens it against the collapse state and
// the live search filter. The panel module draws the rows; the hook owns when
// to re-cook and the per-session hide / lock sets.

use super::expanded::UNATTRIBUTED;
use concinnity_cook::world::LoadedWorld;

// The group holding the world.jsonl lines themselves.
pub(crate) const WORLD_GROUP: &str = "World";

// An asset's provenance badge, as the row shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Badge {
    Authored,
    Imported,
    Injected,
}

impl Badge {
    pub(crate) fn caption(self) -> &'static str {
        match self {
            Badge::Authored => "authored",
            Badge::Imported => "imported",
            Badge::Injected => "injected",
        }
    }
}

// One listed asset: its name, type, and provenance badge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutlinerAsset {
    pub name: String,
    pub asset_type: String,
    pub badge: Badge,
}

// The assets one origin produced: the world's own lines (`WORLD_GROUP`), a
// scene import's or injection pass's output, or the unattributed expansions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutlinerGroup {
    pub label: String,
    pub assets: Vec<OutlinerAsset>,
}

// Build the grouped model from a cooked world. Every live asset appears
// exactly once, under whatever produced it; an authored line that shadows a
// generated asset lists as authored under the world group (the outliner shows
// the live world, not the copy-promotion view).
pub(crate) fn groups_from(loaded: &LoadedWorld) -> Vec<OutlinerGroup> {
    let mut groups: Vec<OutlinerGroup> = Vec::new();
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
        let entry = OutlinerAsset {
            name: asset.name.clone(),
            asset_type: asset.asset_type.clone(),
            badge,
        };
        match groups.iter_mut().find(|g| g.label == label) {
            Some(g) => g.assets.push(entry),
            None => groups.push(OutlinerGroup {
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
        let key = |g: &OutlinerGroup| {
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
pub(crate) enum OutlinerRow {
    Header {
        // Index into the groups list, for the collapse toggle.
        group: usize,
        label: String,
        count: usize,
        open: bool,
    },
    Asset {
        name: String,
        asset_type: String,
        badge: Badge,
    },
}

// Flatten the groups into rows against the collapse state and the filter.
// With a blank filter every header shows and only the groups in `open` unfold;
// a live filter unfolds every group with a match and drops the rest entirely,
// so a match inside a collapsed group is never invisible.
pub(crate) fn rows(groups: &[OutlinerGroup], open: &[usize], filter: &str) -> Vec<OutlinerRow> {
    let filtering = !filter.trim().is_empty();
    let mut out = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let listed: Vec<&OutlinerAsset> = g
            .assets
            .iter()
            .filter(|a| filter_matches(filter, &a.name, &a.asset_type))
            .collect();
        if filtering && listed.is_empty() {
            continue;
        }
        let is_open = filtering || open.contains(&gi);
        out.push(OutlinerRow::Header {
            group: gi,
            label: g.label.clone(),
            count: listed.len(),
            open: is_open,
        });
        if !is_open {
            continue;
        }
        for a in listed {
            out.push(OutlinerRow::Asset {
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
            args: serde_json::json!({}),
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
                args: serde_json::json!({}),
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

    fn group<'a>(groups: &'a [OutlinerGroup], label: &str) -> &'a OutlinerGroup {
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

    #[test]
    fn filter_matches_name_or_type_case_insensitively() {
        assert!(filter_matches("", "cam", "Camera3D"));
        assert!(filter_matches("  ", "cam", "Camera3D"));
        assert!(filter_matches("CAM", "cam", "Camera3D"));
        assert!(filter_matches("material", "fox_mat_a", "Material"));
        assert!(!filter_matches("light", "cam", "Camera3D"));
    }

    #[test]
    fn rows_collapse_by_default_and_unfold_open_groups() {
        let groups = groups_from(&loaded());
        let collapsed = rows(&groups, &[], "");
        assert_eq!(collapsed.len(), 4, "four headers, no asset rows");
        assert!(
            collapsed
                .iter()
                .all(|r| matches!(r, OutlinerRow::Header { open: false, .. }))
        );

        let open = rows(&groups, &[0], "");
        assert!(matches!(
            &open[0],
            OutlinerRow::Header {
                group: 0,
                count: 2,
                open: true,
                ..
            }
        ));
        assert!(matches!(&open[1], OutlinerRow::Asset { name, .. } if name == "cam"));
        assert!(
            matches!(&open[3], OutlinerRow::Header { group: 1, .. }),
            "only the opened group unfolds"
        );
    }

    #[test]
    fn a_live_filter_unfolds_matches_and_drops_empty_groups() {
        let groups = groups_from(&loaded());
        let filtered = rows(&groups, &[], "fox_mat");
        // World's shadowing line and the fox group match; the rest drop.
        let headers: Vec<&str> = filtered
            .iter()
            .filter_map(|r| match r {
                OutlinerRow::Header { label, .. } => Some(label.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(headers, vec![WORLD_GROUP, "fox"]);
        assert!(
            filtered
                .iter()
                .filter_map(|r| match r {
                    OutlinerRow::Header { open, count, .. } => Some((*open, *count)),
                    _ => None,
                })
                .all(|(open, count)| open && count > 0),
            "matching groups auto-unfold and count what they list"
        );
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|r| match r {
                OutlinerRow::Asset { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["fox_mat_wood", "fox_mat_a", "fox_mat_b"]);

        // A type filter reaches assets by their type string.
        let fonts = rows(&groups, &[], "font");
        assert!(
            fonts
                .iter()
                .any(|r| matches!(r, OutlinerRow::Asset { name, .. } if name == "hud_font"))
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
