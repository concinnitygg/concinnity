//! The editor's "grouped asset list": the world's assets shown under a type
//! sub-header with the names indented and alphabetized. Both the Assets browse
//! panel (`assets_panel.rs`) and the Template detail panel (`template_panel.rs`) render
//! this identical list, so the row model, the grouping, the row geometry / style,
//! and the per-row + scrollbar draw all live here once. The two panels differ only
//! in their extra chrome (the Assets panel adds hover / selection tints, a
//! triple-dot menu, and delete); that stays in `assets_panel.rs`, layered over this base.

use concinnity_cook::authoring::world::entry_type_name;

// Visible rows in the body before it scrolls (shared by the grouped list and the
// Assets panel's combo option list, which reuses this window height).
pub(crate) const MAX_ROWS: usize = 12;

// Row geometry, in window pixels.
pub(crate) const ROW_H: f32 = 28.0;

// One rendered browse-list row: a type sub-header, or an indented asset name that
// carries the index of its entry (used by the Assets panel's Delete / edit menu;
// unused by the read-only Template list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ListRow {
    pub is_header: bool,
    pub text: String,
    // The `entries` index for a name row; `None` for a header.
    pub entry: Option<usize>,
}

// The distinct asset types present in `entries`, alphabetical.
fn distinct_types(entries: &[serde_json::Value]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for e in entries {
        if let Some(ty) = entry_type_name(e)
            && !out.iter().any(|s| s == ty)
        {
            out.push(ty.to_string());
        }
    }
    out.sort();
    out
}

// Group `entries` into browse rows: a sub-header per type (types alphabetical),
// then an indented row per asset name (names alphabetical within a type) carrying
// its `entries` index. `type_filter` keeps only the matching type's group.
pub(crate) fn grouped_rows(
    entries: &[serde_json::Value],
    type_filter: Option<&str>,
) -> Vec<ListRow> {
    let mut rows = Vec::new();
    let handles = concinnity_cook::authoring::world::entry_handles(entries);
    for ty in distinct_types(entries) {
        if let Some(f) = type_filter
            && ty != f
        {
            continue;
        }
        let mut named: Vec<(usize, String)> = entries
            .iter()
            .zip(&handles)
            .enumerate()
            .filter_map(|(i, (e, name))| {
                let name = name.as_deref()?;
                (entry_type_name(e) == Some(ty.as_str())).then(|| (i, name.to_string()))
            })
            .collect();
        if named.is_empty() {
            continue;
        }
        named.sort_by(|a, b| a.1.cmp(&b.1));
        rows.push(ListRow {
            is_header: true,
            text: ty,
            entry: None,
        });
        for (i, name) in named {
            rows.push(ListRow {
                is_header: false,
                text: name,
                entry: Some(i),
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, ty: &str) -> serde_json::Value {
        serde_json::json!({"type": ty, "args": {"$id": name}})
    }

    // Types sort alphabetically; names sort alphabetically within their type; name
    // rows carry their original `entries` index (not the sorted position) so the
    // Assets panel's edit / delete still target the right entry.
    #[test]
    fn grouped_rows_sort_types_and_names_and_keep_indices() {
        let entries = vec![
            entry("zeta", "PointLight"),
            entry("beta", "Decal"),
            entry("alpha", "PointLight"),
        ];
        let rows = grouped_rows(&entries, None);
        assert!(rows[0].is_header && rows[0].text == "Decal");
        assert_eq!((rows[1].text.as_str(), rows[1].entry), ("beta", Some(1)));
        assert!(rows[2].is_header && rows[2].text == "PointLight");
        // alpha before zeta (alphabetical), each keeping its original index.
        assert_eq!((rows[3].text.as_str(), rows[3].entry), ("alpha", Some(2)));
        assert_eq!((rows[4].text.as_str(), rows[4].entry), ("zeta", Some(0)));
    }

    #[test]
    fn grouped_rows_filter_keeps_one_group() {
        let entries = vec![
            entry("a", "PointLight"),
            entry("b", "Decal"),
            entry("c", "PointLight"),
        ];
        let rows = grouped_rows(&entries, Some("PointLight"));
        assert_eq!(rows.len(), 3, "one header + two names");
        assert!(rows[0].is_header && rows[0].text == "PointLight");
    }
}
