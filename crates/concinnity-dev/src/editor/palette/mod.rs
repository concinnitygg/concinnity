// src/editor/palette/mod.rs
//
// The command palette's data model: every actionable thing the editor can
// reach -- panels, world assets, console commands, display options -- as one
// ranked list. Pure data and ranking only; the overlay's geometry lives in
// `editor/palette_panel.rs` and the drive in `hook/palette_edit.rs`.

pub(crate) mod providers;

use super::filter;
use super::registry::PanelKey;
use super::view_menu;

// What committing a palette row does. Every arm routes through an existing
// editor path (panel toggles, the selection, console dispatch, the Display
// menu's state); the palette owns none of the behavior itself.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PaletteAction {
    OpenPanel(PanelKey),
    // Select the named asset, reveal it in the tree, and frame the camera on it.
    SelectEntity(String),
    // Open the named asset's editing surface (the Behavior panel for a
    // behavior, the edit form otherwise).
    OpenAsset(String),
    // A command that takes arguments: seed the input with "/name " and keep
    // typing.
    CommandMode(&'static str),
    // A complete console line, dispatched as if typed into the Console panel.
    RunCommand(String),
    // One Display-menu row, applied exactly as a menu click would.
    SetOption(view_menu::MenuRow),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Category {
    Panel,
    Entity,
    Asset,
    Command,
    Option,
}

impl Category {
    // The short tag drawn on a result row.
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Category::Panel => "panel",
            Category::Entity => "entity",
            Category::Asset => "asset",
            Category::Command => "command",
            Category::Option => "option",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PaletteItem {
    pub label: String,
    pub hint: String,
    pub category: Category,
    pub action: PaletteAction,
}

// How many committed rows the empty-query list remembers, newest first.
pub(crate) const RECENT_CAP: usize = 6;

// The items `query` keeps, best first, as indices into `items`.
//
// A query starting with '/' narrows to the commands alone, ranked by the name
// after the slash with any arguments ignored, so "/add Sprite" still ranks
// /add first. An empty query is the launch list instead: recently committed
// rows, then the panels and commands (both short lists); assets appear once
// something is typed, so a large world never floods the default view.
pub(crate) fn matches(items: &[PaletteItem], recent: &[String], query: &str) -> Vec<usize> {
    let typed = query.trim();
    if let Some(rest) = typed.strip_prefix('/') {
        let name = rest.split_whitespace().next().unwrap_or("");
        let commands: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.category == Category::Command)
            .map(|(i, _)| i)
            .collect();
        let ranked = filter::ranked(
            commands.iter().map(|&i| {
                (
                    items[i].label.trim_start_matches('/'),
                    items[i].hint.as_str(),
                )
            }),
            name,
        );
        return ranked.into_iter().map(|r| commands[r]).collect();
    }
    if typed.is_empty() {
        return launch_list(items, recent);
    }
    filter::ranked(
        items.iter().map(|it| (it.label.as_str(), it.hint.as_str())),
        typed,
    )
}

fn launch_list(items: &[PaletteItem], recent: &[String]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for label in recent {
        if let Some(i) = items.iter().position(|it| &it.label == label)
            && !out.contains(&i)
        {
            out.push(i);
        }
    }
    for (i, it) in items.iter().enumerate() {
        if matches!(it.category, Category::Panel | Category::Command) && !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(label: &str, hint: &str, category: Category) -> PaletteItem {
        PaletteItem {
            label: label.to_string(),
            hint: hint.to_string(),
            category,
            action: PaletteAction::RunCommand("/help".to_string()),
        }
    }

    fn fixture() -> Vec<PaletteItem> {
        vec![
            item("Console", "open panel", Category::Panel),
            item("stone_wall", "Mesh in World", Category::Entity),
            item("wall_greeter", "Behavior in World", Category::Asset),
            item("/cook", "compile the world", Category::Command),
            item("/add", "add an asset", Category::Command),
            item("Show: Fog", "display option", Category::Option),
        ]
    }

    fn labels(items: &[PaletteItem], hits: &[usize]) -> Vec<String> {
        hits.iter().map(|&i| items[i].label.clone()).collect()
    }

    #[test]
    fn ranking_crosses_categories_with_prefix_first() {
        let items = fixture();
        let hits = matches(&items, &[], "wall");
        assert_eq!(
            labels(&items, &hits),
            ["wall_greeter", "stone_wall"],
            "prefix beats contains, regardless of category"
        );
    }

    #[test]
    fn empty_query_lists_panels_and_commands_only() {
        let items = fixture();
        let hits = matches(&items, &[], "");
        assert_eq!(labels(&items, &hits), ["Console", "/cook", "/add"]);
        assert_eq!(matches(&items, &[], "   "), hits);
    }

    #[test]
    fn recent_commits_lead_the_empty_query_list() {
        let items = fixture();
        let recent = vec!["stone_wall".to_string(), "/add".to_string()];
        let hits = matches(&items, &recent, "");
        assert_eq!(
            labels(&items, &hits),
            ["stone_wall", "/add", "Console", "/cook"],
            "recents first (assets included), then panels and commands, no repeats"
        );
    }

    #[test]
    fn a_stale_recent_label_is_skipped() {
        let items = fixture();
        let recent = vec!["deleted_asset".to_string()];
        let hits = matches(&items, &recent, "");
        assert_eq!(labels(&items, &hits), ["Console", "/cook", "/add"]);
    }

    #[test]
    fn slash_narrows_to_commands() {
        let items = fixture();
        let hits = matches(&items, &[], "/");
        assert_eq!(labels(&items, &hits), ["/cook", "/add"]);
        let hits = matches(&items, &[], "/ad");
        assert_eq!(labels(&items, &hits), ["/add"]);
    }

    #[test]
    fn command_mode_ranks_by_name_ignoring_arguments() {
        let items = fixture();
        let hits = matches(&items, &[], "/add Sprite crate");
        assert_eq!(labels(&items, &hits), ["/add"]);
    }

    #[test]
    fn no_match_keeps_nothing() {
        let items = fixture();
        assert!(matches(&items, &[], "zzz").is_empty());
        assert!(matches(&items, &[], "/zzz").is_empty());
    }
}
