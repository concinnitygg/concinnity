// src/editor/palette/providers.rs
//
// The palette's result sources. Each provider is a pure enumeration into
// `PaletteItem`s: actions carry names and keys only, and executing them is the
// drive's job (`hook/palette_edit.rs`), so a provider needs no world to be
// exercised.

use super::{Category, PaletteAction, PaletteItem};
use crate::editor::asset_tree::TreeGroup;
use crate::editor::console;
use crate::editor::registry;
use crate::editor::view_menu;

// One row per view-toggleable panel (the ones that open on demand).
pub(crate) fn panel_items() -> Vec<PaletteItem> {
    registry::view_toggles()
        .map(|p| PaletteItem {
            label: p.view_row().unwrap_or("").to_string(),
            hint: "open panel".to_string(),
            category: Category::Panel,
            action: PaletteAction::OpenPanel(p.key()),
        })
        .collect()
}

// One row per asset of the cooked tree. A behavior opens its own panel;
// everything else is a world entity the palette selects and frames. The origin
// group rides in the hint, so typing a scene's name narrows to its assets.
pub(crate) fn asset_items(groups: &[TreeGroup]) -> Vec<PaletteItem> {
    groups
        .iter()
        .flat_map(|g| {
            g.assets.iter().map(|a| {
                let behavior = a.asset_type == "Behavior";
                PaletteItem {
                    label: a.name.clone(),
                    hint: format!("{} in {}", a.asset_type, g.label),
                    category: if behavior {
                        Category::Asset
                    } else {
                        Category::Entity
                    },
                    action: if behavior {
                        PaletteAction::OpenAsset(a.name.clone())
                    } else {
                        PaletteAction::SelectEntity(a.name.clone())
                    },
                }
            })
        })
        .collect()
}

// One row per console command, dispatched through the console's own registry:
// a command with arguments seeds command mode, an argument-less one runs
// outright.
pub(crate) fn command_items() -> Vec<PaletteItem> {
    console::COMMANDS
        .iter()
        .map(|spec| {
            let takes_args = spec.usage.contains(['<', '[']);
            PaletteItem {
                label: format!("/{}", spec.name),
                hint: spec.blurb.to_string(),
                category: Category::Command,
                action: if takes_args {
                    PaletteAction::CommandMode(spec.name)
                } else {
                    PaletteAction::RunCommand(format!("/{}", spec.name))
                },
            }
        })
        .collect()
}

// One row per actionable Display-menu entry (headings carry nothing to do).
pub(crate) fn option_items() -> Vec<PaletteItem> {
    view_menu::rows()
        .into_iter()
        .filter_map(|row| {
            let label = match row {
                view_menu::MenuRow::Mode(m) => format!("View mode: {}", m.label()),
                view_menu::MenuRow::Heading(_) => return None,
                view_menu::MenuRow::Flag(_, label) => format!("Show: {label}"),
                view_menu::MenuRow::Billboards => "Show: Billboards".to_string(),
                view_menu::MenuRow::Extent(_, label) => format!("Extents: {label}"),
            };
            Some(PaletteItem {
                label,
                hint: "display option".to_string(),
                category: Category::Option,
                action: PaletteAction::SetOption(row),
            })
        })
        .collect()
}

// Every provider's items in one list: panels, then assets, then commands,
// then options -- the declaration order equal-rank matches keep.
pub(crate) fn all_items(groups: &[TreeGroup]) -> Vec<PaletteItem> {
    let mut items = panel_items();
    items.extend(asset_items(groups));
    items.extend(command_items());
    items.extend(option_items());
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::asset_tree::{Badge, TreeAsset};

    fn groups() -> Vec<TreeGroup> {
        vec![TreeGroup {
            label: "World".to_string(),
            assets: vec![
                TreeAsset {
                    name: "crate_a".to_string(),
                    asset_type: "Sprite".to_string(),
                    badge: Badge::Authored,
                    promote: None,
                },
                TreeAsset {
                    name: "greeter".to_string(),
                    asset_type: "Behavior".to_string(),
                    badge: Badge::Authored,
                    promote: None,
                },
            ],
        }]
    }

    #[test]
    fn panel_items_cover_every_view_toggle() {
        let items = panel_items();
        assert_eq!(items.len(), registry::view_toggle_count());
        for it in &items {
            assert_eq!(it.category, Category::Panel);
            assert!(matches!(it.action, PaletteAction::OpenPanel(_)));
            assert!(!it.label.is_empty());
        }
    }

    #[test]
    fn command_items_cover_the_console_registry() {
        let items = command_items();
        assert_eq!(items.len(), console::COMMANDS.len());
        let cook = items.iter().find(|it| it.label == "/cook").unwrap();
        assert_eq!(
            cook.action,
            PaletteAction::RunCommand("/cook".to_string()),
            "an argument-less command runs outright"
        );
        let add = items.iter().find(|it| it.label == "/add").unwrap();
        assert_eq!(
            add.action,
            PaletteAction::CommandMode("add"),
            "a command with arguments seeds command mode"
        );
    }

    #[test]
    fn asset_items_route_behaviors_to_their_panel() {
        let items = asset_items(&groups());
        assert_eq!(items.len(), 2);
        let entity = &items[0];
        assert_eq!(entity.category, Category::Entity);
        assert_eq!(
            entity.action,
            PaletteAction::SelectEntity("crate_a".to_string())
        );
        assert_eq!(entity.hint, "Sprite in World");
        let behavior = &items[1];
        assert_eq!(behavior.category, Category::Asset);
        assert_eq!(
            behavior.action,
            PaletteAction::OpenAsset("greeter".to_string())
        );
    }

    #[test]
    fn option_items_cover_the_display_menu_without_headings() {
        let items = option_items();
        let actionable = view_menu::rows()
            .into_iter()
            .filter(|r| !matches!(r, view_menu::MenuRow::Heading(_)))
            .count();
        assert_eq!(items.len(), actionable);
        for it in &items {
            assert_eq!(it.category, Category::Option);
            assert!(matches!(it.action, PaletteAction::SetOption(_)));
        }
    }

    #[test]
    fn all_items_concatenates_in_provider_order() {
        let items = all_items(&groups());
        let first_asset = items
            .iter()
            .position(|it| it.category != Category::Panel)
            .unwrap();
        assert_eq!(first_asset, panel_items().len());
        assert!(items.iter().any(|it| it.category == Category::Command));
        assert!(items.iter().any(|it| it.category == Category::Option));
    }
}
