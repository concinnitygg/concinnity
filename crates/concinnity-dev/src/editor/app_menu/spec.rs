//! What the menu bar holds, as data: one entry per line of it, built from the
//! editor's current panel state. Nothing here names AppKit, so the layout, the
//! checkmarks and the tag round trip are all testable without a window.

use crate::editor::panels::registry;

// The name the application menu's own items read. The menu's title is taken
// from the bundle by AppKit and cannot be set here.
const APP_NAME: &str = "Concinnity";

/// What choosing an item asks the editor to do. Items AppKit answers on its
/// own (hide, about) carry no command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MenuCommand {
    /// Show or hide the panel behind row `i` of [`registry::view_toggles`],
    /// which is the row the View panel's own checkbox `i` drives.
    PanelToggle(usize),
    /// Leave the session the way closing the window does.
    Quit,
}

// A command travels through AppKit as an NSMenuItem tag, which is a plain
// integer. Panel tags start above the fixed commands so the two can never
// collide as panels are added.
const QUIT_TAG: isize = 1;
const PANEL_TAG_BASE: isize = 0x100;

impl MenuCommand {
    pub(super) fn tag(self) -> isize {
        match self {
            MenuCommand::Quit => QUIT_TAG,
            MenuCommand::PanelToggle(i) => PANEL_TAG_BASE + i as isize,
        }
    }

    pub(super) fn from_tag(tag: isize) -> Option<Self> {
        match tag {
            QUIT_TAG => Some(MenuCommand::Quit),
            t if t >= PANEL_TAG_BASE => {
                Some(MenuCommand::PanelToggle((t - PANEL_TAG_BASE) as usize))
            }
            _ => None,
        }
    }
}

/// Which panels are open, one bit per row of [`registry::view_toggles`]. The
/// menu is rebuilt from this only when it changes, so it is worth keeping
/// small and comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PanelMarks(u32);

impl PanelMarks {
    /// Gather the open state of every toggleable panel, in registry order.
    pub(crate) fn from_open(open: impl Iterator<Item = bool>) -> Self {
        Self(
            open.take(u32::BITS as usize)
                .enumerate()
                .filter(|(_, open)| *open)
                .map(|(i, _)| 1 << i)
                .sum(),
        )
    }

    fn contains(self, i: usize) -> bool {
        i < u32::BITS as usize && self.0 & (1 << i) != 0
    }
}

/// One line of a menu: a divider, or an item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Entry {
    Separator,
    Item(ItemSpec),
}

impl Entry {
    pub(super) fn item(&self) -> Option<&ItemSpec> {
        match self {
            Entry::Separator => None,
            Entry::Item(item) => Some(item),
        }
    }
}

/// What an item does when chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ItemKind {
    /// Sent down the responder chain under this selector, which the
    /// application object itself answers.
    Standard(&'static str),
    /// Routed back to the editor on the next frame.
    Command(MenuCommand),
}

/// A key equivalent. Command is implied; `option` adds the second modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KeyEquivalent {
    pub key: &'static str,
    pub option: bool,
}

/// One item of one menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ItemSpec {
    pub title: String,
    pub kind: ItemKind,
    /// Drawn with a checkmark.
    pub checked: bool,
    pub key: Option<KeyEquivalent>,
}

fn standard(title: String, selector: &'static str, key: Option<KeyEquivalent>) -> Entry {
    Entry::Item(ItemSpec {
        title,
        kind: ItemKind::Standard(selector),
        checked: false,
        key,
    })
}

fn command(title: String, command: MenuCommand, checked: bool) -> Entry {
    Entry::Item(ItemSpec {
        title,
        kind: ItemKind::Command(command),
        checked,
        key: None,
    })
}

/// The application menu: the leftmost one, which AppKit reserves for the app
/// itself. Fixed, so it is built once and never refreshed.
pub(super) fn app_entries() -> Vec<Entry> {
    vec![
        standard(
            format!("About {APP_NAME}"),
            "orderFrontStandardAboutPanel:",
            None,
        ),
        Entry::Separator,
        standard(
            format!("Hide {APP_NAME}"),
            "hide:",
            Some(KeyEquivalent {
                key: "h",
                option: false,
            }),
        ),
        standard(
            "Hide Others".to_string(),
            "hideOtherApplications:",
            Some(KeyEquivalent {
                key: "h",
                option: true,
            }),
        ),
        standard("Show All".to_string(), "unhideAllApplications:", None),
        Entry::Separator,
        Entry::Item(ItemSpec {
            title: format!("Quit {APP_NAME}"),
            // Routed through the editor rather than AppKit's `terminate:`, so
            // the session leaves by the same path as closing the window.
            kind: ItemKind::Command(MenuCommand::Quit),
            checked: false,
            key: Some(KeyEquivalent {
                key: "q",
                option: false,
            }),
        }),
    ]
}

/// The View menu: one checkbox per panel the View panel lists, in the same
/// registry order, so the two ways of opening a panel show the same thing.
pub(super) fn view_entries(marks: PanelMarks) -> Vec<Entry> {
    registry::view_toggles()
        .enumerate()
        .map(|(i, panel)| {
            command(
                panel.view_row().unwrap_or_default().to_string(),
                MenuCommand::PanelToggle(i),
                marks.contains(i),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commands(entries: &[Entry]) -> Vec<MenuCommand> {
        entries
            .iter()
            .filter_map(|e| match e.item()?.kind {
                ItemKind::Command(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    fn all_open(open: bool) -> PanelMarks {
        PanelMarks::from_open(std::iter::repeat_n(open, registry::view_toggle_count()))
    }

    // The View menu offers the View panel's rows, under the same captions and
    // in the same order. This is what keeps the two ways of opening a panel
    // from drifting as panels are added.
    #[test]
    fn view_entries_are_the_view_panels_rows() {
        let entries = view_entries(PanelMarks::default());
        let titles: Vec<_> = entries
            .iter()
            .filter_map(Entry::item)
            .map(|i| i.title.as_str())
            .collect();
        let expected: Vec<_> = registry::view_toggles()
            .map(|p| p.view_row().expect("a toggle row"))
            .collect();
        assert_eq!(titles, expected);
        assert_eq!(
            commands(&entries),
            (0..expected.len())
                .map(MenuCommand::PanelToggle)
                .collect::<Vec<_>>()
        );
    }

    // Every toggleable panel has to fit the mark set, which is what a bit per
    // panel buys: adding panels past the width would silently lose their
    // checkmarks.
    #[test]
    fn every_toggleable_panel_fits_the_marks() {
        assert!(registry::view_toggle_count() <= u32::BITS as usize);
        assert_ne!(all_open(true), all_open(false));
    }

    // A row's mark follows its own panel, in both directions and without
    // disturbing its neighbours.
    #[test]
    fn a_row_is_marked_from_its_own_panel() {
        let count = registry::view_toggle_count();
        for open in 0..count {
            let marks = PanelMarks::from_open((0..count).map(|i| i == open));
            let marked: Vec<_> = view_entries(marks)
                .into_iter()
                .filter_map(|e| e.item().filter(|i| i.checked).map(|i| i.title.clone()))
                .collect();
            let expected = registry::view_toggles()
                .nth(open)
                .and_then(|p| p.view_row())
                .expect("a toggle row");
            assert_eq!(marked, vec![expected.to_string()]);
        }
    }

    // Nothing is marked when nothing is open, and everything is when it all
    // is: the two ends the per-row test brackets.
    #[test]
    fn the_marks_span_none_to_all() {
        let none = view_entries(all_open(false));
        let all = view_entries(all_open(true));
        assert!(none.iter().filter_map(Entry::item).all(|i| !i.checked));
        assert!(all.iter().filter_map(Entry::item).all(|i| i.checked));
    }

    // Tags are how a command survives the trip through AppKit, so every one
    // the menu can hold must come back as itself.
    #[test]
    fn every_command_survives_its_tag() {
        let mut all = vec![MenuCommand::Quit];
        all.extend((0..registry::view_toggle_count()).map(MenuCommand::PanelToggle));
        for command in all {
            assert_eq!(MenuCommand::from_tag(command.tag()), Some(command));
        }
    }

    // An item AppKit answers itself carries tag 0, which must not read as a
    // command.
    #[test]
    fn a_tagless_item_is_no_command() {
        assert_eq!(MenuCommand::from_tag(0), None);
    }

    // Quit is the app menu's last line and the only one the editor answers;
    // everything above it is AppKit's own.
    #[test]
    fn quit_is_the_only_app_entry_the_editor_answers() {
        let entries = app_entries();
        assert_eq!(commands(&entries), vec![MenuCommand::Quit]);
        let last = entries.last().expect("an entry").item().expect("an item");
        assert_eq!(last.kind, ItemKind::Command(MenuCommand::Quit));
    }
}
