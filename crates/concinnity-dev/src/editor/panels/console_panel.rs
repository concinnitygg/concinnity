//! The Console panel: a terminal-style floating panel with a scrolling log
//! filling the body and a command line under it (the panel's one `TextField`,
//! which also carries the /del name autocomplete as ghost text). Log lines
//! color by severity (`theme::LOG_*`). The log model and command dispatch live
//! in `console.rs` and `hook/edit/console.rs`; this module builds the panel and
//! shows the log in it, only when the log changed.

use concinnity_engine::components::{
    LayoutChild, LayoutSize, List, TextField, UiItem, UiState, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::console::{ConsoleLog, Severity};
use super::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// The default (and minimum) width; the user can widen the panel past this.
pub(crate) const CONSOLE_W: f32 = 560.0;
const LINE_H: f32 = 22.0;
const INPUT_H: f32 = 28.0;
// Log rows at the default height; a longer log scrolls.
const LINE_POOL: usize = 14;

// The log's list and how much of the log it holds.
const LOG: &str = "editor.console.log";
const SHOWN: &str = "editor.console.shown";

// The panel's slots: the log list, then the command line.
const LIST_SLOT: usize = 0;
const INPUT_SLOT: usize = 1;

fn severity_color(severity: Severity) -> [f32; 3] {
    match severity {
        Severity::Info => theme::LOG_INFO,
        Severity::Warn => theme::LOG_WARN,
        Severity::Error => theme::LOG_ERROR,
        Severity::Command => theme::LOG_COMMAND,
    }
}

// Where the panel sits until the user drags it: bottom-left, terminal-style.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    let s = size();
    [
        8.0,
        (vp[1] - s[1] - 8.0).max(crate::editor::hud::body_top()),
    ]
}

// The panel's default (and minimum) footprint.
pub(crate) fn size() -> [f32; 2] {
    [CONSOLE_W, panel_height(LINE_POOL as f32 * LINE_H + INPUT_H)]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let log = ui.list(List {
        items: LOG.to_string(),
        row_height: LINE_H,
        ..Default::default()
    })?;
    let list = log.item?.id();
    let input = ui.text_field(TextField {
        placeholder: "/help".to_string(),
        height: INPUT_H,
        ..Default::default()
    })?;
    let field = input.item?.id();
    let input = LayoutChild {
        width: LayoutSize::Fill(1.0),
        ..input
    };
    let spec = PanelSpec {
        key: PanelKey::Console,
        title: "Console",
        title_state: None,
        size: size(),
        min_size: Some(size()),
    };
    let mut panel = ui.panel(&spec, vec![log, input])?;
    panel.slots = vec![list, field];
    Some(panel)
}

// The command line's id.
pub(crate) fn input(built: &BuiltPanel) -> Option<AssetId> {
    built.slots.get(INPUT_SLOT).copied()
}

// Whether the log list sits at its tail, so new lines keep it there.
pub(crate) fn at_tail(world: &World, built: &BuiltPanel) -> bool {
    let Some(list) = built
        .slots
        .get(LIST_SLOT)
        .and_then(|&id| world.get_by_id::<List>(id))
    else {
        return true;
    };
    let count = world.resource::<UiState>().map_or(0, |s| s.list(LOG).len());
    list.scroll >= list.max_scroll(count) - 0.5
}

// What the console shows this frame.
pub(crate) struct ConsoleView<'a> {
    // Lines pushed in all, which says when the log changed.
    pub pushed: u64,
    // Whether the log follows its tail.
    pub pinned: bool,
    // Whether the command line holds keyboard focus.
    pub focus: bool,
    // The autocomplete's inline ghost suffix for the command line.
    pub ghost: &'a str,
}

// Show the log, rebuilt only when lines arrived, and the command line's focus
// and ghost.
pub(crate) fn sync(
    view: &ConsoleView<'_>,
    log: impl FnOnce() -> Option<Vec<UiItem>>,
    world: &mut World,
    built: &BuiltPanel,
) {
    let shown = i32::try_from(view.pushed % (1 << 30)).unwrap_or(0);
    let current = world
        .resource::<UiState>()
        .and_then(|s| s.get(SHOWN).cloned());
    if current != Some(UiValue::Int(shown))
        && let Some(items) = log()
    {
        let count = items.len();
        state::with_state(world, |s| {
            s.set_list(LOG, items);
            s.set(SHOWN, UiValue::Int(shown));
        });
        if view.pinned
            && let Some(list) = built
                .slots
                .get(LIST_SLOT)
                .and_then(|&id| world.get_mut_by_id::<List>(id))
        {
            list.scroll = list.max_scroll(count);
        }
    }
    if let Some(field) = input(built).and_then(|id| world.get_mut_by_id::<TextField>(id)) {
        field.focused = view.focus;
        if field.ghost != view.ghost {
            field.ghost.clear();
            field.ghost.push_str(view.ghost);
        }
    }
}

// The log's lines as list items, colored by severity.
pub(crate) fn items(log: &ConsoleLog) -> Vec<UiItem> {
    log.lines()
        .map(|line| UiItem {
            color: Some(severity_color(line.severity)),
            ..UiItem::new(&line.text)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::panels::console::ConsoleLog;
    use crate::editor::ui::field;

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    fn view(pushed: u64, focus: bool) -> ConsoleView<'static> {
        ConsoleView {
            pushed,
            pinned: true,
            focus,
            ghost: "tern",
        }
    }

    #[test]
    fn the_log_shows_each_line_in_its_severity_color() {
        let mut log = ConsoleLog::new(16);
        log.push(Severity::Info, "hello");
        log.push(Severity::Error, "bad\nworse");
        let items = items(&log);
        assert_eq!(items.len(), 3, "a multi-line push splits");
        assert_eq!(items[1].color, Some(theme::LOG_ERROR));
    }

    #[test]
    fn sync_rebuilds_the_log_only_when_lines_arrive() {
        let mut world = World::new();
        let panel = built(&mut world);
        let mut log = ConsoleLog::new(16);
        log.push(Severity::Info, "one");
        sync(&view(1, true), || Some(items(&log)), &mut world, &panel);
        let revision = world.resource::<UiState>().unwrap().revision();
        let mut rebuilt = false;
        sync(
            &view(1, true),
            || {
                rebuilt = true;
                None
            },
            &mut world,
            &panel,
        );
        assert!(!rebuilt, "an unchanged log is not read again");
        assert_eq!(world.resource::<UiState>().unwrap().revision(), revision);
        let field = world
            .get_by_id::<TextField>(input(&panel).unwrap())
            .unwrap();
        assert!(field.focused);
        assert_eq!(field.ghost, "tern");
    }

    #[test]
    fn the_command_line_reads_and_replaces_its_text() {
        let mut world = World::new();
        let panel = built(&mut world);
        field::set_text(&mut world, input(&panel), "/help");
        assert_eq!(field::text(&world, input(&panel)), "/help");
        let field = world
            .get_by_id::<TextField>(input(&panel).unwrap())
            .unwrap();
        assert_eq!(field.caret, 5);
    }
}
