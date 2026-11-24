//! The Shader source panel: one of a Shader's `.hlsl` files in a code text area
//! (see `source.rs` and `diagnostics.rs` for the data,
//! `hook/edit/shader_source.rs` for the actions), its editor lent by the hook
//! each frame. The title names the Shader and the file's stage, the header
//! carries the path, an unsaved-edits mark, the reference toggle and Save, a
//! status line reports the last compile and jumps to it when pressed, and the
//! gutter marks the lines its diagnostics name. The reference column sits at
//! the right, beside the text.

use concinnity_engine::components::{
    Binding, Button, LayoutChild, LayoutDirection, LayoutSize, LineMark, List, TextAlign, TextArea,
    TextLanguage, UiEvent, UiItem, UiState, UiValue, WidgetStyle,
};
use concinnity_engine::ecs::World;
use concinnity_engine::presentation::text_edit::TextEditor;
use concinnity_engine::presentation::text_edit::highlight::Token;

use super::super::registry::PanelKey;
use super::diagnostics::Tone;
use super::list_panel::tone_color;
use super::reference::RefRow;
use crate::editor::text_area::markers::{GutterMarker, Severity};
use crate::editor::theme;
use crate::editor::ui::build::{emit, fill};
use crate::editor::ui::{BuiltPanel, PanelSpec, ROW_H, UiBuilder, panel_height, state};

// The default (and minimum) width; the user can widen the panel past this.
const SOURCE_W: f32 = 660.0;
// The reference column's width.
pub(crate) const COLUMN_W: f32 = 250.0;
const HEADER_H: f32 = 32.0;
const BTN_W: f32 = 70.0;
const REF_BTN_W: f32 = 90.0;
const DIRTY_W: f32 = 70.0;
// The text area's height at the default panel size.
const AREA_H: f32 = 520.0;
const REF_ROW_H: f32 = 22.0;

const PATH_COLOR: [f32; 3] = [0.62, 0.66, 0.76];
const HEADING_COLOR: [f32; 3] = [0.58, 0.66, 0.80];

// The state the panel shows, and the events its controls send.
const TITLE: &str = "editor.source.title";
const PATH: &str = "editor.source.path";
const DIRTY: &str = "editor.source.dirty";
const STATUS: &str = "editor.source.status";
const REF_OPEN: &str = "editor.source.ref_open";
const REF_ROWS: &str = "editor.source.ref_rows";
const REF_SHOWN: &str = "editor.source.ref_shown";
const SAVE: &str = "editor.source.save";
const JUMP: &str = "editor.source.jump";
const TOGGLE_REF: &str = "editor.source.toggle_ref";
const REF_PICK: &str = "editor.source.ref_pick";

// The panel's slots.
const AREA_SLOT: usize = 0;
const STATUS_SLOT: usize = 1;
const REF_SLOT: usize = 2;

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceAction {
    // Write the file; the hot-reload watcher recompiles it.
    Save,
    // The status line: jump to the error it reports.
    Status,
    // Show or hide the reference column.
    ToggleReference,
    // Row `i` of the reference column.
    Reference(usize),
}

// The action a source-panel event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<SourceAction> {
    Some(match event.name.as_str() {
        SAVE => SourceAction::Save,
        JUMP => SourceAction::Status,
        TOGGLE_REF => SourceAction::ToggleReference,
        REF_PICK => match event.value {
            Some(UiValue::Int(i)) => SourceAction::Reference(usize::try_from(i).ok()?),
            _ => return None,
        },
        _ => return None,
    })
}

// Where the panel sits until the user drags it: left of center, below the
// top bar, so it opens beside the Shaders list rather than over it.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [
        (vw * 0.5 - SOURCE_W).max(8.0),
        crate::editor::hud::body_top() + 8.0,
    ]
}

// The room the reference column takes from the text, when shown.
fn reference_w(reference: bool) -> f32 {
    if reference { COLUMN_W + 4.0 } else { 0.0 }
}

pub(crate) fn size(reference: bool) -> [f32; 2] {
    [
        SOURCE_W + reference_w(reference),
        panel_height(HEADER_H + ROW_H + AREA_H),
    ]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let path = ui.bound_label(PATH, PATH_COLOR, TextAlign::Left)?;
    let dirty = ui.bound_label(DIRTY, theme::LOG_WARN, TextAlign::Right)?;
    let dirty = LayoutChild {
        width: LayoutSize::Fixed(DIRTY_W),
        ..dirty
    };
    let reference = ui.button_with(
        Button {
            label: "Reference".to_string(),
            active: Some(Binding::State(REF_OPEN.to_string())),
            action: Some(emit(TOGGLE_REF, None)),
            ..Default::default()
        },
        [REF_BTN_W, HEADER_H - 8.0],
    )?;
    let save = ui.button("Save", SAVE, None, [BTN_W, HEADER_H - 8.0])?;
    let header = ui.row(vec![path, dirty, reference, save], HEADER_H)?;
    let status = ui.button_with(
        Button {
            text_binding: Some(Binding::State(STATUS.to_string())),
            action: Some(emit(JUMP, None)),
            style: WidgetStyle {
                background: [0.0; 4],
                ..ui.style()
            },
            ..Default::default()
        },
        [SOURCE_W - 16.0, ROW_H],
    )?;
    let status_id = status.item?.id();
    let area = ui.text_area(TextArea {
        language: TextLanguage::Hlsl,
        on_save: SAVE.to_string(),
        ..Default::default()
    })?;
    let area_id = area.item?.id();
    let column = ui.list(List {
        items: REF_ROWS.to_string(),
        row_height: REF_ROW_H,
        indent: 12.0,
        on_select: REF_PICK.to_string(),
        on_activate: REF_PICK.to_string(),
        on_expand: REF_PICK.to_string(),
        ..Default::default()
    })?;
    let column_id = column.item?.id();
    let column = LayoutChild {
        width: LayoutSize::Fixed(COLUMN_W),
        ..column
    };
    let body = ui.stack(LayoutDirection::Row, vec![area, column], 4.0, 0.0)?;
    let spec = PanelSpec {
        key: PanelKey::ShaderSource,
        title: "Shader source",
        title_state: Some(TITLE),
        size: size(true),
        min_size: Some(size(false)),
    };
    let mut panel = ui.panel(&spec, vec![header, status, fill(body)])?;
    panel.slots = vec![area_id, status_id, column_id];
    Some(panel)
}

// What the panel shows this frame.
pub(crate) struct SourceView<'a> {
    // "water fragment", in its two words.
    pub title: [&'a str; 2],
    pub path: &'a str,
    pub dirty: bool,
    // Whether the text area holds the keyboard.
    pub focus: bool,
    // The status line: the hovered reference name's signature and summary,
    // else the file's own status and its tone.
    pub hover: Option<[&'a str; 2]>,
    pub status: Option<(&'a str, Tone)>,
    pub markers: &'a [GutterMarker],
    // While the reference column is shown, a key that changes whenever its
    // rows do.
    pub reference: Option<i32>,
}

fn mark_color(severity: Severity) -> [f32; 3] {
    match severity {
        Severity::Error => theme::LOG_ERROR,
        Severity::Warning => theme::LOG_WARN,
    }
}

// A reference row as a list item: a group's heading, folding its names under
// it, or a name in the code's color.
fn ref_item(row: &RefRow<'_>) -> UiItem {
    match row {
        RefRow::Heading { folded, .. } => UiItem {
            expandable: true,
            expanded: !folded,
            color: Some(HEADING_COLOR),
            ..UiItem::new(&row.text())
        },
        RefRow::Name(_) => UiItem {
            depth: 1,
            color: Some(Token::Engine.color()),
            ..UiItem::new(&row.text())
        },
    }
}

// Errors first, so a line marked twice shows its most severe.
fn marks(markers: &[GutterMarker]) -> impl Iterator<Item = LineMark> + '_ {
    [Severity::Error, Severity::Warning]
        .into_iter()
        .flat_map(move |severity| markers.iter().filter(move |m| m.severity == severity))
        .map(|m| LineMark {
            line: m.line,
            color: mark_color(m.severity),
        })
}

// Show the file's title, path, status, and marks, and the reference column's
// rows, read from `rows` only when their key changed.
pub(crate) fn sync<'r>(
    view: &SourceView<'_>,
    rows: impl FnOnce() -> Vec<RefRow<'r>>,
    world: &mut World,
    built: &BuiltPanel,
) {
    let [a, b] = view.title;
    state::set_text(world, TITLE, format_args!("{a} {b}"));
    state::set_text(world, PATH, format_args!("{}", view.path));
    let dirty = if view.dirty { "unsaved" } else { "" };
    state::set_text(world, DIRTY, format_args!("{dirty}"));
    let tone = match (view.hover, view.status) {
        (Some([signature, summary]), _) => {
            state::set_text(world, STATUS, format_args!("{signature}: {summary}"));
            Some(Tone::Info)
        }
        (None, Some((text, tone))) => {
            state::set_text(world, STATUS, format_args!("{text}"));
            Some(tone)
        }
        (None, None) => {
            state::set_text(world, STATUS, format_args!(""));
            None
        }
    };
    let shown = world
        .resource::<UiState>()
        .and_then(|s| s.get(REF_SHOWN).cloned());
    let refill = view
        .reference
        .filter(|&key| shown != Some(UiValue::Int(key)));
    let items = refill.map(|_| rows().iter().map(ref_item).collect::<Vec<_>>());
    state::with_state(world, |s| {
        s.set(REF_OPEN, UiValue::Bool(view.reference.is_some()));
        if let (Some(key), Some(items)) = (refill, items) {
            s.set_list(REF_ROWS, items);
            s.set(REF_SHOWN, UiValue::Int(key));
        }
    });
    let slot = |i: usize| built.slots.get(i).copied();
    if let Some(area) = slot(AREA_SLOT).and_then(|id| world.get_mut_by_id::<TextArea>(id)) {
        area.focused = view.focus;
        if !area.marks.iter().copied().eq(marks(view.markers)) {
            area.marks.clear();
            area.marks.extend(marks(view.markers));
        }
    }
    if let Some(b) = slot(STATUS_SLOT).and_then(|id| world.get_mut_by_id::<Button>(id)) {
        b.style.text = tone.map_or(theme::LABEL_DIM, tone_color);
    }
    if let Some(list) = slot(REF_SLOT).and_then(|id| world.get_mut_by_id::<List>(id)) {
        list.visible = view.reference.is_some();
    }
}

// The reference row under the pointer, if any.
pub(crate) fn reference_hovered(world: &World, built: &BuiltPanel) -> Option<usize> {
    built
        .slots
        .get(REF_SLOT)
        .and_then(|&id| world.get_by_id::<List>(id))
        .filter(|l| l.visible)
        .and_then(|l| l.hovered)
}

// Whether the text area holds the keyboard, as the widget last showed it.
pub(crate) fn area_focused(world: &World, built: &BuiltPanel) -> bool {
    built
        .slots
        .get(AREA_SLOT)
        .and_then(|&id| world.get_by_id::<TextArea>(id))
        .is_some_and(|a| a.focused)
}

// Swap `editor` with the text area's: lending it to the area, or taking it
// back.
pub(crate) fn swap_editor(editor: &mut TextEditor, world: &mut World, built: &BuiltPanel) -> bool {
    let Some(area) = built
        .slots
        .get(AREA_SLOT)
        .and_then(|&id| world.get_mut_by_id::<TextArea>(id))
    else {
        return false;
    };
    std::mem::swap(editor, &mut *area.editor);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::panels::shader::reference::Reference;
    use concinnity_engine::ecs::asset_id::AssetId;
    use concinnity_engine::render::shader::programs::vocabulary::ENTRIES;

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    fn view(reference: Option<i32>) -> SourceView<'static> {
        SourceView {
            title: ["water", "fragment"],
            path: "water.hlsl",
            dirty: true,
            focus: true,
            hover: None,
            status: Some(("2 errors", Tone::Error)),
            markers: &[],
            reference,
        }
    }

    #[test]
    fn the_reference_column_lists_its_groups_and_names_once_per_key() {
        let mut world = World::new();
        let panel = built(&mut world);
        let reference = Reference::default();
        sync(
            &view(Some(1)),
            || reference.rows(ENTRIES),
            &mut world,
            &panel,
        );
        let s = world.resource::<UiState>().unwrap();
        let items = s.list(REF_ROWS);
        assert_eq!(items.len(), reference.rows(ENTRIES).len());
        assert!(items[0].expandable && items[0].expanded);
        assert_eq!(s.get(DIRTY), Some(&UiValue::Text("unsaved".into())));
        assert_eq!(s.get(TITLE), Some(&UiValue::Text("water fragment".into())));
        let status = world.get_by_id::<Button>(panel.slots[STATUS_SLOT]).unwrap();
        assert_eq!(status.style.text, theme::LOG_ERROR);
        let mut read = false;
        sync(
            &view(Some(1)),
            || {
                read = true;
                Vec::new()
            },
            &mut world,
            &panel,
        );
        assert!(!read, "an unchanged key is not read again");
    }

    #[test]
    fn a_hidden_reference_hides_its_column() {
        let mut world = World::new();
        let panel = built(&mut world);
        sync(&view(None), Vec::new, &mut world, &panel);
        assert!(
            !world
                .get_by_id::<List>(panel.slots[REF_SLOT])
                .unwrap()
                .visible
        );
    }

    #[test]
    fn a_line_marked_twice_shows_its_error_and_a_new_severity_shows() {
        let mut world = World::new();
        let panel = built(&mut world);
        let marker = |severity| GutterMarker {
            line: 2,
            column: 0,
            severity,
            message: String::new(),
        };
        let both = [marker(Severity::Warning), marker(Severity::Error)];
        let v = SourceView {
            markers: &both,
            ..view(None)
        };
        sync(&v, Vec::new, &mut world, &panel);
        let area = world.get_by_id::<TextArea>(panel.slots[AREA_SLOT]).unwrap();
        assert_eq!(area.marks[0].color, theme::LOG_ERROR);
        let warning = [marker(Severity::Warning)];
        let v = SourceView {
            markers: &warning,
            ..view(None)
        };
        sync(&v, Vec::new, &mut world, &panel);
        let area = world.get_by_id::<TextArea>(panel.slots[AREA_SLOT]).unwrap();
        assert_eq!(area.marks.len(), 1);
        assert_eq!(area.marks[0].color, theme::LOG_WARN);
    }

    #[test]
    fn a_steady_frame_allocates_nothing() {
        let mut world = World::new();
        let panel = built(&mut world);
        let reference = Reference::default();
        let markers = [GutterMarker {
            line: 1,
            column: 0,
            severity: Severity::Error,
            message: String::new(),
        }];
        let v = SourceView {
            markers: &markers,
            ..view(Some(3))
        };
        sync(&v, || reference.rows(ENTRIES), &mut world, &panel);
        let min = concinnity_testing::quietest_allocs(0, || {
            sync(&v, || reference.rows(ENTRIES), &mut world, &panel);
        });
        assert_eq!(min, 0, "a steady sync allocated {min} times");
    }
}
