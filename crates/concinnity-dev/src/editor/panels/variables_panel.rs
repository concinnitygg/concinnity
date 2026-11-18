//! The Variables panel: the world's variable table as one row per variable,
//! its type and starting value beside its name. The header names the selected
//! variable (or says whether the table is authoritative) and adds one; the
//! toolbar acts on the selected row (retype it, type its value, remove it), and
//! a row for a name the behaviors use but the table leaves out offers to declare
//! it instead.
//!
//! The table is one asset, so this panel edits one asset's args the way the
//! Behavior panel edits one behavior's: directly, committing as it goes, with the
//! build's own checker reporting on the result. `hook/edit/variables.rs` owns the
//! actions and `editor/panels/variables.rs` turns the args into these rows.

use concinnity_engine::components::{
    Binding, Button, List, TextAlign, TextField, TextLabel, UiEvent, UiItem, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::registry::PanelKey;
use super::variables::Row;
use crate::editor::theme;
use crate::editor::ui::build::emit;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

pub(crate) const VARIABLES_W: f32 = 460.0;
const ROW_H: f32 = 22.0;
const CTRL_H: f32 = 26.0;
const NEW_W: f32 = 58.0;
const BTN_W: f32 = 70.0;
// Visible rows at the default height; more appear as the panel grows.
const ROW_POOL: usize = 12;

// An undeclared name is what the table is missing, not what it holds.
const MISSING_LABEL: [f32; 3] = theme::LOG_WARN;

// The state the panel shows, and the events its controls send.
const ROWS: &str = "editor.vars.rows";
const SELECTED: &str = "editor.vars.selected";
const SELECT: &str = "editor.vars.select";
const CAPTION: &str = "editor.vars.caption";
const STATUS: &str = "editor.vars.status";
const TYPE_CAPTION: &str = "editor.vars.type_caption";
const NEW: &str = "editor.vars.new";
const TYPE: &str = "editor.vars.type";
const DELETE: &str = "editor.vars.delete";

// The panel's slots.
const NAME_SLOT: usize = 0;
const VALUE_SLOT: usize = 1;
const TYPE_SLOT: usize = 2;
const DELETE_SLOT: usize = 3;
const CAPTION_SLOT: usize = 4;

pub(crate) fn size() -> [f32; 2] {
    [
        VARIABLES_W,
        panel_height(3.0 * CTRL_H + ROW_POOL as f32 * ROW_H + 2.0 * ROW_H),
    ]
}

pub(crate) fn default_origin(vp_w: f32) -> [f32; 2] {
    [(vp_w - VARIABLES_W - 40.0).max(20.0), 140.0]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let name = ui.text_field(TextField {
        placeholder: "name".to_string(),
        height: CTRL_H,
        ..Default::default()
    })?;
    let caption = ui.bound_label(CAPTION, theme::LABEL_DIM, TextAlign::Left)?;
    let new = ui.button("New", NEW, None, [NEW_W, CTRL_H])?;
    let slots_head = [name.item?.id(), caption.item?.id()];
    let header = ui.row(vec![name, caption, new], CTRL_H)?;
    let retype = ui.button_with(
        Button {
            text_binding: Some(Binding::State(TYPE_CAPTION.to_string())),
            action: Some(emit(TYPE, None)),
            ..Default::default()
        },
        [BTN_W, CTRL_H],
    )?;
    let delete = ui.button("Del", DELETE, None, [BTN_W, CTRL_H])?;
    let value = ui.text_field(TextField {
        placeholder: "value".to_string(),
        height: CTRL_H,
        ..Default::default()
    })?;
    let slots_tool = [value.item?.id(), retype.item?.id(), delete.item?.id()];
    let toolbar = ui.row(vec![retype, delete, value], CTRL_H)?;
    let rows = ui.list(List {
        items: ROWS.to_string(),
        selected: Some(Binding::State(SELECTED.to_string())),
        row_height: ROW_H,
        on_select: SELECT.to_string(),
        ..Default::default()
    })?;
    let status = ui.bound_label(STATUS, theme::LOG_WARN, TextAlign::Left)?;
    let spec = PanelSpec {
        key: PanelKey::Variables,
        title: "Variables",
        title_state: None,
        size: size(),
        min_size: Some(size()),
    };
    let mut panel = ui.panel(&spec, vec![header, toolbar, rows, status])?;
    panel.slots = vec![
        slots_head[0],
        slots_tool[0],
        slots_tool[1],
        slots_tool[2],
        slots_head[1],
    ];
    Some(panel)
}

// The name and value fields' ids.
pub(crate) fn fields(built: &BuiltPanel) -> [Option<AssetId>; 2] {
    [NAME_SLOT, VALUE_SLOT].map(|slot| built.slots.get(slot).copied())
}

// What the panel shows this frame.
#[expect(
    clippy::struct_excessive_bools,
    reason = "a per-frame snapshot of independent panel states"
)]
pub(crate) struct VariablesView<'a> {
    pub rows: &'a [Row],
    pub selected: Option<usize>,
    pub authoritative: bool,
    pub name_focus: bool,
    pub value_focus: bool,
    pub status: Option<&'a str>,
    pub live: bool,
}

impl VariablesView<'_> {
    fn selected_row(&self) -> Option<&Row> {
        self.selected.and_then(|i| self.rows.get(i))
    }

    fn missing(&self) -> usize {
        self.rows.iter().filter(|r| !r.declared()).count()
    }
}

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VariablesAction {
    New,
    Select(usize),
    // The type button: declare an undeclared row, retype a declared one.
    Type,
    Remove,
}

// The action a Variables event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<VariablesAction> {
    Some(match event.name.as_str() {
        NEW => VariablesAction::New,
        TYPE => VariablesAction::Type,
        DELETE => VariablesAction::Remove,
        SELECT => match event.value {
            Some(UiValue::Int(i)) => VariablesAction::Select(usize::try_from(i).ok()?),
            _ => return None,
        },
        _ => return None,
    })
}

// The rows as list items: the name, then its type and value. An undeclared
// name reads in the warning color and says so; a live session's behavior local
// reads in the command color, inspectable but not part of the table.
fn items(rows: &[Row], live: bool) -> Vec<UiItem> {
    rows.iter()
        .map(|row| {
            let (color, ty) = if row.local {
                (theme::LOG_COMMAND, format!("{} local", row.ty))
            } else if row.declared() {
                (theme::LABEL, row.ty.clone())
            } else {
                (MISSING_LABEL, "undeclared".to_string())
            };
            let at = if live { "now" } else { "starts at" };
            let detail = if row.value.is_empty() {
                ty
            } else {
                format!("{ty}  {at} {}", row.value)
            };
            UiItem {
                detail,
                color: Some(color),
                ..UiItem::new(&row.name)
            }
        })
        .collect()
}

// What the header says about the table as a whole. A declared table is held to
// every name its behaviors use, so what it is missing is the thing worth saying.
fn table_caption(view: &VariablesView<'_>) -> (String, [f32; 3]) {
    let missing = view.missing();
    match (view.authoritative, missing) {
        (false, 0) => (
            "no table: variables are implicit ints".to_string(),
            theme::LABEL_DIM,
        ),
        (false, n) => (
            format!("no table: {n} implicit int variable{}", plural(n)),
            theme::LABEL_DIM,
        ),
        (true, 0) => ("every variable is declared".to_string(), theme::LABEL_DIM),
        (true, n) => (
            format!("{n} variable{} used but not declared", plural(n)),
            MISSING_LABEL,
        ),
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

// Show the rows, rebuilt only when they changed, and the controls the
// selection allows.
pub(crate) fn sync(view: &VariablesView<'_>, world: &mut World, built: &BuiltPanel) {
    let items = items(view.rows, view.live);
    let row = view.selected_row();
    let declared = row.is_some_and(Row::declared);
    let (caption, caption_color) = if declared {
        (String::new(), theme::LABEL_DIM)
    } else {
        table_caption(view)
    };
    let selected = view
        .selected
        .and_then(|i| i32::try_from(i).ok())
        .unwrap_or(-1);
    state::with_state(world, |s| {
        if s.list(ROWS) != items.as_slice() {
            s.set_list(ROWS, items);
        }
        s.set(SELECTED, UiValue::Int(selected));
    });
    state::set_text(world, CAPTION, format_args!("{caption}"));
    state::set_text(world, STATUS, format_args!("{}", view.status.unwrap_or("")));
    let type_caption = match row {
        Some(r) if !r.declared() => "Declare",
        _ => "Type",
    };
    state::set_text(world, TYPE_CAPTION, format_args!("{type_caption}"));
    let slot = |i: usize| built.slots.get(i).copied();
    if let Some(label) = slot(CAPTION_SLOT).and_then(|id| world.get_mut_by_id::<TextLabel>(id)) {
        label.color = caption_color;
    }
    let editable = row.is_some_and(|r| !r.local);
    let set_button = |world: &mut World, i: usize, live: bool| {
        if let Some(b) = slot(i).and_then(|id| world.get_mut_by_id::<Button>(id)) {
            b.disabled = !live;
        }
    };
    set_button(world, TYPE_SLOT, editable);
    set_button(world, DELETE_SLOT, editable && declared);
    for (i, focus) in [(NAME_SLOT, view.name_focus), (VALUE_SLOT, view.value_focus)] {
        if let Some(field) = slot(i).and_then(|id| world.get_mut_by_id::<TextField>(id)) {
            field.visible = editable && declared;
            field.focused = field.visible && focus;
        }
    }
}

// Which of the two fields a press focused, as `[name, value]`.
pub(crate) fn focused(world: &World, built: &BuiltPanel) -> [bool; 2] {
    fields(built).map(|id| {
        id.and_then(|id| world.get_by_id::<TextField>(id))
            .is_some_and(|f| f.focused)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;

    fn row(name: &str, at: Option<usize>) -> Row {
        Row {
            name: name.into(),
            at,
            ty: if at.is_some() {
                "int".into()
            } else {
                String::new()
            },
            value: if at.is_some() {
                "3".into()
            } else {
                String::new()
            },
            local: false,
        }
    }

    fn view(rows: &[Row], selected: Option<usize>) -> VariablesView<'_> {
        VariablesView {
            rows,
            selected,
            authoritative: true,
            name_focus: false,
            value_focus: false,
            status: None,
            live: false,
        }
    }

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    #[test]
    fn rows_show_type_and_value_and_warn_when_undeclared() {
        let rows = [row("score", Some(0)), row("lives", None)];
        let items = items(&rows, false);
        assert_eq!(items[0].text, "score");
        assert_eq!(items[0].detail, "int  starts at 3");
        assert_eq!(items[1].detail, "undeclared");
        assert_eq!(items[1].color, Some(MISSING_LABEL));
    }

    #[test]
    fn the_controls_follow_the_selection() {
        let mut world = World::new();
        let panel = built(&mut world);
        let rows = [row("score", Some(0)), row("lives", None)];
        sync(&view(&rows, Some(1)), &mut world, &panel);
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.get(TYPE_CAPTION), Some(&UiValue::Text("Declare".into())));
        assert_eq!(
            s.get(CAPTION),
            Some(&UiValue::Text("1 variable used but not declared".into()))
        );
        let delete = world.get_by_id::<Button>(panel.slots[DELETE_SLOT]).unwrap();
        assert!(delete.disabled, "an undeclared row has nothing to remove");
        let name = world
            .get_by_id::<TextField>(panel.slots[NAME_SLOT])
            .unwrap();
        assert!(!name.visible);

        sync(&view(&rows, Some(0)), &mut world, &panel);
        let name = world
            .get_by_id::<TextField>(panel.slots[NAME_SLOT])
            .unwrap();
        assert!(name.visible, "a declared row can be renamed");
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.get(CAPTION), Some(&UiValue::Text(String::new())));
    }

    #[test]
    fn a_selection_names_its_row() {
        let event = UiEvent {
            name: SELECT.into(),
            element: None,
            value: Some(UiValue::Int(2)),
        };
        assert_eq!(action(&event), Some(VariablesAction::Select(2)));
    }
}
