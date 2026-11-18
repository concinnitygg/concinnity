//! The editor "Assets" panel: every asset of the expanded world as one
//! collapsible tree grouped by origin (built by `asset_tree.rs`), a search field
//! over it, and the per-row editor-session hide / lock toggles. Under the title
//! bar the header is a square "+" (add) button and the search field, then a
//! status line (the asset count, or a cook failure), then the tree.
//!
//! Clicking a name selects the asset in the viewport and opens its add / edit
//! form -- a separate floating panel (`form_panel.rs`); a Shift, Ctrl, or Cmd
//! click adds it to the selection or takes it out. An asset the build
//! generates has no world.jsonl line of its own; its form is seeded from the
//! entry the expansion produced and only confirming it appends that line, which
//! then overrides the expansion. The passes that emit unconditionally (menu,
//! story, and prefab primitives) cannot be overridden that way, so their rows
//! select but open no form. A row with a line of its own has a context menu to
//! delete it, and to export a skinned mesh.
//!
//! While the type picker is open, the "+" becomes an "x" that returns to the
//! tree and the search field narrows the picker's option list, which takes the
//! tree's place; picking a type opens the add form.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::{
    Binding, Button, List, ListToggle, Menu, TextAlign, TextField, TextLabel, UiEvent, UiItem,
    UiState, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;
use std::sync::OnceLock;

use super::asset_tree::{Badge, TreeRow};
use super::registry::PanelKey;
use crate::editor::icons::{self, Glyph};
use crate::editor::ui::build::emit;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};
use crate::editor::{hud, theme};

// The addable asset types the "+" picker offers for a plain add: every type
// flagged `useful_blank` in the registry (both the component and resource-asset
// lists). The flag marks a type that (a) recompiles cleanly when added with
// default args and (b) is useful when added blank; the choices are guarded here
// by `add_types_cook_with_default_args` (runs every offered type through the
// real cook pipeline) and `add_types_are_the_curated_blank_useful_addable_set`
// (every addable-and-cooks-blank type must be flagged or in that test's
// EXCLUDED list with a reason, so a newly registered type is a deliberate
// choice, never a silent omission). Types that cook blank but stay unflagged:
// world-config singletons want the edit-or-add flow below, engine-injected
// HUDs (DebugHud / StatHud) are added by `cn cook`, and types defined by a
// nested array or a source file (Model, Scene, Story, ...) are inert until the
// nested / source form controls exist -- Behavior is flagged because it has
// one: its body is authored in the Behavior panel. Types that cannot even cook
// blank --
// needing a source or a required cross-reference (Mesh / AudioClip / Joint) --
// can never be flagged.
pub(crate) fn add_types() -> impl Iterator<Item = &'static str> {
    static TYPES: OnceLock<Vec<&'static str>> = OnceLock::new();
    TYPES
        .get_or_init(|| {
            let components = RegisteredType::all()
                .iter()
                .filter(|t| t.useful_blank())
                .map(|t| t.as_str());
            components.collect()
        })
        .iter()
        .copied()
}

// The world-config singletons (the registry's `singleton` flag on declarable
// types): exactly one instance belongs to a world. They are offered in the "+"
// picker alongside `add_types`, but picking one EDITS the world's existing
// instance when it has one and only ADDS when it does not (the hook's
// `open_form` is handed the existing entry's index) -- an edit-or-add flow,
// never a blind second append. Held apart from `add_types` so the plain add
// path can keep assuming multi-instance. Like the addables, each must cook
// blank (guarded).
pub(crate) fn config_types() -> impl Iterator<Item = &'static str> {
    static TYPES: OnceLock<Vec<&'static str>> = OnceLock::new();
    TYPES
        .get_or_init(|| {
            RegisteredType::all()
                .iter()
                .filter(|t| t.singleton() && t.addable())
                .map(|t| t.as_str())
                .collect()
        })
        .iter()
        .copied()
}

// Whether `ty` is a world-config singleton (edit-or-add rather than blind append).
pub(crate) fn is_singleton(ty: &str) -> bool {
    RegisteredType::parse(ty).is_some_and(RegisteredType::singleton)
}

// Why a world holding `entries` cannot take another `ty`, for a type with a
// per-world limit past one.
pub(crate) fn add_refused(ty: &str, entries: &[serde_json::Value]) -> Option<String> {
    match ty {
        "Shader" => {
            let count = super::shader::list::shader_count(entries);
            (!super::shader::list::can_add_shader(count)).then(super::shader::list::limit_reason)
        }
        _ => None,
    }
}

// Every type the "+" picker offers: the multi-instance addables plus the config
// singletons.
pub(crate) fn picker_types() -> impl Iterator<Item = &'static str> {
    add_types().chain(config_types())
}

/// One row of the "+" picker: the type to add, and the one-line summary from
/// its own documentation. The summary is empty for a type whose rustdoc opens
/// with something other than prose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PickerOption {
    pub name: String,
    pub summary: String,
}

impl PickerOption {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            summary: crate::docs::type_summaries()
                .get(name)
                .cloned()
                .unwrap_or_default(),
        }
    }
}

// The default (and minimum) panel width; the user can widen it past this.
pub(crate) const PANEL_W: f32 = 500.0;
const HEADER_H: f32 = 32.0;
const STATUS_H: f32 = 22.0;
const ROW_H: f32 = 26.0;
// Rows shown at the default height; a taller panel shows more.
const DEFAULT_ROWS: usize = 14;

const EYE_TINT: [f32; 4] = [0.82, 0.84, 0.90, 1.0];
const EYE_TINT_HIDDEN: [f32; 4] = [0.48, 0.49, 0.55, 1.0];
// Amber reads as "held" without the alarm of red.
const LOCK_TINT: [f32; 4] = [0.90, 0.72, 0.40, 1.0];
const LOCK_TINT_HINT: [f32; 4] = [0.56, 0.57, 0.63, 0.75];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];

// The state the panel shows, and the events its controls send.
pub(crate) const TREE: &str = "editor.assets.tree";
const TREE_SHOWN: [&str; 2] = ["editor.assets.tree.hi", "editor.assets.tree.lo"];
pub(crate) const PICKER: &str = "editor.assets.picker";
const PICKER_SHOWN: [&str; 2] = ["editor.assets.picker.hi", "editor.assets.picker.lo"];
const STATUS: &str = "editor.assets.status";
const PLUS: &str = "editor.assets.plus";
const TOGGLE_PICKER: &str = "editor.assets.toggle_picker";
pub(crate) const ROW: &str = "editor.assets.row";
const EXTEND: &str = "editor.assets.extend";
const EXPAND: &str = "editor.assets.expand";
const HIDE: &str = "editor.assets.hide";
const LOCK: &str = "editor.assets.lock";
pub(crate) const CONTEXT: &str = "editor.assets.context";
const PICK: &str = "editor.assets.pick";
pub(crate) const MENU_OPEN: &str = "editor.assets.menu_open";
pub(crate) const MENU_ITEMS: &str = "editor.assets.menu_items";
const MENU_PICK: &str = "editor.assets.menu_pick";

// The panel's slots.
const SEARCH_SLOT: usize = 0;
const STATUS_SLOT: usize = 1;
const TREE_SLOT: usize = 2;
const PICKER_SLOT: usize = 3;

// The toggle columns' bits on an asset row.
const SHOWN_BIT: u8 = 1;
const LOCKED_BIT: u8 = 2;

// Per-badge type-label colors, so provenance reads at a glance beside the
// asset's type.
fn badge_color(badge: Badge) -> [f32; 3] {
    match badge {
        Badge::Authored => theme::LABEL_DIM,
        Badge::Imported => [0.45, 0.72, 0.62],
        Badge::Injected => [0.70, 0.58, 0.88],
        Badge::Overridden => [0.62, 0.76, 1.0],
    }
}

// Where the panel sits until the user drags it: right-aligned below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw - PANEL_W, hud::body_top()]
}

// The panel's footprint for `rows` rows of tree.
fn size_for(rows: usize) -> [f32; 2] {
    [
        PANEL_W,
        panel_height(HEADER_H + STATUS_H + rows as f32 * ROW_H),
    ]
}

// The size the panel opens at.
pub(crate) fn size() -> [f32; 2] {
    size_for(DEFAULT_ROWS)
}

// A resolved panel press, in the tree's own terms: row picks carry the group
// and the index within it, so the hook resolves the asset (and its promote
// entry) through the same model the rows were flattened from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelAction {
    // The "+" button: open the type picker (from the tree) or close it.
    TogglePicker,
    // Choose picker option `i`.
    PickOption(usize),
    // A group header: fold / unfold it.
    ToggleGroup(usize),
    // An asset row's body: select it and open its editing surface.
    SelectRow(usize, usize),
    // A modified click on an asset row: add it to the selection or take it
    // out.
    ExtendRow(usize, usize),
    // An asset row's eye: flip the editor-session hide.
    ToggleHide(usize, usize),
    // An asset row's lock: flip the pick lock.
    ToggleLock(usize, usize),
    // An asset row's context menu.
    OpenRowMenu(usize, usize),
    // The open row menu's Delete.
    RowDelete,
    // The open row menu's Export (skinned meshes only): write the asset as
    // .glb beside the project.
    RowExport,
}

// The (group, index) of tree row `i`, when it lists an asset.
fn asset_at(rows: &[TreeRow], i: usize) -> Option<(usize, usize)> {
    match rows.get(i)? {
        TreeRow::Asset { group, index, .. } => Some((*group, *index)),
        TreeRow::Header { .. } => None,
    }
}

// The entries of the menu of a row: Delete for a row with a line of its own,
// and Export for a skinned mesh among those.
pub(crate) fn row_menu(row: &TreeRow) -> &'static [RowMenuItem] {
    match row {
        TreeRow::Asset {
            badge: Badge::Authored | Badge::Overridden,
            asset_type: RegisteredType::SkinnedMesh | RegisteredType::CharacterModel,
            ..
        } => &[RowMenuItem::Delete, RowMenuItem::Export],
        TreeRow::Asset {
            badge: Badge::Authored | Badge::Overridden,
            ..
        } => &[RowMenuItem::Delete],
        _ => &[],
    }
}

// One entry of a row's menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowMenuItem {
    Delete,
    Export,
}

impl RowMenuItem {
    fn caption(self) -> &'static str {
        match self {
            Self::Delete => "Delete",
            Self::Export => "Export glTF",
        }
    }
}

// The action an Assets event asks for, if `event` is one: list indices
// resolved against the tree `rows` shown, and a menu pick against the menu of
// `menu_row`.
pub(crate) fn action(
    event: &UiEvent,
    rows: &[TreeRow],
    menu_row: Option<&TreeRow>,
) -> Option<PanelAction> {
    if event.name == TOGGLE_PICKER {
        return Some(PanelAction::TogglePicker);
    }
    let i = match event.value {
        Some(UiValue::Int(i)) => usize::try_from(i).ok()?,
        _ => return None,
    };
    let asset = || asset_at(rows, i);
    Some(match event.name.as_str() {
        PICK => PanelAction::PickOption(i),
        ROW => match rows.get(i)? {
            TreeRow::Header { group, .. } => PanelAction::ToggleGroup(*group),
            TreeRow::Asset { group, index, .. } => PanelAction::SelectRow(*group, *index),
        },
        EXPAND => match rows.get(i)? {
            TreeRow::Header { group, .. } => PanelAction::ToggleGroup(*group),
            TreeRow::Asset { .. } => return None,
        },
        EXTEND => {
            let (g, a) = asset()?;
            PanelAction::ExtendRow(g, a)
        }
        HIDE => {
            let (g, a) = asset()?;
            PanelAction::ToggleHide(g, a)
        }
        LOCK => {
            let (g, a) = asset()?;
            PanelAction::ToggleLock(g, a)
        }
        CONTEXT => {
            let (g, a) = asset()?;
            if row_menu(rows.get(i)?).is_empty() {
                return None;
            }
            PanelAction::OpenRowMenu(g, a)
        }
        MENU_PICK => match row_menu(menu_row?).get(i)? {
            RowMenuItem::Delete => PanelAction::RowDelete,
            RowMenuItem::Export => PanelAction::RowExport,
        },
        _ => return None,
    })
}

fn icon_toggle(
    on: Glyph,
    off: Glyph,
    on_color: [f32; 4],
    off_color: [f32; 4],
    event: &str,
) -> ListToggle {
    ListToggle {
        on: icons::icon(on),
        off: icons::icon(off),
        on_color,
        off_color,
        on_toggle: event.to_string(),
    }
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let plus = ui.button_with(
        Button {
            label: "+".to_string(),
            text_binding: Some(Binding::State(PLUS.to_string())),
            action: Some(emit(TOGGLE_PICKER, None)),
            ..Default::default()
        },
        [HEADER_H - 6.0, HEADER_H - 6.0],
    )?;
    let search = ui.text_field(TextField {
        placeholder: "search".to_string(),
        height: HEADER_H - 6.0,
        ..Default::default()
    })?;
    let search_id = search.item?.id();
    let header = ui.row(vec![plus, search], HEADER_H)?;
    let status = ui.bound_label(STATUS, theme::LABEL_DIM, TextAlign::Left)?;
    let status_id = status.item?.id();
    let status = ui.row(vec![status], STATUS_H)?;
    let tree = ui.list(List {
        items: TREE.to_string(),
        row_height: ROW_H,
        indent: 6.0,
        on_select: ROW.to_string(),
        on_expand: EXPAND.to_string(),
        on_extend: EXTEND.to_string(),
        on_context: CONTEXT.to_string(),
        toggles: vec![
            icon_toggle(icons::EYE, icons::EYE_SHUT, EYE_TINT, EYE_TINT_HIDDEN, HIDE),
            icon_toggle(icons::LOCK, &[], LOCK_TINT, LOCK_TINT_HINT, LOCK),
        ],
        ..Default::default()
    })?;
    let tree_id = tree.item?.id();
    let picker = ui.list(List {
        items: PICKER.to_string(),
        row_height: ROW_H,
        on_select: PICK.to_string(),
        visible: false,
        ..Default::default()
    })?;
    let picker_id = picker.item?.id();
    ui.menu(Menu {
        open: Some(Binding::State(MENU_OPEN.to_string())),
        items: MENU_ITEMS.to_string(),
        on_pick: MENU_PICK.to_string(),
        at_cursor: true,
        ..Default::default()
    })?;
    let spec = PanelSpec {
        key: PanelKey::Assets,
        title: "Assets",
        title_state: None,
        size: size(),
        min_size: Some(size_for(4)),
    };
    let mut panel = ui.panel(&spec, vec![header, status, tree, picker])?;
    panel.slots = vec![search_id, status_id, tree_id, picker_id];
    Some(panel)
}

// The search field's id.
pub(crate) fn search_field(built: &BuiltPanel) -> Option<AssetId> {
    built.slots.get(SEARCH_SLOT).copied()
}

// The tree as list items, each asset row decorated by `decor`: whether it is
// selected, hidden, and locked.
pub(crate) fn tree_items(rows: &[TreeRow], mut decor: impl FnMut(&str) -> RowDecor) -> Vec<UiItem> {
    rows.iter()
        .map(|row| match row {
            TreeRow::Header {
                label, count, open, ..
            } => UiItem {
                detail: count.to_string(),
                expandable: true,
                expanded: *open,
                color: Some(theme::HEADING),
                ..UiItem::new(label)
            },
            TreeRow::Asset {
                name,
                asset_type,
                badge,
                ..
            } => {
                let d = decor(name);
                let mut toggles = 0;
                if !d.hidden {
                    toggles |= SHOWN_BIT;
                }
                if d.locked {
                    toggles |= LOCKED_BIT;
                }
                UiItem {
                    detail: asset_type.as_str().to_string(),
                    detail_color: Some(badge_color(*badge)),
                    depth: 1,
                    dimmed: d.hidden,
                    selected: d.selected,
                    toggles: Some(toggles),
                    ..UiItem::new(name)
                }
            }
        })
        .collect()
}

// How an asset row shows its session state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RowDecor {
    pub selected: bool,
    pub hidden: bool,
    pub locked: bool,
}

// What the panel shows this frame.
pub(crate) struct AssetsView<'a> {
    // Bumped whenever the tree's items change.
    pub tree_rev: u64,
    pub tree: &'a [UiItem],
    // The type picker's options while it is open, bumped as they change.
    pub picker: Option<(u64, &'a [PickerOption])>,
    // Total assets across every group, for the status line.
    pub total: usize,
    // Why the tree is empty, when a cook of the working entries failed.
    pub error: Option<&'a str>,
    pub search_focus: bool,
    // The open row menu's row, by its place in the tree.
    pub menu: Option<&'a TreeRow>,
    // A row to scroll into view.
    pub reveal: Option<usize>,
}

pub(crate) fn sync(view: &AssetsView<'_>, world: &mut World, built: &BuiltPanel) {
    let slot = |i: usize| built.slots.get(i).copied();
    if !state::is_shown(world, TREE_SHOWN, view.tree_rev) {
        let items = view.tree.to_vec();
        state::with_state(world, |s| {
            s.set_list(TREE, items);
            state::mark_shown(s, TREE_SHOWN, view.tree_rev);
        });
    }
    if let Some((rev, options)) = view.picker
        && !state::is_shown(world, PICKER_SHOWN, rev)
    {
        let items = options
            .iter()
            .map(|o| UiItem {
                detail: o.summary.clone(),
                ..UiItem::new(&o.name)
            })
            .collect();
        state::with_state(world, |s| {
            s.set_list(PICKER, items);
            state::mark_shown(s, PICKER_SHOWN, rev);
        });
    }
    let picking = view.picker.is_some();
    state::set_text(
        world,
        PLUS,
        format_args!("{}", if picking { "x" } else { "+" }),
    );
    let color = match (view.error, view.picker) {
        (Some(e), _) => {
            state::set_text(world, STATUS, format_args!("{e}"));
            ERROR_LABEL
        }
        (None, Some((_, options))) => {
            let text = if options.is_empty() {
                "No matching types"
            } else {
                "Add an asset"
            };
            state::set_text(world, STATUS, format_args!("{text}"));
            theme::LABEL_DIM
        }
        (None, None) if view.tree.is_empty() => {
            state::set_text(world, STATUS, format_args!("No matching assets"));
            theme::LABEL_DIM
        }
        (None, None) => {
            state::set_text(world, STATUS, format_args!("Assets ({})", view.total));
            theme::LABEL_DIM
        }
    };
    if let Some(l) = slot(STATUS_SLOT).and_then(|id| world.get_mut_by_id::<TextLabel>(id)) {
        l.color = color;
    }
    if let Some(f) = slot(SEARCH_SLOT).and_then(|id| world.get_mut_by_id::<TextField>(id)) {
        f.focused = view.search_focus;
    }
    if let Some(list) = slot(TREE_SLOT).and_then(|id| world.get_mut_by_id::<List>(id)) {
        list.visible = !picking;
        if let Some(i) = view.reveal {
            list.reveal(i, view.tree.len());
        }
    }
    if let Some(list) = slot(PICKER_SLOT).and_then(|id| world.get_mut_by_id::<List>(id)) {
        list.visible = picking;
    }
    let menu = view.menu.map_or(&[][..], row_menu);
    state::with_state(world, |s| {
        let listed = s.list(MENU_ITEMS);
        let same = listed.len() == menu.len()
            && listed.iter().zip(menu).all(|(a, b)| a.text == b.caption());
        if !same {
            let items = menu.iter().map(|m| UiItem {
                color: (*m == RowMenuItem::Delete).then_some(DELETE_LABEL),
                ..UiItem::new(m.caption())
            });
            s.set_list(MENU_ITEMS, items.collect());
        }
        s.set(MENU_OPEN, UiValue::Bool(!menu.is_empty()));
    });
}

// Whether the row menu the hook keeps open is still open: a press off it, or
// Escape, closes it.
pub(crate) fn menu_open(world: &World) -> bool {
    world
        .resource::<UiState>()
        .is_some_and(|s| s.get(MENU_OPEN) == Some(&UiValue::Bool(true)))
}

// Whether the search field holds the keyboard, as the widget last showed it.
pub(crate) fn search_focused(world: &World, built: &BuiltPanel) -> bool {
    search_field(built)
        .and_then(|id| world.get_by_id::<TextField>(id))
        .is_some_and(|f| f.focused)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    fn rows() -> Vec<TreeRow> {
        vec![
            TreeRow::Header {
                group: 0,
                label: "World".into(),
                count: 2,
                open: true,
            },
            TreeRow::Asset {
                group: 0,
                index: 0,
                name: "hero".into(),
                asset_type: RegisteredType::SkinnedMesh,
                badge: Badge::Authored,
            },
            TreeRow::Asset {
                group: 0,
                index: 1,
                name: "rock".into(),
                asset_type: RegisteredType::Prop,
                badge: Badge::Imported,
            },
        ]
    }

    fn event(name: &str, i: i32) -> UiEvent {
        UiEvent {
            name: name.into(),
            element: None,
            value: Some(UiValue::Int(i)),
        }
    }

    #[test]
    fn list_events_resolve_to_the_rows_they_name() {
        let rows = rows();
        let act = |name: &str, i: i32| action(&event(name, i), &rows, None);
        assert_eq!(act(ROW, 0), Some(PanelAction::ToggleGroup(0)));
        assert_eq!(act(EXPAND, 0), Some(PanelAction::ToggleGroup(0)));
        assert_eq!(act(ROW, 2), Some(PanelAction::SelectRow(0, 1)));
        assert_eq!(act(EXTEND, 1), Some(PanelAction::ExtendRow(0, 0)));
        assert_eq!(act(HIDE, 2), Some(PanelAction::ToggleHide(0, 1)));
        assert_eq!(act(LOCK, 1), Some(PanelAction::ToggleLock(0, 0)));
        assert_eq!(act(HIDE, 0), None, "a header has no toggles");
        assert_eq!(act(CONTEXT, 1), Some(PanelAction::OpenRowMenu(0, 0)));
        assert_eq!(
            act(CONTEXT, 2),
            None,
            "an imported row has no line to delete"
        );
        assert_eq!(act(PICK, 3), Some(PanelAction::PickOption(3)));
    }

    #[test]
    fn a_skinned_mesh_menu_offers_export_after_delete() {
        let rows = rows();
        let pick = |k: i32| action(&event(MENU_PICK, k), &rows, Some(&rows[1]));
        assert_eq!(pick(0), Some(PanelAction::RowDelete));
        assert_eq!(pick(1), Some(PanelAction::RowExport));
        assert_eq!(pick(2), None);
        assert_eq!(row_menu(&rows[2]), &[] as &[RowMenuItem]);
    }

    #[test]
    fn asset_rows_carry_their_session_state() {
        let items = tree_items(&rows(), |name| RowDecor {
            selected: name == "hero",
            hidden: name == "rock",
            locked: name == "hero",
        });
        assert!(items[0].expandable && items[0].expanded);
        assert_eq!(items[0].detail, "2");
        assert!(items[1].selected && !items[1].dimmed);
        assert_eq!(items[1].toggles, Some(SHOWN_BIT | LOCKED_BIT));
        assert_eq!(items[2].toggles, Some(0), "hidden and unlocked");
        assert!(items[2].dimmed);
        assert_eq!(items[2].detail_color, Some(badge_color(Badge::Imported)));
    }

    #[test]
    fn the_picker_takes_the_trees_place_and_the_plus_reads_x() {
        let mut world = World::new();
        let panel = built(&mut world);
        let options = [PickerOption::new("PointLight")];
        let view = AssetsView {
            tree_rev: 1,
            tree: &[],
            picker: Some((1, &options)),
            total: 0,
            error: None,
            search_focus: true,
            menu: None,
            reveal: None,
        };
        sync(&view, &mut world, &panel);
        let shown = |slot: usize| world.get_by_id::<List>(panel.slots[slot]).unwrap().visible;
        assert!(!shown(TREE_SLOT) && shown(PICKER_SLOT));
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.get(PLUS), Some(&UiValue::Text("x".into())));
        assert_eq!(s.list(PICKER)[0].text, "PointLight");
        assert!(search_focused(&world, &panel));
    }

    #[test]
    fn a_cook_error_replaces_the_count_in_red() {
        let mut world = World::new();
        let panel = built(&mut world);
        let view = AssetsView {
            tree_rev: 1,
            tree: &[],
            picker: None,
            total: 0,
            error: Some("bad line"),
            search_focus: false,
            menu: None,
            reveal: None,
        };
        sync(&view, &mut world, &panel);
        let label = world
            .get_by_id::<TextLabel>(panel.slots[STATUS_SLOT])
            .unwrap();
        assert_eq!(label.color, ERROR_LABEL);
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.get(STATUS), Some(&UiValue::Text("bad line".into())));
    }

    // Cook `ty` blank in a minimal world: alongside a Window so the world
    // renders, except for singletons, which cook alone (the fixed Window line
    // would collide with the Window probe under the singleton shape rule; a
    // lone singleton still cooks, as a non-rendering world when it does not
    // render itself).
    fn cook_blank(ty: &str) -> std::io::Result<()> {
        let world = if is_singleton(ty) {
            format!("[\"{ty}\",{{\"$id\":\"probe\"}}]\n")
        } else {
            format!(
                "[\"Window\",{{\"$id\":\"gfx\"}}]\n\
                 [\"{ty}\",{{\"$id\":\"probe\"}}]\n"
            )
        };
        concinnity_cook::build_pipeline_from_str(
            &world,
            crate::project::assets_dir().as_deref(),
            crate::cook_platform().into(),
        )?;
        Ok(())
    }

    // Every offered add-type is a real External type whose default args cook in a
    // minimal rendering world. This is the guard the curated list leans on: a type
    // that needs a source file or a required cross-reference (Mesh, AudioClip,
    // Joint, ...) fails here and must not be listed.
    #[test]
    fn add_types_cook_with_default_args() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();
        for ty in picker_types() {
            let ct = RegisteredType::parse(ty)
                .unwrap_or_else(|| panic!("{ty} must be a known asset type"));
            assert!(ct.addable(), "{ty} must be External / addable");
            cook_blank(ty).unwrap_or_else(|e| panic!("{ty} must cook with default args: {e}"));
        }
    }

    // The set of offered types is exactly the addable types that cook with default
    // args AND are useful when added blank: no world-config singleton (those want an
    // edit-or-add flow), no engine-injected HUD, and no type whose value is a nested
    // array / source file it can't be given here. Enforced so a newly-registered
    // addable-and-blank-useful type is a deliberate `useful_blank` flag choice in
    // the registry, not forgotten.
    #[test]
    fn add_types_are_the_curated_blank_useful_addable_set() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();
        // Types that cook blank but are deliberately NOT offered, each for a reason
        // above. Keeping this explicit means the assertion below flags anything new.
        const EXCLUDED: &[&str] = &[
            // Engine-injected HUDs (added by `cn cook`, not by hand).
            "DebugHud",
            "StatHud",
            // Defined by nested content / a source the scalar form can't supply, so
            // a blank instance is inert.
            "Animation",
            "CharacterShape",
            "File",
            "Layout",
            "Model",
            "PropBody",
            "PropReflectionProbe",
            "RollingDrive",
            "RigidBody",
            "Scene",
            "ScrollPanel",
            "Spawner",
            "Story",
            "UiStateValues",
        ];
        // The config singletons are offered too (edit-or-add), so "offered" spans
        // both lists.
        let offered: std::collections::HashSet<&str> = picker_types().collect();
        let excluded: std::collections::HashSet<&str> = EXCLUDED.iter().copied().collect();
        for (_t, reg) in RegisteredType::addable_types() {
            let ty = reg.type_name;
            let cooks = cook_blank(ty).is_ok();
            if !cooks {
                // Needs a source / required reference: never offerable blank, and it
                // does not belong in the EXCLUDED (cooks-but-curated-out) list.
                assert!(
                    !offered.contains(ty),
                    "{ty} cannot cook blank yet is offered"
                );
                continue;
            }
            assert!(
                offered.contains(ty) ^ excluded.contains(ty),
                "{ty} cooks blank: flag it `useful_blank` in the registry or add it to the EXCLUDED list (with a reason), not both/neither"
            );
        }
    }
}
