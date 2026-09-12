// src/editor/panels/mod.rs
//
// The editor's floating panels: every surface the HUD opens over the viewport,
// plus the chrome they are built from. Each panel splits the same way -- a
// layout half (`*_panel.rs`) that is pure geometry over the panel's own state,
// and, where the panel's model is worth testing on its own, a data half beside
// it (`console.rs` under `console_panel.rs`, `variables.rs` under
// `variables_panel.rs`). Neither half owns state or reads input: the hook holds
// the state, drives the layout each frame, and routes the clicks
// (`hook/panels.rs` for the per-panel `Panel` impls, `hook/edit/` for what an
// action does to the world).
//
// `registry` is the panel system itself -- one entry per panel, which the
// reserved-id allocation, the View panel's rows, HUD injection, the focus
// stack, dragging and click routing all derive from. `panel` is the Assets
// panel, over the tree `asset_tree` builds and the grouping `asset_list`
// shares with the outliner. `form` and `form_panel` are the add / edit form
// every panel opens, and `list_panel` the shared chrome behind the simple row
// lists (Preview, Templates, View).
//
// Three panels keep a directory of their own instead, their model being more
// than one file: `editor/behavior/`, `editor/palette/` and `editor/worlds/`.

pub(crate) mod asset_list;
pub(crate) mod asset_tree;
pub(crate) mod character_shape;
pub(crate) mod character_shape_panel;
pub(crate) mod console;
pub(crate) mod console_panel;
pub(crate) mod content_panel;
pub(crate) mod form;
pub(crate) mod form_panel;
pub(crate) mod health;
pub(crate) mod health_panel;
pub(crate) mod import_panel;
pub(crate) mod lighting;
pub(crate) mod lighting_panel;
pub(crate) mod list_panel;
pub(crate) mod panel;
pub(crate) mod preview;
pub(crate) mod registry;
pub(crate) mod story;
pub(crate) mod story_panel;
pub(crate) mod template;
pub(crate) mod template_panel;
pub(crate) mod variables;
pub(crate) mod variables_panel;
pub(crate) mod view;
