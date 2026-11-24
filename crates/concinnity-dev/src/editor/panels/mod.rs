//! The editor's floating panels: every surface the HUD opens over the viewport,
//! built from engine widgets. Each panel splits the same way -- a widget half
//! (`*_panel.rs`) that builds the panel and shows the panel's own state on it,
//! and, where the panel's model is worth testing on its own, a data half beside
//! it (`console.rs` under `console_panel.rs`, `variables.rs` under
//! `variables_panel.rs`). Neither half owns state or reads input: the hook holds
//! the state, syncs the widgets each frame, and handles the events they send
//! (`hook/panels.rs` for the per-panel `Panel` impls, `hook/edit/` for what an
//! action does to the world).
//!
//! `registry` is the panel system itself -- one entry per panel, which the View
//! panel's rows, building, the focus stack and the close buttons all derive
//! from. `assets_panel` is the Assets panel, over the tree `asset_tree` builds
//! and the grouping `asset_list` shares with the outliner. `form` and
//! `form_panel` are the add / edit form every panel opens.
//!
//! A feature whose panels take more than a pair of files keeps them in a
//! folder: `shader/` (the Shaders list, the Shader source panel it spawns and
//! the shader reference), `sdf/` (the SDF field list and its form) and
//! `planet/` (the Planet panel and its tool strip). Three keep a directory
//! outside `panels/`, their model being more than one file: `editor/behavior/`,
//! `editor/palette/` and `editor/worlds/`.

pub(crate) mod asset_list;
pub(crate) mod asset_tree;
pub(crate) mod assets_panel;
pub(crate) mod character_shape;
pub(crate) mod character_shape_panel;
pub(crate) mod console;
pub(crate) mod console_panel;
pub(crate) mod content_panel;
pub(crate) mod form;
pub(crate) mod form_extras;
pub(crate) mod form_panel;
pub(crate) mod health;
pub(crate) mod health_panel;
pub(crate) mod import_panel;
pub(crate) mod lighting;
pub(crate) mod lighting_panel;
pub(crate) mod planet;
pub(crate) mod preview;
pub(crate) mod registry;
pub(crate) mod sdf;
pub(crate) mod shader;
pub(crate) mod story;
pub(crate) mod story_panel;
pub(crate) mod template;
pub(crate) mod template_panel;
pub(crate) mod variables;
pub(crate) mod variables_panel;
pub(crate) mod view;
