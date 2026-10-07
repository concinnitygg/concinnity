//! EditorHook: what each panel's actions do, one module per surface. A panel's
//! layout half reports what was clicked (`editor/*_panel.rs`); the module here
//! carries that report out. Most of them mutate the working authored entry list
//! and commit it the way every other edit commits (`hook/edits.rs`: the undo
//! snapshot, the live-preview rebuild, and SAVE's atomic write), so a panel
//! action is never a special case of persistence.
//!
//! Two of them mutate no entry and answer a console command instead: `select`
//! resolves /select into a new selection, and `export` compiles the working
//! entries to write one named mesh out as glb.
//!
//! One module per surface: `asset_tree`, `behavior`, `character_shape`,
//! `console`, `content`, `export`, `import`, `lighting`, `map`, `overrides`,
//! `palette`, `select`, `shaders` (with its `shader_edits`, `shader_form` and
//! `shader_source`), `story`, `variables` and `worlds`.
//!
//! Each panel keeps its session state in a `*_state` sibling (`assets_state`,
//! `behavior_state`, `console_state`, `content_state`, `import_state`,
//! `lighting_state`, `map_state`, `palette_state`, `shape_state`,
//! `shaders_state`, `story_state`, `templates_state`, `variables_state` and
//! `worlds_state`), a plain struct with the resets that touch nothing else.
//! Whether a panel is open is the hook's (`EditorHook::open`), not its state's.

pub(super) mod asset_tree;
pub(super) mod assets_state;
pub(super) mod behavior;
pub(super) mod behavior_state;
pub(super) mod character_shape;
pub(super) mod console;
pub(super) mod console_state;
pub(super) mod content;
pub(super) mod content_state;
pub(super) mod export;
pub(super) mod import;
pub(super) mod import_state;
pub(super) mod lighting;
pub(super) mod lighting_state;
pub(super) mod map;
pub(super) mod map_state;
pub(super) mod overrides;
pub(super) mod palette;
pub(super) mod palette_state;
pub(super) mod path_cache;
pub(super) mod sdf_edits;
pub(super) mod sdf_form;
pub(super) mod select;
pub(super) mod shader_edits;
pub(super) mod shader_form;
pub(super) mod shader_source;
pub(super) mod shaders;
pub(super) mod shaders_state;
pub(super) mod shape_state;
pub(super) mod story;
pub(super) mod story_state;
pub(super) mod templates_state;
pub(super) mod variables;
pub(super) mod variables_state;
pub(super) mod worlds;
pub(super) mod worlds_state;
