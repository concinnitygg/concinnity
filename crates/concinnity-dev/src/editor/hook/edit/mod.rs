// src/editor/hook/edit/mod.rs
//
// EditorHook: what each panel's actions do, one module per surface. A panel's
// layout half reports what was clicked (`editor/*_panel.rs`); the module here
// carries that report out. Most of them mutate the working authored entry list
// and commit it the way every other edit commits (`hook/edits.rs`: the undo
// snapshot, the live-preview rebuild, and SAVE's atomic write), so a panel
// action is never a special case of persistence.
//
// Two of them mutate no entry and answer a console command instead: `select`
// resolves /select into a new selection, and `export` compiles the working
// entries to write one named mesh out as glb.
//
// One module per surface: `asset_tree`, `behavior`, `character_shape`,
// `console`, `content`, `export`, `import`, `lighting`, `overrides`,
// `palette`, `select`, `story`, `variables` and `worlds`.

pub(super) mod asset_tree;
pub(super) mod behavior;
pub(super) mod character_shape;
pub(super) mod console;
pub(super) mod content;
pub(super) mod export;
pub(super) mod import;
pub(super) mod lighting;
pub(super) mod overrides;
pub(super) mod palette;
pub(super) mod select;
pub(super) mod story;
pub(super) mod variables;
pub(super) mod worlds;
