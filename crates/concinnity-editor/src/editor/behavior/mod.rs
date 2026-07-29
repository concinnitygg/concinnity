// src/editor/behavior/mod.rs
//
// The Behavior panel's model half: the authored args of one `Behavior` seen as
// an editable node graph. `path` addresses a place inside those args, `palette`
// holds the closed node / expression vocabulary and the JSON a fresh one starts
// as, `outline` flattens the whole asset into the panel's indented rows, `edit`
// applies what a row's controls do, `navigate` says where a step of the
// selection lands, and `fault` resolves the checker's complaint to the row it is
// about. Nothing here touches the world or the HUD: the layout half
// is `editor/behavior_panel.rs` and the actions live in `hook/behavior_edit.rs`.
//
// The panel edits the authored JSON directly rather than a typed twin, because
// that JSON is exactly what `check_with_variables` reads -- so the status line
// reports on the same value the build will.

pub(crate) mod edit;
pub(crate) mod fault;
pub(crate) mod fields;
pub(crate) mod graph;
pub(crate) mod navigate;
pub(crate) mod outline;
pub(crate) mod palette;
pub(crate) mod path;
pub(crate) mod relations;
