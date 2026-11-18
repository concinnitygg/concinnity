//! The editor's panels built from engine widgets. Each is a floating `Layout`
//! on the editor's screen, built once when a world is injected: the engine lays
//! it out, draws it, drags and sizes it, and drives its widgets, which read
//! the editor's model from `UiState` and report what the user did as
//! `UiEvent`s the hook handles.

pub(crate) mod build;
mod built;
pub(crate) mod field;
pub(crate) mod rows;
pub(crate) mod shapes;
pub(crate) mod state;

pub(crate) use build::{PanelSpec, ROW_H, UiBuilder, panel_height};
pub(crate) use built::{BuiltPanel, BuiltUi};

// The event a panel's close button sends, carrying the panel's registry index.
pub(crate) const CLOSE_EVENT: &str = "editor.close";
