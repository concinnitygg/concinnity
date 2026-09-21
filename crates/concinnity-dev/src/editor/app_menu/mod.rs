//! The menu bar the editor installs at the top of the macOS screen: the
//! application menu, and a View menu carrying the same panel toggles as the
//! View panel. That panel stays: this is a second way into the same rows, not
//! a replacement, so a session looks the same on every platform.
//!
//! Editor-only, and macOS-only. A shipped runtime installs no menu at all.
//!
//! The split follows the rest of the editor. `spec` turns the editor's panel
//! state into a platform-neutral description of the menu -- titles,
//! checkmarks, and the tag each item carries -- which is pure data and tested
//! without a window. `menu` is the only half that names AppKit: it builds the
//! NSMenu tree once and pushes later state changes into it.

mod menu;
mod spec;

pub(crate) use menu::{install, sync, take_chosen};
pub(crate) use spec::{MenuCommand, PanelMarks};
