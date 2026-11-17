//! Authored UI: the shorthands a world declares for overlay UI, and the typed
//! expansion that turns each into the elements it stands for.
//!
//! A shorthand is written like any other asset, in world.jsonl or as a typed
//! value handed to the world builder. Either way the cook replaces it with the
//! `Screen`, `Sprite`, `TextLabel`, `HitRegion`, and `KeyBinding` entries that
//! [`Expand::expand`] returns, so the runtime only ever sees those primitives.
//!
//! [`Expand`] and [`ExpandContext`] are the expansion itself, for a tool that
//! builds on this crate and needs a shorthand's entries without a cook. An
//! application authors through the `concinnity` crate's `ui` module, which
//! carries the shorthands only.

mod elements;
mod expand;
mod main_menu;
mod menu_defaults;
mod option_select;
mod panel;
mod row_setting;
mod screen_scope;
mod slider;

pub use expand::{Expand, ExpandContext};
pub use main_menu::schema::{MainMenu, MainMenuItem, MenuItemAction, SettingsProfile};
pub use option_select::schema::OptionSelect;
pub use panel::schema::Panel;
pub use slider::schema::Slider;

pub(crate) use expand::expand_all;
pub(crate) use main_menu::{generates_settings, item_action, settings_entry_screen};
pub(crate) use menu_defaults::inject_menu_defaults;
pub(crate) use screen_scope::scope_entries_to_screen;
