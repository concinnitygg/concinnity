//! The action vocabulary as a world authors it.

#[cfg(feature = "schema")]
mod schema;

use serde::{Deserialize, Serialize};

use crate::components::{Scene, Screen, SettingVerb, StoryCommand};
use crate::ecs::NameRef;
use crate::ecs::asset_fields::{
    ActionField, AssetFields, FieldTable, RefField, ReferenceField, join,
};
use crate::settings::SettingKey;

/// What a [HitRegion](#hitregion) click or a [KeyBinding](#keybinding) press
/// does, as a world authors it.
///
/// An action that names nothing is a bare string; one that does is an object
/// with the action as its only key:
/// - `"quit"`: stop the application
/// - `{"scene": "<name>"}`: jump to the named [Scene](#scene)
/// - `{"show": "<name>"}`: show the named [Screen](#screen), replacing the top of the stack
/// - `{"push": "<name>"}`: open the named [Screen](#screen) on top of what is showing
/// - `{"toggle": "<name>"}`: close the named [Screen](#screen) if it is on top, open it otherwise
/// - `"hide"`: close the top [Screen](#screen)
/// - `{"story": "<verb>"}`: drive the story (`start`, `continue`, `advance`,
///   `auto`, `skip`, `log`, `save`, `load`, `pause`, `settings`,
///   `settings_back`), or `{"story": {"choose": <i>}}` / `{"story": {"slot": <i>}}`
///
/// Generated settings menus also emit `{"group_toggle": <i>}` and
/// `{"setting": {"key": "<setting>", "verb": "<verb>"}}`.
///
/// `C` and `S` are how the action names its [Scene](#scene) and
/// [Screen](#screen): by `$id` while a build reads it ([`NamedAction`]), or by
/// resolved id at runtime (a [`UiAction`](crate::components::UiAction)).
///
/// ```rust
/// # use concinnity_core::components::{AuthoredAction, NamedAction};
/// let action: NamedAction = serde_json::from_value(serde_json::json!({"show": "pause"})).unwrap();
/// assert_eq!(action, AuthoredAction::Show("pause".into()));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredAction<C, S> {
    /// Stop the application.
    Quit,
    /// Change to a scene, dismissing every open screen.
    Scene(C),
    /// Show a screen, replacing the top of the stack.
    Show(S),
    /// Open a screen on top of what is showing.
    Push(S),
    /// Close a screen if it is on top, open it otherwise.
    Toggle(S),
    /// Close the top screen.
    Hide,
    /// Drive the story system.
    Story(StoryCommand),
    /// Expand or collapse a settings-screen group by index.
    GroupToggle(usize),
    /// Operate a settings row.
    Setting {
        /// The setting the row edits.
        key: SettingKey,
        /// What the row does to it.
        verb: SettingVerb,
    },
}

/// An action naming its targets by `$id`: what a build-time shorthand holds.
pub type NamedAction = AuthoredAction<NameRef<Scene>, NameRef<Screen>>;

impl<C: ReferenceField, S: ReferenceField> AuthoredAction<C, S> {
    /// Record an action field at `prefix` that also takes the bare names
    /// `extras`, with the targets it may name as its reference fields.
    pub fn collect_action_fields(
        prefix: &str,
        extras: &'static [&'static str],
        out: &mut FieldTable,
    ) {
        out.actions.push(ActionField {
            path: join(prefix, None),
            extras,
        });
        out.refs.push(RefField {
            path: join(prefix, Some("scene")),
            targets: C::TARGETS,
        });
        for verb in ["show", "push", "toggle"] {
            out.refs.push(RefField {
                path: join(prefix, Some(verb)),
                targets: S::TARGETS,
            });
        }
    }
}

impl<C: ReferenceField, S: ReferenceField> AssetFields for AuthoredAction<C, S> {
    fn collect_fields(prefix: &str, out: &mut FieldTable) {
        Self::collect_action_fields(prefix, &[], out);
    }
}

#[cfg(test)]
mod tests;
