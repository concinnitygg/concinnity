//! The runtime, rebindable key map for the gameplay movement keys. Each backend
//! decodes physical keys into the same semantic booleans (forward, jump, ...);
//! this map says which canonical InputKey drives each action, so the settings menu can
//! remap them at runtime. The map is canonical (backend-agnostic InputKey values); a
//! backend resolves it to its own native key codes when it is pushed via
//! `RenderBackend::set_keymap`.

use crate::components::InputKey;
use serde::{Deserialize, Serialize};

/// A rebindable gameplay action. The four movement directions, sprint, jump, and
/// interact. Pause (Escape) is deliberately not here: it carries cursor-release /
/// menu semantics that are fixed per-backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bindable {
    /// Move forward.
    Forward,
    /// Move backward.
    Backward,
    /// Strafe left.
    Left,
    /// Strafe right.
    Right,
    /// Hold to move faster.
    Sprint,
    /// Jump.
    Jump,
    /// Interact with the entity under the cursor.
    Interact,
}

impl Bindable {
    /// Every rebindable action, in Controls-tab row order.
    pub const ALL: [Bindable; 7] = [
        Bindable::Forward,
        Bindable::Backward,
        Bindable::Left,
        Bindable::Right,
        Bindable::Sprint,
        Bindable::Jump,
        Bindable::Interact,
    ];

    /// The settings key string used in `setting:<key>:rebind` actions and the
    /// engine settings registry.
    pub fn setting_key(self) -> &'static str {
        match self {
            Bindable::Forward => "key_forward",
            Bindable::Backward => "key_backward",
            Bindable::Left => "key_left",
            Bindable::Right => "key_right",
            Bindable::Sprint => "key_sprint",
            Bindable::Jump => "key_jump",
            Bindable::Interact => "key_interact",
        }
    }

    /// The action for a settings key string, or `None` if it is not a rebind key.
    pub fn from_setting_key(key: &str) -> Option<Bindable> {
        Bindable::ALL.into_iter().find(|b| b.setting_key() == key)
    }
}

/// The canonical action -> key map. Persisted in `ControlsSettings` and pushed to
/// the active backend. Each field is `#[serde(default)]` so adding an action in a
/// future build never invalidates an existing settings file (a missing field
/// falls back to its default rather than failing the whole load).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyMap {
    #[serde(default = "def_forward")]
    /// InputKey bound to [`Bindable::Forward`].
    pub forward: InputKey,
    #[serde(default = "def_backward")]
    /// InputKey bound to [`Bindable::Backward`].
    pub backward: InputKey,
    #[serde(default = "def_left")]
    /// InputKey bound to [`Bindable::Left`].
    pub left: InputKey,
    #[serde(default = "def_right")]
    /// InputKey bound to [`Bindable::Right`].
    pub right: InputKey,
    #[serde(default = "def_sprint")]
    /// InputKey bound to [`Bindable::Sprint`].
    pub sprint: InputKey,
    #[serde(default = "def_jump")]
    /// InputKey bound to [`Bindable::Jump`].
    pub jump: InputKey,
    #[serde(default = "def_interact")]
    /// InputKey bound to [`Bindable::Interact`].
    pub interact: InputKey,
}

impl KeyMap {
    /// The default bindings: the keys that were hardcoded before rebinding.
    pub const DEFAULT: KeyMap = KeyMap {
        forward: InputKey::W,
        backward: InputKey::S,
        left: InputKey::A,
        right: InputKey::D,
        sprint: InputKey::Shift,
        jump: InputKey::Space,
        interact: InputKey::E,
    };

    /// The key currently bound to an action.
    pub fn get(self, action: Bindable) -> InputKey {
        match action {
            Bindable::Forward => self.forward,
            Bindable::Backward => self.backward,
            Bindable::Left => self.left,
            Bindable::Right => self.right,
            Bindable::Sprint => self.sprint,
            Bindable::Jump => self.jump,
            Bindable::Interact => self.interact,
        }
    }

    /// Bind an action to a key directly (no conflict handling).
    pub fn set(&mut self, action: Bindable, key: InputKey) {
        match action {
            Bindable::Forward => self.forward = key,
            Bindable::Backward => self.backward = key,
            Bindable::Left => self.left = key,
            Bindable::Right => self.right = key,
            Bindable::Sprint => self.sprint = key,
            Bindable::Jump => self.jump = key,
            Bindable::Interact => self.interact = key,
        }
    }

    /// The action a key is bound to, or `None` if unbound. The map keeps each key
    /// bound to at most one action (the invariant `rebind` maintains), so this is
    /// the unique holder.
    pub fn action_for_key(self, key: InputKey) -> Option<Bindable> {
        Bindable::ALL.into_iter().find(|&b| self.get(b) == key)
    }

    /// Bind `action` to `new_key`, swapping with whichever action already holds
    /// `new_key` so every action stays bound. Rebinding an action to its own key
    /// is a no-op.
    pub fn rebind(&mut self, action: Bindable, new_key: InputKey) {
        let old_key = self.get(action);
        if old_key == new_key {
            return;
        }
        if let Some(other) = self.action_for_key(new_key)
            && other != action
        {
            self.set(other, old_key);
        }
        self.set(action, new_key);
    }
}

impl Default for KeyMap {
    fn default() -> Self {
        Self::DEFAULT
    }
}

fn def_forward() -> InputKey {
    KeyMap::DEFAULT.forward
}
fn def_backward() -> InputKey {
    KeyMap::DEFAULT.backward
}
fn def_left() -> InputKey {
    KeyMap::DEFAULT.left
}
fn def_right() -> InputKey {
    KeyMap::DEFAULT.right
}
fn def_sprint() -> InputKey {
    KeyMap::DEFAULT.sprint
}
fn def_jump() -> InputKey {
    KeyMap::DEFAULT.jump
}
fn def_interact() -> InputKey {
    KeyMap::DEFAULT.interact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_wasd_shift_space_e() {
        let m = KeyMap::default();
        assert_eq!(m.forward, InputKey::W);
        assert_eq!(m.backward, InputKey::S);
        assert_eq!(m.left, InputKey::A);
        assert_eq!(m.right, InputKey::D);
        assert_eq!(m.sprint, InputKey::Shift);
        assert_eq!(m.jump, InputKey::Space);
        assert_eq!(m.interact, InputKey::E);
    }

    #[test]
    fn setting_key_round_trips() {
        for b in Bindable::ALL {
            assert_eq!(Bindable::from_setting_key(b.setting_key()), Some(b));
        }
        assert_eq!(Bindable::from_setting_key("vsync"), None);
        assert_eq!(Bindable::from_setting_key("key_nope"), None);
    }

    #[test]
    fn get_set_round_trip() {
        let mut m = KeyMap::default();
        m.set(Bindable::Forward, InputKey::Up);
        assert_eq!(m.get(Bindable::Forward), InputKey::Up);
    }

    #[test]
    fn set_get_cover_every_action_arm() {
        // Drive set + get across all seven actions with distinct keys, hitting
        // every match arm in both methods.
        let keys = [
            InputKey::Up,
            InputKey::Down,
            InputKey::Left,
            InputKey::Right,
            InputKey::Q,
            InputKey::R,
            InputKey::T,
        ];
        let mut m = KeyMap::default();
        for (b, k) in Bindable::ALL.into_iter().zip(keys) {
            m.set(b, k);
        }
        for (b, k) in Bindable::ALL.into_iter().zip(keys) {
            assert_eq!(m.get(b), k);
        }
    }

    #[test]
    fn empty_cbor_map_uses_all_defaults() {
        // A settings file predating every field (an empty map) still loads, each
        // field falling back through its `serde(default = "def_*")` helper.
        let empty: std::collections::BTreeMap<String, InputKey> = std::collections::BTreeMap::new();
        let mut bytes = Vec::new();
        ciborium::into_writer(&empty, &mut bytes).unwrap();
        let loaded: KeyMap = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded, KeyMap::DEFAULT);
    }

    #[test]
    fn action_for_key_finds_the_holder() {
        let m = KeyMap::default();
        assert_eq!(m.action_for_key(InputKey::W), Some(Bindable::Forward));
        assert_eq!(m.action_for_key(InputKey::Space), Some(Bindable::Jump));
        // A key bound to nothing.
        assert_eq!(m.action_for_key(InputKey::Q), None);
    }

    #[test]
    fn rebind_to_free_key_just_sets_it() {
        let mut m = KeyMap::default();
        m.rebind(Bindable::Forward, InputKey::Q);
        assert_eq!(m.forward, InputKey::Q);
        // The others are untouched.
        assert_eq!(m.backward, InputKey::S);
    }

    #[test]
    fn rebind_to_own_key_is_a_noop() {
        let mut m = KeyMap::default();
        m.rebind(Bindable::Forward, InputKey::W);
        assert_eq!(m, KeyMap::default());
    }

    #[test]
    fn rebind_to_occupied_key_swaps() {
        // Bind Forward to S, which Backward holds: they swap, so Backward
        // inherits Forward's old key (W) and every action stays bound.
        let mut m = KeyMap::default();
        m.rebind(Bindable::Forward, InputKey::S);
        assert_eq!(m.forward, InputKey::S);
        assert_eq!(m.backward, InputKey::W);
        // No key is bound twice.
        for b in Bindable::ALL {
            assert_eq!(m.action_for_key(m.get(b)), Some(b));
        }
    }

    #[test]
    fn cbor_round_trip_and_missing_field_defaults() {
        // A full map survives a CBOR round trip.
        let m = KeyMap {
            forward: InputKey::Up,
            ..KeyMap::default()
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&m, &mut bytes).unwrap();
        let back: KeyMap = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, m);

        // A map written without one field (an older build) still loads, the
        // missing field falling back to its default rather than failing.
        #[derive(Serialize)]
        struct Partial {
            forward: InputKey,
            backward: InputKey,
            left: InputKey,
            right: InputKey,
            sprint: InputKey,
            jump: InputKey,
            // `interact` omitted.
        }
        let partial = Partial {
            forward: InputKey::Up,
            backward: InputKey::S,
            left: InputKey::A,
            right: InputKey::D,
            sprint: InputKey::Shift,
            jump: InputKey::Space,
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&partial, &mut bytes).unwrap();
        let loaded: KeyMap = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded.forward, InputKey::Up);
        assert_eq!(loaded.interact, InputKey::E);
    }
}
