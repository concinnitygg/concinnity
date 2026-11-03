//! The set of keyboard keys held down right now, as a window reports it.

use crate::components::InputKey;

const _: () = assert!(InputKey::ALL.len() <= 128, "a held-key set holds 128 keys");

/// Which [`InputKey`]s are held down. A key is in the set from its press to its
/// release, whatever it is bound to.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HeldKeys(u128);

impl HeldKeys {
    /// No key held.
    pub const NONE: Self = Self(0);

    /// Mark `key` held (`down`) or released.
    pub fn set(&mut self, key: InputKey, down: bool) {
        let bit = Self::bit(key);
        if down {
            self.0 |= bit;
        } else {
            self.0 &= !bit;
        }
    }

    /// Whether `key` is held.
    pub fn contains(self, key: InputKey) -> bool {
        self.0 & Self::bit(key) != 0
    }

    /// The set with `key` held too.
    #[must_use]
    pub fn with(mut self, key: InputKey) -> Self {
        self.set(key, true);
        self
    }

    fn bit(key: InputKey) -> u128 {
        1 << (key as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keys go in on a press and out on a release, independently of each other,
    // the last declared key included.
    #[test]
    fn keys_are_held_from_press_to_release() {
        let last = *InputKey::ALL.last().unwrap();
        let mut held = HeldKeys::NONE.with(InputKey::W).with(last);
        assert!(held.contains(InputKey::W) && held.contains(last));
        assert!(!held.contains(InputKey::A));
        held.set(InputKey::W, false);
        assert!(!held.contains(InputKey::W) && held.contains(last));
        held.set(last, false);
        assert_eq!(held, HeldKeys::NONE);
    }
}
