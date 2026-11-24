// What the settings menu shows for each setting, published for the overlay to
// read through `setting:` bindings.

use alloc::string::String;
use hashbrown::HashMap;

use crate::settings::SettingKey;

/// What each setting shows: the display text of its current value, and a
/// slider's position along its range, 0 to 1. Republished by the settings
/// system whenever a value changes, for labels bound to `setting:` values and
/// for slider handles to read.
#[derive(Debug, Clone, Default)]
pub struct SettingValues {
    texts: HashMap<SettingKey, String>,
    fractions: HashMap<SettingKey, f32>,
    revision: u64,
}

impl SettingValues {
    /// The display text of `key`'s current value.
    pub fn text(&self, key: SettingKey) -> Option<&str> {
        self.texts.get(&key).map(String::as_str)
    }

    /// Where slider `key` sits along its range, 0 to 1.
    pub fn fraction(&self, key: SettingKey) -> Option<f32> {
        self.fractions.get(&key).copied()
    }

    /// Show `text` for `key`, reusing the text held for it.
    pub fn set_text(&mut self, key: SettingKey, text: &str) {
        let held = self.texts.entry(key).or_default();
        if held != text {
            held.clear();
            held.push_str(text);
            self.revision = self.revision.wrapping_add(1);
        }
    }

    /// Place slider `key` at `fraction` of its range.
    pub fn set_fraction(&mut self, key: SettingKey, fraction: f32) {
        let fraction = fraction.clamp(0.0, 1.0);
        if self.fractions.insert(key, fraction) != Some(fraction) {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    /// A count that changes whenever a text or position does.
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_real_changes_move_the_revision() {
        let mut values = SettingValues::default();
        values.set_text(SettingKey::Vsync, "On");
        values.set_fraction(SettingKey::Exposure, 2.0);
        let after = values.revision();
        assert_eq!(values.text(SettingKey::Vsync), Some("On"));
        assert_eq!(values.fraction(SettingKey::Exposure), Some(1.0), "clamped");
        values.set_text(SettingKey::Vsync, "On");
        values.set_fraction(SettingKey::Exposure, 1.0);
        assert_eq!(values.revision(), after);
        values.set_text(SettingKey::Vsync, "Off");
        assert_ne!(values.revision(), after);
    }
}
