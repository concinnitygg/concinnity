// The app's UI state: named values and lists the overlay reads and writes, and
// the app's own systems fill.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::components::UiValue;

/// One row of a list in the app's [`UiState`]: what a [List](#list) draws for
/// it, and where it sits in a tree.
///
/// A tree is a list whose items carry a `depth`: the app flattens it, listing
/// an expanded item's children right after it one level deeper, and leaves a
/// collapsed item's children out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UiItem {
    /// The row's text.
    pub text: String,
    /// Dimmer text drawn at the row's right end, such as a count or a type.
    pub detail: String,
    /// How many levels deep the item sits in a tree: 0 at the top.
    pub depth: u16,
    /// Whether the item has children to show, so its row draws an expander.
    pub expandable: bool,
    /// Whether its children are listed after it.
    pub expanded: bool,
    /// Whether the row draws dimmed, such as an item hidden or unavailable.
    pub dimmed: bool,
    /// A color the row's text draws in, in place of the list's, such as a log
    /// line's severity.
    pub color: Option<[f32; 3]>,
}

impl UiItem {
    /// A plain row showing `text`.
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            ..Self::default()
        }
    }
}

/// The app's UI state: named values the overlay reads and writes through
/// `state:` bindings, and named lists a [List](#list) shows, which the app's
/// own systems read and write directly. Kept apart from the world's variables.
#[derive(Debug, Clone, Default)]
pub struct UiState {
    values: BTreeMap<String, UiValue>,
    lists: BTreeMap<String, Vec<UiItem>>,
    revision: u64,
}

impl UiState {
    /// The value held under `name`.
    pub fn get(&self, name: &str) -> Option<&UiValue> {
        self.values.get(name)
    }

    /// Hold `value` under `name`.
    pub fn set(&mut self, name: &str, value: UiValue) {
        if self.values.get(name) != Some(&value) {
            self.values.insert(name.to_string(), value);
            self.bump();
        }
    }

    /// Hold `value` under `name` unless something already is.
    pub fn seed(&mut self, name: &str, value: UiValue) {
        if !self.values.contains_key(name) {
            self.set(name, value);
        }
    }

    /// The items of list `name`, empty when the app has filled none.
    pub fn list(&self, name: &str) -> &[UiItem] {
        self.lists.get(name).map_or(&[], Vec::as_slice)
    }

    /// Replace list `name` with `items`.
    pub fn set_list(&mut self, name: &str, items: Vec<UiItem>) {
        if self.lists.get(name).map(Vec::as_slice) != Some(items.as_slice()) {
            self.lists.insert(name.to_string(), items);
            self.bump();
        }
    }

    /// Change list `name` in place, creating it empty when absent. Counts as a
    /// change whatever `edit` does.
    pub fn edit_list(&mut self, name: &str, edit: impl FnOnce(&mut Vec<UiItem>)) {
        if !self.lists.contains_key(name) {
            self.lists.insert(name.to_string(), Vec::new());
        }
        if let Some(items) = self.lists.get_mut(name) {
            edit(items);
        }
        self.bump();
    }

    /// A count that changes whenever a value or list does, so a reader can
    /// skip rereading an unchanged state.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn bump(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn state_counts_only_real_changes() {
        let mut state = UiState::default();
        state.set("a", UiValue::Bool(true));
        let after_set = state.revision();
        state.set("a", UiValue::Bool(true));
        assert_eq!(state.revision(), after_set, "an unchanged set");
        state.seed("a", UiValue::Bool(false));
        assert_eq!(
            state.get("a"),
            Some(&UiValue::Bool(true)),
            "seed keeps a held value"
        );
        state.seed("b", UiValue::Int(1));
        assert_eq!(state.get("b"), Some(&UiValue::Int(1)));
    }

    #[test]
    fn lists_are_filled_whole_or_edited_in_place() {
        let mut state = UiState::default();
        assert!(state.list("files").is_empty());
        state.set_list("files", vec![UiItem::new("a"), UiItem::new("b")]);
        let filled = state.revision();
        state.set_list("files", vec![UiItem::new("a"), UiItem::new("b")]);
        assert_eq!(state.revision(), filled, "the same items");
        state.edit_list("files", |items| items[1].expanded = true);
        assert_ne!(state.revision(), filled);
        assert!(state.list("files")[1].expanded);
        state.edit_list("new", |items| items.push(UiItem::new("c")));
        assert_eq!(state.list("new").len(), 1);
    }
}
