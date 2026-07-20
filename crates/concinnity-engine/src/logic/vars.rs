// Shared integer variable store for declarative logic.
//
// A type-keyed resource inserted by ReactionSystem's init. Reaction conditions
// read it and the `set` action writes it. An unset variable reads as 0, so a
// plain flag is a variable holding 1. Deliberately separate from the story
// system's variables, whose reset-on-start and per-slot save semantics belong
// to a playthrough, not the world.

use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Variables {
    values: BTreeMap<String, i32>,
}

impl Variables {
    pub fn get(&self, name: &str) -> i32 {
        self.values.get(name).copied().unwrap_or(0)
    }

    // Assign `value`, or add it to the current value when `add` is true.
    pub fn apply(&mut self, name: &str, value: i32, add: bool) {
        let slot = self.values.entry(name.to_string()).or_insert(0);
        *slot = if add {
            slot.saturating_add(value)
        } else {
            value
        };
    }
}
