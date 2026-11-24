//! Writing the editor's model into the world's UI state each frame, only where
//! it changed, so a steady frame allocates nothing.

use std::cell::RefCell;
use std::fmt::{self, Write as _};

use concinnity_engine::components::{UiState, UiValue};
use concinnity_engine::ecs::World;

thread_local! {
    // Text formatted here and compared before it is stored.
    static SCRATCH: RefCell<String> = const { RefCell::new(String::new()) };
}

// Change the world's UI state, made when the world has none.
pub(crate) fn with_state(world: &mut World, change: impl FnOnce(&mut UiState)) {
    if let Some(state) = world.resource_mut::<UiState>() {
        change(state);
    } else {
        let mut state = UiState::default();
        change(&mut state);
        world.insert_resource(state);
    }
}

// Set `slot` to `value` when it differs, so a steady frame writes nothing.
pub(crate) fn set_if_changed<T: PartialEq>(slot: &mut T, value: T) {
    if *slot != value {
        *slot = value;
    }
}

// The boolean held under `name`, false when none is.
pub(crate) fn get_bool(world: &World, name: &str) -> bool {
    world
        .resource::<UiState>()
        .is_some_and(|s| s.get(name) == Some(&UiValue::Bool(true)))
}

pub(crate) fn set_bool(world: &mut World, name: &str, value: bool) {
    with_state(world, |state| state.set(name, UiValue::Bool(value)));
}

// Hold `text` under `name`, formatted, unless it already holds that text.
pub(crate) fn set_text(world: &mut World, name: &str, text: fmt::Arguments<'_>) {
    SCRATCH.with_borrow_mut(|scratch| {
        scratch.clear();
        #[expect(
            clippy::let_underscore_must_use,
            reason = "writing to a String cannot fail"
        )]
        let _ = scratch.write_fmt(text);
        with_state(world, |state| {
            if !matches!(state.get(name), Some(UiValue::Text(t)) if t == scratch.as_str()) {
                state.set(name, UiValue::Text(scratch.clone()));
            }
        });
    });
}

// Whether `count` (a revision, or a count of samples) is the one last marked
// shown under `names`. The state holds 32-bit ints, so it is kept as its two
// halves; a world with nothing marked yet has shown nothing.
pub(crate) fn is_shown(world: &World, names: [&str; 2], count: u64) -> bool {
    let halves = halves(count);
    world.resource::<UiState>().is_some_and(|s| {
        names
            .iter()
            .zip(halves)
            .all(|(name, half)| s.get(name) == Some(&UiValue::Int(half)))
    })
}

// Mark `count` shown under `names`.
pub(crate) fn mark_shown(state: &mut UiState, names: [&str; 2], count: u64) {
    for (name, half) in names.iter().zip(halves(count)) {
        state.set(name, UiValue::Int(half));
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "each half is exactly 32 bits, kept as a bit pattern"
)]
fn halves(count: u64) -> [i32; 2] {
    [(count >> 32) as u32 as i32, count as u32 as i32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bool_reads_back_and_defaults_false() {
        let mut world = World::new();
        assert!(!get_bool(&world, "b"));
        set_bool(&mut world, "b", true);
        assert!(get_bool(&world, "b"));
    }

    #[test]
    fn a_slot_takes_only_a_different_value() {
        let mut slot = String::from("a");
        let at = slot.as_ptr();
        set_if_changed(&mut slot, String::from("a"));
        assert_eq!(slot.as_ptr(), at, "an equal value leaves the slot alone");
        set_if_changed(&mut slot, String::from("b"));
        assert_eq!(slot, "b");
    }

    #[test]
    fn a_count_is_shown_whole() {
        let mut world = World::new();
        let names = ["a.hi", "a.lo"];
        assert!(!is_shown(&world, names, 0), "nothing is shown yet");
        let big = (7_u64 << 32) | 5;
        with_state(&mut world, |s| mark_shown(s, names, big));
        assert!(is_shown(&world, names, big));
        assert!(!is_shown(&world, names, 5), "the high half counts");
        assert!(!is_shown(&world, names, big + 1));
    }

    #[test]
    fn text_is_stored_once_until_it_changes() {
        let mut world = World::new();
        let revision = |w: &World| w.resource::<UiState>().map(UiState::revision);
        set_text(&mut world, "step", format_args!("{} m", 0.5));
        let rev = revision(&world);
        set_text(&mut world, "step", format_args!("{} m", 0.5));
        assert_eq!(revision(&world), rev);
        set_text(&mut world, "step", format_args!("{} m", 1));
        assert_eq!(
            world.resource::<UiState>().unwrap().get("step"),
            Some(&UiValue::Text("1 m".into()))
        );
    }
}
