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

#[cfg(test)]
mod tests {
    use super::*;

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
