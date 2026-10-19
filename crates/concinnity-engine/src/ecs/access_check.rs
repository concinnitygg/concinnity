//! The debug-build access validator: installs the core `access_check` hooks,
//! keeping the stepping system's declared access per thread (core is no_std and
//! holds no per-thread state). `World::step` announces the stepping system
//! through one hook; every context accessor's touch is verified against it
//! through the other.
#![cfg(debug_assertions)]

use concinnity_core::ecs::Access;
use concinnity_core::ecs::access_check::{self, Touch};

std::thread_local! {
    static ACTIVE: std::cell::Cell<Option<(Access, &'static str)>> =
        const { std::cell::Cell::new(None) };
}

// Mark the given system's declared access active on this thread for the
// duration of its step. `None` clears it (init, decompose, and editor drives
// run unvalidated).
fn set_active(active: Option<(Access, &'static str)>) {
    ACTIVE.with(|a| a.set(active));
}

pub(crate) fn install_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        access_check::install(check);
        access_check::install_active(set_active);
    });
}

fn check(touch: &Touch) {
    if let Some((access, system)) = ACTIVE.with(|a| a.get()) {
        access_check::verify(access, system, touch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::Sprite;
    use concinnity_core::components::TextLabel;
    use concinnity_core::ecs::access_check::set_active;
    use concinnity_core::ecs::{Access, MenuActive, World};
    use concinnity_core::{component_mask, resource_mask};

    #[test]
    #[should_panic(expected = "Sprite without declaring it")]
    fn undeclared_component_read_panics() {
        install_hook();
        set_active(Some((
            Access::new().writes_components(component_mask![TextLabel]),
            "TestSystem",
        )));
        let mut world = World::new();
        let _ = world.context().query::<Sprite>().count();
    }

    #[test]
    #[should_panic(expected = "structural change")]
    fn undeclared_structural_change_panics() {
        install_hook();
        set_active(Some((Access::new(), "TestSystem")));
        let mut world = World::new();
        world.context().push(TextLabel::default());
    }

    #[test]
    #[should_panic(expected = "not in the access-id registry")]
    fn unregistered_resource_touch_panics() {
        install_hook();
        set_active(Some((Access::new(), "TestSystem")));
        struct Unregistered;
        let mut world = World::new();
        let _ = world.context().resource::<Unregistered>();
    }

    #[test]
    fn declared_touches_pass_and_exclusive_passes_everything() {
        install_hook();
        let mut world = World::new();
        set_active(Some((
            Access::new()
                .writes_components(component_mask![TextLabel])
                .reads_resources(resource_mask![MenuActive]),
            "TestSystem",
        )));
        {
            let mut c = world.context();
            let _ = c.query::<TextLabel>().count();
            let _ = c.query_mut::<TextLabel>().count();
            let _ = c.resource::<MenuActive>();
        }
        set_active(Some((Access::new().exclusive(), "TestSystem")));
        {
            let mut c = world.context();
            c.push(TextLabel::default());
            let _ = c.query::<Sprite>().count();
        }
        set_active(None);
    }
}
