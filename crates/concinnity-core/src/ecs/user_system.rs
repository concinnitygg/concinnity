// src/ecs/user_system.rs
//
// A system registered on a world from outside the engine's table, and the name
// check that keeps it addressable.
//
// The schedule keys everything on the entry name -- the profile row, the log
// line, and the `after` / `before` edges the table's rows declare against each
// other. A registered system reusing a table entry's name would inherit that
// entry's edges and shadow it in every name lookup, so the merge refuses the
// collision rather than resolving it.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::ecs::{Phase, System};

// A system registered on a world, held until `start` merges it into the
// schedule at its phase.
pub(crate) struct UserSystem {
    pub(crate) phase: Phase,
    pub(crate) name: &'static str,
    pub(crate) system: Box<dyn System>,
}

impl core::fmt::Debug for UserSystem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UserSystem")
            .field("phase", &self.phase)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

// The name a registration cannot have: one a table entry already uses, or one
// an earlier registration already took. `None` when every name is distinct.
pub(crate) fn colliding_name(
    registered: &[&'static str],
    table: &[&'static str],
) -> Option<&'static str> {
    let mut seen: Vec<&'static str> = Vec::with_capacity(registered.len());
    for name in registered {
        if table.contains(name) || seen.contains(name) {
            return Some(name);
        }
        seen.push(name);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::colliding_name;

    #[test]
    fn distinct_names_pass() {
        assert_eq!(
            colliding_name(&["PlayerController", "Spawner"], &["GraphicsSystem"]),
            None
        );
    }

    // A registration reusing a table entry's name would inherit that entry's
    // ordering edges, so it is refused.
    #[test]
    fn a_table_name_collides() {
        assert_eq!(
            colliding_name(&["GraphicsSystem"], &["GraphicsSystem"]),
            Some("GraphicsSystem")
        );
    }

    // Two registrations sharing a name are equally unaddressable.
    #[test]
    fn a_repeated_registration_collides() {
        assert_eq!(
            colliding_name(&["Ai", "Ai"], &["GraphicsSystem"]),
            Some("Ai")
        );
    }

    // Nothing registered is nothing to collide with, whatever the table holds.
    #[test]
    fn no_registrations_never_collide() {
        assert_eq!(colliding_name(&[], &["GraphicsSystem"]), None);
    }
}
