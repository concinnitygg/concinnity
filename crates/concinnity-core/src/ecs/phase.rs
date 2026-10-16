//! Where in a tick a system runs. Every table row carries one, and a system
//! registered from outside the engine names one instead of naming a table entry:
//! the table stays the one document, its entries stay internal, and its order is
//! free to change as long as each row keeps its phase.
//!
//! The merge rule is one line: phases run in declaration order, and within a
//! phase every table row runs before every registered system. Table order is
//! therefore unchanged by any registration, which is what keeps a table
//! readable as the tick.

/// Where in a tick a system runs.
///
/// A system registered with [`World::add_system`](crate::ecs::World::add_system)
/// names a phase rather than a neighboring system, and runs after every engine
/// system in that phase.
///
/// The tick simulates first and submits its frame last, so the frame shows
/// everything the tick resolved:
///
/// | Phase | Runs after | Runs before |
/// | --- | --- | --- |
/// | [`Early`](Phase::Early) | the menu gate, this tick's input and a planet world's frame move | this tick's world logic |
/// | [`Logic`](Phase::Logic) | the world's behavior bodies | the simulation |
/// | [`Late`](Phase::Late) | this tick's physics, cameras, animation, story and audio | spawns, settings and streaming |
/// | [`PreRender`](Phase::PreRender) | spawns, settings and streaming | this frame is submitted |
/// | [`PostRender`](Phase::PostRender) | this frame's submit | the next tick |
///
/// Input is sampled before the tick, so movement belongs in
/// [`Early`](Phase::Early): a system there reads this tick's input and writes
/// the intent this tick's physics consumes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Before the world's logic: this tick's input is in and nothing has been
    /// simulated yet. Where movement intent belongs.
    Early,
    /// Beside the world's logic. Spawn, despawn and scene requests made here
    /// still drain this tick.
    Logic,
    /// After this tick's physics, cameras and animation have all resolved.
    /// The default, and where most work belongs.
    #[default]
    Late,
    /// After the tick's spawns, settings and streaming, before the frame is
    /// submitted. The last place a transform reaches this frame's draw.
    PreRender,
    /// After the frame is submitted: sees the frame that just went.
    PostRender,
}

impl Phase {
    /// Every phase, in run order.
    pub const ALL: [Phase; 5] = [
        Phase::Early,
        Phase::Logic,
        Phase::Late,
        Phase::PreRender,
        Phase::PostRender,
    ];

    /// The phase name, for schedule reporting.
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Early => "Early",
            Phase::Logic => "Logic",
            Phase::Late => "Late",
            Phase::PreRender => "PreRender",
            Phase::PostRender => "PostRender",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Phase;

    // `ALL` is the run order, and it is the enum's own ordering: the merge walks
    // `ALL`, while a sort of registered systems uses `Ord`, so the two agreeing
    // is what puts a registration in the phase it named.
    #[test]
    fn all_is_sorted_and_complete() {
        let mut sorted = Phase::ALL;
        sorted.sort();
        assert_eq!(sorted, Phase::ALL);
        assert_eq!(Phase::ALL.first(), Some(&Phase::Early));
        assert_eq!(Phase::ALL.last(), Some(&Phase::PostRender));
    }

    // The default is after the simulation, where everything has resolved.
    #[test]
    fn the_default_phase_is_late() {
        assert_eq!(Phase::default(), Phase::Late);
    }

    // Every phase reports its own name, and no two share one.
    #[test]
    fn names_are_distinct() {
        let mut names: alloc::vec::Vec<&str> = Phase::ALL.iter().map(|p| p.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Phase::ALL.len());
    }
}
