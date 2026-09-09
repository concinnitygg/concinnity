// src/ecs/phase.rs
//
// Where in a tick a system runs. Every table row carries one, and a system
// registered from outside the engine names one instead of naming a table entry:
// the table stays the one document, its entries stay internal, and its order is
// free to change as long as each row keeps its phase.
//
// The merge rule is one line: phases run in declaration order, and within a
// phase every table row runs before every registered system. Table order is
// therefore unchanged by any registration, which is what keeps a table
// readable as the tick.

/// Where in a tick a system runs.
///
/// A system registered with [`World::add_system`](crate::ecs::World::add_system)
/// names a phase rather than a neighbouring system, and runs after every engine
/// system in that phase.
///
/// The engine submits its frame partway through the tick, so the phases are not
/// symmetric around it:
///
/// | Phase | Runs after | Runs before |
/// | --- | --- | --- |
/// | [`Early`](Phase::Early) | the previous tick, in full | this tick's world logic |
/// | [`Logic`](Phase::Logic) | the world's behaviour bodies | the requests they emit drain |
/// | [`PreRender`](Phase::PreRender) | spawns, settings and streaming | this frame is submitted |
/// | [`Late`](Phase::Late) | the frame, this tick's input, physics, cameras, animation | the next tick |
///
/// Input is sampled after the frame is submitted, so movement belongs in
/// [`Early`](Phase::Early): a system there reads the input sampled at the end of
/// the previous tick and writes the intent this tick's physics consumes. That is
/// the same one-tick lag the engine's own camera controllers run at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Before the world's logic: the previous tick has fully resolved and
    /// nothing has been simulated yet. Where movement intent belongs.
    Early,
    /// Beside the world's logic. Spawn, despawn and scene requests made here
    /// still drain this tick.
    Logic,
    /// After the tick's spawns, settings and streaming, before the frame is
    /// submitted. The last place a transform reaches this frame's draw.
    PreRender,
    /// After the frame: this tick's input, physics, cameras and animation have
    /// all resolved. The default, and where most work belongs.
    #[default]
    Late,
}

impl Phase {
    /// Every phase, in run order.
    pub const ALL: [Phase; 4] = [Phase::Early, Phase::Logic, Phase::PreRender, Phase::Late];

    /// The phase name, for schedule reporting.
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Early => "Early",
            Phase::Logic => "Logic",
            Phase::PreRender => "PreRender",
            Phase::Late => "Late",
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
        assert_eq!(Phase::ALL.last(), Some(&Phase::Late));
    }

    // The default is the end of the tick, where everything has resolved.
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
