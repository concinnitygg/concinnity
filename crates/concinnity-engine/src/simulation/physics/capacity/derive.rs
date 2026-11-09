// How much a world's physics is allowed to cost, derived from what the world
// actually contains. Cook counts the authored content and the runtime counts
// the loaded components; both feed the same derivation, so the number the
// simulation reserves is the number cook promised.
//
// The split between counts and budget is deliberate: counts are what a world
// has, the budget is what the simulation reserves for it. Only the second is
// worth shipping, and only the first is worth comparing across the two sides.

/// Tallies of a world's authored physics content.
///
/// Both cook (over the authored asset list) and the runtime (over the loaded
/// component columns) produce one of these, which is what lets the two sides
/// be compared for agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhysicsCounts {
    /// Collider-bearing entities with no dynamics: walls, scenery, floors.
    pub static_colliders: u32,
    /// Collider-bearing entities that are freely simulated.
    pub dynamic_colliders: u32,
    /// Sensor regions that report crossings but never collide.
    pub trigger_volumes: u32,
    /// Joints whose bodies both resolve to collider-bearing entities.
    pub joints: u32,
    /// The subset of `joints` anchored to the world rather than a second body.
    pub world_anchored_joints: u32,
    /// The player capsule: `1` when the world declares one, else `0`.
    pub player_capsules: u32,
    /// Kinematic capsules driven by root motion, one per character rig.
    pub rig_capsules: u32,
}

/// The reservation a world's physics content implies.
///
/// Grouped by the kind of body the simulation builds rather than by the asset
/// that asked for it, because that is what the simulation sizes its storage
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhysicsBudget {
    /// Immovable bodies: static colliders plus the world's floor.
    pub fixed: u32,
    /// Freely simulated bodies.
    pub dynamic: u32,
    /// Position-driven bodies: the player capsule and the character rigs.
    pub kinematic: u32,
    /// Sensor bodies, one per trigger volume.
    pub sensors: u32,
    /// Joints connecting two bodies.
    pub joints: u32,
    /// Hidden static bodies minted to anchor a world-anchored joint.
    pub anchors: u32,
    /// Bodies held back for entities created after init.
    pub spawn_headroom: u32,
}

impl PhysicsBudget {
    /// Derive the reservation for a world's `counts`, holding back
    /// `spawn_headroom` bodies for entities created at runtime.
    ///
    /// The floor body is always built, so `fixed` is one past the static
    /// collider count.
    pub const fn derive(counts: &PhysicsCounts, spawn_headroom: u32) -> Self {
        Self {
            fixed: counts.static_colliders.saturating_add(1),
            dynamic: counts.dynamic_colliders,
            kinematic: counts.player_capsules.saturating_add(counts.rig_capsules),
            sensors: counts.trigger_volumes,
            joints: counts.joints,
            anchors: counts.world_anchored_joints,
            spawn_headroom,
        }
    }

    /// Bodies the authored world builds at init.
    pub const fn body_total(&self) -> u32 {
        self.fixed
            .saturating_add(self.dynamic)
            .saturating_add(self.kinematic)
            .saturating_add(self.sensors)
            .saturating_add(self.anchors)
    }

    /// The hard ceiling on live bodies: everything init builds, plus the
    /// headroom held back for runtime spawns. The simulation refuses to build
    /// a body past this.
    pub const fn body_cap(&self) -> u32 {
        self.body_total().saturating_add(self.spawn_headroom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_world_still_reserves_its_floor() {
        let budget = PhysicsBudget::derive(&PhysicsCounts::default(), 0);
        assert_eq!(budget.fixed, 1, "the floor body is always built");
        assert_eq!(budget.body_total(), 1);
        assert_eq!(budget.body_cap(), 1);
    }

    #[test]
    fn each_count_lands_in_its_own_category() {
        let counts = PhysicsCounts {
            static_colliders: 4,
            dynamic_colliders: 3,
            trigger_volumes: 2,
            joints: 5,
            world_anchored_joints: 2,
            player_capsules: 1,
            rig_capsules: 6,
        };
        let budget = PhysicsBudget::derive(&counts, 0);
        assert_eq!(budget.fixed, 5, "4 static colliders plus the floor");
        assert_eq!(budget.dynamic, 3);
        assert_eq!(budget.kinematic, 7, "1 player capsule plus 6 rig capsules");
        assert_eq!(budget.sensors, 2);
        assert_eq!(budget.joints, 5);
        assert_eq!(budget.anchors, 2);
        // Joints constrain bodies rather than being ones, so they stay out of
        // the total; their hidden anchors do not.
        assert_eq!(budget.body_total(), 5 + 3 + 7 + 2 + 2);
    }

    #[test]
    fn headroom_lifts_the_cap_but_not_the_total() {
        let counts = PhysicsCounts {
            static_colliders: 2,
            ..PhysicsCounts::default()
        };
        let budget = PhysicsBudget::derive(&counts, 32);
        assert_eq!(budget.body_total(), 3);
        assert_eq!(budget.body_cap(), 35);
    }

    #[test]
    fn derivation_is_the_same_from_either_side() {
        // The property the runtime debug-asserts: identical counts must give
        // an identical budget, whoever counted them.
        let counts = PhysicsCounts {
            static_colliders: 9,
            dynamic_colliders: 4,
            rig_capsules: 2,
            ..PhysicsCounts::default()
        };
        assert_eq!(
            PhysicsBudget::derive(&counts, 8),
            PhysicsBudget::derive(&counts, 8)
        );
    }

    #[test]
    fn absurd_counts_saturate_instead_of_wrapping() {
        let counts = PhysicsCounts {
            static_colliders: u32::MAX,
            dynamic_colliders: u32::MAX,
            ..PhysicsCounts::default()
        };
        let budget = PhysicsBudget::derive(&counts, u32::MAX);
        assert_eq!(budget.fixed, u32::MAX);
        assert_eq!(budget.body_cap(), u32::MAX);
    }
}
