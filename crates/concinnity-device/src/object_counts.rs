//! The scene-size counters every backend publishes in `RenderStats`, derived
//! from plain counts so `objects` and `skinned_visible` mean the same thing on
//! each.

/// Scene size for one frame's render stats.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObjectCounts {
    /// Static draw objects, every instanced-cluster instance, and every skinned
    /// slot, hidden pool slots included, so it stays flat across skinned
    /// spawn and despawn.
    pub(crate) objects: u32,
    /// Skinned slots currently visible: authored meshes plus runtime-spawned
    /// instances, excluding the hidden pre-reserved pool slots.
    pub(crate) skinned_visible: u32,
}

/// Count a frame's scene from its static object count, each instanced cluster's
/// instance count, and each skinned slot's visibility.
pub(crate) fn object_counts(
    static_objects: usize,
    cluster_instances: impl IntoIterator<Item = usize>,
    skinned_visibility: impl IntoIterator<Item = bool>,
) -> ObjectCounts {
    let instances: usize = cluster_instances.into_iter().sum();
    let (skinned_slots, skinned_visible) = skinned_visibility
        .into_iter()
        .fold((0usize, 0usize), |(slots, visible), v| {
            (slots + 1, visible + usize::from(v))
        });
    ObjectCounts {
        objects: (static_objects + instances + skinned_slots) as u32,
        skinned_visible: skinned_visible as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_world_counts_nothing() {
        assert_eq!(object_counts(0, [], []), ObjectCounts::default());
    }

    #[test]
    fn clusters_add_every_instance() {
        let counts = object_counts(3, [4, 0, 2], []);
        assert_eq!(
            counts,
            ObjectCounts {
                objects: 9,
                skinned_visible: 0
            }
        );
    }

    #[test]
    fn hidden_pool_slots_count_as_objects_but_not_visible() {
        let counts = object_counts(1, [2], [true, false, false, true, false]);
        assert_eq!(
            counts,
            ObjectCounts {
                objects: 8,
                skinned_visible: 2
            }
        );
    }
}
