// The handle a simulation addresses its bodies by. A slot index paired with a
// generation, packed into one word: the generation is what makes a handle to a
// removed body read as absent rather than silently addressing whatever took
// its slot.
//
// Packed index-major so the derived `Ord` orders by slot, giving callers a
// stable key for a body pair without reaching into the body storage.

/// Opaque handle to a body inside a simulation.
///
/// The simulation maps it onto its own storage; callers treat it as an opaque
/// key. Handles order deterministically, so a pair of them makes a stable
/// map key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyHandle(u64);

impl BodyHandle {
    /// Build a handle from a slot index and generation.
    pub const fn from_parts(index: u32, generation: u32) -> Self {
        Self(((index as u64) << 32) | generation as u64)
    }

    /// The slot index this handle addresses.
    pub const fn index(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// The generation this handle was minted at.
    pub const fn generation(self) -> u32 {
        self.0 as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_round_trip() {
        for (index, generation) in [(0, 0), (1, 0), (0, 1), (7, 3), (u32::MAX, u32::MAX)] {
            let handle = BodyHandle::from_parts(index, generation);
            assert_eq!(handle.index(), index);
            assert_eq!(handle.generation(), generation);
        }
    }

    #[test]
    fn a_reused_slot_is_a_different_handle() {
        let first = BodyHandle::from_parts(4, 0);
        let reused = BodyHandle::from_parts(4, 1);
        assert_ne!(first, reused, "the generation must distinguish the slot");
    }

    #[test]
    fn handles_order_by_slot_then_generation() {
        let mut handles = [
            BodyHandle::from_parts(2, 0),
            BodyHandle::from_parts(1, 5),
            BodyHandle::from_parts(1, 2),
        ];
        handles.sort();
        assert_eq!(
            handles,
            [
                BodyHandle::from_parts(1, 2),
                BodyHandle::from_parts(1, 5),
                BodyHandle::from_parts(2, 0),
            ]
        );
    }
}
