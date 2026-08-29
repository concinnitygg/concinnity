// The membership/filter bit pair a collider or query carries. Layer *names*
// belong to the authored world, so resolving them is the driver's job; by the
// time a mask reaches the simulation it is two words of bits.

/// Membership and filter bits applied to a collider or query.
///
/// Two colliders interact only when each one's memberships intersect the
/// other's filter, so a one-way filter still blocks the pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerMask {
    /// Layers this collider or query belongs to.
    pub memberships: u32,
    /// Layers this collider or query interacts with.
    pub filter: u32,
}

impl LayerMask {
    /// Member of every layer, interacting with everything.
    pub const ALL: LayerMask = LayerMask {
        memberships: u32::MAX,
        filter: u32::MAX,
    };

    /// Whether two masks admit each other. The same rule governs a query
    /// against a collider, with the query carrying one of the two masks.
    pub(crate) const fn interacts_with(self, other: LayerMask) -> bool {
        (self.memberships & other.filter) != 0 && (other.memberships & self.filter) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn everything_admits_everything() {
        assert!(LayerMask::ALL.interacts_with(LayerMask::ALL));
    }

    #[test]
    fn a_one_way_filter_blocks_the_pair_from_both_sides() {
        let open = LayerMask::ALL;
        let closed = LayerMask {
            memberships: u32::MAX,
            filter: 0,
        };
        assert!(!open.interacts_with(closed));
        assert!(!closed.interacts_with(open));
    }

    #[test]
    fn disjoint_layers_never_meet() {
        let first = LayerMask {
            memberships: 0b01,
            filter: 0b01,
        };
        let second = LayerMask {
            memberships: 0b10,
            filter: 0b10,
        };
        assert!(!first.interacts_with(second));
        assert!(first.interacts_with(first));
    }
}
