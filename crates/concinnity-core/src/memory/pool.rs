// A fixed-capacity pool for objects of one type: storage reserved once, slots
// handed out and taken back through a free list.
//
// The case it exists for is a population that churns but never grows without
// bound -- spawned entities, in-flight loads, voices. Each of those costs an
// allocation and a free per item from the global allocator, and the frees leave
// holes behind; a pool pays for the whole population once and reuses the same
// slots forever.
//
// Handles carry a generation, so a handle to a removed object reads as absent
// rather than silently addressing whatever took its slot.

use alloc::vec::Vec;

// A slot's occupant, or the emptiness left when it was removed.
struct Slot<T> {
    value: Option<T>,
    // Bumped when a slot is vacated, which is what makes old handles stale.
    generation: u32,
}

/// A reference to one object in a pool. Copyable and small: pass it around
/// instead of the object.
///
/// The generation is what makes a handle to a removed object read as absent
/// rather than silently addressing whatever took its slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolHandle {
    index: u32,
    generation: u32,
}

impl PoolHandle {
    /// Rebuild a handle from parts a caller stored elsewhere.
    ///
    /// Parts that never named a live object read as absent, exactly as a
    /// stale handle does, so this widens no access the pool did not grant.
    pub const fn from_parts(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Position in the pool, for a caller keeping a table alongside it.
    pub const fn index(self) -> usize {
        self.index as usize
    }

    /// The generation this handle was minted at.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// A fixed-capacity slot pool handing out generation-checked handles.
pub struct Pool<T> {
    slots: Vec<Slot<T>>,
    // Vacant slots, most recently vacated first.
    free: Vec<u32>,
    len: usize,
}

impl<T> Pool<T> {
    /// Reserve room for `capacity` objects. The pool never allocates again: it
    /// hands out `None` when full rather than growing.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut free = Vec::with_capacity(capacity);
        for index in 0..capacity {
            slots.push(Slot {
                value: None,
                generation: 0,
            });
            // Reversed, so the first insert takes slot 0 and a fresh pool fills
            // in order.
            free.push((capacity - 1 - index) as u32);
        }
        Self {
            slots,
            free,
            len: 0,
        }
    }

    /// Slots the pool reserved.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Live objects.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the pool holds no live objects.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(test)]
    pub(crate) fn is_full(&self) -> bool {
        self.free.is_empty()
    }

    /// Bytes the pool reserved, occupied or not: what it costs the process
    /// whatever its occupancy, and so what it reports to a byte budget.
    pub fn reserved_bytes(&self) -> u64 {
        (self.capacity() * size_of::<Slot<T>>()) as u64
    }

    /// Place `value` in a free slot. `None` when the pool is full, which is the
    /// caller's cue to drop the request or widen the pool at setup.
    pub fn insert(&mut self, value: T) -> Option<PoolHandle> {
        let index = self.free.pop()?;
        let slot = &mut self.slots[index as usize];
        slot.value = Some(value);
        self.len += 1;
        Some(PoolHandle {
            index,
            generation: slot.generation,
        })
    }

    /// Take the object back out, freeing its slot. `None` when the handle is
    /// stale or already removed.
    pub fn remove(&mut self, handle: PoolHandle) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        self.len -= 1;
        Some(value)
    }

    /// Borrow the object a handle names, if the handle is still live.
    pub fn get(&self, handle: PoolHandle) -> Option<&T> {
        let slot = self.slots.get(handle.index as usize)?;
        (slot.generation == handle.generation).then_some(slot.value.as_ref()?)
    }

    /// Mutably borrow the object a handle names, if it is still live.
    pub fn get_mut(&mut self, handle: PoolHandle) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_mut()
    }

    /// Borrow whatever occupies a slot, by position rather than by handle.
    ///
    /// For a caller that keeps its own tables alongside the pool and indexes
    /// them by slot: the position came from the pool, so re-checking a
    /// generation it never left would only cost a branch.
    pub fn get_at(&self, index: usize) -> Option<&T> {
        self.slots.get(index)?.value.as_ref()
    }

    /// Mutably borrow whatever occupies a slot, by position.
    pub fn get_at_mut(&mut self, index: usize) -> Option<&mut T> {
        self.slots.get_mut(index)?.value.as_mut()
    }

    /// The handle naming whatever occupies a slot, so a caller that walks
    /// positions can hand one back out. `None` when the slot is vacant.
    pub fn handle_at(&self, index: usize) -> Option<PoolHandle> {
        let slot = self.slots.get(index)?;
        slot.value.as_ref()?;
        Some(PoolHandle {
            index: index as u32,
            generation: slot.generation,
        })
    }

    /// Whether a handle still names a live object.
    pub fn contains(&self, handle: PoolHandle) -> bool {
        self.get(handle).is_some()
    }

    /// Every live object with its handle, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (PoolHandle, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let value = slot.value.as_ref()?;
            Some((
                PoolHandle {
                    index: index as u32,
                    generation: slot.generation,
                },
                value,
            ))
        })
    }

    /// Every live object with its handle, mutably, in slot order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (PoolHandle, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(index, slot)| {
                let generation = slot.generation;
                let value = slot.value.as_mut()?;
                Some((
                    PoolHandle {
                        index: index as u32,
                        generation,
                    },
                    value,
                ))
            })
    }

    /// Drop every occupant, keeping the reserved storage. Outstanding handles go
    /// stale, as they would if each object were removed individually.
    pub fn clear(&mut self) {
        self.free.clear();
        for (index, slot) in self.slots.iter_mut().enumerate().rev() {
            if slot.value.take().is_some() {
                slot.generation = slot.generation.wrapping_add(1);
            }
            self.free.push(index as u32);
        }
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_read_back_through_their_handles() {
        let mut pool = Pool::with_capacity(4);
        let a = pool.insert("a").expect("room");
        let b = pool.insert("b").expect("room");

        assert_eq!(pool.get(a), Some(&"a"));
        assert_eq!(pool.get(b), Some(&"b"));
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.capacity(), 4);
    }

    #[test]
    fn a_fresh_pool_fills_its_slots_in_order() {
        let mut pool = Pool::with_capacity(3);
        for expected in 0..3 {
            assert_eq!(pool.insert(expected).expect("room").index(), expected);
        }
    }

    #[test]
    fn removal_frees_the_slot_for_reuse() {
        let mut pool = Pool::with_capacity(2);
        let a = pool.insert(1).expect("room");
        let b = pool.insert(2).expect("room");
        assert!(pool.is_full());

        assert_eq!(pool.remove(a), Some(1));
        assert_eq!(pool.len(), 1);
        let c = pool.insert(3).expect("the freed slot");
        assert_eq!(c.index(), a.index(), "the vacated slot is reused");
        assert_eq!(pool.get(b), Some(&2));
        assert_eq!(pool.get(c), Some(&3));
    }

    // A caller walking positions has to be able to hand a handle back out,
    // and the handle it gets must be the one the pool minted.
    #[test]
    fn a_slot_hands_back_the_handle_naming_its_occupant() {
        let mut pool = Pool::with_capacity(2);
        let a = pool.insert("a").expect("room");
        assert_eq!(pool.handle_at(a.index()), Some(a));
        assert_eq!(pool.handle_at(1), None, "vacant");
        assert_eq!(pool.handle_at(99), None, "out of range");

        pool.remove(a);
        assert_eq!(pool.handle_at(a.index()), None);
        let b = pool.insert("b").expect("the freed slot");
        assert_eq!(pool.handle_at(b.index()), Some(b));
        assert_ne!(
            pool.handle_at(b.index()),
            Some(a),
            "the generation moved on"
        );
    }

    // The point of the generation: a handle to a removed object must not reach
    // whatever took its slot.
    #[test]
    fn a_stale_handle_does_not_reach_the_slots_new_occupant() {
        let mut pool = Pool::with_capacity(1);
        let old = pool.insert("first").expect("room");
        assert_eq!(pool.remove(old), Some("first"));
        let new = pool.insert("second").expect("the freed slot");

        assert_eq!(new.index(), old.index());
        assert_eq!(pool.get(old), None);
        assert!(!pool.contains(old));
        assert_eq!(pool.get_mut(old), None);
        assert_eq!(pool.remove(old), None);
        assert_eq!(pool.get(new), Some(&"second"));
    }

    // Slot access is for callers holding their own table: it must reach the
    // live occupant and report an empty or out-of-range slot as absent.
    #[test]
    fn slot_access_reaches_the_occupant_and_skips_the_vacancies() {
        let mut pool = Pool::with_capacity(3);
        let a = pool.insert(1).expect("room");
        let b = pool.insert(2).expect("room");
        assert_eq!(pool.get_at(a.index()), Some(&1));
        assert_eq!(pool.get_at(b.index()), Some(&2));
        assert_eq!(pool.get_at(2), None);
        assert_eq!(pool.get_at(99), None);

        *pool.get_at_mut(b.index()).expect("live") = 20;
        assert_eq!(pool.get(b), Some(&20));
        pool.remove(a);
        assert_eq!(pool.get_at(a.index()), None);
        assert_eq!(pool.get_at_mut(99), None);
    }

    #[test]
    fn a_full_pool_declines_rather_than_growing() {
        let mut pool = Pool::with_capacity(2);
        assert!(pool.insert(1).is_some());
        assert!(pool.insert(2).is_some());
        assert!(pool.insert(3).is_none());
        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn a_zero_capacity_pool_holds_nothing() {
        let mut pool = Pool::with_capacity(0);
        assert!(pool.is_full());
        assert!(pool.insert(1).is_none());
        assert_eq!(pool.iter().count(), 0);
    }

    #[test]
    fn objects_can_be_mutated_in_place() {
        let mut pool = Pool::with_capacity(2);
        let h = pool.insert(10).expect("room");
        *pool.get_mut(h).expect("live") += 5;
        assert_eq!(pool.get(h), Some(&15));

        for (_, value) in pool.iter_mut() {
            *value *= 2;
        }
        assert_eq!(pool.get(h), Some(&30));
    }

    #[test]
    fn iteration_visits_live_objects_only() {
        let mut pool = Pool::with_capacity(4);
        let a = pool.insert(1).expect("room");
        let _b = pool.insert(2).expect("room");
        let c = pool.insert(3).expect("room");
        pool.remove(a);

        let live: alloc::vec::Vec<i32> = pool.iter().map(|(_, v)| *v).collect();
        assert_eq!(live, [2, 3]);
        // Handles from iteration address the objects they were read from.
        let (handle, _) = pool.iter().next().expect("a live object");
        assert_eq!(pool.get(handle), Some(&2));
        assert_eq!(pool.get(c), Some(&3));
    }

    #[test]
    fn clear_empties_the_pool_and_stales_its_handles() {
        let mut pool = Pool::with_capacity(3);
        let a = pool.insert(1).expect("room");
        let b = pool.insert(2).expect("room");
        pool.clear();

        assert!(pool.is_empty());
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.get(a), None);
        assert_eq!(pool.get(b), None);
        // And the storage is all available again, in order.
        assert_eq!(pool.insert(9).expect("room").index(), 0);
    }

    // A handle a caller stored as parts and rebuilt must still address the same
    // object, and a rebuilt handle whose parts never named one must not.
    #[test]
    fn handles_rebuild_from_their_parts() {
        let mut pool = Pool::with_capacity(2);
        let a = pool.insert("a").expect("room");
        let rebuilt = PoolHandle::from_parts(a.index() as u32, a.generation());
        assert_eq!(rebuilt, a);
        assert_eq!(pool.get(rebuilt), Some(&"a"));

        assert_eq!(pool.get(PoolHandle::from_parts(0, 7)), None);
        assert_eq!(pool.get(PoolHandle::from_parts(99, 0)), None);
    }

    #[test]
    fn a_reused_slot_reports_a_later_generation() {
        let mut pool = Pool::with_capacity(1);
        let first = pool.insert(1).expect("room");
        assert_eq!(first.generation(), 0);
        pool.remove(first);
        let second = pool.insert(2).expect("the freed slot");
        assert_eq!(second.index(), first.index());
        assert_eq!(second.generation(), 1);
    }

    // The pool's cost is its reservation, not its occupancy: that is the figure
    // a byte budget needs.
    #[test]
    fn reserved_bytes_counts_the_reservation_not_the_occupancy() {
        let mut pool = Pool::<u64>::with_capacity(16);
        let reserved = pool.reserved_bytes();
        assert!(reserved >= 16 * size_of::<u64>() as u64);
        pool.insert(1);
        assert_eq!(pool.reserved_bytes(), reserved);
    }

    // Dropping the pool must drop its occupants, not leak them.
    #[test]
    fn occupants_are_dropped_with_the_pool() {
        use alloc::rc::Rc;

        let witness = Rc::new(());
        {
            let mut pool = Pool::with_capacity(2);
            pool.insert(Rc::clone(&witness));
            assert_eq!(Rc::strong_count(&witness), 2);
        }
        assert_eq!(Rc::strong_count(&witness), 1);
    }

    // And so must removing one.
    #[test]
    fn a_removed_occupant_is_handed_back_intact() {
        use alloc::rc::Rc;

        let witness = Rc::new(());
        let mut pool = Pool::with_capacity(2);
        let h = pool.insert(Rc::clone(&witness)).expect("room");
        let taken = pool.remove(h).expect("live");
        assert_eq!(Rc::strong_count(&witness), 2);
        drop(taken);
        assert_eq!(Rc::strong_count(&witness), 1);
    }
}
