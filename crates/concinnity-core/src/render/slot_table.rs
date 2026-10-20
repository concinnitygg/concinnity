//! A capacity-bounded table of slots addressed by stable ids, with a tombstone
//! free list so an insert / remove cycle reuses ids instead of growing.

use alloc::vec::Vec;
use core::fmt;

/// Returned by [`SlotTable::insert`] when every slot up to the table's capacity
/// is live. The capacity is the backend's own per-slot descriptor reservation,
/// so the backend words the error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtCapacity;

/// Why [`SlotTable::remove`] rejected an id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveError {
    /// No slot has ever been handed out under this id.
    OutOfRange,
    /// The id names a slot an earlier remove already tombstoned.
    AlreadyRemoved,
}

impl fmt::Display for RemoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange => f.write_str("out of range"),
            Self::AlreadyRemoved => f.write_str("already removed"),
        }
    }
}

/// Slots of `T` addressed by the id [`SlotTable::insert`] returns. An id is
/// stable until [`SlotTable::remove`] tombstones it, so it can index a
/// backend's parallel per-slot resources.
#[derive(Debug, Clone)]
pub(crate) struct SlotTable<T> {
    slots: Vec<Option<T>>,
    free_slots: Vec<usize>,
    live: usize,
    capacity: usize,
}

impl<T> SlotTable<T> {
    /// An empty table holding at most `capacity` slots.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            free_slots: Vec::new(),
            live: 0,
            capacity,
        }
    }

    /// Whether another [`Self::insert`] would succeed, so a caller can skip
    /// building resources for a value the table would reject.
    pub(crate) fn has_room(&self) -> bool {
        !self.free_slots.is_empty() || self.slots.len() < self.capacity
    }

    /// Place `value` in a slot and return its id, reusing a tombstoned slot
    /// before growing the table.
    pub(crate) fn insert(&mut self, value: T) -> Result<usize, AtCapacity> {
        let id = if let Some(id) = self.free_slots.pop() {
            self.slots[id] = Some(value);
            id
        } else {
            if self.slots.len() >= self.capacity {
                return Err(AtCapacity);
            }
            self.slots.push(Some(value));
            self.slots.len() - 1
        };
        self.live += 1;
        Ok(id)
    }

    /// Tombstone the slot at `id`, returning its value and freeing the id for
    /// the next [`Self::insert`].
    pub(crate) fn remove(&mut self, id: usize) -> Result<T, RemoveError> {
        let slot = self.slots.get_mut(id).ok_or(RemoveError::OutOfRange)?;
        let value = slot.take().ok_or(RemoveError::AlreadyRemoved)?;
        self.free_slots.push(id);
        self.live -= 1;
        Ok(value)
    }

    /// Whether no slot is live.
    pub(crate) fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Whether no slot has ever been handed out, tombstoned or not.
    pub(crate) fn never_used(&self) -> bool {
        self.slots.is_empty()
    }

    /// The value at `id`, if that slot is live.
    #[cfg(test)]
    pub(crate) fn get(&self, id: usize) -> Option<&T> {
        self.slots.get(id)?.as_ref()
    }

    /// The live slots with their ids, in id order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| Some((id, slot.as_ref()?)))
    }

    /// The live slots with their ids, in id order, mutably.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(id, slot)| Some((id, slot.as_mut()?)))
    }

    /// Drop every slot, live or tombstoned, keeping the capacity.
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.free_slots.clear();
        self.live = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_removed_id_is_reused_at_capacity() {
        let mut table = SlotTable::new(1);
        let id = table.insert('a').expect("room for one");
        assert_eq!(table.insert('b'), Err(AtCapacity));
        assert_eq!(table.remove(id), Ok('a'));
        assert_eq!(table.insert('c'), Ok(id));
    }

    #[test]
    fn insert_hands_out_ascending_ids_up_to_capacity() {
        let mut t = SlotTable::new(2);
        assert!(t.has_room());
        assert_eq!(t.insert(10), Ok(0));
        assert_eq!(t.insert(11), Ok(1));
        assert!(!t.has_room());
        assert_eq!(t.insert(12), Err(AtCapacity));
        assert!(!t.is_empty());
    }

    #[test]
    fn a_removed_slot_is_reused_before_the_table_grows() {
        let mut t = SlotTable::new(2);
        let first = t.insert('a').unwrap();
        t.insert('b').unwrap();
        assert_eq!(t.remove(first), Ok('a'));
        assert!(t.has_room());
        assert_eq!(t.insert('c'), Ok(first));
        assert_eq!(t.insert('d'), Err(AtCapacity));
    }

    #[test]
    fn remove_rejects_unknown_and_repeated_ids() {
        let mut t = SlotTable::new(2);
        assert_eq!(t.remove(0), Err(RemoveError::OutOfRange));
        let id = t.insert(()).unwrap();
        assert_eq!(t.remove(id), Ok(()));
        assert_eq!(t.remove(id), Err(RemoveError::AlreadyRemoved));
        assert!(t.is_empty());
        assert!(!t.never_used(), "a tombstone still counts as used");
    }

    #[test]
    fn iteration_skips_tombstones_and_keeps_ids() {
        let mut t = SlotTable::new(4);
        for v in 0..3 {
            t.insert(v).unwrap();
        }
        t.remove(1).unwrap();
        assert_eq!(t.iter().collect::<Vec<_>>(), vec![(0, &0), (2, &2)]);
        for (_, v) in t.iter_mut() {
            *v += 10;
        }
        assert_eq!(t.get(2), Some(&12));
        assert_eq!(t.get(1), None);
        assert_eq!(t.get(9), None);
    }

    #[test]
    fn clear_forgets_every_slot() {
        let mut t = SlotTable::new(1);
        t.insert(1).unwrap();
        t.clear();
        assert!(t.never_used());
        assert_eq!(t.insert(2), Ok(0));
    }

    #[test]
    fn remove_errors_read_as_the_backend_messages() {
        use alloc::format;
        assert_eq!(format!("{}", RemoveError::OutOfRange), "out of range");
        assert_eq!(
            format!("{}", RemoveError::AlreadyRemoved),
            "already removed"
        );
    }
}
