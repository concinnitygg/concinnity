// The driver's lookup containers: a sorted `Vec` behind a map and a set API.
//
// Two properties decide this rather than a tree or a hash table. Every
// container here is reserved from the world's budget at init and never grown,
// so the step path must not allocate; a `Vec` reserved once cannot, while a
// tree allocates per node and a frame-drained tree allocates per frame. And a
// key order that is the same on every run is what keeps the contact events a
// frame publishes in one order, whatever the solve did.
//
// Lookups are a binary search over a contiguous run of keys, which for the
// entry counts a body budget reserves is the shape a cache likes anyway.

use alloc::vec::Vec;

/// A map from `K` to `V` kept sorted by key.
#[derive(Debug)]
pub(crate) struct SortedMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K, V> Default for SortedMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: Ord + Copy, V> SortedMap<K, V> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    // Where `key` is, or where it would go.
    fn seek(&self, key: &K) -> Result<usize, usize> {
        self.entries.binary_search_by(|(k, _)| k.cmp(key))
    }

    /// Insert or overwrite, returning what was there.
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.seek(&key) {
            Ok(at) => Some(core::mem::replace(&mut self.entries[at].1, value)),
            Err(at) => {
                self.entries.insert(at, (key, value));
                None
            }
        }
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.seek(key).ok().map(|at| &self.entries[at].1)
    }

    pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.seek(key).ok().map(|at| &mut self.entries[at].1)
    }

    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        self.seek(key).ok().map(|at| self.entries.remove(at).1)
    }

    /// Take every entry in key order, keeping the reservation.
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        self.entries.drain(..)
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) {
        self.entries.retain(|(k, v)| keep(k, v));
    }
}

/// A set of `K` kept sorted.
#[derive(Debug)]
pub(crate) struct SortedSet<K> {
    keys: Vec<K>,
}

impl<K> Default for SortedSet<K> {
    fn default() -> Self {
        Self { keys: Vec::new() }
    }
}

impl<K: Ord + Copy> SortedSet<K> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
        }
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.keys.capacity()
    }

    /// Add `key`, reporting whether it was new.
    pub(crate) fn insert(&mut self, key: K) -> bool {
        match self.keys.binary_search(&key) {
            Ok(_) => false,
            Err(at) => {
                self.keys.insert(at, key);
                true
            }
        }
    }

    pub(crate) fn remove(&mut self, key: &K) -> bool {
        match self.keys.binary_search(key) {
            Ok(at) => {
                self.keys.remove(at);
                true
            }
            Err(_) => false,
        }
    }

    pub(crate) fn contains(&self, key: &K) -> bool {
        self.keys.binary_search(key).is_ok()
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) {
        self.keys.retain(|k| keep(k));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_map_reads_back_what_it_stored_whatever_the_insert_order() {
        let mut map = SortedMap::default();
        for key in [7u32, 1, 4, 9, 2] {
            assert_eq!(map.insert(key, key * 10), None);
        }
        assert_eq!(map.len(), 5);
        for key in [1u32, 2, 4, 7, 9] {
            assert_eq!(map.get(&key), Some(&(key * 10)));
        }
        assert_eq!(map.get(&3), None);
    }

    #[test]
    fn inserting_a_live_key_overwrites_and_reports_the_old_value() {
        let mut map = SortedMap::default();
        map.insert(2u32, 'a');
        assert_eq!(map.insert(2, 'b'), Some('a'));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&2), Some(&'b'));
        *map.get_mut(&2).expect("the entry is live") = 'c';
        assert_eq!(map.get(&2), Some(&'c'));
    }

    #[test]
    fn removing_drops_only_the_named_key() {
        let mut map = SortedMap::default();
        for key in [1u32, 2, 3] {
            map.insert(key, key);
        }
        assert_eq!(map.remove(&2), Some(2));
        assert_eq!(map.remove(&2), None);
        assert_eq!(map.get(&1), Some(&1));
        assert_eq!(map.get(&3), Some(&3));
    }

    // Draining hands entries back in key order however they went in, which is
    // what makes what a frame publishes from one the same on every run.
    #[test]
    fn draining_yields_key_order_and_keeps_the_reservation() {
        let mut map = SortedMap::with_capacity(8);
        for key in [5u32, 1, 3] {
            map.insert(key, key);
        }
        let drained: Vec<u32> = map.drain().map(|(k, _)| k).collect();
        assert_eq!(drained, [1, 3, 5]);
        assert_eq!(map.len(), 0);
        assert!(map.capacity() >= 8, "the drain kept the reservation");
    }

    #[test]
    fn retain_keeps_the_entries_it_is_told_to() {
        let mut map = SortedMap::default();
        for key in 0u32..6 {
            map.insert(key, key);
        }
        map.retain(|k, _| k % 2 == 0);
        let left: Vec<u32> = map.drain().map(|(k, _)| k).collect();
        assert_eq!(left, [0, 2, 4]);
    }

    #[test]
    fn a_set_admits_each_key_once() {
        let mut set = SortedSet::default();
        assert!(set.insert(4u32));
        assert!(!set.insert(4));
        assert!(set.insert(1));
        assert!(set.contains(&1) && set.contains(&4));
        assert!(!set.contains(&2));
        assert!(set.remove(&4));
        assert!(!set.remove(&4));
        assert!(!set.contains(&4));
    }

    #[test]
    fn a_set_retains_the_keys_it_is_told_to() {
        let mut set = SortedSet::with_capacity(8);
        for key in 0u32..6 {
            set.insert(key);
        }
        set.retain(|k| *k >= 3);
        assert!(!set.contains(&2));
        assert!(set.contains(&3));
        assert!(set.capacity() >= 8);
    }
}
