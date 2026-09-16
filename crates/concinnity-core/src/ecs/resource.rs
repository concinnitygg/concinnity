// Type-keyed singleton store: each type has at most one instance, fetched by
// type. The home for engine-wide singletons (frame input, the render backend,
// the profiler) that would otherwise be faked as one-element collections.
//
// Values are required to be `Send` so the world that owns the store can move
// to a simulation thread. Thread-affine state (a GPU backend) may still be
// stored behind a `Send` handle, but its owner must keep it on the thread its
// invariants require.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::any::{Any, TypeId};

#[derive(Default)]
/// Type-keyed singleton storage: one value per resource type.
pub struct Resources {
    map: BTreeMap<TypeId, Box<dyn Any + Send>>,
}

impl Resources {
    /// An empty store.
    pub fn new() -> Resources {
        Resources::default()
    }

    /// Insert a resource, returning the previous instance of the same type if
    /// one was present. Replaces in place when the type is already present, so
    /// a per-frame republish reuses the existing allocation.
    pub fn insert<T: Any + Send>(&mut self, value: T) -> Option<T> {
        if let Some(slot) = self.map.get_mut(&TypeId::of::<T>()) {
            let existing = (slot.as_mut() as &mut dyn Any)
                .downcast_mut::<T>()
                .expect("resource slot type matches its TypeId key");
            return Some(core::mem::replace(existing, value));
        }
        self.map.insert(TypeId::of::<T>(), Box::new(value));
        None
    }

    /// Borrow the resource of type `T`, if one is present.
    pub fn get<T: Any>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|boxed| (boxed.as_ref() as &dyn Any).downcast_ref::<T>())
    }

    /// Mutably borrow the resource of type `T`, if one is present.
    pub fn get_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|boxed| (boxed.as_mut() as &mut dyn Any).downcast_mut::<T>())
    }

    /// Remove and return the resource of type `T`, if one is present.
    pub fn remove<T: Any>(&mut self) -> Option<T> {
        self.map.remove(&TypeId::of::<T>()).and_then(downcast::<T>)
    }

    /// Take the resource value, leaving `T::default()` parked in its slot so a
    /// later `insert` republish reuses the allocation. `None` when the type was
    /// never inserted; a per-frame take/put cycle never re-boxes.
    pub fn take<T: Any + Send + Default>(&mut self) -> Option<T> {
        self.get_mut::<T>().map(core::mem::take)
    }

    /// Mutably borrow three distinct resource types at once, each `None` when
    /// absent.
    ///
    /// # Panics
    ///
    /// When any two of `A`, `B` and `C` are the same type.
    pub fn get_disjoint_mut<A: Any, B: Any, C: Any>(
        &mut self,
    ) -> (Option<&mut A>, Option<&mut B>, Option<&mut C>) {
        let (a, b, c) = (TypeId::of::<A>(), TypeId::of::<B>(), TypeId::of::<C>());
        assert!(
            a != b && a != c && b != c,
            "a disjoint resource borrow names one type twice"
        );
        let (mut ra, mut rb, mut rc) = (None, None, None);
        for (id, slot) in self.map.iter_mut() {
            let value = slot.as_mut() as &mut dyn Any;
            if *id == a {
                ra = value.downcast_mut::<A>();
            } else if *id == b {
                rb = value.downcast_mut::<B>();
            } else if *id == c {
                rc = value.downcast_mut::<C>();
            }
        }
        (ra, rb, rc)
    }

    /// Whether a resource of type `T` is present.
    pub fn contains<T: Any>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }
}

fn downcast<T: Any>(boxed: Box<dyn Any + Send>) -> Option<T> {
    (boxed as Box<dyn Any>).downcast::<T>().ok().map(|v| *v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Default)]
    struct Seconds(f32);

    #[test]
    fn insert_get_and_remove_by_type() {
        let mut resources = Resources::new();
        assert!(!resources.contains::<Seconds>());
        assert_eq!(resources.insert(Seconds(0.016)), None);
        assert!(resources.contains::<Seconds>());
        assert_eq!(resources.get::<Seconds>(), Some(&Seconds(0.016)));
        assert_eq!(resources.remove::<Seconds>(), Some(Seconds(0.016)));
        assert!(!resources.contains::<Seconds>());
    }

    #[test]
    fn insert_returns_previous_value() {
        let mut resources = Resources::new();
        resources.insert(Seconds(1.0));
        assert_eq!(resources.insert(Seconds(2.0)), Some(Seconds(1.0)));
    }

    #[test]
    fn get_mut_edits_in_place() {
        let mut resources = Resources::new();
        resources.insert(Seconds(1.0));
        resources.get_mut::<Seconds>().unwrap().0 = 5.0;
        assert_eq!(resources.get::<Seconds>(), Some(&Seconds(5.0)));
    }

    #[test]
    fn insert_replaces_in_place_without_reboxing() {
        let mut resources = Resources::new();
        resources.insert(Seconds(1.0));
        let before = resources.get::<Seconds>().unwrap() as *const Seconds;
        assert_eq!(resources.insert(Seconds(2.0)), Some(Seconds(1.0)));
        let after = resources.get::<Seconds>().unwrap() as *const Seconds;
        assert_eq!(before, after, "republish must reuse the existing box");
    }

    #[test]
    fn take_leaves_a_default_parked_in_the_slot() {
        let mut resources = Resources::new();
        assert_eq!(resources.take::<Seconds>(), None);
        resources.insert(Seconds(3.0));
        let before = resources.get::<Seconds>().unwrap() as *const Seconds;
        assert_eq!(resources.take::<Seconds>(), Some(Seconds(3.0)));
        let after = resources.get::<Seconds>().unwrap() as *const Seconds;
        assert_eq!(before, after, "take must leave the box parked");
        assert_eq!(resources.get::<Seconds>(), Some(&Seconds(0.0)));
    }

    #[test]
    fn disjoint_borrow_hands_out_each_present_type() {
        let mut resources = Resources::new();
        resources.insert(Seconds(1.0));
        resources.insert(7u32);
        let (time, count, missing) = resources.get_disjoint_mut::<Seconds, u32, i64>();
        time.unwrap().0 = 2.0;
        *count.unwrap() += 1;
        assert!(missing.is_none());
        assert_eq!(resources.get::<Seconds>(), Some(&Seconds(2.0)));
        assert_eq!(resources.get::<u32>(), Some(&8));
    }

    #[test]
    #[should_panic(expected = "names one type twice")]
    fn disjoint_borrow_rejects_a_repeated_type() {
        let mut resources = Resources::new();
        let _ = resources.get_disjoint_mut::<u32, i64, u32>();
    }

    #[test]
    fn distinct_types_are_independent() {
        let mut resources = Resources::new();
        resources.insert(Seconds(1.0));
        resources.insert(7u32);
        assert_eq!(resources.get::<Seconds>(), Some(&Seconds(1.0)));
        assert_eq!(resources.get::<u32>(), Some(&7));
    }
}
