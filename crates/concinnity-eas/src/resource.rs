// concinnity-eas/src/resource.rs
//
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
pub struct Resources {
    map: BTreeMap<TypeId, Box<dyn Any + Send>>,
}

impl Resources {
    pub fn new() -> Resources {
        Resources::default()
    }

    // Insert a resource, returning the previous instance of the same type if
    // one was present. Replaces in place when the type is already present, so
    // a per-frame republish reuses the existing allocation.
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

    pub fn get<T: Any>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|boxed| (boxed.as_ref() as &dyn Any).downcast_ref::<T>())
    }

    pub fn get_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|boxed| (boxed.as_mut() as &mut dyn Any).downcast_mut::<T>())
    }

    pub fn remove<T: Any>(&mut self) -> Option<T> {
        self.map.remove(&TypeId::of::<T>()).and_then(downcast::<T>)
    }

    // Take the resource value, leaving `T::default()` parked in its slot so a
    // later `insert` republish reuses the allocation. `None` when the type was
    // never inserted; a per-frame take/put cycle never re-boxes.
    pub fn take<T: Any + Send + Default>(&mut self) -> Option<T> {
        self.get_mut::<T>().map(core::mem::take)
    }

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
    struct FrameTime(f32);

    #[test]
    fn insert_get_and_remove_by_type() {
        let mut resources = Resources::new();
        assert!(!resources.contains::<FrameTime>());
        assert_eq!(resources.insert(FrameTime(0.016)), None);
        assert!(resources.contains::<FrameTime>());
        assert_eq!(resources.get::<FrameTime>(), Some(&FrameTime(0.016)));
        assert_eq!(resources.remove::<FrameTime>(), Some(FrameTime(0.016)));
        assert!(!resources.contains::<FrameTime>());
    }

    #[test]
    fn insert_returns_previous_value() {
        let mut resources = Resources::new();
        resources.insert(FrameTime(1.0));
        assert_eq!(resources.insert(FrameTime(2.0)), Some(FrameTime(1.0)));
    }

    #[test]
    fn get_mut_edits_in_place() {
        let mut resources = Resources::new();
        resources.insert(FrameTime(1.0));
        resources.get_mut::<FrameTime>().unwrap().0 = 5.0;
        assert_eq!(resources.get::<FrameTime>(), Some(&FrameTime(5.0)));
    }

    #[test]
    fn insert_replaces_in_place_without_reboxing() {
        let mut resources = Resources::new();
        resources.insert(FrameTime(1.0));
        let before = resources.get::<FrameTime>().unwrap() as *const FrameTime;
        assert_eq!(resources.insert(FrameTime(2.0)), Some(FrameTime(1.0)));
        let after = resources.get::<FrameTime>().unwrap() as *const FrameTime;
        assert_eq!(before, after, "republish must reuse the existing box");
    }

    #[test]
    fn take_leaves_a_default_parked_in_the_slot() {
        let mut resources = Resources::new();
        assert_eq!(resources.take::<FrameTime>(), None);
        resources.insert(FrameTime(3.0));
        let before = resources.get::<FrameTime>().unwrap() as *const FrameTime;
        assert_eq!(resources.take::<FrameTime>(), Some(FrameTime(3.0)));
        let after = resources.get::<FrameTime>().unwrap() as *const FrameTime;
        assert_eq!(before, after, "take must leave the box parked");
        assert_eq!(resources.get::<FrameTime>(), Some(&FrameTime(0.0)));
    }

    #[test]
    fn distinct_types_are_independent() {
        let mut resources = Resources::new();
        resources.insert(FrameTime(1.0));
        resources.insert(7u32);
        assert_eq!(resources.get::<FrameTime>(), Some(&FrameTime(1.0)));
        assert_eq!(resources.get::<u32>(), Some(&7));
    }
}
