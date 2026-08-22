// concinnity-eas/src/event_store.rs
//
// Owner of every `Events<E>` queue, keyed by event type. Queues are created
// lazily on first mutable access and the store can rotate all of them at once,
// so a frame driver never maintains a per-type rotation list (a queue missing
// from such a list would buffer its events forever).

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::any::{Any, TypeId};

use crate::event::Events;

// Object-safe view of one queue: rotation without knowing the event type,
// plus downcast access back to the concrete `Events<E>`. `Send` so the store
// (inside the world) can move to a simulation thread.
trait AnyEventQueue: Send {
    fn update(&mut self);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<E: Send + 'static> AnyEventQueue for Events<E> {
    fn update(&mut self) {
        Events::update(self);
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Default)]
/// Type-keyed event queues, one per event type in use.
pub struct EventStore {
    queues: BTreeMap<TypeId, Box<dyn AnyEventQueue>>,
}

impl EventStore {
    /// An empty store.
    pub fn new() -> EventStore {
        EventStore::default()
    }

    /// Borrow the queue for event type E, if one has been created.
    pub fn get<E: 'static>(&self) -> Option<&Events<E>> {
        self.queues
            .get(&TypeId::of::<E>())
            .and_then(|queue| queue.as_any().downcast_ref::<Events<E>>())
    }

    /// Mutably borrow the queue for event type E, creating an empty one on
    /// first access so writers and readers never miss it.
    pub fn get_mut_or_create<E: Send + 'static>(&mut self) -> &mut Events<E> {
        self.queues
            .entry(TypeId::of::<E>())
            .or_insert_with(|| Box::new(Events::<E>::new()))
            .as_any_mut()
            .downcast_mut::<Events<E>>()
            .expect("queue stored under E's TypeId is Events<E>")
    }

    /// Advance every queue one frame (see `Events::update`). Queues are
    /// independent, so rotation order does not matter.
    pub fn update_all(&mut self) {
        for queue in self.queues.values_mut() {
            queue.update();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_before_first_write_is_none() {
        let store = EventStore::new();
        assert!(store.get::<u32>().is_none());
    }

    #[test]
    fn queue_persists_across_accesses() {
        let mut store = EventStore::new();
        store.get_mut_or_create::<u32>().send(7);
        assert_eq!(store.get::<u32>().unwrap().len(), 1);
        // A second mutable access returns the same queue, not a fresh one.
        assert_eq!(store.get_mut_or_create::<u32>().len(), 1);
    }

    #[test]
    fn update_all_rotates_every_queue() {
        let mut store = EventStore::new();
        store.get_mut_or_create::<u32>().send(1);
        store.get_mut_or_create::<&str>().send("a");
        // Two-frame retention: one rotation keeps the events readable, the
        // second retires them from every queue.
        store.update_all();
        assert_eq!(store.get::<u32>().unwrap().len(), 1);
        assert_eq!(store.get::<&str>().unwrap().len(), 1);
        store.update_all();
        assert!(store.get::<u32>().unwrap().is_empty());
        assert!(store.get::<&str>().unwrap().is_empty());
    }
}
