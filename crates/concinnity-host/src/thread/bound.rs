//! A value pinned to the thread that made it, which may still be stored in
//! state that moves between threads.

use std::mem::ManuallyDrop;
use std::thread::ThreadId;

/// Owns a value that must only be used, and dropped, on the thread that wrapped
/// it, such as a graphics context bound to its window's thread.
///
/// The wrapper itself may move to another thread with the state that holds it,
/// but the value is only reachable from its owner thread: on any other thread
/// every accessor answers `None` and logs the misuse. Dropped off its owner
/// thread, the value is leaked rather than torn down on a thread its
/// invariants forbid.
///
/// ```rust
/// # use concinnity_host::thread::ThreadBound;
/// let mut bound = ThreadBound::new(vec![1, 2, 3]);
/// bound.get_mut().unwrap().push(4);
/// assert_eq!(bound.get().map(Vec::len), Some(4));
/// let elsewhere = std::thread::spawn(move || bound.get().is_none()).join();
/// assert_eq!(elsewhere.ok(), Some(true));
/// ```
pub struct ThreadBound<T> {
    value: ManuallyDrop<T>,
    owner: ThreadId,
}

// SAFETY: the value is never reached through the wrapper off `owner`: every
// accessor and `Drop` checks the calling thread first, and a drop elsewhere
// leaks the value instead of running its destructor. So moving the wrapper
// moves no use of `T` to another thread.
unsafe impl<T> Send for ThreadBound<T> {}

impl<T> ThreadBound<T> {
    /// Pin `value` to the calling thread.
    pub fn new(value: T) -> Self {
        Self {
            value: ManuallyDrop::new(value),
            owner: std::thread::current().id(),
        }
    }

    /// Whether the calling thread owns the value.
    pub fn is_owner_thread(&self) -> bool {
        std::thread::current().id() == self.owner
    }

    /// The value, or `None` off its owner thread.
    #[track_caller]
    pub fn get(&self) -> Option<&T> {
        self.checked().then_some(&*self.value)
    }

    /// The value, or `None` off its owner thread.
    #[track_caller]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.checked() {
            Some(&mut *self.value)
        } else {
            None
        }
    }

    /// Unwrap the value, or `None` off its owner thread, where it is leaked.
    #[track_caller]
    pub fn into_inner(self) -> Option<T> {
        // Never dropped through `Drop`, so a misuse is reported once and the
        // value leaks off its owner thread.
        let mut this = ManuallyDrop::new(self);
        if !this.checked() {
            return None;
        }
        // SAFETY: `this` is never used or dropped again, so the value is
        // taken out exactly once, on its owner thread.
        Some(unsafe { ManuallyDrop::take(&mut this.value) })
    }

    #[track_caller]
    fn checked(&self) -> bool {
        let owned = self.is_owner_thread();
        if !owned {
            tracing::error!(
                caller = %std::panic::Location::caller(),
                "a thread-bound {} was reached off the thread that owns it",
                std::any::type_name::<T>()
            );
        }
        owned
    }
}

impl<T> Drop for ThreadBound<T> {
    fn drop(&mut self) {
        if self.checked() {
            // SAFETY: dropped once, here, on the owner thread; the field is
            // not used again.
            unsafe { ManuallyDrop::drop(&mut self.value) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn the_owner_thread_reaches_and_unwraps_the_value() {
        let mut bound = ThreadBound::new(5u32);
        assert!(bound.is_owner_thread());
        *bound.get_mut().unwrap() += 1;
        assert_eq!(bound.get(), Some(&6));
        assert_eq!(bound.into_inner(), Some(6));
    }

    #[test]
    fn another_thread_reaches_nothing() {
        let mut bound = ThreadBound::new(5u32);
        let (bound, seen) = std::thread::spawn(move || {
            let seen = (
                bound.is_owner_thread(),
                bound.get().is_some(),
                bound.get_mut().is_some(),
            );
            (bound, seen)
        })
        .join()
        .unwrap();
        assert_eq!(seen, (false, false, false));
        assert_eq!(bound.get(), Some(&5), "back home it is reachable again");
    }

    // A value dropped on its owner thread is torn down; dropped or unwrapped
    // anywhere else it is leaked, never destroyed on the wrong thread.
    #[test]
    fn only_the_owner_thread_destroys_the_value() {
        let drops = Arc::new(AtomicUsize::new(0));
        drop(ThreadBound::new(Counted(Arc::clone(&drops))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let away = ThreadBound::new(Counted(Arc::clone(&drops)));
        std::thread::spawn(move || drop(away)).join().unwrap();
        let unwrapped = ThreadBound::new(Counted(Arc::clone(&drops)));
        let none = std::thread::spawn(move || unwrapped.into_inner().is_none())
            .join()
            .unwrap();
        assert!(none);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
