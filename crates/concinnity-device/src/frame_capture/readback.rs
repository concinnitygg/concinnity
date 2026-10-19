// Host-visible buffers frame captures copy into, reused across captures and
// lent to the writer thread until it has decoded them.

use std::sync::{Arc, Condvar, Mutex};

use concinnity_core::render::error::RenderResult;

// Whether a buffer's bytes are lent out, and the signal that they came back.
#[derive(Default)]
struct Held {
    lent: Mutex<bool>,
    returned: Condvar,
}

impl Held {
    fn lend(&self) {
        *self
            .lent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    }

    fn give_back(&self) {
        *self
            .lent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
        self.returned.notify_all();
    }

    fn is_lent(&self) -> bool {
        *self
            .lent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn wait_returned(&self) {
        let mut lent = self
            .lent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *lent {
            lent = self
                .returned
                .wait(lent)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

struct Slot<B> {
    buffer: B,
    mapped: *const u8,
    len: usize,
    held: Arc<Held>,
}

/// The read-back buffers of one backend, `B` being its buffer handle. A buffer
/// lent out stays allocated and mapped until its lease is dropped; dropping
/// the pool waits for every lease to come back first.
pub(crate) struct ReadbackPool<B> {
    slots: Vec<Slot<B>>,
}

/// A buffer allocated for the pool and the host address its bytes are mapped
/// at, which must stay valid for the buffer's life.
pub(crate) struct Mapped<B> {
    pub buffer: B,
    pub mapped: *const u8,
}

/// A loan of one pool buffer's mapped bytes.
pub(crate) struct ReadbackLease {
    mapped: *const u8,
    len: usize,
    held: Arc<Held>,
}

// SAFETY: the lease is the only reader of the bytes it points at while it
// lives (the pool lends a buffer to one lease at a time and never writes it),
// and the pool keeps the mapping alive until the lease is dropped.
unsafe impl Send for ReadbackLease {}

impl ReadbackLease {
    /// The lent bytes.
    ///
    /// # Safety
    ///
    /// The GPU work that writes them must have completed.
    pub(crate) unsafe fn bytes(&self) -> &[u8] {
        // SAFETY: the pool allocated `len` mapped bytes at `mapped` and keeps
        // them mapped while this lease lives; the caller vouches the GPU is done
        // writing them.
        unsafe { std::slice::from_raw_parts(self.mapped, self.len) }
    }
}

impl Drop for ReadbackLease {
    fn drop(&mut self) {
        self.held.give_back();
    }
}

impl<B> Default for ReadbackPool<B> {
    fn default() -> Self {
        Self { slots: Vec::new() }
    }
}

impl<B: Clone> ReadbackPool<B> {
    /// Keep a buffer of `len` bytes ready, allocating one with `alloc` when the
    /// pool has none of that size, and release idle buffers of any other size.
    /// A no-op once one exists, lent out or not.
    pub(crate) fn reserve(
        &mut self,
        len: usize,
        alloc: impl FnOnce(usize) -> RenderResult<Mapped<B>>,
    ) -> RenderResult<()> {
        self.release_idle_other_than(len);
        if !self.slots.iter().any(|s| s.len == len) {
            self.allocate(len, alloc)?;
        }
        Ok(())
    }

    /// Lend out a buffer of `len` bytes, allocating one with `alloc` only when
    /// every buffer of that size is already lent.
    pub(crate) fn lend(
        &mut self,
        len: usize,
        alloc: impl FnOnce(usize) -> RenderResult<Mapped<B>>,
    ) -> RenderResult<(B, ReadbackLease)> {
        self.release_idle_other_than(len);
        let index = match self
            .slots
            .iter()
            .position(|s| s.len == len && !s.held.is_lent())
        {
            Some(index) => index,
            None => self.allocate(len, alloc)?,
        };
        let slot = &self.slots[index];
        slot.held.lend();
        let lease = ReadbackLease {
            mapped: slot.mapped,
            len: slot.len,
            held: Arc::clone(&slot.held),
        };
        Ok((slot.buffer.clone(), lease))
    }

    fn allocate(
        &mut self,
        len: usize,
        alloc: impl FnOnce(usize) -> RenderResult<Mapped<B>>,
    ) -> RenderResult<usize> {
        let Mapped { buffer, mapped } = alloc(len)?;
        self.slots.push(Slot {
            buffer,
            mapped,
            len,
            held: Arc::default(),
        });
        Ok(self.slots.len() - 1)
    }

    fn release_idle_other_than(&mut self, len: usize) {
        self.slots.retain(|s| s.len == len || s.held.is_lent());
    }

    #[cfg(test)]
    fn sizes(&self) -> Vec<usize> {
        self.slots.iter().map(|s| s.len).collect()
    }
}

impl<B> Drop for ReadbackPool<B> {
    fn drop(&mut self) {
        for slot in &self.slots {
            slot.held.wait_returned();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::cell::RefCell;

    // A pool of plain byte vectors standing in for mapped GPU buffers, counting
    // how many it allocated.
    struct Host {
        allocated: Cell<usize>,
        memory: RefCell<Vec<Box<[u8]>>>,
    }

    impl Host {
        fn new() -> Self {
            Self {
                allocated: Cell::new(0),
                memory: RefCell::default(),
            }
        }

        fn alloc(&self, fill: u8) -> impl FnOnce(usize) -> RenderResult<Mapped<usize>> + '_ {
            move |len| {
                let id = self.allocated.get();
                self.allocated.set(id + 1);
                let bytes = vec![fill; len].into_boxed_slice();
                let mapped = bytes.as_ptr();
                self.memory.borrow_mut().push(bytes);
                Ok(Mapped { buffer: id, mapped })
            }
        }
    }

    #[test]
    fn a_returned_buffer_is_lent_again_rather_than_reallocated() {
        let host = Host::new();
        let mut pool = ReadbackPool::default();
        let (first, lease) = pool.lend(16, host.alloc(0)).unwrap();
        drop(lease);
        let (second, _lease) = pool.lend(16, host.alloc(0)).unwrap();
        assert_eq!(first, second);
        assert_eq!(host.allocated.get(), 1);
    }

    #[test]
    fn a_buffer_still_lent_out_is_never_lent_twice() {
        let host = Host::new();
        let mut pool = ReadbackPool::default();
        let (first, _a) = pool.lend(16, host.alloc(0)).unwrap();
        let (second, _b) = pool.lend(16, host.alloc(0)).unwrap();
        assert_ne!(first, second);
        assert_eq!(host.allocated.get(), 2);
    }

    #[test]
    fn a_reserved_buffer_serves_the_first_capture() {
        let host = Host::new();
        let mut pool = ReadbackPool::default();
        pool.reserve(16, host.alloc(0)).unwrap();
        pool.reserve(16, host.alloc(0)).unwrap();
        let _lease = pool.lend(16, host.alloc(0)).unwrap();
        assert_eq!(host.allocated.get(), 1);
    }

    #[test]
    fn reserving_while_the_only_buffer_is_lent_allocates_nothing() {
        let host = Host::new();
        let mut pool = ReadbackPool::default();
        let _lease = pool.lend(16, host.alloc(0)).unwrap();
        pool.reserve(16, host.alloc(0)).unwrap();
        assert_eq!(host.allocated.get(), 1);
    }

    #[test]
    fn a_resize_releases_idle_buffers_but_keeps_lent_ones() {
        let host = Host::new();
        let mut pool = ReadbackPool::default();
        let (_, lent) = pool.lend(16, host.alloc(0)).unwrap();
        pool.reserve(32, host.alloc(0)).unwrap();
        pool.reserve(16, host.alloc(0)).unwrap();
        // The 32-byte buffer was idle when the size went back to 16.
        assert_eq!(pool.sizes(), [16]);
        drop(lent);
        pool.reserve(64, host.alloc(0)).unwrap();
        assert_eq!(pool.sizes(), [64]);
    }

    #[test]
    fn a_lease_reads_its_buffers_bytes() {
        let host = Host::new();
        let mut pool = ReadbackPool::default();
        let (_, lease) = pool.lend(4, host.alloc(7)).unwrap();
        // SAFETY: nothing writes the host stand-in's bytes.
        assert_eq!(unsafe { lease.bytes() }, [7, 7, 7, 7]);
    }

    #[test]
    fn dropping_the_pool_waits_for_a_lease_held_on_another_thread() {
        let host = Host::new();
        let mut pool = ReadbackPool::default();
        let (_, lease) = pool.lend(4, host.alloc(0)).unwrap();
        let returned = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&returned);
        let reader = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            drop(lease);
        });
        drop(pool);
        assert!(returned.load(std::sync::atomic::Ordering::SeqCst));
        reader.join().unwrap();
    }
}
