//! Fan-outs the tests share: one that really runs units on threads, and one that
//! visits them in an order no pool would, to show an order never leaks out.

use super::FanOut;

/// Every unit of work on its own scoped thread, reporting `.0` workers. Below
/// two units there is nothing to split, so they run on the calling thread.
#[derive(Debug)]
pub struct Threads(pub usize);

impl FanOut for Threads {
    fn workers(&self) -> usize {
        self.0
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        if items.len() < 2 {
            items.iter_mut().for_each(body);
            return;
        }
        let body = &body;
        std::thread::scope(|scope| {
            for item in items.iter_mut() {
                scope.spawn(move || body(item));
            }
        });
    }
}

/// Reports `.0` workers and visits the units back to front, on the calling
/// thread.
#[derive(Debug)]
pub struct Reversed(pub usize);

impl FanOut for Reversed {
    fn workers(&self) -> usize {
        self.0
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        items.iter_mut().rev().for_each(body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // Each unit is touched exactly once however the split lands, and a split
    // of one or none runs in place.
    #[test]
    fn threads_touch_every_unit_once() {
        let mut items: Vec<u32> = (0..16).collect();
        Threads(8).for_each(&mut items, |item| *item += 1);
        assert_eq!(items, (1..17).collect::<Vec<u32>>());
        assert_eq!(Threads(8).workers(), 8);
        let mut one = [7u32];
        Threads(8).for_each(&mut one, |item| *item += 1);
        assert_eq!(one, [8]);
    }

    #[test]
    fn reversed_visits_back_to_front() {
        let mut seen = Vec::new();
        let mut items = [0u32, 1, 2];
        let order = spin::Mutex::new(&mut seen);
        Reversed(3).for_each(&mut items, |item| order.lock().push(*item));
        assert_eq!(seen, [2, 1, 0]);
    }
}
