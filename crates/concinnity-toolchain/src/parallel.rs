// Running a build script's independent compiles side by side. Each one is a
// subprocess pipeline that spends most of its time waiting on dxc or the Metal
// toolchain, so one worker per core keeps them all busy.

use std::sync::atomic::{AtomicUsize, Ordering};

/// `f` over every item on a pool of scoped threads, the results in the order
/// of `items` however the work was interleaved, so whatever a caller generates
/// from them is deterministic.
pub fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(items.len());
    if workers <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, R)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(index) else {
                            return done;
                        };
                        done.push((index, f(item)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|e| std::panic::resume_unwind(e))
            })
            .collect()
    });
    results.sort_unstable_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_keep_the_order_of_the_items() {
        let items: Vec<u32> = (0..257).collect();
        let doubled = parallel_map(&items, |n| {
            // Uneven work, so completion order differs from item order.
            if n % 7 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            n * 2
        });
        assert_eq!(doubled, items.iter().map(|n| n * 2).collect::<Vec<_>>());
    }

    #[test]
    fn every_item_runs_exactly_once() {
        let calls = AtomicUsize::new(0);
        let items = [(); 100];
        let results = parallel_map(&items, |()| calls.fetch_add(1, Ordering::Relaxed));
        assert_eq!(calls.load(Ordering::Relaxed), 100);
        let mut seen = results;
        seen.sort_unstable();
        assert_eq!(seen, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn no_items_yield_no_results() {
        let results: Vec<u8> = parallel_map(&[] as &[u8], |_| unreachable!());
        assert!(results.is_empty());
    }

    // A panicking compile has to fail the build it ran in, not vanish with its
    // worker.
    #[test]
    #[should_panic(expected = "item 3")]
    fn a_panic_in_one_item_reaches_the_caller() {
        let items: Vec<u32> = (0..8).collect();
        parallel_map(&items, |n| assert_ne!(*n, 3, "item 3"));
    }
}
