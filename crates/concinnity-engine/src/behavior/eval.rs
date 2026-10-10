// The job pool as an evaluation scheduler: this host's answer to "how does a
// tick's firing instances get worked through". The buckets share nothing, so
// they go straight to `parallel_for` and join before the tick continues. A lone
// bucket (a tick too small to fan out) runs on the calling thread, still timed.

use concinnity_core::behavior::{EvalBucket, EvalScheduler};
use concinnity_core::profile::FanOutTiming;
use concinnity_host::thread::jobs::pool;

#[derive(Debug)]
pub(crate) struct Pool;

impl EvalScheduler for Pool {
    fn workers(&self) -> usize {
        pool().thread_count().max(1)
    }

    fn run(
        &self,
        buckets: &mut [EvalBucket],
        eval: &(dyn Fn(&mut EvalBucket) + Send + Sync),
    ) -> FanOutTiming {
        pool().parallel_for_timed(buckets, eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn a_lone_bucket_runs_on_the_calling_thread() {
        let caller = std::thread::current().id();
        let ran_on = Mutex::new(None);
        let mut buckets = [EvalBucket::default()];
        Pool.run(&mut buckets, &|_| {
            *ran_on.lock().unwrap() = Some(std::thread::current().id());
        });
        assert_eq!(ran_on.into_inner().unwrap(), Some(caller));
    }
}
