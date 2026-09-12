// The job pool as an evaluation scheduler: this host's answer to "how does a
// tick's firing instances get worked through". The buckets share nothing, so
// they go straight to `parallel_for` and join before the tick continues.

use concinnity_core::behavior::{EvalBucket, EvalScheduler};
use concinnity_host::thread::jobs::pool;

#[derive(Debug)]
pub(crate) struct Pool;

impl EvalScheduler for Pool {
    fn workers(&self) -> usize {
        pool().thread_count().max(1)
    }

    fn run(&self, buckets: &mut [EvalBucket], eval: &(dyn Fn(&mut EvalBucket) + Send + Sync)) {
        pool().parallel_for(buckets, eval);
    }
}
