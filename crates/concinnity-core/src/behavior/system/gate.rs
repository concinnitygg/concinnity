// The fan-out gate for a tick's behavior runs.

use crate::profile::FanOutGate;

// Below this much estimated work a split does not pay. Measured on a 12-core
// Mac with the workers asleep between ticks, as they are at frame rate: the
// fan-out loses below about 0.3 ms of work, breaks even near 0.4 ms, and halves
// the wait from about 0.8 ms.
const MIN_FANOUT_WORK_NS: u64 = 500_000;

// Whether a tick's runs are worth fanning out, from what a run cost the last
// time the host measured one.
pub(super) type EvalGate = FanOutGate<MIN_FANOUT_WORK_NS>;
