// Waiting in a test on work another thread finishes.

use std::fmt::Display;
use std::time::{Duration, Instant};

// Long enough for a loaded CI host; a hang still fails rather than stalling.
const TIMEOUT: Duration = Duration::from_secs(10);
const PAUSE: Duration = Duration::from_millis(1);

/// Call `attempt` until it returns a value, pausing between tries, and return
/// it. Panics naming `what` once ten seconds pass without one, so work that
/// never finishes fails the test instead of hanging it.
pub fn eventually<T>(what: impl Display, attempt: impl FnMut() -> Option<T>) -> T {
    eventually_within(TIMEOUT, what, attempt)
}

fn eventually_within<T>(
    timeout: Duration,
    what: impl Display,
    mut attempt: impl FnMut() -> Option<T>,
) -> T {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(value) = attempt() {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(PAUSE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_the_first_value() {
        let mut tries = 0;
        let got = eventually("the third try", || {
            tries += 1;
            (tries == 3).then_some(tries)
        });
        assert_eq!(got, 3);
    }

    #[test]
    #[should_panic(expected = "timed out waiting for nothing")]
    fn names_what_it_waited_for_when_it_gives_up() {
        eventually_within(Duration::from_millis(20), "nothing", || None::<()>);
    }
}
