// Track evaluation: a list of keys, a time, and the value the track holds.

use crate::components::Ease;

/// A point one track reaches: the value it holds at `end_seconds`, and how the
/// run up to it is paced. `N` is the width of the value, 3 for a position
/// offset and 2 for a yaw / pitch pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Key<const N: usize> {
    /// The value reached at `end_seconds`.
    pub value: [f32; N],
    /// Seconds from the track's start at which the value is reached.
    pub end_seconds: f32,
    /// How the run up to this key is paced.
    pub ease: Ease,
}

/// The value a track holds `seconds` in, having started from `start`.
///
/// Within a key's span the value eases from the previous key toward this one.
/// Past the last key the track holds its final value, so a shorter track waits
/// at its end rather than snapping back.
pub(super) fn sample<const N: usize>(start: [f32; N], keys: &[Key<N>], seconds: f32) -> [f32; N] {
    let mut from = start;
    let mut from_seconds = 0.0;
    for key in keys {
        if seconds < key.end_seconds {
            let span = key.end_seconds - from_seconds;
            let t = if span > 0.0 {
                (seconds - from_seconds) / span
            } else {
                1.0
            };
            return lerp(from, key.value, key.ease.eval(t));
        }
        from = key.value;
        from_seconds = key.end_seconds;
    }
    from
}

/// Which key covers `seconds`, or `None` for an empty track. Past the last key
/// the last one still answers, so a finished track keeps reporting the segment
/// it ended in.
pub(super) fn active<const N: usize>(keys: &[Key<N>], seconds: f32) -> Option<usize> {
    let last = keys.len().checked_sub(1)?;
    Some(
        keys.iter()
            .position(|key| seconds < key.end_seconds)
            .unwrap_or(last),
    )
}

/// How long a track runs, which is its last key's end.
pub(super) fn duration<const N: usize>(keys: &[Key<N>]) -> f32 {
    keys.last().map_or(0.0, |key| key.end_seconds)
}

// Component-wise interpolation; `t` is already eased and clamped.
fn lerp<const N: usize>(a: [f32; N], b: [f32; N], t: f32) -> [f32; N] {
    let mut out = a;
    for i in 0..N {
        out[i] = a[i] + (b[i] - a[i]) * t;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> [Key<1>; 2] {
        [
            Key {
                value: [10.0],
                end_seconds: 2.0,
                ease: Ease::Linear,
            },
            Key {
                value: [30.0],
                end_seconds: 6.0,
                ease: Ease::Linear,
            },
        ]
    }

    #[test]
    fn an_empty_track_holds_its_start_and_runs_for_no_time() {
        let keys: [Key<3>; 0] = [];
        assert_eq!(sample([1.0, 2.0, 3.0], &keys, 5.0), [1.0, 2.0, 3.0]);
        assert_eq!(duration(&keys), 0.0);
        assert_eq!(active(&keys, 0.0), None);
    }

    #[test]
    fn the_first_key_is_reached_from_the_start_value() {
        let keys = keys();
        assert_eq!(sample([0.0], &keys, 0.0), [0.0]);
        assert_eq!(sample([0.0], &keys, 1.0), [5.0]);
        // A non-zero start is where the first run begins, not an offset on it.
        assert_eq!(sample([4.0], &keys, 1.0), [7.0]);
    }

    #[test]
    fn a_later_key_is_reached_from_the_one_before_it() {
        let keys = keys();
        // The second key spans 2s..6s, so half way through it is at 4s.
        assert_eq!(sample([0.0], &keys, 4.0), [20.0]);
        assert_eq!(sample([0.0], &keys, 6.0), [30.0]);
    }

    #[test]
    fn a_finished_track_holds_its_last_value() {
        let keys = keys();
        assert_eq!(sample([0.0], &keys, 60.0), [30.0]);
        assert_eq!(duration(&keys), 6.0);
    }

    #[test]
    fn a_zero_length_key_snaps_rather_than_dividing_by_its_span() {
        let keys = [
            Key {
                value: [5.0],
                end_seconds: 0.0,
                ease: Ease::Linear,
            },
            Key {
                value: [9.0],
                end_seconds: 1.0,
                ease: Ease::Linear,
            },
        ];
        // At t=0 the instant key has already ended, so the run to the second
        // key is what is in progress.
        assert_eq!(sample([0.0], &keys, 0.0), [5.0]);
        assert_eq!(sample([0.0], &keys, 0.5), [7.0]);
    }

    #[test]
    fn easing_bends_the_rate_without_moving_the_endpoints() {
        let eased = [Key {
            value: [10.0],
            end_seconds: 2.0,
            ease: Ease::InOut,
        }];
        assert_eq!(sample([0.0], &eased, 0.0), [0.0]);
        assert_eq!(sample([0.0], &eased, 2.0), [10.0]);
        // Symmetric about the midpoint, and slower than linear at the start.
        assert_eq!(sample([0.0], &eased, 1.0), [5.0]);
        assert!(sample([0.0], &eased, 0.5)[0] < 2.5);
    }

    #[test]
    fn the_active_key_tracks_the_clock_and_sticks_at_the_end() {
        let keys = keys();
        assert_eq!(active(&keys, 0.0), Some(0));
        assert_eq!(active(&keys, 1.9), Some(0));
        assert_eq!(active(&keys, 2.0), Some(1));
        assert_eq!(active(&keys, 5.9), Some(1));
        assert_eq!(active(&keys, 600.0), Some(1));
    }

    #[test]
    fn interpolation_runs_over_every_component() {
        let keys = [Key {
            value: [2.0, 4.0, 6.0],
            end_seconds: 1.0,
            ease: Ease::Linear,
        }];
        assert_eq!(sample([0.0; 3], &keys, 0.5), [1.0, 2.0, 3.0]);
    }
}
