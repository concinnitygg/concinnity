// src/gfx/root_motion.rs
//
// Root-motion track: the character-displacement curve stripped out of a
// clip's root joint at build time. The pose keeps the root anchored in
// place; the runtime samples this track's frame-to-frame delta instead and
// feeds it to whatever moves the character (a physics capsule, or the mesh
// transform directly). Pure math, unit-tested here.

/// One key of a root-motion curve: the root joint's stripped translation at
/// `time` seconds from the clip start.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct RootKey {
    pub time: f32,
    pub translation: [f32; 3],
}

// The displacement curve of one clip, in the mesh's model space. Keys are in
// ascending time order (the build bakes them from the root joint's keyframe
// track, so they inherit its ordering).
#[derive(Debug, Clone, Default)]
pub struct RootTrack {
    pub keys: Vec<RootKey>,
}

impl RootTrack {
    // The curve's translation at clip-local time `t`, clamped to the key
    // range; between keys the translation lerps.
    pub fn sample(&self, t: f32) -> [f32; 3] {
        match self.keys.as_slice() {
            [] => [0.0; 3],
            [only] => only.translation,
            keys => {
                if t <= keys[0].time {
                    return keys[0].translation;
                }
                let last = keys[keys.len() - 1];
                if t >= last.time {
                    return last.translation;
                }
                for w in keys.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if t >= a.time && t <= b.time {
                        let span = (b.time - a.time).max(1e-6);
                        let f = (t - a.time) / span;
                        return lerp3(a.translation, b.translation, f);
                    }
                }
                last.translation
            }
        }
    }

    // The displacement covered between two *unwrapped* clip times
    // (`t0 <= t1`, in the same seconds the clip clock runs on). A looping
    // clip adds one full per-cycle displacement for every wrap crossed, so a
    // multi-loop frame (or a long hitch) loses no ground; a non-looping clip
    // clamps both ends.
    pub fn delta(&self, t0: f32, t1: f32, duration: f32, looping: bool) -> [f32; 3] {
        if !looping || duration <= 1e-6 {
            return sub3(self.sample(t1), self.sample(t0));
        }
        let cycles = (t1 / duration).floor() - (t0 / duration).floor();
        let per_cycle = sub3(self.sample(duration), self.sample(0.0));
        let within = sub3(
            self.sample(t1.rem_euclid(duration)),
            self.sample(t0.rem_euclid(duration)),
        );
        [
            within[0] + cycles * per_cycle[0],
            within[1] + cycles * per_cycle[1],
            within[2] + cycles * per_cycle[2],
        ]
    }
}

pub fn lerp3(a: [f32; 3], b: [f32; 3], f: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1s clip walking +2 on X per cycle, linear.
    fn walk_x() -> RootTrack {
        RootTrack {
            keys: vec![
                RootKey {
                    time: 0.0,
                    translation: [0.0; 3],
                },
                RootKey {
                    time: 1.0,
                    translation: [2.0, 0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn sample_clamps_and_lerps() {
        let track = walk_x();
        assert_eq!(track.sample(-1.0), [0.0; 3]);
        assert_eq!(track.sample(2.0), [2.0, 0.0, 0.0]);
        assert!((track.sample(0.25)[0] - 0.5).abs() < 1e-6);
        assert_eq!(RootTrack::default().sample(0.5), [0.0; 3]);
    }

    #[test]
    fn delta_within_one_cycle() {
        let track = walk_x();
        let d = track.delta(0.25, 0.75, 1.0, true);
        assert!((d[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn delta_across_a_wrap_adds_the_cycle_displacement() {
        let track = walk_x();
        // 0.75 -> 1.25 covers the wrap: 0.5s of walking = +1.0 X.
        let d = track.delta(0.75, 1.25, 1.0, true);
        assert!((d[0] - 1.0).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn delta_across_multiple_wraps_loses_no_ground() {
        let track = walk_x();
        // 3.4 cycles = +6.8 X.
        let d = track.delta(0.1, 3.5, 1.0, true);
        assert!((d[0] - 6.8).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn non_looping_delta_clamps_at_the_end() {
        let track = walk_x();
        let d = track.delta(0.5, 5.0, 1.0, false);
        assert!((d[0] - 1.0).abs() < 1e-6, "holds the final key: {d:?}");
    }

    #[test]
    fn zero_duration_degrades_to_clamped_endpoint_sampling() {
        let track = walk_x();
        // A degenerate duration cannot wrap; the delta falls back to plain
        // clamped endpoint sampling instead of dividing by zero.
        let d = track.delta(0.0, 1.0, 0.0, true);
        assert!((d[0] - 2.0).abs() < 1e-6, "{d:?}");
    }
}
