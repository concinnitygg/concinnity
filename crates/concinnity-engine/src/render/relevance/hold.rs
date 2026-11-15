//! Each effect's answer carried from frame to frame, so a view resting at a
//! threshold never toggles it.

use super::{Effect, Margins, Relevant};

/// The margin an effect that stood down must reach to come back: a quarter
/// past the threshold it fell below.
const RISE: f32 = 1.25;

/// How long, in seconds, an effect's rule must fail before it stands down.
const HOLD_SECONDS: f32 = 0.5;

/// Whether each effect contributes, held across frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelevanceState {
    relevant: Relevant,
    // When each relevant effect's rule began failing, if it is failing now.
    failing_since: [Option<f32>; Effect::COUNT],
}

impl Default for RelevanceState {
    /// Every effect contributing.
    fn default() -> Self {
        Self {
            relevant: Relevant::ALL,
            failing_since: [None; Effect::COUNT],
        }
    }
}

impl RelevanceState {
    /// The effects contributing as of the last update.
    pub fn relevant(&self) -> Relevant {
        self.relevant
    }

    /// Take the frame `now` seconds into the world, whose view gives
    /// `margins`, and return the effects that contribute to it. An effect
    /// comes back the frame its margin reaches [`RISE`], and stands down once
    /// its margin has stayed under 1 for [`HOLD_SECONDS`].
    pub fn update(&mut self, margins: &Margins, now: f32) -> Relevant {
        for effect in Effect::ALL {
            let margin = margins.get(effect);
            let failing = &mut self.failing_since[effect as usize];
            let was = self.relevant.contains(effect);
            let on = if !was {
                margin >= RISE
            } else if margin >= 1.0 {
                true
            } else {
                // A clock that ran backwards starts the wait over.
                let since = failing.filter(|&s| s <= now).unwrap_or(now);
                *failing = Some(since);
                now - since < HOLD_SECONDS
            };
            if !(was && on && margin < 1.0) {
                *failing = None;
            }
            self.relevant = self.relevant.with(effect, on);
        }
        self.relevant
    }

    /// Count every effect as contributing again, for a view that does not
    /// continue the last.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
