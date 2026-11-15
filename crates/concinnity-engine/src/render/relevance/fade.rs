//! Each effect's answer carried from frame to frame, with its share faded in
//! and out, so a view resting at a threshold never toggles an effect and the
//! effects that blend in by their share never pop crossing one.

use super::{Effect, Margins, Relevant};

/// The margin an effect that stood down must reach to come back: a quarter
/// past the threshold it fell below.
const RISE: f32 = 1.25;

/// How long, in seconds, an effect's share takes to fall once its rule fails.
/// It stands down when the share reaches 0.
const FADE_OUT_SECONDS: f32 = 0.5;

/// How long, in seconds, the share of an effect that came back takes to rise
/// to full.
const FADE_IN_SECONDS: f32 = 0.25;

/// Each effect's share, from 0 where it adds nothing to 1 in full. Ambient
/// occlusion, the bounce and the reflections are blended in by it; for the
/// other effects it only times how long a failing rule keeps them running.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fades([f32; Effect::COUNT]);

impl Fades {
    /// Every effect in full.
    pub const FULL: Self = Self([1.0; Effect::COUNT]);

    /// The share of `effect`.
    pub fn get(&self, effect: Effect) -> f32 {
        self.0[effect as usize]
    }
}

impl Default for Fades {
    /// Every effect in full.
    fn default() -> Self {
        Self::FULL
    }
}

/// Whether each effect contributes, and its share, held across frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RelevanceState {
    relevant: Relevant,
    fades: Fades,
    // The clock of the last update, to advance the fades from.
    last: Option<f32>,
}

impl RelevanceState {
    /// The effects contributing as of the last update.
    pub fn relevant(&self) -> Relevant {
        self.relevant
    }

    /// Each effect's share as of the last update; none for an effect that
    /// does not contribute.
    pub fn fades(&self) -> Fades {
        self.fades
    }

    /// Take the frame `now` seconds into the world, whose view gives
    /// `margins`, and return the effects that contribute to it.
    ///
    /// An effect whose margin falls under 1 has its share fall over
    /// [`FADE_OUT_SECONDS`] and stands down once it reaches 0; a margin back
    /// at 1 before then raises it again. One that stood down comes back the
    /// frame its margin reaches [`RISE`] at a share of 0, which rises over
    /// [`FADE_IN_SECONDS`].
    pub fn update(&mut self, margins: &Margins, now: f32) -> Relevant {
        // A clock that ran backwards advances nothing.
        let dt = self.last.map_or(0.0, |last| (now - last).max(0.0));
        self.last = Some(now);
        for effect in Effect::ALL {
            let margin = margins.get(effect);
            let fade = &mut self.fades.0[effect as usize];
            let on = if !self.relevant.contains(effect) {
                *fade = 0.0;
                margin >= RISE
            } else if margin >= 1.0 {
                *fade = (*fade + dt / FADE_IN_SECONDS).min(1.0);
                true
            } else {
                *fade = (*fade - dt / FADE_OUT_SECONDS).max(0.0);
                *fade > 0.0
            };
            self.relevant = self.relevant.with(effect, on);
        }
        self.relevant
    }

    /// Count every effect as contributing in full again, for a view that does
    /// not continue the last.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
