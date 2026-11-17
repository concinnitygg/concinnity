//! Each effect's answer carried from frame to frame, with its share faded in
//! and out, so a view resting at a threshold never toggles an effect and the
//! effects that blend in by their share never pop crossing one.

use super::scope::ReflectionScope;
use super::{Effect, Margins, Relevant};
use crate::render::view::ScreenRect;

/// The margin an effect that stood down must reach to come back: a quarter
/// past the threshold it fell below.
const RISE: f32 = 1.25;

/// How long, in seconds, an effect's share takes to fall once its rule fails,
/// and how long its rule must keep failing for it to stand down.
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

// One rule's answer carried from frame to frame.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Track {
    on: bool,
    fade: f32,
    // How long the rule has failed without a break, in seconds.
    failing: f32,
}

impl Default for Track {
    fn default() -> Self {
        Self {
            on: true,
            fade: 1.0,
            failing: 0.0,
        }
    }
}

impl Track {
    // Take a frame `dt` seconds after the last whose rule stands at `margin`.
    fn step(&mut self, margin: f32, dt: f32) {
        if !self.on {
            self.fade = 0.0;
            self.failing = 0.0;
            self.on = margin >= RISE;
        } else if margin >= 1.0 {
            self.fade = (self.fade + dt / FADE_IN_SECONDS).min(1.0);
            self.failing = 0.0;
        } else {
            self.fade = (self.fade - dt / FADE_OUT_SECONDS).max(0.0);
            self.failing += dt;
            self.on = self.failing < FADE_OUT_SECONDS;
        }
    }
}

/// Whether each effect contributes, and its share, held across frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelevanceState {
    effects: [Track; Effect::COUNT],
    // The reflections on the surfaces that can fill the screen, and on the
    // prop, each held like an effect; the reflections run while either does.
    whole_reflections: Track,
    prop_reflections: Track,
    prop_rect: ScreenRect,
    // The clock of the last update, to advance the fades from.
    last: Option<f32>,
}

impl Default for RelevanceState {
    /// Every effect contributing in full, the reflections everywhere.
    fn default() -> Self {
        Self {
            effects: [Track::default(); Effect::COUNT],
            whole_reflections: Track::default(),
            prop_reflections: Track::default(),
            prop_rect: ScreenRect::EMPTY,
            last: None,
        }
    }
}

impl RelevanceState {
    /// The effects contributing as of the last update.
    pub fn relevant(&self) -> Relevant {
        Effect::ALL.into_iter().fold(Relevant::NONE, |set, effect| {
            set.with(effect, self.effects[effect as usize].on)
        })
    }

    /// Each effect's share as of the last update; none for an effect that
    /// does not contribute.
    pub fn fades(&self) -> Fades {
        Fades(self.effects.map(|t| t.fade))
    }

    /// The share of each pixel the reflections show on as of the last
    /// update.
    pub fn reflection_scope(&self) -> ReflectionScope {
        ReflectionScope {
            whole: self.whole_reflections.fade,
            prop: self.prop_reflections.fade,
            rect: self.prop_rect,
        }
    }

    /// Take the frame `now` seconds into the world, whose view gives
    /// `margins`, and return the effects that contribute to it.
    ///
    /// An effect whose margin falls under 1 has its share fall over
    /// [`FADE_OUT_SECONDS`] and stands down once its margin has stayed under
    /// 1 that long; a margin back at 1 before then raises it again. One that
    /// stood down comes back the frame its margin reaches [`RISE`] at a share
    /// of 0, which rises over [`FADE_IN_SECONDS`], so a rule that holds only
    /// now and then keeps its effect on rather than toggling it. The
    /// reflections on each part of the screen are held the same way, and the
    /// reflections run while either part holds them, so the two never
    /// disagree.
    pub fn update(&mut self, margins: &Margins, now: f32) -> Relevant {
        // A clock that ran backwards advances nothing.
        let dt = self.last.map_or(0.0, |last| (now - last).max(0.0));
        self.last = Some(now);
        for effect in Effect::ALL {
            if effect != Effect::Reflections {
                self.effects[effect as usize].step(margins.get(effect), dt);
            }
        }
        let reflections = &margins.reflections;
        let (whole, prop) = (&mut self.whole_reflections, &mut self.prop_reflections);
        whole.step(reflections.whole, dt);
        prop.step(reflections.prop, dt);
        self.effects[Effect::Reflections as usize] = Track {
            on: whole.on || prop.on,
            fade: whole.fade.max(prop.fade),
            failing: whole.failing.min(prop.failing),
        };
        self.prop_rect = reflections.rect;
        self.relevant()
    }

    /// Count every effect as contributing in full again, for a view that does
    /// not continue the last.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
