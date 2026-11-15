//! Each effect's rule: how far past its threshold the frame's view puts it.

use super::Effect;
use crate::components::Relevance;
use crate::gfx::render_types::ViewSampleGrid;
use crate::render::frame::plan::EffectInputs;
use crate::render::view::{Seen, SubjectKind, ViewFacts};

/// Each effect's physical reach this frame, in meters, from its own settings
/// and the world's content; `None` for an effect that does not run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EffectReach {
    /// How far the ambient occlusion searches for occluders.
    pub ambient_occlusion: Option<f32>,
    /// How far the screen-space bounce traces.
    pub indirect_lighting: Option<f32>,
    /// How far the reflections trace.
    pub reflections: Option<f32>,
    /// What the world holds that the effects reach.
    pub scene: SceneReach,
}

/// The reach the world's content gives the effects this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneReach {
    /// How far from the camera the cascades shade.
    pub shadow_cascades: Option<f32>,
    /// How far from the camera the nearest medium the cascades shade lies:
    /// fog thick enough within their reach at 0, a raymarched volume in view
    /// at its distance.
    pub shadowed_media: Option<f32>,
    /// The longest wave the ocean simulation shows.
    pub longest_wave: Option<f32>,
    /// How far the tip of a planted blade can move between two frames, as
    /// its wind sways it.
    pub grass_sway: Option<f32>,
}

impl EffectReach {
    /// The reach of the screen-space effects `effects` holds the live
    /// settings of, over what the world holds, `scene`. Where the bounce's
    /// resolve `folds_occlusion` in, the occlusion is read as far as the
    /// bounce reaches; the reflections reach as far as the longer of their
    /// traces.
    pub fn of(effects: &EffectInputs, folds_occlusion: bool, scene: SceneReach) -> Self {
        let longer = |a: Option<f32>, b: Option<f32>| match (a, b) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let bounce = effects.ssgi.map(|(s, _)| s.max_distance);
        Self {
            ambient_occlusion: longer(
                effects.ssao.map(|s| s.radius),
                bounce.filter(|_| folds_occlusion),
            ),
            indirect_lighting: bounce,
            reflections: longer(
                effects.rt.map(|s| s.max_distance),
                effects.ssr.map(|s| s.max_distance),
            ),
            scene,
        }
    }
}

/// How far past its threshold each effect's measure stands this frame: 1 at
/// the threshold, above it where the effect contributes, infinite where no
/// rule applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Margins([f32; Effect::COUNT]);

impl Margins {
    /// Every effect contributing without condition.
    pub const UNBOUNDED: Self = Self([f32::INFINITY; Effect::COUNT]);

    /// The margins `rules` give the view `facts` describes, for effects
    /// reaching as far as `reach`.
    pub fn of(rules: &Relevance, facts: &ViewFacts, reach: &EffectReach) -> Self {
        if !rules.enabled {
            return Self::UNBOUNDED;
        }
        let view = Surfaces::new(facts);
        let opaque = |kind| kind != SubjectKind::Water;
        let reached = |reach: Option<f32>, nearest: Option<f32>| {
            ratio(
                reach.map(|r| nearest.map_or(0.0, |d| facts.pixels(r, d))),
                rules.min_reach_pixels,
            )
        };
        let near_any = view.nearest(|_| true, true);
        let ground = view.nearest(|kind| kind == SubjectKind::Planet, true);
        let scene = &reach.scene;
        let mut out = Self::UNBOUNDED;
        out.set(
            Effect::AmbientOcclusion,
            reached(reach.ambient_occlusion, near_any),
        );
        out.set(
            Effect::IndirectLighting,
            reached(reach.indirect_lighting, near_any),
        );
        out.set(
            Effect::Reflections,
            reached(reach.reflections, view.nearest(opaque, true)),
        );
        let shaded = match (near_any, scene.shadowed_media) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        out.set(
            Effect::ShadowCascades,
            scene.shadow_cascades.map_or(f32::INFINITY, |r| {
                shaded.map_or(0.0, |d| r / d.max(f32::MIN_POSITIVE))
            }),
        );
        out.set(
            Effect::PropProbe,
            prop_margin(facts, rules.min_subject_pixels),
        );
        out.set(
            Effect::WaterSim,
            ratio(
                scene.longest_wave.map(|wave| {
                    view.nearest(|kind| kind == SubjectKind::Water, false)
                        .map_or(0.0, |d| facts.pixels(wave, d))
                }),
                rules.min_wave_pixels,
            ),
        );
        let sway = scene
            .grass_sway
            .and_then(|sway| ground.map(|d| facts.pixels(sway, d)));
        out.set(
            Effect::Disocclusion,
            motion_margin(facts, sway, rules.min_motion_pixels),
        );
        out
    }

    /// The margin of `effect`.
    pub fn get(&self, effect: Effect) -> f32 {
        self.0[effect as usize]
    }

    /// The margins with `effect`'s at `margin`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with(mut self, effect: Effect, margin: f32) -> Self {
        self.set(effect, margin);
        self
    }

    fn set(&mut self, effect: Effect, margin: f32) {
        self.0[effect as usize] = margin;
    }
}

// `measure` against `threshold`: unbounded without a measure or a threshold.
fn ratio(measure: Option<f32>, threshold: f32) -> f32 {
    match measure {
        Some(m) if threshold > 0.0 => m / threshold,
        _ => f32::INFINITY,
    }
}

// The probed prop's size on screen against `min_pixels`: none out of view,
// unbounded without a prop.
fn prop_margin(facts: &ViewFacts, min_pixels: f32) -> f32 {
    let prop = facts.subjects().find(|s| s.kind == SubjectKind::Prop);
    ratio(
        prop.map(|p| {
            if p.in_view {
                p.extent_px(facts.focal_px)
            } else {
                0.0
            }
        }),
        min_pixels,
    )
}

// How far the view's surfaces moved against it, with grass swaying `sway`
// pixels in view, against `min_pixels`: unbounded until a measurement says how
// far its nearest still surface is.
fn motion_margin(facts: &ViewFacts, sway: Option<f32>, min_pixels: f32) -> f32 {
    if facts.seen.is_none() {
        return f32::INFINITY;
    }
    let moved = facts.parallax_px.max(facts.mover_px);
    ratio(Some(sway.map_or(moved, |s| moved.max(s))), min_pixels)
}

// The surfaces a frame's view holds, from its latest measurement while that
// still stands for the view, else from where its subjects lie.
struct Surfaces<'a> {
    facts: &'a ViewFacts,
    // The measurement, while the view has moved less than a sample cell since
    // it was taken.
    standing: Option<Seen>,
}

impl<'a> Surfaces<'a> {
    fn new(facts: &'a ViewFacts) -> Self {
        let [width, height] = facts.render_size;
        let cell = ViewSampleGrid::over(width, height).cell as f32;
        let standing = facts.seen.filter(|seen| seen.drift_px <= cell);
        Self { facts, standing }
    }

    // How far the nearest surface in view lies, among the subjects `acts_on`
    // accepts and, with `other`, the surfaces no subject holds; `None` when
    // none of them is in view.
    //
    // A subject in view counts at its nearest seen part while the measurement
    // stands and saw it, else at its nearest point, seen or not. The surfaces
    // no subject holds count where the measurement saw them, and at the
    // camera itself when it no longer stands but saw some, or when nothing
    // has been measured.
    fn nearest(&self, acts_on: impl Fn(SubjectKind) -> bool, other: bool) -> Option<f32> {
        let subjects = self
            .facts
            .subjects()
            .filter(|s| s.in_view && acts_on(s.kind))
            .map(|s| match (self.standing, s.seen.nearest) {
                (Some(_), Some(seen)) => seen,
                _ => s.distance,
            });
        let others = match (other, self.facts.seen, self.standing) {
            (false, _, _) => None,
            (true, None, _) => Some(0.0),
            (true, Some(_), Some(standing)) => standing.other.nearest,
            (true, Some(seen), None) => seen.other.nearest.map(|_| 0.0),
        };
        subjects.chain(others).reduce(f32::min)
    }
}
