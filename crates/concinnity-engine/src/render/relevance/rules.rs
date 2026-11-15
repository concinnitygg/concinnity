//! Each effect's rule: how far past its threshold the frame's view puts it.

use super::Effect;
use crate::components::Relevance;
use crate::render::environment::grass::lod::{GRASS_END_PIXELS, GrassReach};
use crate::render::frame::plan::EffectInputs;
use crate::render::view::{Detail, SubjectKind, ViewFacts};

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
    /// How tall the planted grass stands and how far its wind sways it.
    pub grass: Option<GrassReach>,
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
    ///
    /// Each effect measures against the surfaces in view it acts on, each at
    /// the size it shows ([`ViewFacts::details`]), and contributes while any
    /// one of them asks for it. The ambient occlusion, the bounce and the
    /// cascades act between two surfaces, so they act on a surface only while
    /// another lies within their reach of it: a prop floating far over
    /// everything else asks for none of them, however near the camera.
    pub fn of(rules: &Relevance, facts: &ViewFacts, reach: &EffectReach) -> Self {
        if !rules.enabled {
            return Self::UNBOUNDED;
        }
        // The most pixels `length` spans on a surface `acts_on` accepts.
        let widest = |length: f32, acts_on: &dyn Fn(&Detail) -> bool| {
            facts
                .details()
                .filter(|d| acts_on(d))
                .map(|d| d.pixels(length))
                .reduce(f32::max)
        };
        let reached = |reach: Option<f32>, acts_on: &dyn Fn(&Detail) -> bool| {
            ratio(
                reach.map(|r| widest(r, acts_on).unwrap_or(0.0)),
                rules.min_reach_pixels,
            )
        };
        let between = |reach: Option<f32>| {
            reached(reach, &|d: &Detail| reach.is_some_and(|r| d.clearance <= r))
        };
        let scene = &reach.scene;
        let mut out = Self::UNBOUNDED;
        out.set(Effect::AmbientOcclusion, between(reach.ambient_occlusion));
        out.set(Effect::IndirectLighting, between(reach.indirect_lighting));
        out.set(
            Effect::Reflections,
            reached(reach.reflections, &|d: &Detail| {
                d.kind != Some(SubjectKind::Water)
            }),
        );
        out.set(
            Effect::ShadowCascades,
            scene.shadow_cascades.map_or(f32::INFINITY, |r| {
                let surface = facts
                    .details()
                    .filter(|d| d.clearance <= r)
                    .map(|d| d.nearest)
                    .reduce(f32::min);
                let shaded = match (surface, scene.shadowed_media) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
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
                    widest(wave, &|d: &Detail| d.kind == Some(SubjectKind::Water)).unwrap_or(0.0)
                }),
                rules.min_wave_pixels,
            ),
        );
        let ground = |d: &Detail| grows_on(d.kind);
        out.set(
            Effect::Grass,
            ratio(
                scene
                    .grass
                    .map(|g| widest(g.blade_height, &ground).unwrap_or(0.0)),
                GRASS_END_PIXELS,
            ),
        );
        let sway = scene.grass.and_then(|g| widest(g.sway, &ground));
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

// Whether grass may grow on a surface of `kind`: the planet's ground, or the
// surfaces no subject holds, terrain among them.
fn grows_on(kind: Option<SubjectKind>) -> bool {
    matches!(kind, None | Some(SubjectKind::Planet))
}
