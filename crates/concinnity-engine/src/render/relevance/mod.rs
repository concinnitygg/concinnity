//! Whether each effect contributes to what the camera sees this frame:
//! [`Relevant`].
//!
//! Each effect has a physical reach ([`EffectReach`]): how far from a surface
//! its ambient occlusion searches, how far its bounce or its reflections
//! trace, how far its cascades shadow, how long the ocean's longest wave is.
//! The frame's [`ViewFacts`](crate::render::view::ViewFacts) say how far away
//! the nearest surface it acts on lies, so the reach can be measured in
//! pixels and held against the [`Relevance`](crate::components::Relevance)
//! thresholds. [`RelevanceState`] carries each effect's answer from frame to
//! frame: an effect comes back once its rule holds with a quarter to spare,
//! and stands down only once its rule has failed for half a second, so it
//! never flickers at a threshold.

mod hold;
mod media;
mod rules;
#[cfg(test)]
mod tests;

pub use hold::RelevanceState;
pub use media::shadowed_media;
pub use rules::{EffectReach, Margins, SceneReach};

/// An effect whose passes stand down where it does not contribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    /// Screen-space ambient occlusion.
    AmbientOcclusion,
    /// The screen-space bounce's trace.
    IndirectLighting,
    /// The ray-traced or screen-space reflections on opaque surfaces.
    Reflections,
    /// The sun's cascaded shadows.
    ShadowCascades,
    /// The reflection probe a prop rides.
    PropProbe,
    /// The ocean simulation.
    WaterSim,
    /// The disocclusion mark a temporal upscaler reads.
    Disocclusion,
}

impl Effect {
    /// How many effects there are.
    pub const COUNT: usize = 7;

    /// Every effect, in declaration order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::AmbientOcclusion,
        Self::IndirectLighting,
        Self::Reflections,
        Self::ShadowCascades,
        Self::PropProbe,
        Self::WaterSim,
        Self::Disocclusion,
    ];

    /// The effect's name in reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::AmbientOcclusion => "ambient_occlusion",
            Self::IndirectLighting => "indirect_lighting",
            Self::Reflections => "reflections",
            Self::ShadowCascades => "shadow_cascades",
            Self::PropProbe => "prop_probe",
            Self::WaterSim => "water_sim",
            Self::Disocclusion => "disocclusion",
        }
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// The effects that contribute to this frame's view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relevant(u8);

impl Relevant {
    /// Every effect.
    pub const ALL: Self = Self((1 << Effect::COUNT) - 1);

    /// No effect.
    pub const NONE: Self = Self(0);

    /// Whether `effect` contributes.
    pub fn contains(self, effect: Effect) -> bool {
        self.0 & effect.bit() != 0
    }

    /// The set with `effect` contributing or not.
    #[must_use]
    pub fn with(self, effect: Effect, on: bool) -> Self {
        if on {
            Self(self.0 | effect.bit())
        } else {
            Self(self.0 & !effect.bit())
        }
    }
}

impl Default for Relevant {
    /// Every effect: what a frame with nothing measured assumes.
    fn default() -> Self {
        Self::ALL
    }
}
