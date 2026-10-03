// Positional audio-emitter schema.

use crate::components::AudioBus;
use crate::components::Prop;
use crate::components::Vocabulary;
use crate::ecs::AudioClipHandle;
use crate::ecs::Ref;

/// A point source of sound in the world.
///
/// Plays its `clip` (an [AudioClip](#audioclip) reference) from `position`,
/// attenuated and panned relative to the camera. When `prop` names a
/// [Prop](#prop), the emitter tracks that prop's position every frame, so the
/// sound follows a moving object.
///
/// The sound is at full volume inside `min_distance`, fades according to
/// `rolloff` between `min_distance` and `max_distance`, and is inaudible
/// beyond `max_distance`.
///
/// ```rust
/// # use concinnity_core::components::AudioEmitter;
/// AudioEmitter {
///     position: [6.0, 4.0, -6.0],
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct AudioEmitter {
    /// The [AudioClip](#audioclip) this emitter plays.
    pub clip: Option<AudioClipHandle>,
    /// World-space position of the sound source.
    pub position: [f32; 3],
    /// Linear gain multiplier applied to the clip.
    #[asset(default = 1.0)]
    pub volume: f32,
    /// Whether the clip restarts when it ends.
    // A positional emitter is normally ambience, so it loops by default.
    #[asset(default = true)]
    pub looping: bool,
    /// Optional [Prop](#prop) whose position the emitter tracks each frame.
    pub prop: Option<Ref<Prop>>,
    /// Distance from the listener at which the sound plays at full volume.
    #[asset(default = 1.0)]
    pub min_distance: f32,
    /// Distance from the listener beyond which the sound is inaudible. Must
    /// exceed `min_distance`.
    #[asset(default = 50.0)]
    pub max_distance: f32,
    /// How volume falls between `min_distance` and `max_distance`.
    pub rolloff: Rolloff,
    /// Mix bus the emitter routes through. Defaults to `sfx`.
    pub bus: Option<AudioBus>,
}

/// How an [AudioEmitter](#audioemitter)'s volume falls with distance.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum Rolloff {
    /// Natural falloff, steep near the source. The default.
    #[default]
    #[vocab("logarithmic")]
    Logarithmic,
    /// Gradual falloff spread evenly across the range.
    #[vocab("linear")]
    Linear,
    /// No distance falloff: constant volume everywhere (panning still applies).
    #[vocab("none")]
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolloff_names_parse_in_lowercase() {
        let r = |s: &str| serde_json::from_str::<Rolloff>(s).unwrap();
        assert_eq!(r(r#""logarithmic""#), Rolloff::Logarithmic);
        assert_eq!(r(r#""linear""#), Rolloff::Linear);
        assert_eq!(r(r#""none""#), Rolloff::None);
    }
}
