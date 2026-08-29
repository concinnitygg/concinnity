// Positional audio-emitter schema.

use crate::components::AudioBus;
use crate::ecs::AudioClipHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;
use crate::ecs::de_opt_audio_clip_handle;

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AudioEmitter {
    /// The [AudioClip](#audioclip) this emitter plays.
    #[serde(deserialize_with = "de_opt_audio_clip_handle")]
    pub clip: Option<AudioClipHandle>,
    /// World-space position of the sound source.
    pub position: [f32; 3],
    /// Linear gain multiplier applied to the clip.
    pub volume: f32,
    /// Whether the clip restarts when it ends.
    pub looping: bool,
    /// Optional [Prop](#prop) whose position the emitter tracks each frame.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub prop: Option<AssetId>,
    /// Distance from the listener at which the sound plays at full volume.
    pub min_distance: f32,
    /// Distance from the listener beyond which the sound is inaudible. Must
    /// exceed `min_distance`.
    pub max_distance: f32,
    /// How volume falls between `min_distance` and `max_distance`.
    pub rolloff: Rolloff,
    /// Mix bus the emitter routes through. Defaults to `sfx`.
    pub bus: Option<AudioBus>,
}

/// How an [AudioEmitter](#audioemitter)'s volume falls with distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Rolloff {
    /// Natural falloff, steep near the source. The default.
    #[default]
    Logarithmic,
    /// Gradual falloff spread evenly across the range.
    Linear,
    /// No distance falloff: constant volume everywhere (panning still applies).
    None,
}

impl Default for AudioEmitter {
    fn default() -> Self {
        Self {
            clip: None,
            position: [0.0; 3],
            volume: 1.0,
            looping: true,
            prop: None,
            min_distance: 1.0,
            max_distance: 50.0,
            rolloff: Rolloff::Logarithmic,
            bus: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_emitter_loops_at_the_origin() {
        // A positional emitter is normally ambience, so it loops by default.
        let e = AudioEmitter::default();
        assert!(e.looping);
        assert_eq!(e.volume, 1.0);
        assert_eq!(e.position, [0.0, 0.0, 0.0]);
        assert!(e.clip.is_none());
        assert!(e.prop.is_none());
        assert_eq!(e.min_distance, 1.0);
        assert_eq!(e.max_distance, 50.0);
        assert_eq!(e.rolloff, Rolloff::Logarithmic);
        assert!(e.bus.is_none());
    }

    #[test]
    fn an_emitter_attached_to_a_prop_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let e: AudioEmitter = serde_json::from_str(
            r#"{"clip":"hum","prop":"lamp","position":[1,2,3],"volume":0.5,"looping":false}"#,
        )
        .unwrap();
        assert_eq!(e.clip, Some(AudioClipHandle(3)));
        assert_eq!(e.prop, Some(AssetId(4)));
        assert_eq!(e.position, [1.0, 2.0, 3.0]);
        assert!(!e.looping);

        let bytes = postcard::to_allocvec(&e).unwrap();
        let back: AudioEmitter = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.clip, Some(AudioClipHandle(3)));
        assert_eq!(back.prop, Some(AssetId(4)));
        assert_eq!(back.volume, 0.5);
    }

    #[test]
    fn authored_rolloff_and_bus_parse_and_round_trip() {
        crate::test_support::install_resolvers();
        let e: AudioEmitter = serde_json::from_str(
            r#"{"clip":"hum","min_distance":2.5,"max_distance":80.0,"rolloff":"linear","bus":"voice"}"#,
        )
        .unwrap();
        assert_eq!(e.min_distance, 2.5);
        assert_eq!(e.max_distance, 80.0);
        assert_eq!(e.rolloff, Rolloff::Linear);
        assert_eq!(e.bus, Some(crate::components::AudioBus::Voice));

        let bytes = postcard::to_allocvec(&e).unwrap();
        let back: AudioEmitter = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.max_distance, 80.0);
        assert_eq!(back.rolloff, Rolloff::Linear);
        assert_eq!(back.bus, Some(crate::components::AudioBus::Voice));

        let none: AudioEmitter = serde_json::from_str(r#"{"rolloff":"none"}"#).unwrap();
        assert_eq!(none.rolloff, Rolloff::None);
    }
}
