// Positional audio-emitter schema.

use crate::{AssetId, AudioClipHandle, de_opt_asset_ref, de_opt_audio_clip_handle};

/// A point source of sound in the world.
///
/// Plays its `clip` (an [AudioClip](#audioclip) reference) from `position`,
/// attenuated and panned relative to the camera. When `prop` names a
/// [Prop](#prop), the emitter tracks that prop's position every frame, so the
/// sound follows a moving object.
///
/// ```jsonl
/// {"name":"fire_sound","type":"AudioEmitter","args":{"clip":"fire_loop","position":[6.0,4.0,-6.0]}}
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
}

impl Default for AudioEmitter {
    fn default() -> Self {
        Self {
            clip: None,
            position: [0.0; 3],
            volume: 1.0,
            looping: true,
            prop: None,
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
}
