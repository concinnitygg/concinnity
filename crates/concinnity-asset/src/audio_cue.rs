// Audio-cue schema.

use crate::{AssetId, AudioBus, AudioClipHandle, de_opt_asset_ref, de_opt_audio_clip_handle};

/// Plays audio when a [Screen](#screen) is shown.
///
/// A cue links a [Screen](#screen) to an [AudioClip](#audioclip): whenever UI
/// navigation makes the screen active (a `screen:show` or `screen:toggle` action, a
/// [KeyBinding](#keybinding), dismissing an overlay back to it, or being the
/// world's initial screen), the clip plays. Cues play flat on the main mix with
/// no 3D position; use an [AudioEmitter](#audioemitter) for positional sound.
///
/// The `kind` decides the playback behavior:
///
/// - `music`: loops until replaced. Showing a screen whose music cue is already
///   playing leaves the track running, so navigating between screens that share
///   a cue is seamless. A screen with a *different* music cue replaces the
///   track; a screen with *no* music cue leaves the current music playing.
/// - `sound`: a one-shot effect, played every time the screen is shown.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AudioCue {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Screen](#screen) whose activation triggers this cue.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
    /// The [AudioClip](#audioclip) to play.
    #[serde(deserialize_with = "de_opt_audio_clip_handle")]
    pub clip: Option<AudioClipHandle>,
    /// Playback behavior: a looping `music` track or a one-shot `sound`.
    pub kind: CueKind,
    /// Linear gain applied to the clip (1.0 leaves it unchanged).
    pub volume: f32,
    /// Mix bus the cue routes through. Defaults to `music` for a music cue
    /// and `sfx` for a sound cue; set `voice` for dialogue.
    pub bus: Option<AudioBus>,
    /// Voice priority for a `sound` cue. When all voice slots are busy, a new
    /// sound silences the oldest lowest-priority voice; a sound outranked by
    /// everything playing is skipped. Higher wins; the default is 0.
    pub priority: i32,
}

/// How an [AudioCue](#audiocue) plays its clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CueKind {
    /// Loops until a screen with a different music cue is shown. Re-triggering
    /// the currently playing clip is a no-op, so shared cues are seamless.
    Music,
    /// A one-shot effect, played on every activation of the screen.
    #[default]
    Sound,
}

impl Default for AudioCue {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            screen: None,
            clip: None,
            kind: CueKind::Sound,
            volume: 1.0,
            bus: None,
            priority: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_cue_is_a_one_shot_sound_at_unit_gain() {
        let c = AudioCue::default();
        assert_eq!(c.kind, CueKind::Sound);
        assert_eq!(c.volume, 1.0);
        assert!(c.clip.is_none());
        assert!(c.screen.is_none());
        assert!(c.bus.is_none());
        assert_eq!(c.priority, 0);
        assert_eq!(CueKind::default(), CueKind::Sound);
    }

    #[test]
    fn a_music_cue_parses_its_clip_and_screen_by_name() {
        crate::test_support::install_resolvers();
        let c: AudioCue =
            serde_json::from_str(r#"{"clip":"theme","screen":"menu","kind":"music","volume":0.4}"#)
                .unwrap();
        assert_eq!(c.clip, Some(AudioClipHandle(5)));
        assert_eq!(c.screen, Some(AssetId(4)));
        assert_eq!(c.kind, CueKind::Music);
        assert_eq!(c.volume, 0.4);
        assert_eq!(
            serde_json::to_string(&CueKind::Music).unwrap(),
            r#""music""#
        );

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: AudioCue = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.clip, Some(AudioClipHandle(5)));
        assert_eq!(back.screen, Some(AssetId(4)));
        assert_eq!(back.kind, CueKind::Music);
        assert_eq!(back.asset_id, AssetId::default());
    }

    #[test]
    fn a_voice_cue_parses_its_bus_and_priority() {
        crate::test_support::install_resolvers();
        let c: AudioCue =
            serde_json::from_str(r#"{"clip":"line","screen":"menu","bus":"voice","priority":5}"#)
                .unwrap();
        assert_eq!(c.bus, Some(AudioBus::Voice));
        assert_eq!(c.priority, 5);

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: AudioCue = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.bus, Some(AudioBus::Voice));
        assert_eq!(back.priority, 5);
    }
}
