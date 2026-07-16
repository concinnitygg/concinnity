// Audio-cue schema.

use crate::{AssetId, AudioClipHandle, de_opt_asset_ref, de_opt_audio_clip_handle};

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
///
/// ```jsonl
/// {"name":"inn_theme","type":"AudioCue","args":{"screen":"page_inn","clip":"theme","kind":"music"}}
/// {"name":"door_sfx","type":"AudioCue","args":{"screen":"page_door","clip":"door_creak"}}
/// ```
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
        }
    }
}
