// Audio-cue schema.

use crate::components::AudioBus;
use crate::components::Screen;
use crate::components::Vocabulary;
use crate::ecs::AudioClipHandle;
use crate::ecs::Ref;

/// Plays audio when a [Screen](#screen) is shown.
///
/// A cue links a [Screen](#screen) to an [AudioClip](#audioclip): whenever UI
/// navigation makes the screen active (a `show` or `toggle` action, a
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
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct AudioCue {
    /// The [Screen](#screen) whose activation triggers this cue.
    pub screen: Option<Ref<Screen>>,
    /// The [AudioClip](#audioclip) to play.
    pub clip: Option<AudioClipHandle>,
    /// Playback behavior: a looping `music` track or a one-shot `sound`.
    pub kind: CueKind,
    /// Linear gain applied to the clip (1.0 leaves it unchanged).
    #[asset(default = 1.0)]
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
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum CueKind {
    /// A one-shot effect, played on every activation of the screen.
    #[default]
    #[vocab("sound")]
    Sound,
    /// Loops until a screen with a different music cue is shown. Re-triggering
    /// the currently playing clip is a no-op, so shared cues are seamless.
    #[vocab("music")]
    Music,
}

#[cfg(test)]
mod tests {
    use super::*;

    // NAMES is what the editor's picker offers, so it has to be what serde
    // accepts. A variant added without extending both lists fails here.
    #[test]
    fn every_cue_kind_name_is_what_serde_writes() {
        assert_eq!(CueKind::ALL.len(), CueKind::NAMES.len());
        for (kind, name) in CueKind::ALL.iter().zip(CueKind::NAMES) {
            assert_eq!(kind.as_str(), *name);
            assert_eq!(
                serde_json::to_string(kind).expect("serializes"),
                alloc::format!("\"{name}\"")
            );
        }
    }
}
