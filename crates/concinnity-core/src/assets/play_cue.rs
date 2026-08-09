// src/assets/play_cue.rs

use crate::assets::CueKind;
use crate::ecs::AudioClipHandle;

// Runtime-only event asking the audio system to play a clip: the direct
// equivalent of an AudioCue firing, for systems that trigger audio from their
// own state (the story system plays page music and one-shots this way). World
// authors never declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct PlayCue {
    pub clip: AudioClipHandle,
    pub kind: CueKind,
    pub volume: f32,
    // Voice priority for a one-shot request (higher wins a full voice pool).
    pub priority: i32,
}
