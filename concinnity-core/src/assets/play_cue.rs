// src/assets/play_cue.rs

use crate::assets::CueKind;
use crate::ecs::asset_id::AssetId;

// Runtime-only event asking the audio system to play a clip: the direct
// equivalent of an AudioCue firing, for systems that trigger audio from their
// own state (the story system plays page music and one-shots this way). World
// authors never declare this type directly.
#[derive(Debug, Clone, Copy)]
pub struct PlayCue {
    pub clip: AssetId,
    pub kind: CueKind,
    pub volume: f32,
}
