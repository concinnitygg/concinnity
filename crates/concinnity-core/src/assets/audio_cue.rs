// src/assets/audio_cue.rs

use crate::assets::AudioCue;
use crate::ecs::{AssetOrigin, Component};

impl Component for AudioCue {
    const NAME: &'static str = "AudioCue";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("clip", "AudioClip"), ("view", "View")]
    }

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
