// src/assets/audio_emitter.rs

use crate::assets::AudioEmitter;
use crate::ecs::{AssetOrigin, Component};

impl Component for AudioEmitter {
    const NAME: &'static str = "AudioEmitter";

    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("clip", "AudioClip"), ("prop", "Prop")]
    }

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }
}
