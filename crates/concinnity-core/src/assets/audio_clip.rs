// src/assets/audio_clip.rs

use std::collections::HashSet;

use crate::assets::AudioClip;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator, PipelineContext};

impl Component for AudioClip {
    const NAME: &'static str = "AudioClip";

    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

/// Blob indices that hold an `AudioClip` payload.
///
/// `AudioSystem` reads these payloads at its `init`, but it inits *after* the
/// graphics systems, which free blob payloads once their own GPU uploads are
/// done. The graphics systems consult this so they leave the audio blobs
/// resident for `AudioSystem` to read.
pub fn audio_clip_blob_indices(ctx: &PipelineContext) -> HashSet<u32> {
    ctx.query::<AudioClip>()
        .filter_map(|c| c.locator.as_ref().map(|l| l.blob_index))
        .collect()
}

impl crate::build::SourceBacked for AudioClip {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}
