// Baked audio-clip schema.

use crate::{AssetId, PayloadLocator};
use alloc::string::String;

/// A baked audio clip: the sound an [AudioEmitter](#audioemitter) plays.
///
/// The build reads the `source` file (any format the engine can decode:
/// `.ogg`, `.wav`, `.flac`, `.mp3`) and packs it into the world.
///
/// An `AudioClip` is inert on its own: reference it from an
/// [AudioEmitter](#audioemitter)'s `clip` field to place the sound in the world.
///
/// ```jsonl
/// {"name":"fire_loop","type":"AudioClip","args":{"source":"audio/fire_crackle.ogg"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct AudioClip {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source audio file.
    pub source: String,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}
