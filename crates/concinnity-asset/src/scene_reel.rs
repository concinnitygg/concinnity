// Scene-playlist schema.

use crate::AssetId;
use alloc::vec::Vec;

/// An ordered playlist of named [Scene](#scene)s.
///
/// The current scene's [Prop](#prop)s are shown, then it advances to the next
/// based on that scene's `duration_secs`. Timing and transition style are
/// declared on each [Scene](#scene) asset. Props not prefixed by any scene name
/// remain visible in all scenes.
///
/// ```jsonl
/// {"name":"day",  "type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
/// {"name":"night","type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
/// {"name":"day_sun",  "type":"Prop","args":{"model":"model_sun_disc","position":[0,80,-200]}}
/// {"name":"night_moon","type":"Prop","args":{"model":"model_moon_disc","position":[0,80,-200]}}
/// {"name":"reel","type":"SceneReel","args":{"looping":true,"scenes":["day","night"]}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SceneReel {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Ordered list of [Scene](#scene) assets to play.
    pub scenes: Vec<AssetId>,
    /// When true, wraps back to the first scene after the last one ends.
    pub looping: bool,
    /// Index of the entry that is active at world start.
    pub start_index: u32,
}
