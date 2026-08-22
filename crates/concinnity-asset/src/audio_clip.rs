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
/// ```rust
/// # use concinnity_asset::AudioClip;
/// AudioClip {
///     source: "audio/fire_crackle.ogg".into(),
///     ..Default::default()
/// };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_clip_names_no_source() {
        let c = AudioClip::default();
        assert!(c.source.is_empty());
        assert_eq!(c.asset_id, AssetId::default());
        assert!(c.locator.is_none());
    }

    #[test]
    fn the_source_path_is_the_only_authored_field() {
        let c: AudioClip = serde_json::from_str(r#"{"source":"audio/theme.wav"}"#).unwrap();
        assert_eq!(c.source, "audio/theme.wav");
        // Identity and payload location are injected, so neither is authorable
        // nor carried on the wire.
        assert_eq!(
            serde_json::to_string(&c).unwrap(),
            r#"{"source":"audio/theme.wav"}"#
        );

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: AudioClip = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source, "audio/theme.wav");
        assert_eq!(back.asset_id, AssetId::default());
        assert!(back.locator.is_none());
    }
}
