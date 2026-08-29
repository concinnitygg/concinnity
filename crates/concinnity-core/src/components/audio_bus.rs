// Audio mix-bus schema shared by the audio asset types.

/// A mix bus grouping related sounds under one user volume.
///
/// Every sound routes through one of three buses under the master output:
/// `music` for looping tracks, `sfx` for effects and positional emitters, and
/// `voice` for dialogue. Each bus has its own volume in the settings menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioBus {
    /// Looping music tracks.
    Music,
    /// Sound effects and positional emitters.
    Sfx,
    /// Dialogue and narration.
    Voice,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_names_parse_lowercase() {
        for (name, bus) in [
            ("\"music\"", AudioBus::Music),
            ("\"sfx\"", AudioBus::Sfx),
            ("\"voice\"", AudioBus::Voice),
        ] {
            let parsed: AudioBus = serde_json::from_str(name).unwrap();
            assert_eq!(parsed, bus);
            assert_eq!(serde_json::to_string(&bus).unwrap(), name);
        }
    }

    #[test]
    fn bus_round_trips_through_postcard() {
        for bus in [AudioBus::Music, AudioBus::Sfx, AudioBus::Voice] {
            let bytes = postcard::to_allocvec(&bus).unwrap();
            let back: AudioBus = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back, bus);
        }
    }
}
