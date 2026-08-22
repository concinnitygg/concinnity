// src/assets/audio_command.rs

/// Runtime-only event sent by GraphicsSystem when a volume setting changes,
/// read by AudioSystem from its `Events<AudioCommand>` queue.
///
/// Volumes are owned by the audio engine, not the renderer, so a change made in
/// the settings menu is handed across as this event rather than read from disk
/// each frame: the audio system scales its output on the same tick. World
/// authors never declare this type directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioCommand {
    /// The output stage whose volume changed.
    pub target: AudioTarget,
    /// New volume as a linear gain (0.0 = silent, 1.0 = full).
    pub gain: f32,
}

/// The output stage an [`AudioCommand`] addresses: the master mix or one of
/// the three buses under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioTarget {
    /// The master mix every bus feeds.
    Master,
    /// The music bus.
    Music,
    /// The sound-effects bus.
    Sfx,
    /// The voice bus.
    Voice,
}
