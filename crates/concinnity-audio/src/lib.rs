//! A thin wrapper around the kira audio engine for 3D positional sound and
//! screen-triggered cues. kira (and its cpal / symphonia dependencies) is
//! confined to this crate: callers work entirely in the engine's `[f32; 3]`
//! representation and the schema types from concinnity-core.
//!
//! `AudioEngine` owns one kira `AudioManager`, a listener, the music / sfx /
//! voice mix buses, one spatial track per emitter, the one-shot voice pool,
//! and the background decode worker. `AudioSystem` builds it at init and
//! drives it every frame. The engine depends on this crate and constructs
//! `AudioSystem` through its system registry; the dependency arrow is
//! concinnity-audio <- concinnity-engine.

// Clip decode bookkeeping (in flight / ready / failed + deferred plays).
mod clips;
// The background decode worker thread.
mod decode;
// The kira-backed mixer facade.
mod engine;
// Contact-impulse to impact-gain mapping.
mod impact;
// Occlusion smoothing and its volume / lowpass mapping.
mod occlusion;
// Authored attenuation mapped onto kira's spatial parameters.
mod rolloff;
// The internal positional-audio system that drives `AudioEngine` from the
// world's `AudioEmitter` / `AudioCue` components.
mod system;
// The capped one-shot voice pool.
mod voices;

// The audio system the engine registry wraps.
pub use system::AudioSystem;

pub(crate) use engine::{AudioEngine, EmitterId, EmitterParams};

/// Persisted mix volumes handed to [`AudioSystem::new`] (linear gains; `None`
/// leaves a stage at unity). Resolved from the settings store by the engine's
/// audio gate, so this crate needs no dependency on the engine.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AudioVolumes {
    /// Master output volume.
    pub master: Option<f32>,
    /// Music bus volume.
    pub music: Option<f32>,
    /// Sound-effects bus volume (one-shots and positional emitters).
    pub sfx: Option<f32>,
    /// Voice / dialogue bus volume.
    pub voice: Option<f32>,
}

// A minimal 16-bit PCM mono WAV of `frames` silent samples, decodable by the
// symphonia backend without any file on disk. Shared by this crate's tests.
#[cfg(test)]
pub(crate) mod test_wav {
    pub(crate) fn pcm_wav_mono(frames: u32) -> Vec<u8> {
        let sample_rate: u32 = 44_100;
        let bits = 16u16;
        let channels = 1u16;
        let block_align = channels * bits / 8;
        let byte_rate = sample_rate * block_align as u32;
        let data_len = frames * block_align as u32;
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data_len).to_le_bytes());
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
        w.extend_from_slice(&1u16.to_le_bytes()); // audio format: PCM
        w.extend_from_slice(&channels.to_le_bytes());
        w.extend_from_slice(&sample_rate.to_le_bytes());
        w.extend_from_slice(&byte_rate.to_le_bytes());
        w.extend_from_slice(&block_align.to_le_bytes());
        w.extend_from_slice(&bits.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data_len.to_le_bytes());
        w.extend(vec![0u8; data_len as usize]);
        w
    }
}
