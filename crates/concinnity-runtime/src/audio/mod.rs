//! The kira adapter behind core's audio: the [`AudioOut`] a world's
//! `AudioSystem` plays through, its background decode thread, and authored
//! attenuation mapped onto kira's spatial tracks. kira (and its cpal /
//! symphonia dependencies) is confined to this module.
//!
//! [`AudioOut`]: concinnity_engine::audio::out::AudioOut

// The background decode worker thread.
mod decode;
// The kira-backed mixer.
mod out;
// Authored attenuation mapped onto kira's spatial parameters.
mod rolloff;

pub(crate) use out::open;
