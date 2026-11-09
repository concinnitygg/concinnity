use super::*;
use concinnity_engine::components::Rolloff;
use concinnity_testing::fixtures::wav::pcm16_mono;
use kira::backend::mock::MockBackend;

// The real mixer on kira's mock backend: no device, fully headless.
type MockOut = KiraOut<MockBackend>;

fn mock_out() -> MockOut {
    MockOut::start().expect("the mock backend always starts")
}

impl MockOut {
    // Let the mock renderer consume queued mixer commands and process a chunk
    // of audio, so handle-visible state advances.
    fn pump(&mut self) {
        self.manager.backend_mut().on_start_processing();
        self.manager.backend_mut().process();
    }

    fn bus_sounds(&self, bus: AudioBus) -> usize {
        match bus {
            AudioBus::Music => self.buses.music.num_sounds(),
            AudioBus::Sfx => self.buses.sfx.num_sounds(),
            AudioBus::Voice => self.buses.voice.num_sounds(),
        }
    }

    // Decode `key` and wait for it; the worker is a real thread.
    fn ready(&mut self, key: u64, bytes: Vec<u8>) -> Result<(), String> {
        assert!(self.decode(key, bytes));
        concinnity_testing::eventually(format_args!("clip {key} to decode"), || self.decoded()).1
    }
}

const ACOUSTICS: Acoustics = Acoustics {
    bus: AudioBus::Sfx,
    distances: (2.0, 80.0),
    rolloff: Rolloff::Linear,
};

const OPEN: f64 = 20_000.0;

#[test]
fn clips_play_on_the_bus_they_name() {
    let mut out = mock_out();
    out.ready(1, pcm16_mono(44_100, &[0; 64])).expect("decodes");
    assert!(
        out.play(1, PlayOn::Bus(AudioBus::Voice), 0.0, false)
            .is_some()
    );
    assert!(
        out.play(1, PlayOn::Bus(AudioBus::Sfx), 0.0, false)
            .is_some()
    );
    assert_eq!(out.bus_sounds(AudioBus::Voice), 1);
    assert_eq!(out.bus_sounds(AudioBus::Sfx), 1);
    assert_eq!(out.bus_sounds(AudioBus::Music), 0);
}

#[test]
fn an_undecoded_clip_does_not_play() {
    let mut out = mock_out();
    assert!(
        out.play(4, PlayOn::Bus(AudioBus::Sfx), 0.0, false)
            .is_none()
    );
    assert!(out.ready(7, b"not an audio file".to_vec()).is_err());
    assert!(
        out.play(7, PlayOn::Bus(AudioBus::Sfx), 0.0, false)
            .is_none()
    );
}

#[test]
fn a_finished_voice_reads_finished_and_a_stopped_one_is_forgotten() {
    let mut out = mock_out();
    out.ready(1, pcm16_mono(44_100, &[0; 64])).expect("decodes");
    let voice = out
        .play(1, PlayOn::Bus(AudioBus::Sfx), 0.0, false)
        .expect("plays");
    assert!(!out.finished(voice));
    // The mock renderer runs at 1 Hz, so each pump processes 2 seconds: a
    // 64-frame clip finishes almost immediately.
    concinnity_testing::eventually("the voice to finish", || {
        out.pump();
        out.finished(voice).then_some(())
    });
    assert!(out.voices.is_empty(), "a finished voice is forgotten");
    let looping = out
        .play(1, PlayOn::Bus(AudioBus::Music), 0.0, true)
        .expect("plays");
    out.stop(looping);
    assert!(out.finished(looping));
}

#[test]
fn emitters_join_their_bus_and_take_plays() {
    let mut out = mock_out();
    assert!(out.add_emitter(EmitterId(2), [1.0, 0.0, 0.0], &ACOUSTICS, OPEN));
    assert_eq!(out.buses.sfx.num_sub_tracks(), 1, "spatial track under sfx");
    assert_eq!(out.buses.music.num_sub_tracks(), 0);
    out.ready(9, pcm16_mono(44_100, &[0; 64])).expect("decodes");
    assert!(
        out.play(9, PlayOn::Emitter(EmitterId(2)), -2.0, true)
            .is_some()
    );
    out.move_emitter(EmitterId(2), [2.0, 0.0, 0.0]);
    out.muffle_emitter(EmitterId(2), -6.0, 4_000.0);
    out.pump();
    out.remove_emitter(EmitterId(2));
    assert!(
        out.play(9, PlayOn::Emitter(EmitterId(2)), 0.0, true)
            .is_none()
    );
}

#[test]
fn positioned_sounds_ride_transient_sfx_tracks() {
    let mut out = mock_out();
    out.ready(1, pcm16_mono(44_100, &[0; 4096]))
        .expect("decodes");
    assert!(
        out.play(1, PlayOn::At([2.0, 0.0, 1.0]), 0.0, false)
            .is_some()
    );
    assert_eq!(out.buses.sfx.num_sub_tracks(), 1);
    assert_eq!(out.buses.music.num_sub_tracks(), 0);
}

// Emitters come and go with their clips; no handle outlives its emitter.
#[test]
fn emitter_clips_keep_no_voice_handles() {
    let mut out = mock_out();
    out.ready(9, pcm16_mono(44_100, &[0; 64])).expect("decodes");
    for _ in 0..8 {
        assert!(out.add_emitter(EmitterId(0), [1.0, 0.0, 0.0], &ACOUSTICS, OPEN));
        assert!(
            out.play(9, PlayOn::Emitter(EmitterId(0)), 0.0, true)
                .is_some()
        );
        out.remove_emitter(EmitterId(0));
    }
    assert!(out.voices.is_empty());
}
