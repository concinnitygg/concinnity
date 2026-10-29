use super::*;
use crate::audio::out::recording::RecordingOut;
use crate::components::Rolloff;
use alloc::vec;

type TestMixer = Mixer<RecordingOut>;

fn mixer() -> TestMixer {
    Mixer::new(Some(Box::new(RecordingOut::default())))
}

fn out(mixer: &TestMixer) -> &RecordingOut {
    mixer.out.as_deref().expect("an open output")
}

fn out_mut(mixer: &mut TestMixer) -> &mut RecordingOut {
    mixer.out.as_deref_mut().expect("an open output")
}

// Queue `key` and let its decode land.
fn ready(mixer: &mut TestMixer, key: u64) {
    mixer.queue_clip(key, vec![0; 4]);
    mixer.tick();
    assert_eq!(mixer.clips.state(key), Some(&ClipState::Ready));
}

const PARAMS: Acoustics = Acoustics {
    bus: AudioBus::Sfx,
    distances: (1.0, 50.0),
    rolloff: Rolloff::Logarithmic,
};

#[test]
fn a_mixer_without_output_is_inert_but_records_volumes() {
    let mut mixer = TestMixer::new(None);
    assert!(!mixer.is_enabled());
    assert!(mixer.add_emitter([0.0; 3], PARAMS).is_none());
    mixer.queue_clip(1, vec![0; 4]);
    mixer.tick();
    mixer.set_emitter_position(EmitterId(0), [1.0, 2.0, 3.0]);
    mixer.set_emitter_occlusion(EmitterId(0), 0.5);
    mixer.set_listener([4.0, 5.0, 6.0], 1.0, 0.5);
    assert!(!mixer.play_emitter_clip(EmitterId(0), 1, true, 1.0));
    assert!(!mixer.play_music(1, 1.0, AudioBus::Music));
    assert!(!mixer.play_sound(1, 1.0, AudioBus::Sfx, 0));
    assert_eq!(mixer.playing_sounds(), 0);
    assert_eq!(mixer.last_volume(AudioTarget::Master), 1.0);
    mixer.set_volume(AudioTarget::Voice, 0.5);
    assert_eq!(mixer.last_volume(AudioTarget::Voice), 0.5);
}

#[test]
fn one_shots_route_to_their_bus() {
    let mut mixer = mixer();
    ready(&mut mixer, 1);
    assert!(mixer.play_sound(1, 1.0, AudioBus::Voice, 0));
    assert!(mixer.play_sound(1, 1.0, AudioBus::Sfx, 0));
    let ons: Vec<PlayOn> = out(&mixer).playing.iter().map(|p| p.2).collect();
    assert_eq!(
        ons,
        [PlayOn::Bus(AudioBus::Voice), PlayOn::Bus(AudioBus::Sfx)]
    );
    assert_eq!(mixer.playing_sounds(), 2);
}

#[test]
fn a_play_before_its_decode_starts_on_a_later_tick() {
    let mut mixer = mixer();
    mixer.queue_clip(5, vec![0; 4]);
    assert!(mixer.play_sound(5, 1.0, AudioBus::Sfx, 0));
    assert!(out(&mixer).playing.is_empty(), "not started yet");
    mixer.tick();
    assert_eq!(
        out(&mixer).playing.len(),
        1,
        "started when the decode landed"
    );
    assert_eq!(mixer.playing_sounds(), 1);
}

#[test]
fn the_voice_cap_steals_low_and_refuses_outranked() {
    let mut mixer = mixer();
    ready(&mut mixer, 1);
    for _ in 0..MAX_VOICES {
        assert!(mixer.play_sound(1, 1.0, AudioBus::Sfx, 5));
    }
    assert!(mixer.play_sound(1, 1.0, AudioBus::Sfx, 5));
    assert_eq!(mixer.playing_sounds(), MAX_VOICES);
    assert_eq!(out(&mixer).stopped, [VoiceId(0)], "the oldest was stolen");
    assert!(!mixer.play_sound(1, 1.0, AudioBus::Sfx, 1));
    assert_eq!(mixer.playing_sounds(), MAX_VOICES);
}

#[test]
fn finished_voices_are_reaped() {
    let mut mixer = mixer();
    ready(&mut mixer, 1);
    assert!(mixer.play_sound(1, 1.0, AudioBus::Sfx, 0));
    out_mut(&mut mixer).finished.insert(VoiceId(0));
    mixer.tick();
    assert_eq!(mixer.playing_sounds(), 0);
}

#[test]
fn music_replays_are_seamless_and_swaps_replace() {
    let mut mixer = mixer();
    ready(&mut mixer, 1);
    ready(&mut mixer, 2);
    assert!(mixer.play_music(1, 1.0, AudioBus::Music));
    assert!(mixer.play_music(1, 0.5, AudioBus::Music));
    assert_eq!(out(&mixer).playing.len(), 1, "the running track is kept");
    assert!(mixer.play_music(2, 1.0, AudioBus::Music));
    assert_eq!(mixer.music.map(|m| m.0), Some(2));
    assert_eq!(out(&mixer).stopped, [VoiceId(0)], "the old track stopped");
    assert!(out(&mixer).playing.iter().all(|p| p.3), "music loops");
}

#[test]
fn emitters_join_their_bus_and_dedupe_occlusion() {
    let mut mixer = mixer();
    let id = mixer.add_emitter([1.0, 0.0, 0.0], PARAMS).expect("emitter");
    assert_eq!(
        out(&mixer).emitters[&id.0],
        (AudioBus::Sfx, [1.0, 0.0, 0.0])
    );
    ready(&mut mixer, 9);
    assert!(mixer.play_emitter_clip(id, 9, true, 0.8));
    mixer.set_emitter_position(id, [2.0, 0.0, 0.0]);
    assert_eq!(out(&mixer).emitters[&id.0].1, [2.0, 0.0, 0.0]);
    mixer.set_emitter_occlusion(id, 0.5);
    mixer.set_emitter_occlusion(id, 0.5);
    assert_eq!(
        out(&mixer).muffles.len(),
        1,
        "an unchanged factor sends nothing"
    );
}

#[test]
fn removed_emitters_free_their_slot_without_shifting_ids() {
    let mut mixer = mixer();
    let first = mixer.add_emitter([1.0, 0.0, 0.0], PARAMS).unwrap();
    let second = mixer.add_emitter([2.0, 0.0, 0.0], PARAMS).unwrap();
    assert_ne!(first, second);
    mixer.remove_emitter(first);
    mixer.remove_emitter(first);
    ready(&mut mixer, 1);
    assert!(!mixer.play_emitter_clip(first, 1, true, 1.0));
    assert!(mixer.play_emitter_clip(second, 1, true, 1.0));
    let third = mixer.add_emitter([4.0, 0.0, 0.0], PARAMS).unwrap();
    assert_eq!(third, first, "slot recycled");
    assert!(mixer.play_emitter_clip(third, 1, true, 1.0));
}

#[test]
fn an_emitter_the_output_refuses_takes_no_id() {
    let mut mixer = mixer();
    out_mut(&mut mixer).refuse_emitters = true;
    assert!(mixer.add_emitter([0.0; 3], PARAMS).is_none());
    out_mut(&mut mixer).refuse_emitters = false;
    assert_eq!(mixer.add_emitter([0.0; 3], PARAMS), Some(EmitterId(0)));
}

#[test]
fn positioned_one_shots_count_as_voices() {
    let mut mixer = mixer();
    ready(&mut mixer, 1);
    assert!(mixer.play_sound_at([2.0, 0.0, 1.0], 1, 0.8, 0));
    assert_eq!(mixer.playing_sounds(), 1);
    assert_eq!(out(&mixer).playing[0].2, PlayOn::At([2.0, 0.0, 1.0]));
}

#[test]
fn a_failed_decode_refuses_plays_without_wedging_the_mixer() {
    let mut mixer = mixer();
    mixer.queue_clip(7, b"bad".to_vec());
    assert!(mixer.play_sound(7, 1.0, AudioBus::Sfx, 0), "deferred");
    mixer.tick();
    assert_eq!(mixer.clips.state(7), Some(&ClipState::Failed));
    assert!(!mixer.play_sound(7, 1.0, AudioBus::Sfx, 0));
    assert_eq!(mixer.playing_sounds(), 0);
    ready(&mut mixer, 8);
    assert!(mixer.play_sound(8, 1.0, AudioBus::Sfx, 0));
}

#[test]
fn identity_orientation_at_zero_yaw_pitch() {
    let q = orientation(0.0, 0.0);
    assert!((q[3] - 1.0).abs() < 1.0e-6);
    assert!(q[0].abs() < 1.0e-6 && q[1].abs() < 1.0e-6 && q[2].abs() < 1.0e-6);
}

#[test]
fn the_orientation_stays_unit_length() {
    for &(yaw, pitch) in &[(0.5, 0.3), (-1.2, 0.8), (3.0, -0.6), (-2.7, -1.1)] {
        let q = orientation(yaw, pitch);
        let len = math::sqrt(q.iter().map(|c| c * c).sum());
        assert!(
            (len - 1.0).abs() < 1.0e-5,
            "not unit ({len}) at {yaw}/{pitch}"
        );
    }
}

#[test]
fn gain_to_db_reference_points() {
    assert!(gain_to_db(1.0).abs() < 1.0e-4);
    assert!((gain_to_db(0.5) - (-6.0206)).abs() < 0.01);
    assert!(gain_to_db(0.0) < -60.0);
    assert!(gain_to_db(0.0).is_finite());
}
