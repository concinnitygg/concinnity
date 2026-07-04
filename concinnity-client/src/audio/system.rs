// src/audio/system.rs
//
// Audio playback: 3D positional emitters and view-triggered cues. An internal
// system (not a declarable asset): `World::start` constructs one whenever the
// world contains any `AudioEmitter` or `AudioCue`, so a world with neither
// never opens an audio device.

use std::collections::HashMap;

use crate::assets::{
    AudioClip, AudioCommand, AudioCue, AudioEmitter, Camera3D, CueKind, PlayCue, Story, Transform,
    ViewShown,
};
use crate::audio::{AudioEngine, EmitterId};
use crate::ecs::asset_id::AssetId;
use crate::ecs::decompose::EntityByName;
use crate::ecs::{PipelineContext, StepResult, System};

// Audio behavior. Constructed internally by `World::start` when the world
// declares any `AudioEmitter` or `AudioCue`; never a world-declared asset, so
// it carries no config.
pub struct AudioSystem {
    engine: AudioEngine,
    // One entry per `AudioEmitter` that became a live engine emitter.
    emitters: Vec<EmitterBinding>,
    // View-triggered cues, keyed by the View whose activation fires them.
    cues: HashMap<AssetId, Vec<CueBinding>>,
    // Encoded clip payloads for the cue clips, keyed by AudioClip id.
    cue_clip_bytes: HashMap<AssetId, Vec<u8>>,
    // Cursor into the Events<AudioCommand> queue (live master-volume changes).
    audio_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<ViewShown> queue (cue triggers).
    view_shown_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<PlayCue> queue (direct play requests, e.g. the
    // story system's page audio).
    play_cue_cursor: crate::ecs::EventCursor,
    // Cues that matched a shown view so far; observable engine-independent
    // progress for headless tests (playback needs a device and a payload).
    cues_matched: usize,
}

// Links one engine emitter to the world data that positions it.
struct EmitterBinding {
    id: EmitterId,
    // The Prop this emitter follows each frame, if any.
    follows: Option<AssetId>,
}

// One AudioCue resolved to its clip and playback style.
struct CueBinding {
    clip: AssetId,
    kind: CueKind,
    volume: f32,
}

impl std::fmt::Debug for AudioSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioSystem")
            .field("engine", &self.engine)
            .field("emitters", &self.emitters.len())
            .field("cue_views", &self.cues.len())
            .finish()
    }
}

impl AudioSystem {
    // Fresh audio engine with no live emitters or cues. Emitters and cues are
    // bound from the world's components in [`System::init`].
    pub fn new() -> Self {
        Self {
            engine: AudioEngine::new(),
            emitters: Vec::new(),
            cues: HashMap::new(),
            cue_clip_bytes: HashMap::new(),
            audio_cmd_cursor: crate::ecs::EventCursor::default(),
            view_shown_cursor: crate::ecs::EventCursor::default(),
            play_cue_cursor: crate::ecs::EventCursor::default(),
            cues_matched: 0,
        }
    }
}

impl System for AudioSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        // Snapshot the emitters, then resolve every clip's payload locator so
        // the borrow of `ctx` for the AudioClip query is released before the
        // `read_payload` calls below.
        let emitter_snaps: Vec<AudioEmitter> = ctx.query::<AudioEmitter>().cloned().collect();
        let clip_locators: HashMap<AssetId, crate::ecs::PayloadLocator> = ctx
            .query::<AudioClip>()
            .filter_map(|c| c.locator.clone().map(|l| (c.asset_id, l)))
            .collect();

        // The persisted master volume (settings menu) scales every emitter via
        // the main mix track, so it can be changed live (see `step`). `None`
        // leaves output at unity. Clips play at their authored gain; the master
        // is a separate output-stage multiplier.
        let master = crate::config::Settings::load()
            .audio
            .master_volume
            .unwrap_or(1.0);
        self.engine.set_master_volume(master);

        for emitter in emitter_snaps {
            let Some(id) = self.engine.add_emitter(emitter.position) else {
                continue;
            };
            match emitter.clip.and_then(|clip| clip_locators.get(&clip)) {
                Some(locator) => match ctx.read_payload(locator) {
                    Ok(bytes) => {
                        self.engine
                            .play_clip(id, bytes, emitter.looping, emitter.volume);
                    }
                    Err(e) => tracing::warn!("AudioSystem: clip payload read failed: {e}"),
                },
                None => tracing::warn!(
                    "AudioSystem: emitter has no clip with a compiled payload, silent"
                ),
            }
            self.emitters.push(EmitterBinding {
                id,
                follows: emitter.prop,
            });
        }

        // Bind the view-triggered cues and cache their clip payloads, so
        // firing a cue never touches the blob mid-frame.
        let cue_snaps: Vec<AudioCue> = ctx.query::<AudioCue>().cloned().collect();
        for cue in cue_snaps {
            let (Some(view), Some(clip)) = (cue.view, cue.clip) else {
                tracing::warn!("AudioSystem: cue without a view and a clip, ignored");
                continue;
            };
            if let std::collections::hash_map::Entry::Vacant(slot) = self.cue_clip_bytes.entry(clip)
            {
                match clip_locators.get(&clip) {
                    Some(locator) => match ctx.read_payload(locator) {
                        Ok(bytes) => {
                            slot.insert(bytes.to_vec());
                        }
                        Err(e) => tracing::warn!("AudioSystem: cue payload read failed: {e}"),
                    },
                    None => {
                        tracing::warn!("AudioSystem: cue clip has no compiled payload, silent")
                    }
                }
            }
            self.cues.entry(view).or_default().push(CueBinding {
                clip,
                kind: cue.kind,
                volume: cue.volume,
            });
        }

        // A story plays clips by direct PlayCue request rather than through
        // view-keyed cues, so cache every clip payload up front.
        if ctx.query::<Story>().next().is_some() {
            let uncached: Vec<(AssetId, crate::ecs::PayloadLocator)> = clip_locators
                .iter()
                .filter(|(id, _)| !self.cue_clip_bytes.contains_key(id))
                .map(|(id, locator)| (*id, locator.clone()))
                .collect();
            for (id, locator) in uncached {
                match ctx.read_payload(&locator) {
                    Ok(bytes) => {
                        self.cue_clip_bytes.insert(id, bytes.to_vec());
                    }
                    Err(e) => tracing::warn!("AudioSystem: story clip payload read failed: {e}"),
                }
            }
        }

        tracing::info!(
            "AudioSystem: {} emitter(s), {} cue view(s), engine {}",
            self.emitters.len(),
            self.cues.len(),
            if self.engine.is_enabled() {
                "enabled"
            } else {
                "disabled"
            },
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Apply any live master-volume change sent this tick by GraphicsSystem,
        // which runs first. The last one this tick wins.
        if let Some(events) = ctx.events::<AudioCommand>() {
            for cmd in events.read(&mut self.audio_cmd_cursor) {
                self.engine.set_master_volume(cmd.master_volume);
            }
        }

        // Fire cues for views shown since the last step. UiInputSystem runs
        // after this system, so a navigation is heard one tick later.
        if !self.cues.is_empty()
            && let Some(events) = ctx.events::<ViewShown>()
        {
            let shown: Vec<AssetId> = events
                .read(&mut self.view_shown_cursor)
                .into_iter()
                .map(|e| e.view)
                .collect();
            for view in shown {
                let Some(bindings) = self.cues.get(&view) else {
                    continue;
                };
                self.cues_matched += bindings.len();
                for cue in bindings {
                    let Some(bytes) = self.cue_clip_bytes.get(&cue.clip) else {
                        continue;
                    };
                    match cue.kind {
                        CueKind::Music => {
                            self.engine.play_music(cue.clip.0 as u64, bytes, cue.volume);
                        }
                        CueKind::Sound => {
                            self.engine.play_sound(bytes, cue.volume);
                        }
                    }
                }
            }
        }

        // Play direct requests (the story system's page audio). The story
        // system runs earlier in the schedule, so these are heard this tick.
        if let Some(events) = ctx.events::<PlayCue>() {
            let requests: Vec<PlayCue> = events
                .read(&mut self.play_cue_cursor)
                .into_iter()
                .copied()
                .collect();
            for cue in requests {
                self.cues_matched += 1;
                let Some(bytes) = self.cue_clip_bytes.get(&cue.clip) else {
                    continue;
                };
                match cue.kind {
                    CueKind::Music => {
                        self.engine.play_music(cue.clip.0 as u64, bytes, cue.volume);
                    }
                    CueKind::Sound => {
                        self.engine.play_sound(bytes, cue.volume);
                    }
                }
            }
        }

        // The listener rides the camera.
        if let Some((pos, yaw, pitch)) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.position, c.yaw, c.pitch))
        {
            self.engine.set_listener(pos, yaw, pitch);
        }

        // Prop-bound emitters track their followed prop's current position, read
        // from its Transform via the name index.
        if self.emitters.iter().any(|b| b.follows.is_some()) {
            for binding in &self.emitters {
                if let Some(prop_id) = binding.follows
                    && let Some(entity) =
                        ctx.resource::<EntityByName>().and_then(|n| n.get(prop_id))
                    && let Some(t) = ctx.get::<Transform>(entity)
                {
                    self.engine.set_emitter_position(binding.id, t.position);
                }
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::assets::AudioEmitter;
    use crate::ecs::World;

    // An `AudioEmitter` in the world spawns the internal AudioSystem; without
    // one, no audio device is opened.
    #[test]
    fn audio_emitter_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(AudioEmitter::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["AudioSystem"]);
    }

    #[test]
    fn no_audio_emitter_means_no_system() {
        let mut world = World::new_empty();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // An `AudioCue` alone (no emitter) also spawns the audio system: a UI-only
    // world can play view-triggered audio.
    #[test]
    fn audio_cue_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(crate::assets::AudioCue::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert!(names.contains(&"AudioSystem"), "{names:?}");
    }

    // The full trigger chain: the initial view's activation (announced by
    // UiInputSystem at init) reaches the audio system, which matches the
    // view's cue on the first step. Playback itself needs a device and a
    // compiled payload, so the test observes the match counter.
    #[test]
    fn initial_view_fires_its_cue() {
        use crate::assets::{AudioClip, AudioCue, View};
        use crate::ecs::asset_id::AssetId;

        let mut world = World::new_empty();
        let view = AssetId(90);
        let clip = AssetId(91);
        world.add_component(View {
            asset_id: view,
            initial: true,
            fade_in_secs: 0.0,
        });
        world.add_component(AudioClip {
            asset_id: clip,
            ..Default::default()
        });
        world.add_component(AudioCue {
            view: Some(view),
            clip: Some(clip),
            ..Default::default()
        });
        world.start().unwrap();
        world.step();

        let matched = world
            .systems()
            .iter()
            .find_map(|s| match s {
                crate::ecs::SystemAsset::AudioSystem(a) => Some(a.cues_matched),
                _ => None,
            })
            .expect("world has an AudioSystem");
        assert_eq!(matched, 1, "the initial view's cue should have matched");
    }

    // The live master gain the world's AudioSystem currently holds. Mirrors how
    // the ControlsCommand test observes Camera3D.yaw: reach into the running
    // system to assert the command actually took effect (the gain is engine
    // state, not a queryable component).
    fn applied_master_volume(world: &World) -> f32 {
        world
            .systems()
            .iter()
            .find_map(|s| match s {
                crate::ecs::SystemAsset::AudioSystem(a) => Some(a.engine.last_master_volume),
                _ => None,
            })
            .expect("world has an AudioSystem")
    }

    // A master-volume AudioCommand sent mid-tick is read AND applied by the
    // audio system the same tick, so the new master takes effect without a
    // restart (the settings-menu master-volume row).
    #[test]
    fn audio_command_applies_master_volume_live() {
        use crate::assets::AudioCommand;

        let mut world = World::new_empty();
        world.add_component(AudioEmitter::default());
        world.start().unwrap();
        // Init applied the persisted master (unity by default in a test).
        assert!((applied_master_volume(&world) - 1.0).abs() < 1.0e-6);

        // GraphicsSystem would send this when the master-volume row is cycled;
        // the audio system reads it this same tick.
        world
            .events_mut::<AudioCommand>()
            .send(AudioCommand { master_volume: 0.5 });
        world.step();

        assert!(
            (applied_master_volume(&world) - 0.5).abs() < 1.0e-6,
            "master volume should be applied live this tick"
        );
    }

    // Several AudioCommands sent in one tick (e.g. a rapid double-cycle) are
    // all read in order; the last one sent is applied last and wins.
    #[test]
    fn audio_command_last_write_wins_per_tick() {
        use crate::assets::AudioCommand;

        let mut world = World::new_empty();
        world.add_component(AudioEmitter::default());
        world.start().unwrap();

        world
            .events_mut::<AudioCommand>()
            .send(AudioCommand { master_volume: 0.5 });
        world.events_mut::<AudioCommand>().send(AudioCommand {
            master_volume: 0.25,
        });
        world.step();

        assert!((applied_master_volume(&world) - 0.25).abs() < 1.0e-6);
    }
}
