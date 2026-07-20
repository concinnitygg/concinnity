// Reaction schema: declarative when/if/then logic.

use alloc::string::String;
use alloc::vec::Vec;

use crate::story::CmpOp;
use crate::{AssetId, AudioClipHandle, de_opt_asset_ref, de_opt_audio_clip_handle};

/// A declarative logic rule: when an event fires and its conditions pass, run
/// a list of actions.
///
/// A reaction is the world's when/if/then unit. `on` names the event that
/// fires it, `conditions` gate it against shared integer variables (all must
/// pass), and `actions` run in order when it fires. Variables start each run
/// at `0` and are written by the `set` action, so a flag is a variable holding
/// `1`. Rules chain: a reaction with a `variable` source fires when another
/// reaction (or any system) changes that variable.
///
/// `once` limits a reaction to a single firing; `cooldown` enforces a minimum
/// number of seconds between firings; `delay` postpones the actions after the
/// firing decision (conditions are checked at fire time, not after the delay).
/// Timers, delays, and cooldowns freeze while a menu is open, like the rest of
/// the world clock.
///
/// ```jsonl
/// {"name":"greet","type":"Reaction","args":{"on":"start","actions":[{"set":{"name":"visits","value":1,"add":true}}]}}
/// {"name":"drip","type":"Reaction","args":{"on":{"timer":{"interval":5.0,"repeat":true}},"actions":[{"spawn":{"template":"drop","position":[0,3,0],"lifetime":4.0}}]}}
/// {"name":"chime","type":"Reaction","args":{"on":{"variable":"visits"},"conditions":[{"name":"visits","op":"ge","value":3}],"actions":[{"sound":{"clip":"bell"}}],"once":true}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Reaction {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The event that fires this reaction: `"start"` (world start), `{"timer":
    /// {"interval": seconds, "repeat": bool}}`, or `{"variable": "name"}` (the
    /// named variable changed value).
    pub on: ReactionSource,
    /// Conditions on shared variables; every one must pass for the reaction to
    /// fire. An empty list always passes.
    pub conditions: Vec<Condition>,
    /// The actions run, in order, each time the reaction fires.
    pub actions: Vec<ReactionAction>,
    /// Fire at most once per run.
    pub once: bool,
    /// Seconds between the firing decision and the actions running (`0` runs
    /// them immediately).
    pub delay: f32,
    /// Minimum seconds between firings (`0` allows every firing).
    pub cooldown: f32,
}

/// The event that fires a [Reaction](#reaction).
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReactionSource {
    /// Fires once when the world starts.
    #[default]
    Start,
    /// Fires when `interval` seconds have elapsed; with `repeat`, every
    /// `interval` seconds.
    Timer {
        /// Seconds before the reaction fires.
        #[serde(default)]
        interval: f32,
        /// `true` fires every `interval` seconds; `false` fires once.
        #[serde(default)]
        repeat: bool,
    },
    /// Fires whenever the named shared variable changes value.
    Variable(String),
}

/// A test against one shared variable, gating a [Reaction](#reaction). An
/// unset variable reads as `0`, so a plain flag test is `ne 0` and its
/// negation `eq 0`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Condition {
    /// The variable the condition tests.
    pub name: String,
    /// How the variable compares against `value`: `eq`, `ne`, `lt`, `le`,
    /// `gt`, or `ge`.
    pub op: CmpOp,
    /// The literal compared against.
    pub value: i32,
}

/// One action run by a firing [Reaction](#reaction).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReactionAction {
    /// Write a shared variable: assign `value`, or add it to the current
    /// value when `add` is `true`.
    Set {
        /// The variable name.
        name: String,
        /// The value assigned (or added).
        #[serde(default)]
        value: i32,
        /// `false` assigns `value`; `true` adds it to the current value.
        #[serde(default)]
        add: bool,
    },
    /// Create a copy of an existing placement at a world position. The copy is
    /// transient: it cannot be addressed by name afterwards.
    Spawn {
        /// The placement to copy (e.g. a [Prop](#prop)).
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        template: Option<AssetId>,
        /// World-space position of the copy.
        #[serde(default)]
        position: [f32; 3],
        /// Euler rotation of the copy in degrees.
        #[serde(default)]
        rotation_deg: [f32; 3],
        /// Scale of the copy (`[1, 1, 1]` keeps the template's size).
        #[serde(default = "unit_scale")]
        scale: [f32; 3],
        /// Seconds the copy lives before auto-despawning (`0` lives forever).
        #[serde(default)]
        lifetime: f32,
    },
    /// Remove a named entity and its children from the world.
    Despawn {
        /// The entity to remove.
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        target: Option<AssetId>,
    },
    /// Re-point a named entity's parent edge.
    Reparent {
        /// The entity to move.
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        child: Option<AssetId>,
        /// The new parent, or unset to detach to a root.
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        parent: Option<AssetId>,
    },
    /// Play an [AudioClip](#audioclip) flat on the main mix (no 3D position).
    Sound {
        /// The clip to play.
        #[serde(default, deserialize_with = "de_opt_audio_clip_handle")]
        clip: Option<AudioClipHandle>,
        /// Playback behavior: a looping `music` track or a one-shot `sound`.
        #[serde(default)]
        kind: crate::CueKind,
        /// Linear gain applied to the clip (`1.0` leaves it unchanged).
        #[serde(default = "unit_volume")]
        volume: f32,
    },
    /// Jump the world's [SceneReel](#scenereel) to a named [Scene](#scene).
    Scene {
        /// The scene to jump to.
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        scene: Option<AssetId>,
        /// The transition: `"Cut"` or `"FadeBlack"`.
        #[serde(default = "default_transition")]
        transition: String,
    },
    /// Show a [Screen](#screen), replacing the top of the screen stack.
    Screen {
        /// The screen to show.
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        screen: Option<AssetId>,
    },
    /// Control the world's [Story](#story) playback.
    Story(StoryPlayback),
}

/// The [Story](#story) playback command a [Reaction](#reaction) action sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoryPlayback {
    /// Start the story from its beginning.
    #[default]
    Start,
    /// Resume the story from its auto-save.
    Continue,
}

impl Reaction {
    /// Whether any action plays an audio clip, so the runtime knows this
    /// reaction needs the audio system and its clip payloads cached.
    pub fn plays_sound(&self) -> bool {
        self.actions
            .iter()
            .any(|a| matches!(a, ReactionAction::Sound { .. }))
    }
}

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn unit_volume() -> f32 {
    1.0
}

fn default_transition() -> String {
    String::from("FadeBlack")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_shapes_parse() {
        let r: Reaction = serde_json::from_str(r#"{"on":"start"}"#).unwrap();
        assert_eq!(r.on, ReactionSource::Start);

        let r: Reaction =
            serde_json::from_str(r#"{"on":{"timer":{"interval":2.5,"repeat":true}}}"#).unwrap();
        assert_eq!(
            r.on,
            ReactionSource::Timer {
                interval: 2.5,
                repeat: true
            }
        );

        let r: Reaction = serde_json::from_str(r#"{"on":{"variable":"score"}}"#).unwrap();
        assert_eq!(r.on, ReactionSource::Variable("score".into()));
    }

    #[test]
    fn defaults_fill_omitted_fields() {
        let r: Reaction = serde_json::from_str("{}").unwrap();
        assert_eq!(r.on, ReactionSource::Start);
        assert!(r.conditions.is_empty());
        assert!(r.actions.is_empty());
        assert!(!r.once);
        assert_eq!(r.delay, 0.0);
        assert_eq!(r.cooldown, 0.0);
    }

    #[test]
    fn action_defaults_parse() {
        let r: Reaction = serde_json::from_str(
            r#"{"actions":[{"set":{"name":"n"}},{"spawn":{"template":3}},{"sound":{"clip":2}},{"story":"continue"}]}"#,
        )
        .unwrap();
        match &r.actions[1] {
            ReactionAction::Spawn {
                scale, lifetime, ..
            } => {
                assert_eq!(*scale, [1.0, 1.0, 1.0]);
                assert_eq!(*lifetime, 0.0);
            }
            other => panic!("expected spawn, got {other:?}"),
        }
        match &r.actions[2] {
            ReactionAction::Sound { volume, kind, .. } => {
                assert_eq!(*volume, 1.0);
                assert_eq!(*kind, crate::CueKind::Sound);
            }
            other => panic!("expected sound, got {other:?}"),
        }
        assert!(matches!(
            r.actions[3],
            ReactionAction::Story(StoryPlayback::Continue)
        ));
    }

    #[test]
    fn condition_defaults_to_flag_test() {
        let c: Condition = serde_json::from_str(r#"{"name":"has_key"}"#).unwrap();
        assert_eq!(c.op, CmpOp::Ne);
        assert_eq!(c.value, 0);
    }

    #[test]
    fn baked_round_trip_is_postcard_stable() {
        let r: Reaction = serde_json::from_str(
            r#"{"on":{"timer":{"interval":1.0}},"conditions":[{"name":"n","op":"ge","value":2}],
                "actions":[{"despawn":{"target":7}},{"scene":{"scene":9,"transition":"Cut"}}],
                "once":true,"delay":0.5,"cooldown":2.0}"#,
        )
        .unwrap();
        let bytes = postcard::to_allocvec(&r).unwrap();
        let back: Reaction = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.actions.len(), 2);
        assert!(back.once);
        assert_eq!(back.delay, 0.5);
        match &back.actions[0] {
            ReactionAction::Despawn { target } => assert_eq!(*target, Some(AssetId(7))),
            other => panic!("expected despawn, got {other:?}"),
        }
    }
}
