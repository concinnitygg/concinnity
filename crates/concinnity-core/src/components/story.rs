// Branching-story graph schema.

use crate::components::Vocabulary;
use crate::components::{Screen, Sprite, TextLabel};
use crate::ecs::AudioClipHandle;
use crate::ecs::Ref;
use crate::ecs::TextureHandle;
use alloc::string::String;
use alloc::vec::Vec;

/// A compiled branching story graph, played at runtime by the story system.
///
/// A `Story` is normally produced by a [StoryImport](#storyimport) expansion
/// at build time rather than written by hand: the Markdown source compiles
/// into this graph plus the stage scaffolding (a single dialogue
/// [Screen](#screen) whose labels and sprites the story system mutates page by
/// page). All references are pre-resolved: dialog text is pre-wrapped,
/// speakers carry their display name and color, stage images carry their
/// on-canvas rectangle, and jump / choice targets are node indices into
/// `nodes`.
///
/// The story system reads the graph and drives the stage screen named
/// `<name>_stage`: it fills the dialogue and name-plate labels (revealing
/// text at `text_speed`), swaps the backdrop and portrait sprite textures,
/// shows the choice menu when a node ends in one, and plays page audio.
/// Clicking the stage (or pressing Space) advances; `{"story": "start"}` restarts
/// from the first node.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Story {
    /// The story title, as shown on the generated title screen.
    pub title: String,
    /// The node graph in document order. Play starts at the first node; a
    /// node whose last page has no jump and no choices falls through to the
    /// next node, and the last node ends the story.
    pub nodes: Vec<StoryNode>,
    /// Dialogue reveal speed in characters per second. `0` shows each page
    /// instantly.
    #[asset(default = 45.0)]
    pub text_speed: f32,
    /// The generated stage assets the story system drives. All references
    /// are resolved to ids at build time, like every other cross-reference.
    pub scaffold: StoryScaffold,
    /// Stable key naming this story's save file (position + flags,
    /// auto-saved page by page under the project data directory). Empty
    /// disables saving.
    pub save_key: String,
}

/// The stage scaffolding a [Story](#story)'s build expansion generated: the
/// [Screen](#screen)s, [Sprite](#sprite)s, and [TextLabel](#textlabel)s the
/// story system mutates page by page.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryScaffold {
    /// The stage [Screen](#screen) the story plays inside.
    pub screen: Option<Ref<Screen>>,
    /// The [Screen](#screen) shown when the story ends.
    pub ending: Option<Ref<Screen>>,
    /// Backdrop [Sprite](#sprite).
    pub bg: Option<Ref<Sprite>>,
    /// Stage-left portrait [Sprite](#sprite).
    pub left: Option<Ref<Sprite>>,
    /// Stage-center portrait [Sprite](#sprite).
    pub center: Option<Ref<Sprite>>,
    /// Stage-right portrait [Sprite](#sprite).
    pub right: Option<Ref<Sprite>>,
    /// Dialog box backdrop [Sprite](#sprite).
    pub dialog_box: Option<Ref<Sprite>>,
    /// Speaker name-plate [TextLabel](#textlabel).
    pub name_label: Option<Ref<TextLabel>>,
    /// Dialog text [TextLabel](#textlabel).
    pub text_label: Option<Ref<TextLabel>>,
    /// Choice button box [Sprite](#sprite)s, one per option slot.
    pub option_boxes: Vec<Ref<Sprite>>,
    /// Choice button [TextLabel](#textlabel)s, one per option slot.
    pub options: Vec<Ref<TextLabel>>,
    /// The title screen's Start [TextLabel](#textlabel). The story lays the
    /// title menu out at runtime, keeping only the buttons that apply
    /// contiguous (Continue and Load appear only when a save exists), so these
    /// labels are moved and cleared per the save state on disk.
    pub start_label: Option<Ref<TextLabel>>,
    /// The title screen's Quit [TextLabel](#textlabel).
    pub quit_label: Option<Ref<TextLabel>>,
    /// The title screen's Continue [TextLabel](#textlabel), hidden while no
    /// save exists.
    pub continue_label: Option<Ref<TextLabel>>,
    /// The title screen [Screen](#screen), returned to when the load overlay is
    /// dismissed before play started.
    pub title: Option<Ref<Screen>>,
    /// The title screen's Load [TextLabel](#textlabel), hidden while no
    /// slot save exists.
    pub load_label: Option<Ref<TextLabel>>,
    /// The pause-menu [Screen](#screen) (the injected Escape overlay), shown over
    /// the stage and returned from to the stage. Unset when the world declares
    /// no pause menu.
    pub pause: Option<Ref<Screen>>,
    /// The settings-screen entry [Screen](#screen) opened by the pause menu's and
    /// the title screen's Settings items. Unset when there is no pause menu.
    pub settings: Option<Ref<Screen>>,
    /// The title screen's Settings [TextLabel](#textlabel), laid out with the
    /// other title buttons and hidden when there is no settings screen.
    pub settings_label: Option<Ref<TextLabel>>,
    /// The small pulsing [Sprite](#sprite) shown when a fully revealed page
    /// waits for input.
    pub advance_marker: Option<Ref<Sprite>>,
    /// Quick-row Log [TextLabel](#textlabel) (dialogue history toggle).
    pub log_label: Option<Ref<TextLabel>>,
    /// Quick-row Auto [TextLabel](#textlabel) (auto-advance toggle).
    pub auto_label: Option<Ref<TextLabel>>,
    /// Quick-row Skip [TextLabel](#textlabel) (fast-forward toggle).
    pub skip_label: Option<Ref<TextLabel>>,
    /// Quick-row Save [TextLabel](#textlabel) (opens the slot overlay).
    pub save_label: Option<Ref<TextLabel>>,
    /// Full-canvas dim [Sprite](#sprite) behind the backlog and slot
    /// overlays.
    pub overlay_dim: Option<Ref<Sprite>>,
    /// The backlog overlay's history [TextLabel](#textlabel).
    pub backlog_label: Option<Ref<TextLabel>>,
    /// The slot overlay's heading [TextLabel](#textlabel) ("Save" / "Load").
    pub slot_title: Option<Ref<TextLabel>>,
    /// Slot row box [Sprite](#sprite)s.
    pub slot_boxes: Vec<Ref<Sprite>>,
    /// Slot row [TextLabel](#textlabel)s.
    pub slot_labels: Vec<Ref<TextLabel>>,
}

/// One jump target in a [Story](#story): a run of pages optionally ending in
/// a choice menu.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryNode {
    /// The heading slug this node was compiled from (diagnostics only).
    pub slug: String,
    /// The click-through pages, in order.
    pub pages: Vec<StoryPage>,
    /// The choice menu shown after the last page. Empty = no menu.
    pub choices: Vec<StoryChoice>,
    /// Stage dressing current at the choice menu.
    pub choice_stage: StoryStage,
    /// Music current at the choice menu ([AudioClip](#audioclip) reference).
    pub choice_music: Option<AudioClipHandle>,
    /// One-shots played when the choice menu shows.
    pub choice_sounds: Vec<AudioClipHandle>,
    /// Flag operations run when the choice menu shows.
    pub choice_ops: Vec<StoryOp>,
    /// Conditional jumps evaluated before the choice menu shows.
    pub choice_gates: Vec<StoryGate>,
}

/// One click-through page of a [StoryNode](#storynode).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryPage {
    /// The speaking character, shown as a name plate. `None` = narration.
    pub speaker: Option<StorySpeaker>,
    /// The dialog text, pre-wrapped with explicit newlines.
    pub text: String,
    /// BehaviorNode index advancing jumps to, overriding the default next-page /
    /// fall-through order.
    pub jump: Option<u32>,
    /// Music current at this page ([AudioClip](#audioclip) reference).
    /// Re-triggering the already-playing track is seamless.
    pub music: Option<AudioClipHandle>,
    /// One-shot effects played when the page shows.
    pub sounds: Vec<AudioClipHandle>,
    /// Stage dressing current at this page.
    pub stage: StoryStage,
    /// Flag operations run when the page shows.
    pub ops: Vec<StoryOp>,
    /// Conditional jumps evaluated before the page shows: the first gate
    /// whose condition passes redirects play to its target node instead.
    pub gates: Vec<StoryGate>,
}

/// A resolved speaker attribution on a [StoryPage](#storypage).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StorySpeaker {
    /// Display name for the name plate.
    pub name: String,
    /// Name-plate text color.
    pub color: [f32; 3],
}

/// The stage dressing current at a page or choice menu: the backdrop and the
/// character portraits standing on stage.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryStage {
    /// Backdrop image. `None` = flat dark fill.
    pub bg: Option<StoryImage>,
    /// Portrait at stage left.
    pub left: Option<StoryImage>,
    /// Portrait at stage center.
    pub center: Option<StoryImage>,
    /// Portrait at stage right.
    pub right: Option<StoryImage>,
}

/// One placed stage image: which [Texture](#texture) to sample and where it
/// sits on the reference canvas.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryImage {
    /// [Texture](#texture) to sample.
    pub texture: TextureHandle,
    /// Left edge on the reference canvas.
    pub x: f32,
    /// Top edge on the reference canvas.
    pub y: f32,
    /// Width on the reference canvas.
    pub width: f32,
    /// Height on the reference canvas.
    pub height: f32,
}

/// One option in a [StoryNode](#storynode)'s choice menu.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryChoice {
    /// Button text.
    pub label: String,
    /// BehaviorNode index chosen; play continues at that node's first page.
    pub target: u32,
    /// Condition gating the option: shown only while it passes. `None` is
    /// always shown.
    pub condition: Option<StoryCondition>,
}

/// One variable operation in a [Story](#story)'s script. All story state is
/// named integer variables, starting at `0` each playthrough: a plain flag
/// is a variable set to `1` and cleared to `0`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryOp {
    /// The variable name.
    pub name: String,
    /// The value assigned (or added).
    pub value: i32,
    /// `false` assigns `value`; `true` adds it to the current value.
    pub add: bool,
}

/// One conditional jump in a [Story](#story)'s script.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryGate {
    /// The variable the condition tests.
    pub name: String,
    /// How the variable compares against `value`.
    pub op: StoryCompareOp,
    /// The literal compared against.
    pub value: i32,
    /// BehaviorNode index play jumps to when the condition passes.
    pub target: u32,
}

/// A condition on a [StoryChoice](#storychoice).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StoryCondition {
    /// The variable the condition tests.
    pub name: String,
    /// How the variable compares against `value`.
    pub op: StoryCompareOp,
    /// The literal compared against.
    pub value: i32,
}

/// A comparison operator in a [Story](#story) condition. An unset variable
/// reads as `0`, so a plain flag test is `Ne 0` and its negation `Eq 0`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum StoryCompareOp {
    /// Equal.
    #[vocab("eq")]
    Eq,
    /// Not equal.
    #[default]
    #[vocab("ne")]
    Ne,
    /// Less than.
    #[vocab("lt")]
    Lt,
    /// Less than or equal.
    #[vocab("le")]
    Le,
    /// Greater than.
    #[vocab("gt")]
    Gt,
    /// Greater than or equal.
    #[vocab("ge")]
    Ge,
}

impl StoryCompareOp {
    /// Evaluate `lhs <op> rhs`.
    pub fn eval(self, lhs: i32, rhs: i32) -> bool {
        match self {
            StoryCompareOp::Eq => lhs == rhs,
            StoryCompareOp::Ne => lhs != rhs,
            StoryCompareOp::Lt => lhs < rhs,
            StoryCompareOp::Le => lhs <= rhs,
            StoryCompareOp::Gt => lhs > rhs,
            StoryCompareOp::Ge => lhs >= rhs,
        }
    }
}

/// Runtime event carrying a freshly re-compiled [Story](#story) graph. The
/// story system swaps its graph for the new one in place, keeping the
/// current position (matched by node slug) and raised flags, so edits to a
/// story's source land in the running game. A plain event, not a declarable
/// asset.
#[derive(Debug, Clone)]
pub struct StoryReload {
    /// The replacement graph. Matched to its story system by the scaffold's
    /// stage screen reference.
    pub story: Story,
}

/// The [Story](#story) playback command a [Behavior](#behavior) node sends.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum StoryPlayback {
    /// Start the story from its beginning.
    #[default]
    #[vocab("start")]
    Start,
    /// Resume the story from its auto-save.
    #[vocab("continue")]
    Continue,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn an_omitted_comparison_defaults_to_not_equal() {
        // A gate written with only a name and a value reads as "flag is set",
        // the common case in an imported markdown story.
        let g: StoryGate = crate::test_support::from_json(r#"{"name":"met_ana","target":3}"#);
        assert_eq!(g.op, StoryCompareOp::Ne);
        assert!(g.op.eval(1, 0));
    }

    #[test]
    fn an_empty_name_in_a_sound_list_is_an_error() {
        crate::test_support::install_resolvers();
        let page = serde_json::from_str::<StoryPage>(r#"{"sounds":["door",""]}"#);
        let err = page.unwrap_err().to_string();
        assert!(err.contains("empty reference name"), "{err}");
        let node = serde_json::from_str::<StoryNode>(r#"{"choice_sounds":[""]}"#);
        let err = node.unwrap_err().to_string();
        assert!(err.contains("empty reference name"), "{err}");
    }

    // NAMES is what the editor's picker offers, so it has to be what serde
    // accepts. A variant added without extending both lists fails here.
    #[test]
    fn every_playback_name_is_what_serde_writes() {
        assert_eq!(StoryPlayback::ALL.len(), StoryPlayback::NAMES.len());
        for (cmd, name) in StoryPlayback::ALL.iter().zip(StoryPlayback::NAMES) {
            assert_eq!(cmd.as_str(), *name);
            assert_eq!(
                serde_json::to_string(cmd).expect("serializes"),
                alloc::format!("\"{name}\"")
            );
        }
    }

    #[test]
    fn every_comparison_agrees_with_the_operator_it_names() {
        for (lhs, rhs) in [(1, 2), (2, 2), (3, 2)] {
            assert_eq!(StoryCompareOp::Eq.eval(lhs, rhs), lhs == rhs);
            assert_eq!(StoryCompareOp::Ne.eval(lhs, rhs), lhs != rhs);
            assert_eq!(StoryCompareOp::Lt.eval(lhs, rhs), lhs < rhs);
            assert_eq!(StoryCompareOp::Le.eval(lhs, rhs), lhs <= rhs);
            assert_eq!(StoryCompareOp::Gt.eval(lhs, rhs), lhs > rhs);
            assert_eq!(StoryCompareOp::Ge.eval(lhs, rhs), lhs >= rhs);
        }
    }

    #[test]
    fn comparison_and_playback_names_parse_in_lowercase() {
        let op = |s: &str| serde_json::from_str::<StoryCompareOp>(s).unwrap();
        assert_eq!(op(r#""eq""#), StoryCompareOp::Eq);
        assert_eq!(op(r#""ne""#), StoryCompareOp::Ne);
        assert_eq!(op(r#""lt""#), StoryCompareOp::Lt);
        assert_eq!(op(r#""le""#), StoryCompareOp::Le);
        assert_eq!(op(r#""gt""#), StoryCompareOp::Gt);
        assert_eq!(op(r#""ge""#), StoryCompareOp::Ge);
        assert_eq!(
            serde_json::to_string(&StoryCompareOp::Ge).unwrap(),
            r#""ge""#
        );

        assert_eq!(
            serde_json::from_str::<StoryPlayback>(r#""continue""#).unwrap(),
            StoryPlayback::Continue
        );
        assert_eq!(
            serde_json::to_string(&StoryPlayback::Start).unwrap(),
            r#""start""#
        );
    }

    #[test]
    fn a_reload_carries_the_replacement_graph() {
        let reload = StoryReload {
            story: Story::default(),
        };
        assert!(reload.story.nodes.is_empty());
        assert!(alloc::format!("{reload:?}").contains("StoryReload"));
    }
}
