// Branching-story graph schema.

use crate::{
    AssetId, AudioClipHandle, TextureHandle, de_audio_clip_handle_vec, de_opt_asset_ref,
    de_opt_audio_clip_handle, de_texture_handle,
};
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
/// Clicking the stage (or pressing Space) advances; `story:start` restarts
/// from the first node.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Story {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The story title, as shown on the generated title screen.
    pub title: String,
    /// The node graph in document order. Play starts at the first node; a
    /// node whose last page has no jump and no choices falls through to the
    /// next node, and the last node ends the story.
    pub nodes: Vec<StoryNode>,
    /// Dialogue reveal speed in characters per second. `0` shows each page
    /// instantly.
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryScaffold {
    /// The stage [Screen](#screen) the story plays inside.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
    /// The [Screen](#screen) shown when the story ends.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub ending: Option<AssetId>,
    /// Backdrop [Sprite](#sprite).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub bg: Option<AssetId>,
    /// Stage-left portrait [Sprite](#sprite).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub left: Option<AssetId>,
    /// Stage-center portrait [Sprite](#sprite).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub center: Option<AssetId>,
    /// Stage-right portrait [Sprite](#sprite).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub right: Option<AssetId>,
    /// Dialog box backdrop [Sprite](#sprite).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub dialog_box: Option<AssetId>,
    /// Speaker name-plate [TextLabel](#textlabel).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub name_label: Option<AssetId>,
    /// Dialog text [TextLabel](#textlabel).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub text_label: Option<AssetId>,
    /// Choice button box [Sprite](#sprite)s, one per option slot.
    pub option_boxes: Vec<AssetId>,
    /// Choice button [TextLabel](#textlabel)s, one per option slot.
    pub options: Vec<AssetId>,
    /// The title screen's Start [TextLabel](#textlabel). The story lays the
    /// title menu out at runtime, keeping only the buttons that apply
    /// contiguous (Continue and Load appear only when a save exists), so these
    /// labels are moved and cleared per the save state on disk.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub start_label: Option<AssetId>,
    /// The title screen's Quit [TextLabel](#textlabel).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub quit_label: Option<AssetId>,
    /// The title screen's Continue [TextLabel](#textlabel), hidden while no
    /// save exists.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub continue_label: Option<AssetId>,
    /// The title screen [Screen](#screen), returned to when the load overlay is
    /// dismissed before play started.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub title: Option<AssetId>,
    /// The title screen's Load [TextLabel](#textlabel), hidden while no
    /// slot save exists.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub load_label: Option<AssetId>,
    /// The pause-menu [Screen](#screen) (the injected Escape overlay), shown over
    /// the stage and returned from to the stage. Unset when the world declares
    /// no pause menu.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub pause: Option<AssetId>,
    /// The settings-screen entry [Screen](#screen) opened by the pause menu's and
    /// the title screen's Settings items. Unset when there is no pause menu.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub settings: Option<AssetId>,
    /// The title screen's Settings [TextLabel](#textlabel), laid out with the
    /// other title buttons and hidden when there is no settings screen.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub settings_label: Option<AssetId>,
    /// The small pulsing [Sprite](#sprite) shown when a fully revealed page
    /// waits for input.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub advance_marker: Option<AssetId>,
    /// Quick-row Log [TextLabel](#textlabel) (dialogue history toggle).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub log_label: Option<AssetId>,
    /// Quick-row Auto [TextLabel](#textlabel) (auto-advance toggle).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub auto_label: Option<AssetId>,
    /// Quick-row Skip [TextLabel](#textlabel) (fast-forward toggle).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub skip_label: Option<AssetId>,
    /// Quick-row Save [TextLabel](#textlabel) (opens the slot overlay).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub save_label: Option<AssetId>,
    /// Full-canvas dim [Sprite](#sprite) behind the backlog and slot
    /// overlays.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub overlay_dim: Option<AssetId>,
    /// The backlog overlay's history [TextLabel](#textlabel).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub backlog_label: Option<AssetId>,
    /// The slot overlay's heading [TextLabel](#textlabel) ("Save" / "Load").
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub slot_title: Option<AssetId>,
    /// Slot row box [Sprite](#sprite)s.
    pub slot_boxes: Vec<AssetId>,
    /// Slot row [TextLabel](#textlabel)s.
    pub slot_labels: Vec<AssetId>,
}

/// One jump target in a [Story](#story): a run of pages optionally ending in
/// a choice menu.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    #[serde(deserialize_with = "de_opt_audio_clip_handle")]
    pub choice_music: Option<AudioClipHandle>,
    /// One-shots played when the choice menu shows.
    #[serde(deserialize_with = "de_audio_clip_handle_vec")]
    pub choice_sounds: Vec<AudioClipHandle>,
    /// Flag operations run when the choice menu shows.
    pub choice_ops: Vec<StoryOp>,
    /// Conditional jumps evaluated before the choice menu shows.
    pub choice_gates: Vec<StoryGate>,
}

/// One click-through page of a [StoryNode](#storynode).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryPage {
    /// The speaking character, shown as a name plate. `None` = narration.
    pub speaker: Option<StorySpeaker>,
    /// The dialog text, pre-wrapped with explicit newlines.
    pub text: String,
    /// Node index advancing jumps to, overriding the default next-page /
    /// fall-through order.
    pub jump: Option<u32>,
    /// Music current at this page ([AudioClip](#audioclip) reference).
    /// Re-triggering the already-playing track is seamless.
    #[serde(deserialize_with = "de_opt_audio_clip_handle")]
    pub music: Option<AudioClipHandle>,
    /// One-shot effects played when the page shows.
    #[serde(deserialize_with = "de_audio_clip_handle_vec")]
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StorySpeaker {
    /// Display name for the name plate.
    pub name: String,
    /// Name-plate text color.
    pub color: [f32; 3],
}

/// The stage dressing current at a page or choice menu: the backdrop and the
/// character portraits standing on stage.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryImage {
    /// [Texture](#texture) to sample.
    #[serde(deserialize_with = "de_texture_handle")]
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryChoice {
    /// Button text.
    pub label: String,
    /// Node index chosen; play continues at that node's first page.
    pub target: u32,
    /// Condition gating the option: shown only while it passes. `None` is
    /// always shown.
    pub condition: Option<StoryCondition>,
}

/// One variable operation in a [Story](#story)'s script. All story state is
/// named integer variables, starting at `0` each playthrough: a plain flag
/// is a variable set to `1` and cleared to `0`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryGate {
    /// The variable the condition tests.
    pub name: String,
    /// How the variable compares against `value`.
    pub op: CmpOp,
    /// The literal compared against.
    pub value: i32,
    /// Node index play jumps to when the condition passes.
    pub target: u32,
}

/// A condition on a [StoryChoice](#storychoice).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryCondition {
    /// The variable the condition tests.
    pub name: String,
    /// How the variable compares against `value`.
    pub op: CmpOp,
    /// The literal compared against.
    pub value: i32,
}

/// A comparison operator in a [Story](#story) condition. An unset variable
/// reads as `0`, so a plain flag test is `Ne 0` and its negation `Eq 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CmpOp {
    /// Equal.
    Eq,
    /// Not equal.
    #[default]
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

impl CmpOp {
    /// Evaluate `lhs <op> rhs`.
    pub fn eval(self, lhs: i32, rhs: i32) -> bool {
        match self {
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Gt => lhs > rhs,
            CmpOp::Ge => lhs >= rhs,
        }
    }
}

impl Default for Story {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            title: String::new(),
            nodes: Vec::new(),
            text_speed: 45.0,
            scaffold: StoryScaffold::default(),
            save_key: String::new(),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoryPlayback {
    /// Start the story from its beginning.
    #[default]
    Start,
    /// Resume the story from its auto-save.
    Continue,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_blank_story_has_no_nodes_and_types_at_the_default_speed() {
        let s = Story::default();
        assert!(s.nodes.is_empty());
        assert!(s.title.is_empty());
        assert_eq!(s.text_speed, 45.0);
        // No save key means the story never touches persisted state.
        assert!(s.save_key.is_empty());
        assert!(s.scaffold.screen.is_none());
        assert!(s.scaffold.options.is_empty());
    }

    #[test]
    fn every_comparison_agrees_with_the_operator_it_names() {
        for (lhs, rhs) in [(1, 2), (2, 2), (3, 2)] {
            assert_eq!(CmpOp::Eq.eval(lhs, rhs), lhs == rhs);
            assert_eq!(CmpOp::Ne.eval(lhs, rhs), lhs != rhs);
            assert_eq!(CmpOp::Lt.eval(lhs, rhs), lhs < rhs);
            assert_eq!(CmpOp::Le.eval(lhs, rhs), lhs <= rhs);
            assert_eq!(CmpOp::Gt.eval(lhs, rhs), lhs > rhs);
            assert_eq!(CmpOp::Ge.eval(lhs, rhs), lhs >= rhs);
        }
    }

    #[test]
    fn an_omitted_comparison_defaults_to_not_equal() {
        // A gate written with only a name and a value reads as "flag is set",
        // which is the common case in an imported markdown story.
        assert_eq!(CmpOp::default(), CmpOp::Ne);
        let g: StoryGate = serde_json::from_str(r#"{"name":"met_ana","target":3}"#).unwrap();
        assert_eq!(g.op, CmpOp::Ne);
        assert!(g.op.eval(1, 0));
    }

    #[test]
    fn comparison_and_playback_names_parse_in_lowercase() {
        let op = |s: &str| serde_json::from_str::<CmpOp>(s).unwrap();
        assert_eq!(op(r#""eq""#), CmpOp::Eq);
        assert_eq!(op(r#""ne""#), CmpOp::Ne);
        assert_eq!(op(r#""lt""#), CmpOp::Lt);
        assert_eq!(op(r#""le""#), CmpOp::Le);
        assert_eq!(op(r#""gt""#), CmpOp::Gt);
        assert_eq!(op(r#""ge""#), CmpOp::Ge);
        assert_eq!(serde_json::to_string(&CmpOp::Ge).unwrap(), r#""ge""#);

        assert_eq!(StoryPlayback::default(), StoryPlayback::Start);
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
    fn a_compiled_graph_parses_its_pages_choices_and_audio() {
        crate::test_support::install_resolvers();
        let s: Story = serde_json::from_str(
            r#"{"title":"Ash","text_speed":30.0,"save_key":"ash",
                "nodes":[{"slug":"intro",
                  "pages":[{"speaker":{"name":"Ana","color":[1,0,0]},"text":"Hello",
                            "music":"theme","sounds":["door","",3],
                            "stage":{"bg":{"texture":"bg_room","width":1280,"height":720}},
                            "ops":[{"name":"visits","value":1,"add":true}],
                            "gates":[{"name":"visits","op":"gt","value":2,"target":4}]}],
                  "choices":[{"label":"Stay","target":1,
                              "condition":{"name":"visits","op":"ge","value":1}}],
                  "choice_sounds":["click"]}]}"#,
        )
        .unwrap();

        assert_eq!(s.title, "Ash");
        assert_eq!(s.text_speed, 30.0);
        let node = &s.nodes[0];
        assert_eq!(node.slug, "intro");
        assert_eq!(node.choice_sounds, vec![AudioClipHandle(5)]);

        let page = &node.pages[0];
        assert_eq!(page.text, "Hello");
        assert_eq!(page.speaker.as_ref().expect("speaker").name, "Ana");
        assert_eq!(page.music, Some(AudioClipHandle(5)));
        // Empty entries drop out of a sound list rather than becoming handle 0.
        assert_eq!(page.sounds, vec![AudioClipHandle(4), AudioClipHandle(3)]);
        assert_eq!(page.jump, None);
        let bg = page.stage.bg.as_ref().expect("background image");
        assert_eq!(bg.texture, TextureHandle(7));
        assert_eq!((bg.width, bg.height), (1280.0, 720.0));
        assert!(page.stage.left.is_none());
        assert_eq!(page.ops[0].name, "visits");
        assert!(page.ops[0].add);
        assert_eq!(page.gates[0].op, CmpOp::Gt);
        assert_eq!(page.gates[0].target, 4);

        let choice = &node.choices[0];
        assert_eq!(choice.label, "Stay");
        assert_eq!(choice.target, 1);
        assert_eq!(choice.condition.as_ref().expect("condition").op, CmpOp::Ge);
    }

    #[test]
    fn a_graph_round_trips_through_postcard() {
        // Stories ride the blob in the baked form, so the whole nested graph has
        // to survive a format that carries no field names.
        let mut s = Story {
            title: alloc::string::String::from("Ash"),
            ..Story::default()
        };
        s.nodes.push(StoryNode {
            slug: alloc::string::String::from("intro"),
            pages: vec![StoryPage {
                text: alloc::string::String::from("Hello"),
                jump: Some(2),
                music: Some(AudioClipHandle(1)),
                sounds: vec![AudioClipHandle(2)],
                stage: StoryStage {
                    center: Some(StoryImage {
                        texture: TextureHandle(3),
                        width: 512.0,
                        ..StoryImage::default()
                    }),
                    ..StoryStage::default()
                },
                ..StoryPage::default()
            }],
            choice_gates: vec![StoryGate {
                op: CmpOp::Le,
                target: 7,
                ..StoryGate::default()
            }],
            ..StoryNode::default()
        });
        s.scaffold.slot_labels.push(AssetId(9));

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: Story = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.title, "Ash");
        let page = &back.nodes[0].pages[0];
        assert_eq!(page.jump, Some(2));
        assert_eq!(page.music, Some(AudioClipHandle(1)));
        assert_eq!(page.sounds, vec![AudioClipHandle(2)]);
        assert_eq!(
            page.stage.center.as_ref().expect("center image").texture,
            TextureHandle(3)
        );
        assert_eq!(back.nodes[0].choice_gates[0].op, CmpOp::Le);
        assert_eq!(back.scaffold.slot_labels, vec![AssetId(9)]);
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
