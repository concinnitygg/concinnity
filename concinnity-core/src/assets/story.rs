// src/assets/story.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// A compiled branching story graph, played at runtime by the story system.
///
/// A `Story` is normally produced by a [StoryImport](#storyimport) expansion
/// at build time rather than written by hand: the Markdown source compiles
/// into this graph plus the stage scaffolding (a single dialogue
/// [View](#view) whose labels and sprites the story system mutates page by
/// page). All references are pre-resolved: dialog text is pre-wrapped,
/// speakers carry their display name and color, stage images carry their
/// on-canvas rectangle, and jump / choice targets are node indices into
/// `nodes`.
///
/// The story system reads the graph and drives the stage view named
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
/// [View](#view)s, [Sprite](#sprite)s, and [TextLabel](#textlabel)s the
/// story system mutates page by page.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryScaffold {
    /// The stage [View](#view) the story plays inside.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub view: Option<AssetId>,
    /// The [View](#view) shown when the story ends.
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
    /// The title screen [View](#view), returned to when the load overlay is
    /// dismissed before play started.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub title: Option<AssetId>,
    /// The title screen's Load [TextLabel](#textlabel), hidden while no
    /// slot save exists.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub load_label: Option<AssetId>,
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
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub choice_music: Option<AssetId>,
    /// One-shots played when the choice menu shows.
    pub choice_sounds: Vec<AssetId>,
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
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub music: Option<AssetId>,
    /// One-shot effects played when the page shows.
    pub sounds: Vec<AssetId>,
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
    pub texture: AssetId,
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

impl Component for Story {
    const NAME: &'static str = "Story";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
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
    /// stage view reference.
    pub story: Story,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let s: Story = serde_json::from_str("{}").unwrap();
        assert!(s.nodes.is_empty());
        assert_eq!(s.text_speed, 45.0);
    }

    #[test]
    fn graph_round_trips_and_resolves_references() {
        let json = r#"{
            "title": "T",
            "text_speed": 30.0,
            "scaffold": {
                "view": "s_stage",
                "ending": "s_ending",
                "bg": "s_stage_bg",
                "text_label": "s_stage_text",
                "option_boxes": ["s_stage_opt0_box"],
                "options": ["s_stage_opt0_lbl"]
            },
            "nodes": [{
                "slug": "inn",
                "pages": [{
                    "speaker": {"name": "Ayame", "color": [1.0, 0.85, 0.8]},
                    "text": "You came.",
                    "music": "s_clip0",
                    "sounds": ["s_clip1"],
                    "stage": {
                        "bg": {"texture": "s_img0", "x": 0, "y": 0, "width": 1280, "height": 720},
                        "center": {"texture": "s_img1", "x": 412, "y": 20, "width": 456, "height": 700}
                    }
                }],
                "choices": [
                    {"label": "Into the wood", "target": 1},
                    {"label": "Toward the shore", "target": 2}
                ]
            }]
        }"#;
        let s: Story = serde_json::from_str(json).unwrap();
        assert_eq!(s.nodes.len(), 1);
        let page = &s.nodes[0].pages[0];
        assert_eq!(page.speaker.as_ref().unwrap().name, "Ayame");
        assert_eq!(page.stage.center.as_ref().unwrap().width, 456.0);
        assert_eq!(s.nodes[0].choices[1].target, 2);
        // Name-string references resolved to ids through the interner (the
        // build-time path); the omitted dialog_box stays unset.
        use crate::ecs::asset_id::intern;
        assert_eq!(s.scaffold.view, Some(intern("s_stage")));
        assert_eq!(s.scaffold.options, vec![intern("s_stage_opt0_lbl")]);
        assert_eq!(s.scaffold.option_boxes, vec![intern("s_stage_opt0_box")]);
        assert_eq!(s.scaffold.dialog_box, None);
        assert_eq!(page.music, Some(intern("s_clip0")));
        assert_eq!(page.sounds, vec![intern("s_clip1")]);
        assert_eq!(page.stage.bg.as_ref().unwrap().texture, intern("s_img0"));
        // Ids serialize back out as integers (the blob path carries only
        // numbers).
        let back = serde_json::to_value(&s).unwrap();
        assert!(back["nodes"][0]["pages"][0]["music"].is_number());
    }
}
