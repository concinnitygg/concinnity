// src/assets/story.rs

use crate::ecs::asset_id::AssetId;
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
    /// Music current at the choice menu ([AudioClip](#audioclip) asset name).
    pub choice_music: Option<String>,
    /// One-shots played when the choice menu shows.
    pub choice_sounds: Vec<String>,
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
    /// Music current at this page ([AudioClip](#audioclip) asset name).
    /// Re-triggering the already-playing track is seamless.
    pub music: Option<String>,
    /// One-shot effects played when the page shows.
    pub sounds: Vec<String>,
    /// Stage dressing current at this page.
    pub stage: StoryStage,
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
    /// [Texture](#texture) asset name.
    pub texture: String,
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
}

impl Default for Story {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            title: String::new(),
            nodes: Vec::new(),
            text_speed: 45.0,
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
    fn graph_round_trips() {
        let json = r#"{
            "title": "T",
            "text_speed": 30.0,
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
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["nodes"][0]["pages"][0]["music"], "s_clip0");
    }
}
