// src/assets/story.rs

use crate::assets::Story;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

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
    fn graph_round_trips_and_resolves_references() {
        crate::ecs::asset_id::reset_interner();
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
