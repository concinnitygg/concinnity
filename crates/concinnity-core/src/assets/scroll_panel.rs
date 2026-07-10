// src/assets/scroll_panel.rs

use crate::assets::ScrollPanel;
use crate::ecs::{AssetOrigin, Component};

impl Component for ScrollPanel {
    const NAME: &'static str = "ScrollPanel";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{ScrollGroup, ScrollRow};
    use crate::ecs::asset_id::{AssetId, intern_all, reset_interner};

    #[test]
    fn bare_args_deserialize_with_defaults() {
        let p: ScrollPanel = serde_json::from_str("{}").unwrap();
        assert!(p.view.is_none());
        assert!(p.rows.is_empty());
        assert!(p.groups.is_empty());
        assert!(p.thumb.is_none());
    }

    #[test]
    fn rows_and_refs_resolve_from_names() {
        reset_interner();
        // Declaration order assigns ids: row0a=0, row0b=1, hdr=2, body=3,
        // thumb=4, track=5, view=6.
        intern_all(&["row0a", "row0b", "hdr", "body", "thumb", "track", "menu"]);
        let json = r#"{
            "view": "menu",
            "x": 10, "y": 20, "width": 300, "height": 200,
            "rows": [
                {"elements": ["row0a", "row0b"], "base_y": 20, "height": 40, "group": -1},
                {"elements": ["body"], "base_y": 60, "height": 40, "group": 0}
            ],
            "groups": [{"collapsed": true, "header": "hdr", "title": "Advanced"}],
            "thumb": "thumb", "track": "track",
            "track_x": 305, "track_y": 20, "track_w": 6, "track_h": 200
        }"#;
        let p: ScrollPanel = serde_json::from_str(json).unwrap();
        assert_eq!(p.view, Some(AssetId(6)));
        assert_eq!(p.rows.len(), 2);
        assert_eq!(p.rows[0].elements, vec![AssetId(0), AssetId(1)]);
        assert_eq!(p.rows[0].group, -1);
        assert_eq!(p.rows[1].elements, vec![AssetId(3)]);
        assert_eq!(p.rows[1].group, 0);
        assert_eq!(p.groups.len(), 1);
        assert!(p.groups[0].collapsed);
        assert_eq!(p.groups[0].header, Some(AssetId(2)));
        assert_eq!(p.groups[0].title, "Advanced");
        assert_eq!(p.thumb, Some(AssetId(4)));
        assert_eq!(p.track, Some(AssetId(5)));
    }

    #[test]
    fn round_trips_through_serde() {
        let p = ScrollPanel {
            view: Some(AssetId(2)),
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
            rows: vec![ScrollRow {
                elements: vec![AssetId(5)],
                base_y: 2.0,
                height: 40.0,
                group: 0,
            }],
            groups: vec![ScrollGroup {
                collapsed: false,
                header: Some(AssetId(7)),
                title: "Advanced".to_string(),
            }],
            thumb: Some(AssetId(8)),
            track: Some(AssetId(9)),
            track_x: 5.0,
            track_y: 6.0,
            track_w: 7.0,
            track_h: 8.0,
        };
        let v = serde_json::to_value(&p).unwrap();
        let back: ScrollPanel = serde_json::from_value(v).unwrap();
        assert_eq!(back.rows.len(), 1);
        assert_eq!(back.rows[0].elements, vec![AssetId(5)]);
        assert_eq!(back.groups[0].title, "Advanced");
    }
}
