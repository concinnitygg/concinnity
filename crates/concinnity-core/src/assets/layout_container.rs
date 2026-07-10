// src/assets/layout_container.rs

use crate::assets::LayoutContainer;
use crate::ecs::{AssetOrigin, Component};

impl Component for LayoutContainer {
    const NAME: &'static str = "LayoutContainer";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Justify, LabelBox, LayoutRow, Placement};
    use crate::ecs::asset_id::{AssetId, intern_all, reset_interner};

    // The vertical inset matches the horizontal padding here so the existing
    // placement expectations (text origin = box top-left + pad) hold; the
    // renderer can supply a different `top_inset` when the box hugs the glyphs.
    fn boxed(w: f32, h: f32, pad: f32) -> LabelBox {
        LabelBox {
            w,
            h,
            pad,
            top_inset: pad,
        }
    }

    /// A single left-justified row places boxes edge-to-edge with `col_gap`
    /// between them, and insets each origin by the label's padding.
    #[test]
    fn single_row_left_packs_with_gap() {
        let c = LayoutContainer {
            x: 10.0,
            y: 20.0,
            col_gap: 4.0,
            row_gap: 5.0,
            rows: vec![LayoutRow {
                cols: vec![AssetId(1), AssetId(2)],
                justify: Justify::Left,
            }],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(30.0, 16.0, 2.0)),
            AssetId(2) => Some(boxed(50.0, 16.0, 2.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        assert_eq!(p.len(), 2);
        // First box at container origin; origin inset by its padding.
        assert_eq!(
            p[0],
            Placement {
                id: AssetId(1),
                x: 12.0,
                y: 22.0
            }
        );
        // Second box starts after first box width + col_gap = 10 + 30 + 4 = 44.
        assert_eq!(
            p[1],
            Placement {
                id: AssetId(2),
                x: 46.0,
                y: 22.0
            }
        );
    }

    /// Unknown / unmeasurable labels are dropped and reserve no space.
    #[test]
    fn unknown_labels_are_skipped() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 10.0,
            row_gap: 0.0,
            rows: vec![LayoutRow {
                cols: vec![AssetId(1), AssetId(99), AssetId(2)],
                justify: Justify::Left,
            }],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(20.0, 10.0, 0.0)),
            AssetId(2) => Some(boxed(20.0, 10.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].id, AssetId(1));
        assert_eq!(p[0].x, 0.0);
        // The missing label leaves no gap: second visible box at 0 + 20 + 10.
        assert_eq!(p[1].id, AssetId(2));
        assert_eq!(p[1].x, 30.0);
    }

    /// A second row stacks below the first by the first row's box height plus
    /// the row gap. A lone label on that row starts at the container's left,
    /// occupying the row beneath the wider row above.
    #[test]
    fn second_row_stacks_below_and_spans() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 5.0,
            row_gap: 6.0,
            rows: vec![
                LayoutRow {
                    cols: vec![AssetId(1), AssetId(2)],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![AssetId(3)],
                    justify: Justify::Left,
                },
            ],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(40.0, 18.0, 0.0)),
            AssetId(2) => Some(boxed(40.0, 18.0, 0.0)),
            AssetId(3) => Some(boxed(120.0, 14.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        assert_eq!(p.len(), 3);
        // Row 2 label drops by row 1 box height (18) + row_gap (6) = 24.
        let passes = p.iter().find(|pl| pl.id == AssetId(3)).unwrap();
        assert_eq!(passes.x, 0.0);
        assert_eq!(passes.y, 24.0);
    }

    /// Centering a narrow row offsets it by half the slack to the widest row.
    #[test]
    fn center_justify_offsets_by_half_slack() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 0.0,
            row_gap: 0.0,
            rows: vec![
                LayoutRow {
                    cols: vec![AssetId(1)],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![AssetId(2)],
                    justify: Justify::Center,
                },
            ],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(100.0, 10.0, 0.0)),
            AssetId(2) => Some(boxed(40.0, 10.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        let narrow = p.iter().find(|pl| pl.id == AssetId(2)).unwrap();
        // slack = 100 - 40 = 60; centered offset = 30.
        assert_eq!(narrow.x, 30.0);
    }

    /// SpaceBetween spreads a short row across the content width, distributing
    /// slack into the gaps between labels.
    #[test]
    fn space_between_distributes_slack_into_gaps() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 0.0,
            row_gap: 0.0,
            rows: vec![
                // Widest row sets content width to 200.
                LayoutRow {
                    cols: vec![AssetId(10)],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![AssetId(1), AssetId(2), AssetId(3)],
                    justify: Justify::SpaceBetween,
                },
            ],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(10) => Some(boxed(200.0, 10.0, 0.0)),
            AssetId(1) | AssetId(2) | AssetId(3) => Some(boxed(20.0, 10.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        let row = |id| p.iter().find(|pl: &&Placement| pl.id == id).unwrap().x;
        // Three 20px boxes in 200px → 140px slack over 2 gaps = 70px each.
        assert_eq!(row(AssetId(1)), 0.0);
        assert_eq!(row(AssetId(2)), 90.0); // 20 + 70
        assert_eq!(row(AssetId(3)), 180.0); // 90 + 20 + 70
    }

    /// Args round-trip through JSON the way the build pipeline reserializes
    /// them: label names intern to ids, justify parses kebab-case, and missing
    /// fields fall back to the defaults.
    #[test]
    fn args_deserialize_from_world_json() {
        reset_interner();
        intern_all(&["fps_chip", "vram_chip", "passes_chip"]);
        let json = r#"{
            "x": 8, "y": 8, "col_gap": 5, "row_gap": 5,
            "rows": [
                {"cols": ["fps_chip", "vram_chip"]},
                {"cols": ["passes_chip"], "justify": "space-between"}
            ]
        }"#;
        let c: LayoutContainer = serde_json::from_str(json).unwrap();
        assert_eq!(c.x, 8.0);
        assert_eq!(c.rows.len(), 2);
        assert_eq!(c.rows[0].cols, vec![AssetId(0), AssetId(1)]);
        assert_eq!(c.rows[0].justify, Justify::Left); // defaulted
        assert_eq!(c.rows[1].cols, vec![AssetId(2)]);
        assert_eq!(c.rows[1].justify, Justify::SpaceBetween);
    }
}
