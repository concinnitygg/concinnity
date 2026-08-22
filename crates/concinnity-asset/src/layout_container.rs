// Row-based label layout container schema.

use crate::AssetId;
use alloc::vec::Vec;

/// Horizontal placement of a row's labels within the container's content width
/// (the width of the widest row). Ignored when a row is as wide as the content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Justify {
    /// Pack labels against the left edge (the default).
    #[default]
    Left,
    /// Center the row within the content width.
    Center,
    /// Pack labels against the right edge.
    Right,
    /// Spread the row across the full content width, distributing the slack
    /// evenly between labels. A single-label row falls back to `Left`.
    SpaceBetween,
}

/// One horizontal row of labels inside a `LayoutContainer`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LayoutRow {
    /// The [TextLabel](#textlabel)s in this row, laid out left to right. Their
    /// own `x`/`y` are ignored: the container positions them.
    pub cols: Vec<AssetId>,
    /// How this row is placed within the container's content width.
    pub justify: Justify,
}

impl Default for LayoutRow {
    fn default() -> Self {
        Self {
            cols: Vec::new(),
            justify: Justify::Left,
        }
    }
}

/// Positions a set of [TextLabel](#textlabel)s as a stack of rows, so a HUD does
/// not have to hand-place every chip. Each row lays its labels out left to
/// right; rows stack top to bottom. The container owns the labels' on-screen
/// position: the labels keep their own styling (font, colour, background,
/// padding) but their `x`/`y` are overwritten each frame.
///
/// Sizing is content-driven: a label is measured at its current text, so the
/// layout reflows as live HUD values change width. A row with a single label
/// sits on its own line beneath the previous row, which is how a wide chip
/// (e.g. a multi-pass timing line) ends up spanning the width of the row above.
///
/// Labels referenced by `cols` are matched by name; a label whose font is not
/// loaded, or which is hidden, is skipped and reserves no space.
///
/// ```rust
/// # use concinnity_asset::LayoutContainer;
/// LayoutContainer {
///     x: 10.0,
///     y: 10.0,
///     col_gap: 6.0,
///     row_gap: 6.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LayoutContainer {
    /// Left edge of the container in window pixels.
    pub x: f32,
    /// Top edge of the container in window pixels.
    pub y: f32,
    /// Pixels between adjacent labels in a row, measured between their
    /// background boxes.
    pub col_gap: f32,
    /// Pixels between adjacent rows, measured between their background boxes.
    pub row_gap: f32,
    /// Rows of labels, top to bottom.
    pub rows: Vec<LayoutRow>,
    /// When false, the container leaves its labels where they are instead of
    /// repositioning them.
    pub visible: bool,
}

impl Default for LayoutContainer {
    fn default() -> Self {
        Self {
            x: 10.0,
            y: 10.0,
            col_gap: 6.0,
            row_gap: 6.0,
            rows: Vec::new(),
            visible: true,
        }
    }
}

/// A label's measured extent, used by [`LayoutContainer::layout`] to place it.
/// `w`/`h` are the full background-box size in pixels (the text extent grown by
/// `pad` on every side). `pad` is the horizontal inset from the box's left edge
/// to the text origin. `top_inset` is the vertical inset from the box's top edge
/// down to the text origin (the label's `y`); it is distinct from `pad` because
/// the box hugs the visible glyphs, which sit below the text origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelBox {
    /// Background-box width in pixels.
    pub w: f32,
    /// Background-box height in pixels.
    pub h: f32,
    /// Horizontal inset from the box's left edge to the text origin.
    pub pad: f32,
    /// Vertical inset from the box's top edge down to the text origin.
    pub top_inset: f32,
}

/// The resolved top-left text origin for one label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelPlacement {
    /// The label this placement is for.
    pub id: AssetId,
    /// Text-origin x in pixels from the window's top-left.
    pub x: f32,
    /// Text-origin y in pixels from the window's top-left.
    pub y: f32,
}

impl LayoutContainer {
    /// Resolve the text origin for every label in the container.
    ///
    /// `size_of` returns a label's measured box, or `None` to drop it from the
    /// layout (unknown label, unloaded font, hidden). The result is pure
    /// geometry: callers measure with their font metrics, this places the
    /// boxes. Boxes within a row are laid edge-to-edge with `col_gap` between
    /// them; rows are stacked with `row_gap` between them.
    pub fn layout(&self, size_of: impl Fn(AssetId) -> Option<LabelBox>) -> Vec<LabelPlacement> {
        let mut out = Vec::new();
        self.layout_into(size_of, &mut out);
        out
    }

    /// [`LayoutContainer::layout`] writing into a caller-owned buffer, so a
    /// per-frame caller reuses its capacity. Clears `out` first.
    pub fn layout_into(
        &self,
        size_of: impl Fn(AssetId) -> Option<LabelBox>,
        out: &mut Vec<LabelPlacement>,
    ) {
        out.clear();
        // A row's measurable width: cells laid edge-to-edge with the column
        // gap, dropping unknown labels.
        let row_width = |row: &LayoutRow| -> f32 {
            let mut sum = 0.0_f32;
            let mut n = 0usize;
            for &id in &row.cols {
                if let Some(b) = size_of(id) {
                    sum += b.w;
                    n += 1;
                }
            }
            if n == 0 {
                0.0
            } else {
                sum + self.col_gap * (n - 1) as f32
            }
        };

        // Content width is the widest row; narrower rows justify within it.
        let content_w = self.rows.iter().map(row_width).fold(0.0_f32, f32::max);

        let mut y_cursor = self.y;
        for row in &self.rows {
            let mut n = 0usize;
            let mut row_h = 0.0_f32;
            for &id in &row.cols {
                if let Some(b) = size_of(id) {
                    n += 1;
                    row_h = row_h.max(b.h);
                }
            }
            if n > 0 {
                let rw = row_width(row);
                let slack = (content_w - rw).max(0.0);
                let (start, gap) = match row.justify {
                    Justify::Left => (0.0, self.col_gap),
                    Justify::Right => (slack, self.col_gap),
                    Justify::Center => (slack / 2.0, self.col_gap),
                    Justify::SpaceBetween => {
                        if n > 1 {
                            (0.0, self.col_gap + slack / (n - 1) as f32)
                        } else {
                            (0.0, self.col_gap)
                        }
                    }
                };
                let mut x_cursor = self.x + start;
                for &id in &row.cols {
                    let Some(b) = size_of(id) else {
                        continue;
                    };
                    // Box occupies [x_cursor, x_cursor + b.w]; the text origin the
                    // renderer wants is inset from the box's top-left by the
                    // horizontal padding and the (possibly different) vertical
                    // inset, since the box hugs the visible glyphs.
                    out.push(LabelPlacement {
                        id,
                        x: x_cursor + b.pad,
                        y: y_cursor + b.top_inset,
                    });
                    x_cursor += b.w + gap;
                }
            }
            y_cursor += row_h + self.row_gap;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // A label 10 wide and 20 tall per unit of its id, with no padding, so a
    // placement's x/y is the box corner and widths stay easy to add up.
    fn box_of(id: AssetId) -> Option<LabelBox> {
        Some(LabelBox {
            w: 10.0 * id.0 as f32,
            h: 20.0,
            pad: 0.0,
            top_inset: 0.0,
        })
    }

    fn row(justify: Justify, cols: &[u32]) -> LayoutRow {
        LayoutRow {
            cols: cols.iter().copied().map(AssetId).collect(),
            justify,
        }
    }

    fn container(rows: Vec<LayoutRow>) -> LayoutContainer {
        LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 2.0,
            row_gap: 4.0,
            rows,
            ..LayoutContainer::default()
        }
    }

    #[test]
    fn defaults_place_an_empty_container_in_the_top_left() {
        let c = LayoutContainer::default();
        assert_eq!((c.x, c.y), (10.0, 10.0));
        assert_eq!((c.col_gap, c.row_gap), (6.0, 6.0));
        assert!(c.visible);
        assert!(c.rows.is_empty());
        assert!(c.layout(box_of).is_empty());
        assert_eq!(LayoutRow::default().justify, Justify::Left);
        assert_eq!(Justify::default(), Justify::Left);
    }

    #[test]
    fn a_row_lays_its_labels_out_edge_to_edge_with_the_column_gap() {
        let c = container(vec![row(Justify::Left, &[1, 2])]);
        let out = c.layout(box_of);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].x, out[0].y), (0.0, 0.0));
        // Second box starts after the first box's width plus the gap.
        assert_eq!(out[1].x, 12.0);
    }

    #[test]
    fn rows_stack_by_the_tallest_box_plus_the_row_gap() {
        let c = container(vec![row(Justify::Left, &[1]), row(Justify::Left, &[1])]);
        let out = c.layout(box_of);
        assert_eq!(out[0].y, 0.0);
        assert_eq!(out[1].y, 24.0);
    }

    #[test]
    fn a_narrow_row_justifies_within_the_widest_row() {
        // Wide row is 10 + 2 + 20 = 32; narrow row is 10, so 22 of slack.
        let rows = |j| vec![row(Justify::Left, &[1, 2]), row(j, &[1])];
        let x_of_narrow = |j| container(rows(j)).layout(box_of)[2].x;
        assert_eq!(x_of_narrow(Justify::Left), 0.0);
        assert_eq!(x_of_narrow(Justify::Center), 11.0);
        assert_eq!(x_of_narrow(Justify::Right), 22.0);
        // A single-label row has nothing to spread between, so it packs left.
        assert_eq!(x_of_narrow(Justify::SpaceBetween), 0.0);
    }

    #[test]
    fn space_between_spreads_the_slack_across_the_gaps() {
        let c = container(vec![
            row(Justify::Left, &[4]),
            row(Justify::SpaceBetween, &[1, 1, 1]),
        ]);
        // Wide row is 40; the three narrow boxes are 30 + 2 gaps = 34, so 6 of
        // slack split over 2 gaps: each gap grows from 2 to 5.
        let out = c.layout(box_of);
        assert_eq!(out[1].x, 0.0);
        assert_eq!(out[2].x, 15.0);
        assert_eq!(out[3].x, 30.0);
    }

    #[test]
    fn a_label_that_cannot_be_measured_is_dropped_and_reserves_no_space() {
        // An unknown label, an unloaded font, or a hidden label measures to None.
        let c = container(vec![row(Justify::Left, &[1, 2, 3])]);
        let out = c.layout(|id| if id.0 == 2 { None } else { box_of(id) });
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, AssetId(1));
        assert_eq!(out[1].id, AssetId(3));
        // Box 3 follows box 1 directly, as though box 2 was never declared.
        assert_eq!(out[1].x, 12.0);
    }

    #[test]
    fn an_empty_row_still_advances_the_cursor_by_the_row_gap() {
        let c = container(vec![row(Justify::Left, &[]), row(Justify::Left, &[1])]);
        let out = c.layout(box_of);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].y, 4.0);
    }

    #[test]
    fn padding_insets_the_text_origin_from_the_box_corner() {
        // The box hugs the visible glyphs, which sit below the text origin, so
        // the vertical inset is its own number rather than the padding.
        let c = container(vec![row(Justify::Left, &[1])]);
        let out = c.layout(|id| {
            Some(LabelBox {
                pad: 3.0,
                top_inset: 7.0,
                ..box_of(id).unwrap()
            })
        });
        assert_eq!((out[0].x, out[0].y), (3.0, 7.0));
    }

    #[test]
    fn rows_parse_from_authored_args_and_round_trip_through_postcard() {
        crate::test_support::install_resolvers();
        let c: LayoutContainer = serde_json::from_str(
            r#"{"x":10,"y":10,"col_gap":6,"row_gap":6,
                "rows":[{"cols":["fps_chip","ev_chip"],"justify":"space-between"},
                        {"cols":["passes_chip"]}]}"#,
        )
        .unwrap();
        assert_eq!(c.rows.len(), 2);
        assert_eq!(c.rows[0].justify, Justify::SpaceBetween);
        assert_eq!(c.rows[1].justify, Justify::Left);
        assert_eq!(c.rows[0].cols, [AssetId(8), AssetId(7)]);

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: LayoutContainer = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.rows[0].justify, Justify::SpaceBetween);
        assert_eq!(back.rows[1].cols, [AssetId(11)]);
    }

    #[test]
    fn layout_into_clears_the_reused_buffer_before_placing() {
        let c = container(vec![row(Justify::Left, &[1, 2])]);
        let mut out = Vec::new();
        c.layout_into(box_of, &mut out);
        c.layout_into(box_of, &mut out);
        assert_eq!(out.len(), 2, "a reused buffer holds one solve, not two");
        assert_eq!(out, c.layout(box_of));
    }

    #[test]
    fn justify_names_parse_in_kebab_case() {
        let j = |s: &str| serde_json::from_str::<Justify>(s).unwrap();
        assert_eq!(j(r#""left""#), Justify::Left);
        assert_eq!(j(r#""center""#), Justify::Center);
        assert_eq!(j(r#""right""#), Justify::Right);
        assert_eq!(j(r#""space-between""#), Justify::SpaceBetween);
        assert_eq!(
            serde_json::to_string(&Justify::SpaceBetween).unwrap(),
            r#""space-between""#
        );
    }
}
