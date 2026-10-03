//! Behavior checks for data-only components whose `Component` impls are
//! generated centrally (see `cn_impl_components!`): the clamped accessors of
//! `VoxelWorld`, the row packing of `LayoutContainer`, and the runtime-only
//! state of `TextInput`. One submodule per component.

use crate::ecs::Ref;
use alloc::vec;

use crate::components::*;

mod voxel_world {
    use super::*;

    #[test]
    fn degenerate_args_are_floored_and_clamped() {
        let w = VoxelWorld {
            chunk_blocks: [0, 0, 0],
            block_size: -1.0,
            view_radius: 9999,
            load_budget: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.chunk_blocks(), [1, 1, 1]);
        assert!(w.block_size() > 0.0);
        assert_eq!(w.view_radius(), 32);
        assert_eq!(w.load_budget(), 1);
    }

    #[test]
    fn a_zero_impostor_radius_disables_the_far_band() {
        let w = VoxelWorld {
            impostor_radius: 0,
            ..VoxelWorld::default()
        };
        // Clamped up to the view radius, which leaves no far band.
        assert_eq!(w.impostor_radius(), w.view_radius());
        assert!(!w.impostors_enabled());
    }

    #[test]
    fn impostor_radius_enables_the_far_band_and_clamps() {
        let w = VoxelWorld {
            view_radius: 5,
            impostor_radius: 16,
            impostor_step: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.impostor_radius(), 16);
        assert!(w.impostors_enabled());
        // step floored at 1.
        assert_eq!(w.impostor_step(), 1);

        // An impostor radius below the view radius disables impostors.
        let w2 = VoxelWorld {
            view_radius: 8,
            impostor_radius: 4,
            ..VoxelWorld::default()
        };
        assert_eq!(w2.impostor_radius(), w2.view_radius());
        assert!(!w2.impostors_enabled());
    }
}

mod layout_container {
    use super::*;
    use crate::components::{Justify, LabelBox, LabelPlacement, LayoutRow};
    use crate::ecs::asset_id::AssetId;

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
                cols: vec![Ref::new(AssetId(1)), Ref::new(AssetId(2))],
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
            LabelPlacement {
                id: AssetId(1),
                x: 12.0,
                y: 22.0
            }
        );
        // Second box starts after first box width + col_gap = 10 + 30 + 4 = 44.
        assert_eq!(
            p[1],
            LabelPlacement {
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
                cols: vec![
                    Ref::new(AssetId(1)),
                    Ref::new(AssetId(99)),
                    Ref::new(AssetId(2)),
                ],
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
                    cols: vec![Ref::new(AssetId(1)), Ref::new(AssetId(2))],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![Ref::new(AssetId(3))],
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
                    cols: vec![Ref::new(AssetId(1))],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![Ref::new(AssetId(2))],
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
                    cols: vec![Ref::new(AssetId(10))],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![
                        Ref::new(AssetId(1)),
                        Ref::new(AssetId(2)),
                        Ref::new(AssetId(3)),
                    ],
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
        let row = |id| p.iter().find(|pl: &&LabelPlacement| pl.id == id).unwrap().x;
        // Three 20px boxes in 200px → 140px slack over 2 gaps = 70px each.
        assert_eq!(row(AssetId(1)), 0.0);
        assert_eq!(row(AssetId(2)), 90.0); // 20 + 70
        assert_eq!(row(AssetId(3)), 180.0); // 90 + 20 + 70
    }
}

mod text_input {
    use super::*;

    #[test]
    fn runtime_state_is_not_serialized() {
        // `focused` / `caret` are runtime-only, so `args` (the
        // public schema) never carries them.
        let t = TextInput {
            focused: true,
            caret: 3,
            ..Default::default()
        };
        let v = serde_json::to_value(&t).unwrap();
        assert!(v.get("focused").is_none());
        assert!(v.get("caret").is_none());
        assert!(v.is_object());
    }
}
