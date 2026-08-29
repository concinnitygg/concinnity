// src/components/component_tests.rs
//
// Serde / default / round-trip coverage for the data-only components whose
// `Component` impls are generated centrally (see `cn_impl_components!`). These
// checks are uniform in shape and were kept beside each type before its
// hand-written impl was removed; they live together here now that the per-type
// modules are gone. One submodule per component.
//
// The authoring-only vocabulary is not covered here: those types are not
// components, and their schema checks live beside them in their own modules.

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::components::*;

mod graphics_config {
    use super::*;
    use crate::components::ShadowUpdate;

    #[test]
    fn shadow_update_defaults_to_hybrid() {
        assert_eq!(
            GraphicsConfig::default().shadow_update,
            ShadowUpdate::Hybrid
        );
        assert_eq!(ShadowUpdate::default(), ShadowUpdate::Hybrid);
    }

    #[test]
    fn shadow_update_round_trips_via_snake_case_json() {
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_update":"every_frame"}"#).expect("parse");
        assert_eq!(cfg.shadow_update, ShadowUpdate::EveryFrame);
        // Omitting the field falls back to the hybrid default.
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_update, ShadowUpdate::Hybrid);
    }

    #[test]
    fn vsync_defaults_off_and_round_trips() {
        // Omitted -> uncapped (false).
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert!(!cfg.vsync);
        // Explicit true is honoured.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"vsync":true}"#).expect("parse");
        assert!(cfg.vsync);
    }

    #[test]
    fn fps_cap_defaults_to_unlimited_and_round_trips() {
        // Omitted -> 0 (uncapped).
        assert_eq!(GraphicsConfig::default().fps_cap, 0);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.fps_cap, 0);
        // Explicit cap is honoured.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"fps_cap":60}"#).expect("parse");
        assert_eq!(cfg.fps_cap, 60);
    }

    #[test]
    fn shadow_distance_defaults_to_80_and_round_trips() {
        assert_eq!(GraphicsConfig::default().shadow_distance, 80);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_distance":160}"#).expect("parse");
        assert_eq!(cfg.shadow_distance, 160);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_distance, 80);
    }

    #[test]
    fn shadow_cascades_defaults_to_4_and_round_trips() {
        assert_eq!(GraphicsConfig::default().shadow_cascades, 4);
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"shadow_cascades":2}"#).expect("parse");
        assert_eq!(cfg.shadow_cascades, 2);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_cascades, 4);
    }

    #[test]
    fn anisotropy_defaults_to_8_and_round_trips() {
        // The default matches the value the backends historically hardcoded.
        assert_eq!(GraphicsConfig::default().anisotropy, 8);
        // An authored value is honoured; omitting the field falls back to 8.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"anisotropy":16}"#).expect("parse");
        assert_eq!(cfg.anisotropy, 16);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.anisotropy, 8);
    }

    #[test]
    fn shadow_update_round_trips_through_args() {
        let cfg = GraphicsConfig {
            shadow_update: ShadowUpdate::EveryFrame,
            ..Default::default()
        };
        assert_eq!(
            serde_json::from_value::<GraphicsConfig>(serde_json::to_value(&cfg).unwrap())
                .unwrap()
                .shadow_update,
            ShadowUpdate::EveryFrame
        );
    }
}

mod key_binding {
    use super::*;

    #[test]
    fn deserializes_escape_to_view_toggle() {
        let json = r#"{"key":"Escape","action":"screen:toggle:pause_menu"}"#;
        let kb: KeyBinding = serde_json::from_str(json).unwrap();
        assert_eq!(kb.key, "Escape");
        assert_eq!(kb.action, "screen:toggle:pause_menu");
    }

    #[test]
    fn deserializes_with_defaults_to_empty_strings() {
        let kb: KeyBinding = serde_json::from_str("{}").unwrap();
        assert!(kb.key.is_empty());
        assert!(kb.action.is_empty());
    }
}

mod screen {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let s: Screen = serde_json::from_str("{}").unwrap();
        assert_eq!(s.fade_in_secs, 0.0);
        assert!(!s.initial);
        assert!(s.toggle_key.is_empty());
        assert_eq!(s.input, ScreenInput::Capture);
        assert!(s.pauses_world);
        assert!(s.focus.is_none());
        assert_eq!(s.layer, 0);
    }

    #[test]
    fn deserializes_with_authored_fields() {
        let s: Screen = serde_json::from_str(
            r#"{"initial":true,"toggle_key":"Backtick","input":"passthrough","pauses_world":false,"layer":-2}"#,
        )
        .unwrap();
        assert!(s.initial);
        assert_eq!(s.toggle_key, "Backtick");
        assert_eq!(s.input, ScreenInput::Passthrough);
        assert!(!s.pauses_world);
        assert_eq!(s.layer, -2);
    }
}

mod story {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let s: Story = serde_json::from_str("{}").unwrap();
        assert!(s.nodes.is_empty());
        assert_eq!(s.text_speed, 45.0);
    }

    #[test]
    fn graph_round_trips_and_resolves_references() {
        crate::test_support::reset_interner();
        let json = r#"{
                "title": "T",
                "text_speed": 30.0,
                "scaffold": {
                    "screen": "s_stage",
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
        use crate::test_support::intern;
        assert_eq!(s.scaffold.screen, Some(intern("s_stage")));
        assert_eq!(s.scaffold.options, vec![intern("s_stage_opt0_lbl")]);
        assert_eq!(s.scaffold.option_boxes, vec![intern("s_stage_opt0_box")]);
        assert_eq!(s.scaffold.dialog_box, None);
        // Audio-clip and texture references resolve to their per-kind handle.
        // With no handle resolver installed (this is a bare serde test, not a
        // build), the deserializer falls back to the name interner, exactly like
        // an `AssetId`, so the handle value is the interned id of the name.
        use crate::ecs::{AudioClipHandle, TextureHandle};
        assert_eq!(page.music, Some(AudioClipHandle(intern("s_clip0").0)));
        assert_eq!(page.sounds, vec![AudioClipHandle(intern("s_clip1").0)]);
        assert_eq!(
            page.stage.bg.as_ref().unwrap().texture,
            TextureHandle(intern("s_img0").0)
        );
        // Ids serialize back out as integers (the blob path carries only
        // numbers).
        let back = serde_json::to_value(&s).unwrap();
        assert!(back["nodes"][0]["pages"][0]["music"].is_number());
    }
}

mod sprite {
    use super::*;
    use crate::components::SpriteFit;

    #[test]
    fn deserializes_with_all_fields() {
        let json = r#"{
                "x": 10, "y": 20, "width": 300, "height": 200,
                "tint": [0.5, 0.5, 0.5, 0.8], "visible": true
            }"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert_eq!(s.x, 10.0);
        assert_eq!(s.width, 300.0);
        assert_eq!(s.tint, [0.5, 0.5, 0.5, 0.8]);
        assert!(s.visible);
        assert!(s.texture.is_none());
    }

    #[test]
    fn deserializes_with_defaults() {
        let s: Sprite = serde_json::from_str("{}").unwrap();
        assert_eq!(s.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(s.visible);
        assert_eq!(s.width, 100.0);
        assert!(!s.follow_cursor);
        assert_eq!(s.fit, SpriteFit::Fit);
        assert_eq!(s.corner_radius, 0.0);
        assert_eq!(s.border_width, 0.0);
    }

    #[test]
    fn border_round_trips() {
        let json = r#"{"border_width":1.5,"border_color":[0.5,0.25,0.75,1.0]}"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert_eq!(s.border_width, 1.5);
        assert_eq!(s.border_color, [0.5, 0.25, 0.75, 1.0]);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["border_width"], 1.5);
        assert_eq!(back["border_color"][0], 0.5);
    }

    #[test]
    fn corner_radius_round_trips() {
        let s: Sprite = serde_json::from_str(r#"{"corner_radius":12.5}"#).unwrap();
        assert_eq!(s.corner_radius, 12.5);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["corner_radius"], 12.5);
    }

    #[test]
    fn fit_deserializes_lowercase_and_round_trips() {
        let s: Sprite = serde_json::from_str(r#"{"fit":"cover"}"#).unwrap();
        assert_eq!(s.fit, SpriteFit::Cover);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["fit"], "cover");
    }

    #[test]
    fn follow_cursor_round_trips() {
        let json = r#"{"follow_cursor":true,"width":16,"height":16}"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert!(s.follow_cursor);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["follow_cursor"], true);
    }

    #[test]
    fn deserializes_with_texture_reference() {
        crate::test_support::reset_interner();
        // The interner is global and not reset here; we just check the field
        // is populated when a string name is supplied (it interns lazily).
        let json = r#"{"texture":"tex_intro","tint":[1,1,1,1]}"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert!(s.texture.is_some());
    }
}

mod streaming_config {
    use super::*;

    #[test]
    fn default_is_a_moderate_budget_and_a_full_cap() {
        let c = StreamingConfig::default();
        assert_eq!(c.texture_budget, 4);
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.budget(), 4);
        assert_eq!(c.cap(), 96);
        assert_eq!(c.mesh_budget(), 4);
        assert_eq!(c.mesh_cap(), 4096);
    }

    #[test]
    fn zero_budget_and_cap_are_floored_at_one() {
        let c = StreamingConfig {
            texture_budget: 0,
            texture_cap: 0,
            mesh_budget: 0,
            mesh_cap: 0,
            ..Default::default()
        };
        // A 0 here would otherwise stall streaming forever.
        assert_eq!(c.budget(), 1);
        assert_eq!(c.cap(), 1);
        assert_eq!(c.mesh_budget(), 1);
        assert_eq!(c.mesh_cap(), 1);
    }

    #[test]
    fn deserialises_from_jsonl_args_with_defaults_for_omitted_fields() {
        let c: StreamingConfig =
            serde_json::from_str(r#"{"texture_budget":2,"mesh_budget":2}"#).expect("parse");
        assert_eq!(c.texture_budget, 2);
        assert_eq!(c.mesh_budget, 2);
        // Omitted fields fall back to the defaults.
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.mesh_cap, 4096);

        // An empty object is all defaults.
        let c: StreamingConfig = serde_json::from_str("{}").expect("parse");
        assert_eq!(c.texture_budget, 4);
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.mesh_budget, 4);
        assert_eq!(c.mesh_cap, 4096);
        // The byte-budget fields default to 0 (derive from GPU memory).
        assert_eq!(c.texture_budget_mb, 0);
        assert_eq!(c.mesh_budget_mb, 0);
    }

    #[test]
    fn round_trips_through_args() {
        let c = StreamingConfig {
            texture_budget: 7,
            texture_cap: 32,
            mesh_budget: 3,
            mesh_cap: 64,
            texture_budget_mb: 512,
            mesh_budget_mb: 256,
        };
        let back: StreamingConfig =
            serde_json::from_value(serde_json::to_value(&c).unwrap()).unwrap();
        assert_eq!(back.texture_budget, 7);
        assert_eq!(back.texture_cap, 32);
        assert_eq!(back.mesh_budget, 3);
        assert_eq!(back.mesh_cap, 64);
        assert_eq!(back.texture_budget_mb, 512);
        assert_eq!(back.mesh_budget_mb, 256);
    }
}

mod app_config {
    use super::*;

    // The bake keeps only what the running process needs: the state location
    // and the budgets. The distribution strings are consumed at build / export
    // time and never ship in the blob.
    #[test]
    fn bake_keeps_the_home_and_the_budgets() {
        let args: crate::components::cook::AppConfig = serde_json::from_str(
            r#"{"name":"My Game","home":"state","max_memory_mb":512,"job_threads":2}"#,
        )
        .unwrap();
        let baked = AppConfig::bake(args);
        assert_eq!(baked.home, "state");
        assert_eq!(baked.max_memory_mb, 512);
        assert_eq!(baked.job_threads, 2);
    }
}

mod voxel_world {
    use super::*;

    #[test]
    fn default_is_a_modest_window() {
        let w = VoxelWorld::default();
        assert_eq!(w.chunk_blocks(), [16, 24, 16]);
        assert_eq!(w.view_radius(), 5);
        assert_eq!(w.load_budget(), 3);
        assert_eq!(w.chunk_world_size(), (16.0, 16.0));
    }

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
    fn deserialises_from_jsonl_args_with_defaults_for_omitted_fields() {
        let w: VoxelWorld = serde_json::from_str(r#"{"seed":7,"view_radius":8}"#).expect("parse");
        assert_eq!(w.seed, 7);
        assert_eq!(w.view_radius(), 8);
        // omitted fields fall back to the defaults
        assert_eq!(w.chunk_blocks(), [16, 24, 16]);
        assert_eq!(w.load_budget(), 3);
    }

    #[test]
    fn round_trips_through_args() {
        let w = VoxelWorld {
            seed: 99,
            chunk_blocks: [8, 32, 8],
            block_size: 2.0,
            view_radius: 4,
            impostor_radius: 12,
            impostor_step: 2,
            load_budget: 5,
            palette: Vec::new(),
            material: None,
        };
        let back: VoxelWorld = serde_json::from_value(serde_json::to_value(&w).unwrap()).unwrap();
        assert_eq!(back.seed, 99);
        assert_eq!(back.chunk_blocks, [8, 32, 8]);
        assert_eq!(back.block_size, 2.0);
        assert_eq!(back.impostor_radius, 12);
        assert_eq!(back.impostor_step, 2);
        assert_eq!(back.load_budget, 5);
    }

    #[test]
    fn impostors_disabled_by_default() {
        let w = VoxelWorld::default();
        // Default impostor_radius 0 -> clamped up to view_radius -> no far band.
        assert_eq!(w.impostor_radius(), w.view_radius());
        assert!(!w.impostors_enabled());
        assert_eq!(w.impostor_step(), 4);
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
    use crate::test_support::{intern_all, reset_interner};

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
        let row = |id| p.iter().find(|pl: &&LabelPlacement| pl.id == id).unwrap().x;
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

mod scroll_panel {
    use super::*;
    use crate::components::{ScrollGroup, ScrollRow};
    use crate::ecs::asset_id::AssetId;
    use crate::test_support::{intern_all, reset_interner};

    #[test]
    fn bare_args_deserialize_with_defaults() {
        let p: ScrollPanel = serde_json::from_str("{}").unwrap();
        assert!(p.screen.is_none());
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
                "screen": "menu",
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
        assert_eq!(p.screen, Some(AssetId(6)));
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
            screen: Some(AssetId(2)),
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

mod text_input {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let t: TextInput = serde_json::from_str("{}").unwrap();
        assert_eq!(t.content, "");
        assert_eq!(t.width, 240.0);
        assert_eq!(t.max_len, 0);
        assert!(t.visible);
        assert!(!t.focused);
        assert_eq!(t.caret, 0);
    }

    #[test]
    fn deserializes_with_fields() {
        let json = r#"{
                "placeholder": "Name", "content": "hi",
                "x": 10, "y": 20, "width": 300, "height": 48, "max_len": 24
            }"#;
        let t: TextInput = serde_json::from_str(json).unwrap();
        assert_eq!(t.placeholder, "Name");
        assert_eq!(t.content, "hi");
        assert_eq!(t.width, 300.0);
        assert_eq!(t.max_len, 24);
    }

    #[test]
    fn runtime_state_is_not_serialized() {
        // `focused` / `caret` / `asset_id` are runtime-only, so `args` (the
        // public schema) never carries them.
        let t = TextInput {
            focused: true,
            caret: 3,
            ..Default::default()
        };
        let v = serde_json::to_value(&t).unwrap();
        assert!(v.get("focused").is_none());
        assert!(v.get("caret").is_none());
        assert!(v.get("asset_id").is_none());
        assert!(v.is_object());
    }
}
