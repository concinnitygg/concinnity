// Settings-row plumbing shared by the per-frame SettingCommand drain and the
// init-time row captures: label / sprite writers, the gray-out helpers for
// disabled rows, and the captures GraphicsSystem's init runs on the
// SettingsState it resolves.

use concinnity_core::components::{
    GamepadAction, HitRegion, ScrollPanel, Sprite, TextLabel, WindowMode,
};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::render::{display_mode, keymap};
use concinnity_host::thread::asset_id::AssetId;

use super::SettingsState;
use crate::gfx::system::{PadRebindViz, RebindViz, SliderViz};
use crate::settings;
use crate::settings::action;

// Muted gray applied to the labels of a capability-disabled settings row, so it
// reads as unavailable next to the live rows.
pub(crate) const DISABLED_ROW_COLOR: [f32; 3] = [0.42, 0.42, 0.47];

// The full set of label ids to gray for a set of capability-gated rows: the
// gated value labels themselves (the fallback when a row is not in a scroll
// panel), plus every element of any scroll row that contains one of them, so a
// row dims as a whole (its name + value + stepper glyphs) rather than only its
// value. `rows` is each scroll row's element id list.
pub(crate) fn expand_dim_set(
    gated: &std::collections::HashSet<AssetId>,
    rows: &[Vec<AssetId>],
) -> std::collections::HashSet<AssetId> {
    let mut dim = gated.clone();
    for row in rows {
        if row.iter().any(|id| gated.contains(id)) {
            dim.extend(row.iter().copied());
        }
    }
    dim
}

// Gray a captured set of settings-row labels (or restore their authored
// colors), for a row disabled at runtime: the show_fps / show_vram rows under
// the "Display performance stats" master, and the Resolution row outside
// fullscreen. A free function taking the captured (id, color) list so it can
// run inside the settings drain, where the backend is a live borrow. The
// matching input inertness comes from the `DisabledSettingRows` resource,
// published after the drain.
pub(crate) fn set_rows_grayed(
    ctx: &mut PipelineContext,
    rows: &[(AssetId, [f32; 3])],
    grayed: bool,
) {
    for &(id, orig) in rows {
        let color = if grayed { DISABLED_ROW_COLOR } else { orig };
        for l in ctx.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.color = color;
                break;
            }
        }
    }
}

// The (label id, authored color) list of every settings row whose key is in
// `keys`, each expanded to its whole scroll row (background + name + value +
// glyphs) so the row grays as a unit; the authored colors drive the restore.
// Runs at init while the HitRegions / ScrollPanels are still present.
pub(crate) fn capture_row_labels(
    ctx: &mut PipelineContext,
    keys: &[&str],
) -> Vec<(AssetId, [f32; 3])> {
    // Collect the rows' value-label ids (every region of a row -- steppers'
    // prev/next or a dropdown's open -- references its value label).
    let mut anchors: std::collections::HashSet<AssetId> = std::collections::HashSet::new();
    for r in ctx.query::<HitRegion>() {
        let Some((key, _)) = action::parse(&r.action) else {
            continue;
        };
        if keys.contains(&key)
            && let Some(label) = r.label
        {
            anchors.insert(label);
        }
    }
    if anchors.is_empty() {
        return Vec::new();
    }
    let rows: Vec<Vec<AssetId>> = ctx
        .query::<ScrollPanel>()
        .flat_map(|p| p.rows.iter().map(|r| r.elements.clone()))
        .collect();
    let dim = expand_dim_set(&anchors, &rows);
    ctx.query::<TextLabel>()
        .filter(|l| dim.contains(&l.asset_id))
        .map(|l| (l.asset_id, l.color))
        .collect()
}

// Overwrite the text of the TextLabel with the given id, if present.
pub(crate) fn set_label_content(ctx: &mut PipelineContext, id: AssetId, text: &str) {
    crate::ecs::by_asset_id::set_text(ctx, id, text);
}

// Set a cycle row's value label from its init-captured id. Used to update a
// row other than the one that was clicked (the master preset relabels the
// quality toggles + render scale; a quality-toggle change relabels the master
// row). The menu's HitRegions are drained after init, so the row -> label map
// is captured once rather than re-queried here.
pub(crate) fn set_cached_row_label(
    labels: &std::collections::HashMap<String, AssetId>,
    ctx: &mut PipelineContext,
    key: &str,
    text: &str,
) {
    if let Some(&id) = labels.get(key) {
        set_label_content(ctx, id, text);
    }
}

// Move the Sprite with the given id to `x` (its left edge), if present. Used to
// slide a slider's handle along its track.
pub(crate) fn set_sprite_x(ctx: &mut PipelineContext, id: AssetId, x: f32) {
    crate::ecs::by_asset_id::update::<Sprite>(ctx, Some(id), |s| s.x = x);
}

impl SettingsState {
    // Capture each slider row's runtime bookkeeping from its drag HitRegion +
    // handle Sprite, then sync the handle position and value label to the live
    // value. Runs once at init, before UiInputSystem drains the HitRegions and
    // hides the screen elements. `persisted` serves every controls slider, which
    // reads the store rather than the render params.
    pub(crate) fn init_sliders(
        &mut self,
        ctx: &mut PipelineContext,
        persisted: &crate::config::Settings,
    ) {
        let sprite_w: std::collections::HashMap<AssetId, f32> = ctx
            .query::<Sprite>()
            .map(|s| (s.asset_id, s.width))
            .collect();
        let mut sliders: Vec<SliderViz> = Vec::new();
        for r in ctx.query::<HitRegion>() {
            let Some(key) = action::key_with_verb(&r.action, "drag") else {
                continue;
            };
            let (Some(handle_id), Some(value_id)) = (r.drag_handle, r.label) else {
                continue;
            };
            let handle_w = sprite_w.get(&handle_id).copied().unwrap_or(0.0);
            sliders.push(SliderViz {
                key: key.to_string(),
                track_x: r.x,
                track_w: r.width,
                handle_w,
                handle_id,
                value_id,
            });
        }
        for s in &sliders {
            let Some(slider) = settings::slider(&s.key) else {
                continue;
            };
            let value = slider.current_value(
                &self.post_process,
                &self.post_config,
                self.ambient_intensity,
                persisted,
            );
            let hx = s.track_x + slider.fraction(value) * (s.track_w - s.handle_w).max(0.0);
            set_sprite_x(ctx, s.handle_id, hx);
            set_label_content(ctx, s.value_id, &(slider.format)(value));
        }
        self.sliders = sliders;
    }

    // Capture each key-rebind row's bookkeeping from its `setting:key_*:rebind`
    // HitRegion, then sync each value label to the live bound key. Runs once at
    // init (after the keymap is seeded), before UiInputSystem drains the
    // HitRegions.
    pub(crate) fn init_rebind_rows(&mut self, ctx: &mut PipelineContext) {
        let mut rows: Vec<RebindViz> = Vec::new();
        let mut pad_rows: Vec<PadRebindViz> = Vec::new();
        for r in ctx.query::<HitRegion>() {
            let (Some(key), Some(value_id)) = (action::key_with_verb(&r.action, "rebind"), r.label)
            else {
                continue;
            };
            // A `key_*` setting is a keyboard rebind row; a `pad_*` setting is
            // a gamepad rebind row.
            if let Some(action) = keymap::Bindable::from_setting_key(key) {
                rows.push(RebindViz { action, value_id });
            } else if let Some(action) = GamepadAction::from_setting_key(key) {
                pad_rows.push(PadRebindViz { action, value_id });
            }
        }
        for row in &rows {
            let name = self.keymap.get(row.action).display_name();
            set_label_content(ctx, row.value_id, name);
        }
        for row in &pad_rows {
            let name = self.gamepad_map.get(row.action).display_name();
            set_label_content(ctx, row.value_id, name);
        }
        self.rebind_rows = rows;
        self.pad_rebind_rows = pad_rows;
    }

    // Capture each cycle row's setting key -> value-label id, so a runtime change
    // can relabel a row other than the one clicked (the master preset relabels
    // its dependents; a quality-toggle change relabels the master row). Runs at
    // init, before UiInputSystem drains the HitRegions.
    pub(crate) fn init_cycle_value_labels(&mut self, ctx: &mut PipelineContext) {
        let mut labels = std::collections::HashMap::new();
        for r in ctx.query::<HitRegion>() {
            if let (Some(key), Some(value_id)) = (action::cycle_key(&r.action), r.label) {
                labels.insert(key.to_string(), value_id);
            }
        }
        self.cycle_value_labels = labels;
    }

    // Capture the show_fps / show_vram row labels (with their authored colors)
    // so the master "Display performance stats" toggle can gray them out at
    // runtime and restore them, and apply the initial gray from the resolved
    // toggle. Runs once at init while the HitRegions / ScrollPanels are present.
    pub(crate) fn capture_perf_sub_rows(&mut self, ctx: &mut PipelineContext) {
        self.perf_sub_row_labels = capture_row_labels(ctx, &["show_fps", "show_vram"]);
        set_rows_grayed(ctx, &self.perf_sub_row_labels, !self.perf_stats);
    }

    // Capture the Resolution row's labels and apply the initial gray from the
    // resolved window mode: the row only applies in fullscreen (windowed sizes
    // come from the window itself, borderless covers the display), so it is
    // grayed + inert in the other modes. Same init window as the perf rows.
    pub(crate) fn capture_resolution_row(&mut self, ctx: &mut PipelineContext) {
        self.resolution_row_labels = capture_row_labels(ctx, &["resolution"]);
        set_rows_grayed(
            ctx,
            &self.resolution_row_labels,
            self.window_args.mode != WindowMode::Fullscreen,
        );
    }

    // The mode the Resolution row displays and cycles from: the user's choice,
    // else the display's own mode, else the authored window size (a backend
    // that cannot read the display; snaps to the nearest listed mode).
    pub(crate) fn effective_resolution(&self) -> display_mode::DisplayMode {
        self.resolution
            .or(self.current_mode)
            .unwrap_or(display_mode::DisplayMode {
                width: self.window_args.width,
                height: self.window_args.height,
                refresh_hz: 0,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::ScrollRow;
    use concinnity_core::ecs::Arena;
    use concinnity_core::ecs::ComponentSlot;
    use concinnity_core::ecs::ComponentStorage;
    use concinnity_core::ecs::FrameContext;
    use concinnity_core::ecs::Resources;
    use concinnity_core::gfx::profile;
    use concinnity_host::store::blob::BlobData;
    use std::collections::HashSet;

    // A gated value label pulls in every element of the scroll row that holds
    // it (the row's background, name, value, and stepper glyphs), so the whole
    // row grays out; unrelated rows are untouched.
    #[test]
    fn dim_set_expands_a_gated_value_label_to_its_whole_row() {
        let value = AssetId(3);
        let gated: HashSet<AssetId> = [value].into_iter().collect();
        let rows = vec![
            // Row A: bg, name, prev_glyph, value, next_glyph (value is gated).
            vec![AssetId(1), AssetId(2), value, AssetId(4), AssetId(5)],
            // Row B: an unrelated row.
            vec![AssetId(10), AssetId(11)],
        ];
        let dim = expand_dim_set(&gated, &rows);
        for id in [1, 2, 3, 4, 5] {
            assert!(dim.contains(&AssetId(id)), "row A element {id} should dim");
        }
        assert!(!dim.contains(&AssetId(10)), "an unrelated row stays lit");
        assert!(!dim.contains(&AssetId(11)), "an unrelated row stays lit");
    }

    // With no scroll rows (a hand-authored menu outside a panel), only the gated
    // value label itself dims -- a graceful fallback, not a panic.
    #[test]
    fn dim_set_without_rows_falls_back_to_the_value_label() {
        let gated: HashSet<AssetId> = [AssetId(7)].into_iter().collect();
        assert_eq!(expand_dim_set(&gated, &[]), gated);
    }

    // Owns the storage a PipelineContext borrows from. The helpers under test
    // only touch components, so the blob / profile / resources stay empty.
    struct TestWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: profile::FrameProfile,
        resources: Resources,
        scratch: Arena,
    }

    impl TestWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::new(vec![Some(Vec::new())]),
                profile: profile::FrameProfile::default(),
                resources: Resources::new(),
                scratch: Arena::with_capacity(64 * 1024),
            }
        }

        fn push<C: ComponentSlot>(&mut self, c: C) {
            self.components.push_typed(c);
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: FrameContext::new(&self.scratch),
            }
        }
    }

    fn label(id: u32, color: [f32; 3]) -> TextLabel {
        TextLabel {
            asset_id: AssetId(id),
            color,
            ..Default::default()
        }
    }

    fn region(action: &str, label: Option<u32>) -> HitRegion {
        HitRegion {
            action: action.to_string(),
            label: label.map(AssetId),
            ..Default::default()
        }
    }

    // Writing a label's content hits the one matching id and leaves the rest
    // alone; an id with no label is a no-op rather than a panic.
    #[test]
    fn set_label_content_writes_only_the_matching_label() {
        let mut world = TestWorld::new();
        world.push(label(1, [1.0; 3]));
        world.push(label(2, [1.0; 3]));
        let mut ctx = world.ctx();

        set_label_content(&mut ctx, AssetId(2), "High");
        let contents: Vec<&str> = ctx
            .query::<TextLabel>()
            .map(|l| l.content.as_str())
            .collect();
        assert_eq!(contents, ["", "High"]);

        set_label_content(&mut ctx, AssetId(9), "Ultra");
        let contents: Vec<&str> = ctx
            .query::<TextLabel>()
            .map(|l| l.content.as_str())
            .collect();
        assert_eq!(contents, ["", "High"], "an absent id changes nothing");
    }

    // Moving a slider handle hits the one matching sprite; an absent id is a
    // no-op.
    #[test]
    fn set_sprite_x_moves_only_the_matching_sprite() {
        let mut world = TestWorld::new();
        world.push(Sprite {
            asset_id: AssetId(1),
            x: 0.0,
            ..Default::default()
        });
        world.push(Sprite {
            asset_id: AssetId(2),
            x: 0.0,
            ..Default::default()
        });
        let mut ctx = world.ctx();

        set_sprite_x(&mut ctx, AssetId(2), 42.0);
        assert_eq!(
            ctx.query::<Sprite>().map(|s| s.x).collect::<Vec<_>>(),
            [0.0, 42.0]
        );

        set_sprite_x(&mut ctx, AssetId(9), 99.0);
        assert_eq!(
            ctx.query::<Sprite>().map(|s| s.x).collect::<Vec<_>>(),
            [0.0, 42.0],
            "an absent id changes nothing"
        );
    }

    // Graying a captured row set recolors every listed label, and ungraying
    // restores each label's own authored color rather than a shared default.
    #[test]
    fn set_rows_grayed_grays_then_restores_authored_colors() {
        let authored = [[0.9, 0.9, 0.9], [0.2, 0.6, 1.0]];
        let mut world = TestWorld::new();
        world.push(label(1, authored[0]));
        world.push(label(2, authored[1]));
        let rows = [(AssetId(1), authored[0]), (AssetId(2), authored[1])];
        let mut ctx = world.ctx();

        set_rows_grayed(&mut ctx, &rows, true);
        assert!(
            ctx.query::<TextLabel>()
                .all(|l| l.color == DISABLED_ROW_COLOR)
        );

        set_rows_grayed(&mut ctx, &rows, false);
        assert_eq!(
            ctx.query::<TextLabel>()
                .map(|l| l.color)
                .collect::<Vec<_>>(),
            authored
        );
    }

    // A capture keys off the rows' regions and returns the whole scroll row's
    // labels with their authored colors, skipping unrelated rows, regions with
    // no label, and non-setting actions.
    #[test]
    fn capture_row_labels_returns_a_matching_rows_labels_and_colors() {
        let mut world = TestWorld::new();
        for id in [1, 2, 3, 4, 5, 20] {
            world.push(label(id, [id as f32 / 100.0; 3]));
        }
        world.push(region("setting:shadows:next", Some(3)));
        world.push(region("setting:other:next", Some(20)));
        world.push(region("quit", Some(1)));
        world.push(region("setting:shadows:prev", None));
        world.push(ScrollPanel {
            rows: vec![
                ScrollRow {
                    elements: vec![AssetId(1), AssetId(2), AssetId(3), AssetId(4), AssetId(5)],
                    ..Default::default()
                },
                ScrollRow {
                    elements: vec![AssetId(20)],
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let mut ctx = world.ctx();

        let captured = capture_row_labels(&mut ctx, &["shadows"]);
        let expected: Vec<(AssetId, [f32; 3])> = [1, 2, 3, 4, 5]
            .into_iter()
            .map(|id| (AssetId(id), [id as f32 / 100.0; 3]))
            .collect();
        assert_eq!(captured, expected);
    }

    // A key no region carries captures nothing, so a row absent from a world's
    // menu simply has no gray-out set.
    #[test]
    fn capture_row_labels_without_a_matching_key_captures_nothing() {
        let mut world = TestWorld::new();
        world.push(label(1, [1.0; 3]));
        world.push(region("setting:shadows:next", Some(1)));
        let mut ctx = world.ctx();

        assert!(capture_row_labels(&mut ctx, &["resolution"]).is_empty());
    }

    // A perf sub-row captured while the stats master is off starts grayed, and
    // one captured while it is on keeps its authored color.
    #[test]
    fn capture_perf_sub_rows_applies_the_initial_gray() {
        let authored = [0.8, 0.8, 0.8];
        for (perf_stats, expected) in [(false, DISABLED_ROW_COLOR), (true, authored)] {
            let mut world = TestWorld::new();
            world.push(label(1, authored));
            world.push(region("setting:show_fps:next", Some(1)));
            let mut state = SettingsState::for_tests();
            state.perf_stats = perf_stats;
            let mut ctx = world.ctx();

            state.capture_perf_sub_rows(&mut ctx);
            assert_eq!(state.perf_sub_row_labels, [(AssetId(1), authored)]);
            assert_eq!(ctx.query::<TextLabel>().next().unwrap().color, expected);
        }
    }

    // The Resolution row starts grayed in every mode but fullscreen.
    #[test]
    fn capture_resolution_row_grays_outside_fullscreen() {
        let authored = [0.8, 0.8, 0.8];
        for (mode, expected) in [
            (WindowMode::Windowed, DISABLED_ROW_COLOR),
            (WindowMode::Borderless, DISABLED_ROW_COLOR),
            (WindowMode::Fullscreen, authored),
        ] {
            let mut world = TestWorld::new();
            world.push(label(1, authored));
            world.push(region("setting:resolution:next", Some(1)));
            let mut state = SettingsState::for_tests();
            state.window_args.mode = mode;
            let mut ctx = world.ctx();

            state.capture_resolution_row(&mut ctx);
            assert_eq!(state.resolution_row_labels, [(AssetId(1), authored)]);
            assert_eq!(
                ctx.query::<TextLabel>().next().unwrap().color,
                expected,
                "{mode:?}"
            );
        }
    }

    // Each cycle row's value label is captured under its key; drag and rebind
    // rows are not cycle rows.
    #[test]
    fn init_cycle_value_labels_maps_each_cycle_key_to_its_label() {
        let mut world = TestWorld::new();
        world.push(region("setting:vsync:next", Some(1)));
        world.push(region("setting:vsync:prev", Some(1)));
        world.push(region("setting:exposure:drag", Some(2)));
        let mut state = SettingsState::for_tests();
        let mut ctx = world.ctx();

        state.init_cycle_value_labels(&mut ctx);
        assert_eq!(state.cycle_value_labels.len(), 1);
        assert_eq!(state.cycle_value_labels.get("vsync"), Some(&AssetId(1)));
    }

    // The Resolution row reads the user's choice first, then the display's own
    // mode, then the authored window size.
    #[test]
    fn effective_resolution_falls_back_from_choice_to_display_to_window() {
        let mode = |width, height, refresh_hz| display_mode::DisplayMode {
            width,
            height,
            refresh_hz,
        };
        let mut state = SettingsState::for_tests();
        state.window_args.width = 640;
        state.window_args.height = 360;
        assert_eq!(state.effective_resolution(), mode(640, 360, 0));

        state.current_mode = Some(mode(2560, 1440, 144));
        assert_eq!(state.effective_resolution(), mode(2560, 1440, 144));

        state.resolution = Some(mode(1920, 1080, 60));
        assert_eq!(state.effective_resolution(), mode(1920, 1080, 60));
    }
}
