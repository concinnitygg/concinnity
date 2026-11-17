// Settings-row plumbing shared by the per-frame SettingCommand drain and the
// init-time row captures: publishing shown values and slider positions, the
// gray-out helpers for disabled rows, and the captures graphics init runs on
// the SettingsState it resolves.

use alloc::vec::Vec;

use crate::components::{Binding, HitRegion, Identity, ScrollPanel, TextLabel, WindowMode};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::settings::{SettingKey, SettingValues, StoredSettings, registry};
use crate::window::display_mode;

use super::SettingsState;

/// Muted gray applied to the labels of a capability-disabled settings row, so
/// it reads as unavailable next to the live rows.
pub const DISABLED_ROW_COLOR: [f32; 3] = [0.42, 0.42, 0.47];

// The full set of label ids to gray for a set of capability-gated rows: the
// gated value labels themselves (the fallback when a row is not in a scroll
// panel), plus every element of any scroll row that contains one of them, so a
// row dims as a whole (its name + value + stepper glyphs) rather than only its
// value. `rows` is each scroll row's element id list.
pub(crate) fn expand_dim_set(
    gated: &hashbrown::HashSet<AssetId>,
    rows: &[Vec<AssetId>],
) -> hashbrown::HashSet<AssetId> {
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
    ctx: &mut PipelineContext<'_>,
    rows: &[(AssetId, [f32; 3])],
    grayed: bool,
) {
    for &(id, orig) in rows {
        let color = if grayed { DISABLED_ROW_COLOR } else { orig };
        crate::ecs::by_asset_id::update::<TextLabel>(ctx, Some(id), |l| l.color = color);
    }
}

// The (label id, authored color) list of every settings row whose key is in
// `keys`, each expanded to its whole scroll row (background + name + value +
// glyphs) so the row grays as a unit; the authored colors drive the restore.
// Runs at init while the HitRegions / ScrollPanels are still present.
pub(crate) fn capture_row_labels(
    ctx: &PipelineContext<'_>,
    keys: &[SettingKey],
) -> Vec<(AssetId, [f32; 3])> {
    // Collect the rows' value-label ids (every region of a row, a stepper's
    // prev/next or a dropdown's open, references its value label).
    let mut anchors: hashbrown::HashSet<AssetId> = hashbrown::HashSet::new();
    for r in ctx.query::<HitRegion>() {
        if let Some(key) = r.setting()
            && keys.contains(&key)
            && let Some(label) = r.label
        {
            anchors.insert(label.id());
        }
    }
    if anchors.is_empty() {
        return Vec::new();
    }
    let rows: Vec<Vec<AssetId>> = ctx
        .query::<ScrollPanel>()
        .flat_map(|p| {
            p.rows
                .iter()
                .map(|r| r.elements.iter().map(|e| e.id()).collect())
        })
        .collect();
    let dim = expand_dim_set(&anchors, &rows);
    ctx.join2::<TextLabel, Identity>()
        .filter(|(_, _, identity)| dim.contains(&identity.id()))
        .map(|(_, l, identity)| (identity.id(), l.color))
        .collect()
}

// Every setting the overlay shows or operates, each once.
pub(crate) fn menu_keys(ctx: &PipelineContext<'_>) -> Vec<SettingKey> {
    let shown = ctx
        .query::<TextLabel>()
        .filter_map(|l| match &l.text_binding {
            Some(Binding::Setting(key)) => Some(*key),
            _ => None,
        });
    let operated = ctx.query::<HitRegion>().filter_map(HitRegion::setting);
    let mut keys: Vec<SettingKey> = Vec::new();
    for key in shown.chain(operated) {
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

// Show `text` as `key`'s value, for every label bound to it.
pub(crate) fn show(ctx: &mut PipelineContext<'_>, key: SettingKey, text: &str) {
    if let Some(values) = ctx.resource_mut::<SettingValues>() {
        values.set_text(key, text);
    }
}

// Place slider `key` at `fraction` of its range, for every handle bound to it.
pub(crate) fn place_slider(ctx: &mut PipelineContext<'_>, key: SettingKey, fraction: f32) {
    if let Some(values) = ctx.resource_mut::<SettingValues>() {
        values.set_fraction(key, fraction);
    }
}

impl SettingsState {
    // Publish every shown slider's position and value text, and every shown
    // rebind row's bound key or button. Runs once at init. `persisted`
    // serves every controls slider, which reads the store rather than the
    // render params.
    pub(crate) fn publish_rows(
        &self,
        ctx: &mut PipelineContext<'_>,
        persisted: &StoredSettings,
        keys: &[SettingKey],
    ) {
        for &key in keys {
            match key {
                SettingKey::KeyRebind(action) => {
                    show(ctx, key, self.keymap.get(action).display_name());
                }
                SettingKey::PadRebind(action) => {
                    show(ctx, key, self.gamepad_map.get(action).display_name());
                }
                _ => {
                    if let Some(slider) = registry::slider(key) {
                        let value = slider.current_value(&self.resolved.graphics, persisted);
                        place_slider(ctx, key, slider.fraction(value));
                        show(ctx, key, &(slider.format)(value));
                    }
                }
            }
        }
    }

    // Capture the show_fps / show_vram row labels (with their authored colors)
    // so the master "Display performance stats" toggle can gray them out at
    // runtime and restore them, and apply the initial gray from the resolved
    // toggle. Runs once at init while the HitRegions / ScrollPanels are present.
    pub(crate) fn capture_perf_sub_rows(&mut self, ctx: &mut PipelineContext<'_>) {
        self.perf_sub_row_labels =
            capture_row_labels(ctx, &[SettingKey::ShowFps, SettingKey::ShowVram]);
        set_rows_grayed(ctx, &self.perf_sub_row_labels, !self.resolved.perf_stats);
    }

    // Capture the Resolution row's labels and apply the initial gray from the
    // resolved window mode: the row only applies in fullscreen (windowed sizes
    // come from the window itself, borderless covers the display), so it is
    // grayed + inert in the other modes. Same init window as the perf rows.
    pub(crate) fn capture_resolution_row(&mut self, ctx: &mut PipelineContext<'_>) {
        self.resolution_row_labels = capture_row_labels(ctx, &[SettingKey::Resolution]);
        set_rows_grayed(
            ctx,
            &self.resolution_row_labels,
            self.window_args.mode != WindowMode::Fullscreen,
        );
    }

    // Capture the SSR row's labels and apply the initial gray. Screen-space
    // reflections stay a live choice, the fallback where ray tracing is off,
    // but no frame runs them while ray-traced reflections do.
    pub(crate) fn capture_ssr_row(&mut self, ctx: &mut PipelineContext<'_>) {
        self.ssr_row_labels = capture_row_labels(ctx, &[SettingKey::Ssr]);
        self.gray_superseded_ssr_row(ctx);
    }

    // Gray the SSR row while ray-traced reflections are on and the device runs
    // them, or while a launch flag fixes it; restore it otherwise.
    pub(crate) fn gray_superseded_ssr_row(&self, ctx: &mut PipelineContext<'_>) {
        let superseded =
            self.resolved.graphics.reflections.ray_traced && self.capabilities.ray_tracing;
        let grayed = superseded || self.fixed_by_launch(SettingKey::Ssr);
        set_rows_grayed(ctx, &self.ssr_row_labels, grayed);
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
    use crate::components::ScrollRow;
    use crate::ecs::{Ref, World};
    use alloc::vec;
    use hashbrown::HashSet;

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
    // value label itself dims, a graceful fallback rather than a panic.
    #[test]
    fn dim_set_without_rows_falls_back_to_the_value_label() {
        let gated: HashSet<AssetId> = [AssetId(7)].into_iter().collect();
        assert_eq!(expand_dim_set(&gated, &[]), gated);
    }

    fn label(color: [f32; 3]) -> TextLabel {
        TextLabel {
            color,
            ..Default::default()
        }
    }

    fn region(action: serde_json::Value, label: Option<u32>) -> HitRegion {
        HitRegion {
            action: Some(serde_json::from_value(action).unwrap()),
            label: label.map(AssetId).map(Ref::new),
            ..Default::default()
        }
    }

    // Showing a value and placing a slider publish into SettingValues; with
    // no SettingValues resource they are no-ops rather than panics.
    #[test]
    fn show_and_place_slider_publish_into_setting_values() {
        let mut world = World::new();
        let mut ctx = world.context();
        show(&mut ctx, SettingKey::Vsync, "On");
        place_slider(&mut ctx, SettingKey::Exposure, 0.5);
        assert!(ctx.resource::<SettingValues>().is_none());

        ctx.insert_resource(SettingValues::default());
        show(&mut ctx, SettingKey::Vsync, "On");
        place_slider(&mut ctx, SettingKey::Exposure, 0.5);
        let values = ctx.resource::<SettingValues>().unwrap();
        assert_eq!(values.text(SettingKey::Vsync), Some("On"));
        assert_eq!(values.fraction(SettingKey::Exposure), Some(0.5));
    }

    // The menu's keys are every setting a label shows or a region operates,
    // each once, in the order first met.
    #[test]
    fn menu_keys_lists_shown_and_operated_settings_once() {
        let mut world = World::new();
        world.push(TextLabel {
            text_binding: Some(Binding::Setting(SettingKey::Vsync)),
            ..Default::default()
        });
        world.push(TextLabel {
            text_binding: Some(Binding::State("title".into())),
            ..Default::default()
        });
        world.push(region(
            serde_json::json!({"step": {"binding": "setting:vsync", "by": 1}}),
            None,
        ));
        world.push(HitRegion {
            slider: Some(Binding::Setting(SettingKey::Exposure)),
            ..Default::default()
        });
        world.push(region(
            serde_json::json!({"setting": {"key": "window_mode", "verb": "open"}}),
            None,
        ));
        world.push(region(serde_json::json!("quit"), None));
        let ctx = world.context();

        assert_eq!(
            menu_keys(&ctx),
            [
                SettingKey::Vsync,
                SettingKey::Exposure,
                SettingKey::WindowMode
            ]
        );
    }

    // A shown slider publishes its fraction and value text, and a shown rebind
    // row its bound key.
    #[test]
    fn publish_rows_publishes_sliders_and_rebinds() {
        use crate::input::keymap::Bindable;
        let mut world = World::new();
        world.insert_resource(SettingValues::default());
        let state = SettingsState::for_tests();
        let stored = StoredSettings::default();
        let forward = SettingKey::KeyRebind(Bindable::Forward);
        let mut ctx = world.context();

        state.publish_rows(&mut ctx, &stored, &[SettingKey::Exposure, forward]);
        let values = ctx.resource::<SettingValues>().unwrap();
        let exposure = registry::slider(SettingKey::Exposure).unwrap();
        let value = exposure.current_value(&state.resolved.graphics, &stored);
        assert_eq!(
            values.fraction(SettingKey::Exposure),
            Some(exposure.fraction(value))
        );
        assert_eq!(
            values.text(SettingKey::Exposure),
            Some((exposure.format)(value).as_str())
        );
        assert_eq!(
            values.text(forward),
            Some(state.keymap.get(Bindable::Forward).display_name())
        );
    }

    // Graying a captured row set recolors every listed label, and ungraying
    // restores each label's own authored color rather than a shared default.
    #[test]
    fn set_rows_grayed_grays_then_restores_authored_colors() {
        let authored = [[0.9, 0.9, 0.9], [0.2, 0.6, 1.0]];
        let mut world = World::new();
        world.push_identified(AssetId(1), label(authored[0]));
        world.push_identified(AssetId(2), label(authored[1]));
        let rows = [(AssetId(1), authored[0]), (AssetId(2), authored[1])];
        let mut ctx = world.context();

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
        let mut world = World::new();
        for id in [1, 2, 3, 4, 5, 20] {
            world.push_identified(AssetId(id), label([id as f32 / 100.0; 3]));
        }
        world.push(region(
            serde_json::json!({"step": {"binding": "setting:shadow_map_size", "by": 1}}),
            Some(3),
        ));
        world.push(region(
            serde_json::json!({"step": {"binding": "setting:vsync", "by": 1}}),
            Some(20),
        ));
        world.push(region(serde_json::json!("quit"), Some(1)));
        world.push(region(
            serde_json::json!({"step": {"binding": "setting:shadow_map_size", "by": -1}}),
            None,
        ));
        world.push(ScrollPanel {
            rows: vec![
                ScrollRow {
                    elements: vec![
                        Ref::new(AssetId(1)),
                        Ref::new(AssetId(2)),
                        Ref::new(AssetId(3)),
                        Ref::new(AssetId(4)),
                        Ref::new(AssetId(5)),
                    ],
                    ..Default::default()
                },
                ScrollRow {
                    elements: vec![Ref::new(AssetId(20))],
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let ctx = world.context();

        let captured = capture_row_labels(&ctx, &[SettingKey::ShadowMapSize]);
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
        let mut world = World::new();
        world.push_identified(AssetId(1), label([1.0; 3]));
        world.push(region(
            serde_json::json!({"step": {"binding": "setting:shadow_map_size", "by": 1}}),
            Some(1),
        ));
        let ctx = world.context();

        assert!(capture_row_labels(&ctx, &[SettingKey::Resolution]).is_empty());
    }

    // A perf sub-row captured while the stats master is off starts grayed, and
    // one captured while it is on keeps its authored color.
    #[test]
    fn capture_perf_sub_rows_applies_the_initial_gray() {
        let authored = [0.8, 0.8, 0.8];
        for (perf_stats, expected) in [(false, DISABLED_ROW_COLOR), (true, authored)] {
            let mut world = World::new();
            world.push_identified(AssetId(1), label(authored));
            world.push(region(
                serde_json::json!({"step": {"binding": "setting:show_fps", "by": 1}}),
                Some(1),
            ));
            let mut state = SettingsState::for_tests();
            state.resolved.perf_stats = perf_stats;
            let mut ctx = world.context();

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
            let mut world = World::new();
            world.push_identified(AssetId(1), label(authored));
            world.push(region(
                serde_json::json!({"step": {"binding": "setting:resolution", "by": 1}}),
                Some(1),
            ));
            let mut state = SettingsState::for_tests();
            state.window_args.mode = mode;
            let mut ctx = world.context();

            state.capture_resolution_row(&mut ctx);
            assert_eq!(state.resolution_row_labels, [(AssetId(1), authored)]);
            assert_eq!(
                ctx.query::<TextLabel>().next().unwrap().color,
                expected,
                "{mode:?}"
            );
        }
    }

    // The SSR row grays only while ray-traced reflections are on and the device
    // can run them.
    #[test]
    fn capture_ssr_row_grays_while_rt_reflections_run() {
        let authored = [0.8, 0.8, 0.8];
        for (rt, capable, expected) in [
            (true, true, DISABLED_ROW_COLOR),
            (true, false, authored),
            (false, true, authored),
        ] {
            let mut world = World::new();
            world.push_identified(AssetId(1), label(authored));
            world.push(region(
                serde_json::json!({"step": {"binding": "setting:ssr", "by": 1}}),
                Some(1),
            ));
            let mut state = SettingsState::for_tests();
            state.resolved.graphics.reflections.ray_traced = rt;
            state.capabilities.ray_tracing = capable;
            let mut ctx = world.context();

            state.capture_ssr_row(&mut ctx);
            assert_eq!(state.ssr_row_labels, [(AssetId(1), authored)]);
            assert_eq!(
                ctx.query::<TextLabel>().next().unwrap().color,
                expected,
                "rt {rt}, capable {capable}"
            );
        }
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
