// src/editor/hook/panels.rs
//
// The `Panel` registry implementations: one stateless unit per floating panel,
// binding its module (geometry + draw) to the hook state that backs it. The
// shared machinery -- dragging, focus, close buttons, injection, draw layers,
// the hidden pass -- lives on the registry consumers; each impl supplies only
// what is panel-specific.

use super::*;
use crate::editor::registry::{Panel, PanelKey};

pub(crate) struct AssetsPanel;

impl Panel for AssetsPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Assets
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        panel::max_size()
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Assets")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.panel_open
    }
    // Opening re-cooks the tree and focuses a cleared search field, ready to
    // type.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.toggle_assets();
        if hook.panel_open {
            hook.tree_stale = true;
            hook.tree_scroll = 0;
            hook.search_focus = true;
            widget::seed_field(world, panel::SEARCH_INPUT, "");
        }
    }
    // Closing keeps the tree state (like a View-checkbox untick); only the
    // transient picker / row-menu overlays are dropped.
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.panel_open = false;
        hook.picker_open = false;
        hook.row_menu = None;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        panel::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        vec![(panel::SEARCH_INPUT, "search")]
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Assets);
        let action = {
            let data = hook.panel_data(world);
            let view = hook.make_view(&data, [mx, my]);
            panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_panel(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Assets);
        panel::cursor_over_body(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, world: &mut World, delta: f32) {
        hook.scroll_tree(delta, world);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.tree_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Assets);
        let data = hook.panel_data(world);
        let view = hook.make_view(&data, mouse);
        panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        panel::apply(world, None, [0.0, 0.0], panel::size());
    }
}

pub(crate) struct EditPanel;

impl Panel for EditPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Edit
    }
    fn resizable(&self) -> bool {
        true
    }
    // The form is part of the assets UI: shown / interactive only while the
    // browse panel is on.
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.form_open() && hook.panel_open
    }
    fn close(&self, hook: &mut EditorHook, world: &mut World) {
        hook.apply_form(FormAction::Close, world);
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        form_panel::size(hook.form_fields.len())
    }
    // The field list tracks the type's args; the height resizes only when there
    // are more fields than the default window shows.
    fn max_size(&self, hook: &EditorHook) -> [f32; 2] {
        form_panel::max_size(hook.form_fields.len())
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        form_panel::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        form_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        form_panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        let mut ids = vec![(form_panel::NAME_INPUT, "name")];
        ids.extend((0..form::FIELD_POOL_MAX).map(|j| (form_panel::form_input(j), "")));
        ids
    }
    fn overlay_ids(&self, hook: &EditorHook) -> Vec<AssetId> {
        match hook.field_dropdown.is_some() {
            true => form_panel::dropdown_ids(),
            false => Vec::new(),
        }
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Edit);
        let action = {
            let data = hook.panel_data(world);
            let view = hook.make_form_view(&data, [mx, my]);
            form_panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_form(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Edit);
        form_panel::cursor_over(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, world: &mut World, delta: f32) {
        hook.scroll_form(delta, world);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Edit);
        let data = hook.panel_data(world);
        let view = hook.make_form_view(&data, mouse);
        form_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        form_panel::apply(world, None, [0.0, 0.0], form_panel::size(0));
    }
}

pub(crate) struct HealthPanel;

impl Panel for HealthPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Health
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Health")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.health_open
    }
    fn toggle(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.health_open = !hook.health_open;
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.health_open = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        health_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        health_panel::default_origin(vp)
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        health_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        health_panel::all_label_ids()
    }
    // Read-only: a body press is swallowed so it cannot reach the world.
    fn press(
        &self,
        _hook: &mut EditorHook,
        _world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        health_panel::hit_test(mx, my, o)
    }
    // The snapshot is refreshed on the hook's throttled sample, not here: `draw`
    // only has `&EditorHook`, and the syscalls behind it must not run per frame.
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        health_panel::apply(world, hook.health.snapshot(), o, mouse);
    }
    fn hide(&self, world: &mut World) {
        health_panel::hide_all(world);
    }
}

pub(crate) struct PreviewPanel;

impl Panel for PreviewPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Preview
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Preview")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.preview_open
    }
    fn toggle(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.preview_open = !hook.preview_open;
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.preview_open = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        preview::size()
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        preview::default_origin()
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        preview::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        preview::all_label_ids()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let _ = world;
        match preview::hit_test(mx, my, o) {
            Some(PreviewAction::TogglePlay) => {
                hook.sim_toggle_play();
                true
            }
            Some(PreviewAction::ToggleFly) => {
                hook.toggle_fly();
                true
            }
            Some(PreviewAction::ToggleAxes) => {
                hook.axes_visible = !hook.axes_visible;
                true
            }
            Some(PreviewAction::Consume) => true,
            None => false,
        }
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        preview::apply(
            world,
            o,
            preview::PreviewState {
                playing: hook.sim.playing(),
                fly: hook.fly,
                axes: hook.axes_visible,
            },
            mouse,
        );
    }
    fn hide(&self, world: &mut World) {
        preview::hide_all(world);
    }
}

pub(crate) struct ViewPanel;

impl Panel for ViewPanel {
    fn key(&self) -> PanelKey {
        PanelKey::View
    }
    fn resizable(&self) -> bool {
        true
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.view_open
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.view_open = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        view::size()
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        view::default_origin()
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        view::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        view::all_label_ids()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::View);
        match view::hit_test(mx, my, o, s) {
            Some(ViewAction::Toggle(i)) => {
                hook.toggle_view_row(i, world);
                true
            }
            Some(ViewAction::Consume) => true,
            None => false,
        }
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::View);
        view::apply(world, o, s, &hook.view_rows(), mouse);
    }
    fn hide(&self, world: &mut World) {
        view::hide_all(world);
    }
}

pub(crate) struct TemplatesPanel;

impl Panel for TemplatesPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Templates
    }
    fn resizable(&self) -> bool {
        true
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Templates")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.templates_open
    }
    fn toggle(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.templates_open = !hook.templates_open;
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.templates_open = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        templates::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        templates::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        templates::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        templates::all_label_ids()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let _ = world;
        let s = hook.effective_size(PanelKey::Templates);
        match templates::hit_test(mx, my, o, s) {
            Some(TemplatesAction::Pick(i)) => {
                hook.open_template_detail(i);
                true
            }
            Some(TemplatesAction::Consume) => true,
            None => false,
        }
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Templates);
        templates::apply(world, o, s, hook.open_template, mouse);
    }
    fn hide(&self, world: &mut World) {
        templates::hide_all(world);
    }
}

pub(crate) struct TemplateDetailPanel;

impl TemplateDetailPanel {
    // The open template's grouped-row count (1 when none is open, matching the
    // panel's minimum footprint).
    fn row_count(&self, hook: &EditorHook) -> usize {
        hook.open_template
            .map_or(1, |i| hook.template_rows(i).len())
    }
}

impl Panel for TemplateDetailPanel {
    fn key(&self) -> PanelKey {
        PanelKey::TemplateDetail
    }
    fn resizable(&self) -> bool {
        true
    }
    // Part of the Templates UI: shown only while the Templates list is open and
    // a template is picked.
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.templates_open && hook.open_template.is_some()
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.close_template_detail();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        template_panel::size(self.row_count(hook))
    }
    // The list tracks the template's asset count; the height resizes only when
    // there are more rows than the default window shows.
    fn max_size(&self, hook: &EditorHook) -> [f32; 2] {
        template_panel::max_size(self.row_count(hook))
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        template_panel::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        template_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        template_panel::all_label_ids()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let _ = world;
        let Some(i) = hook.open_template else {
            return false;
        };
        let s = hook.effective_size(PanelKey::TemplateDetail);
        let action = {
            let data = hook.template_detail_data(i);
            let view = hook.make_template_view(&data, [mx, my]);
            template_panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_template_detail(a);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        if hook.open_template.is_none() {
            return false;
        }
        let s = hook.effective_size(PanelKey::TemplateDetail);
        template_panel::cursor_over(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_template_list(delta);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let Some(i) = hook.open_template else {
            return;
        };
        let s = hook.effective_size(PanelKey::TemplateDetail);
        let data = hook.template_detail_data(i);
        let view = hook.make_template_view(&data, mouse);
        template_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        template_panel::apply(world, None, [0.0, 0.0], template_panel::size(0));
    }
}

pub(crate) struct LightingPanel;

impl Panel for LightingPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Lighting
    }
    fn resizable(&self) -> bool {
        true
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Lighting")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.lighting_open
    }
    // Opening (re)seeds the text controls from the current entries and drops
    // any stale focus / status from the last session.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.lighting_open = !hook.lighting_open;
        if hook.lighting_open {
            hook.lighting_focus = None;
            hook.lighting_status = None;
            hook.seed_lighting(world);
        }
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.lighting_open = false;
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        lighting_panel::size(lighting::rows(&hook.lighting_present()).len())
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        lighting_panel::default_origin()
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        lighting_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        lighting_panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        lighting_panel::all_field_ids()
            .into_iter()
            .map(|id| (id, ""))
            .collect()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Lighting);
        let action = {
            let data = hook.lighting_data();
            let view = hook.make_lighting_view(&data, [mx, my]);
            lighting_panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_lighting_action(a, world);
                true
            }
            None => false,
        }
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Lighting);
        let data = hook.lighting_data();
        let view = hook.make_lighting_view(&data, mouse);
        lighting_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        lighting_panel::apply(world, None, [0.0, 0.0], lighting_panel::size(0));
    }
}

pub(crate) struct StoryPanel;

impl Panel for StoryPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Story
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        story_panel::max_size()
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Story")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.story_open
    }
    // Opening (re)loads the source file, so the panel always starts from the
    // on-disk truth.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.story_open = !hook.story_open;
        if hook.story_open {
            hook.load_story(world);
        }
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.story_open = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        story_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        story_panel::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        story_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        story_panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        story_panel::all_field_ids()
            .into_iter()
            .map(|id| (id, ""))
            .collect()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Story);
        let action = {
            let view = hook.make_story_view([mx, my]);
            story_panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_story_action(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Story);
        story_panel::cursor_over_lines(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_story(delta);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.story_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Story);
        let view = hook.make_story_view(mouse);
        story_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        story_panel::apply(world, None, [0.0, 0.0], story_panel::size());
    }
}

pub(crate) struct ConsolePanel;

impl Panel for ConsolePanel {
    fn key(&self) -> PanelKey {
        PanelKey::Console
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        console_panel::max_size()
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Console")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.console_open
    }
    // Opening focuses a cleared command line (backtick does the same through
    // the hook's key drive).
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.toggle_console(world);
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.console_open = false;
        hook.console_focus = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        console_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        console_panel::default_origin(vp)
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        console_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        console_panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        console_panel::all_field_ids()
            .into_iter()
            .map(|id| (id, "/help"))
            .collect()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Console);
        match console_panel::hit_test(mx, my, o, s) {
            Some(a) => {
                hook.apply_console_action(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Console);
        console_panel::cursor_over_log(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_console(delta);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.console_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Console);
        let (lines, total, first) = hook.console_window();
        let ghost = hook.console_ghost(world);
        let view = hook.make_console_view(&lines, total, first, &ghost, mouse);
        console_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        console_panel::apply(world, None, [0.0, 0.0], console_panel::size());
    }
}

pub(crate) struct BehaviorPanel;

impl Panel for BehaviorPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Behavior
    }
    fn resizable(&self) -> bool {
        true
    }
    // The outline's row pool caps how tall it is worth growing the panel; a
    // chart has no such pool, so there it grows to the screen.
    fn max_size(&self, hook: &EditorHook) -> [f32; 2] {
        match hook.behavior_mode {
            ViewMode::Outline => behavior_panel::max_size(),
            _ => [f32::INFINITY, f32::INFINITY],
        }
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Behavior")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.behavior_open
    }
    // Opening re-reads the world's behaviors, so the panel always starts from
    // the current entry list rather than a stale selection.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.behavior_open = !hook.behavior_open;
        if hook.behavior_open {
            hook.open_behavior(world);
        }
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.behavior_open = false;
        hook.behavior_focus = false;
        hook.behavior_name_focus = false;
        hook.behavior_remove_armed = false;
        hook.behavior_picking = false;
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        match hook.behavior_mode {
            ViewMode::Chart => behavior_panel::chart_size(),
            ViewMode::Overview => behavior_panel::overview_size(),
            ViewMode::Outline => behavior_panel::size(),
        }
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        behavior_panel::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        behavior_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        behavior_panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        vec![
            (behavior_panel::VALUE_INPUT, "value"),
            (behavior_panel::NAME_INPUT, "name"),
            (behavior_panel::FILTER_INPUT, "filter"),
        ]
    }
    fn overlay_ids(&self, hook: &EditorHook) -> Vec<AssetId> {
        let mut ids = behavior_panel::status_ids();
        if hook.behavior_picking {
            ids.extend(behavior_panel::palette_ids());
        }
        ids
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Behavior);
        let action = {
            let data = hook.behavior_data();
            let view = hook.make_behavior_view(&data, [mx, my]);
            behavior_panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_behavior_action(a, world, [mx, my]);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Behavior);
        behavior_panel::cursor_over_body(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_behavior(delta);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.behavior_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Behavior);
        let data = hook.behavior_data();
        let view = hook.make_behavior_view(&data, mouse);
        behavior_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        behavior_panel::apply(world, None, [0.0, 0.0], behavior_panel::size());
    }
}

pub(crate) struct VariablesPanel;

impl Panel for VariablesPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Variables
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        variables_panel::max_size()
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Variables")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.variables_open
    }
    // Opening re-reads the world, so the panel always starts from the current
    // table and the names its behaviors use.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.variables_open = !hook.variables_open;
        if hook.variables_open {
            hook.open_variables(world);
        }
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.variables_open = false;
        hook.variables_name_focus = false;
        hook.variables_value_focus = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        variables_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        variables_panel::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        variables_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        variables_panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        vec![
            (variables_panel::NAME_INPUT, "name"),
            (variables_panel::VALUE_INPUT, "value"),
        ]
    }
    fn overlay_ids(&self, _hook: &EditorHook) -> Vec<AssetId> {
        variables_panel::status_ids()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Variables);
        let action = {
            let data = hook.variables_data();
            let view = hook.make_variables_view(&data, [mx, my]);
            variables_panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_variables_action(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Variables);
        variables_panel::cursor_over_body(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_variables(delta);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.variables_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Variables);
        let data = hook.variables_data();
        let view = hook.make_variables_view(&data, mouse);
        variables_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        variables_panel::apply(world, None, [0.0, 0.0], variables_panel::size());
    }
}

pub(crate) struct ImportPanel;

impl Panel for ImportPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Import
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        import_panel::max_size()
    }
    fn view_row(&self) -> Option<&'static str> {
        Some("Import")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.import_open
    }
    // Opening clears stale state and focuses the path field, ready to type.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.import_open = !hook.import_open;
        if hook.import_open {
            hook.import_status = None;
            hook.import_scroll = 0;
            hook.import_focus = true;
            widget::seed_field(world, import_panel::PATH_INPUT, "");
        }
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.import_open = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        import_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        import_panel::default_origin(vp[0])
    }
    fn sprite_ids(&self) -> Vec<AssetId> {
        import_panel::all_sprite_ids()
    }
    fn label_ids(&self) -> Vec<AssetId> {
        import_panel::all_label_ids()
    }
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        import_panel::all_field_ids()
            .into_iter()
            .map(|id| (id, "path/to/file.glb"))
            .collect()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Import);
        let action = {
            let rows = hook.import_rows();
            let view = hook.make_import_view(&rows, [mx, my]);
            import_panel::hit_test(&view, mx, my, o, s)
        };
        match action {
            Some(a) => {
                hook.apply_import_action(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Import);
        import_panel::cursor_over_list(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_imports(delta);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.import_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Import);
        let rows = hook.import_rows();
        let view = hook.make_import_view(&rows, mouse);
        import_panel::apply(world, Some(&view), o, s);
    }
    fn hide(&self, world: &mut World) {
        import_panel::apply(world, None, [0.0, 0.0], import_panel::size());
    }
}
