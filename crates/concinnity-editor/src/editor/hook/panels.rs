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
    fn view_row(&self) -> Option<&'static str> {
        Some("Assets")
    }
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.panel_open
    }
    fn toggle(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.toggle_assets();
    }
    // Closing keeps the browse state (like a View-checkbox untick); only the
    // transient combo / row-menu overlays are dropped.
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.panel_open = false;
        hook.combo = Combo::Closed;
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
        vec![(panel::FILTER_INPUT, "filter")]
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let action = {
            let data = hook.panel_data(world);
            let view = hook.make_view(&data, [mx, my]);
            panel::hit_test(&view, mx, my, o)
        };
        match action {
            Some(a) => {
                hook.apply_panel(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(
        &self,
        _hook: &EditorHook,
        _world: &World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        panel::cursor_over_body(mx, my, o)
    }
    fn scroll(&self, hook: &mut EditorHook, world: &mut World, delta: f32) {
        hook.scroll_list(delta, world);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let data = hook.panel_data(world);
        let view = hook.make_view(&data, mouse);
        panel::apply(world, Some(&view), o);
    }
    fn hide(&self, world: &mut World) {
        panel::apply(world, None, [0.0, 0.0]);
    }
}

pub(crate) struct EditPanel;

impl Panel for EditPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Edit
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
        ids.extend((0..form::FIELD_POOL).map(|j| (form_panel::form_input(j), "")));
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
        let action = {
            let data = hook.panel_data(world);
            let view = hook.make_form_view(&data, [mx, my]);
            form_panel::hit_test(&view, mx, my, o)
        };
        match action {
            Some(a) => {
                hook.apply_form(a, world);
                true
            }
            None => false,
        }
    }
    fn wheel_over(&self, hook: &EditorHook, world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let data = hook.panel_data(world);
        let view = hook.make_form_view(&data, [mx, my]);
        form_panel::cursor_over(&view, mx, my, o)
    }
    fn scroll(&self, hook: &mut EditorHook, world: &mut World, delta: f32) {
        hook.scroll_form(delta, world);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let data = hook.panel_data(world);
        let view = hook.make_form_view(&data, mouse);
        form_panel::apply(world, Some(&view), o);
    }
    fn hide(&self, world: &mut World) {
        form_panel::apply(world, None, [0.0, 0.0]);
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
            Some(PreviewAction::ToggleCapture) => {
                hook.world_capture = !hook.world_capture;
                true
            }
            Some(PreviewAction::Consume) => true,
            None => false,
        }
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        preview::apply(world, o, hook.world_capture, mouse);
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
        match view::hit_test(mx, my, o) {
            Some(ViewAction::Toggle(i)) => {
                hook.toggle_view_row(i, world);
                true
            }
            Some(ViewAction::Consume) => true,
            None => false,
        }
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        view::apply(world, o, &hook.view_rows(), mouse);
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
        match templates::hit_test(mx, my, o) {
            Some(TemplatesAction::Pick(i)) => {
                hook.open_template_detail(i);
                true
            }
            Some(TemplatesAction::Consume) => true,
            None => false,
        }
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        templates::apply(world, o, hook.open_template, mouse);
    }
    fn hide(&self, world: &mut World) {
        templates::hide_all(world);
    }
}

pub(crate) struct TemplateDetailPanel;

impl Panel for TemplateDetailPanel {
    fn key(&self) -> PanelKey {
        PanelKey::TemplateDetail
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
        let n = hook
            .open_template
            .map_or(1, |i| hook.template_rows(i).len());
        template_panel::size(n)
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
        let action = {
            let data = hook.template_detail_data(i);
            let view = hook.make_template_view(&data, [mx, my]);
            template_panel::hit_test(&view, mx, my, o)
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
        let Some(i) = hook.open_template else {
            return false;
        };
        let data = hook.template_detail_data(i);
        let view = hook.make_template_view(&data, [mx, my]);
        template_panel::cursor_over(&view, mx, my, o)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_template_list(delta);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let Some(i) = hook.open_template else {
            return;
        };
        let data = hook.template_detail_data(i);
        let view = hook.make_template_view(&data, mouse);
        template_panel::apply(world, Some(&view), o);
    }
    fn hide(&self, world: &mut World) {
        template_panel::apply(world, None, [0.0, 0.0]);
    }
}

pub(crate) struct LightingPanel;

impl Panel for LightingPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Lighting
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
        let action = {
            let data = hook.lighting_data();
            let view = hook.make_lighting_view(&data, [mx, my]);
            lighting_panel::hit_test(&view, mx, my, o)
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
        let data = hook.lighting_data();
        let view = hook.make_lighting_view(&data, mouse);
        lighting_panel::apply(world, Some(&view), o);
    }
    fn hide(&self, world: &mut World) {
        lighting_panel::apply(world, None, [0.0, 0.0]);
    }
}
