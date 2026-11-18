//! The `Panel` registry implementations: one stateless unit per floating panel,
//! binding its module (geometry + draw) to the hook state that backs it. The
//! shared machinery -- dragging, focus, close buttons, injection, draw layers,
//! open state, the hidden pass -- lives on the registry and its consumers; each
//! impl supplies only what is panel-specific.

use concinnity_engine::components::{FrameInput, UiEvent};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::EditorHook;
use crate::editor::behavior::panel::ViewMode;
use crate::editor::hud_ids::HudIds;
use crate::editor::panels::form_panel::{self, FormAction};
use crate::editor::panels::preview::{self, PreviewAction};
use crate::editor::panels::registry::{self, Panel, PanelKey};
use crate::editor::panels::template;
use crate::editor::panels::view;
use crate::editor::panels::{
    assets_panel, character_shape_panel, console_panel, content_panel, health_panel, import_panel,
    lighting, lighting_panel, planet, shader, story_panel, template_panel, variables_panel,
};
use crate::editor::ui::{BuiltPanel, UiBuilder, field};
use crate::editor::viewport::snap;
use crate::editor::{behavior, map, palette, widget, worlds};

// Apply a body press the panel resolved, if it resolved one: whether the press
// was taken.
fn handled<A>(action: Option<A>, apply: impl FnOnce(A)) -> bool {
    action.map(apply).is_some()
}

pub(crate) struct AssetsPanel;

impl Panel for AssetsPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Assets
    }
    // Toggling either way drops the transient overlays, like closing.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        let open = !hook.open[PanelKey::Assets];
        hook.open[PanelKey::Assets] = open;
        self.on_close(hook, world);
        if open {
            self.on_open(hook, world);
        }
    }
    // Opening re-cooks the tree and focuses a cleared search field, ready to
    // type.
    fn on_open(&self, hook: &mut EditorHook, world: &mut World) {
        hook.assets.stale = true;
        hook.assets.search_focus = true;
        EditorHook::clear_assets_search(world);
    }
    // Closing keeps the tree state (like a View-checkbox untick); only the
    // transient picker / row-menu overlays are dropped.
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.assets.picker_open = false;
        hook.assets.row_menu = None;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        assets_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        assets_panel::default_origin(vp[0])
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.tree_keys(world, input);
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        assets_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_assets(world, built);
    }
    fn restore(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        field::set_text(
            world,
            assets_panel::search_field(built),
            &hook.assets.filter,
        );
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, built: &mut BuiltPanel) {
        hook.prepare_assets(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        assets_panel::sync(&hook.make_assets_view(), world, built);
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = assets_panel::action(event, &hook.assets.rows, hook.menu_row()) else {
            return false;
        };
        hook.apply_panel(action);
        true
    }
}

pub(crate) struct EditPanel;

impl Panel for EditPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Edit
    }
    // The form is part of the UI of the panel it opened from: shown /
    // interactive only while that panel is.
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.form_open() && registry::panel(hook.form.host).is_open(hook)
    }
    fn close(&self, hook: &mut EditorHook, world: &mut World) {
        hook.apply_form(FormAction::Close, world);
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        form_panel::size(hook.form_row_count())
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        form_panel::default_origin(vp[0])
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        form_panel::build(ui)
    }
    // A world the panel was just built into holds none of the form's rows.
    fn restore(&self, hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) {
        hook.form.shown = None;
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_form(world, built);
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, built: &mut BuiltPanel) {
        hook.prepare_form(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        if let (Some(view), Some(shown)) = (hook.make_form_view(), &hook.form.shown) {
            form_panel::sync(&view, shown, world, built);
        }
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = form_panel::action(event, &hook.form.fields, hook.open_form_menu())
        else {
            return false;
        };
        hook.apply_form(action, world);
        true
    }
}

pub(crate) struct HealthPanel;

impl Panel for HealthPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Health
    }
    // Grows with the breakdown: the panel is as tall as the tags something is
    // reporting into.
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        health_panel::size(hook.health.snapshot())
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        health_panel::default_origin(vp)
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        health_panel::build(ui)
    }
    // The snapshot is refreshed on the hook's throttled sample, not here, and
    // shown only once per sample.
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        health_panel::sync(hook.health.snapshot(), hook.health.samples(), world, built);
    }
}

pub(crate) struct PreviewPanel;

impl Panel for PreviewPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Preview
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        preview::size()
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        preview::default_origin()
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        preview::build(ui)
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, _built: &BuiltPanel) {
        let state = preview::PreviewState {
            playing: hook.sim.playing(),
            fly: hook.fly,
            axes: hook.axes_visible,
            snap: hook.snap,
            align: hook.align_to_surface,
        };
        preview::sync(state, world);
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = preview::action(event) else {
            return false;
        };
        match action {
            PreviewAction::TogglePlay => hook.sim_toggle_play(),
            PreviewAction::ToggleFly => hook.toggle_fly(),
            PreviewAction::ToggleAxes => hook.axes_visible = !hook.axes_visible,
            PreviewAction::ToggleSnapMove => {
                hook.snap.translate.enabled = !hook.snap.translate.enabled;
            }
            PreviewAction::CycleSnapMoveStep => hook.snap.translate.cycle(&snap::TRANSLATE_STEPS),
            PreviewAction::ToggleSnapRotate => {
                hook.snap.rotate.enabled = !hook.snap.rotate.enabled;
            }
            PreviewAction::CycleSnapRotateStep => hook.snap.rotate.cycle(&snap::ROTATE_STEPS),
            PreviewAction::ToggleAlign => hook.align_to_surface = !hook.align_to_surface,
            PreviewAction::DropToFloor => {
                hook.drop_selection_to_floor(world);
            }
        }
        true
    }
}

pub(crate) struct ContentPanel;

impl Panel for ContentPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Content
    }
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.assets.stale = true;
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.content.search_focus = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        content_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        content_panel::default_origin(vp)
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        content_panel::build(ui)
    }
    fn restore(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        field::set_text(
            world,
            content_panel::search_field(built),
            &hook.content.filter,
        );
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_content(world, built);
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, built: &mut BuiltPanel) {
        hook.prepare_content(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        content_panel::sync(&hook.make_content_view(), world, built);
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = content_panel::action(event) else {
            return false;
        };
        hook.apply_content_action(action);
        true
    }
}

// Opened from the top bar's View button rather than from a row of its own.
pub(crate) struct ViewPanel;

impl Panel for ViewPanel {
    fn key(&self) -> PanelKey {
        PanelKey::View
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        view::size()
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        view::default_origin()
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        view::build(ui)
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, _built: &BuiltPanel) {
        let open = registry::VIEW_ROWS
            .iter()
            .map(|&(key, _)| registry::panel(key).is_open(hook));
        view::sync(open, world);
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(row) = view::toggled_row(event) else {
            return false;
        };
        hook.toggle_view_row(row, world);
        true
    }
}

pub(crate) struct TemplatesPanel;

impl Panel for TemplatesPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Templates
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        template::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        template::default_origin(vp[0])
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        template::build(ui)
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, _built: &BuiltPanel) {
        template::sync(hook.templates.detail, world);
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        let Some(i) = template::picked(event) else {
            return false;
        };
        hook.open_template_detail(i);
        true
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
        hook.open[PanelKey::Templates] && hook.templates.detail.is_some()
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.close_template_detail();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        let rows = hook
            .templates
            .detail
            .map_or(1, |i| EditorHook::template_rows(i).len());
        template_panel::size(rows)
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        template_panel::default_origin(vp[0])
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        template_panel::build(ui)
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, _built: &BuiltPanel) {
        if let Some(i) = hook.templates.detail {
            template_panel::sync(i, world);
        }
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        if !template_panel::applies(event) {
            return false;
        }
        hook.apply_open_template();
        true
    }
}

pub(crate) struct LightingPanel;

impl Panel for LightingPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Lighting
    }
    fn ids(&self) -> &'static HudIds {
        lighting_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    // Opening (re)seeds the text controls from the current entries and drops
    // any stale focus / status from the last session.
    fn on_open(&self, hook: &mut EditorHook, world: &mut World) {
        hook.lighting.focus = None;
        hook.lighting.status = None;
        hook.seed_lighting(world);
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        lighting_panel::size(lighting::rows(&hook.lighting_present()).len())
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        lighting_panel::default_origin()
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
        handled(action, |a| hook.apply_lighting_action(a, world))
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Lighting);
        let data = hook.lighting_data();
        let view = hook.make_lighting_view(&data, mouse);
        lighting_panel::place(world, &view, o, s);
    }
}

pub(crate) struct PlanetPanel;

impl Panel for PlanetPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Planet
    }
    fn ids(&self) -> &'static HudIds {
        planet::panel::ids()
    }
    fn on_open(&self, hook: &mut EditorHook, world: &mut World) {
        hook.planet.focus = None;
        hook.planet.status = None;
        hook.seed_planet(world);
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.planet_tool.put_down();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        planet::panel::size(hook.planet_rows().len(), hook.planet_strip())
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        planet::panel::default_origin(vp)
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Planet);
        let action = {
            let rows = hook.planet_rows();
            let view = hook.make_planet_view(&rows, "", [mx, my]);
            planet::panel::hit_test(&view, mx, my, o, s)
        };
        handled(action, |a| hook.apply_planet_action(a, world))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Planet);
        let rows = hook.planet_rows().len();
        let strip = hook.planet_strip();
        widget::point_in(mx, my, planet::panel::body_rect(o, s[0], strip, rows))
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_planet(delta);
    }
    fn frame_keys(&self, hook: &mut EditorHook, _world: &mut World, input: &FrameInput) {
        hook.planet_tool_keys(input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Planet);
        let rows = hook.planet_rows();
        let pick = hook.planet_pick_label();
        let view = hook.make_planet_view(&rows, &pick, mouse);
        planet::panel::place(world, &view, o, s);
    }
}

pub(crate) struct CharacterShapePanel;

impl Panel for CharacterShapePanel {
    fn key(&self) -> PanelKey {
        PanelKey::CharacterShape
    }
    fn ids(&self) -> &'static HudIds {
        character_shape_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.shape.status = None;
        hook.shape.scroll = 0;
    }
    // The panel shows up to `DEFAULT_ROWS` rows by default and can be dragged
    // taller to show every row; never taller than its content.
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        character_shape_panel::size(
            hook.shape
                .rows
                .clamp(1, character_shape_panel::DEFAULT_ROWS),
        )
    }
    fn max_size(&self, hook: &EditorHook) -> [f32; 2] {
        let rows = hook.shape.rows.clamp(1, character_shape_panel::MAX_ROWS);
        [f32::INFINITY, character_shape_panel::size(rows)[1]]
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        character_shape_panel::default_origin()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::CharacterShape);
        let data = hook.shape_data(world);
        let action = {
            let view = hook.make_shape_view(&data, [mx, my]);
            character_shape_panel::hit_test(&view, mx, my, o, s)
        };
        handled(action, |a| {
            hook.apply_shape_action(a, &data, [mx, my], world);
        })
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::CharacterShape);
        character_shape_panel::cursor_over_rows(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_shape(delta);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::CharacterShape);
        let data = hook.shape_data(world);
        let view = hook.make_shape_view(&data, mouse);
        character_shape_panel::place(world, &view, o, s);
    }
}

pub(crate) struct StoryPanel;

impl Panel for StoryPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Story
    }
    // Opening (re)loads the source file, so the panel always starts from the
    // on-disk truth.
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.load_story();
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        story_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        story_panel::default_origin(vp[0])
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        story_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_story(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        story_panel::sync(&hook.story_view(), world, built);
    }
    fn lend(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) -> bool {
        story_panel::swap_editor(&mut hook.story.area, world, built)
    }
    fn reclaim(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        story_panel::swap_editor(&mut hook.story.area, world, built);
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = story_panel::action(event) else {
            return false;
        };
        hook.apply_story_action(action);
        true
    }
}

pub(crate) struct ShadersPanel;

impl Panel for ShadersPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Shaders
    }
    // Opening builds the rows before the panel's first draw.
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.shader_rows();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        shader::list_panel::size(hook.shaders.rows.len())
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        shader::list_panel::default_origin(vp[0])
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        shader::list_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, _built: &BuiltPanel) {
        hook.observe_shaders(world);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, _built: &BuiltPanel) {
        shader::list_panel::sync(&hook.make_shaders_view(), world);
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = shader::list_panel::action(event) else {
            return false;
        };
        hook.apply_shaders_action(action, world);
        true
    }
}

pub(crate) struct ShaderSourcePanel;

impl Panel for ShaderSourcePanel {
    fn key(&self) -> PanelKey {
        PanelKey::ShaderSource
    }
    // Opened from the Shaders list, on one file.
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.shaders.source.is_some()
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.close_shader_source();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        shader::source_panel::size(hook.shaders.reference.open)
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        shader::source_panel::default_origin(vp[0])
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        shader::source_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_shader_source(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.sync_shader_source(world, built);
    }
    fn lend(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) -> bool {
        hook.shaders
            .source
            .as_mut()
            .is_some_and(|src| shader::source_panel::swap_editor(&mut src.area, world, built))
    }
    fn reclaim(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        if let Some(src) = hook.shaders.source.as_mut() {
            shader::source_panel::swap_editor(&mut src.area, world, built);
        }
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = shader::source_panel::action(event) else {
            return false;
        };
        hook.apply_source_action(action);
        true
    }
}

pub(crate) struct ConsolePanel;

impl Panel for ConsolePanel {
    fn key(&self) -> PanelKey {
        PanelKey::Console
    }
    // Opening focuses a cleared command line (backtick does the same through
    // the hook's key drive).
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.toggle_console(world);
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.console.focus = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        console_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        console_panel::default_origin(vp)
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.console_keys(world, input);
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        console_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_console(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        let view = hook.console_view(hook.console_ghost_now(world));
        let log = || hook.console_sink.read(console_panel::items);
        console_panel::sync(&view, log, world, built);
    }
}

pub(crate) struct BehaviorPanel;

impl Panel for BehaviorPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Behavior
    }
    fn ids(&self) -> &'static HudIds {
        behavior::panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    // The outline's row pool caps how tall it is worth growing the panel; a
    // chart has no such pool, so there it grows to the screen.
    fn max_size(&self, hook: &EditorHook) -> [f32; 2] {
        match hook.behavior.mode {
            ViewMode::Outline => behavior::panel::max_size(),
            _ => [f32::INFINITY, f32::INFINITY],
        }
    }
    // Opening re-reads the world's behaviors, so the panel always starts from
    // the current entry list rather than a stale selection.
    fn on_open(&self, hook: &mut EditorHook, world: &mut World) {
        hook.open_behavior(world);
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.behavior.blur_inputs();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        match hook.behavior.mode {
            ViewMode::Chart => behavior::panel::chart_size(),
            ViewMode::Overview => behavior::panel::overview_size(),
            ViewMode::Outline => behavior::panel::size(),
        }
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        behavior::panel::default_origin(vp[0])
    }
    fn overlay_ids(&self, hook: &EditorHook) -> Vec<AssetId> {
        let mut ids = behavior::panel::status_ids();
        if hook.behavior.picking {
            ids.extend(behavior::panel::palette_ids());
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
            behavior::panel::hit_test(&view, mx, my, o, s)
        };
        handled(action, |a| hook.apply_behavior_action(a, world, [mx, my]))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Behavior);
        behavior::panel::cursor_over_body(mx, my, o, s)
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
        behavior::panel::place(world, Some(&view), o, s);
    }
}

pub(crate) struct MapPanel;

impl Panel for MapPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Map
    }
    fn ids(&self) -> &'static HudIds {
        map::panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    // Opening is a fresh look at the map, so the canvas is put back on where
    // the world starts rather than on wherever it was last left.
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.map.reroot();
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.map.pan_drag = None;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        map::panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        map::panel::default_origin(vp[0])
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Map);
        let chart = hook.map_chart();
        let action = {
            let view = hook.make_map_view(&chart, [mx, my]);
            map::panel::hit_test(&view, mx, my, o, s)
        };
        handled(action, |a| hook.apply_map_action(a, [mx, my], world))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Map);
        map::panel::cursor_over_body(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_map(delta);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let chart = hook.map_chart();
        let view = hook.make_map_view(&chart, mouse);
        map::panel::place(world, Some(&view), o, hook.effective_size(PanelKey::Map));
    }
}

pub(crate) struct VariablesPanel;

impl Panel for VariablesPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Variables
    }
    // Opening re-reads the world, so the panel always starts from the current
    // table and the names its behaviors use.
    fn on_open(&self, hook: &mut EditorHook, world: &mut World) {
        hook.open_variables(world);
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.variables.name_focus = false;
        hook.variables.value_focus = false;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        variables_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        variables_panel::default_origin(vp[0])
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.variables_keys(world, input);
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        variables_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_variables(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        let data = hook.variables_data();
        let view = hook.make_variables_view(&data);
        variables_panel::sync(&view, world, built);
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = variables_panel::action(event) else {
            return false;
        };
        hook.apply_variables_action(action, world);
        true
    }
}

pub(crate) struct PalettePanel;

impl Panel for PalettePanel {
    fn key(&self) -> PanelKey {
        PanelKey::Palette
    }
    fn ids(&self) -> &'static HudIds {
        palette::panel::ids()
    }
    // Opening rebuilds the item list and clears the query, ready to type.
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        hook.toggle_palette(world);
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        palette::panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        palette::panel::default_origin(vp)
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
            let view = hook.make_palette_view([mx, my]);
            palette::panel::hit_test(&view, mx, my, o)
        };
        handled(action, |hit| hook.apply_palette_hit(hit, world))
    }
    fn wheel_over(
        &self,
        _hook: &EditorHook,
        _world: &World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        palette::panel::cursor_over_rows(mx, my, o)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_palette(delta);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.palette_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let view = hook.make_palette_view(mouse);
        palette::panel::place(world, Some(&view), o);
    }
}

pub(crate) struct ImportPanel;

impl Panel for ImportPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Import
    }
    // Opening clears stale state and focuses the path field, ready to type.
    fn on_open(&self, hook: &mut EditorHook, world: &mut World) {
        hook.import.status = None;
        hook.import.focus = true;
        EditorHook::clear_import_field(world);
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        import_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        import_panel::default_origin(vp[0])
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.import_keys(world, input);
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        import_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_import(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        import_panel::sync(
            &hook.make_import_view(),
            || hook.import_rows(),
            world,
            built,
        );
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = import_panel::action(event) else {
            return false;
        };
        hook.apply_import_action(action, world);
        true
    }
}

pub(crate) struct WorldsPanel;

impl Panel for WorldsPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Worlds
    }
    // Opening re-reads the project's worlds.
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.open_worlds_panel();
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.worlds.menu = None;
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        hook.worlds_layout().size()
    }
    // The start screen docks to the window's left edge; the registry hands the
    // default anchor no hook, so the switcher's is what it asks for and the
    // start screen re-derives its own (`EditorHook::origin`).
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        worlds::Layout::new(worlds::Mode::Session, vp, 0.0).default_origin()
    }
    fn is_built(&self) -> bool {
        true
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        worlds::panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, _built: &BuiltPanel) {
        hook.observe_worlds(world);
    }
    fn prepare(&self, hook: &mut EditorHook, _world: &mut World, _built: &mut BuiltPanel) {
        hook.prepare_worlds();
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        worlds::panel::sync(&hook.make_worlds_view(), world, built);
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = hook.worlds_event(event) else {
            return false;
        };
        hook.apply_worlds_action(action, world);
        true
    }
}
