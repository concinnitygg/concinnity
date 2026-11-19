//! The `Panel` registry implementations: one stateless unit per floating panel,
//! binding its module (its build and sync) to the hook state that backs it. The
//! shared machinery -- focus, close buttons, building, draw layers, open state --
//! lives on the registry and its consumers; each impl supplies only what is
//! panel-specific.

use concinnity_engine::components::{FrameInput, UiEvent};
use concinnity_engine::ecs::World;

use super::EditorHook;
use crate::editor::behavior::panel::ViewMode;
use crate::editor::panels::form_panel::{self, FormAction};
use crate::editor::panels::preview::{self, PreviewAction};
use crate::editor::panels::registry::{self, Panel, PanelKey};
use crate::editor::panels::template;
use crate::editor::panels::view;
use crate::editor::panels::{
    assets_panel, character_shape_panel, console_panel, content_panel, health_panel, import_panel,
    lighting_panel, planet, shader, story_panel, template_panel, variables_panel,
};
use crate::editor::ui::{BuiltPanel, UiBuilder, field};
use crate::editor::viewport::snap;
use crate::editor::{behavior, map, palette, worlds};

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
    // Opening (re)seeds the texts from the current entries and drops any
    // stale focus / status from the last session.
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.lighting.focus = None;
        hook.lighting.status = None;
        hook.seed_lighting();
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        lighting_panel::default_size()
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        lighting_panel::default_origin()
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        lighting_panel::build(ui)
    }
    // A world the panel was just built into holds none of its rows.
    fn restore(&self, hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) {
        hook.lighting.shown = None;
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, _built: &BuiltPanel) {
        hook.observe_lighting(world);
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, built: &mut BuiltPanel) {
        hook.prepare_lighting(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, _built: &BuiltPanel) {
        if let (Some(view), Some(ids)) = (hook.make_lighting_view(), &hook.lighting.shown) {
            lighting_panel::sync(&view, ids, world);
        }
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = lighting_panel::action(event) else {
            return false;
        };
        hook.apply_lighting_action(action);
        true
    }
}

pub(crate) struct PlanetPanel;

impl Panel for PlanetPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Planet
    }
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.planet.focus = None;
        hook.planet.status = None;
        hook.seed_planet();
    }
    fn on_close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.planet_tool.put_down();
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        planet::panel::size(planet::panel::VISIBLE_ROWS)
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        planet::panel::default_origin(vp)
    }
    fn frame_keys(&self, hook: &mut EditorHook, _world: &mut World, input: &FrameInput) {
        hook.planet_tool_keys(input);
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        planet::panel::build(ui)
    }
    // A world the panel was just built into holds none of its rows.
    fn restore(&self, hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) {
        hook.planet.shown = None;
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, _built: &BuiltPanel) {
        hook.observe_planet(world);
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, built: &mut BuiltPanel) {
        hook.prepare_planet(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        if let (Some(view), Some(ids)) = (hook.make_planet_view(), &hook.planet.shown) {
            planet::panel::sync(&view, ids, built, world);
        }
    }
    fn on_event(&self, hook: &mut EditorHook, _world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = planet::panel::action(event) else {
            return false;
        };
        hook.apply_planet_action(action);
        true
    }
}

pub(crate) struct CharacterShapePanel;

impl Panel for CharacterShapePanel {
    fn key(&self) -> PanelKey {
        PanelKey::CharacterShape
    }
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.shape.status = None;
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        character_shape_panel::size(character_shape_panel::DEFAULT_ROWS)
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        character_shape_panel::default_origin()
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        character_shape_panel::build(ui)
    }
    // A world the panel was just built into holds none of its rows.
    fn restore(&self, hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) {
        hook.shape.shown = None;
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, built: &mut BuiltPanel) {
        hook.prepare_shape(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        if let (Some(view), Some(ids)) = (hook.make_shape_view(), &hook.shape.shown) {
            character_shape_panel::sync(&view, ids, built, world);
        }
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(action) = character_shape_panel::action(event) else {
            return false;
        };
        hook.apply_shape_action(action, world);
        true
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
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.behavior_keys(world, input);
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        behavior::panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_behavior(world, built);
    }
    fn restore(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.restore_behavior(world, built);
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, _built: &mut BuiltPanel) {
        hook.prepare_behavior(world);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        if let Some(view) = hook.make_behavior_view() {
            behavior::panel::sync(&view, world, built);
        }
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        hook.on_behavior_event(world, event)
    }
}

pub(crate) struct MapPanel;

impl Panel for MapPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Map
    }
    // Opening is a fresh look at the map, so the canvas is put back on where
    // the world starts rather than on wherever it was last left.
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.map.reroot();
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        map::panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        map::panel::default_origin(vp[0])
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        map::panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_map(world, built);
    }
    fn prepare(&self, hook: &mut EditorHook, world: &mut World, _built: &mut BuiltPanel) {
        hook.prepare_map(world);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        map::panel::sync(world, built, hook.map.pan);
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        hook.on_map_event(world, event)
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
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        variables_panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_variables(world, built);
    }
    fn prepare(&self, hook: &mut EditorHook, _world: &mut World, _built: &mut BuiltPanel) {
        hook.prepare_variables();
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        if let Some(view) = hook.make_variables_view() {
            variables_panel::sync(&view, world, built);
        }
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
    // Opening rebuilds the item list and clears the query, ready to type.
    fn toggle(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.toggle_palette();
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        palette::panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        palette::panel::default_origin(vp)
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.palette_keys(world, input);
    }
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
        palette::panel::build(ui)
    }
    fn observe(&self, hook: &mut EditorHook, world: &mut World, built: &BuiltPanel) {
        hook.observe_palette(world, built);
    }
    fn sync(&self, hook: &EditorHook, world: &mut World, built: &BuiltPanel) {
        palette::panel::sync(&hook.make_palette_view(), world, built);
    }
    fn on_event(&self, hook: &mut EditorHook, world: &mut World, event: &UiEvent) -> bool {
        let Some(row) = palette::panel::pick(event) else {
            return false;
        };
        hook.pick_palette_row(row, world);
        true
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
