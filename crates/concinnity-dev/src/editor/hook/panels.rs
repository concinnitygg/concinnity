//! The `Panel` registry implementations: one stateless unit per floating panel,
//! binding its module (geometry + draw) to the hook state that backs it. The
//! shared machinery -- dragging, focus, close buttons, injection, draw layers,
//! open state, the hidden pass -- lives on the registry and its consumers; each
//! impl supplies only what is panel-specific.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::EditorHook;
use crate::editor::behavior;
use crate::editor::behavior::panel::ViewMode;
use crate::editor::hud_ids::HudIds;
use crate::editor::map;
use crate::editor::palette;
use crate::editor::panels::assets_panel;
use crate::editor::panels::character_shape_panel;
use crate::editor::panels::console_panel;
use crate::editor::panels::content_panel;
use crate::editor::panels::form_panel::{self, FormAction};
use crate::editor::panels::health_panel;
use crate::editor::panels::import_panel;
use crate::editor::panels::lighting;
use crate::editor::panels::lighting_panel;
use crate::editor::panels::preview::{self, PreviewAction};
use crate::editor::panels::registry::{self, Panel, PanelKey};
use crate::editor::panels::shader_list_panel;
use crate::editor::panels::shader_source_panel;
use crate::editor::panels::story_panel;
use crate::editor::panels::template::{self, TemplatesAction};
use crate::editor::panels::template_panel;
use crate::editor::panels::variables_panel;
use crate::editor::panels::view::{self, ViewAction};
use crate::editor::text_area::layout::Metrics;
use crate::editor::viewport::snap;
use crate::editor::widget;
use crate::editor::worlds;

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
    fn ids(&self) -> &'static HudIds {
        assets_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        assets_panel::max_size()
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
        hook.assets.scroll = 0;
        hook.assets.search_focus = true;
        widget::seed_field(world, assets_panel::SEARCH_INPUT, "");
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
            let selected = hook.selected_names();
            let view = hook.make_view(&data, &selected, [mx, my]);
            assets_panel::hit_test(&view, mx, my, o, s)
        };
        handled(action, |a| hook.apply_panel(a, world))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Assets);
        assets_panel::cursor_over_body(mx, my, o, s)
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
        let selected = hook.selected_names();
        let view = hook.make_view(&data, &selected, mouse);
        assets_panel::place(world, &view, o, s);
    }
}

pub(crate) struct EditPanel;

impl Panel for EditPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Edit
    }
    fn ids(&self) -> &'static HudIds {
        form_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
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
    // The field list tracks the type's args; the height resizes only when there
    // are more fields than the default window shows.
    fn max_size(&self, hook: &EditorHook) -> [f32; 2] {
        form_panel::max_size(hook.form_row_count())
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        form_panel::default_origin(vp[0])
    }
    fn overlay_ids(&self, hook: &EditorHook) -> Vec<AssetId> {
        if hook.form.field_dropdown.is_some() {
            return form_panel::dropdown_ids();
        }
        if hook.form.override_menu.is_some() || hook.form.entity_menu_open {
            return form_panel::override_menu_ids();
        }
        Vec::new()
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
        handled(action, |a| hook.apply_form(a, world))
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
        form_panel::place(world, &view, o, s);
    }
}

pub(crate) struct HealthPanel;

impl Panel for HealthPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Health
    }
    fn ids(&self) -> &'static HudIds {
        health_panel::ids()
    }
    // Grows with the breakdown: the panel is as tall as the tags something is
    // reporting into.
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        health_panel::size(hook.health.snapshot())
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        health_panel::default_origin(vp)
    }
    // Read-only: a body press is swallowed so it cannot reach the world.
    fn press(
        &self,
        hook: &mut EditorHook,
        _world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        health_panel::hit_test(mx, my, o, hook.health.snapshot())
    }
    // The snapshot is refreshed on the hook's throttled sample, not here: `draw`
    // only has `&EditorHook`, and the syscalls behind it must not run per frame.
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        health_panel::place(world, hook.health.snapshot(), o, mouse);
    }
}

pub(crate) struct PreviewPanel;

impl Panel for PreviewPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Preview
    }
    fn ids(&self) -> &'static HudIds {
        preview::ids()
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        preview::size()
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        preview::default_origin()
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        handled(preview::hit_test(mx, my, o), |a| match a {
            PreviewAction::TogglePlay => hook.sim_toggle_play(),
            PreviewAction::ToggleFly => hook.toggle_fly(),
            PreviewAction::ToggleAxes => hook.axes_visible = !hook.axes_visible,
            PreviewAction::ToggleSnapMove => {
                hook.snap.translate.enabled = !hook.snap.translate.enabled;
            }
            PreviewAction::CycleSnapMoveStep => {
                hook.snap.translate.cycle(&snap::TRANSLATE_STEPS);
            }
            PreviewAction::ToggleSnapRotate => {
                hook.snap.rotate.enabled = !hook.snap.rotate.enabled;
            }
            PreviewAction::CycleSnapRotateStep => {
                hook.snap.rotate.cycle(&snap::ROTATE_STEPS);
            }
            PreviewAction::ToggleAlign => hook.align_to_surface = !hook.align_to_surface,
            PreviewAction::DropToFloor => {
                hook.drop_selection_to_floor(world);
            }
            PreviewAction::Consume => {}
        })
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        preview::place(
            world,
            o,
            preview::PreviewState {
                playing: hook.sim.playing(),
                fly: hook.fly,
                axes: hook.axes_visible,
                snap: hook.snap,
                align: hook.align_to_surface,
            },
            mouse,
        );
    }
}

pub(crate) struct ContentPanel;

impl Panel for ContentPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Content
    }
    fn ids(&self) -> &'static HudIds {
        content_panel::ids()
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
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let shown = hook.content_cells(world).0.len();
        let action = content_panel::hit_test(mx, my, o, shown);
        handled(action, |a| hook.apply_content_action(a, world, [mx, my]))
    }
    fn wheel_over(
        &self,
        _hook: &EditorHook,
        _world: &World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        content_panel::cursor_over_body(mx, my, o)
    }
    fn scroll(&self, hook: &mut EditorHook, world: &mut World, delta: f32) {
        hook.scroll_content(delta, world);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let (cells, total) = hook.content_cells(world);
        content_panel::place(
            world,
            &content_panel::ContentView {
                cells: &cells,
                total,
                type_caption: hook.content_type_caption(),
                search_focus: hook.content.search_focus,
                mouse,
            },
            o,
        );
    }
}

// Opened from the top bar's View button rather than from a row of its own.
pub(crate) struct ViewPanel;

impl Panel for ViewPanel {
    fn key(&self) -> PanelKey {
        PanelKey::View
    }
    fn ids(&self) -> &'static HudIds {
        view::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        view::size()
    }
    fn default_origin(&self, _vp: [f32; 2]) -> [f32; 2] {
        view::default_origin()
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
        handled(view::hit_test(mx, my, o, s), |a| match a {
            ViewAction::Toggle(i) => hook.toggle_view_row(i, world),
            ViewAction::Consume => {}
        })
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::View);
        view::place(world, o, s, &hook.view_rows(), mouse);
    }
}

pub(crate) struct TemplatesPanel;

impl Panel for TemplatesPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Templates
    }
    fn ids(&self) -> &'static HudIds {
        template::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        template::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        template::default_origin(vp[0])
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        _world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Templates);
        handled(template::hit_test(mx, my, o, s), |a| match a {
            TemplatesAction::Pick(i) => hook.open_template_detail(i),
            TemplatesAction::Consume => {}
        })
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Templates);
        template::place(world, o, s, hook.templates.detail, mouse);
    }
}

pub(crate) struct TemplateDetailPanel;

impl TemplateDetailPanel {
    // The open template's grouped-row count (1 when none is open, matching the
    // panel's minimum footprint).
    fn row_count(&self, hook: &EditorHook) -> usize {
        hook.templates
            .detail
            .map_or(1, |i| hook.template_rows(i).len())
    }
}

impl Panel for TemplateDetailPanel {
    fn key(&self) -> PanelKey {
        PanelKey::TemplateDetail
    }
    fn ids(&self) -> &'static HudIds {
        template_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
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
    fn press(
        &self,
        hook: &mut EditorHook,
        _world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        if hook.templates.detail.is_none() {
            return false;
        }
        let s = hook.effective_size(PanelKey::TemplateDetail);
        let action = template_panel::hit_test(mx, my, o, s);
        handled(action, |a| hook.apply_template_detail(a))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        if hook.templates.detail.is_none() {
            return false;
        }
        let s = hook.effective_size(PanelKey::TemplateDetail);
        template_panel::cursor_over(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_template_list(delta);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let Some(i) = hook.templates.detail else {
            return;
        };
        let s = hook.effective_size(PanelKey::TemplateDetail);
        let data = hook.template_detail_data(i);
        let view = hook.make_template_view(&data, mouse);
        template_panel::place(world, &view, o, s);
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
            hook.apply_shape_action(a, &data, [mx, my], world)
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
    fn ids(&self) -> &'static HudIds {
        story_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        story_panel::max_size()
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
    fn press(
        &self,
        hook: &mut EditorHook,
        _world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Story);
        let action = {
            let view = hook.make_story_view([mx, my]);
            story_panel::hit_test(&view, mx, my, o, s)
        };
        handled(action, |a| hook.apply_story_action(a, mx, my))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Story);
        story_panel::cursor_over_area(mx, my, o, s)
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
        story_panel::place(world, Some(&view), o, s, Metrics::code());
    }
}

pub(crate) struct ShadersPanel;

impl ShadersPanel {
    // The rows as last built; the frame drive rebuilds them first while the
    // panel shows.
    fn row_count(&self, hook: &EditorHook) -> usize {
        hook.shaders.rows.len()
    }
}

impl Panel for ShadersPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Shaders
    }
    fn ids(&self) -> &'static HudIds {
        shader_list_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, hook: &EditorHook) -> [f32; 2] {
        shader_list_panel::max_size(self.row_count(hook))
    }
    // Opening builds the rows before the panel's first draw.
    fn on_open(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.shader_rows();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        shader_list_panel::size(self.row_count(hook))
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        shader_list_panel::default_origin(vp[0])
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Shaders);
        let rows = hook.shader_rows().to_vec();
        let action = {
            let view = hook.make_shaders_view(&rows, [mx, my]);
            shader_list_panel::hit_test(&view, mx, my, o, s)
        };
        // A press anywhere else closes the row menu without being taken.
        if action.is_none() {
            hook.shaders.menu = None;
        }
        handled(action, |a| hook.apply_shaders_action(a, &rows, world))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::Shaders);
        shader_list_panel::cursor_over(mx, my, o, s)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_shaders(delta);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let s = hook.effective_size(PanelKey::Shaders);
        let view = hook.make_shaders_view(&hook.shaders.rows, mouse);
        shader_list_panel::place(world, Some(&view), o, s);
    }
}

pub(crate) struct ShaderSourcePanel;

impl Panel for ShaderSourcePanel {
    fn key(&self) -> PanelKey {
        PanelKey::ShaderSource
    }
    fn ids(&self) -> &'static HudIds {
        shader_source_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        shader_source_panel::max_size()
    }
    // Opened from the Shaders list, on one file.
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.shaders.source.is_some()
    }
    fn close(&self, hook: &mut EditorHook, _world: &mut World) {
        hook.close_shader_source();
    }
    fn size(&self, hook: &EditorHook) -> [f32; 2] {
        shader_source_panel::size(hook.shaders.reference.open)
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        shader_source_panel::default_origin(vp[0])
    }
    fn press(
        &self,
        hook: &mut EditorHook,
        _world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::ShaderSource);
        let action = shader_source_panel::hit_test(mx, my, o, s, hook.shaders.reference.open);
        handled(action, |a| hook.apply_source_action(a, mx, my))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let s = hook.effective_size(PanelKey::ShaderSource);
        let reference = hook.shaders.reference.open;
        shader_source_panel::cursor_over_area(mx, my, o, s, reference)
            || (reference && shader_source_panel::cursor_over_reference(mx, my, o, s))
    }
    fn scroll_at(&self, hook: &mut EditorHook, _world: &mut World, delta: f32, mx: f32, my: f32) {
        hook.scroll_shader_source(delta, mx, my);
    }
    fn frame_keys(&self, hook: &mut EditorHook, world: &mut World, input: &FrameInput) {
        hook.shader_source_keys(world, input);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        hook.draw_shader_source(world, o, mouse);
    }
}

pub(crate) struct ConsolePanel;

impl Panel for ConsolePanel {
    fn key(&self) -> PanelKey {
        PanelKey::Console
    }
    fn ids(&self) -> &'static HudIds {
        console_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        console_panel::max_size()
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
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let s = hook.effective_size(PanelKey::Console);
        let action = console_panel::hit_test(mx, my, o, s);
        handled(action, |a| hook.apply_console_action(a, world))
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
        console_panel::place(world, Some(&view), o, s);
    }
    // The command line's autocomplete ghost goes with it.
    fn hide(&self, world: &mut World) {
        console_panel::hide_all(world);
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
    fn ids(&self) -> &'static HudIds {
        variables_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        variables_panel::max_size()
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
        handled(action, |a| hook.apply_variables_action(a, world))
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
        variables_panel::place(world, Some(&view), o, s);
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
    fn ids(&self) -> &'static HudIds {
        import_panel::ids()
    }
    fn resizable(&self) -> bool {
        true
    }
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        import_panel::max_size()
    }
    // Opening clears stale state and focuses the path field, ready to type.
    fn on_open(&self, hook: &mut EditorHook, world: &mut World) {
        hook.import.status = None;
        hook.import.scroll = 0;
        hook.import.focus = true;
        widget::seed_field(world, import_panel::PATH_INPUT, "");
    }
    fn size(&self, _hook: &EditorHook) -> [f32; 2] {
        import_panel::size()
    }
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        import_panel::default_origin(vp[0])
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
        handled(action, |a| hook.apply_import_action(a, world))
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
        import_panel::place(world, Some(&view), o, s);
    }
}

pub(crate) struct WorldsPanel;

impl Panel for WorldsPanel {
    fn key(&self) -> PanelKey {
        PanelKey::Worlds
    }
    fn ids(&self) -> &'static HudIds {
        worlds::ids()
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
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool {
        let action = worlds::hit_test(&hook.make_worlds_view([mx, my]), mx, my, o);
        handled(action, |a| hook.apply_worlds_action(a, world))
    }
    fn wheel_over(&self, hook: &EditorHook, _world: &World, mx: f32, my: f32, o: [f32; 2]) -> bool {
        hook.worlds_layout().cursor_over_list(mx, my, o)
    }
    fn scroll(&self, hook: &mut EditorHook, _world: &mut World, delta: f32) {
        hook.scroll_worlds(delta);
    }
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]) {
        let view = hook.make_worlds_view(mouse);
        worlds::place(world, Some(&view), o);
    }
}
