// The widgets' draw: each visible widget on an open screen is synthesized into
// plain Sprites and TextLabels and fed through the same shapers as the authored
// overlay, clipped to its own rect and lifted to its screen's layer (a popup to
// the popup layer over it).
//   mod.rs       the per-frame pass and the pen the widgets draw with
//   list.rs      List, Grid
//   canvas.rs    Canvas
//   simple.rs    Tabs, Checkbox, Button, ValueSlider
//   text_area.rs TextArea, and the font grid it is measured to
//   popups.rs    Menu, Modal, Toasts

mod canvas;
mod list;
mod popups;
mod simple;
mod text_area;

use alloc::string::String;
use core::fmt::Write as _;

use crate::components::{
    Button, Canvas, Checkbox, Grid, Icon, Identity, Layout, List, Menu, Modal, Screen, Sprite,
    SpriteFit, Tabs, TextAlign, TextArea, TextLabel, Toasts, UiImage, ValueSlider, WidgetStyle,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext, Ref, TextureHandle};
use crate::gfx::overlay::UI_REFERENCE_SIZE;
use crate::presentation::ui::panel_stack::{PanelStack, intersect};
use crate::presentation::ui::screen::Popup;
use crate::protocol::{ScreenSpaces, ScreenStack};
use crate::render::overlay::call_buffer::TextCallBuffer;
use crate::render::overlay::maps::{OverlayLayers, Placement};
use crate::render::overlay::widgets::WidgetScratch;
use crate::render::overlay::{OverlayAssets, sprite, text};

// The dim a modal root draws over the rest of its screen.
const MODAL_DIM: [f32; 4] = [0.02, 0.02, 0.03, 0.55];

pub(super) use popups::lay_out_modals;
pub(super) use text_area::measure_text_areas;

// What every widget's draw reads: the frame's faces, the open screens and
// their layers and spaces, and the frame's size.
pub(super) struct ControlFrame<'a> {
    pub(super) assets: &'a OverlayAssets,
    pub(super) stack: Option<&'a ScreenStack>,
    pub(super) panels: &'a PanelStack,
    // Each element's draw layer, where a screen or an override sets one.
    pub(super) layers: &'a OverlayLayers,
    pub(super) spaces: &'a ScreenSpaces,
    pub(super) viewport: [f32; 2],
    pub(super) default_atlas_slot: Option<usize>,
    pub(super) caret_visible: bool,
}

impl ControlFrame<'_> {
    // Whether widget `entity` of `screen` draws: its screen is open (one on
    // no screen always is) and no hidden layout holds it.
    fn shown(&self, entity: Entity, screen: Option<Ref<Screen>>) -> bool {
        !self.panels.hides(entity)
            && screen.is_none_or(|s| self.stack.is_none_or(|st| st.open.contains(&s.id())))
    }

    // The placement of widget `entity` of `screen`: on the layer an override
    // gives the widget, else its screen's (or that layer's `popup`), and its
    // panel's rank, clipped to `clip`.
    fn placement(
        &self,
        (entity, widget): (Entity, Option<AssetId>),
        screen: Option<Ref<Screen>>,
        popup: Option<Popup>,
        clip: Option<[f32; 4]>,
    ) -> Placement {
        let id = screen.map(Ref::id);
        let base = widget
            .and_then(|w| self.layers.get(&w))
            .or_else(|| id.and_then(|id| self.stack.and_then(|st| st.layers.get(&id))))
            .copied()
            .unwrap_or(0);
        // A widget inside a scrolling layout draws only within it.
        let clip = match (clip, self.panels.clip_of(entity)) {
            (Some(own), Some(outer)) => Some(intersect(own, outer)),
            (own, outer) => own.or(outer),
        };
        Placement {
            clip,
            layer: popup.map_or(base, |p| p.over(base)),
            rank: self.panels.rank_of(entity),
            canvas: self.spaces.on_canvas(id),
        }
    }

    // The size of the space a widget of `screen` lays out in.
    fn area(&self, screen: Option<Ref<Screen>>) -> [f32; 2] {
        if self.spaces.on_canvas(screen.map(Ref::id)) {
            UI_REFERENCE_SIZE
        } else {
            self.viewport
        }
    }

    // Feed the elements built into `out` through the shapers at `placement`.
    fn flush(&self, out: &WidgetScratch, placement: Placement, buffer: &mut TextCallBuffer) {
        sprite::build_sprite_calls_into(
            buffer,
            out.sprites.iter().map(|s| (s, placement)),
            self.default_atlas_slot,
            &self.assets.sprite_texture_slots,
            self.viewport,
        );
        text::build_text_calls_into(
            buffer,
            out.labels.iter().map(|l| (l, placement)),
            &self.assets.fonts,
            self.viewport,
        );
    }
}

// A widget's entity and asset id.
fn named(ctx: &PipelineContext<'_>, entity: Entity) -> (Entity, Option<AssetId>) {
    (entity, ctx.get::<Identity>(entity).map(|i| i.id()))
}

// Draw every visible widget on an open screen into `buffer`.
pub(super) fn build_controls(
    ctx: &PipelineContext<'_>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
    buffer: &mut TextCallBuffer,
) {
    let state = ctx.resource::<crate::components::UiState>();
    for (entity, list) in ctx.query_with_entity::<List>() {
        if list.visible && frame.shown(entity, list.screen) {
            list::draw(list, ctx, state, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), list.screen, None, Some(list.rect())),
                buffer,
            );
        }
    }
    for (entity, grid) in ctx.query_with_entity::<Grid>() {
        if grid.visible && frame.shown(entity, grid.screen) {
            list::draw_grid(grid, ctx, state, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), grid.screen, None, Some(grid.rect())),
                buffer,
            );
        }
    }
    for (entity, c) in ctx.query_with_entity::<Canvas>() {
        if c.visible && frame.shown(entity, c.screen) {
            canvas::draw(c, state, frame, out);
            let clip = c.clip.then(|| c.rect());
            frame.flush(
                out,
                frame.placement(named(ctx, entity), c.screen, None, clip),
                buffer,
            );
        }
    }
    for (entity, layout) in ctx.query_with_entity::<Layout>() {
        if layout.scroll && layout.visible && frame.shown(entity, layout.screen) {
            let Some(rect) = layout.rect else {
                continue;
            };
            list::draw_layout_scrollbar(layout, rect, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), layout.screen, None, None),
                buffer,
            );
        }
    }
    // A modal root's dim, over the rest of its screen and under the root.
    for (entity, rank) in frame.panels.modals() {
        let Some(layout) = ctx.get::<Layout>(entity) else {
            continue;
        };
        let style = WidgetStyle::default();
        let mut pen = Pen::new(out, frame, &style, layout.screen, SpriteFit::default());
        let area = frame.area(layout.screen);
        pen.rect([0.0, 0.0, area[0], area[1]], MODAL_DIM, 0.0);
        let placement = Placement {
            rank,
            ..frame.placement(named(ctx, entity), layout.screen, None, None)
        };
        frame.flush(out, placement, buffer);
    }
    for (entity, tabs) in ctx.query_with_entity::<Tabs>() {
        if tabs.visible && frame.shown(entity, tabs.screen) {
            simple::draw_tabs(tabs, ctx, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), tabs.screen, None, None),
                buffer,
            );
        }
    }
    for (entity, check) in ctx.query_with_entity::<Checkbox>() {
        if check.visible && frame.shown(entity, check.screen) {
            simple::draw_checkbox(check, ctx, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), check.screen, None, None),
                buffer,
            );
        }
    }
    for (entity, button) in ctx.query_with_entity::<Button>() {
        if button.visible && frame.shown(entity, button.screen) {
            simple::draw_button(button, ctx, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), button.screen, None, None),
                buffer,
            );
        }
    }
    for (entity, slider) in ctx.query_with_entity::<ValueSlider>() {
        if slider.visible && frame.shown(entity, slider.screen) {
            simple::draw_value_slider(slider, ctx, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), slider.screen, None, None),
                buffer,
            );
        }
    }
    for (entity, area) in ctx.query_with_entity::<TextArea>() {
        if area.visible && area.read_at.is_some() && frame.shown(entity, area.screen) {
            let rect = [area.x, area.y, area.width, area.height];
            text_area::draw(area, frame, out);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), area.screen, None, Some(rect)),
                buffer,
            );
        }
    }
    for (entity, menu) in ctx.query_with_entity::<Menu>() {
        if frame.shown(entity, menu.screen) && popups::menu_open(menu, ctx) {
            popups::draw_menu(menu, ctx, state, frame, out);
            let popup = Some(Popup::Menu);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), menu.screen, popup, None),
                buffer,
            );
        }
    }
    for (entity, modal) in ctx.query_with_entity::<Modal>() {
        if frame.shown(entity, modal.screen) && popups::modal_open(modal, ctx) {
            popups::draw_modal(modal, frame, out);
            let popup = Some(Popup::Modal);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), modal.screen, popup, None),
                buffer,
            );
        }
    }
    for (entity, toasts) in ctx.query_with_entity::<Toasts>() {
        if toasts.visible && frame.shown(entity, toasts.screen) && !toasts.shown.is_empty() {
            popups::draw_toasts(toasts, frame, out);
            let popup = Some(Popup::Toast);
            frame.flush(
                out,
                frame.placement(named(ctx, entity), toasts.screen, popup, None),
                buffer,
            );
        }
    }
}

// The sprites and labels one widget draws with, carrying its screen, fit, and
// style.
struct Pen<'a> {
    out: &'a mut WidgetScratch,
    fonts: &'a text::FontSet,
    style: &'a WidgetStyle,
    screen: Option<Ref<Screen>>,
    fit: SpriteFit,
}

impl<'a> Pen<'a> {
    fn new(
        out: &'a mut WidgetScratch,
        frame: &'a ControlFrame<'_>,
        style: &'a WidgetStyle,
        screen: Option<Ref<Screen>>,
        fit: SpriteFit,
    ) -> Self {
        out.begin();
        Self {
            out,
            fonts: &frame.assets.fonts,
            style,
            screen,
            fit,
        }
    }

    // Height of a line of the style's text.
    fn line_h(&self) -> f32 {
        self.fonts
            .resolve(self.style.font)
            .map_or(16.0, |f| f.size_px * self.style.scale)
    }

    // The top of a line of text centered in a row from `y` of height `h`.
    fn text_y(&self, y: f32, h: f32) -> f32 {
        y + (h - self.line_h()) * 0.5
    }

    // A filled rectangle with rounded corners.
    fn rect(&mut self, rect: [f32; 4], tint: [f32; 4], radius: f32) {
        self.framed(rect, tint, radius, [0.0; 4]);
    }

    // A filled rectangle framed by a hairline of `border`.
    fn framed(&mut self, rect: [f32; 4], tint: [f32; 4], radius: f32, border: [f32; 4]) {
        let width = if border[3] > 0.0 { 1.0 } else { 0.0 };
        self.sprite(rect, tint, radius, (border, width), None);
    }

    // An outline of `width` in `color` around `rect`, with no fill.
    fn ring(&mut self, rect: [f32; 4], color: [f32; 4], radius: f32, width: f32) {
        self.sprite(rect, [0.0; 4], radius, (color, width), None);
    }

    // `icon` in `color`, scaled into the square `cell`, `[x, y, side]`.
    fn icon(&mut self, icon: &Icon, cell: [f32; 3], color: [f32; 4]) {
        for part in &icon.parts {
            let (rect, radius) = part.placed(cell);
            if part.ring > 0.0 {
                self.ring(rect, color, radius, part.ring * cell[2]);
            } else {
                self.rect(rect, color, radius);
            }
        }
    }

    // `image` fitted inside `area`.
    fn image(&mut self, image: &UiImage, area: [f32; 4]) {
        let rect = image.fitted(area);
        self.sprite(rect, [1.0; 4], 0.0, ([0.0; 4], 0.0), Some(image.texture));
    }

    // One sprite at `rect`: its fill, rounding, `(color, width)` outline, and
    // texture.
    fn sprite(
        &mut self,
        rect: [f32; 4],
        tint: [f32; 4],
        radius: f32,
        (border_color, border_width): ([f32; 4], f32),
        texture: Option<TextureHandle>,
    ) {
        self.out.sprites.push(Sprite {
            x: rect[0],
            y: rect[1],
            width: rect[2],
            height: rect[3],
            texture,
            tint,
            follow_cursor: false,
            visible: true,
            screen: self.screen,
            fit: self.fit,
            corner_radius: radius,
            border_width,
            border_color,
        });
    }

    // A line of text at `(x, y)`, cut with an ellipsis past `width` when it is
    // positive.
    fn text(
        &mut self,
        content: &str,
        at: [f32; 2],
        color: [f32; 3],
        align: TextAlign,
        width: f32,
    ) -> &mut TextLabel {
        let mut drawn = self.out.string("");
        let scale = self.style.scale;
        match self.fonts.resolve(self.style.font) {
            Some(font) if width > 0.0 && text::text_advance_width(content, font, scale) > width => {
                text::ellipsize_into(content, font, scale, width, &mut drawn);
            }
            _ => drawn.push_str(content),
        }
        self.push_label(drawn, at, color, align)
    }

    // How wide `content` draws on one line.
    fn advance(&self, content: &str) -> f32 {
        self.fonts.resolve(self.style.font).map_or(0.0, |font| {
            text::text_advance_width(content, font, self.style.scale)
        })
    }

    // Text from `(x, y)` broken at spaces onto lines that fit `width`, at most
    // `max_lines` of them when it is above zero.
    fn wrapped(
        &mut self,
        content: &str,
        at: [f32; 2],
        color: [f32; 3],
        width: f32,
        max_lines: usize,
    ) {
        let mut drawn = self.out.string("");
        match self.fonts.resolve(self.style.font) {
            Some(font) => text::wrap_into(content, font, self.style.scale, width, &mut drawn),
            None => drawn.push_str(content),
        }
        if max_lines > 0
            && let Some((cut, _)) = drawn.match_indices('\n').nth(max_lines - 1)
        {
            drawn.truncate(cut);
        }
        self.push_label(drawn, at, color, TextAlign::Left);
    }

    // A line showing `value` with `decimals` digits after the point.
    fn decimal(
        &mut self,
        value: f32,
        decimals: u8,
        at: [f32; 2],
        color: [f32; 3],
        align: TextAlign,
    ) {
        let mut content = self.out.string("");
        let precision = usize::from(decimals);
        #[expect(
            clippy::let_underscore_must_use,
            reason = "writing to a String cannot fail"
        )]
        let _ = write!(content, "{value:.precision$}");
        self.push_label(content, at, color, align);
    }

    // A line showing `value` as a number.
    fn number(&mut self, value: usize, at: [f32; 2], color: [f32; 3], align: TextAlign) {
        let mut content = self.out.string("");
        #[expect(
            clippy::let_underscore_must_use,
            reason = "writing to a String cannot fail"
        )]
        let _ = write!(content, "{value}");
        self.push_label(content, at, color, align);
    }

    fn push_label(
        &mut self,
        content: String,
        at: [f32; 2],
        color: [f32; 3],
        align: TextAlign,
    ) -> &mut TextLabel {
        let color_runs = self.out.runs();
        self.out.labels.push(TextLabel {
            font: self.style.font,
            content,
            x: at[0],
            y: at[1],
            color,
            color_runs,
            scale: self.style.scale,
            centered: false,
            align,
            fit: self.fit,
            background: [0.0; 4],
            padding: 0.0,
            wrap_width: 0.0,
            max_lines: 0,
            text_binding: None,
            visible: true,
            screen: self.screen,
        });
        let last = self.out.labels.len() - 1;
        &mut self.out.labels[last]
    }
}

// A highlight inset from a row's rect, so it reads as a pill.
fn inset(rect: [f32; 4]) -> [f32; 4] {
    const X: f32 = 4.0;
    const Y: f32 = 1.5;
    [
        rect[0] + X,
        rect[1] + Y,
        (rect[2] - 2.0 * X).max(0.0),
        (rect[3] - 2.0 * Y).max(0.0),
    ]
}

// A color dimmed toward transparent, for a disabled control.
fn faded(c: [f32; 4]) -> [f32; 4] {
    [c[0], c[1], c[2], c[3] * 0.45]
}

#[cfg(test)]
mod tests;
