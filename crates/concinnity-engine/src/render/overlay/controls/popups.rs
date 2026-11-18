// The popups' draw: an open Menu, an open Modal over its dimmed screen, and a
// Toasts stack. Also lays each dialog out from its measured message, so its
// buttons take clicks where they are drawn.

use alloc::vec::Vec;
use core::fmt::Write as _;

use crate::components::{Menu, Modal, TextAlign, TextLabel, Toasts, UiState};
use crate::ecs::PipelineContext;
use crate::gfx::overlay::UI_REFERENCE_SIZE;
use crate::presentation::ui::bound;
use crate::presentation::ui::widgets::modal::{GAP, PADDING};
use crate::protocol::ScreenSpaces;
use crate::render::overlay::text::{self, FontSet};
use crate::render::overlay::widgets::WidgetScratch;

use super::{ControlFrame, Pen, inset};

const BUTTON_TINT: [f32; 4] = [0.19, 0.20, 0.25, 1.0];
// Width of a toast's edge stripe.
const STRIPE_W: f32 = 4.0;
// Seconds a toast takes to fade out at the end of its time.
const FADE_S: f32 = 0.4;
// Inset of entry text from a menu's edges.
const ENTRY_PAD: f32 = 10.0;

pub(super) fn menu_open(menu: &Menu, ctx: &PipelineContext<'_>) -> bool {
    menu.open
        .as_ref()
        .and_then(|b| bound::bool_of(ctx, b))
        .unwrap_or(false)
}

pub(super) fn modal_open(modal: &Modal, ctx: &PipelineContext<'_>) -> bool {
    modal
        .open
        .as_ref()
        .and_then(|b| bound::bool_of(ctx, b))
        .unwrap_or(false)
}

pub(super) fn draw_menu(
    menu: &Menu,
    state: Option<&UiState>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
) {
    let style = &menu.style;
    let mut pen = Pen::new(out, frame, style, menu.screen, menu.fit);
    let count = menu.entry_count(state);
    pen.framed(
        menu.rect(count),
        style.background,
        style.corner_radius,
        style.border,
    );
    let items = state.map_or(&[][..], |s| s.list(&menu.items));
    for i in 0..count {
        let row = menu.row_rect(i);
        let (label, shortcut, disabled, separator, tint) = match menu.entries.get(i) {
            Some(e) if menu.items.is_empty() => (
                e.label.as_str(),
                e.shortcut.as_str(),
                e.disabled,
                e.separator,
                None,
            ),
            _ => items.get(i).map_or(("", "", true, false, None), |it| {
                (
                    it.text.as_str(),
                    it.detail.as_str(),
                    it.dimmed,
                    false,
                    it.color,
                )
            }),
        };
        if separator && i > 0 {
            pen.rect(
                [row[0] + ENTRY_PAD, row[1], row[2] - 2.0 * ENTRY_PAD, 1.0],
                style.border,
                0.0,
            );
        }
        if menu.hovered == Some(i) && !disabled {
            pen.rect(inset(row), style.hover, style.corner_radius);
        }
        let y = pen.text_y(row[1], row[3]);
        let color = match tint {
            _ if disabled => style.dim_text,
            Some(tint) => tint,
            None => style.text,
        };
        let right = row[0] + row[2] - ENTRY_PAD;
        pen.text(
            label,
            [row[0] + ENTRY_PAD, y],
            color,
            TextAlign::Left,
            row[2] - 2.0 * ENTRY_PAD,
        );
        if !shortcut.is_empty() {
            pen.text(shortcut, [right, y], style.dim_text, TextAlign::Right, 0.0);
        }
    }
}

// Lay every open dialog out for this frame from its measured message.
// `flags` is scratch: each dialog's open state and whether it is on the canvas,
// read before the dialogs are borrowed to lay out.
pub(in crate::render::overlay) fn lay_out_modals(
    ctx: &mut PipelineContext<'_>,
    fonts: &FontSet,
    viewport: [f32; 2],
    scratch: &mut WidgetScratch,
    flags: &mut Vec<(bool, bool)>,
) {
    flags.clear();
    {
        let no_spaces = ScreenSpaces::default();
        let spaces = ctx.resource::<ScreenSpaces>().unwrap_or(&no_spaces);
        for modal in ctx.query::<Modal>() {
            let canvas = spaces.on_canvas(modal.screen.map(crate::ecs::Ref::id));
            flags.push((modal_open(modal, ctx), canvas));
        }
    }
    for (modal, &(open, canvas)) in ctx.query_mut::<Modal>().zip(flags.iter()) {
        if !open {
            continue;
        }
        let line_h = fonts
            .resolve(modal.style.font)
            .map_or(16.0, |f| f.size_px * modal.style.scale);
        let area = if canvas { UI_REFERENCE_SIZE } else { viewport };
        let width = modal.width.min(area[0]) - 2.0 * PADDING;
        scratch.begin();
        let mut message = scratch.string("");
        match fonts.resolve(modal.style.font) {
            Some(font) => text::wrap_into(
                &modal.message,
                font,
                modal.style.scale,
                width.max(1.0),
                &mut message,
            ),
            None => message.push_str(&modal.message),
        }
        let label = message_label(modal, message);
        let message_h = text::measure_label_box(&label, fonts).map_or(line_h, |b| b.h);
        scratch.labels.push(label);
        modal.lay_out(area, line_h, message_h);
    }
}

// The dialog's message, already broken into lines, placed at the origin.
fn message_label(modal: &Modal, content: alloc::string::String) -> TextLabel {
    TextLabel {
        font: modal.style.font,
        content,
        scale: modal.style.scale,
        screen: modal.screen,
        fit: modal.fit,
        ..TextLabel::default()
    }
}

pub(super) fn draw_modal(modal: &Modal, frame: &ControlFrame<'_>, out: &mut WidgetScratch) {
    let style = &modal.style;
    let area = frame.area(modal.screen);
    let mut pen = Pen::new(out, frame, style, modal.screen, modal.fit);
    pen.rect([0.0, 0.0, area[0], area[1]], modal.dim, 0.0);
    let card = modal.drawn.card;
    pen.framed(card, style.background, style.corner_radius, style.border);
    let x = card[0] + PADDING;
    let inner = card[2] - 2.0 * PADDING;
    let mut y = card[1] + PADDING;
    pen.text(&modal.title, [x, y], style.text, TextAlign::Left, inner);
    y += pen.line_h() + GAP;
    if !modal.message.is_empty() {
        pen.wrapped(&modal.message, [x, y], style.dim_text, inner);
    }
    for (i, (button, rect)) in modal.buttons.iter().zip(&modal.drawn.buttons).enumerate() {
        let fill = if modal.hovered == Some(i) {
            style.hover
        } else if button.primary {
            style.accent
        } else {
            BUTTON_TINT
        };
        pen.rect(*rect, fill, style.corner_radius);
        let center = [rect[0] + rect[2] * 0.5, pen.text_y(rect[1], rect[3])];
        pen.text(&button.label, center, style.text, TextAlign::Center, 0.0);
    }
}

pub(super) fn draw_toasts(toasts: &Toasts, frame: &ControlFrame<'_>, out: &mut WidgetScratch) {
    let style = &toasts.style;
    let area = frame.area(toasts.screen);
    let mut pen = Pen::new(out, frame, style, toasts.screen, toasts.fit);
    let visible = toasts.visible_count();
    let newest = toasts.shown.len().saturating_sub(1);
    for slot in 0..visible {
        let toast = &toasts.shown[newest - slot];
        let fade = (toast.left / FADE_S).clamp(0.0, 1.0);
        let alpha = |c: [f32; 4]| [c[0], c[1], c[2], c[3] * fade];
        let card = toasts.card_rect(slot, area);
        pen.framed(
            card,
            alpha(style.background),
            style.corner_radius,
            alpha(style.border),
        );
        let stripe = [card[0], card[1], STRIPE_W, card[3]];
        pen.rect(
            stripe,
            alpha(toast.tone.accent()),
            style.corner_radius.min(STRIPE_W),
        );
        let x = card[0] + STRIPE_W + ENTRY_PAD;
        let y = pen.text_y(card[1], card[3]);
        let width = card[2] - STRIPE_W - 2.0 * ENTRY_PAD;
        pen.text(&toast.text, [x, y], style.text, TextAlign::Left, width);
    }
    let waiting = toasts.shown.len() - visible;
    if waiting > 0 {
        let row = toasts.card_rect(visible, area);
        let y = pen.text_y(row[1], row[3]);
        let at = [row[0] + ENTRY_PAD, y];
        let label = pen.text("+", at, style.dim_text, TextAlign::Left, 0.0);
        #[expect(
            clippy::let_underscore_must_use,
            reason = "writing to a String cannot fail"
        )]
        let _ = write!(label.content, "{waiting} more");
    }
}
