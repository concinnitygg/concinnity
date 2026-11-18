// The popups' draw: an open Menu, an open Modal over its dimmed screen, and a
// Toasts stack. Also lays each dialog out from its measured message, so its
// buttons take clicks where they are drawn.

use alloc::vec::Vec;
use core::fmt::Write as _;

use crate::components::{
    Menu, Modal, TextAlign, TextLabel, ToastProgress, Toasts, UiState, WidgetStyle,
};
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
// How far each level of a menu item's depth indents it.
const DEPTH_INDENT: f32 = 14.0;
// A menu entry's mark, and the gap after it and before a detail.
const MARK: f32 = 12.0;
const MARK_GAP: f32 = 8.0;
// The most of a menu row a detail takes.
const DETAIL_SHARE: f32 = 0.6;
// Inset of a toast's message and bar from its card's top and bottom.
const TOAST_PAD: f32 = 8.0;
// Height of a toast's progress bar.
const BAR_H: f32 = 6.0;
// An ongoing bar's segment, as a share of the track, and the seconds it takes
// to glide there and back.
const SWEEP_SHARE: f32 = 0.3;
const SWEEP_S: f32 = 1.2;

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

// One menu entry as drawn: from `entries`, or an item of the list `items`.
struct MenuRow<'a> {
    label: &'a str,
    detail: &'a str,
    disabled: bool,
    separator: bool,
    tint: Option<[f32; 3]>,
    depth: u16,
    // Whether the entry shows a mark, whether it is a dot, and whether it is
    // on.
    mark: Option<(bool, bool)>,
}

pub(super) fn draw_menu(
    menu: &Menu,
    ctx: &PipelineContext<'_>,
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
    let marks = menu.items.is_empty() && menu.entries.iter().any(|e| e.mark.is_some());
    for i in 0..count {
        let entry = match menu.entries.get(i) {
            Some(e) if menu.items.is_empty() => MenuRow {
                label: &e.label,
                detail: &e.shortcut,
                disabled: e.disabled,
                separator: e.separator,
                tint: None,
                depth: 0,
                mark: e.mark.as_ref().map(|m| {
                    let (binding, radio) = m.parts();
                    (radio, bound::bool_of(ctx, binding).unwrap_or(false))
                }),
            },
            _ => items.get(i).map_or(
                MenuRow {
                    label: "",
                    detail: "",
                    disabled: true,
                    separator: false,
                    tint: None,
                    depth: 0,
                    mark: None,
                },
                |it| MenuRow {
                    label: &it.text,
                    detail: &it.detail,
                    disabled: it.dimmed,
                    separator: false,
                    tint: it.color,
                    depth: it.depth,
                    mark: None,
                },
            ),
        };
        draw_menu_row(&mut pen, menu, i, &entry, marks);
    }
}

fn draw_menu_row(pen: &mut Pen<'_>, menu: &Menu, i: usize, entry: &MenuRow<'_>, marks: bool) {
    let style = &menu.style;
    let row = menu.row_rect(i);
    if entry.separator && i > 0 {
        pen.rect(
            [row[0] + ENTRY_PAD, row[1], row[2] - 2.0 * ENTRY_PAD, 1.0],
            style.border,
            0.0,
        );
    }
    if menu.hovered == Some(i) && !entry.disabled {
        pen.rect(inset(row), style.hover, style.corner_radius);
    }
    let y = pen.text_y(row[1], row[3]);
    let mut x = row[0] + ENTRY_PAD + f32::from(entry.depth) * DEPTH_INDENT;
    if marks {
        if let Some((radio, on)) = entry.mark {
            draw_mark(pen, style, [x, row[1] + (row[3] - MARK) * 0.5], radio, on);
        }
        x += MARK + MARK_GAP;
    }
    let color = match entry.tint {
        _ if entry.disabled => style.dim_text,
        Some(tint) => tint,
        None => style.text,
    };
    let right = row[0] + row[2] - ENTRY_PAD;
    let mut room = right - x;
    if !entry.detail.is_empty() {
        let detail = pen.advance(entry.detail).min(room * DETAIL_SHARE);
        pen.text(
            entry.detail,
            [right, y],
            style.dim_text,
            TextAlign::Right,
            detail,
        );
        room -= detail + MARK_GAP;
    }
    pen.text(entry.label, [x, y], color, TextAlign::Left, room.max(0.0));
}

// An entry's mark at `at`: a box filled while on, or a ring with a dot.
fn draw_mark(pen: &mut Pen<'_>, style: &WidgetStyle, at: [f32; 2], radio: bool, on: bool) {
    let rect = [at[0], at[1], MARK, MARK];
    let radius = if radio {
        MARK * 0.5
    } else {
        style.corner_radius.min(MARK * 0.3)
    };
    let [r, g, b] = style.dim_text;
    pen.ring(rect, [r, g, b, 1.0], radius, 1.5);
    if on {
        let m = MARK * 0.25;
        let dot = [at[0] + m, at[1] + m, MARK - 2.0 * m, MARK - 2.0 * m];
        let r = if radio { dot[2] * 0.5 } else { 1.5 };
        pen.rect(dot, style.accent, r);
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
        pen.wrapped(&modal.message, [x, y], style.dim_text, inner, 0);
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
    let line_h = pen.line_h();
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
        let width = card[2] - STRIPE_W - 2.0 * ENTRY_PAD;
        let bar = if toast.progress.is_some() {
            BAR_H + TOAST_PAD
        } else {
            0.0
        };
        let room = card[3] - 2.0 * TOAST_PAD - bar;
        let lines = concinnity_core::math::floor(room / line_h).max(1.0);
        let y = card[1] + TOAST_PAD + (room - lines * line_h).max(0.0) * 0.5;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a whole, positive count of lines"
        )]
        let lines = lines as usize;
        pen.wrapped(&toast.text, [x, y], style.text, width, lines);
        if let Some(progress) = toast.progress {
            let track = [x, card[1] + card[3] - TOAST_PAD - BAR_H, width, BAR_H];
            pen.rect(track, alpha(BUTTON_TINT), BAR_H * 0.5);
            let fill = progress_fill(track, progress, toast.age);
            pen.rect(fill, alpha(style.accent), BAR_H * 0.5);
        }
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

// The filled part of a progress bar's `track`: the share done, or a short
// segment gliding along it while the work cannot be counted, placed by `age`.
fn progress_fill(track: [f32; 4], progress: ToastProgress, age: f32) -> [f32; 4] {
    match progress {
        ToastProgress::Fraction(f) => [track[0], track[1], track[2] * f.clamp(0.0, 1.0), track[3]],
        ToastProgress::Ongoing => {
            let w = track[2] * SWEEP_SHARE;
            let t = concinnity_core::math::fract(age / SWEEP_S);
            let pos = if t < 0.5 { 2.0 * t } else { 2.0 - 2.0 * t };
            [track[0] + pos * (track[2] - w), track[1], w, track[3]]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bar_fills_its_share_or_glides_along_its_track() {
        let track = [10.0, 0.0, 100.0, 6.0];
        assert_eq!(
            progress_fill(track, ToastProgress::Fraction(0.25), 0.0),
            [10.0, 0.0, 25.0, 6.0]
        );
        for age in [0.0, 0.3, 0.6, 0.9, 5.0] {
            let fill = progress_fill(track, ToastProgress::Ongoing, age);
            assert!(
                fill[0] >= 10.0 && fill[0] + fill[2] <= 110.0 + 1e-3,
                "{fill:?}"
            );
        }
        let start = progress_fill(track, ToastProgress::Ongoing, 0.0);
        let half = progress_fill(track, ToastProgress::Ongoing, SWEEP_S * 0.5);
        assert_eq!(start[0], 10.0);
        assert!(
            (half[0] + half[2] - 110.0).abs() < 1e-3,
            "the far end at half"
        );
    }
}
