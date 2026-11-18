// The one-row widgets' draw: a Tabs strip, a Checkbox, and a Button.

use crate::components::{Button, Checkbox, Tabs, TextAlign};
use crate::ecs::PipelineContext;
use crate::presentation::ui::bound;
use crate::render::overlay::widgets::WidgetScratch;

use super::{ControlFrame, Pen, faded};

// Gap between a checkbox's box and its caption.
const CAPTION_GAP: f32 = 8.0;

pub(super) fn draw_tabs(
    tabs: &Tabs,
    ctx: &PipelineContext<'_>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
) {
    let style = &tabs.style;
    let mut pen = Pen::new(out, frame, style, tabs.screen, tabs.fit);
    let strip = [tabs.x, tabs.y, tabs.width, tabs.height];
    pen.framed(strip, style.background, style.corner_radius, style.border);
    let selected = tabs
        .selected
        .as_ref()
        .and_then(|b| bound::int_of(ctx, b))
        .map_or(0, |i| usize::try_from(i).unwrap_or(usize::MAX));
    for (i, title) in tabs.tabs.iter().enumerate() {
        let rect = tabs.tab_rect(i);
        let fill = if i == selected {
            Some(style.selected)
        } else if tabs.hovered == Some(i) {
            Some(style.hover)
        } else {
            None
        };
        if let Some(fill) = fill {
            pen.rect(super::inset(rect), fill, style.corner_radius);
        }
        let color = if i == selected {
            style.text
        } else {
            style.dim_text
        };
        let center = [rect[0] + rect[2] * 0.5, pen.text_y(rect[1], rect[3])];
        pen.text(title, center, color, TextAlign::Center, 0.0);
    }
}

pub(super) fn draw_checkbox(
    check: &Checkbox,
    ctx: &PipelineContext<'_>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
) {
    let style = &check.style;
    let mut pen = Pen::new(out, frame, style, check.screen, check.fit);
    if check.hovered && !check.disabled {
        pen.rect(check.rect(), style.hover, style.corner_radius);
    }
    let on = check
        .checked
        .as_ref()
        .and_then(|b| bound::bool_of(ctx, b))
        .unwrap_or(false);
    let off_fill = [0.30, 0.30, 0.34, 1.0];
    let fill = if on { style.accent } else { off_fill };
    let fill = if check.disabled { faded(fill) } else { fill };
    let b = check.box_rect();
    pen.rect(b, fill, style.corner_radius.min(b[2] * 0.3));
    if on {
        // The mark: a smaller square in the text color inside the box.
        let m = b[2] * 0.3;
        let [r, g, bl] = style.text;
        pen.rect(
            [b[0] + m, b[1] + m, b[2] - 2.0 * m, b[3] - 2.0 * m],
            [r, g, bl, 1.0],
            1.5,
        );
    }
    let x = b[0] + b[2] + CAPTION_GAP;
    let width = check.x + check.width - x;
    let caption = if check.disabled {
        style.dim_text
    } else {
        style.text
    };
    if check.note.is_empty() {
        let y = pen.text_y(check.y, check.height);
        pen.text(&check.label, [x, y], caption, TextAlign::Left, width);
    } else {
        let half = check.height * 0.5;
        let y = pen.text_y(check.y, half);
        pen.text(&check.label, [x, y], caption, TextAlign::Left, width);
        let y = pen.text_y(check.y + half, half);
        pen.text(&check.note, [x, y], style.dim_text, TextAlign::Left, width);
    }
}

pub(super) fn draw_button(
    button: &Button,
    ctx: &PipelineContext<'_>,
    frame: &ControlFrame<'_>,
    out: &mut WidgetScratch,
) {
    let style = &button.style;
    let mut pen = Pen::new(out, frame, style, button.screen, button.fit);
    let active = button
        .active
        .as_ref()
        .and_then(|b| bound::bool_of(ctx, b))
        .unwrap_or(false);
    let fill = if button.disabled {
        faded(style.background)
    } else if active {
        style.selected
    } else if button.hovered {
        style.hover
    } else {
        style.background
    };
    let rect = button.rect();
    if fill[3] > 0.0 {
        pen.framed(rect, fill, style.corner_radius, style.border);
    }
    let color = if button.disabled {
        style.dim_text
    } else {
        style.text
    };
    let center = [rect[0] + rect[2] * 0.5, pen.text_y(rect[1], rect[3])];
    pen.text(&button.label, center, color, TextAlign::Center, rect[2]);
}
