// Transient overlay element synthesis: the floating dropdown list and the
// text-input field box/caret. Both are built as plain Sprites + TextLabels and
// fed through the same shapers as the authored overlay elements.

use crate::assets::{Sprite, TextInput, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::gfx::text;

// Build the transient overlay Sprites + TextLabels for an open dropdown list.
// These are fed through the same sprite / text shapers as the menu but with no
// clip bands, so the list draws unclipped on top of the menu (escaping the
// scroll band's scissor). Geometry is reference space for a screen-owned row (and
// window pixels otherwise), matching the input hit-test in `ui`.
pub(super) fn build_dropdown_overlay(
    screen: &crate::ecs::DropdownView,
    loaded_fonts: &std::collections::HashMap<crate::ecs::FontHandle, text::LoadedFont>,
) -> (Vec<Sprite>, Vec<TextLabel>) {
    // Panel fill (near-opaque so rows behind it do not show through), a framing
    // border, and the selected / hovered row highlights.
    const PANEL_BG: [f32; 4] = [0.06, 0.08, 0.14, 0.98];
    const BORDER: [f32; 4] = [0.28, 0.34, 0.52, 1.0];
    const SELECTED_BG: [f32; 4] = [0.14, 0.20, 0.34, 1.0];
    const HOVER_BG: [f32; 4] = [0.22, 0.30, 0.48, 1.0];
    const TRACK: [f32; 4] = [0.12, 0.15, 0.25, 0.95];
    const THUMB: [f32; 4] = [0.45, 0.52, 0.70, 0.9];
    const TEXT_PAD: f32 = 10.0;
    const BORDER_PX: f32 = 2.0;

    use concinnity_core::gfx::dropdown;
    let count = screen.options.len();
    let layout = dropdown::layout(screen.anchor, count);
    // The layout windows a long list; rows show options `first..`, so the
    // selected / hovered option indices map to row indices (off-window ones
    // simply draw no highlight).
    let first = screen.first.min(dropdown::max_first(count));
    let row_of = |option: usize| {
        option
            .checked_sub(first)
            .filter(|r| *r < layout.items.len())
    };
    let mut sprites: Vec<Sprite> = Vec::new();
    let mk_sprite = |rect: [f32; 4], tint: [f32; 4]| Sprite {
        asset_id: AssetId::default(),
        x: rect[0],
        y: rect[1],
        width: rect[2],
        height: rect[3],
        texture: None,
        tint,
        follow_cursor: false,
        visible: true,
        screen: screen.screen,
        fit: crate::assets::SpriteFit::Fit,
        corner_radius: 0.0,
    };

    // Border quad (a little larger, drawn first) then the panel fill on top.
    let [lx, ly, lw, lh] = layout.list;
    sprites.push(mk_sprite(
        [
            lx - BORDER_PX,
            ly - BORDER_PX,
            lw + 2.0 * BORDER_PX,
            lh + 2.0 * BORDER_PX,
        ],
        BORDER,
    ));
    sprites.push(mk_sprite(layout.list, PANEL_BG));
    // The currently-applied option, then the hovered one on top of it (each
    // only when its option is inside the shown window).
    if let Some(rect) = row_of(screen.selected).and_then(|r| layout.items.get(r)) {
        sprites.push(mk_sprite(*rect, SELECTED_BG));
    }
    if let Some(rect) = screen
        .hovered
        .and_then(row_of)
        .and_then(|r| layout.items.get(r))
    {
        sprites.push(mk_sprite(*rect, HOVER_BG));
    }
    // A scrolled list gets a scrollbar inside its right edge: a faint
    // full-height track with the draggable thumb over it.
    if let Some(rect) = dropdown::track_rect(&layout, count) {
        sprites.push(mk_sprite(rect, TRACK));
    }
    if let Some(rect) = dropdown::thumb_rect(&layout, first, count) {
        sprites.push(mk_sprite(rect, THUMB));
    }

    // One text label per SHOWN option, vertically centered in its row (the text
    // draws after the sprites, so it sits over the highlights).
    let line_h = screen
        .font
        .and_then(|f| loaded_fonts.get(&f))
        .map(|f| f.size_px * screen.scale)
        .unwrap_or(0.0);
    let labels: Vec<TextLabel> = screen
        .options
        .iter()
        .skip(first)
        .zip(&layout.items)
        .map(|(opt, rect)| TextLabel {
            asset_id: AssetId::default(),
            font: screen.font,
            content: opt.clone(),
            x: rect[0] + TEXT_PAD,
            y: rect[1] + (rect[3] - line_h) / 2.0,
            color: screen.color,
            scale: screen.scale,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: screen.screen,
        })
        .collect();

    (sprites, labels)
}

// The visible slice of a single-line field's text, fit to its box width. `avail`
// is the drawable text width; `advance` measures the rendered width of a prefix
// with the real font metrics. Returns the substring to draw, the x offset to add
// to the field's left text edge (always >= 0, so nothing bleeds left), and the
// caret's x offset from that same edge. A field that fits is returned untouched;
// one that overflows is truncated from the head with an ellipsis while unfocused,
// or horizontally scrolled to keep the caret in screen while focused.
fn fit_line(
    content: &str,
    caret_byte: usize,
    avail: f32,
    focused: bool,
    advance: impl Fn(&str) -> f32,
) -> (String, f32, f32) {
    const ELLIPSIS: &str = "...";
    let caret_byte = caret_byte.min(content.len());
    let full = advance(content);
    if avail <= 0.0 || full <= avail {
        return (content.to_string(), 0.0, advance(&content[..caret_byte]));
    }
    if focused {
        // Pin the caret to the right edge once the text runs past the box, easing
        // back to the left as the caret returns.
        let caret_w = advance(&content[..caret_byte]);
        let scroll = (caret_w - avail).max(0.0);
        // Byte boundaries: the start, then the end of each char.
        let mut bounds = vec![0usize];
        bounds.extend(content.char_indices().map(|(b, c)| b + c.len_utf8()));
        // Drop chars fully scrolled off the left so nothing bleeds past that edge.
        let start = *bounds
            .iter()
            .find(|&&b| advance(&content[..b]) >= scroll)
            .unwrap_or(&0);
        // Keep chars up to the last boundary still inside the box.
        let end = bounds
            .iter()
            .rev()
            .find(|&&b| advance(&content[..b]) - scroll <= avail)
            .copied()
            .unwrap_or(content.len())
            .max(start);
        let visible = content.get(start..end).unwrap_or("").to_string();
        (
            visible,
            advance(&content[..start]) - scroll,
            caret_w - scroll,
        )
    } else {
        // Truncate from the head, leaving room for an ellipsis.
        let ell = advance(ELLIPSIS);
        let mut end = 0usize;
        for (b, c) in content.char_indices() {
            let nb = b + c.len_utf8();
            if advance(&content[..nb]) + ell > avail {
                break;
            }
            end = nb;
        }
        (format!("{}{ELLIPSIS}", &content[..end]), 0.0, 0.0)
    }
}

// Synthesise the transient Sprites + TextLabels that draw a TextInput field: a
// background box, the typed content (or the dimmer placeholder while empty and
// unfocused), and a caret bar while focused. Fed through the same shapers as the
// authored overlay elements, carrying the field's `screen` / `fit` so screen mapping
// and visibility apply. Mirrors `build_dropdown_overlay`. The text is fit to the
// box (`fit_line`) so a long value never bleeds past the field's edges.
pub(super) fn build_text_input_overlay(
    ti: &TextInput,
    loaded_fonts: &std::collections::HashMap<crate::ecs::FontHandle, text::LoadedFont>,
    caret_visible: bool,
) -> (Vec<Sprite>, Vec<TextLabel>) {
    const CARET_W: f32 = 2.0;
    let font = ti.font.and_then(|f| loaded_fonts.get(&f));
    let line_h = font
        .map(|f| f.size_px * ti.scale)
        .unwrap_or(ti.height * 0.6);
    // Text baseline math centres the cap band in `[y, y + line_h]`, so placing
    // the line box's top here vertically centres the text in the field.
    let text_y = ti.y + (ti.height - line_h) / 2.0;

    let bg = Sprite {
        asset_id: AssetId::default(),
        x: ti.x,
        y: ti.y,
        width: ti.width,
        height: ti.height,
        texture: None,
        tint: ti.background,
        follow_cursor: false,
        visible: true,
        screen: ti.screen,
        fit: ti.fit,
        corner_radius: ti.corner_radius,
    };
    let mut sprites = vec![bg];

    // Placeholder only while empty and unfocused; otherwise the live content.
    let showing_placeholder = ti.content.is_empty() && !ti.focused;
    let (raw, color) = if showing_placeholder {
        (ti.placeholder.as_str(), ti.placeholder_color)
    } else {
        (ti.content.as_str(), ti.text_color)
    };
    // The byte offset of the caret within the live content (only consulted for the
    // focused, content-showing case; harmless otherwise).
    let caret_byte = {
        let caret = ti.caret.min(ti.content.chars().count());
        ti.content
            .char_indices()
            .nth(caret)
            .map(|(b, _)| b)
            .unwrap_or(ti.content.len())
    };
    // Fit the text to the box. Without a loaded font we cannot measure, so pass it
    // through (it will not be rendered until a font loads).
    let avail = (ti.width - 2.0 * ti.padding - CARET_W).max(0.0);
    let (content, x_offset, caret_off) = match font {
        Some(f) => fit_line(raw, caret_byte, avail, ti.focused, |s| {
            text::text_advance_width(s, f, ti.scale)
        }),
        None => (raw.to_string(), 0.0, 0.0),
    };

    let label = TextLabel {
        asset_id: AssetId::default(),
        font: ti.font,
        content,
        x: ti.x + ti.padding + x_offset,
        y: text_y,
        color,
        scale: ti.scale,
        centered: false,
        align: crate::assets::TextAlign::Left,
        fit: ti.fit,
        background: [0.0, 0.0, 0.0, 0.0],
        padding: 0.0,
        visible: true,
        screen: ti.screen,
    };

    // Caret: a thin bar at the caret's fit position, only while the field holds
    // focus and the font loaded, and only on the visible half of the blink cycle.
    if ti.focused && font.is_some() && caret_visible {
        let caret_x = ti.x + ti.padding + caret_off;
        sprites.push(Sprite {
            asset_id: AssetId::default(),
            x: caret_x,
            y: text_y,
            width: CARET_W,
            height: line_h,
            texture: None,
            tint: [ti.caret_color[0], ti.caret_color[1], ti.caret_color[2], 1.0],
            follow_cursor: false,
            visible: true,
            screen: ti.screen,
            fit: ti.fit,
            corner_radius: 0.0,
        });
    }

    (sprites, vec![label])
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed-width mock metric (every char 10px) makes `fit_line` widths exact.
    fn mock_advance(s: &str) -> f32 {
        s.chars().count() as f32 * 10.0
    }

    // Text that fits the box is returned untouched, with the caret at its measured
    // position.
    #[test]
    fn fit_line_passes_through_text_that_fits() {
        let (text, xoff, caret) = fit_line("abc", 3, 100.0, false, mock_advance);
        assert_eq!(text, "abc");
        assert_eq!(xoff, 0.0);
        assert_eq!(caret, 30.0);
    }

    // An unfocused overflow is truncated from the head with an ellipsis that fits
    // inside the box.
    #[test]
    fn fit_line_truncates_unfocused_overflow_with_ellipsis() {
        // avail 65, ellipsis "..." = 30px: keep chars while width + 30 <= 65 (3 chars).
        let (text, xoff, _) = fit_line("abcdefghij", 0, 65.0, false, mock_advance);
        assert_eq!(text, "abc...");
        assert_eq!(xoff, 0.0);
    }

    // A focused overflow scrolls so the caret (at the end here) stays at the box's
    // right edge, dropping the head that ran off the left.
    #[test]
    fn fit_line_scrolls_focused_overflow_to_keep_the_caret_visible() {
        // 100px of text, 50px box, caret at end: scroll 50 -> show the last 5 chars.
        let (text, xoff, caret) = fit_line("abcdefghij", 10, 50.0, true, mock_advance);
        assert_eq!(text, "fghij");
        assert_eq!(xoff, 0.0);
        assert_eq!(caret, 50.0, "caret pinned to the right edge");
        assert!(caret <= 50.0, "caret never past the box");
    }

    // With the caret at the start, a focused overflow shows the head (no scroll).
    #[test]
    fn fit_line_focused_caret_at_start_shows_the_head() {
        let (text, xoff, caret) = fit_line("abcdefghij", 0, 50.0, true, mock_advance);
        assert_eq!(text, "abcde");
        assert_eq!(xoff, 0.0);
        assert_eq!(caret, 0.0);
    }
}
