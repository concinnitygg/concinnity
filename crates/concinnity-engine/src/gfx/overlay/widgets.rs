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
        border_width: 0.0,
        border_color: [0.0, 0.0, 0.0, 1.0],
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
        border_width: 0.0,
        border_color: [0.0, 0.0, 0.0, 1.0],
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
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 1.0],
        });
    }

    (sprites, vec![label])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{DropdownView, FontHandle};

    // A fixed-width mock metric (every char 10px) makes `fit_line` widths exact.
    fn mock_advance(s: &str) -> f32 {
        s.chars().count() as f32 * 10.0
    }

    const FONT: FontHandle = FontHandle(0);

    fn make_glyph(advance_px: f32) -> crate::build::font::GlyphMetrics {
        crate::build::font::GlyphMetrics {
            char_code: 0,
            atlas_x: 0,
            atlas_y: 0,
            atlas_w: 8,
            atlas_h: 12,
            advance_px,
            bearing_x: 0.0,
            bearing_y: 12.0,
        }
    }

    // A fixed-width synthetic font (every glyph 10px in a 16px em) makes the
    // built geometry exact.
    fn loaded_fonts() -> std::collections::HashMap<FontHandle, text::LoadedFont> {
        let metrics: std::collections::HashMap<u32, crate::build::font::GlyphMetrics> = ('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
            .chain(['.', ' '])
            .map(|c| (c as u32, make_glyph(10.0)))
            .collect();
        let cap_px = text::derive_cap_px(&metrics, 16.0);
        std::collections::HashMap::from([(
            FONT,
            text::LoadedFont {
                atlas_slot: 0,
                cap_px,
                metrics,
                atlas_w: 128,
                atlas_h: 128,
                size_px: 16.0,
                supersample: 1.0,
            },
        )])
    }

    fn no_fonts() -> std::collections::HashMap<FontHandle, text::LoadedFont> {
        std::collections::HashMap::new()
    }

    // A list anchored to a 200x40 control at (400, 100), which has room to open
    // downward from y = 140.
    fn dropdown_view(options: &[&str]) -> DropdownView {
        DropdownView {
            anchor: [400.0, 100.0, 200.0, 40.0],
            options: options.iter().map(|s| s.to_string()).collect(),
            selected: 0,
            first: 0,
            hovered: None,
            screen: Some(AssetId(5)),
            font: Some(FONT),
            scale: 1.0,
            color: [1.0, 1.0, 1.0],
        }
    }

    fn text_input() -> TextInput {
        TextInput {
            font: Some(FONT),
            placeholder: "type here".to_string(),
            ..Default::default()
        }
    }

    fn rect(s: &Sprite) -> [f32; 4] {
        [s.x, s.y, s.width, s.height]
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

    // The list stacks back to front: the border quad, the panel fill over it, then
    // the selected row's highlight and the hovered row's on top. The option text
    // follows the sprites, so it draws over the highlights.
    #[test]
    fn build_dropdown_overlay_layers_border_panel_then_highlights() {
        let mut view = dropdown_view(&["aa", "bb", "cc"]);
        view.selected = 1;
        view.hovered = Some(2);
        let (sprites, labels) = build_dropdown_overlay(&view, &loaded_fonts());
        assert_eq!(sprites.len(), 4);
        // The border sits 2px outside the list rect on every side.
        assert_eq!(rect(&sprites[0]), [398.0, 138.0, 204.0, 124.0]);
        assert_eq!(rect(&sprites[1]), [400.0, 140.0, 200.0, 120.0]);
        // Highlights land on the selected row then the hovered one.
        assert_eq!(rect(&sprites[2]), [400.0, 180.0, 200.0, 40.0]);
        assert_eq!(rect(&sprites[3]), [400.0, 220.0, 200.0, 40.0]);
        assert_ne!(sprites[2].tint, sprites[3].tint);
        // Every sprite carries the view's screen, so it maps like the menu it drops from.
        assert!(sprites.iter().all(|s| s.screen == view.screen && s.visible));
        // One label per shown option, inset by the text pad and centred on a 16px line.
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].content, "aa");
        assert_eq!((labels[0].x, labels[0].y), (410.0, 152.0));
        assert_eq!((labels[2].x, labels[2].y), (410.0, 232.0));
        assert!(labels.iter().all(|l| l.font == view.font && l.visible));
    }

    // A list longer than the layout window shows options from `first` onward:
    // selected / hovered options outside that window draw no highlight, and the
    // scrolled list gains a scrollbar track with its thumb over it.
    #[test]
    fn build_dropdown_overlay_windows_a_scrolled_list() {
        let options: Vec<String> = (0..16).map(|i| format!("o{i}")).collect();
        let refs: Vec<&str> = options.iter().map(|s| s.as_str()).collect();
        let mut view = dropdown_view(&refs);
        view.first = 8;
        view.selected = 0;
        view.hovered = Some(3);
        let (sprites, labels) = build_dropdown_overlay(&view, &loaded_fonts());
        // Border + panel + track + thumb: both highlighted options are off the window.
        assert_eq!(sprites.len(), 4);
        assert_eq!(labels.len(), 8);
        assert_eq!(labels[0].content, "o8");
        assert_eq!(labels[7].content, "o15");
        // The track spans the list's full height; the thumb rides inside it, here
        // at the bottom of its travel.
        let (track, thumb) = (&sprites[2], &sprites[3]);
        assert_eq!((track.y, track.height), (140.0, 320.0));
        assert!(thumb.x >= track.x && thumb.x + thumb.width <= track.x + track.width);
        assert_eq!(thumb.y + thumb.height, track.y + track.height);
    }

    // A list that fits needs no scrollbar, and an out-of-range scroll position
    // clamps rather than windowing rows away.
    #[test]
    fn build_dropdown_overlay_fitting_list_has_no_scrollbar() {
        let mut view = dropdown_view(&["aa", "bb", "cc"]);
        view.first = 99;
        let (sprites, labels) = build_dropdown_overlay(&view, &loaded_fonts());
        // Border + panel + the selected highlight only.
        assert_eq!(sprites.len(), 3);
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].content, "aa");
    }

    // Without a loaded font there is no line height to centre on, so the text
    // falls back to the row's midpoint.
    #[test]
    fn build_dropdown_overlay_without_a_loaded_font_centres_on_a_zero_line() {
        let view = dropdown_view(&["aa"]);
        let (_, labels) = build_dropdown_overlay(&view, &no_fonts());
        assert_eq!(labels[0].y, 160.0);
    }

    // An empty, unfocused field shows the dimmer placeholder over a box mirroring
    // the field's geometry, with no caret.
    #[test]
    fn build_text_input_overlay_shows_the_placeholder_while_empty_and_unfocused() {
        let ti = text_input();
        let (sprites, labels) = build_text_input_overlay(&ti, &loaded_fonts(), true);
        assert_eq!(sprites.len(), 1);
        assert_eq!(rect(&sprites[0]), [ti.x, ti.y, ti.width, ti.height]);
        assert_eq!(sprites[0].tint, ti.background);
        assert_eq!(sprites[0].corner_radius, ti.corner_radius);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].content, "type here");
        assert_eq!(labels[0].color, ti.placeholder_color);
        // Text starts at the padding inset; the 16px line centres in the 40px box.
        assert_eq!((labels[0].x, labels[0].y), (8.0, 12.0));
    }

    // Content, or focus on an empty field, replaces the placeholder with the live
    // text in the content colour.
    #[test]
    fn build_text_input_overlay_shows_content_over_the_placeholder() {
        let mut ti = text_input();
        ti.content = "abc".to_string();
        let (_, labels) = build_text_input_overlay(&ti, &loaded_fonts(), false);
        assert_eq!(labels[0].content, "abc");
        assert_eq!(labels[0].color, ti.text_color);
        // Focusing an empty field drops the placeholder: it is being typed into.
        let focused = TextInput {
            focused: true,
            ..text_input()
        };
        let (_, labels) = build_text_input_overlay(&focused, &loaded_fonts(), false);
        assert_eq!(labels[0].content, "");
        assert_eq!(labels[0].color, focused.text_color);
    }

    // The caret bar draws only while the field holds focus and the blink is on its
    // visible half, sitting at the caret's fit position.
    #[test]
    fn build_text_input_overlay_draws_the_caret_only_on_a_focused_visible_blink() {
        let mut ti = text_input();
        ti.content = "abc".to_string();
        ti.focused = true;
        ti.caret = 3;
        let (sprites, _) = build_text_input_overlay(&ti, &loaded_fonts(), true);
        assert_eq!(sprites.len(), 2);
        // padding (8) + three 10px glyphs; a 2px bar spanning the 16px line box.
        assert_eq!(rect(&sprites[1]), [38.0, 12.0, 2.0, 16.0]);
        assert_eq!(sprites[1].tint[3], 1.0, "the caret always draws opaque");
        // The blink's dark half draws the box alone.
        assert_eq!(
            build_text_input_overlay(&ti, &loaded_fonts(), false)
                .0
                .len(),
            1
        );
        // So does an unfocused field.
        ti.focused = false;
        assert_eq!(
            build_text_input_overlay(&ti, &loaded_fonts(), true).0.len(),
            1
        );
    }

    // Without a loaded font nothing can be measured: the text passes through
    // unfit, the line height falls back to a fraction of the field, and no caret
    // draws even while focused.
    #[test]
    fn build_text_input_overlay_without_a_loaded_font_passes_text_through() {
        let mut ti = text_input();
        // Long enough that a measured field would scroll or truncate it.
        ti.content = "abcdefghijklmnopqrstuvwxyz".to_string();
        ti.focused = true;
        let (sprites, labels) = build_text_input_overlay(&ti, &no_fonts(), true);
        assert_eq!(sprites.len(), 1);
        assert_eq!(labels[0].content, ti.content);
        // line_h = height * 0.6 = 24, centred in the 40px box.
        assert_eq!((labels[0].x, labels[0].y), (8.0, 8.0));
    }
}
