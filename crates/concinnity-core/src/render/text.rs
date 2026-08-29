//! Font atlas data and text draw-call assembly. No backend ownership; the
//! renderer uploads the atlas textures; this module only builds the quad
//! geometry from TextLabel components each frame.

use crate::components::{LabelBox, SpriteFit, TextAlign, TextLabel};
use crate::ecs::FontHandle;
use crate::gfx::overlay::OverlayTransform;
use crate::gfx::render_types::{TextDrawCall, TextVertex};
use crate::render::overlay_maps::{ClipRects, OverlayLayers};
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// One face's per-glyph metrics, keyed by Unicode code point.
pub type FontMetrics = HashMap<u32, crate::gfx::font::GlyphMetrics>;

/// Per-font data kept in memory after init() so step() can build text quads each frame.
pub struct LoadedFont {
    /// Index into the backend's text atlas texture array.
    pub atlas_slot: usize,
    /// Per-glyph metrics keyed by Unicode code point.
    pub metrics: FontMetrics,
    /// Atlas width in pixels.
    pub atlas_w: u32,
    /// Atlas height in pixels.
    pub atlas_h: u32,
    /// Rasterisation height (px) used to position glyphs vertically.
    pub size_px: f32,
    /// Cap height (logical px): the bearing of an uppercase reference glyph, used
    /// to vertically center the visible text within its line box. The full em
    /// (`size_px`) is taller than the visible glyphs, so centering on the em alone
    /// leaves a gap above the caps; centering the cap band fixes that.
    pub cap_px: f32,
    /// Atlas supersample factor: glyph `atlas_w`/`atlas_h` are stored in atlas
    /// texels, which are this many times larger than the glyph's size in logical
    /// (layout) pixels. The on-screen quad divides by it so the text lays out at
    /// its requested size while the extra texels supersample the glyph.
    pub supersample: f32,
}

/// The faces a frame draws with: every font loaded at init, keyed by handle,
/// plus the face a label naming no font of its own falls back to.
///
/// Nothing compiles a font for text that names none, whichever way the world was
/// assembled, so the renderer registers the engine's built-in face as the
/// fallback whenever some text needs it. A world whose text all names a Font
/// leaves the fallback unset and pays nothing for it.
#[derive(Default)]
pub struct FontSet {
    faces: HashMap<FontHandle, LoadedFont>,
    fallback: Option<FontHandle>,
}

impl FontSet {
    /// Register `font` under `handle`, replacing any face already there.
    pub fn insert(&mut self, handle: FontHandle, font: LoadedFont) {
        self.faces.insert(handle, font);
    }

    /// Draw labels naming no font with the face under `handle`.
    pub fn set_fallback(&mut self, handle: FontHandle) {
        self.fallback = Some(handle);
    }

    /// The face registered under `handle`.
    pub fn get(&self, handle: FontHandle) -> Option<&LoadedFont> {
        self.faces.get(&handle)
    }

    /// The face a label draws with: the one it names, or the fallback when it
    /// names none (or names one that never loaded).
    pub fn resolve(&self, font: Option<FontHandle>) -> Option<&LoadedFont> {
        font.and_then(|h| self.faces.get(&h))
            .or_else(|| self.fallback.and_then(|h| self.faces.get(&h)))
    }

    /// Number of loaded faces.
    pub fn len(&self) -> usize {
        self.faces.len()
    }

    /// Whether no face loaded at all, in which case no text draws.
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// Atlas slot of an arbitrary loaded face, which the sprite shaper uses as
    /// the texture slot for its untextured (sentinel-UV) quads.
    pub fn any_atlas_slot(&self) -> Option<usize> {
        self.faces.values().next().map(|f| f.atlas_slot)
    }
}

/// Cap height (logical px) for vertical centering: the bearing of an uppercase
/// reference glyph ('H'), falling back to the tallest uppercase glyph, then to a
/// fraction of the em when no metrics are available.
pub fn derive_cap_px(metrics: &FontMetrics, size_px: f32) -> f32 {
    if let Some(h) = metrics.get(&('H' as u32))
        && h.bearing_y > 0.0
    {
        return h.bearing_y;
    }
    let max_upper = ('A'..='Z')
        .filter_map(|c| metrics.get(&(c as u32)))
        .map(|m| m.bearing_y)
        .fold(0.0_f32, f32::max);
    if max_upper > 0.0 {
        return max_upper;
    }
    0.7 * size_px
}

/// Width of the widest line of `content` in scaled pixels: what a label's
/// background box and its centre / right alignment are both sized against.
pub fn widest_line_width(content: &str, font: &LoadedFont, scale: f32) -> f32 {
    content
        .split('\n')
        .map(|line| text_advance_width(line, font, scale))
        .fold(0.0_f32, f32::max)
}

/// Advance width of `content` in scaled pixels for the given font, for placing a
/// caret, sizing a field's text, or centring a line. Newlines carry no advance,
/// so a multi-line string measures as if its lines were concatenated.
pub fn text_advance_width(content: &str, font: &LoadedFont, scale: f32) -> f32 {
    content
        .chars()
        .filter(|&ch| ch != '\n')
        .map(|ch| advance_px(ch, font, scale))
        .sum()
}

// One glyph's advance in scaled pixels, substituting the space glyph's advance
// for a missing metric.
fn advance_px(ch: char, font: &LoadedFont, scale: f32) -> f32 {
    font.metrics
        .get(&(ch as u32))
        .map(|m| m.advance_px * scale)
        .unwrap_or_else(|| {
            font.metrics
                .get(&(b' ' as u32))
                .map(|m| m.advance_px * scale)
                .unwrap_or(0.0)
        })
}

// The content a label actually draws: its text broken to `wrap_width` and
// capped at `max_lines`. Wrapping measures in the label's own pixel space with
// `label.scale`, which gives the same breaks as measuring in window pixels: a
// screen-owned label scales its advances and its wrap width by the same overlay
// factor. A centered label has no container (it is fitted to the viewport), so
// it is left alone. Borrows the authored content whenever no line breaks or
// truncates, so a fitting label allocates nothing.
fn laid_out<'a>(label: &'a TextLabel, font: &LoadedFont) -> Cow<'a, str> {
    if label.centered || (label.wrap_width <= 0.0 && label.max_lines == 0) {
        return Cow::Borrowed(&label.content);
    }
    let mut lines: Vec<&str> = Vec::new();
    let mut authored_lines = 0usize;
    for authored in label.content.split('\n') {
        authored_lines += 1;
        if label.wrap_width > 0.0 {
            wrap_line(authored, font, label.scale, label.wrap_width, &mut lines);
        } else {
            lines.push(authored);
        }
    }
    let max = label.max_lines as usize;
    let truncated = max > 0 && lines.len() > max;
    if truncated {
        lines.truncate(max);
    }
    if !truncated && lines.len() == authored_lines {
        return Cow::Borrowed(&label.content);
    }
    let ellipsized = truncated
        .then(|| lines.pop())
        .flatten()
        .map(|last| with_ellipsis(last, font, label.scale, label.wrap_width));
    let mut out = String::with_capacity(label.content.len() + 4);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(line);
    }
    if let Some(last) = ellipsized {
        if !lines.is_empty() {
            out.push('\n');
        }
        out.push_str(&last);
    }
    Cow::Owned(out)
}

// Greedily pack `line`'s words into `out` as subslices of `line`, breaking at
// spaces. A word too wide to fit a line of its own is split mid-word, since
// leaving it whole would put it back outside the container wrapping exists to
// respect. Widths accumulate one glyph advance at a time in authored order,
// matching a from-scratch measure of the same text.
fn wrap_line<'a>(line: &'a str, font: &LoadedFont, scale: f32, width: f32, out: &mut Vec<&'a str>) {
    let advance = |ch: char| advance_px(ch, font, scale);
    // The line under construction, `line[start..end]`, and its measured width.
    let (mut start, mut end) = (0usize, 0usize);
    let mut current_width = 0.0_f32;
    // Byte offset of the next word (words are separated by single spaces).
    let mut pos = 0usize;
    for word in line.split(' ') {
        let word_end = pos + word.len();
        // The candidate: the word appended to the current line (joined by the
        // space between them), or the word alone when the line is empty.
        let (cand_start, cand_width) = if end > start {
            let mut w = current_width;
            for ch in line[end..word_end].chars() {
                w += advance(ch);
            }
            (start, w)
        } else {
            let mut w = 0.0_f32;
            for ch in word.chars() {
                w += advance(ch);
            }
            (pos, w)
        };
        if cand_width <= width {
            (start, end, current_width) = (cand_start, word_end, cand_width);
            pos = word_end + 1;
            continue;
        }
        if end > start {
            out.push(&line[start..end]);
        }
        // The word now starts a line of its own; split it if even that overflows.
        (start, end) = (pos, word_end);
        loop {
            let (mut w, mut chars) = (0.0_f32, 0usize);
            for ch in line[start..end].chars() {
                w += advance(ch);
                chars += 1;
            }
            if w <= width || chars <= 1 {
                current_width = w;
                break;
            }
            // The longest head (at least one char) that fits the width.
            let mut acc = 0.0_f32;
            let mut head_end = start;
            for (i, ch) in line[start..end].char_indices() {
                let next = acc + advance(ch);
                if head_end > start && next > width {
                    break;
                }
                acc = next;
                head_end = start + i + ch.len_utf8();
            }
            out.push(&line[start..head_end]);
            start = head_end;
        }
        pos = word_end + 1;
    }
    out.push(&line[start..end]);
}

// `line` shortened until it and a trailing ellipsis fit `width`: the longest
// prefix whose width plus the ellipsis fits, found in one forward scan. Prefix
// widths accumulate one glyph advance at a time in authored order, with the
// ellipsis advances added after, matching a from-scratch measure of the same
// candidate. A zero width (capping lines without wrapping them) leaves the
// line as it is.
fn with_ellipsis(line: &str, font: &LoadedFont, scale: f32, width: f32) -> String {
    const ELLIPSIS: &str = "...";
    let mut out = String::with_capacity(line.len() + ELLIPSIS.len());
    let mut end = line.len();
    if width > 0.0 {
        let ellipsis_w: f32 = ELLIPSIS.chars().map(|ch| advance_px(ch, font, scale)).sum();
        end = 0;
        let mut prefix_w = 0.0_f32;
        for (i, ch) in line.char_indices() {
            let w = prefix_w + advance_px(ch, font, scale);
            if w + ellipsis_w > width {
                break;
            }
            prefix_w = w;
            end = i + ch.len_utf8();
        }
    }
    out.push_str(&line[..end]);
    out.push_str(ELLIPSIS);
    out
}

// Baseline position relative to a label's top-left `y`, so the cap-height band
// is vertically centered within the line box `[y, y + line_height]`. Pinning the
// baseline to the box bottom (the old behaviour) left a large gap above the
// glyphs; centering the cap band makes short UI text sit centered in its box.
fn baseline_offset(font: &LoadedFont, scale: f32) -> f32 {
    let line_height = font.size_px * scale;
    (line_height + font.cap_px * scale) / 2.0
}

// The visible glyphs' vertical extent above and below the first line's baseline,
// in scaled pixels: how far the tallest glyph rises (ascent) and the lowest
// glyph drops (descent). A tight background box and the layout measurement both
// hug this, so the box wraps the ink with `padding` on every side instead of the
// full em line box.
fn content_v_extent(content: &str, font: &LoadedFont, scale: f32) -> (f32, f32) {
    let mut top_above = 0.0_f32;
    let mut bot_below = 0.0_f32;
    for ch in content.chars() {
        if ch == '\n' {
            continue;
        }
        if let Some(m) = font.metrics.get(&(ch as u32)) {
            if m.atlas_h == 0 {
                continue;
            }
            top_above = top_above.max(m.bearing_y * scale);
            bot_below = bot_below.max((m.atlas_h as f32 / font.supersample - m.bearing_y) * scale);
        }
    }
    (top_above, bot_below)
}

/// Measure a label's background-box extent for layout: a box hugging the visible
/// glyphs grown by the label's padding on every side, plus one line height per
/// extra `\n`-split line. Mirrors the background-box math in `build_text_calls`.
/// `top_inset` is the gap from the box top down to the text origin (the label's
/// `y`), which `LayoutContainer` uses to place the box. Returns `None` for a
/// hidden label, or one with no font to draw with at all, so a
/// `LayoutContainer` drops it and reserves no space.
pub fn measure_label_box(label: &TextLabel, loaded_fonts: &FontSet) -> Option<LabelBox> {
    if !label.visible {
        return None;
    }
    let font = loaded_fonts.resolve(label.font)?;
    let scale = label.scale;
    let line_height = font.size_px * scale;
    let content = laid_out(label, font);
    let lines = content.split('\n').count().max(1) as f32;
    let text_w = widest_line_width(&content, font, scale);
    let pad = label.padding;
    let (top_above, bot_below) = content_v_extent(&content, font, scale);
    let base_off = baseline_offset(font, scale);
    Some(LabelBox {
        w: text_w + 2.0 * pad,
        h: top_above + bot_below + (lines - 1.0) * line_height + 2.0 * pad,
        pad,
        // Box top is `base_off - top_above - pad` below the origin; the inset is
        // the origin's distance below the box top.
        top_inset: top_above + pad - base_off,
    })
}

/// Build one TextDrawCall per TextLabel, laying out character quads using the
/// loaded font metrics. When `win_w` and `win_h` are both > 0.0, labels with
/// `centered = true` are repositioned to the centre of the viewport. `clips`
/// maps an element id to a reference-space clip band; a label found there has
/// its call scissored to that band (mapped to the window), so a scrollable
/// panel's off-band rows do not bleed over its chrome.
pub fn build_text_calls(
    labels: &[TextLabel],
    loaded_fonts: &FontSet,
    win_w: f32,
    win_h: f32,
    clips: &ClipRects,
    layers: &OverlayLayers,
) -> Vec<TextDrawCall> {
    let mut out = crate::render::call_buffer::TextCallBuffer::default();
    build_text_calls_into(&mut out, labels, loaded_fonts, win_w, win_h, clips, layers);
    out.take()
}

/// `build_text_calls`, appending onto an existing draw list so a caller
/// assembling a frame from several element groups reuses one buffer (and, in
/// steady state, the pooled geometry of the spent frame it recycled).
pub fn build_text_calls_into(
    out: &mut crate::render::call_buffer::TextCallBuffer,
    labels: &[TextLabel],
    loaded_fonts: &FontSet,
    win_w: f32,
    win_h: f32,
    clips: &ClipRects,
    layers: &OverlayLayers,
) {
    // Screen-owned labels are overlay UI authored in the reference canvas; map
    // them to the live window so menus scale with the window. HUD labels
    // (view == None) keep literal window pixels.
    let overlay = OverlayTransform::from_viewport([win_w, win_h]);
    // Alternate mappings a view-owned label may opt into via `fit`.
    let bottom = OverlayTransform::bottom_anchored_from_viewport([win_w, win_h]);
    let cover = OverlayTransform::cover_from_viewport([win_w, win_h]);
    for label in labels {
        if !label.visible {
            continue;
        }
        let font = match loaded_fonts.resolve(label.font) {
            Some(f) => f,
            None => continue,
        };
        // Everything below reads the laid-out content, not the authored string,
        // so alignment, the background box, and the glyph run agree on the lines
        // that are actually drawn.
        let content = laid_out(label, font);
        // One quad per glyph plus the optional background box; the byte length
        // upper-bounds the glyph count.
        let quads = content.len() + 1;
        let (mut vertices, mut indices) = out.geometry();
        vertices.reserve(4 * quads);
        indices.reserve(6 * quads);

        // For centered labels, auto-scale to fill ~85% of the viewport while
        // preserving the text's aspect ratio. The label's scale field is used
        // for non-centered labels only.
        // Set when horizontal alignment measures it, so the background box
        // below does not measure the same content a second time.
        let mut widest_line: Option<f32> = None;
        let (x0, y0, scale) = if label.centered && win_w > 0.0 && win_h > 0.0 {
            let w1 = text_advance_width(&content, font, 1.0);
            let h1 = font.size_px;
            let scale = if w1 > 0.0 && h1 > 0.0 {
                let sw = win_w * 0.85 / w1;
                let sh = win_h * 0.85 / h1;
                sw.min(sh)
            } else {
                label.scale
            };
            let tw = text_advance_width(&content, font, scale);
            let th = h1 * scale;
            ((win_w - tw) / 2.0, (win_h - th) / 2.0, scale)
        } else {
            // The anchor point and scale: a view-owned label maps through its
            // `fit` transform, a HUD label stays in literal window pixels.
            let (ax, ay, scale) = if label.screen.is_some() {
                let t = match label.fit {
                    SpriteFit::Bottom => bottom,
                    SpriteFit::Cover => cover,
                    SpriteFit::Fit => overlay,
                };
                let (sx, sy) = t.forward(label.x, label.y);
                (sx, sy, label.scale * t.scale())
            } else {
                (label.x, label.y, label.scale)
            };
            // Horizontal alignment shifts the anchor by the rendered width,
            // measured with the real metrics so centered UI text sits exactly
            // on its anchor at any scale (centering by the widest line).
            let x0 = match label.align {
                TextAlign::Left => ax,
                TextAlign::Center | TextAlign::Right => {
                    let w = widest_line_width(&content, font, scale);
                    widest_line = Some(w);
                    if label.align == TextAlign::Center {
                        ax - w / 2.0
                    } else {
                        ax - w
                    }
                }
            };
            (x0, ay, scale)
        };

        let mut x_cursor = x0;
        // baseline: positioned so the cap-height band is centered within the line
        // box, so short UI text sits vertically centered rather than pinned to
        // the box bottom. Advanced by one line height on each newline so
        // multi-line labels lay out down the screen.
        let line_height = font.size_px * scale;
        let mut baseline = y0 + baseline_offset(font, scale);
        let aw = font.atlas_w as f32;
        let ah = font.atlas_h as f32;

        // Background box: a filled quad behind the glyphs, sized to hug the
        // visible glyphs grown by `padding` on every side (not the full em line
        // box, which left a large gap above the caps). Emitted first so the
        // glyphs composite on top. It carries a sentinel UV (a negative u) that
        // the text shader reads as "solid fill", with the box alpha passed
        // through in v. Empty content draws nothing at all (so a blanked label
        // fully disappears).
        if label.background[3] > 0.0 && !content.is_empty() {
            let lines = content.split('\n').count().max(1) as f32;
            let text_w = widest_line.unwrap_or_else(|| widest_line_width(&content, font, scale));
            let pad = label.padding;
            let (top_above, bot_below) = content_v_extent(&content, font, scale);
            let last_baseline = baseline + (lines - 1.0) * line_height;
            let (x0b, y0b) = (x0 - pad, baseline - top_above - pad);
            let (x1b, y1b) = (x0 + text_w + pad, last_baseline + bot_below + pad);
            let bg = [
                label.background[0],
                label.background[1],
                label.background[2],
            ];
            let ba = label.background[3];
            let box_vtx = |x: f32, y: f32| TextVertex {
                pos: [x, y],
                uv: [-1.0, ba],
                color: bg,
                mode: 0.0,
            };
            vertices.extend_from_slice(&[
                box_vtx(x0b, y0b),
                box_vtx(x1b, y0b),
                box_vtx(x1b, y1b),
                box_vtx(x0b, y1b),
            ]);
            indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        }

        for ch in content.chars() {
            if ch == '\n' {
                x_cursor = x0;
                baseline += line_height;
                continue;
            }
            let m = match font.metrics.get(&(ch as u32)) {
                Some(m) => m,
                None => {
                    if let Some(sp) = font.metrics.get(&(b' ' as u32)) {
                        x_cursor += sp.advance_px * scale;
                    }
                    continue;
                }
            };
            if m.atlas_w == 0 || m.atlas_h == 0 {
                x_cursor += m.advance_px * scale;
                continue;
            }
            // atlas_w/atlas_h are in supersampled atlas texels; divide by the
            // supersample factor to get the glyph's logical size before scaling
            // to the screen. The UVs below still address the full texel extent.
            let gw = m.atlas_w as f32 / font.supersample * scale;
            let gh = m.atlas_h as f32 / font.supersample * scale;
            let gx = x_cursor + m.bearing_x * scale;
            let gy = baseline - m.bearing_y * scale;
            let u0 = m.atlas_x as f32 / aw;
            let v0 = m.atlas_y as f32 / ah;
            let u1 = (m.atlas_x as f32 + m.atlas_w as f32) / aw;
            let v1 = (m.atlas_y as f32 + m.atlas_h as f32) / ah;
            let base = vertices.len() as u16;
            vertices.extend_from_slice(&[
                TextVertex {
                    pos: [gx, gy],
                    uv: [u0, v0],
                    color: label.color,
                    mode: 0.0,
                },
                TextVertex {
                    pos: [gx + gw, gy],
                    uv: [u1, v0],
                    color: label.color,
                    mode: 0.0,
                },
                TextVertex {
                    pos: [gx + gw, gy + gh],
                    uv: [u1, v1],
                    color: label.color,
                    mode: 0.0,
                },
                TextVertex {
                    pos: [gx, gy + gh],
                    uv: [u0, v1],
                    color: label.color,
                    mode: 0.0,
                },
            ]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            x_cursor += m.advance_px * scale;
        }
        if vertices.is_empty() {
            out.park(vertices, indices);
        } else {
            out.calls.push(TextDrawCall {
                vertices,
                indices,
                atlas_slot: font.atlas_slot,
                clip_rect: clips
                    .get(&label.asset_id)
                    .map(|b| band_to_window(&overlay, *b)),
                layer: layers.get(&label.asset_id).copied().unwrap_or(0),
            });
        }
    }
}

// Map a reference-space clip band `[x, y, width, height]` to a window-space
// rectangle through the overlay transform, so the backend can scissor to it.
pub(crate) fn band_to_window(overlay: &OverlayTransform, band: [f32; 4]) -> [f32; 4] {
    let (x0, y0) = overlay.forward(band[0], band[1]);
    let (x1, y1) = overlay.forward(band[0] + band[2], band[1] + band[3]);
    [x0, y0, x1 - x0, y1 - y0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;
    use crate::gfx::font::GlyphMetrics;

    use alloc::string::ToString;
    // No clip bands: every label draws unclipped.
    fn no_clips() -> ClipRects {
        ClipRects::new()
    }
    fn no_layers() -> OverlayLayers {
        OverlayLayers::new()
    }

    fn make_glyph(atlas_w: u16, atlas_h: u16, advance_px: f32) -> GlyphMetrics {
        GlyphMetrics {
            char_code: 0,
            atlas_x: 0,
            atlas_y: 0,
            atlas_w,
            atlas_h,
            advance_px,
            bearing_x: 0.0,
            bearing_y: atlas_h as f32,
        }
    }

    fn make_font(chars: &[(char, GlyphMetrics)]) -> LoadedFont {
        let metrics: FontMetrics = chars.iter().map(|(c, m)| (*c as u32, *m)).collect();
        let cap_px = derive_cap_px(&metrics, 16.0);
        LoadedFont {
            atlas_slot: 0,
            cap_px,
            metrics,
            atlas_w: 128,
            atlas_h: 128,
            size_px: 16.0,
            // 1x: the unit tests express glyph sizes directly in atlas texels.
            supersample: 1.0,
        }
    }

    fn make_label(font: FontHandle, content: &str, x: f32) -> TextLabel {
        TextLabel {
            asset_id: AssetId::default(),
            font: Some(font),
            content: content.to_string(),
            x,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: crate::components::TextAlign::Left,
            fit: crate::components::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            wrap_width: 0.0,
            max_lines: 0,
            visible: true,
            screen: None,
        }
    }

    // A font whose every glyph advances 10px, so a wrap width in pixels reads
    // directly as a character count.
    fn even_font() -> LoadedFont {
        let glyphs: Vec<(char, GlyphMetrics)> = ('a'..='z')
            .chain(['A', ' ', '-', '.', '\''])
            .map(|c| (c, make_glyph(8, 8, 10.0)))
            .collect();
        make_font(&glyphs)
    }

    fn wrapped(content: &str, width: f32, max_lines: u32) -> Vec<String> {
        let font = even_font();
        let mut label = make_label(FontHandle(0), content, 0.0);
        label.wrap_width = width;
        label.max_lines = max_lines;
        laid_out(&label, &font)
            .split('\n')
            .map(String::from)
            .collect()
    }

    #[test]
    fn text_wraps_at_word_boundaries_within_its_width() {
        // 5 glyphs per line: "aaa bbb" is 7 wide, so the words split.
        assert_eq!(wrapped("aaa bbb", 50.0, 0), ["aaa", "bbb"]);
        // Exactly filling a line does not spill onto the next one.
        assert_eq!(wrapped("aa bb", 50.0, 0), ["aa bb"]);
        assert_eq!(wrapped("aa bb cc", 50.0, 0), ["aa bb", "cc"]);
    }

    #[test]
    fn authored_newlines_stay_breaks_and_wrap_within_themselves() {
        assert_eq!(wrapped("aa\nbb cc dd", 50.0, 0), ["aa", "bb cc", "dd"]);
    }

    #[test]
    fn a_word_too_long_for_a_line_splits_rather_than_overflowing() {
        assert_eq!(wrapped("aaaaaaaa", 50.0, 0), ["aaaaa", "aaa"]);
        // Every produced line is inside the width, which is the whole point.
        let font = even_font();
        for line in wrapped("aaaaaaaaaaaaaa bb", 50.0, 0) {
            assert!(text_advance_width(&line, &font, 1.0) <= 50.0, "{line:?}");
        }
    }

    #[test]
    fn max_lines_cuts_the_overflow_with_an_ellipsis_that_still_fits() {
        // Three lines' worth of text into a two-line box.
        assert_eq!(wrapped("aa bb cc dd ee ff", 50.0, 0).len(), 3);
        let lines = wrapped("aa bb cc dd ee ff", 50.0, 2);
        assert_eq!(lines.len(), 2);
        assert!(lines[1].ends_with("..."), "{lines:?}");
        let font = even_font();
        for line in &lines {
            assert!(text_advance_width(line, &font, 1.0) <= 50.0, "{line:?}");
        }
    }

    #[test]
    fn text_that_fits_is_left_exactly_as_it_was() {
        assert_eq!(wrapped("aa bb", 500.0, 4), ["aa bb"]);
        assert_eq!(wrapped("", 50.0, 2), [""]);
    }

    // Wrapping has to reach the glyph run, the alignment measure, and the
    // background box together, or a wrapped label draws its box around the
    // unwrapped text.
    #[test]
    fn a_wrapped_label_draws_and_measures_the_lines_it_wrapped_to() {
        let font_id = FontHandle(0);
        let mut fonts = FontSet::default();
        fonts.insert(font_id, even_font());
        let mut label = make_label(font_id, "aa bb cc", 0.0);
        label.wrap_width = 50.0;
        label.background = [0.0, 0.0, 0.0, 1.0];

        let boxed = measure_label_box(&label, &fonts).unwrap();
        // Two lines of five glyphs, not one line of eight.
        assert!(boxed.w <= 50.0, "{boxed:?}");
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            200.0,
            200.0,
            &no_clips(),
            &no_layers(),
        );
        let right = calls[0]
            .vertices
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::MIN, f32::max);
        assert!(right <= 50.0, "glyphs ran past the wrap width: {right}");
    }

    #[test]
    fn empty_labels_returns_empty_calls() {
        let fonts = FontSet::default();
        assert!(build_text_calls(&[], &fonts, 0.0, 0.0, &no_clips(), &no_layers()).is_empty());
    }

    // A world assembled in code has no compiled Font for its labels to name, so
    // the renderer registers a fallback face and every font-less label draws
    // with it.
    #[test]
    fn a_label_naming_no_font_draws_with_the_fallback() {
        let g = make_glyph(8, 8, 10.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let mut label = make_label(FontHandle(0), "A", 0.0);
        label.font = None;

        // With no fallback there is no face to lay the glyphs out with.
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert!(calls.is_empty());
        assert!(measure_label_box(&label, &fonts).is_none());

        fonts.set_fallback(FontHandle(0));
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        assert!(measure_label_box(&label, &fonts).is_some());
    }

    // The fallback only stands in for a label that resolves to no face of its
    // own; a label naming a loaded font keeps it.
    #[test]
    fn a_named_font_wins_over_the_fallback() {
        let wide = make_glyph(8, 8, 20.0);
        let narrow = make_glyph(8, 8, 5.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', wide)]));
        fonts.insert(FontHandle(1), make_font(&[('A', narrow)]));
        fonts.set_fallback(FontHandle(0));

        let label = make_label(FontHandle(1), "AA", 0.0);
        assert_eq!(measure_label_box(&label, &fonts).unwrap().w, 10.0);
    }

    // A label naming a font that never loaded falls back rather than vanishing,
    // so a missing face costs styling instead of the text itself.
    #[test]
    fn an_unloaded_font_falls_back() {
        let g = make_glyph(8, 8, 10.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        fonts.set_fallback(FontHandle(0));

        let label = make_label(FontHandle(99), "A", 0.0);
        assert!(measure_label_box(&label, &fonts).is_some());
    }

    #[test]
    fn unknown_font_produces_no_call() {
        let fonts = FontSet::default();
        let label = make_label(FontHandle(99), "hello", 0.0);
        assert!(
            build_text_calls(
                core::slice::from_ref(&label),
                &fonts,
                0.0,
                0.0,
                &no_clips(),
                &no_layers()
            )
            .is_empty()
        );
    }

    #[test]
    fn single_glyph_produces_quad() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let label = make_label(FontHandle(0), "A", 0.0);
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4);
        assert_eq!(calls[0].indices.len(), 6);
        assert_eq!(calls[0].atlas_slot, 0);
    }

    #[test]
    fn background_prepends_a_box_quad() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let mut label = make_label(FontHandle(0), "A", 0.0);
        label.background = [0.0, 0.3, 0.1, 0.85];
        label.padding = 4.0;
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        // 4 box verts prepended + 4 glyph verts; 6 box indices + 6 glyph.
        assert_eq!(calls[0].vertices.len(), 8);
        assert_eq!(calls[0].indices.len(), 12);
        // The box quad comes first: sentinel u (< 0), box alpha carried in v.
        for v in &calls[0].vertices[..4] {
            assert!(v.uv[0] < 0.0, "box vert should carry the sentinel u");
            assert!((v.uv[1] - 0.85).abs() < 1e-4, "box alpha travels in v");
        }
        // Glyph verts keep real, non-negative atlas UVs.
        assert!(calls[0].vertices[4].uv[0] >= 0.0);
    }

    #[test]
    fn derive_cap_px_uses_uppercase_reference() {
        // 'H' is the cap-height reference, even when a lowercase glyph is taller.
        let mut m = HashMap::new();
        m.insert('H' as u32, make_glyph(8, 10, 9.0)); // bearing_y = 10
        m.insert('g' as u32, make_glyph(8, 14, 9.0)); // taller, but lowercase
        assert!((derive_cap_px(&m, 16.0) - 10.0).abs() < 1e-4);
        // With no glyphs, fall back to a fraction of the em.
        let empty = HashMap::new();
        assert!((derive_cap_px(&empty, 20.0) - 14.0).abs() < 1e-4);
    }

    #[test]
    fn background_box_hugs_glyph_with_symmetric_padding() {
        // The box wraps the visible glyph with `padding` above and below (instead
        // of the full em line box, which left a large gap above the caps).
        let g = make_glyph(10, 12, 11.0); // bearing_y = 12, no descent
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let mut label = make_label(FontHandle(0), "A", 0.0);
        label.background = [0.1, 0.1, 0.1, 1.0];
        label.padding = 4.0;
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        let v = &calls[0].vertices;
        // Verts 0..4 are the box; 4..8 the glyph quad.
        let (box_top, box_bot) = (v[0].pos[1], v[2].pos[1]);
        let (glyph_top, glyph_bot) = (v[4].pos[1], v[6].pos[1]);
        assert!(
            (glyph_top - box_top - 4.0).abs() < 1e-4,
            "top pad = {}",
            glyph_top - box_top
        );
        assert!(
            (box_bot - glyph_bot - 4.0).abs() < 1e-4,
            "bottom pad = {}",
            box_bot - glyph_bot
        );
    }

    #[test]
    fn background_with_empty_content_draws_nothing() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let mut label = make_label(FontHandle(0), "", 0.0);
        label.background = [0.0, 0.3, 0.1, 0.85];
        // A blanked label (e.g. a toggled-off HUD chip) draws no box.
        assert!(
            build_text_calls(
                core::slice::from_ref(&label),
                &fonts,
                0.0,
                0.0,
                &no_clips(),
                &no_layers()
            )
            .is_empty()
        );
    }

    #[test]
    fn space_advances_cursor_without_quad() {
        let space = make_glyph(0, 0, 8.0);
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[(' ', space), ('A', g)]));
        // Two spaces then 'A': only 'A' produces geometry.
        let label = make_label(FontHandle(0), "  A", 0.0);
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4);
        // 'A' quad starts after 2 × advance_px(space) = 16.0
        let gx = calls[0].vertices[0].pos[0];
        assert!((gx - 16.0).abs() < 1e-4, "expected gx=16.0, got {gx}");
    }

    #[test]
    fn zero_size_glyph_advances_cursor_without_quad() {
        // A glyph whose atlas dimensions are 0×0 is invisible but still advances x.
        let zero = GlyphMetrics {
            char_code: b'X' as u32,
            atlas_x: 0,
            atlas_y: 0,
            atlas_w: 0,
            atlas_h: 0,
            advance_px: 5.0,
            bearing_x: 0.0,
            bearing_y: 0.0,
        };
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('X', zero), ('A', g)]));
        let label = make_label(FontHandle(0), "XA", 0.0);
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4); // only 'A'
        // 'A' starts at x = advance_px('X') = 5.0
        assert!((calls[0].vertices[0].pos[0] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn newline_starts_a_new_line() {
        // "A\nA": the second glyph resets x to the label origin and drops
        // down by one line height (font size_px * scale = 16).
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let label = make_label(FontHandle(0), "A\nA", 0.0);
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        // Two glyphs -> two quads -> 8 vertices, 12 indices.
        assert_eq!(calls[0].vertices.len(), 8);
        assert_eq!(calls[0].indices.len(), 12);
        let first = &calls[0].vertices[0];
        let second = &calls[0].vertices[4];
        // x resets to the label origin on the new line.
        assert!((first.pos[0] - second.pos[0]).abs() < 1e-4);
        // y drops by exactly one line height.
        assert!(
            (second.pos[1] - first.pos[1] - 16.0).abs() < 1e-4,
            "expected +16 line height, got {}",
            second.pos[1] - first.pos[1]
        );
    }

    #[test]
    fn centered_label_is_repositioned() {
        let g = make_glyph(10, 12, 20.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let mut label = make_label(FontHandle(0), "A", 0.0);
        label.centered = true;
        // Viewport 200×100; glyph advance=20, size_px=16, cap_px=12 ('A' bearing).
        // Auto-scale: sw = 200*0.85/20 = 8.5, sh = 100*0.85/16 = 5.3125 -> scale = 5.3125
        // tw = 20*5.3125 = 106.25, th = 16*5.3125 = 85.0
        // x0 = (200 - 106.25) / 2 = 46.875, y0 = (100 - 85.0) / 2 = 7.5
        // line_height = 16*5.3125 = 85; baseline centers the cap band:
        // baseline = 7.5 + (85 + 12*5.3125)/2 = 7.5 + 74.375 = 81.875
        // gx = x0 + bearing_x*scale = 46.875, gy = baseline - bearing_y*scale = 81.875 - 63.75 = 18.125
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            200.0,
            100.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        let v = &calls[0].vertices[0];
        assert!((v.pos[0] - 46.875).abs() < 1e-3, "gx={}", v.pos[0]);
        assert!((v.pos[1] - 18.125).abs() < 1e-3, "gy={}", v.pos[1]);
    }

    #[test]
    fn view_owned_label_scales_and_repositions_with_overlay() {
        // A view-owned (overlay) label is authored in the reference canvas and
        // mapped onto the window. At a 2x viewport its origin moves to the
        // forward-mapped position and its scale doubles. A HUD label (view ==
        // None) at the same coordinates stays put.
        let g = make_glyph(10, 12, 20.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));

        let hud = make_label(FontHandle(0), "A", 100.0); // view == None
        let mut overlay_label = make_label(FontHandle(0), "A", 100.0);
        overlay_label.y = 100.0;
        overlay_label.screen = Some(AssetId(5));

        // 2x reference viewport (1280x720 -> 2560x1440): scale 2, centered.
        let vp = (2560.0, 1440.0);
        let hud_calls = build_text_calls(
            core::slice::from_ref(&hud),
            &fonts,
            vp.0,
            vp.1,
            &no_clips(),
            &no_layers(),
        );
        let ovl_calls = build_text_calls(
            core::slice::from_ref(&overlay_label),
            &fonts,
            vp.0,
            vp.1,
            &no_clips(),
            &no_layers(),
        );
        // HUD label keeps its literal origin (x = 100).
        assert!((hud_calls[0].vertices[0].pos[0] - 100.0).abs() < 1e-3);
        // Overlay label: forward(100,100) at scale 2 -> x = 1280 + (100-640)*2 = 200.
        assert!(
            (ovl_calls[0].vertices[0].pos[0] - 200.0).abs() < 1e-3,
            "x={}",
            ovl_calls[0].vertices[0].pos[0]
        );
        // Glyph width doubles (atlas_w 10 -> 20 on screen).
        let w = ovl_calls[0].vertices[1].pos[0] - ovl_calls[0].vertices[0].pos[0];
        assert!((w - 20.0).abs() < 1e-3, "w={w}");
    }

    #[test]
    fn measure_label_box_grows_text_by_padding() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g), ('B', g)]));
        let mut label = make_label(FontHandle(0), "AB", 0.0);
        label.padding = 4.0;
        let b = measure_label_box(&label, &fonts).unwrap();
        // text width = 2 * advance(11) = 22, grown by padding on both sides.
        assert!((b.w - 30.0).abs() < 1e-4, "w={}", b.w);
        // The box hugs the glyphs: ascent(bearing_y=12) + descent(0) + 2*pad(4) = 20.
        assert!((b.h - 20.0).abs() < 1e-4, "h={}", b.h);
        assert!((b.pad - 4.0).abs() < 1e-4);
        // top_inset = ascent(12) + pad(4) - baseline_offset((16+12)/2=14) = 2.
        assert!(
            (b.top_inset - 2.0).abs() < 1e-4,
            "top_inset={}",
            b.top_inset
        );
    }

    #[test]
    fn measure_label_box_skips_hidden_and_unloaded() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        // Hidden label → None even with a loaded font.
        let mut hidden = make_label(FontHandle(0), "A", 0.0);
        hidden.visible = false;
        assert!(measure_label_box(&hidden, &fonts).is_none());
        // Visible label whose font isn't loaded → None.
        let orphan = make_label(FontHandle(99), "A", 0.0);
        assert!(measure_label_box(&orphan, &fonts).is_none());
    }

    #[test]
    fn align_center_and_right_shift_the_anchor() {
        // Two 'A' glyphs, advance 10 each: rendered width = 20. A HUD label
        // (view == None) anchored at x = 100 keeps that x when left-aligned,
        // shifts left by half the width when centered, and by the full width
        // when right-aligned.
        let g = make_glyph(10, 12, 10.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let first_x = |align: TextAlign| {
            let mut l = make_label(FontHandle(0), "AA", 100.0);
            l.align = align;
            build_text_calls(
                core::slice::from_ref(&l),
                &fonts,
                0.0,
                0.0,
                &no_clips(),
                &no_layers(),
            )[0]
            .vertices[0]
                .pos[0]
        };
        assert!((first_x(TextAlign::Left) - 100.0).abs() < 1e-4);
        assert!((first_x(TextAlign::Center) - 90.0).abs() < 1e-4);
        assert!((first_x(TextAlign::Right) - 80.0).abs() < 1e-4);
    }

    #[test]
    fn clip_band_scissors_the_call() {
        // A label registered in `clips` gets its call scissored to the band,
        // mapped through the overlay transform to window space. At the reference
        // viewport the overlay is identity, so the scissor equals the band.
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let mut label = make_label(FontHandle(0), "A", 0.0);
        label.asset_id = AssetId(7);
        let mut clips = ClipRects::new();
        let band = [10.0, 20.0, 300.0, 40.0];
        clips.insert(AssetId(7), band);
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            1280.0,
            720.0,
            &clips,
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].clip_rect, Some(band));
        // A label absent from `clips` (asset_id 0) draws unclipped.
        let other = make_label(FontHandle(0), "A", 0.0);
        let unclipped = build_text_calls(
            core::slice::from_ref(&other),
            &fonts,
            1280.0,
            720.0,
            &clips,
            &no_layers(),
        );
        assert_eq!(unclipped[0].clip_rect, None);
    }

    #[test]
    fn band_to_window_maps_through_the_overlay() {
        // At a 2x viewport a reference band is scaled and recentered.
        let overlay = OverlayTransform::from_viewport([2560.0, 1440.0]);
        let mapped = band_to_window(&overlay, [640.0, 360.0, 100.0, 50.0]);
        // forward(640,360) = center = (1280,720); the band doubles in size.
        assert!((mapped[0] - 1280.0).abs() < 1e-3, "x={}", mapped[0]);
        assert!((mapped[1] - 720.0).abs() < 1e-3, "y={}", mapped[1]);
        assert!((mapped[2] - 200.0).abs() < 1e-3, "w={}", mapped[2]);
        assert!((mapped[3] - 100.0).abs() < 1e-3, "h={}", mapped[3]);
    }

    #[test]
    fn fit_bottom_and_cover_map_view_owned_labels() {
        // A view-owned label maps its anchor through its `fit` transform. At a
        // 4:3 viewport (taller than the 16:9 reference) Bottom pushes the label
        // below plain Fit, and Cover scales it up, so each branch yields a
        // distinct origin.
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[('A', g)]));
        let vp = (1024.0, 768.0);
        let first_y = |fit: SpriteFit| {
            let mut l = make_label(FontHandle(0), "A", 100.0);
            l.y = 600.0;
            l.screen = Some(AssetId(5));
            l.fit = fit;
            build_text_calls(
                core::slice::from_ref(&l),
                &fonts,
                vp.0,
                vp.1,
                &no_clips(),
                &no_layers(),
            )[0]
            .vertices[0]
                .pos[1]
        };
        let fit_y = first_y(SpriteFit::Fit);
        let bottom_y = first_y(SpriteFit::Bottom);
        let cover_y = first_y(SpriteFit::Cover);
        assert!(bottom_y > fit_y, "bottom={bottom_y} fit={fit_y}");
        assert!(
            (cover_y - fit_y).abs() > 1e-3,
            "cover={cover_y} fit={fit_y}"
        );
    }

    #[test]
    fn missing_glyph_falls_back_to_space_advance() {
        // A code point with no metric advances the cursor by the space glyph's
        // advance, so an unknown character still occupies layout width.
        let space = make_glyph(0, 0, 7.0);
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[(' ', space), ('A', g)]));
        // '?' has no metric; it consumes one space advance before 'A'.
        let label = make_label(FontHandle(0), "?A", 0.0);
        let calls = build_text_calls(
            core::slice::from_ref(&label),
            &fonts,
            0.0,
            0.0,
            &no_clips(),
            &no_layers(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4); // only 'A' draws a quad
        assert!((calls[0].vertices[0].pos[0] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn measure_uses_space_advance_for_missing_glyphs() {
        // text_advance_width (via measure_label_box) also substitutes the space
        // advance for an unknown glyph, keeping layout width stable.
        let space = make_glyph(0, 0, 7.0);
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = FontSet::default();
        fonts.insert(FontHandle(0), make_font(&[(' ', space), ('A', g)]));
        let known = make_label(FontHandle(0), "A", 0.0);
        let with_missing = make_label(FontHandle(0), "?A", 0.0);
        let wk = measure_label_box(&known, &fonts).unwrap().w;
        let wm = measure_label_box(&with_missing, &fonts).unwrap().w;
        // The '?' contributes exactly one space advance (7) of extra width.
        assert!((wm - wk - 7.0).abs() < 1e-4, "wk={wk} wm={wm}");
    }
}
