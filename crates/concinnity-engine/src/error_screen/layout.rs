// LabelPlacement for the error screen's message lines and the Quit button, plus the
// button's hit rect.
//
// Everything is laid out in window pixels against the live drawable size, so
// the screen re-centres when the window is resized. The labels are deliberately
// not screen-owned: a `Screen` maps through the reference-canvas overlay
// transform, and this path has no world to own one.

use crate::assets::{TextAlign, TextLabel};
use crate::ecs::FontHandle;
use crate::gfx::text::{LoadedFont, measure_label_box};
use std::collections::HashMap;

// On-screen heights, in window pixels. The header leads, the Quit line matches
// the menu options' size, and the message draws a step smaller than Quit.
const HEADER_PX: f32 = 68.0;
const MESSAGE_PX: f32 = 32.0;
const QUIT_PX: f32 = 52.0;

// Vertical anchors as a fraction of window height. `MESSAGE_Y` is where the
// first message line starts; further lines stack below it.
const HEADER_Y: f32 = 0.26;
const MESSAGE_Y: f32 = 0.42;
// Where Quit sits when the message is short. A message long enough to reach it
// pushes it down instead (see `build`).
const QUIT_Y: f32 = 0.64;

// Gap between stacked message lines, as a fraction of the message height.
const LINE_GAP: f32 = 0.35;

// Clearance kept between the last message line and the Quit button when a long
// message pushes it down, in multiples of the message height.
const QUIT_CLEARANCE: f32 = 1.5;

// Fraction of the window width the message wraps within, so a long path breaks
// across lines instead of running off both edges.
const WRAP_FRACTION: f32 = 0.8;

// Most lines the message draws before ellipsis, so no message can fill the
// screen and push the Quit button out of view.
const MESSAGE_MAX_LINES: u32 = 6;

// Padding around the Quit text that the click test accepts, so the button has a
// comfortable target rather than the exact glyph bounds.
const QUIT_HIT_PAD: f32 = 24.0;

// Linear-space RGB, like every other `TextLabel` colour.
const HEADER_TEXT: &str = "Error";
const HEADER_COLOR: [f32; 3] = [0.85, 0.09, 0.07];
const TEXT_COLOR: [f32; 3] = [0.95, 0.78, 0.12];
const QUIT_COLOR: [f32; 3] = [0.82, 0.82, 0.82];
const QUIT_HOVER_COLOR: [f32; 3] = [1.0, 0.85, 0.3];

pub(super) struct Layout {
    pub(super) labels: Vec<TextLabel>,
    // Quit button bounds in window pixels: [x, y, width, height].
    pub(super) quit_rect: [f32; 4],
}

impl Layout {
    pub(super) fn quit_hit(&self, x: f32, y: f32) -> bool {
        let [rx, ry, rw, rh] = self.quit_rect;
        x >= rx && x <= rx + rw && y >= ry && y <= ry + rh
    }

    // The "Error" header, which `build` always emits first.
    #[cfg(test)]
    fn header_label(&self) -> &TextLabel {
        self.labels.first().expect("the header is always present")
    }

    // The Quit line, which `build` always appends last.
    #[cfg(test)]
    fn quit_label(&self) -> &TextLabel {
        self.labels.last().expect("the Quit line is always present")
    }

    // The message lines, which sit between the header and the Quit line.
    #[cfg(test)]
    fn message_labels(&self) -> &[TextLabel] {
        &self.labels[1..self.labels.len() - 1]
    }
}

// Lay the screen out for the current drawable size. `hovered` tints the Quit
// line, and is computed from the previous frame's rect by the caller.
pub(super) fn build(
    message: &str,
    win_w: f32,
    win_h: f32,
    fonts: &HashMap<FontHandle, LoadedFont>,
    handle: FontHandle,
    hovered: bool,
) -> Layout {
    let size_px = fonts
        .get(&handle)
        .map(|f| f.size_px)
        .unwrap_or(1.0)
        .max(1.0);

    let mut labels: Vec<TextLabel> = vec![TextLabel {
        font: Some(handle),
        content: HEADER_TEXT.to_string(),
        x: win_w * 0.5,
        y: win_h * HEADER_Y,
        color: HEADER_COLOR,
        scale: HEADER_PX / size_px,
        align: TextAlign::Center,
        visible: true,
        ..TextLabel::default()
    }];

    // One label per authored line, each centred on its own. A single label
    // holding the newlines would centre the block and left-align the lines
    // inside it, which reads as ragged for a two-line error.
    let mut y = win_h * MESSAGE_Y;
    for line in message.lines() {
        let label = TextLabel {
            font: Some(handle),
            content: line.to_string(),
            x: win_w * 0.5,
            y,
            color: TEXT_COLOR,
            scale: MESSAGE_PX / size_px,
            align: TextAlign::Center,
            wrap_width: win_w * WRAP_FRACTION,
            max_lines: MESSAGE_MAX_LINES,
            visible: true,
            ..TextLabel::default()
        };
        // Advance by what the line actually occupies, so a wrapped path pushes
        // the following lines down instead of drawing over them.
        let height = measure_label_box(&label, fonts)
            .map(|b| b.h)
            .unwrap_or(MESSAGE_PX);
        y += height * (1.0 + LINE_GAP);
        labels.push(label);
    }

    // Quit keeps its resting position for a short message, and is pushed below
    // a long one rather than being drawn over it.
    let quit_y = (win_h * QUIT_Y).max(y + MESSAGE_PX * QUIT_CLEARANCE);

    let quit_label = TextLabel {
        font: Some(handle),
        content: "Quit".to_string(),
        x: win_w * 0.5,
        y: quit_y,
        color: if hovered {
            QUIT_HOVER_COLOR
        } else {
            QUIT_COLOR
        },
        scale: QUIT_PX / size_px,
        align: TextAlign::Center,
        visible: true,
        ..TextLabel::default()
    };

    // Measure the Quit line with the same metrics that draw it, so the click
    // target tracks the text rather than a guessed rectangle.
    let quit_rect = match measure_label_box(&quit_label, fonts) {
        Some(b) => [
            quit_label.x - b.w * 0.5 - QUIT_HIT_PAD,
            quit_label.y - b.top_inset - QUIT_HIT_PAD,
            b.w + QUIT_HIT_PAD * 2.0,
            b.h + QUIT_HIT_PAD * 2.0,
        ],
        None => [0.0, 0.0, 0.0, 0.0],
    };

    labels.push(quit_label);
    Layout { labels, quit_rect }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fonts() -> (HashMap<FontHandle, LoadedFont>, FontHandle) {
        let font = super::super::font::load().expect("embedded font decodes");
        (font.fonts, super::super::font::HANDLE)
    }

    #[test]
    fn every_line_is_horizontally_centered() {
        let (fonts, handle) = fonts();
        let layout = build("data file missing", 1280.0, 720.0, &fonts, handle, false);

        assert_eq!(layout.labels.len(), 3, "header, one message line, Quit");
        for label in &layout.labels {
            assert_eq!(label.x, 640.0, "centered on the window");
            assert_eq!(label.align, TextAlign::Center);
            // `centered` auto-scales to fill the viewport and ignores `scale`,
            // which is not what this screen wants.
            assert!(!label.centered);
        }
        assert!(
            layout.quit_label().scale > layout.message_labels()[0].scale,
            "the Quit line draws larger than the message"
        );
    }

    // The header leads the screen: above everything, larger than everything,
    // and the only red element.
    #[test]
    fn the_header_leads_in_larger_red_text() {
        let (fonts, handle) = fonts();
        let layout = build("data file missing", 1280.0, 720.0, &fonts, handle, false);
        let header = layout.header_label();

        assert_eq!(header.content, "Error");
        assert_eq!(header.color, HEADER_COLOR);
        assert!(
            header.color[0] > header.color[1] && header.color[0] > header.color[2],
            "the header reads red: {:?}",
            header.color
        );
        assert!(
            header.scale > layout.message_labels()[0].scale,
            "the header draws larger than the message"
        );
        assert!(
            header.y < layout.message_labels()[0].y,
            "the header sits above the message"
        );
    }

    // The message is yellow, which is what separates it from the red header
    // above it and the neutral Quit line below.
    #[test]
    fn the_message_draws_yellow() {
        let (fonts, handle) = fonts();
        let layout = build("boom\nsecond line", 1280.0, 720.0, &fonts, handle, false);

        for line in layout.message_labels() {
            assert_eq!(line.color, TEXT_COLOR);
            let [r, g, b] = line.color;
            assert!(
                r > 0.5 && g > 0.5 && b < 0.3,
                "the message reads yellow: {:?}",
                line.color
            );
        }
        assert_ne!(layout.header_label().color, TEXT_COLOR);
        assert_ne!(layout.quit_label().color, TEXT_COLOR);
    }

    // A message long enough to reach the button's resting position pushes it
    // down instead of being drawn over by it.
    #[test]
    fn a_long_message_pushes_quit_below_itself() {
        let (fonts, handle) = fonts();
        let many_lines = (0..6).map(|i| format!("line {i}")).collect::<Vec<_>>();
        let layout = build(&many_lines.join("\n"), 1280.0, 720.0, &fonts, handle, false);

        let last = layout.message_labels().last().expect("message lines exist");
        let last_box = measure_label_box(last, &fonts).expect("the last line measures");
        assert!(
            layout.quit_label().y > last.y + last_box.h,
            "Quit at {} overlaps the last message line ending at {}",
            layout.quit_label().y,
            last.y + last_box.h
        );
    }

    // Each authored line gets its own centred label. A single label holding the
    // newlines would centre the block and left-align its lines, which leaves a
    // short path hanging off the left of the sentence above it.
    #[test]
    fn each_message_line_is_centered_independently() {
        let (fonts, handle) = fonts();
        let layout = build(
            "Failed to find the data blob:\n.concinnity/data/0",
            1280.0,
            720.0,
            &fonts,
            handle,
            false,
        );

        let lines = layout.message_labels();
        assert_eq!(lines.len(), 2, "one label per authored line");
        assert!(!lines[0].content.contains('\n'));
        for line in lines {
            assert_eq!(line.x, 640.0, "each line centres on the window");
        }
        assert!(lines[1].y > lines[0].y, "the second line stacks below");
        // Stacked without overlapping: the gap clears the first line's box.
        let first = measure_label_box(&lines[0], &fonts).expect("first line measures");
        assert!(lines[1].y >= lines[0].y + first.h);
    }

    #[test]
    fn quit_rect_surrounds_the_quit_label() {
        let (fonts, handle) = fonts();
        let layout = build("boom", 1280.0, 720.0, &fonts, handle, false);
        let quit = layout.quit_label();
        let [rx, ry, rw, rh] = layout.quit_rect;

        assert!(rw > 0.0 && rh > 0.0, "the button has real bounds");
        assert!(layout.quit_hit(quit.x, quit.y), "the label origin hits");
        assert!(
            layout.quit_hit(rx + 1.0, ry + 1.0),
            "the top-left corner hits"
        );
        assert!(
            !layout.quit_hit(rx - 1.0, ry + rh * 0.5),
            "left of the button misses"
        );
        assert!(
            !layout.quit_hit(rx + rw * 0.5, ry - 1.0),
            "above the button misses"
        );
        assert!(
            !layout.quit_hit(rx + rw + 1.0, ry + rh * 0.5),
            "right of it misses"
        );
    }

    #[test]
    fn layout_follows_the_window_size() {
        let (fonts, handle) = fonts();
        let small = build("boom", 800.0, 600.0, &fonts, handle, false);
        let large = build("boom", 1920.0, 1080.0, &fonts, handle, false);

        assert_eq!(small.message_labels()[0].x, 400.0);
        assert_eq!(large.message_labels()[0].x, 960.0);
        assert!(
            large.quit_rect[1] > small.quit_rect[1],
            "the button tracks height"
        );
        // The wrap width follows the window so a long path re-flows on resize.
        assert!(large.message_labels()[0].wrap_width > small.message_labels()[0].wrap_width);
    }

    #[test]
    fn hovering_retints_only_the_quit_line() {
        let (fonts, handle) = fonts();
        let plain = build("boom", 1280.0, 720.0, &fonts, handle, false);
        let hovered = build("boom", 1280.0, 720.0, &fonts, handle, true);

        assert_eq!(
            plain.message_labels()[0].color,
            hovered.message_labels()[0].color
        );
        assert_ne!(plain.quit_label().color, hovered.quit_label().color);
        assert_eq!(hovered.quit_label().color, QUIT_HOVER_COLOR);
    }

    // A long path is exactly what the missing-data message carries, so it must
    // wrap and stay bounded rather than run off the window.
    #[test]
    fn a_long_message_stays_bounded() {
        let (fonts, handle) = fonts();
        let long = "no compiled world data under ".to_string() + &"very/long/path/".repeat(40);
        let layout = build(&long, 1280.0, 720.0, &fonts, handle, false);

        let label = &layout.message_labels()[0];
        assert!(label.wrap_width > 0.0 && label.wrap_width < 1280.0);
        assert_eq!(label.max_lines, MESSAGE_MAX_LINES);
        let measured = measure_label_box(label, &fonts).expect("the message measures");
        assert!(
            measured.w <= label.wrap_width + 1.0,
            "wrapped width {} exceeds the wrap bound {}",
            measured.w,
            label.wrap_width
        );
    }
}
