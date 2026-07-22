// src/editor/theme.rs
//
// The editor chrome's shared visual theme: one palette and metric set so the
// top bar and every floating panel read as a single surface. Panels keep local
// tints only for colours that are truly their own (e.g. the delete-red row);
// anything that repeats across panels lives here.

// The HUD font is 20px; editor labels draw a step smaller for a denser, cleaner
// read. `TEXT_HALF` is half the scaled line height, used to vertically center a
// label in a row or button.
pub(crate) const TEXT_SCALE: f32 = 0.85;
pub(crate) const TEXT_HALF: f32 = 10.0 * TEXT_SCALE;

// Corner rounding: panels get the large radius, buttons / row highlights /
// dropdowns the small one.
pub(crate) const PANEL_RADIUS: f32 = 10.0;
pub(crate) const CONTROL_RADIUS: f32 = 5.0;

// The unified chrome surface: the top bar and every panel background share this
// tint, title bars included (no separate title strip).
pub(crate) const CHROME_TINT: [f32; 4] = [0.11, 0.11, 0.14, 0.97];

// A hairline border framing every floating panel, a step lighter than the
// chrome surface so the panel reads as a raised edge against the scene behind.
pub(crate) const PANEL_BORDER_WIDTH: f32 = 1.5;
pub(crate) const PANEL_BORDER_TINT: [f32; 4] = [0.30, 0.32, 0.40, 1.0];

// Row / option interaction states.
pub(crate) const HOVER_TINT: [f32; 4] = [0.24, 0.27, 0.36, 0.9];
pub(crate) const SELECTED_TINT: [f32; 4] = [0.18, 0.28, 0.46, 1.0];

// Neutral button chip and the accent for an active / open control.
pub(crate) const BUTTON_TINT: [f32; 4] = [0.19, 0.20, 0.25, 1.0];
pub(crate) const ACCENT_TINT: [f32; 4] = [0.26, 0.42, 0.66, 1.0];

// Label colours: normal body text, dimmed secondary text, and headings.
pub(crate) const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
pub(crate) const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
pub(crate) const HEADING: [f32; 3] = [0.93, 0.94, 0.97];

// Console log-line colours by severity, plus the dimmed echo of a submitted
// command line.
pub(crate) const LOG_INFO: [f32; 3] = LABEL;
pub(crate) const LOG_WARN: [f32; 3] = [0.95, 0.78, 0.45];
pub(crate) const LOG_ERROR: [f32; 3] = [0.95, 0.55, 0.55];
pub(crate) const LOG_COMMAND: [f32; 3] = [0.58, 0.72, 0.95];

// How far a row's hover / selection highlight is inset from the row rect, so
// the highlight reads as a floating pill rather than a full-width band.
pub(crate) const HIGHLIGHT_INSET_X: f32 = 4.0;
pub(crate) const HIGHLIGHT_INSET_Y: f32 = 1.5;

// A row's hover / selection highlight: the row rect pulled in by the insets.
pub(crate) fn highlight_rect(rect: [f32; 4]) -> [f32; 4] {
    [
        rect[0] + HIGHLIGHT_INSET_X,
        rect[1] + HIGHLIGHT_INSET_Y,
        (rect[2] - 2.0 * HIGHLIGHT_INSET_X).max(0.0),
        (rect[3] - 2.0 * HIGHLIGHT_INSET_Y).max(0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlight_rect_insets_and_never_goes_negative() {
        let r = highlight_rect([10.0, 20.0, 100.0, 28.0]);
        assert_eq!(r[0], 10.0 + HIGHLIGHT_INSET_X);
        assert_eq!(r[1], 20.0 + HIGHLIGHT_INSET_Y);
        assert_eq!(r[2], 100.0 - 2.0 * HIGHLIGHT_INSET_X);
        assert_eq!(r[3], 28.0 - 2.0 * HIGHLIGHT_INSET_Y);
        // A degenerate rect clamps to zero size instead of inverting.
        let tiny = highlight_rect([0.0, 0.0, 2.0, 1.0]);
        assert_eq!((tiny[2], tiny[3]), (0.0, 0.0));
    }
}
