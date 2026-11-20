//! The editor chrome's shared visual theme: one palette and metric set so the
//! top bar and every floating panel read as a single surface. Panels keep local
//! tints only for colors that are truly their own (e.g. the delete-red row);
//! anything that repeats across panels lives here.

// The HUD font is 20px; editor labels draw a step smaller for a denser, cleaner
// read. `TEXT_HALF` is half the scaled line height, used to vertically center a
// label in a row or button.
pub(crate) const TEXT_SCALE: f32 = 0.85;
pub(crate) const TEXT_HALF: f32 = 10.0 * TEXT_SCALE;
// One line of text, top to bottom.
pub(crate) const LINE_H: f32 = TEXT_HALF * 2.0;
// A panel's title bar, which drags it.
pub(crate) const TITLE_H: f32 = 28.0;

// Corner rounding: panels get the large radius, buttons / row highlights /
// dropdowns the small one.
pub(crate) const PANEL_RADIUS: f32 = 10.0;
pub(crate) const CONTROL_RADIUS: f32 = 5.0;

// The unified chrome surface: the top bar and every panel background share this
// tint, title bars included (no separate title strip).
pub(crate) const CHROME_TINT: [f32; 4] = [0.11, 0.11, 0.14, 0.97];

// The start screen's sidebar tint, translucent over the preview it stands on.
pub(crate) const SIDEBAR_TINT: [f32; 4] = [0.07, 0.07, 0.10, 0.55];

// A hairline border framing every floating panel, a step lighter than the
// chrome surface so the panel reads as a raised edge against the scene behind.
pub(crate) const PANEL_BORDER_WIDTH: f32 = 1.5;
pub(crate) const PANEL_BORDER_TINT: [f32; 4] = [0.30, 0.32, 0.40, 1.0];

// The frame belongs to the surface it encloses, so it is drawn at that
// surface's opacity: a panel let down to a wash is framed by one too, rather
// than by a solid hairline that would read as a hard edge cut into the world.
pub(crate) const fn panel_border(surface_alpha: f32) -> [f32; 4] {
    let [r, g, b, a] = PANEL_BORDER_TINT;
    [r, g, b, a * surface_alpha]
}

// Row / option interaction states.
pub(crate) const HOVER_TINT: [f32; 4] = [0.24, 0.27, 0.36, 0.9];
pub(crate) const SELECTED_TINT: [f32; 4] = [0.18, 0.28, 0.46, 1.0];

// Neutral button chip and the accent for an active / open control.
pub(crate) const BUTTON_TINT: [f32; 4] = [0.19, 0.20, 0.25, 1.0];
// The well a code text area sits in.
pub(crate) const CODE_WELL_TINT: [f32; 4] = [0.07, 0.07, 0.09, 1.0];
// A panel's close button under the pointer.
pub(crate) const CLOSE_HOVER_TINT: [f32; 4] = [0.46, 0.26, 0.28, 1.0];
pub(crate) const ACCENT_TINT: [f32; 4] = [0.26, 0.42, 0.66, 1.0];

// A checkbox's box while on.
pub(crate) const CHECK_ON_TINT: [f32; 4] = [0.30, 0.66, 0.34, 1.0];

// Label colors: normal body text, dimmed secondary text, and headings.
pub(crate) const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
pub(crate) const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
pub(crate) const HEADING: [f32; 3] = [0.93, 0.94, 0.97];

// Console log-line colors by severity, plus the dimmed echo of a submitted
// command line.
pub(crate) const LOG_INFO: [f32; 3] = LABEL;
pub(crate) const LOG_WARN: [f32; 3] = [0.95, 0.78, 0.45];
pub(crate) const LOG_ERROR: [f32; 3] = [0.95, 0.55, 0.55];
pub(crate) const LOG_COMMAND: [f32; 3] = [0.58, 0.72, 0.95];
