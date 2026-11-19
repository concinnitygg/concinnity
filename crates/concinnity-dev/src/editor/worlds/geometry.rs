//! The Worlds panel's two presentations. Every metric a presentation differs by
//! lives in one `Metrics` value, and the window facts a presentation is
//! resolved against (its size, and the chrome floating over the top of the
//! frame) live in one `Layout`.

use crate::editor::ui::build::BODY_PAD;
use crate::editor::{hud, theme};

// Which presentation of the panel is up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    // The start screen: a sidebar docked down the window's left edge, over the
    // world it previews. It cannot be closed or dragged.
    Start,
    // The in-session switcher: one floating panel among the others.
    Session,
}

// Everything the two presentations differ by, in logical units.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Metrics {
    pub width: f32,
    pub pad: f32,
    // Height of the title bar, or zero for a presentation that carries no
    // heading (and therefore no close button either).
    pub title_h: f32,
    pub header_h: f32,
    pub status_h: f32,
    pub list_header_h: f32,
    pub row_h: f32,
    // Rows the floating switcher is sized to.
    pub rows: usize,
    // Text scale for every caption the panel draws.
    pub text_scale: f32,
    // The surface the panel's background is drawn on. The docked sidebar
    // stands over the world it previews, so it takes a wash rather than the
    // opaque chrome a floating panel sits behind.
    pub panel_tint: [f32; 4],
    // Whether the panel is docked to the window's left edge and spans its
    // height, rather than floating at a draggable origin.
    pub docked: bool,
}

pub(crate) const fn metrics(mode: Mode) -> Metrics {
    match mode {
        Mode::Session => Metrics {
            width: 340.0,
            pad: BODY_PAD,
            title_h: theme::TITLE_H,
            header_h: 32.0,
            status_h: theme::LINE_H + 6.0,
            list_header_h: 24.0,
            row_h: 26.0,
            rows: 12,
            text_scale: theme::TEXT_SCALE,
            panel_tint: theme::CHROME_TINT,
            docked: false,
        },
        Mode::Start => Metrics {
            width: 280.0,
            pad: 12.0,
            title_h: 0.0,
            header_h: 36.0,
            status_h: 44.0,
            list_header_h: 24.0,
            row_h: 34.0,
            rows: 0,
            text_scale: 1.0,
            panel_tint: theme::SIDEBAR_TINT,
            docked: true,
        },
    }
}

impl Metrics {
    // Whether the presentation draws a title bar, and with it a close button.
    // The sidebar has neither: closing it would strand the session on an empty
    // scene with no way back.
    pub(crate) fn has_title_bar(&self) -> bool {
        self.title_h > 0.0
    }
}

// The panel resolved against this frame's window: which presentation is up, its
// metrics, and the window facts the docked sidebar is sized and offset by.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    pub mode: Mode,
    m: Metrics,
    vp: [f32; 2],
    // Window chrome floating over the top of the frame, in logical units. Only
    // the docked sidebar meets it; the switcher hangs below the top bar.
    inset: f32,
}

impl Layout {
    pub(crate) fn new(mode: Mode, vp: [f32; 2], inset: f32) -> Self {
        Self {
            mode,
            m: metrics(mode),
            vp,
            inset: inset.max(0.0),
        }
    }

    pub(crate) fn metrics(&self) -> &Metrics {
        &self.m
    }

    // The panel's footprint. The sidebar spans the window it is docked to; the
    // switcher is sized by the rows it shows.
    pub(crate) fn size(&self) -> [f32; 2] {
        if self.m.docked {
            [self.m.width, self.vp[1]]
        } else {
            let chrome = self.m.header_h + self.m.status_h + self.m.list_header_h;
            [
                self.m.width,
                crate::editor::ui::panel_height(chrome + self.m.rows as f32 * self.m.row_h),
            ]
        }
    }

    // Where the panel sits until the user drags it. The sidebar docks to the
    // window's left edge; the switcher hangs below the top bar, centered.
    pub(crate) fn default_origin(&self) -> [f32; 2] {
        if self.m.docked {
            [0.0, 0.0]
        } else {
            [(self.vp[0] - self.m.width) * 0.5, hud::body_top() + 24.0]
        }
    }

    // How far the content sits below the panel's top edge, past its padding.
    // The sidebar clears the window chrome floating over the frame, so its
    // first control lands neither under the OS window buttons nor against the
    // window's edge. The switcher's title bar already does this for it.
    pub(crate) fn content_offset(&self) -> f32 {
        if self.m.docked { self.inset } else { 0.0 }
    }

    // The window the panel leaves to the world behind it: everything right of
    // the docked sidebar. The floating switcher stands over a session's
    // viewport rather than beside it, so there it is the whole window.
    pub(crate) fn preview_rect(&self) -> [f32; 4] {
        if self.m.docked {
            [
                self.m.width,
                0.0,
                (self.vp[0] - self.m.width).max(0.0),
                self.vp[1],
            ]
        } else {
            [0.0, 0.0, self.vp[0], self.vp[1]]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VP: [f32; 2] = [1280.0, 720.0];

    // The start screen is a full-height sidebar docked to the window's left
    // edge: narrower than the switcher, as tall as the window it is drawn in,
    // and with the render owning everything to its right.
    #[test]
    fn the_sidebar_docks_to_the_left_edge_at_full_height() {
        let (start, session) = (metrics(Mode::Start), metrics(Mode::Session));
        assert!(start.width < session.width, "narrower than the switcher");
        assert!(start.row_h > session.row_h);
        assert!(start.text_scale > session.text_scale);
        assert!(start.docked && !session.docked);

        let l = Layout::new(Mode::Start, VP, 0.0);
        assert_eq!(l.default_origin(), [0.0, 0.0]);
        assert_eq!(l.size(), [start.width, VP[1]]);
        assert_eq!(l.preview_rect()[0], start.width, "the render owns the rest");

        // The switcher keeps its anchor below the top bar, centered.
        let s = Layout::new(Mode::Session, VP, 0.0);
        let o = s.default_origin();
        assert!(o[1] >= hud::body_top());
        assert!((o[0] + s.size()[0] * 0.5 - VP[0] * 0.5).abs() < 0.5);
    }

    // A top inset pushes only the sidebar's content down; the switcher floats
    // clear of the window's top edge, and its footprint is fixed whatever the
    // window's height.
    #[test]
    fn the_top_inset_moves_only_the_sidebar_content() {
        assert_eq!(Layout::new(Mode::Start, VP, 28.0).content_offset(), 28.0);
        assert_eq!(Layout::new(Mode::Session, VP, 28.0).content_offset(), 0.0);
        let a = Layout::new(Mode::Session, VP, 0.0);
        let b = Layout::new(Mode::Session, [1280.0, 2400.0], 0.0);
        assert_eq!(a.size(), b.size());
    }
}
