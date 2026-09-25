//! Pointer input over a laid-out text area: a press places the caret or grabs
//! a scrollbar, a held button drags either, and the wheel scrolls.

use super::layout::{Geometry, scroll_on_track};
use super::{Scroll, TextArea};

// Scroll-delta units per line: three lines per wheel notch on the backends
// that report notches, and proportional on a trackpad.
const WHEEL_UNITS_PER_LINE: f32 = 20.0 / 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bar {
    Vertical,
    Horizontal,
}

impl TextArea {
    // A press at (x, y) on the area laid out as `g`, at time `now` in seconds.
    // `false` when it missed the area. The gutter places the caret at the start
    // of the line it numbers.
    pub(crate) fn press_at(
        &mut self,
        g: &Geometry,
        x: f32,
        y: f32,
        extend: bool,
        now: f64,
    ) -> bool {
        if !g.contains(x, y) {
            return false;
        }
        if let Some(bar) = g.bar_at(self, x, y) {
            self.release();
            self.bar = Some(bar);
            self.drag_bar(bar, g, x, y);
            return true;
        }
        let pos = g.pos_at(self, x, y);
        self.press(pos, extend, now);
        true
    }

    // Follow the pointer each frame: while the button is held a press keeps
    // dragging its selection or scrollbar, and letting go ends it.
    pub(crate) fn pointer(&mut self, g: &Geometry, x: f32, y: f32, held: bool) {
        if !held {
            self.bar = None;
            self.release();
            return;
        }
        match self.bar {
            Some(bar) => self.drag_bar(bar, g, x, y),
            None if self.dragging() => {
                let pos = g.pos_at(self, x, y);
                self.drag_to(pos);
            }
            None => {}
        }
    }

    // Whether a press is still being dragged (text or scrollbar).
    pub(crate) fn pointer_busy(&self) -> bool {
        self.dragging() || self.bar.is_some()
    }

    fn drag_bar(&mut self, bar: Bar, g: &Geometry, x: f32, y: f32) {
        let view = g.view();
        let scroll = self.scroll();
        let next = match bar {
            Bar::Vertical => {
                let t = g.v_track();
                let top = scroll_on_track(t[3], y - t[1], view.rows, self.line_count());
                Scroll { top, ..scroll }
            }
            Bar::Horizontal => {
                let t = g.h_track();
                let left = scroll_on_track(t[2], x - t[0], view.cols, self.widest() + 1);
                Scroll { left, ..scroll }
            }
        };
        self.set_scroll(next);
    }

    // A wheel movement of `delta` scroll units (positive moves the text up),
    // across when `horizontal`. Fractions of a line carry to the next event.
    pub(crate) fn wheel(&mut self, delta: f32, horizontal: bool) {
        self.wheel_carry += delta;
        let steps = (self.wheel_carry / WHEEL_UNITS_PER_LINE).trunc();
        self.wheel_carry -= steps * WHEEL_UNITS_PER_LINE;
        let steps = steps as isize;
        if horizontal {
            self.scroll_by(0, steps);
        } else {
            self.scroll_by(steps, 0);
        }
    }
}
