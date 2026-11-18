// A vertical scroll over content taller than its view: how far it scrolls, the
// scrollbar along its right edge, and the scroll a dragged thumb or a revealed
// span asks for. Lists and grids share it.

// Width of the scrollbar strip along a scrolling view's right edge.
pub(crate) const SCROLLBAR_W: f32 = 6.0;
const MIN_THUMB: f32 = 18.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScrollView {
    // The view's `[x, y, width, height]`.
    pub(crate) rect: [f32; 4],
    // Height of everything the view scrolls over.
    pub(crate) content: f32,
    // Pixels scrolled from the top.
    pub(crate) scroll: f32,
}

impl ScrollView {
    pub(crate) fn max_scroll(&self) -> f32 {
        (self.content - self.rect[3]).max(0.0)
    }

    pub(crate) fn scrolls(&self) -> bool {
        self.max_scroll() > 0.0
    }

    // Width left for the content, short of the scrollbar when one shows.
    pub(crate) fn inner_width(&self) -> f32 {
        if self.scrolls() {
            (self.rect[2] - SCROLLBAR_W).max(0.0)
        } else {
            self.rect[2]
        }
    }

    // The scrollbar's track and thumb, or `None` when everything shows.
    pub(crate) fn scrollbar(&self) -> Option<([f32; 4], [f32; 4])> {
        if !self.scrolls() {
            return None;
        }
        let [x, y, w, h] = self.rect;
        let track = [x + w - SCROLLBAR_W, y, SCROLLBAR_W, h];
        let len = (h * h / self.content).clamp(MIN_THUMB.min(h), h);
        let at = (h - len) * (self.scroll / self.max_scroll()).clamp(0.0, 1.0);
        Some((track, [track[0], y + at, SCROLLBAR_W, len]))
    }

    // The scroll that puts the thumb's top `thumb_top` down its track.
    pub(crate) fn scroll_for_thumb(&self, thumb_top: f32) -> f32 {
        let Some((track, thumb)) = self.scrollbar() else {
            return 0.0;
        };
        let travel = (track[3] - thumb[3]).max(1.0);
        ((thumb_top - track[1]) / travel).clamp(0.0, 1.0) * self.max_scroll()
    }

    // The scroll that shows the span from `top` down `height`, moving no
    // further than it must.
    pub(crate) fn revealing(&self, top: f32, height: f32) -> f32 {
        let scroll = if top < self.scroll {
            top
        } else if top + height > self.scroll + self.rect[3] {
            top + height - self.rect[3]
        } else {
            self.scroll
        };
        scroll.clamp(0.0, self.max_scroll())
    }

    // The scroll `delta` wheel pixels move to.
    pub(crate) fn wheeled(&self, delta: f32) -> f32 {
        (self.scroll - delta).clamp(0.0, self.max_scroll())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(content: f32) -> ScrollView {
        ScrollView {
            rect: [0.0, 0.0, 200.0, 100.0],
            content,
            scroll: 0.0,
        }
    }

    #[test]
    fn a_scrollbar_shows_only_while_content_overflows() {
        assert!(view(100.0).scrollbar().is_none());
        assert_eq!(view(100.0).inner_width(), 200.0);
        let mut v = view(400.0);
        assert_eq!(v.scrollbar().unwrap().1[3], 25.0, "a quarter shows");
        assert_eq!(v.inner_width(), 200.0 - SCROLLBAR_W);
        v.scroll = v.max_scroll();
        let (track, thumb) = v.scrollbar().unwrap();
        assert_eq!(thumb[1] + thumb[3], track[1] + track[3], "at the bottom");
        assert_eq!(v.scroll_for_thumb(track[1]), 0.0);
        assert_eq!(v.scroll_for_thumb(track[1] + 75.0), v.max_scroll());
    }

    #[test]
    fn revealing_moves_only_as_far_as_it_must() {
        let mut v = view(400.0);
        assert_eq!(v.revealing(200.0, 20.0), 120.0, "to the bottom edge");
        v.scroll = 120.0;
        assert_eq!(v.revealing(40.0, 20.0), 40.0, "to the top edge");
        assert_eq!(v.revealing(150.0, 20.0), 120.0, "already in view");
        assert_eq!(v.wheeled(-1000.0), v.max_scroll());
    }
}
