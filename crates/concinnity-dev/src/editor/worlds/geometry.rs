// src/editor/worlds/geometry.rs
//
// The Worlds panel's rects and its hit test, for both presentations. Every
// metric a presentation differs by lives in one `Metrics` value, and the window
// facts a presentation is resolved against (its size, and the chrome floating
// over the top of the frame) live in one `Layout`, so the panel is laid out and
// hit-tested by the same arithmetic whichever one is up.

use super::{POOL, WorldsAction, WorldsView};
use crate::editor::hud;
use crate::editor::theme;
use crate::editor::widget::{self, point_in};

// Which presentation of the panel is up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    // The start screen: a sidebar docked down the window's left edge, over the
    // world it previews. It cannot be closed or dragged.
    Start,
    // The in-session switcher: one floating panel among the others.
    Session,
}

// Everything the two presentations differ by. Logical units throughout; every
// rect derives from the panel origin `o`, so a dragged switcher moves as one
// piece.
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
    // Rows the element pool can back; the window's height may show fewer.
    pub max_rows: usize,
    // Text scale for every caption the panel draws.
    pub text_scale: f32,
    // Row caption budget, leaving the triple-dot its reserved slot.
    pub max_row_chars: usize,
    // The surface the panel's background is drawn on. The docked sidebar
    // stands over the world it previews, so it takes a wash rather than the
    // opaque chrome a floating panel sits behind.
    pub panel_tint: [f32; 4],
    // Whether the panel is docked to the window's left edge and spans its
    // height, rather than floating at a draggable origin.
    pub docked: bool,
    // Whether the row menu offers Open. The switcher opens on the row click
    // itself, so there it would say twice what a click already does.
    pub menu_opens: bool,
}

const SCROLLBAR_W: f32 = 5.0;
// Rows the floating switcher shows; its height is sized to exactly these.
const SESSION_ROWS: usize = 12;
// The triple-dot button, sized as the Assets panel's is.
const DOT_SZ: f32 = 20.0;
// The floating row menu.
const MENU_W: f32 = 132.0;
const MENU_ROW_H: f32 = 26.0;

pub(crate) const fn metrics(mode: Mode) -> Metrics {
    match mode {
        Mode::Session => Metrics {
            width: 340.0,
            pad: 10.0,
            title_h: widget::TITLE_H,
            header_h: 32.0,
            status_h: widget::LINE_H + 6.0,
            list_header_h: 24.0,
            row_h: 26.0,
            max_rows: SESSION_ROWS,
            text_scale: theme::TEXT_SCALE,
            max_row_chars: 32,
            panel_tint: theme::CHROME_TINT,
            docked: false,
            menu_opens: false,
        },
        Mode::Start => Metrics {
            width: 280.0,
            pad: 12.0,
            title_h: 0.0,
            header_h: 36.0,
            status_h: 44.0,
            list_header_h: 24.0,
            row_h: 34.0,
            max_rows: POOL,
            text_scale: 1.0,
            max_row_chars: 22,
            panel_tint: theme::SIDEBAR_TINT,
            docked: true,
            menu_opens: true,
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

    // Height of a header control (the name field, New, and the row chips).
    fn control_h(&self) -> f32 {
        self.header_h - 8.0
    }

    // Half the scaled line height, for centering a caption in a row or chip.
    pub(crate) fn text_half(&self) -> f32 {
        10.0 * self.text_scale
    }

    // How many scaled lines of text fit in a box `h` pixels tall.
    pub(crate) fn lines_in(&self, h: f32) -> u32 {
        ((h / (2.0 * self.text_half())).floor() as u32).max(1)
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
        match self.m.docked {
            true => [self.m.width, self.vp[1]],
            false => [
                self.m.width,
                self.chrome_h() + self.m.max_rows as f32 * self.m.row_h + self.m.pad,
            ],
        }
    }

    // Where the panel sits until the user drags it. The sidebar docks to the
    // window's left edge; the switcher hangs below the top bar, centred.
    pub(crate) fn default_origin(&self) -> [f32; 2] {
        match self.m.docked {
            true => [0.0, 0.0],
            false => [(self.vp[0] - self.m.width) * 0.5, hud::body_top() + 24.0],
        }
    }

    // The background sprite. The docked sidebar bleeds its rounding off the
    // window's left, top, and bottom edges, so only the edge facing the render
    // is rounded and the fill reaches the very top of the window.
    pub(crate) fn panel_rect(&self, o: [f32; 2]) -> [f32; 4] {
        let s = self.size();
        match self.m.docked {
            true => {
                let r = theme::PANEL_RADIUS;
                [o[0] - r, o[1] - r, s[0] + r, s[1] + 2.0 * r]
            }
            false => widget::outer_rect(o, s),
        }
    }

    // The window the panel leaves to the world behind it: everything right of
    // the docked sidebar. The floating switcher stands over a session's
    // viewport rather than beside it, so there it is the whole window.
    pub(crate) fn preview_rect(&self) -> [f32; 4] {
        match self.m.docked {
            true => [
                self.m.width,
                0.0,
                (self.vp[0] - self.m.width).max(0.0),
                self.vp[1],
            ],
            false => [0.0, 0.0, self.vp[0], self.vp[1]],
        }
    }

    // The top of the panel's interactive content.
    fn content_top(&self, o: [f32; 2]) -> f32 {
        o[1] + self.content_offset()
    }

    // How far the content sits below the panel's top edge. The sidebar clears
    // the window chrome floating over the frame and then its own padding, so
    // the New button and the first row land neither under the OS window buttons
    // nor against the window's edge. The switcher's title bar already does this
    // for it.
    fn content_offset(&self) -> f32 {
        match self.m.docked {
            true => self.inset + self.m.pad,
            false => 0.0,
        }
    }

    // Height of the chrome above the row list.
    fn chrome_h(&self) -> f32 {
        self.m.title_h + self.m.header_h + self.m.status_h + self.m.list_header_h
    }

    // Rows the panel offers this frame: what its height has room for, capped by
    // the element pool.
    pub(crate) fn rows(&self) -> usize {
        let room = self.size()[1] - self.content_offset() - self.chrome_h() - self.m.pad;
        let fits = (room / self.m.row_h).floor().max(0.0) as usize;
        fits.min(self.m.max_rows)
    }

    pub(crate) fn title_rect(&self, o: [f32; 2]) -> [f32; 4] {
        [o[0], self.content_top(o), self.m.width, self.m.title_h]
    }

    // The `+` that starts an untitled world: a square at the header's right
    // end, so the header carries one control and no typing.
    pub(crate) fn new_rect(&self, o: [f32; 2]) -> [f32; 4] {
        let side = self.m.control_h();
        [
            o[0] + self.m.width - self.m.pad - side,
            self.content_top(o) + self.m.title_h + 4.0,
            side,
            side,
        ]
    }

    // The status line, under the header.
    pub(crate) fn status_rect(&self, o: [f32; 2]) -> [f32; 4] {
        [
            o[0] + self.m.pad,
            self.content_top(o) + self.m.title_h + self.m.header_h,
            (self.m.width - 2.0 * self.m.pad).max(0.0),
            self.m.status_h - 4.0,
        ]
    }

    // The list's caption line, between the status line and the rows.
    pub(crate) fn list_header_pos(&self, o: [f32; 2]) -> [f32; 2] {
        [
            o[0] + self.m.pad,
            self.content_top(o) + self.m.title_h + self.m.header_h + self.m.status_h + 2.0,
        ]
    }

    fn list_top(&self, o: [f32; 2]) -> f32 {
        self.content_top(o) + self.chrome_h()
    }

    // Visible row `slot` (0-based within the window).
    pub(crate) fn row_rect(&self, o: [f32; 2], slot: usize) -> [f32; 4] {
        [
            o[0],
            self.list_top(o) + slot as f32 * self.m.row_h,
            self.m.width - SCROLLBAR_W - 2.0,
            self.m.row_h,
        ]
    }

    // The triple-dot button, in its own reserved slot at row `slot`'s right
    // end. The space is held on every row even though the dots only show on the
    // row offering them, so a caption never has to move to make room.
    pub(crate) fn dot_rect(&self, o: [f32; 2], slot: usize) -> [f32; 4] {
        let r = self.row_rect(o, slot);
        [
            r[0] + r[2] - self.m.pad - DOT_SZ,
            r[1] + (self.m.row_h - DOT_SZ) * 0.5,
            DOT_SZ,
            DOT_SZ,
        ]
    }

    // The row menu, floating off row `slot`: below it, or above when the panel
    // has no room below (the docked sidebar's last rows reach the window's
    // bottom edge). Returns (background, Open row, Delete row); the Open row is
    // empty in the switcher, where Delete then sits alone at the top.
    pub(crate) fn menu_rects(&self, o: [f32; 2], slot: usize) -> ([f32; 4], [f32; 4], [f32; 4]) {
        let opens = self.m.menu_opens;
        let h = MENU_ROW_H * if opens { 2.0 } else { 1.0 };
        let x = o[0] + self.m.width - MENU_W - SCROLLBAR_W - 2.0;
        let r = self.row_rect(o, slot);
        let below = r[1] + r[3];
        let top = match below + h <= o[1] + self.size()[1] {
            true => below,
            false => r[1] - h,
        };
        let open_h = if opens { MENU_ROW_H } else { 0.0 };
        (
            [x, top, MENU_W, h],
            [x, top, MENU_W, open_h],
            [x, top + open_h, MENU_W, MENU_ROW_H],
        )
    }

    // The scrollbar track, or `None` while the listing fits.
    pub(crate) fn scrollbar(&self, o: [f32; 2], total: usize, scroll: usize) -> Option<Bar> {
        let rows = self.rows();
        if total <= rows {
            return None;
        }
        let x = o[0] + self.m.width - SCROLLBAR_W;
        let top = self.list_top(o);
        let h = rows as f32 * self.m.row_h;
        let thumb_h = (h * rows as f32 / total as f32).max(18.0);
        let off = (h - thumb_h) * (scroll as f32 / (total - rows) as f32);
        Some(Bar {
            track: [x, top, SCROLLBAR_W, h],
            thumb: [x, top + off, SCROLLBAR_W, thumb_h],
            radius: SCROLLBAR_W * 0.5,
        })
    }

    // Whether the cursor is over the scrollable listing (for wheel routing).
    pub(crate) fn cursor_over_list(&self, mx: f32, my: f32, o: [f32; 2]) -> bool {
        let p = widget::outer_rect(o, self.size());
        mx >= p[0] && mx < p[0] + p[2] && my >= self.list_top(o) && my < p[1] + p[3]
    }
}

// The scrollbar's two rects.
pub(crate) struct Bar {
    pub track: [f32; 4],
    pub thumb: [f32; 4],
    pub radius: f32,
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means
// the click missed the panel entirely and belongs to whatever is behind it.
// In the switcher a title-bar press never reaches this (the shared routing
// intercepts it first); the sidebar has no drag, so its chrome is swallowed
// here like the rest.
pub(crate) fn hit_test(view: &WorldsView, mx: f32, my: f32, o: [f32; 2]) -> Option<WorldsAction> {
    let l = &view.layout;
    // An open row menu is modal over the panel: its rows pick, and every other
    // press dismisses it rather than reaching what is under it.
    if let Some(i) = view.menu {
        if let Some(slot) = view.slot_of(i) {
            let (_, open, delete) = l.menu_rects(o, slot);
            if point_in(mx, my, delete) {
                return Some(WorldsAction::Delete(i));
            }
            if l.metrics().menu_opens && point_in(mx, my, open) {
                return Some(WorldsAction::Open(i));
            }
        }
        return Some(WorldsAction::CloseMenu);
    }
    if point_in(mx, my, l.new_rect(o)) {
        return Some(WorldsAction::New);
    }
    for slot in 0..l.rows() {
        if !point_in(mx, my, l.row_rect(o, slot)) {
            continue;
        }
        let i = view.scroll + slot;
        if i >= view.rows.len() {
            return Some(WorldsAction::Consume);
        }
        // The triple-dot is checked before the rest of the row, so reaching for
        // the menu never previews or opens the world it belongs to.
        if point_in(mx, my, l.dot_rect(o, slot)) {
            return Some(WorldsAction::OpenMenu(i));
        }
        // The switcher opens on the click. The start screen previews on it,
        // and opens only once that row is the one showing behind the panel.
        return Some(match l.mode {
            Mode::Session => WorldsAction::Open(i),
            Mode::Start if view.previewing == Some(i) => WorldsAction::Open(i),
            Mode::Start => WorldsAction::Select(i),
        });
    }
    point_in(mx, my, widget::outer_rect(o, l.size())).then_some(WorldsAction::Consume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::worlds::WorldRow;

    const VP: [f32; 2] = [1280.0, 720.0];
    // A macOS title bar's worth of chrome floating over the top of the frame.
    const INSET: f32 = 28.0;

    fn rows(n: usize) -> Vec<WorldRow> {
        (0..n)
            .map(|i| WorldRow {
                name: format!("world{i}"),
                path: format!("/p/worlds/world{i}.jsonl"),
                open: i == 0,
            })
            .collect()
    }

    fn layout(mode: Mode) -> Layout {
        Layout::new(mode, VP, 0.0)
    }

    fn view<'a>(l: Layout, rows: &'a [WorldRow]) -> WorldsView<'a> {
        WorldsView {
            rows,
            scroll: 0,
            layout: l,
            selected: None,
            previewing: None,
            menu: None,
            status: None,
            mouse: [0.0, 0.0],
        }
    }

    fn mid(r: [f32; 4]) -> (f32, f32) {
        (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5)
    }

    // The header carries one control: the `+` that starts an untitled world.
    #[test]
    fn hit_test_resolves_the_plus_button() {
        for mode in [Mode::Session, Mode::Start] {
            let l = layout(mode);
            let o = l.default_origin();
            let rows = rows(2);
            let v = view(l, &rows);
            let n = l.new_rect(o);
            let (x, y) = mid(n);
            assert_eq!(hit_test(&v, x, y, o), Some(WorldsAction::New), "{mode:?}");
            assert_eq!(n[2], n[3], "{mode:?}: a square, not a captioned chip");
            assert!(
                n[0] + n[2] <= o[0] + l.size()[0],
                "{mode:?}: inside the panel"
            );
        }
    }

    // The triple-dot wins over the row it sits in, so reaching for the menu
    // never opens the world instead.
    #[test]
    fn a_row_opens_and_its_triple_dot_opens_the_menu_in_the_switcher() {
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(3);
        let v = view(l, &rows);
        let r = l.row_rect(o, 1);
        assert_eq!(
            hit_test(&v, r[0] + 4.0, r[1] + 4.0, o),
            Some(WorldsAction::Open(1))
        );
        let d = l.dot_rect(o, 1);
        let (x, y) = mid(d);
        assert_eq!(hit_test(&v, x, y, o), Some(WorldsAction::OpenMenu(1)));
        assert!(
            d[0] > r[0] && d[0] + d[2] <= r[0] + r[2],
            "the dots sit inside their row"
        );
    }

    // The start screen's first click on a row previews it; the second, once
    // that world is the one showing, opens it.
    #[test]
    fn the_start_screen_selects_then_opens_the_same_row() {
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = rows(3);
        let r = l.row_rect(o, 2);
        let (x, y) = (r[0] + 8.0, r[1] + 8.0);

        let fresh = view(l, &rows);
        assert_eq!(hit_test(&fresh, x, y, o), Some(WorldsAction::Select(2)));

        let showing = WorldsView {
            selected: Some(2),
            previewing: Some(2),
            ..view(l, &rows)
        };
        assert_eq!(hit_test(&showing, x, y, o), Some(WorldsAction::Open(2)));
    }

    // The open menu is modal over the panel: its rows pick, and every other
    // press dismisses it rather than reaching the row underneath.
    #[test]
    fn the_open_menu_picks_and_swallows_everything_else() {
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = rows(3);
        let v = WorldsView {
            menu: Some(1),
            ..view(l, &rows)
        };
        let (_, open, delete) = l.menu_rects(o, 1);
        let (x, y) = mid(open);
        assert_eq!(hit_test(&v, x, y, o), Some(WorldsAction::Open(1)));
        let (x, y) = mid(delete);
        assert_eq!(hit_test(&v, x, y, o), Some(WorldsAction::Delete(1)));
        // Another row, the `+`, and the panel's own chrome all just dismiss it.
        let r = l.row_rect(o, 2);
        assert_eq!(
            hit_test(&v, r[0] + 4.0, r[1] + 4.0, o),
            Some(WorldsAction::CloseMenu)
        );
        let (x, y) = mid(l.new_rect(o));
        assert_eq!(hit_test(&v, x, y, o), Some(WorldsAction::CloseMenu));
    }

    // The switcher opens on the row click itself, so its menu offers Delete
    // alone rather than saying twice what a click already does.
    #[test]
    fn the_switchers_menu_offers_delete_alone() {
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(3);
        let v = WorldsView {
            menu: Some(1),
            ..view(l, &rows)
        };
        let (bg, open, delete) = l.menu_rects(o, 1);
        assert_eq!(open[3], 0.0, "no Open row");
        assert_eq!(delete[1], bg[1], "Delete sits at the top on its own");
        assert_eq!(bg[3], delete[3]);
        let (x, y) = mid(delete);
        assert_eq!(hit_test(&v, x, y, o), Some(WorldsAction::Delete(1)));

        // The sidebar's menu is the taller one, with Open above Delete.
        let start = layout(Mode::Start);
        let (bg, open, delete) = start.menu_rects(start.default_origin(), 1);
        assert!(open[3] > 0.0 && delete[1] == open[1] + open[3]);
        assert_eq!(bg[3], open[3] + delete[3]);
    }

    // The triple-dot keeps its own slot at every row's right end, inside the
    // row and clear of the caption, at either presentation's width.
    #[test]
    fn the_triple_dot_keeps_its_own_slot() {
        for mode in [Mode::Session, Mode::Start] {
            let l = layout(mode);
            let o = l.default_origin();
            let r = l.row_rect(o, 0);
            let d = l.dot_rect(o, 0);
            assert!(d[0] > r[0] + l.metrics().pad, "{mode:?}: clear of the name");
            assert!(d[0] + d[2] <= r[0] + r[2], "{mode:?}: inside the row");
            assert!(d[3] <= l.metrics().row_h, "{mode:?}: and inside its height");
        }
    }

    // A menu on a row near the bottom of a docked sidebar opens upward, so it
    // never runs off the window the panel spans.
    #[test]
    fn a_menu_near_the_bottom_opens_upward() {
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let last = l.rows() - 1;
        let (bg, ..) = l.menu_rects(o, last);
        let r = l.row_rect(o, last);
        assert!(bg[1] + bg[3] <= r[1], "it sits above the row it belongs to");
        assert!(bg[1] >= o[1], "and stays inside the panel");
        // A row with room below still opens downward.
        let (bg, ..) = l.menu_rects(o, 0);
        assert_eq!(bg[1], l.row_rect(o, 0)[1] + l.metrics().row_h);
    }

    // A scrolled listing resolves to the world under the cursor, not the slot.
    #[test]
    fn a_scrolled_row_resolves_to_its_world() {
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(POOL + 5);
        let v = WorldsView {
            scroll: 3,
            ..view(l, &rows)
        };
        let r = l.row_rect(o, 0);
        assert_eq!(
            hit_test(&v, r[0] + 4.0, r[1] + 4.0, o),
            Some(WorldsAction::Open(3))
        );
        // The menu's rows follow the same offset: slot 0 is world 3.
        let open = WorldsView { menu: Some(3), ..v };
        let (x, y) = mid(l.menu_rects(o, 0).2);
        assert_eq!(hit_test(&open, x, y, o), Some(WorldsAction::Delete(3)));
    }

    // Every press inside the panel is swallowed, and every press outside misses
    // entirely, so what is behind stays reachable. The sidebar's guard follows
    // its docked footprint, not the window it is drawn over.
    #[test]
    fn presses_are_rect_guarded_in_both_presentations() {
        for mode in [Mode::Session, Mode::Start] {
            let l = layout(mode);
            let o = l.default_origin();
            let s = l.size();
            let rows = rows(1);
            let v = view(l, &rows);
            let empty = l.row_rect(o, 5);
            assert_eq!(
                hit_test(&v, empty[0] + 4.0, empty[1] + 4.0, o),
                Some(WorldsAction::Consume),
                "{mode:?}: an empty slot is swallowed"
            );
            for (x, y) in [
                (o[0] + s[0] + 2.0, o[1] + 20.0),
                (o[0] + s[0] * 0.5, o[1] + s[1] + 2.0),
            ] {
                assert_eq!(
                    hit_test(&v, x, y, o),
                    None,
                    "{mode:?}: ({x}, {y}) is off the panel"
                );
            }
        }
    }

    // The switcher's title bar is chrome the panel swallows; the sidebar has
    // none at all, so its top strip is header instead.
    #[test]
    fn the_switcher_swallows_its_title_bar_and_the_sidebar_has_none() {
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(1);
        let t = l.title_rect(o);
        assert_eq!(
            hit_test(&view(l, &rows), t[0] + 4.0, t[1] + 4.0, o),
            Some(WorldsAction::Consume)
        );
        assert!(metrics(Mode::Session).has_title_bar());
        assert!(!metrics(Mode::Start).has_title_bar());
        assert_eq!(metrics(Mode::Start).title_h, 0.0);
    }

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
        assert!(start.menu_opens && !session.menu_opens);

        let l = layout(Mode::Start);
        assert_eq!(l.default_origin(), [0.0, 0.0]);
        assert_eq!(l.size(), [start.width, VP[1]]);
        assert!(l.size()[0] < VP[0] * 0.3, "the render owns the rest");

        // The switcher keeps its anchor below the top bar, centred.
        let s = layout(Mode::Session);
        let o = s.default_origin();
        assert!(o[1] >= hud::body_top());
        assert!((o[0] + s.size()[0] * 0.5 - VP[0] * 0.5).abs() < 0.5);
    }

    // The sidebar's background bleeds its rounding off the window's left, top,
    // and bottom edges, so it fills to the very top under the OS window buttons
    // and only the edge facing the render is rounded. The switcher's is its
    // plain footprint.
    #[test]
    fn the_docked_background_bleeds_off_the_window_edges() {
        let l = layout(Mode::Start);
        let bg = l.panel_rect([0.0, 0.0]);
        assert!(bg[0] < 0.0 && bg[1] < 0.0, "left and top run off-window");
        assert!(bg[1] + bg[3] > VP[1], "and so does the bottom");
        assert_eq!(bg[0] + bg[2], l.size()[0], "the inboard edge stays put");

        let s = layout(Mode::Session);
        let o = s.default_origin();
        assert_eq!(s.panel_rect(o), widget::outer_rect(o, s.size()));
    }

    // A non-zero top inset pushes the sidebar's whole content down by exactly
    // that much, so the `+` and the first row clear the OS window buttons
    // floating over the frame. Zero leaves the layout flush.
    #[test]
    fn the_top_inset_displaces_the_sidebar_content() {
        let flush = Layout::new(Mode::Start, VP, 0.0);
        let inset = Layout::new(Mode::Start, VP, INSET);
        let o = [0.0, 0.0];
        // With no chrome to clear, the content sits at the sidebar's own padding.
        assert_eq!(flush.new_rect(o)[1], metrics(Mode::Start).pad + 4.0);
        for (a, b) in [
            (flush.new_rect(o), inset.new_rect(o)),
            (flush.row_rect(o, 0), inset.row_rect(o, 0)),
            (flush.dot_rect(o, 0), inset.dot_rect(o, 0)),
        ] {
            assert_eq!(b[1] - a[1], INSET, "the content clears the chrome");
            assert_eq!(b[0], a[0], "and nothing moves sideways");
        }
        // The panel itself still spans the window: only the content moved.
        assert_eq!(inset.size(), flush.size());
        assert!(inset.panel_rect(o)[1] < 0.0);
        // The rows it lost to the inset come off the visible window.
        assert!(inset.rows() < flush.rows());
    }

    // The switcher floats clear of the window's top edge, so a reported inset
    // must not move it.
    #[test]
    fn the_top_inset_leaves_the_switcher_alone() {
        let flush = Layout::new(Mode::Session, VP, 0.0);
        let inset = Layout::new(Mode::Session, VP, INSET);
        let o = flush.default_origin();
        assert_eq!(inset.default_origin(), o);
        assert_eq!(inset.size(), flush.size());
        assert_eq!(inset.title_rect(o), flush.title_rect(o));
        assert_eq!(inset.row_rect(o, 0), flush.row_rect(o, 0));
        assert_eq!(inset.rows(), flush.rows());
    }

    // The switcher's footprint is fixed: it hangs from an anchor in a window
    // it does not fill, so neither a taller window nor a reported inset changes
    // it.
    #[test]
    fn the_switcher_is_independent_of_the_window_height() {
        let a = Layout::new(Mode::Session, VP, 0.0);
        let b = Layout::new(Mode::Session, [1280.0, 2400.0], 0.0);
        assert_eq!(a.size(), b.size());
        assert_eq!(a.rows(), b.rows());
        assert_eq!(
            a.row_rect(a.default_origin(), 0)[3],
            b.row_rect(b.default_origin(), 0)[3]
        );
    }

    // The visible row window follows the height on offer: the switcher is sized
    // to exactly its rows, while the sidebar takes what a tall window gives it
    // and a short one keeps it honest.
    #[test]
    fn the_row_window_follows_the_height_on_offer() {
        assert_eq!(layout(Mode::Session).rows(), SESSION_ROWS);
        assert_eq!(metrics(Mode::Start).max_rows, POOL);

        let tall = Layout::new(Mode::Start, [1280.0, 2400.0], 0.0);
        assert_eq!(tall.rows(), POOL, "a tall window is capped by the pool");
        let short = Layout::new(Mode::Start, [1280.0, 360.0], 0.0);
        assert!(short.rows() > 0 && short.rows() < layout(Mode::Start).rows());
        // Every row the sidebar offers fits inside the window it is docked to.
        for l in [tall, short, layout(Mode::Start)] {
            let last = l.row_rect([0.0, 0.0], l.rows().saturating_sub(1));
            assert!(last[1] + last[3] <= l.size()[1] + 0.01);
        }
    }

    #[test]
    fn the_wheel_region_covers_the_rows_and_not_the_header() {
        for mode in [Mode::Session, Mode::Start] {
            let l = layout(mode);
            let o = l.default_origin();
            let r = l.row_rect(o, 0);
            assert!(l.cursor_over_list(r[0] + 4.0, r[1] + 4.0, o));
            let n = l.new_rect(o);
            assert!(!l.cursor_over_list(n[0] + 4.0, n[1] + 4.0, o));
        }
    }

    // The scrollbar exists only while the listing overflows the visible rows,
    // and its thumb travels the track as the list scrolls.
    #[test]
    fn the_scrollbar_appears_only_on_overflow() {
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = l.rows();
        assert!(l.scrollbar(o, rows, 0).is_none());
        let top = l
            .scrollbar(o, rows + 4, 0)
            .expect("an overflowing listing has a bar");
        let down = l.scrollbar(o, rows + 4, 4).expect("still overflowing");
        assert!(down.thumb[1] > top.thumb[1]);
        assert!(down.thumb[1] + down.thumb[3] <= down.track[1] + down.track[3] + 0.01);
    }
}
