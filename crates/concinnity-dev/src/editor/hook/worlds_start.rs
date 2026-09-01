// src/editor/hook/worlds_start.rs
//
// EditorHook: the start screen. A session that named no world on the command
// line opens on the Worlds panel alone, and picking a row there previews that
// world live -- compiled in memory and swapped under the running backend by the
// same rebuild every edit takes, but without retargeting the session or leaving
// the screen. Opening then commits what is already showing, so the world the
// user picked is not compiled twice.

use super::*;
use std::path::Path;

impl EditorHook {
    // Everything a frame's input drives while the start screen is up. The panel
    // is the only thing on screen, so the press it does not resolve is swallowed
    // rather than offered to the world behind it (there is nothing there to
    // pick, and a stray click must not disturb the preview).
    pub(super) fn drive_start_input(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) {
        if input.left_click {
            // The confirmation dialog (a delete) swallows every press first.
            if !self.route_modal_click(input, vp, world) {
                self.route_start_click(input, vp, world);
            }
        }
        if self.modal.is_none() && input.scroll_delta.abs() > 0.5 {
            let o = self.origin(PanelKey::Worlds, vp);
            if self
                .worlds_layout()
                .cursor_over_list(input.mouse_x, input.mouse_y, o)
            {
                self.scroll_worlds(input.scroll_delta);
            }
        }
    }

    // Resolve a press against the start screen's panel. Nothing else routes:
    // the top bar is not drawn, the panel neither drags nor closes, and a press
    // that misses the panel does nothing at all.
    fn route_start_click(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        let o = self.origin(PanelKey::Worlds, vp);
        let view = self.make_worlds_view([mx, my]);
        if let Some(action) = worlds::hit_test(&view, mx, my, o) {
            self.apply_worlds_action(action, world);
        }
    }

    // Which presentation the panel draws and hit-tests in this frame.
    pub(super) fn worlds_mode(&self) -> worlds::Mode {
        match self.start_mode {
            true => worlds::Mode::Start,
            false => worlds::Mode::Session,
        }
    }

    // That presentation resolved against this frame's window: the sidebar is
    // sized by the window it docks to and offset by the chrome floating over
    // its top.
    pub(super) fn worlds_layout(&self) -> worlds::Layout {
        worlds::Layout::new(self.worlds_mode(), self.viewport, self.top_inset)
    }

    // The listed row a path sits at, for the selection and preview marks the
    // view carries (both are held by path, so a refreshed listing cannot slide
    // them onto another world).
    pub(super) fn worlds_row_of(&self, path: Option<&String>) -> Option<usize> {
        let path = path?;
        self.worlds_rows.iter().position(|r| &r.path == path)
    }

    // Select listed world `i` and show it behind the screen. The session is not
    // retargeted: the path a SAVE would write, the history, and the watcher all
    // stay where they are until the world is opened.
    pub(super) fn select_world(&mut self, i: usize) {
        let Some(row) = self.worlds_rows.get(i) else {
            return;
        };
        let path = row.path.clone();
        self.worlds_selected = Some(path.clone());
        // A pick the screen opened on is superseded rather than compiled after
        // the row the user actually asked for.
        self.start_preview = None;
        self.preview_world(&path);
    }

    // Frames the loading cover is given before the compile that blocks on it.
    const COVER_FRAMES: u8 = 2;

    // Frames the screen must have been drawn for before the world it opened on
    // is compiled. One is enough to have the listing on screen; the second
    // covers a frame the window manager dropped bringing the window up.
    pub(super) const START_PREVIEW_DELAY: u32 = 2;

    // Stage the world the screen opened on, once the screen has been drawn.
    // Everything the editor needs is up by then -- the window, the panel's own
    // baked elements, and the listing read off disk -- so the compile is spent
    // behind a screen the user can already read, under the loading cover.
    pub(super) fn drive_start_preview(&mut self) {
        if self.start_drawn < Self::START_PREVIEW_DELAY {
            return;
        }
        let Some(path) = self.start_preview.take() else {
            return;
        };
        self.preview_world(&path);
    }

    // Whether the screen is standing over a world it cannot show yet: the pick
    // it opened on, or any row since, while the compile that brings it up is
    // still owed.
    pub(super) fn loading_preview(&self) -> bool {
        self.start_mode && (self.start_preview.is_some() || self.rebuild_preview)
    }

    // The name the loading cover says, which is the world being compiled.
    pub(super) fn loading_name(&self) -> Option<String> {
        let path = self
            .start_preview
            .as_deref()
            .or(self.worlds_preview.as_deref())?;
        Some(session_store::world_key(path))
    }

    // Compile the world at `path` into the background. The rebuild request is a
    // flag the frame loop consumes once, so a burst of row clicks costs one
    // compile of the world last picked rather than one per click.
    fn preview_world(&mut self, path: &str) {
        match world_files::read_entries(Path::new(path)) {
            Ok(entries) => {
                self.worlds_status = None;
                self.worlds_preview = Some(path.to_string());
                self.stage_preview(entries);
            }
            // A world that will not parse cannot be shown: the screen says so
            // and keeps whatever was already behind it.
            Err(e) => self.worlds_status = Some(e),
        }
    }

    // A preview whose compile failed is not showing: the row loses its mark,
    // so the next press on it compiles again rather than opens, and the panel
    // says why. The world that was up stays up.
    pub(super) fn preview_failed(&mut self, error: &str) {
        self.worlds_preview = None;
        self.worlds_status = Some(short_status(error));
    }

    // Drop the background back to the seeded empty scene, which is what a
    // session with no world to show opens on.
    fn clear_preview(&mut self) {
        self.worlds_preview = None;
        self.stage_preview(Vec::new());
    }

    // Hold `entries` as the world the session is showing. Nothing here is an
    // edit: the list reads as saved and carries no history, so the start screen
    // can never present a world as having unsaved changes.
    fn stage_preview(&mut self, entries: Vec<serde_json::Value>) {
        // The world these entries replace is the one the attract camera was
        // framing, so the cycle starts again (on its first shot, from black)
        // over whatever the rebuild brings up.
        self.restart_cinematic();
        self.history = History::default();
        self.baseline = entries.clone();
        self.saved = entries.clone();
        self.entries = entries;
        self.dirty = false;
        self.require_rebuild();
        // The compile blocks the frame it runs on, so it waits for the loading
        // cover to have been drawn and presented first.
        self.rebuild_countdown = Self::COVER_FRAMES;
    }

    // Open listed world `i` from the start screen. The world already showing is
    // adopted as it stands -- it was compiled from these very entries -- so
    // committing costs no second compile; a row that is not the one showing (a
    // preview that failed, or a selection the deleted world left behind) is
    // read and compiled the usual way.
    pub(super) fn open_from_start(&mut self, i: usize, world: &mut World) {
        let Some(row) = self.worlds_rows.get(i) else {
            return;
        };
        let path = row.path.clone();
        if self.worlds_preview.as_deref() == Some(path.as_str()) {
            // The attract camera ends here: the session is about to adopt this
            // very world, and it must open on the camera the world declared.
            self.stop_cinematic(world);
            let entries = self.entries.clone();
            self.retarget(path, entries, Adopt::Showing);
            return;
        }
        match world_files::read_entries(Path::new(&path)) {
            Ok(entries) => {
                // The world showing gets its own camera back before it goes:
                // should the compile fail, it is the world the session keeps.
                self.stop_cinematic(world);
                self.retarget(path, entries, Adopt::No);
            }
            // A file that will not parse leaves the screen as it stands.
            Err(e) => self.worlds_status = Some(e),
        }
    }

    // Follow a delete through on the start screen: a deleted world can no
    // longer be shown, so the background falls back to the seeded empty scene,
    // and the selection moves to the row that took its place (the next one
    // down, or the last row once the list has been used up).
    pub(super) fn reselect_after_delete(&mut self, path: &str, was_at: Option<usize>) {
        if self.worlds_selected.as_deref() == Some(path) {
            let next = was_at
                .and_then(|i| self.worlds_rows.get(i))
                .or_else(|| self.worlds_rows.last());
            self.worlds_selected = next.map(|r| r.path.clone());
        }
        if self.worlds_preview.as_deref() == Some(path) {
            self.clear_preview();
        }
        if self.start_preview.as_deref() == Some(path) {
            self.start_preview = None;
        }
    }

    // Leave the start screen for good: the session now owns a world, so the top
    // bar, the panels, and the viewport overlays all come back.
    pub(super) fn leave_start_screen(&mut self) {
        self.start_mode = false;
        self.worlds_selected = None;
        self.worlds_preview = None;
        self.start_preview = None;
    }
}

impl EditorHook {
    // Draw the loading cover over the area the sidebar leaves to the render,
    // or hide it. The sidebar itself is never covered: the listing stays
    // readable and clickable through the wait.
    pub(super) fn drive_loading_draw(&self, world: &mut World, shown: bool) {
        match shown && self.loading_preview() {
            true => worlds::loading::apply(
                world,
                self.worlds_layout().preview_rect(),
                self.loading_name().as_deref(),
            ),
            false => worlds::loading::hide(world),
        }
    }
}

// Whether a retarget can adopt the world the session is already showing instead
// of compiling the one it is opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Adopt {
    // The start screen's preview already compiled and swapped these entries in.
    Showing,
    // Anything else: the live world stands for another world's entries.
    No,
}
