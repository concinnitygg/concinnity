// src/editor/hook/bookmarks.rs
//
// EditorHook: camera bookmarks. Ctrl+1..9 saves the current camera pose to a
// numbered slot, 1..9 glides back to it. Slots persist per world in the
// project's editor session store (`editor/session_store.rs`).

use super::*;

// The bookmark slot a digit key addresses, if any.
pub(super) fn slot_for(key: crate::components::InputKey) -> Option<usize> {
    use crate::components::InputKey;
    Some(match key {
        InputKey::Num1 => 0,
        InputKey::Num2 => 1,
        InputKey::Num3 => 2,
        InputKey::Num4 => 3,
        InputKey::Num5 => 4,
        InputKey::Num6 => 5,
        InputKey::Num7 => 6,
        InputKey::Num8 => 7,
        InputKey::Num9 => 8,
        _ => return None,
    })
}

impl EditorHook {
    pub(super) fn save_bookmark(&mut self, slot: usize, world: &World) {
        let Some(pose) = camera_pose::read(world) else {
            return;
        };
        self.bookmarks[slot] = Some(pose);
        let Some(path) = session_store::default_path() else {
            self.console_sink.info(&format!(
                "camera bookmark {} set for this session",
                slot + 1
            ));
            return;
        };
        let mut store = session_store::load(&path);
        store
            .worlds
            .entry(session_store::world_key(&self.world_path))
            .or_default()
            .bookmarks = self.bookmarks;
        match session_store::save(&path, &store) {
            Ok(()) => self
                .console_sink
                .info(&format!("camera bookmark {} saved", slot + 1)),
            Err(e) => self
                .console_sink
                .error(&format!("camera bookmark save failed: {e}")),
        }
    }

    pub(super) fn recall_bookmark(&mut self, slot: usize, world: &World) {
        let Some(pose) = self.bookmarks[slot] else {
            return;
        };
        let Some(from) = camera_pose::read(world) else {
            return;
        };
        self.orbit = None;
        self.start_glide(from, pose);
    }
}
