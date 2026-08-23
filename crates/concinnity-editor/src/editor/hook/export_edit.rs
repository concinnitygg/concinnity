// src/editor/hook/export_edit.rs
//
// EditorHook: the console's /export action. Compiles the working entries in
// memory on a worker thread (under the one-cook-at-a-time guard), writes the
// named skinned mesh as `<name>.glb` beside the project's world file, and
// reports through the log sink and a toast.

use super::*;
use crate::editor::gltf_export;
use std::sync::atomic::Ordering;

impl EditorHook {
    // /export: resolve the mesh (an explicit name, or the selection through
    // the same binding the shape panel uses) and export it off-thread.
    pub(super) fn console_export(&mut self, name: Option<&str>, bake: bool) {
        let mesh = match name {
            Some(n) => n.to_string(),
            None => match self.shape_binding() {
                Some(b) => b.mesh,
                None => {
                    self.console_sink
                        .error("select a skinned mesh or name one: /export [name] [bake]");
                    return;
                }
            },
        };
        if self.console_build_running.swap(true, Ordering::SeqCst) {
            self.console_sink.warn("cook already running");
            return;
        }
        let content = match crate::world::write_world_jsonl(&self.entries) {
            Ok(c) => c,
            Err(e) => {
                self.console_build_running.store(false, Ordering::SeqCst);
                self.console_sink.error(&format!("export failed: {e}"));
                return;
            }
        };
        let out = std::path::Path::new(&self.world_path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(format!("{mesh}.glb"));
        let sink = self.console_sink.clone();
        let toasts = self.notifier.clone();
        let running = self.console_build_running.clone();
        let op = self.notifier.begin_op("Exporting");
        sink.info(&format!("export of '{mesh}' started"));
        std::thread::spawn(move || {
            // The bounded pool keeps the compile off rayon's global pool,
            // like the cook worker.
            let outcome = crate::jobs::pool()
                .install(|| gltf_export::export_world_mesh(&content, &mesh, bake))
                .and_then(|bytes| {
                    std::fs::write(&out, &bytes)
                        .map(|_| bytes.len())
                        .map_err(|e| format!("write {}: {e}", out.display()))
                });
            op.finish();
            match outcome {
                Ok(len) => {
                    let what = format!("Exported {} ({:.1} MB)", out.display(), len as f64 / 1e6);
                    sink.info(&what);
                    toasts.success(&what);
                }
                Err(e) => {
                    sink.error(&format!("export failed: {e}"));
                    toasts.error_with(&format!("Export failed: {e}"), notify::Action::OpenConsole);
                }
            }
            running.store(false, Ordering::SeqCst);
        });
    }
}
