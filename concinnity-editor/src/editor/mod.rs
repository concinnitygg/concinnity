// src/editor/mod.rs
//
// The `cn editor` run path. Unlike `cn debug` (which compiles world.jsonl fully
// in memory and stands up a WebSocket command channel), the editor reads the
// already-compiled blobs on startup, overlays an injected editor HUD, and
// persists edits by recompiling on SAVE. An optional debug port reuses the
// existing debug server so `cn debug smoke` / `screenshot` can verify a session.

mod hook;
mod hud;
mod inject;

use crate::app::state::App;
use crate::debug_hook::DebugHook;
use crate::ecs::World;
use crate::world::find_world_jsonl;
use hook::EditorHook;
use tokio_util::sync::CancellationToken;

// Editor entry point (`cn editor`). Loads the compiled world from `.concinnity/
// data/`, injects the editor HUD, and runs the world loop driven by the editor
// hook (plus the debug server when a port is given).
pub(crate) fn run_editor(json_path: Option<&str>, debug_port: Option<u16>) -> std::io::Result<()> {
    concinnity_client::app::run::init_logging();

    // Resolve the world.jsonl path -- the edit target, where readable names
    // live. The compiled blobs are what render; world.jsonl is what we mutate.
    let resolved;
    let json_path = match json_path {
        Some(p) if std::path::Path::new(p).exists() => p.to_string(),
        _ => {
            resolved = find_world_jsonl(None)?;
            resolved
        }
    };

    // The editor reads compiled blobs on startup; a missing build is a hard
    // error (there is nothing to edit until `cn build` has run).
    let mut app = App::new();
    app.load_blob().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no compiled world data (run `concinnity build` first): {e:?}"),
        )
    })?;

    // Parse the authored entry list up front so edits patch it directly.
    let content = std::fs::read_to_string(&json_path)?;
    let entries = concinnity_core::world::parse_world_jsonl(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    // Inject the editor HUD before start, so build_internal_systems constructs
    // the editor HUD system from it alongside the world's own systems.
    inject::editor_hud(app.world_mut());

    let editor_hook = EditorHook::new(json_path, entries);
    let hook: Box<dyn DebugHook> = match debug_port {
        Some(port) => {
            let server = crate::debug::DebugServer::start(port)?;
            MultiHook::boxed(vec![Box::new(editor_hook), Box::new(server)])
        }
        None => Box::new(editor_hook),
    };

    crate::run::start_app(app, Some(hook))
}

// Fan a single per-frame drive out to several hooks. Lets the editor run its own
// hook and the debug server side by side without either owning the other.
struct MultiHook {
    hooks: Vec<Box<dyn DebugHook>>,
}

impl MultiHook {
    fn boxed(hooks: Vec<Box<dyn DebugHook>>) -> Box<dyn DebugHook> {
        Box::new(Self { hooks })
    }
}

impl DebugHook for MultiHook {
    fn tick(&mut self, world: &mut World) {
        for hook in &mut self.hooks {
            hook.tick(world);
        }
    }

    fn attach_shutdown(&mut self, shutdown: CancellationToken) {
        for hook in &mut self.hooks {
            hook.attach_shutdown(shutdown.clone());
        }
    }
}
