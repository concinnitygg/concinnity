// src/editor/mod.rs
//
// The `cn editor` run path. Unlike `cn debug` (which compiles world.jsonl fully
// in memory and stands up a WebSocket command channel), the editor reads the
// already-compiled blobs on startup, overlays an injected editor HUD, and
// persists edits by recompiling on SAVE. An optional debug port reuses the
// existing debug server so `cn debug send` / `screenshot` can inspect a session.

mod asset_list;
mod asset_tree;
mod axes;
mod behavior;
mod behavior_chart;
mod behavior_panel;
mod billboards;
mod character_shape;
mod character_shape_panel;
mod console;
mod console_panel;
mod content_panel;
mod create_menu;
mod cursor;
mod file_dialog;
mod filter;
mod form;
mod form_panel;
mod framing;
mod gizmo;
mod gltf_export;
mod group_transform;
mod health;
mod health_panel;
mod highlight;
mod history;
mod hook;
mod hud;
mod import_panel;
mod inject;
mod lighting;
mod lighting_panel;
mod list_panel;
mod live;
mod marquee;
pub(crate) mod notify;
mod orbit;
mod outlines;
mod overrides;
mod palette;
mod palette_panel;
mod panel;
mod preview;
mod registry;
mod resize;
mod select_related;
mod selection;
mod session_store;
mod sim;
mod snap;
mod story;
mod story_panel;
mod template_panel;
mod templates;
mod theme;
mod thumbs;
mod toast_overlay;
mod variables;
mod variables_panel;
mod view;
mod view_menu;
mod visibility;
mod widget;
mod widget_slider;

use crate::app::state::App;
use crate::debug_hook::DebugHook;
use crate::ecs::World;
use crate::world::{WORLD_JSONL, find_world_jsonl};
use concinnity_engine::shutdown::ShutdownToken;
use hook::EditorHook;

// A minimal renderable world: a lone GraphicsConfig, which the cook pipeline
// expands into a Window plus default shaders. Booted in memory when there is
// nothing renderable to load (no world file, or an authored world with no
// render marker), so the editor still opens a window over a black scene. Named
// distinctively so it never collides with an authored asset, and it is never
// added to the authored entry list, so it can never leak into the user's
// world.jsonl on SAVE.
const SEED_GRAPHICS_CONFIG: &str =
    "{\"name\":\"editor_default_gfx\",\"type\":\"GraphicsConfig\",\"args\":{}}";

/// Editor entry point (`cn editor`). Brings up a renderable world -- building the
/// blobs first if they are missing, and falling back to an empty in-memory world
/// when there is nothing to load -- injects the editor HUD, and runs the world
/// loop driven by the editor hook (plus the debug server when a port is given).
pub fn run_editor(json_path: Option<&str>, debug_port: Option<u16>) -> std::io::Result<()> {
    // Instead of the engine's plain `init_logging`: the same stderr formatter
    // plus a layer mirroring this crate's events into the Console panel's log.
    // The sink exists first so even boot-time errors reach the panel.
    let console_sink = console::ConsoleSink::default();
    console::install_tracing(console_sink.clone());

    // Resolve the edit target -- the world.jsonl where readable names live and
    // where SAVE writes. A missing file is not an error: the editor opens an
    // empty world and creates the file on the first SAVE.
    let (world_path, world_exists) = resolve_edit_target(json_path);

    // Hand the resolved path to the engine so the hot-reload watcher
    // subscribes to this world.jsonl. The engine no longer discovers it;
    // world.jsonl lookup is authoring I/O in concinnity-cook.
    concinnity_engine::app::dev_flags::set_world_jsonl_path(Some(world_path.clone()));

    // Parse the authored entry list up front so edits patch it directly (empty
    // when the file does not exist yet).
    let entries = if world_exists {
        let content = std::fs::read_to_string(&world_path)?;
        crate::world::parse_world_jsonl(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?
    } else {
        Vec::new()
    };

    // Bring up a renderable world: build the blobs if needed, load them, and
    // seed an empty world when there is nothing renderable to show.
    let mut app = crate::project::app();
    boot_world(&mut app, &world_path, world_exists, &entries)?;

    // Inject the editor HUD elements before start (this also drops the world's
    // DebugHud, whose F1 role the editor takes over); the editor's DebugHook
    // tick drives them each frame.
    inject::editor_hud(app.world_mut());

    // Every editor session hot-reloads file-backed assets; with a debug port
    // the DebugServer owns the reload driver (so the WS `reload-assets`
    // command reaches its flag), without one the driver runs as its own hook.
    // Either way the session holds exactly one driver, so a reload is never
    // applied twice.
    let editor_hook = EditorHook::new(world_path, entries).with_console_sink(console_sink);
    let hook: Box<dyn DebugHook> = match debug_port {
        Some(port) => {
            let server =
                crate::debug::DebugServer::start(port)?.with_notifier(editor_hook.notifier());
            MultiHook::boxed(vec![Box::new(editor_hook), Box::new(server)])
        }
        None => {
            let reload = crate::debug::hot_reload::HotReloadDriver::new()
                .with_notifier(editor_hook.notifier());
            MultiHook::boxed(vec![Box::new(editor_hook), Box::new(reload)])
        }
    };

    crate::run::start_app(app, Some(hook))
}

// Resolve the world.jsonl the editor edits, and whether it exists yet. An
// explicit path is taken as-is (present or not, so a brand-new file can be
// named); with no path, the most-recent world is used, falling back to the
// default `world.jsonl` name for a fresh, not-yet-saved world.
fn resolve_edit_target(json_path: Option<&str>) -> (String, bool) {
    match json_path {
        Some(p) => (p.to_string(), std::path::Path::new(p).exists()),
        None => match find_world_jsonl(crate::project::worlds_dir().as_deref(), None) {
            Ok(p) => (p, true),
            Err(_) => (WORLD_JSONL.to_string(), false),
        },
    }
}

// Populate `app` with a renderable world for editing:
//   * build the blobs first if the world has content but has not been compiled
//     yet (`cn build` as a library call);
//   * load the compiled blobs when present;
//   * if there is still nothing renderable (no world file, an authored world
//     with no render marker, or an empty build), boot a minimal in-memory world
//     seeded with a GraphicsConfig so the editor still opens a window.
fn boot_world(
    app: &mut App,
    world_path: &str,
    world_exists: bool,
    entries: &[serde_json::Value],
) -> std::io::Result<()> {
    let blobs_present = || {
        crate::project::data_dir()
            .is_some_and(|d| concinnity_host::store::blob::primary_in(&d).exists())
    };

    // Build if the world has content the compiled blobs do not reflect yet.
    if world_exists && !blobs_present() {
        crate::build_world_to_disk(world_path)?;
    }

    // Load the compiled blobs when they exist; the primary render source.
    if blobs_present() {
        app.load_blob().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to load compiled world data: {e:?}"),
            )
        })?;
        // The blobs carry only interned ids; a boot without an in-process cook
        // has an empty name table, which kills name-keyed picking until the
        // first edit. Restore it from the lock the build wrote. Best effort: a
        // missing lock only means the pre-existing degraded behavior.
        match crate::authoring::name_table::prime_from_lock_file() {
            Ok(n) if n > 0 => tracing::info!("editor: primed {n} asset names from the build lock"),
            Ok(_) => {}
            Err(e) => tracing::warn!("editor: could not prime asset names: {e}"),
        }
        // The in-memory build installs the hot-reload source catalogues as it
        // compiles; a blob boot reconstructs them from the lock + the authored
        // entries so file-backed assets reload here too. Best effort, like the
        // name priming above.
        match crate::authoring::reload_sources::install_from_lock(app.world_mut(), entries) {
            Ok(n) if n > 0 => {
                tracing::info!("editor: recovered {n} hot-reload source(s) from the build lock");
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("editor: could not recover hot-reload sources: {e}"),
        }
    }

    // Fall back to an in-memory seed when nothing renderable was loaded, so a
    // window still opens over a black scene.
    if !concinnity_engine::ecs::renders(app.world()) {
        let base = if world_exists {
            std::fs::read_to_string(world_path)?
        } else {
            String::new()
        };
        let world = crate::build_world_from_str(&seeded_content(&base))?;
        app.load_world(world);
    }

    Ok(())
}

// Guarantee a render marker: append the seed GraphicsConfig to the authored
// content (only reached when the world does not otherwise render, so there is
// no existing GraphicsConfig to collide with).
fn seeded_content(base: &str) -> String {
    if base.trim().is_empty() {
        SEED_GRAPHICS_CONFIG.to_string()
    } else {
        format!("{base}\n{SEED_GRAPHICS_CONFIG}")
    }
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

    fn apply_world_swap(&mut self, app: &mut crate::app::state::App) {
        for hook in &mut self.hooks {
            hook.apply_world_swap(app);
        }
    }

    fn attach_shutdown(&mut self, shutdown: ShutdownToken) {
        for hook in &mut self.hooks {
            hook.attach_shutdown(shutdown.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An empty (or whitespace-only) world seeds to just the render marker, so an
    // empty session still opens a window.
    #[test]
    fn seeded_content_of_empty_is_the_render_marker() {
        assert_eq!(seeded_content(""), SEED_GRAPHICS_CONFIG);
        assert_eq!(seeded_content("   \n"), SEED_GRAPHICS_CONFIG);
    }

    // Authored content keeps its entries and gains the render marker on its own
    // line, so the combined string still parses as one asset per line.
    #[test]
    fn seeded_content_appends_marker_to_authored_content() {
        let base = "{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}";
        let seeded = seeded_content(base);
        let parsed = crate::world::parse_world_jsonl(&seeded).unwrap();
        assert_eq!(parsed.len(), 2, "authored entry plus the seed marker");
        assert_eq!(parsed[0]["name"], "phys");
        assert_eq!(parsed[1]["type"], "GraphicsConfig");
    }

    // The seed marker is itself a well-formed, renderable asset line.
    #[test]
    fn seed_marker_is_a_graphics_config() {
        let parsed = crate::world::parse_world_jsonl(SEED_GRAPHICS_CONFIG).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["type"], "GraphicsConfig");
    }

    // An explicit path is taken verbatim; its existence is reported so a
    // brand-new (not-yet-saved) file boots as an empty world rather than erroring.
    #[test]
    fn resolve_edit_target_honors_an_explicit_path() {
        let (path, exists) = resolve_edit_target(Some("/no/such/cn-editor-world.jsonl"));
        assert_eq!(path, "/no/such/cn-editor-world.jsonl");
        assert!(
            !exists,
            "a missing explicit path is reported absent, not an error"
        );
    }
}
