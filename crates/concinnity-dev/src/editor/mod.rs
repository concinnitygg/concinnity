// src/editor/mod.rs
//
// The `cn editor` run path. Like `cn debug`, the editor compiles world.jsonl in
// memory: the session boots from the authored entries, not from the blobs a
// build left under the state tree, so what opens is always what the world file
// says. It overlays an injected editor HUD and persists edits by writing
// world.jsonl. The blobs are refreshed only by an explicit build (`cn build`,
// or the console's cook command). An optional debug port reuses the existing
// debug server so an MCP client can inspect and drive a session.
//
// `cn editor -f <world>` opens that world. With no world named the session
// opens an empty scene under the Worlds panel (`editor/worlds.rs`), which
// lists the project's worlds and opens, creates, or deletes one.

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
mod modal;
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
mod world_files;
mod worlds;

use crate::app::state::App;
use crate::debug_hook::DebugHook;
use crate::ecs::World;
use crate::world::WORLD_JSONL;
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

/// Editor entry point (`cn editor`). Compiles the authored world in memory,
/// injects the editor HUD, and runs the world loop driven by the editor hook
/// (plus the debug server when a port is given).
pub fn run_editor(json_path: Option<&str>, debug_port: Option<u16>) -> std::io::Result<()> {
    // Instead of the engine's plain `init_logging`: the same stderr formatter
    // plus a layer mirroring this crate's events into the Console panel's log.
    // The sink exists first so even boot-time errors reach the panel.
    let console_sink = console::ConsoleSink::default();
    console::install_tracing(console_sink.clone());

    // Resolve the edit target -- the world.jsonl where readable names live and
    // where SAVE writes -- and whether the session opens on the Worlds panel
    // instead of a world.
    let (world_path, pick_a_world) = resolve_edit_target(json_path);

    // Hand the resolved path to the engine so the hot-reload watcher
    // subscribes to this world.jsonl. The engine no longer discovers it;
    // world.jsonl lookup is authoring I/O in concinnity-cook.
    concinnity_engine::app::dev_flags::set_world_jsonl_path(Some(world_path.clone()));

    // Parse the authored entry list up front so edits patch it directly. A
    // session opening on the start screen edits nothing until a world is picked
    // there, and it boots on nothing: the window comes up on the screen's own
    // listing, and the project's most recent world is compiled behind it a few
    // frames later (`hook/worlds_start.rs`). A world that takes seconds to
    // compile is then waited out on a screen that is up and usable rather than
    // in front of no window at all.
    let (entries, previewing) = if pick_a_world {
        (Vec::new(), start_screen_pick())
    } else if std::path::Path::new(&world_path).exists() {
        let content = std::fs::read_to_string(&world_path)?;
        let entries = crate::world::parse_world_jsonl(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        (entries, None)
    } else {
        (Vec::new(), None)
    };

    // Bring up a renderable world by compiling those entries, seeding a render
    // marker when they alone would not render.
    let mut app = crate::project::app();
    boot_world(&mut app, &entries)?;

    // Inject the editor HUD elements before start (this also drops the world's
    // DebugHud, whose F1 role the editor takes over); the editor's DebugHook
    // tick drives them each frame.
    inject::editor_hud(app.world_mut());

    // Every editor session hot-reloads file-backed assets; with a debug port
    // the DebugServer owns the reload driver (so the WS `reload-assets`
    // command reaches its flag), without one the driver runs as its own hook.
    // Either way the session holds exactly one driver, so a reload is never
    // applied twice.
    let mut editor_hook = EditorHook::new(world_path, entries).with_console_sink(console_sink);
    if pick_a_world {
        editor_hook = editor_hook.with_start_screen(previewing);
    }
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

// Resolve the world the editor opens on, and whether it opens on the Worlds
// panel rather than on that world. An explicit path is taken as-is (present or
// not, so a brand-new file can be named) and loads straight away. With no path
// the session opens an empty scene and the Worlds panel, which picks the world
// to work on; the path stands in until it does, so a SAVE before any pick
// still lands in the project's `worlds/`.
fn resolve_edit_target(json_path: Option<&str>) -> (String, bool) {
    match json_path {
        Some(p) => (p.to_string(), false),
        None => (unsaved_world_path(), true),
    }
}

// The world the start screen preselects: the project's most recent one. Only
// its path -- reading and compiling it is the screen's own work, done once it
// has a window to show the result in. A project with no worlds preselects
// nothing, which is what the screen's empty listing already says.
fn start_screen_pick() -> Option<String> {
    let world = world_files::newest(
        crate::project::worlds_dir().as_deref(),
        crate::project::content_root().as_deref(),
    )?;
    Some(world.path.to_string_lossy().into_owned())
}

// Where the editor puts a world nobody has saved yet.
pub(crate) fn unsaved_world_path() -> String {
    crate::project::worlds_dir()
        .map(|dir| dir.join(WORLD_JSONL).to_string_lossy().into_owned())
        .unwrap_or_else(|| WORLD_JSONL.to_string())
}

// Populate `app` with a renderable world for editing, compiled from the
// authored entries in memory. Nothing under the build root is read: the blobs
// there are refreshed only by an explicit build, so they may lag the world file
// the editor is opening.
fn boot_world(app: &mut App, entries: &[serde_json::Value]) -> std::io::Result<()> {
    let jsonl = crate::world::write_world_jsonl(entries)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let (world, _) = build_renderable(&jsonl)?;
    app.load_world(world);
    Ok(())
}

// Compile world.jsonl content into a ready-to-run world, plus the template
// baselines its expansion merged authored patches over. Content that would not
// render (an empty world, or authored entries with no render marker) is
// recompiled with a seeded GraphicsConfig, so a session always opens a window.
// Boot and every live-preview rebuild come through here, so what the editor
// shows never depends on which of the two produced it.
fn build_renderable(
    jsonl: &str,
) -> std::io::Result<(World, Vec<concinnity_cook::build_only::ShadowedAsset>)> {
    match crate::authoring::build_world_and_shadows(jsonl) {
        Ok(built) if concinnity_engine::ecs::renders(&built.0) => Ok(built),
        _ => crate::authoring::build_world_and_shadows(&seeded_content(jsonl)),
    }
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

    // A project whose build root is a `.concinnity/` of its own, as `cn` opens
    // one, with the machine-wide cache so a boot compiles shaders once.
    fn open_project(dir: &std::path::Path) -> std::path::PathBuf {
        let build_root = dir.join(".concinnity");
        crate::project::open(
            concinnity_host::store::paths::StateTree::at(dir)
                .with_build(&build_root)
                .with_cache(concinnity_testing::shared_cache_dir(
                    "concinnity-dev-tests-cache",
                )),
        );
        build_root
    }

    fn entry(name: &str, ty: &str, args: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"name": name, "type": ty, "args": args})
    }

    // A renderable authored world, plus a label whose content identifies which
    // compile a booted world came from.
    fn renderable_entries(label: &str) -> Vec<serde_json::Value> {
        vec![
            entry("cam", "Camera3D", serde_json::json!({})),
            entry("room", "Room", serde_json::json!({})),
            entry("hint", "TextLabel", serde_json::json!({"content": label})),
        ]
    }

    // The content of the booted world's only TextLabel.
    fn booted_label(app: &App) -> String {
        app.world()
            .query::<crate::components::TextLabel>()
            .next()
            .expect("the authored label is in the booted world")
            .content
            .clone()
    }

    // Boot compiles the authored entries in memory: the world it brings up is
    // the one the entry list describes, and no build output is read or written.
    #[test]
    fn boot_compiles_the_authored_entries_without_touching_the_build_root() {
        let _guard = crate::test_support::lock();
        let dir = concinnity_testing::TempTree::new();
        let build_root = open_project(dir.path());

        let mut app = crate::project::app();
        boot_world(&mut app, &renderable_entries("authored")).expect("the world builds");

        assert!(concinnity_engine::ecs::renders(app.world()));
        assert_eq!(booted_label(&app), "authored");
        assert!(
            !build_root.join("data").exists() && !build_root.join("world-lock.json").exists(),
            "boot writes no blobs and no lock"
        );

        crate::test_support::isolate_state_dir();
    }

    // Blobs a build left behind are ignored: the session shows what the entry
    // list says even when the compiled output on disk says something else, and
    // that output is left exactly as the build wrote it.
    #[test]
    fn boot_ignores_blobs_that_no_longer_match_the_entries() {
        let _guard = crate::test_support::lock();
        let dir = concinnity_testing::TempTree::new();
        let build_root = open_project(dir.path());

        // An explicit build, as `cn build` runs it, over the stale world.
        let world_path = dir.path().join("worlds").join(WORLD_JSONL);
        std::fs::create_dir_all(world_path.parent().unwrap()).unwrap();
        std::fs::write(
            &world_path,
            crate::world::write_world_jsonl(&renderable_entries("stale")).unwrap(),
        )
        .unwrap();
        crate::build_world_to_disk(world_path.to_str().unwrap()).expect("the build writes blobs");
        let blob = concinnity_host::store::blob::primary_in(&build_root.join("data"));
        let before = std::fs::read(&blob).expect("the build wrote a primary blob");

        let mut app = crate::project::app();
        boot_world(&mut app, &renderable_entries("edited")).expect("the world builds");

        assert_eq!(
            booted_label(&app),
            "edited",
            "the entries win over the blobs the last build left"
        );
        assert_eq!(
            std::fs::read(&blob).unwrap(),
            before,
            "boot leaves the build output untouched"
        );

        crate::test_support::isolate_state_dir();
    }

    // Nothing renderable to compile still opens a window: an empty entry list
    // boots the seeded render marker.
    #[test]
    fn boot_seeds_a_render_marker_for_an_empty_entry_list() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();

        let mut app = crate::project::app();
        boot_world(&mut app, &[]).expect("an empty world seeds");
        assert!(concinnity_engine::ecs::renders(app.world()));
    }

    // An explicit path is taken verbatim and loads directly, panel closed --
    // including one that does not exist yet, which boots as an empty world
    // rather than erroring.
    #[test]
    fn resolve_edit_target_honors_an_explicit_path() {
        let (path, pick) = resolve_edit_target(Some("/no/such/cn-editor-world.jsonl"));
        assert_eq!(path, "/no/such/cn-editor-world.jsonl");
        assert!(!pick, "a named world loads instead of the Worlds panel");
    }

    // With no world named, the session opens on the Worlds panel instead of
    // guessing which of the project's worlds the user meant.
    #[test]
    fn resolve_edit_target_without_a_path_opens_the_worlds_panel() {
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();

        let (path, pick) = resolve_edit_target(None);
        assert!(pick, "no named world opens the Worlds panel");
        assert_eq!(path, unsaved_world_path());
    }

    // With no world to discover, the editor opens an unsaved one in the
    // project's `worlds/`, so the first save lands where a build looks.
    #[test]
    fn an_unsaved_world_is_named_inside_the_projects_worlds_directory() {
        // Reading the session's project; opening one is what the guard covers.
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();

        let worlds = crate::project::worlds_dir().expect("the harness opened a project");
        assert_eq!(
            unsaved_world_path(),
            worlds.join(WORLD_JSONL).to_string_lossy()
        );
    }
}
