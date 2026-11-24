//! The `cn edit` run path. Like `cn run -f`, the editor compiles .jsonl in
//! memory: the session boots from the authored entries, not from the blobs a
//! build left under the state tree, so what opens is always what the file
//! says. It sets its own UI up in that world and persists edits by writing
//! .jsonl. The blobs are refreshed only by an explicit build (`cn cook`,
//! or the console's cook command). An optional debug port reuses the existing
//! debug server so an MCP client can inspect and drive a session.
//!
//! `cn edit -f <world>` opens that world. With no world named the session
//! opens an empty scene under the Worlds panel (`editor/worlds/`), which lists
//! the project's worlds and opens, creates, or deletes one.
//!
//! The subsystem splits in one place and splits there all the way down. `hook/`
//! is the editor as a single per-frame drive: it holds every piece of session
//! state, and it is the only module here that reads a frame's input or decides
//! that the world changes. Every module below is what it drives. The panels and
//! the viewport are pure -- geometry, a model, or math over state handed in --
//! which is what lets them be tested without a window; the few that do reach
//! outside (`session_store`, `file_dialog`, `thumbs`, `gltf_export/`, and
//! `live/` writing the running world) say so on their own line.
//!
//! Where a concern lives follows from how big it got. A panel is one or two
//! files under `panels/`; a panel whose model outgrew that keeps its own
//! directory (`behavior/`, `palette/`, `worlds/`). What each module holds is on
//! the line above its declaration.

// How the editor addresses one asset of the world it is editing: the currency
// of the selection, the open form, and every panel that follows it.
mod asset_handle;
// The Behavior panel's model half: one behavior's authored args as an editable
// node graph, plus the palette, outline and chart views over it.
mod behavior;
// How a world that failed to build is reported: a console line per problem,
// and a toast that selects the asset the first one names.
mod build_failure;
// The macOS menu bar: the application menu, and a View menu offering the
// Display chip's rows a second time.
#[cfg(target_os = "macos")]
mod app_menu;
// The monospace face code text draws with, baked once for the process.
mod code_font;
// The viewport's right-click "Create here" menu, anchored at the cursor.
mod create_menu;
// The authored entry list, with the session key every entry is addressed by.
mod entry_list;
// The native file picker behind Import's Browse, and the project-relative
// rewrite its result needs.
mod file_dialog;
// The ranked substring filter every pick list narrows through.
mod filter;
// glTF (.glb) export of a skinned mesh: geometry, skeleton and morph targets.
mod gltf_export;
// Bounded undo / redo stacks over the authored entry list. Pure data.
mod history;
// The editor itself: one per-frame drive holding all of the session state,
// and the only half here that reads input or writes the world.
mod hook;
// The top bar: the full-width strip holding SAVE, the panel chips and the
// transport.
mod hud;
// Vector icons drawn from rounded rectangles, sharp at any scale.
mod icons;
// Setting the editor up in a compiled world, between the in-memory compile and
// `Runtime::start`.
mod install;
// Applying an edit to the running preview world instead of rebuilding it.
mod live;
// The world map: its places and the moves between them, as a chart.
mod map;
// The confirmation dialog: a modal panel with its message, optional name field
// and checkbox, and its buttons.
mod modal;
// The toast inbox the editor and its workers push into, and the feed that
// turns it into toasts.
pub(crate) mod notify;
// Extent outlines for assets with spatial reach but no geometry: trigger
// volumes, light ranges, frusta, probe bounds.
mod outlines;
// Copies of the files an entry owns, for a duplicate that must not share them.
mod owned_files;
// Per-field override state for a template-derived asset, whose authored line
// is a sparse patch over what the expansion generated.
mod overrides;
// The command palette's model: every actionable thing the editor can reach.
mod palette;
// The Planet panel's tools and brush, the strokes they lay and the brush ring.
mod planet_tools;
// Every floating panel: its layout half, its data half, and the registry
// each panel is one entry in.
mod panels;
// The precision authored values are written at.
mod round;
// Pure resolution for the /select console command.
mod select_related;
// The viewport selection, an ordered set of `AssetHandle`s.
mod selection;
// Per-project session state, persisted as one small CBOR file.
mod session_store;
// The Play / Pause / Step / Stop transport over the preview world. Pure
// state; the hook drives it.
mod sim;
// A multi-line code text surface: its pure editing model and its layout.
mod text_area;
// The chrome's shared palette and metrics, so every surface reads as one.
mod theme;
// The editor's view of the thumbnail set a build baked into the cache.
mod thumbs;
// The toast stack: the engine widget the notifications show on.
mod toasts;
// The editor's panels built from engine widgets.
mod ui;
// The Display menu: the viewport view mode and the per-session show flags.
mod view_menu;
// Interaction over the scene rather than over a panel: the gizmo and its
// snapping, box-select, the highlight, the world-space furniture, the
// camera math.
mod viewport;
// Pure composition of the two hide mechanisms, the manual set and an isolate.
mod visibility;
// The Worlds panel and the start screen it becomes with no world open.
mod worlds;

use concinnity_cook::authoring::world::APP_JSONL;
use concinnity_engine::ecs::World;
use concinnity_engine::launch::{LaunchArgs, LaunchRequest};
use concinnity_engine::shutdown::ShutdownToken;
use concinnity_runtime::Runtime;
use hook::EditorHook;

use crate::frame_hook::FrameHook;

// A lone Window booted when there is nothing renderable to load, so the editor
// still opens. Never added to the authored entries.
const SEED_WINDOW: &str = "[\"Window\",{\"$id\":\"editor_default_window\"}]";

/// Editor entry point (`cn edit`). Compiles the authored world in memory,
/// sets the editor up in it, and runs the world loop driven by the editor hook
/// (plus the debug server when the launch names a port).
pub(crate) fn run_editor(args: LaunchArgs, json_path: Option<&str>) -> std::io::Result<()> {
    let debug_port = args.debug_port;
    // Every editor session hot-reloads, with or without a debug port.
    let launch = LaunchRequest {
        dev_loop: true,
        args,
    };
    // Instead of the engine's plain `init_logging`: the same stderr formatter
    // plus a layer mirroring this crate's events into the Console panel's log.
    // The sink exists first so even boot-time errors reach the panel.
    let console_sink = panels::console::ConsoleSink::default();
    panels::console::install_tracing(console_sink.clone(), launch.args.quiet);

    // Resolve the edit target -- the .jsonl file where readable names live and
    // where SAVE writes -- and whether the session opens on the Worlds panel
    // instead of a world.
    let (world_path, pick_a_world) = resolve_edit_target(json_path);

    // Parse the entries up front. A start-screen session boots on nothing and
    // compiles the most recent world behind the listing a few frames later.
    let (entries, previewing) = if pick_a_world {
        (entry_list::EntryList::default(), start_screen_pick())
    } else {
        let entries = worlds::files::read_entries(
            std::path::Path::new(&world_path),
            &crate::project::root_or_cwd(),
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        (entries, None)
    };

    // Bring up a renderable world by compiling those entries, seeding a render
    // marker when they alone would not render.
    let mut runtime = crate::project::runtime().with_launch(launch);
    let root = crate::project::root_or_cwd();
    let origin = entry_list::Origin {
        world: std::path::Path::new(&world_path),
        root: &root,
    };
    boot_world(&mut runtime, &entries, origin)?;

    // Set the editor up in the world before start (this also drops the world's
    // DebugHud, whose F1 role the editor takes over); the editor's FrameHook
    // tick drives its UI each frame.
    install::editor_ui(runtime.world_mut());

    // One hot-reload driver per session: the DebugServer's with a debug port,
    // else its own hook.
    let mut editor_hook = EditorHook::new(world_path, entries)
        .with_console_sink(console_sink)
        .with_app_menu();
    if pick_a_world {
        editor_hook = editor_hook.with_start_screen(previewing);
    }
    let hook: Box<dyn FrameHook> = if let Some(port) = debug_port {
        let server = crate::debug::DebugServer::start(port)?
            .with_notifier(editor_hook.notifier())
            .with_world_path(editor_hook.world_path_handle())
            .with_reload_reports(editor_hook.reload_reports());
        MultiHook::boxed(vec![Box::new(editor_hook), Box::new(server)])
    } else {
        let reload = crate::debug::hot_reload::HotReloadDriver::new()
            .with_notifier(editor_hook.notifier())
            .with_world_path(editor_hook.world_path_handle())
            .with_reload_reports(editor_hook.reload_reports());
        MultiHook::boxed(vec![Box::new(editor_hook), Box::new(reload)])
    };

    crate::run::start_app(runtime, Some(hook))
}

// The world the editor opens and whether it opens on the Worlds panel. An
// explicit path loads at once, present or not; without one the Worlds panel
// picks, and SAVE meanwhile lands at the project root.
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
    let world = worlds::files::newest(crate::project::content_root().as_deref())?;
    Some(world.path.to_string_lossy().into_owned())
}

// Where the editor puts a world nobody has saved yet.
pub(crate) fn unsaved_world_path() -> String {
    crate::project::content_root().map_or_else(
        || APP_JSONL.to_string(),
        |dir| dir.join(APP_JSONL).to_string_lossy().into_owned(),
    )
}

// Populate `runtime` with a renderable world for editing, compiled from the
// authored entries in memory. Nothing under the build root is read: the blobs
// there are refreshed only by an explicit build, so they may lag the file
// the editor is opening.
fn boot_world(
    runtime: &mut Runtime,
    entries: &entry_list::EntryList,
    origin: entry_list::Origin<'_>,
) -> std::io::Result<()> {
    let jsonl = entries.build_text(origin)?;
    let (world, _) = build_renderable(&jsonl)?;
    runtime.load_world(world);
    Ok(())
}

// Compile .jsonl content into a world plus template baselines, seeding a
// Window when it would not render. Boot and every rebuild come through here.
fn build_renderable(
    jsonl: &str,
) -> std::io::Result<(World, Vec<concinnity_cook::build_only::ShadowedAsset>)> {
    match crate::authoring::build_world_and_shadows(jsonl) {
        Ok(built) if built.0.renders() => Ok(built),
        _ => crate::authoring::build_world_and_shadows(&seeded_content(jsonl)),
    }
}

// Guarantee something to draw: append the seed Window to the authored content
// (only reached when the world does not otherwise render, so there is no
// existing Window to collide with).
fn seeded_content(base: &str) -> String {
    if base.trim().is_empty() {
        SEED_WINDOW.to_string()
    } else {
        format!("{base}\n{SEED_WINDOW}")
    }
}

// Fan a single per-frame drive out to several hooks. Lets the editor run its own
// hook and the debug server side by side without either owning the other.
struct MultiHook {
    hooks: Vec<Box<dyn FrameHook>>,
}

impl MultiHook {
    fn boxed(hooks: Vec<Box<dyn FrameHook>>) -> Box<dyn FrameHook> {
        Box::new(Self { hooks })
    }
}

impl FrameHook for MultiHook {
    fn tick(&mut self, world: &mut World) {
        for hook in &mut self.hooks {
            hook.tick(world);
        }
    }

    fn apply_world_swap(&mut self, runtime: &mut Runtime) {
        for hook in &mut self.hooks {
            hook.apply_world_swap(runtime);
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
    use concinnity_cook::authoring::world::{parse_world_jsonl, write_world_jsonl};
    use concinnity_engine::components::TextLabel;

    // An empty (or whitespace-only) world seeds to just the render marker, so an
    // empty session still opens a window.
    #[test]
    fn seeded_content_of_empty_is_the_render_marker() {
        assert_eq!(seeded_content(""), SEED_WINDOW);
        assert_eq!(seeded_content("   \n"), SEED_WINDOW);
    }

    // Authored content keeps its entries and gains the render marker on its own
    // line, so the combined string still parses as one asset per line.
    #[test]
    fn seeded_content_appends_marker_to_authored_content() {
        let base = "[\"PhysicsConfig\",{\"$id\":\"phys\"}]";
        let seeded = seeded_content(base);
        let parsed = parse_world_jsonl(&seeded).unwrap();
        assert_eq!(parsed.len(), 2, "authored entry plus the seed marker");
        assert_eq!(parsed[0]["args"]["$id"], "phys");
        assert_eq!(parsed[1]["type"], "Window");
    }

    // The seed marker is itself a well-formed, renderable asset line.
    #[test]
    fn seed_marker_is_a_window() {
        let parsed = parse_world_jsonl(SEED_WINDOW).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["type"], "Window");
    }

    // A project rooted at `dir`, as `cn` opens one, with the machine-wide cache
    // so a boot compiles shaders once. Returns the build root.
    fn open_project(dir: &std::path::Path) -> std::path::PathBuf {
        crate::project::open(
            concinnity_host::store::paths::StateTree::at(dir).with_cache(
                concinnity_testing::shared_cache_dir("concinnity-dev-tests-cache"),
            ),
        );
        dir.to_path_buf()
    }

    fn entry(name: &str, ty: &str, args: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"type": ty, "args": concinnity_cook::authoring::world::args_with_id(args, name)})
    }

    // A renderable authored world, plus a label whose content identifies which
    // compile a booted world came from.
    // Boot `entries` as the open project's file.
    fn boot(runtime: &mut Runtime, entries: Vec<serde_json::Value>) -> std::io::Result<()> {
        let root = crate::project::root_or_cwd();
        let world = root.join("world.jsonl");
        let origin = entry_list::Origin {
            world: &world,
            root: &root,
        };
        boot_world(runtime, &entry_list::EntryList::new(entries), origin)
    }

    fn renderable_entries(label: &str) -> Vec<serde_json::Value> {
        vec![
            entry("cam", "Camera3D", serde_json::json!({})),
            entry("room", "Room", serde_json::json!({})),
            entry("hint", "TextLabel", serde_json::json!({"content": label})),
        ]
    }

    // The content of the booted world's only TextLabel.
    fn booted_label(runtime: &Runtime) -> String {
        runtime
            .world()
            .query::<TextLabel>()
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

        let mut runtime = crate::project::runtime();
        boot(&mut runtime, renderable_entries("authored")).expect("the world builds");

        assert!(runtime.world().renders());
        assert_eq!(booted_label(&runtime), "authored");
        assert!(
            !build_root.join("data").exists()
                && !crate::project::tree()
                    .expect("the test opened a project")
                    .cook_lock_path()
                    .exists(),
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

        // An explicit build, as `cn cook` runs it, over the stale world.
        let world_path = dir.path().join(APP_JSONL);
        std::fs::write(
            &world_path,
            write_world_jsonl(&renderable_entries("stale")).unwrap(),
        )
        .unwrap();
        let content = std::fs::read_to_string(&world_path).unwrap();
        crate::authoring::build_world_str_to_disk(&content, None).expect("the build writes blobs");
        let blob = concinnity_host::store::blob::primary_in(&build_root.join("data"));
        let before = std::fs::read(&blob).expect("the build wrote a primary blob");

        let mut runtime = crate::project::runtime();
        boot(&mut runtime, renderable_entries("edited")).expect("the world builds");

        assert_eq!(
            booted_label(&runtime),
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

        let mut runtime = crate::project::runtime();
        boot(&mut runtime, Vec::new()).expect("an empty world seeds");
        assert!(runtime.world().renders());
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

    // With no file named, the editor's unsaved one sits at the project root,
    // where the application `init` starts would.
    #[test]
    fn an_unsaved_world_is_named_at_the_project_root() {
        // Reading the session's project; opening one is what the guard covers.
        let _guard = crate::test_support::lock();
        crate::test_support::isolate_state_dir();

        let root = crate::project::content_root().expect("the harness opened a project");
        assert_eq!(unsaved_world_path(), root.join(APP_JSONL).to_string_lossy());
    }
}
