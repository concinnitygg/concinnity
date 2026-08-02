// src/debug/hot_reload/watcher.rs
//
// Filesystem watcher: subscribes to the parent directories of every captured
// source path and flips the shared atomic on a relevant change. Mirrors the
// per-backend shader watcher.

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gfx::graphics_system::hot_reload_sources::*;

// Spawn the watcher. Mirrors the shader-watcher pattern in
// `concinnity_device::metal::hot_reload`: 150 ms debounce, only
// modify/create/remove events fire the flag, only relevant extensions count.
pub(super) fn spawn_watcher(
    sources: &HotReloadSources,
    flag: Arc<AtomicBool>,
) -> Option<notify::RecommendedWatcher> {
    // Procedural meshes are generated, not sourced from a file, so they have no
    // directory to watch.
    let HotReloadSources {
        map,
        color_lut,
        environment_map,
        meshes,
        skinned_meshes,
        procedural_meshes: _,
        shader_stages,
        world_jsonl_path,
    } = sources;
    let debounce = Duration::from_millis(150);
    let last_fire = Mutex::new(Instant::now() - debounce);
    let mut watcher = match notify::recommended_watcher(move |res: notify::Result<Event>| {
        let event = match res {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!("asset hot-reload watcher error: {e}");
                return;
            }
        };
        let Some(kind) = classify_event(&event) else {
            return;
        };
        let mut last = match last_fire.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = Instant::now();
        if now.duration_since(*last) < debounce {
            return;
        }
        *last = now;
        tracing::info!(
            "asset hot-reload: detected change to {:?}, scheduling {kind:?} reload",
            event.paths
        );
        signal(kind, &flag);
    }) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("asset hot-reload: failed to create notify watcher: {e}");
            return None;
        }
    };

    // Build the unique-dir set across textures + the optional LUT + the
    // optional EnvironmentMap so the watcher subscribes to each path at most
    // once.
    let mut dirs: BTreeSet<PathBuf> = map.watch_dirs().into_iter().collect();
    if let Some(lut) = color_lut
        && let Some(parent) = Path::new(&lut.resolved_path).parent()
        && !parent.as_os_str().is_empty()
    {
        dirs.insert(parent.to_path_buf());
    }
    if let Some(env_map) = environment_map
        && let Some(parent) = Path::new(&env_map.resolved_path).parent()
        && !parent.as_os_str().is_empty()
    {
        dirs.insert(parent.to_path_buf());
    }
    for dir in meshes.watch_dirs() {
        dirs.insert(dir);
    }
    for dir in skinned_meshes.watch_dirs() {
        dirs.insert(dir);
    }
    for dir in shader_stages.watch_dirs() {
        dirs.insert(dir);
    }
    if let Some(path) = world_jsonl_path {
        if let Some(parent) = Path::new(path).parent() {
            // An empty parent means the path was already a bare filename in
            // CWD; subscribe to "." in that case so the same `notify` events
            // fire as for any directoried path.
            let dir = if parent.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                parent.to_path_buf()
            };
            dirs.insert(dir);
        }
        // The world's StoryImport sources: a `.md` save in one of these
        // directories re-expands the stories in place.
        dirs.extend(story_source_dirs(path));
    }
    let mut any_watched = false;
    for dir in dirs {
        match watcher.watch(&dir, RecursiveMode::NonRecursive) {
            Ok(()) => {
                tracing::info!(
                    "asset hot-reload: watching {} for asset source changes",
                    dir.display()
                );
                any_watched = true;
            }
            Err(e) => {
                tracing::warn!(
                    "asset hot-reload: failed to watch {} ({}); assets sourced from \
                     that directory will need a manual `reload-assets` to refresh",
                    dir.display(),
                    e
                );
            }
        }
    }
    if any_watched {
        Some(watcher)
    } else {
        // None of the directories could be watched (likely a packaged binary
        // run from outside its checkout). The debug command path still works.
        None
    }
}

// The reload pass a filesystem change kicks. Each save is routed to the
// narrowest pass that can serve it: the backend asset payloads (textures, IBL,
// meshes, skinned, animations) do not live in the world JSONL, which carries
// only Prop transforms and the asset-graph topology, and a shader save needs a
// recompile plus a pipeline rebuild but no texture or mesh decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReloadKind {
    // `.metal` / `.hlsl` / `.glsl`.
    ShaderStages,
    // `.jsonl`, the world file.
    World,
    // `.md`: re-expand the world's StoryImports and hand the fresh graphs to
    // the running story system.
    Stories,
    // `.glb` / `.png` / `.hdr` / `.cube` and the rest: decode the payloads
    // again, and the animation graph alongside them.
    Assets,
}

// The pass `event` kicks, or `None` when it is not a change worth reloading
// for. Pure, so the routing is decided the same way whether it comes from a
// live notify callback or a test.
pub(super) fn classify_event(event: &Event) -> Option<ReloadKind> {
    if !is_asset_event(event) {
        return None;
    }
    let has_ext = |matches: fn(&str) -> bool| {
        event.paths.iter().any(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(matches)
                .unwrap_or(false)
        })
    };
    Some(if has_ext(is_shader_extension) {
        ReloadKind::ShaderStages
    } else if has_ext(|e| e.eq_ignore_ascii_case("jsonl")) {
        ReloadKind::World
    } else if has_ext(|e| e.eq_ignore_ascii_case("md")) {
        ReloadKind::Stories
    } else {
        ReloadKind::Assets
    })
}

// Raise the pending flag for `kind`. Separate from `classify_event` because
// these are process-global statics: the routing above is what a test drives.
fn signal(kind: ReloadKind, flag: &AtomicBool) {
    match kind {
        ReloadKind::ShaderStages => super::set_pending_shader_stages(),
        ReloadKind::World => super::set_pending_world(),
        ReloadKind::Stories => super::set_pending_stories(),
        ReloadKind::Assets => {
            flag.store(true, Ordering::SeqCst);
            // AnimationSystem subscribes via a sibling static flag in
            // crate::app::dev_flags; the asset map lives on GraphicsSystem so
            // a separate signal is the simplest way to notify the animation
            // graph of the same `.glb` save without plumbing a shared Arc.
            crate::app::dev_flags::set_pending_animations();
        }
    }
}

// The parent directories of every StoryImport source declared in the world
// file, so the watcher hears `.md` saves. Read from the raw JSONL (one asset
// declaration per line); a malformed line is skipped like the build would
// reject it later.
pub(super) fn story_source_dirs(world_jsonl_path: &str) -> BTreeSet<PathBuf> {
    let mut dirs = BTreeSet::new();
    let Ok(content) = std::fs::read_to_string(world_jsonl_path) else {
        return dirs;
    };
    for line in content.lines() {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if entry.get("type").and_then(|t| t.as_str()) != Some("StoryImport") {
            continue;
        }
        let Some(source) = entry
            .get("args")
            .and_then(|a| a.get("source"))
            .and_then(|s| s.as_str())
        else {
            continue;
        };
        let parent = Path::new(source).parent().unwrap_or(Path::new(""));
        dirs.insert(if parent.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            parent.to_path_buf()
        });
    }
    dirs
}

// Filter notify events down to those that should kick a reload. We don't
// peek at the exact path against the map here because notify often reports
// paths via temp files / rename sidecars; a coarse extension match is good
// enough at V1 scale and gets debounced anyway. `.cube` covers ColorLut
// sources, `.hdr` covers EnvironmentMap sources, `.jsonl` covers the world
// file, `.md` covers StoryImport sources, `.bin` covers a text `.gltf`'s
// external buffers, alongside the texture / mesh extensions.
pub(super) fn is_asset_event(event: &Event) -> bool {
    if !matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    ) {
        return false;
    }
    event.paths.iter().any(|p| {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(
            ext.to_ascii_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "glb" | "gltf" | "bin" | "cube" | "hdr" | "jsonl" | "md"
        ) || is_shader_extension(ext)
    })
}

// True for the shader-source extensions recognised by world-loaded
// `Shader` hot-reload. Case-insensitive so a `.METAL` save still
// triggers the rebuild. `.metal` files in the engine's bundled shader
// directory are handled by a separate watcher in
// [`crate::metal::hot_reload`]; the asset watcher here only subscribes
// to the parent directories of *captured* Shader stage sources, so the
// two watchers never observe the same file even though they share an
// extension list.
pub(super) fn is_shader_extension(ext: &str) -> bool {
    matches!(ext.to_ascii_lowercase().as_str(), "metal" | "hlsl" | "glsl")
}
