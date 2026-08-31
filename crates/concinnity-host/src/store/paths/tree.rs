// The state tree as a value: a content root, the three roots that may be split
// away from it, and the layout hanging off all four.
//
// Every directory and file name the tree is made of is spelled once, here. A
// caller asks for `saves_dir()` or `build_cache_path()`; that `cache/1` is a
// build's segment is this module's knowledge and nobody else's.

use std::path::{Path, PathBuf};

// The tree's layout, spelled once. Private: a caller asks the tree for a path
// rather than for a segment's name, so nothing outside this module has to know
// that a build's cache is `cache/1`.
const ASSETS_DIR: &str = "assets";
const DATA_DIR: &str = "data";
const WORLD_LOCK_FILE: &str = "world-lock.json";
const WORLDS_DIR: &str = "worlds";
const SAVES_DIR: &str = "saves";
const PREVIEW_SAVES_DIR: &str = "preview-saves";
const SETTINGS_FILE: &str = "settings";
const CRASHES_DIR: &str = "crashes";
const EDITOR_SESSION_FILE: &str = "editor";
const CACHE_DIR: &str = "cache";
const RUNTIME_CACHE_SEGMENT: &str = "0";
const BUILD_CACHE_SEGMENT: &str = "1";

/// Where a project's state lives, and what hangs off it.
///
/// Built by whatever runs the process -- the dev CLI from its project
/// directory, a shipped application from the directory beside its executable,
/// an embedder from whatever its own layout implies -- and passed down. Library
/// code is handed one; it never resolves a root for itself.
///
/// Four roots, because the four fall apart on real installs:
///
/// - the **content** root holds what a person authors (`assets/`, `worlds/`),
/// - the **build** root holds what a build produces (`data/`, the world lock),
///   split away when the authored content should stay visible and the output
///   should not,
/// - the **writable** root holds what the running application writes
///   (`saves/`, `settings`, `crashes/`), split away when the content root is a
///   read-only install such as Program Files,
/// - the **cache** root holds the regenerable segments, split away when the
///   caches should outlive (or be shared across) the content beside them.
///
/// An unsplit tree resolves all four at the content root, which is the
/// single-folder layout a portable install uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateTree {
    content: PathBuf,
    build: Option<PathBuf>,
    writable: Option<PathBuf>,
    cache: Option<PathBuf>,
}

impl StateTree {
    /// A tree with every root at `content`.
    pub fn at<P: Into<PathBuf>>(content: P) -> Self {
        Self {
            content: content.into(),
            build: None,
            writable: None,
            cache: None,
        }
    }

    /// Move what a build produces to `dir`, leaving the authored content where
    /// it is: a dev project keeps `assets/` and `worlds/` in sight and its
    /// blobs, lock, and caches out of the way.
    #[must_use]
    pub fn with_build<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.build = Some(dir.into());
        self
    }

    /// Move the runtime-writable state to `dir`, leaving the content where it
    /// is: a read-only install writes its saves and settings per-user.
    #[must_use]
    pub fn with_writable<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.writable = Some(dir.into());
        self
    }

    /// Move both cache segments to `dir`. Anchors the regenerable artifacts
    /// away from the content and the writable state, so a warm cache can sit
    /// behind a content root that is read-only, or freshly built, or both.
    #[must_use]
    pub fn with_cache<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.cache = Some(dir.into());
        self
    }

    /// The root holding what a person authors.
    pub fn content_root(&self) -> &Path {
        &self.content
    }

    /// The root holding what a build produces.
    pub fn build_root(&self) -> &Path {
        self.build.as_deref().unwrap_or(&self.content)
    }

    /// The root holding what the running application writes.
    pub fn writable_root(&self) -> &Path {
        self.writable.as_deref().unwrap_or(&self.content)
    }

    /// The content root's `assets/` directory, holding authored source files.
    pub fn assets_dir(&self) -> PathBuf {
        self.content.join(ASSETS_DIR)
    }

    /// The content root's `worlds/` directory, holding authored worlds.
    pub fn worlds_dir(&self) -> PathBuf {
        self.content.join(WORLDS_DIR)
    }

    /// The build root's `data/` directory, holding the blobs a build writes
    /// and a run reads.
    pub fn data_dir(&self) -> PathBuf {
        self.build_root().join(DATA_DIR)
    }

    /// The record a build writes beside its blobs: what went into them, and
    /// what the build injected that no authored line asked for.
    pub fn world_lock_path(&self) -> PathBuf {
        self.build_root().join(WORLD_LOCK_FILE)
    }

    /// Directory holding the runtime save files. Created on first write by the
    /// running application, never by a build.
    pub fn saves_dir(&self) -> PathBuf {
        self.writable_root().join(SAVES_DIR)
    }

    /// Sandboxed sibling of [`saves_dir`](Self::saves_dir) for preview
    /// sessions: the save UI keeps working against this directory, the real
    /// saves are never touched, and the sandbox is wiped at each session start.
    pub fn preview_saves_dir(&self) -> PathBuf {
        self.writable_root().join(PREVIEW_SAVES_DIR)
    }

    /// The mutable settings file, written by the in-engine settings menu and
    /// never by a build.
    pub fn settings_path(&self) -> PathBuf {
        self.writable_root().join(SETTINGS_FILE)
    }

    /// Directory holding crash reports. Created on first write; capped by the
    /// writer's retention pruning.
    pub fn crashes_dir(&self) -> PathBuf {
        self.writable_root().join(CRASHES_DIR)
    }

    /// The editor's session store: the per-project state an editor run carries
    /// between launches, which is state rather than cache.
    pub fn editor_session_path(&self) -> PathBuf {
        self.writable_root().join(EDITOR_SESSION_FILE)
    }

    /// The segment a running application writes: one container holding every
    /// regenerable artifact it produces for its own later launches, indexed by
    /// producer and key.
    ///
    /// Deletable at any time; whatever is missing is recomputed. The
    /// application writes this file and no other, so a concurrent build writing
    /// a segment of its own never shares a file with it.
    pub fn runtime_cache_path(&self) -> PathBuf {
        self.cache_root_for_runtime()
            .join(CACHE_DIR)
            .join(RUNTIME_CACHE_SEGMENT)
    }

    /// The runtime segment a bundle ships, read-only. `cn export` warms it with
    /// the shader binaries a first launch would otherwise compile; because those
    /// artifacts are backend IR (DXBC / SPIR-V) rather than machine code, one
    /// warmed at package time is valid on any machine.
    ///
    /// Always resolves against the build root, so it ships with the blobs it
    /// was warmed for and stays readable on a read-only install. That is also
    /// the only thing separating it from
    /// [`runtime_cache_path`](Self::runtime_cache_path): a bundle the player
    /// can write to has one segment serving both roles.
    pub fn bundled_runtime_cache_path(&self) -> PathBuf {
        self.build_root()
            .join(CACHE_DIR)
            .join(RUNTIME_CACHE_SEGMENT)
    }

    /// The segment a build writes: one container holding every payload,
    /// expansion, and baked thumbnail a cook produced, indexed by producer and
    /// key.
    ///
    /// Deletable at any time; whatever is missing is recompiled.
    pub fn build_cache_path(&self) -> PathBuf {
        self.cache_root_for_build()
            .join(CACHE_DIR)
            .join(BUILD_CACHE_SEGMENT)
    }

    // Without a cache root the runtime segment follows what the application
    // writes, which is what keeps it writable on a read-only install.
    fn cache_root_for_runtime(&self) -> &Path {
        self.cache
            .as_deref()
            .unwrap_or_else(|| self.writable_root())
    }

    // Without a cache root the build segment follows the build root: a build
    // writes the `data/` beside it, so a tree it cannot write is a tree it
    // cannot cook into either.
    fn cache_root_for_build(&self) -> &Path {
        self.cache.as_deref().unwrap_or_else(|| self.build_root())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An unsplit tree: one folder, everything under it. The portable install
    // and the dev checkout.
    #[test]
    fn one_root_resolves_the_whole_layout() {
        let tree = StateTree::at("/flat");
        let root = Path::new("/flat");

        assert_eq!(tree.content_root(), root);
        assert_eq!(tree.build_root(), root);
        assert_eq!(tree.writable_root(), root);
        assert_eq!(tree.assets_dir(), root.join("assets"));
        assert_eq!(tree.data_dir(), root.join("data"));
        assert_eq!(tree.world_lock_path(), root.join("world-lock.json"));
        assert_eq!(tree.worlds_dir(), root.join("worlds"));
        assert_eq!(tree.saves_dir(), root.join("saves"));
        assert_eq!(tree.preview_saves_dir(), root.join("preview-saves"));
        assert_eq!(tree.settings_path(), root.join("settings"));
        assert_eq!(tree.crashes_dir(), root.join("crashes"));
        assert_eq!(tree.editor_session_path(), root.join("editor"));
        assert_eq!(tree.runtime_cache_path(), root.join("cache").join("0"));
        assert_eq!(tree.build_cache_path(), root.join("cache").join("1"));
        // One file in both runtime roles, which is what makes the bundled tier
        // vacuous for a bundle the player can write to.
        assert_eq!(tree.bundled_runtime_cache_path(), tree.runtime_cache_path());
    }

    // A read-only install: only what the application writes moves. The content
    // (and the segment a build writes into it) stays put.
    #[test]
    fn a_writable_root_moves_only_what_the_application_writes() {
        let content = Path::new("/opt/MyGame");
        let writable = Path::new("/home/u/.local/share/MyGame");
        let tree = StateTree::at(content).with_writable(writable);

        assert_eq!(tree.content_root(), content);
        assert_eq!(tree.writable_root(), writable);
        assert_eq!(tree.saves_dir(), writable.join("saves"));
        assert_eq!(tree.preview_saves_dir(), writable.join("preview-saves"));
        assert_eq!(tree.settings_path(), writable.join("settings"));
        assert_eq!(tree.crashes_dir(), writable.join("crashes"));
        assert_eq!(tree.editor_session_path(), writable.join("editor"));
        assert_eq!(tree.runtime_cache_path(), writable.join("cache").join("0"));

        assert_eq!(tree.data_dir(), content.join("data"));
        assert_eq!(tree.world_lock_path(), content.join("world-lock.json"));
        assert_eq!(tree.assets_dir(), content.join("assets"));
        assert_eq!(tree.worlds_dir(), content.join("worlds"));
        assert_eq!(tree.build_cache_path(), content.join("cache").join("1"));
        // The shipped segment stays with the content, which is what keeps a
        // read-only install's warmed artifacts readable.
        assert_eq!(
            tree.bundled_runtime_cache_path(),
            content.join("cache").join("0")
        );
    }

    // A cache root moves both regenerable segments and nothing else: the point
    // of the split is a warm cache behind content that is fresh, read-only, or
    // both.
    #[test]
    fn a_cache_root_moves_both_segments_and_nothing_else() {
        let content = Path::new("/build/content");
        let cache = Path::new("/var/cache/mygame");
        let tree = StateTree::at(content).with_cache(cache);

        assert_eq!(tree.runtime_cache_path(), cache.join("cache").join("0"));
        assert_eq!(tree.build_cache_path(), cache.join("cache").join("1"));
        // Still the shipped tier's own definition: beside the content.
        assert_eq!(
            tree.bundled_runtime_cache_path(),
            content.join("cache").join("0")
        );
        assert_eq!(tree.data_dir(), content.join("data"));
        assert_eq!(tree.saves_dir(), content.join("saves"));
    }

    // All four split: the case one knob could never express, and the reason
    // each is a root of its own.
    #[test]
    fn all_four_roots_split_independently() {
        let tree = StateTree::at("/opt/app")
            .with_build("/opt/app/out")
            .with_writable("/home/u/app")
            .with_cache("/var/cache/app");

        assert_eq!(tree.assets_dir(), Path::new("/opt/app/assets"));
        assert_eq!(tree.data_dir(), Path::new("/opt/app/out/data"));
        assert_eq!(tree.saves_dir(), Path::new("/home/u/app/saves"));
        assert_eq!(
            tree.runtime_cache_path(),
            Path::new("/var/cache/app/cache/0")
        );
        assert_eq!(tree.build_cache_path(), Path::new("/var/cache/app/cache/1"));
    }

    // The dev layout: authored content in sight at the project root, every
    // derived byte under one hidden directory beside it.
    #[test]
    fn a_build_root_hides_the_output_and_leaves_the_authored_content_visible() {
        let project = Path::new("/proj");
        let hidden = Path::new("/proj/.concinnity");
        let tree = StateTree::at(project)
            .with_writable(hidden)
            .with_build(hidden);

        assert_eq!(tree.assets_dir(), project.join("assets"));
        assert_eq!(tree.worlds_dir(), project.join("worlds"));

        assert_eq!(tree.data_dir(), hidden.join("data"));
        assert_eq!(tree.world_lock_path(), hidden.join("world-lock.json"));
        assert_eq!(tree.settings_path(), hidden.join("settings"));
        assert_eq!(tree.saves_dir(), hidden.join("saves"));
        assert_eq!(tree.editor_session_path(), hidden.join("editor"));
        // Both segments follow without a cache root of their own, which is what
        // keeps a `cache/` out of the project root.
        assert_eq!(tree.runtime_cache_path(), hidden.join("cache").join("0"));
        assert_eq!(tree.build_cache_path(), hidden.join("cache").join("1"));
        assert_eq!(
            tree.bundled_runtime_cache_path(),
            hidden.join("cache").join("0")
        );
    }

    // A cache root still outranks the build root for the segment a build
    // writes: the two splits answer different questions.
    #[test]
    fn a_cache_root_outranks_the_build_root() {
        let tree = StateTree::at("/proj")
            .with_build("/proj/.concinnity")
            .with_cache("/var/cache/app");

        assert_eq!(tree.build_cache_path(), Path::new("/var/cache/app/cache/1"));
        assert_eq!(tree.data_dir(), Path::new("/proj/.concinnity/data"));
    }

    // The builders are independent: setting one leaves the others resolving at
    // their own defaults.
    #[test]
    fn builders_do_not_disturb_each_other() {
        let base = StateTree::at("/root");
        assert_eq!(
            base.clone().with_cache("/c").saves_dir(),
            base.saves_dir(),
            "a cache root leaves the writable state alone"
        );
        assert_eq!(
            base.clone().with_writable("/w").build_cache_path(),
            base.build_cache_path(),
            "a writable root leaves the build segment with the content"
        );
        assert_eq!(
            base.clone().with_build("/b").assets_dir(),
            base.assets_dir(),
            "a build root leaves the authored content where it is"
        );
        assert_eq!(
            base.clone().with_build("/b").saves_dir(),
            base.saves_dir(),
            "a build root leaves the writable state alone"
        );
    }
}
