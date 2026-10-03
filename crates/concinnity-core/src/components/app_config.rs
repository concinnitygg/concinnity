//! The `AppConfig` asset. Only what the running process needs survives
//! the bake: the state-tree location and the resource budgets. The distribution
//! metadata (name, id, version, author, icon) is read at build / export time
//! from the authored world, so it never ships in the blob.

use alloc::string::String;

/// Names, identifies, and sizes the application.
///
/// Declare at most one `AppConfig` per world. It supplies the display name,
/// bundle identifier, version, author, and icon that the export step reads when
/// it packages the world into a distributable game (the archive and executable
/// name, and on macOS the `.app` bundle metadata and icon). When the world
/// declares no [Window](#window) title of its own, `name` also fills the window
/// title, so a running game shows its own name in the title bar.
///
/// `icon` is a path to a source image (a square PNG, 512x512 or larger)
/// relative to the world; it is read by the packaging step and is not compiled
/// into the world's data. `id` is a reverse-DNS bundle identifier
/// (e.g. `gg.studio.mygame`); when left empty the export derives one from
/// `name`. Empty string fields mean "unset".
///
/// `home` chooses where the running application keeps what it writes: the
/// settings file, the save files, crash reports, and the shader caches. Leave
/// it empty and those sit beside the application's data, which is what a
/// portable install wants. A relative path resolves against that same content
/// directory, so `"state"` puts them in a `state/` subfolder; an absolute path
/// is used verbatim. A read-only install that sets no `home` relocates them to
/// a per-user directory on its own.
///
/// `max_memory_mb` and `job_threads` are `0` for "auto", where the engine sizes
/// both from the host machine. A non-zero value overrides that choice, clamped
/// to what the machine can safely give.
///
/// `headless` keeps a world that could draw from opening a window. A world
/// that draws nothing runs that way already.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct AppConfigArgs {
    /// Display name of the application: the game's window title, the exported
    /// archive and executable name, and the macOS bundle display name.
    #[asset(default = "Concinnity")]
    pub name: String,
    /// Reverse-DNS bundle identifier (e.g. `gg.studio.mygame`). When empty the
    /// export derives one from `name`.
    pub id: String,
    /// Human-readable version string (e.g. `1.0.0`).
    #[asset(default = "0.1.0")]
    pub version: String,
    /// Author or studio name, recorded in the exported bundle's metadata.
    pub author: String,
    /// Path to a source icon image (a square PNG, 512x512 or larger) relative
    /// to the world, used to build the platform icon at export time. Empty for
    /// no custom icon.
    pub icon: String,
    /// Where the running application writes its settings, saves, crash reports,
    /// and shader caches. Empty means beside the application's data; a relative
    /// path resolves against that directory; an absolute path is used verbatim.
    pub home: String,
    /// Soft ceiling on host memory the runtime aims to stay under, in
    /// mebibytes. `0` = auto (a fraction of total RAM, capped by a built-in
    /// ceiling). A non-zero value is clamped so it never exceeds what the
    /// machine can safely give.
    pub max_memory_mb: u32,
    /// Worker threads for the shared job pool. `0` = auto (one per core, less
    /// one for the main thread). A non-zero value never exceeds the core count.
    pub job_threads: u32,
    /// Run with no window and no renderer, whatever the world holds. `false`
    /// (the default) lets the content decide: a world with something to draw
    /// opens a window, one without runs headless either way.
    pub headless: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_keeps_the_home_the_budgets_and_headless() {
        let a = AppConfigArgs {
            name: "Pong".into(),
            home: "state".into(),
            max_memory_mb: 2048,
            job_threads: 8,
            headless: true,
            ..Default::default()
        };
        let c = AppConfig::bake(a);
        assert_eq!(c.home, "state");
        assert_eq!(c.max_memory_mb, 2048);
        assert_eq!(c.job_threads, 8);
        assert!(c.headless);
    }
}

/// Runtime half of the AppConfig asset: where the application keeps what it
/// writes, and its process resource budgets.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    /// Where the running application writes its settings, saves, crash
    /// reports, and shader caches. Empty means beside the application's data.
    pub home: String,
    /// Soft ceiling on host memory the runtime aims to stay under, in
    /// mebibytes. `0` = auto.
    pub max_memory_mb: u32,
    /// Worker threads for the shared job pool. `0` = auto.
    pub job_threads: u32,
    /// Run with no window and no renderer, whatever the world holds.
    pub headless: bool,
}

impl AppConfig {
    /// Translate the authored args into the runtime component: keep the state
    /// location and the resource budgets. Run by cook at build time (the baked
    /// blob record carries the result).
    pub fn bake(args: AppConfigArgs) -> Self {
        Self {
            home: args.home,
            max_memory_mb: args.max_memory_mb,
            job_threads: args.job_threads,
            headless: args.headless,
        }
    }
}
