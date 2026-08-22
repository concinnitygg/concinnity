// Application distribution metadata schema. The runtime component keeps only
// the resource budgets (see the core assets module); everything else here is
// consumed at build / export time and never ships in the blob.

use alloc::string::{String, ToString};

/// Names and identifies the application for distribution.
///
/// Declare at most one `Application` per world. It supplies the display name,
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ApplicationArgs {
    /// Display name of the application: the game's window title, the exported
    /// archive and executable name, and the macOS bundle display name.
    pub name: String,
    /// Reverse-DNS bundle identifier (e.g. `gg.studio.mygame`). When empty the
    /// export derives one from `name`.
    pub id: String,
    /// Human-readable version string (e.g. `1.0.0`).
    pub version: String,
    /// Author or studio name, recorded in the exported bundle's metadata.
    pub author: String,
    /// Path to a source icon image (a square PNG, 512x512 or larger) relative
    /// to the world, used to build the platform icon at export time. Empty for
    /// no custom icon.
    pub icon: String,
    /// Optional overrides for the runtime's process resource budgets. When
    /// omitted (or left at their defaults) the engine sizes both from the host
    /// machine.
    pub limits: AppLimits,
}

/// Optional per-application overrides for the runtime's thread and memory
/// budgets. Each field of `0` means "auto" (the engine picks a value from the
/// host machine); a non-zero value overrides that choice, clamped to what the
/// machine can safely give.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AppLimits {
    /// Soft ceiling on host memory the runtime aims to stay under, in
    /// mebibytes. `0` = auto (a fraction of total RAM, capped by a built-in
    /// ceiling). A non-zero value is clamped so it never exceeds what the
    /// machine can safely give.
    pub max_memory_mb: u32,
    /// Worker threads for the shared job pool. `0` = auto (one per core, less
    /// one for the main thread). A non-zero value never exceeds the core count.
    pub job_threads: u32,
}

impl Default for ApplicationArgs {
    fn default() -> Self {
        Self {
            name: "Concinnity".to_string(),
            id: String::new(),
            version: "0.1.0".to_string(),
            author: String::new(),
            icon: String::new(),
            limits: AppLimits::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_name_an_unversioned_unlimited_application() {
        let a = ApplicationArgs::default();
        assert_eq!(a.name, "Concinnity");
        assert_eq!(a.version, "0.1.0");
        assert!(a.id.is_empty());
        assert!(a.author.is_empty());
        assert!(a.icon.is_empty());
        // Zero is "no budget declared", not "no memory and no threads".
        assert_eq!(a.limits.max_memory_mb, 0);
        assert_eq!(a.limits.job_threads, 0);
    }

    #[test]
    fn export_metadata_parses_and_round_trips_through_postcard() {
        let a: ApplicationArgs = serde_json::from_str(
            r#"{"name":"Ash","id":"com.example.ash","version":"1.2.0","author":"Grant",
                "icon":"icon.png","limits":{"max_memory_mb":2048,"job_threads":8}}"#,
        )
        .unwrap();
        assert_eq!(a.id, "com.example.ash");
        assert_eq!(a.limits.job_threads, 8);

        // Only the budgets ship in the blob, but the whole struct is what the
        // export step reads back, so it has to survive the baked format.
        let bytes = postcard::to_allocvec(&a).unwrap();
        let back: ApplicationArgs = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.name, "Ash");
        assert_eq!(back.version, "1.2.0");
        assert_eq!(back.author, "Grant");
        assert_eq!(back.icon, "icon.png");
        assert_eq!(back.limits.max_memory_mb, 2048);
    }
}
