// Application distribution metadata schema.

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
///
/// ```jsonl
/// {"name":"app","type":"Application","args":{"name":"My Game","id":"gg.studio.mygame","version":"1.0.0","author":"Studio","icon":"art/icon.png"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Application {
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
}

impl Default for Application {
    fn default() -> Self {
        Self {
            name: "Concinnity".to_string(),
            id: String::new(),
            version: "0.1.0".to_string(),
            author: String::new(),
            icon: String::new(),
        }
    }
}
