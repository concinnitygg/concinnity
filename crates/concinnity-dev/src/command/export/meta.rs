//! The app name, identifier, version and icon a bundle carries, read from the world and the command line.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::{WorldJsonlAsset, WorldSource, load_world};
use concinnity_cook::build_only::prepare_world;
use std::io;
use std::path::{Path, PathBuf};

use super::AppMeta;

// Read the app metadata a world file declares. The build already validated the
// world, so an error here is mapped plainly rather than reported per asset.
pub(super) fn world_app_meta(
    source: WorldSource<'_>,
    assets_dir: Option<&Path>,
    cli_name: Option<&str>,
    cli_version: Option<&str>,
) -> io::Result<AppMeta> {
    let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
    let loaded = prepare_world(source, assets_dir).map_err(|errs| invalid(errs.to_string()))?;
    let authored = load_world(source)
        .map_err(|errs| invalid(errs.join("\n")))?
        .iter()
        .map(WorldJsonlAsset::from_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid)?;
    Ok(read_app_meta(
        cli_name,
        cli_version,
        &loaded.assets,
        &authored,
    ))
}

// Read the app metadata from the expanded world, applying the `--name` /
// `--version` overrides and deriving anything the AppConfig asset left unset.
// `authored` is the world before expansion, which still holds its MainMenus.
pub(super) fn read_app_meta(
    cli_name: Option<&str>,
    cli_version: Option<&str>,
    assets: &[WorldJsonlAsset],
    authored: &[WorldJsonlAsset],
) -> AppMeta {
    let display_name = resolve_display_name(cli_name, assets, authored);
    let identifier = string_arg(assets, RegisteredType::AppConfig, "id")
        .unwrap_or_else(|| derive_identifier(&display_name));
    let version = cli_version
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| string_arg(assets, RegisteredType::AppConfig, "version"))
        .unwrap_or_else(|| "0.1.0".to_string());
    let icon = string_arg(assets, RegisteredType::AppConfig, "icon").map(PathBuf::from);
    AppMeta {
        display_name,
        identifier,
        version,
        icon,
    }
}

// The app name, by precedence: an explicit `--name`, then the AppConfig
// asset's name, then an authored MainMenu's title, then the engine default.
pub(super) fn resolve_display_name(
    cli_name: Option<&str>,
    assets: &[WorldJsonlAsset],
    authored: &[WorldJsonlAsset],
) -> String {
    if let Some(n) = cli_name.map(str::trim).filter(|s| !s.is_empty()) {
        return n.to_string();
    }
    if let Some(n) = string_arg(assets, RegisteredType::AppConfig, "name") {
        return n;
    }
    if let Some(n) = string_arg(authored, RegisteredType::MainMenu, "title") {
        return n;
    }
    "Concinnity".to_string()
}

// A reverse-DNS bundle identifier derived from the name when the AppConfig
// asset declares none: `gg.concinnity.<name>`, the name reduced to bundle-id
// characters (ascii alphanumerics, lowercased; other runs become a single `-`).
pub(super) fn derive_identifier(name: &str) -> String {
    let mut comp = String::new();
    let mut pending_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            if pending_dash && !comp.is_empty() {
                comp.push('-');
            }
            pending_dash = false;
            comp.push(c.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
    }
    let comp = comp.trim_matches('-');
    if comp.is_empty() {
        "gg.concinnity.app".to_string()
    } else {
        format!("gg.concinnity.{comp}")
    }
}

// The first non-empty string value of `key` on the first asset of `asset_type`.
pub(super) fn string_arg(
    assets: &[WorldJsonlAsset],
    asset_type: RegisteredType,
    key: &str,
) -> Option<String> {
    assets
        .iter()
        .find(|a| a.asset_type == asset_type)
        .and_then(|a| a.args.get(key))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

// A filesystem-safe slug for the bundle folder, executable, and archive name.
// Keeps alphanumerics, `.`, `-`, `_`; collapses runs of other characters
// (including spaces) to a single `-`. Falls back to "app" when empty.
pub(super) fn slug(name: &str) -> String {
    let mut out = String::new();
    let mut pending_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
            if pending_dash && !out.is_empty() {
                out.push('-');
            }
            pending_dash = false;
            out.push(c);
        } else {
            pending_dash = true;
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "app".to_string()
    } else {
        trimmed.to_string()
    }
}

// Short platform token for distributable artifact file names, from an OS string
// (`std::env::consts::OS`). Export is host-only, so this always reflects the
// bundle's actual platform.
pub(super) fn platform_tag(os: &str) -> &str {
    match os {
        "macos" => "mac",
        "windows" => "win",
        "linux" => "linux",
        other => other,
    }
}

// The base name for a distributable archive: `<slug>-<version>-<platform>` (e.g.
// `My-Game-1.0.0-mac`). The `.app` bundle, the executable, and the in-archive top
// folder keep the bare slug; only the `.zip` / `.dmg` carry the version and
// platform so a release folder can hold builds of several versions side by side.
pub(super) fn artifact_stem(meta: &AppMeta, platform: &str) -> String {
    format!(
        "{}-{}-{}",
        slug(&meta.display_name),
        slug(&meta.version),
        platform
    )
}
