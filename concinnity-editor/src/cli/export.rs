// src/cli/export.rs
//
// `cn export`: package a built world into a distributable game. Builds the world
// (reusing the normal build cache), then assembles a self-contained bundle: the
// runtime player executable beside the world's compiled `data/` blobs, in a
// flat layout the shipped runtime resolves relative to its own executable. The
// result is a folder and, by default, a `.zip`.
//
// The player is the prebuilt `concinnity-runtime` binary that ships beside the
// `cn`/`concinnity` executable; export copies it rather than compiling, so a
// user needs no build toolchain and no engine source. Because the runtime is a
// single compiled binary, a bundle targets exactly the platform this `cn` was
// built for (host-only for now; see the --platform check).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use concinnity_cook::build_from_path;
use concinnity_cook::world::{WorldJsonlAsset, prepare_world};

use super::list::resolve_world_path;

pub fn export(
    json_path: Option<&str>,
    name: Option<&str>,
    platform: Option<&str>,
    out: &str,
    format: &str,
) -> io::Result<()> {
    let make_zip = match format {
        "zip" => true,
        "dir" => false,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown --format '{other}' (expected 'zip' or 'dir')"),
            ));
        }
    };

    // Fail fast, before building, on the two things export cannot recover from:
    // a target that is not this platform, and a missing runtime player.
    check_target_platform(platform)?;
    let runtime = runtime_binary_path()?;

    // Build the world exactly like `cn build` (validates, compiles, writes the
    // blobs + world-lock.json, reuses the build cache).
    let world_path = resolve_world_path(json_path)?;
    build_from_path(&world_path)?;

    // Read the application name from the expanded world. The build above already
    // validated it, so this cannot fail on validation; map any error plainly.
    let content = fs::read_to_string(&world_path)?;
    let loaded = prepare_world(&content)
        .map_err(|errs| io::Error::new(io::ErrorKind::InvalidData, errs.join("\n")))?;
    let display_name = resolve_display_name(name, &loaded.assets);
    let slug = slug(&display_name);

    let out_dir = Path::new(out);
    let bundle_dir = out_dir.join(&slug);
    if bundle_dir.exists() {
        fs::remove_dir_all(&bundle_dir)?;
    }
    fs::create_dir_all(&bundle_dir)?;

    // The renamed player, beside the world's data. The runtime resolves its
    // state root (data/, saves/, settings) relative to this executable.
    let exe_name = exe_file_name(&slug);
    let exe_dst = bundle_dir.join(&exe_name);
    fs::copy(&runtime, &exe_dst)?;
    make_executable(&exe_dst)?;

    let blob_count = copy_blobs(
        &concinnity_core::paths::data_dir(),
        &bundle_dir.join("data"),
    )?;
    if blob_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "the build produced no data blobs to package",
        ));
    }

    println!(
        "Exported \"{}\" -> {} ({} blob{})",
        display_name,
        bundle_dir.display(),
        blob_count,
        if blob_count == 1 { "" } else { "s" },
    );

    if make_zip {
        let zip_path = out_dir.join(format!("{slug}.zip"));
        zip_bundle(&bundle_dir, &slug, &exe_name, &zip_path)?;
        println!("Wrote {}", zip_path.display());
    }

    Ok(())
}

// Copy every compiled blob (the integer-named files) from the build's data
// directory into the bundle's `data/`, skipping the shader-compile
// intermediates the build leaves there (named after their asset). Returns the
// number of blobs copied.
fn copy_blobs(data_src: &Path, data_dst: &Path) -> io::Result<usize> {
    fs::create_dir_all(data_dst)?;
    let mut count = 0;
    for entry in fs::read_dir(data_src)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if is_blob_name(&name) {
            fs::copy(entry.path(), data_dst.join(&*name))?;
            count += 1;
        }
    }
    Ok(count)
}

// A blob file is named by its integer index with no extension (blob_path uses
// `index.to_string()`); everything else in data/ is build scratch.
fn is_blob_name(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit())
}

// Reject a target that is not the platform this `cn` was built for. Cross-
// platform export is not supported yet: the runtime player is a single compiled
// binary and cooked blobs embed platform-native shaders, so a bundle can only
// target the host.
fn check_target_platform(platform: Option<&str>) -> io::Result<()> {
    let Some(requested) = platform else {
        return Ok(());
    };
    let host = std::env::consts::OS;
    if normalize_platform(requested) == host {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "cross-platform export is not supported yet: this `cn` targets '{host}'. \
             Run `cn export` on a {requested} machine to produce a {requested} build."
        ),
    ))
}

fn normalize_platform(p: &str) -> &str {
    match p.to_lowercase().as_str() {
        "mac" | "macos" | "osx" | "darwin" => "macos",
        "win" | "windows" => "windows",
        "linux" => "linux",
        // Unknown values fall through and simply won't match the host.
        _ => "",
    }
}

// Locate the runtime player that ships beside this executable.
fn runtime_binary_path() -> io::Result<PathBuf> {
    let cn = std::env::current_exe()?;
    let dir = cn
        .parent()
        .ok_or_else(|| io::Error::other("cannot locate the cn executable's directory"))?;
    let path = dir.join(exe_file_name("concinnity-runtime"));
    if path.exists() {
        Ok(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "runtime player not found at {} -- the `concinnity-runtime` binary must sit \
                 beside the `cn` executable",
                path.display()
            ),
        ))
    }
}

// Platform executable file name: append `.exe` on Windows.
fn exe_file_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

#[cfg(unix)]
fn make_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> io::Result<()> {
    Ok(())
}

// The application name, by precedence: an explicit `--name`, then the
// Application asset's name, then a MainMenu title, then the engine default.
fn resolve_display_name(cli_name: Option<&str>, assets: &[WorldJsonlAsset]) -> String {
    if let Some(n) = cli_name.map(str::trim).filter(|s| !s.is_empty()) {
        return n.to_string();
    }
    if let Some(n) = string_arg(assets, "application", "name") {
        return n;
    }
    if let Some(n) = string_arg(assets, "mainmenu", "title") {
        return n;
    }
    "Concinnity".to_string()
}

// The first non-empty string value of `key` on the first asset whose normalized
// type matches `type_norm`.
fn string_arg(assets: &[WorldJsonlAsset], type_norm: &str, key: &str) -> Option<String> {
    assets
        .iter()
        .find(|a| normalize_type(&a.asset_type) == type_norm)
        .and_then(|a| a.args.get(key))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn normalize_type(t: &str) -> String {
    t.to_lowercase().replace('_', "")
}

// A filesystem-safe slug for the bundle folder, executable, and archive name.
// Keeps alphanumerics, `.`, `-`, `_`; collapses runs of other characters
// (including spaces) to a single `-`. Falls back to "game" when empty.
fn slug(name: &str) -> String {
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
        "game".to_string()
    } else {
        trimmed.to_string()
    }
}

// Zip the bundle directory under a single top-level `<slug>/` folder, with the
// player executable marked executable so it stays runnable after extraction on
// Unix. Files are added in sorted order for a reproducible archive.
fn zip_bundle(bundle_dir: &Path, top: &str, exe_name: &str, zip_path: &Path) -> io::Result<()> {
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    let mut files = Vec::new();
    collect_files(bundle_dir, &mut files)?;
    files.sort();

    let file = fs::File::create(zip_path)?;
    let mut zw = zip::ZipWriter::new(file);
    for path in files {
        let rel = path
            .strip_prefix(bundle_dir)
            .map_err(io::Error::other)?
            .to_string_lossy()
            .replace('\\', "/");
        let is_exe = rel == exe_name;
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(if is_exe { 0o755 } else { 0o644 });
        zw.start_file(format!("{top}/{rel}"), options)
            .map_err(io::Error::other)?;
        let bytes = fs::read(&path)?;
        zw.write_all(&bytes)?;
    }
    zw.finish().map_err(io::Error::other)?;
    Ok(())
}

// Collect every file under `dir` (recursively) into `out`.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, ty: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args,
        }
    }

    #[test]
    fn slug_is_filesystem_safe() {
        assert_eq!(slug("My Game"), "My-Game");
        assert_eq!(slug("  Spaced  Out  "), "Spaced-Out");
        assert_eq!(slug("weird:/name*?"), "weird-name");
        assert_eq!(slug("keep_dots.and-dashes"), "keep_dots.and-dashes");
        assert_eq!(slug("***"), "game");
        assert_eq!(slug(""), "game");
    }

    #[test]
    fn is_blob_name_matches_only_integer_files() {
        assert!(is_blob_name("0"));
        assert!(is_blob_name("42"));
        assert!(!is_blob_name(""));
        assert!(!is_blob_name("0.metallib"));
        assert!(!is_blob_name("default_vertex_shader.air"));
        assert!(!is_blob_name("settings"));
    }

    #[test]
    fn name_precedence_is_cli_then_application_then_menu_then_default() {
        let app = asset(
            "app",
            "Application",
            serde_json::json!({"name": "App Name"}),
        );
        let menu = asset("m", "MainMenu", serde_json::json!({"title": "Menu Title"}));

        // Explicit --name wins over everything.
        assert_eq!(
            resolve_display_name(Some("CLI Name"), &[app.clone(), menu.clone()]),
            "CLI Name"
        );
        // Then the Application asset.
        assert_eq!(
            resolve_display_name(None, &[app.clone(), menu.clone()]),
            "App Name"
        );
        // Then a MainMenu title.
        assert_eq!(
            resolve_display_name(None, std::slice::from_ref(&menu)),
            "Menu Title"
        );
        // Then the engine default.
        assert_eq!(resolve_display_name(None, &[]), "Concinnity");
        // An empty --name is ignored (falls through to the Application asset).
        assert_eq!(resolve_display_name(Some("  "), &[app]), "App Name");
    }

    #[test]
    fn normalize_platform_accepts_aliases() {
        assert_eq!(normalize_platform("macOS"), "macos");
        assert_eq!(normalize_platform("Darwin"), "macos");
        assert_eq!(normalize_platform("win"), "windows");
        assert_eq!(normalize_platform("Linux"), "linux");
        assert_eq!(normalize_platform("solaris"), "");
    }

    #[test]
    fn host_platform_is_accepted_and_others_rejected() {
        // None and the host are fine; a foreign platform errors.
        check_target_platform(None).unwrap();
        check_target_platform(Some(std::env::consts::OS)).unwrap();
        let foreign = if std::env::consts::OS == "windows" {
            "linux"
        } else {
            "windows"
        };
        assert!(check_target_platform(Some(foreign)).is_err());
    }
}
