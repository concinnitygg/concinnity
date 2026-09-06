// src/cli/export.rs
//
// `cn export`: package a built world into a distributable app. Builds the world
// (reusing the normal build cache), then assembles a self-contained bundle: the
// runtime player executable beside the world's compiled `data/` blobs, in a
// flat layout the shipped runtime resolves relative to its own executable. On
// macOS the bundle is a proper `.app` (optionally wrapped in a `.dmg`);
// elsewhere it is a folder. The result is archived to a `.zip` by default.
//
// The player is the prebuilt `concinnity-run` binary that ships beside the
// `cn`/`concinnity` executable; export copies it rather than compiling, so a
// user needs no build toolchain and no engine source. Because the runtime is a
// single compiled binary, a bundle targets exactly the platform this `cn` was
// built for (host-only for now; see the --platform check).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use concinnity_cook::authoring::world::WorldJsonlAsset;
use concinnity_cook::build_from_path;
use concinnity_cook::build_only::prepare_world;
use concinnity_cook::paths::StateTree;
use concinnity_host::scratch;

use crate::command::resolve_world_path;

// Resolved naming/metadata for the exported app.
struct AppMeta {
    // Display name (window title, macOS bundle display name); may contain spaces.
    display_name: String,
    // Reverse-DNS bundle identifier (macOS CFBundleIdentifier).
    identifier: String,
    // Version string (macOS CFBundle[Short]Version).
    version: String,
    // Source icon path (relative to the world), if the AppConfig asset set one.
    icon: Option<PathBuf>,
}

/// Package a built world into a distributable bundle: the player binary, the
/// blob, and a warmed runtime cache segment, written to `out` as a zip or a
/// directory.
pub fn export(
    json_path: Option<&str>,
    name: Option<&str>,
    version: Option<&str>,
    platform: Option<&str>,
    out: &str,
    format: &str,
    dmg: bool,
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
    if dmg && !cfg!(target_os = "macos") {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "--dmg is only available when exporting on macOS",
        ));
    }

    // Fail fast, before building, on the things export cannot recover from: a
    // target that is not this platform, a missing runtime player, and a runtime
    // built for a different backend than the blobs this `cn` cooks.
    check_target_platform(platform)?;
    let runtime = runtime_binary_path()?;
    let runtime_platform = read_runtime_platform(&runtime)?;
    verify_runtime_backend(runtime_platform.as_deref(), crate::cook_platform())?;

    // Build the world exactly like `cn build` (validates, compiles, writes the
    // blobs + world-lock.json, reuses the build cache).
    let world_path = resolve_world_path(json_path)?;
    let tree = crate::project::require()?;
    build_from_path(&tree, &world_path, crate::cook_platform())?;

    // Read the app metadata from the expanded world. The build above already
    // validated it, so this cannot fail on validation; map any error plainly.
    let content = fs::read_to_string(&world_path)?;
    let loaded = prepare_world(&content, crate::project::assets_dir().as_deref())
        .map_err(|errs| io::Error::new(io::ErrorKind::InvalidData, errs.join("\n")))?;
    let meta = read_app_meta(name, version, &loaded.assets);

    let out_dir = Path::new(out);
    fs::create_dir_all(out_dir)?;
    let data_dir = crate::project::data_dir().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no project state directory to read the built blobs from",
        )
    })?;

    if cfg!(target_os = "macos") {
        export_macos(&meta, &runtime, out_dir, &data_dir, make_zip, dmg)
    } else {
        export_portable(
            &meta,
            &runtime,
            runtime_platform.as_deref(),
            out_dir,
            &data_dir,
            make_zip,
        )
    }
}

// A folder bundle (Windows / Linux): the renamed player beside `data/`, then a
// zip of that folder. `runtime_platform` is the player's stamped shader
// platform (`hlsl`/`glsl`/`metal`, or None when unstamped), used to decide
// which native sidecars belong beside it.
fn export_portable(
    meta: &AppMeta,
    runtime: &Path,
    runtime_platform: Option<&str>,
    out_dir: &Path,
    data_dir: &Path,
    make_zip: bool,
) -> io::Result<()> {
    let slug = slug(&meta.display_name);
    let bundle_dir = out_dir.join(&slug);
    reset_dir(&bundle_dir)?;

    let exe_name = exe_file_name(&slug);
    let exe_dst = bundle_dir.join(&exe_name);
    fs::copy(runtime, &exe_dst)?;
    make_executable(&exe_dst)?;
    copy_runtime_sidecars(runtime, runtime_platform, &bundle_dir)?;

    let blobs = copy_blobs(data_dir, &StateTree::at(&bundle_dir).data_dir())?;
    // Before archiving, so the warmed cache segment is inside the zip. The
    // player resolves its state root to the bundle folder, so `cache/0` sits
    // beside the exe.
    precompile_shaders(&bundle_dir);
    report_export(&meta.display_name, &bundle_dir, blobs);

    if make_zip {
        let stem = artifact_stem(meta, platform_tag(std::env::consts::OS));
        let zip_path = out_dir.join(format!("{stem}.zip"));
        zip_tree(&bundle_dir, &slug, &exe_name, &zip_path)?;
        println!("Wrote {}", zip_path.display());
    }
    Ok(())
}

// A macOS `.app` bundle: Contents/MacOS/<exe>, Contents/Info.plist, and
// Contents/Resources/{<icon>.icns, data/}. The runtime resolves its state root
// to Contents/Resources (see concinnity-run's state_dir_for_exe). Optionally
// zipped and/or wrapped in a `.dmg`.
fn export_macos(
    meta: &AppMeta,
    runtime: &Path,
    out_dir: &Path,
    data_dir: &Path,
    make_zip: bool,
    make_dmg: bool,
) -> io::Result<()> {
    let slug = slug(&meta.display_name);
    let app_dir = out_dir.join(format!("{slug}.app"));
    reset_dir(&app_dir)?;

    let contents = app_dir.join("Contents");
    let macos_dir = contents.join("MacOS");
    let resources = contents.join("Resources");
    fs::create_dir_all(&macos_dir)?;
    fs::create_dir_all(&resources)?;

    let exe_dst = macos_dir.join(&slug);
    fs::copy(runtime, &exe_dst)?;
    make_executable(&exe_dst)?;

    let blobs = copy_blobs(data_dir, &StateTree::at(&resources).data_dir())?;
    // The player resolves its state root to Contents/Resources; a no-op on
    // Metal, whose shaders precompile at build time.
    precompile_shaders(&resources);

    // Build the .icns from the AppConfig's icon, falling back to the bundled
    // engine default so every macOS bundle carries an icon.
    let icon_file = Some(match &meta.icon {
        Some(src) => build_icns(src, &resources, &slug)?,
        None => build_default_icns(&resources, &slug)?,
    });

    fs::write(
        contents.join("Info.plist"),
        info_plist(meta, &slug, icon_file.as_deref()),
    )?;

    report_export(&meta.display_name, &app_dir, blobs);

    let stem = artifact_stem(meta, platform_tag(std::env::consts::OS));
    if make_zip {
        let zip_path = out_dir.join(format!("{stem}.zip"));
        let exe_rel = format!("Contents/MacOS/{slug}");
        zip_tree(&app_dir, &format!("{slug}.app"), &exe_rel, &zip_path)?;
        println!("Wrote {}", zip_path.display());
    }
    if make_dmg {
        let dmg_path = out_dir.join(format!("{stem}.dmg"));
        build_dmg(&app_dir, &slug, &meta.display_name, &dmg_path)?;
        println!("Wrote {}", dmg_path.display());
    }
    Ok(())
}

// Compile the engine's built-in shaders into the bundle's `cache/0`, so a
// player's first launch reuses every artifact instead of compiling (measured
// ~1 s on a DirectX release build). The player reads that segment like any
// other cache, so deleting it costs one slow launch and nothing more.
// Compilation is pure CPU, so this runs in-process with no GPU device or
// window: the compile set is enumerated from the same declarations renderer
// init compiles through, and the pool-sized Vulkan shaders are compiled for the
// world just built (its texture count). The artifacts are backend IR, not
// machine code, so ones compiled here are valid on any machine.
//
// Best-effort by design: a failed program is reported and simply compiles at
// the bundle's first launch, so failures warn rather than failing the export.
#[cfg(any(backend_dx, backend_vk))]
fn precompile_shaders(state_dir: &Path) {
    // The suite asserts bundle layout, not shader output: skip so no test runs
    // the shader compiler, reads the world data anchor, or writes into the
    // developer's cache segment. Mirrors `shader_cache`'s own test opt-out,
    // which does not apply here because concinnity-device is a dependency of
    // this test binary rather than the crate under test.
    if cfg!(test) {
        return;
    }
    println!("Compiling built-in shaders...");
    let report = concinnity_engine::precompile_builtin_shaders(state_dir);
    println!(
        "  cached {} shader binaries ({} compiled, {} reused)",
        report.cached(),
        report.compiled,
        report.reused
    );
    for failure in &report.failed {
        eprintln!(
            "warning: shader precompile failed ({failure}); it will compile on \
             the bundle's first launch"
        );
    }
}

// Metal precompiles its shaders at build time; the bundle needs no cache.
#[cfg(not(any(backend_dx, backend_vk)))]
fn precompile_shaders(_state_dir: &Path) {}

// Announce the finished bundle. `copy_blobs` has already refused a build that
// produced none, so the count here is always at least one.
fn report_export(display_name: &str, bundle: &Path, blobs: usize) {
    println!(
        "Exported \"{}\" -> {} ({} blob{})",
        display_name,
        bundle.display(),
        blobs,
        if blobs == 1 { "" } else { "s" },
    );
}

// Remove `dir` if it exists, then create it fresh, so re-exports are clean.
fn reset_dir(dir: &Path) -> io::Result<()> {
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    fs::create_dir_all(dir)
}

// Copy every compiled blob (the integer-named files) from the build's data
// directory into `data_dst`, skipping the shader-compile intermediates the
// build leaves there (named after their asset). Returns the number copied.
// Copy the built blobs into the bundle as the `data` entry the player looks
// for, and report how many were copied.
//
// A world that fits in one blob ships as a single file named `data`; one that
// overflows ships as a `data/` directory holding `0`, `1`, ... Overflow blobs
// are always siblings of blob 0 named by index, so the single-file form can
// only carry a world that has none -- the player refuses the mismatch rather
// than reading `1` and `2` out of the folder it was launched from.
fn copy_blobs(data_src: &Path, data_dst: &Path) -> io::Result<usize> {
    let blobs = blobs_in(data_src)?;
    if blobs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no compiled blobs in {}", data_src.display()),
        ));
    }
    // Whatever the previous export left here: the two forms occupy the same
    // name, so a shrinking world would otherwise write a file over a directory.
    let _ = fs::remove_file(data_dst);
    let _ = fs::remove_dir_all(data_dst);

    if let [(_, only)] = blobs.as_slice() {
        if let Some(parent) = data_dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(only, data_dst)?;
        return Ok(1);
    }

    fs::create_dir_all(data_dst)?;
    for (index, path) in &blobs {
        fs::copy(path, data_dst.join(index.to_string()))?;
    }
    Ok(blobs.len())
}

// Every blob file in `dir`, paired with its index and ordered by it.
fn blobs_in(dir: &Path) -> io::Result<Vec<(u32, PathBuf)>> {
    let mut blobs = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if let Some(index) = blob_index(&name) {
            blobs.push((index, entry.path()));
        }
    }
    blobs.sort_by_key(|(index, _)| *index);
    Ok(blobs)
}

// A blob file is named by its integer index with no extension (blob_path uses
// `index.to_string()`); everything else in data/ is build scratch.
fn blob_index(name: &str) -> Option<u32> {
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    name.parse().ok()
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
    let path = dir.join(exe_file_name("concinnity-run"));
    if path.exists() {
        Ok(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "runtime player not found at {} -- the `concinnity-run` binary must sit \
                 beside the `cn` executable (in a dev checkout, build it with \
                 `cargo build --features player`)",
                path.display()
            ),
        ))
    }
}

// The fixed prefix of the backend stamp baked into `concinnity-run` (see its
// definition in that binary's main.rs); the shader-platform key follows it.
const RUNTIME_PLATFORM_MARKER: &[u8] = b"cn-runtime-platform:";

// Read the runtime player's stamped shader platform (`hlsl` / `glsl` / `metal`)
// by scanning its binary for the backend marker. Returns None for an older,
// unstamped runtime, warning once so the skipped backend check is visible.
fn read_runtime_platform(runtime: &Path) -> io::Result<Option<String>> {
    let bytes = fs::read(runtime)?;
    let found = find_platform_stamp(&bytes);
    if found.is_none() {
        eprintln!(
            "warning: no backend stamp found in {}; skipping the runtime/cook \
             backend check (rebuild `concinnity-run` to enable it)",
            runtime.display()
        );
    }
    Ok(found)
}

// Reject a runtime player built for a different rendering backend than the one
// this `cn` cooks blobs for. `cn` and `concinnity-run` compile
// independently, so a DX-built `cn` sitting beside a Vulkan-built runtime would
// otherwise silently ship a SPIR-V player with DXBC blobs (or vice versa) that
// fails to load every shader at launch. `found` is the player's stamped shader
// platform (None for an unstamped runtime, already warned about); compare it to
// `cooked`, the shader platform this export's blobs were built for.
fn verify_runtime_backend(
    found: Option<&str>,
    cooked: concinnity_cook::platform::Platform,
) -> io::Result<()> {
    let expected = cooked.key();
    match found {
        // Unstamped runtime: cannot verify, so proceed rather than block a
        // possibly-fine export (the missing-stamp warning already printed).
        None => Ok(()),
        Some(found) if found == expected => Ok(()),
        Some(found) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "runtime/cook backend mismatch: this `cn` cooks {} blobs, but the \
                 `concinnity-run` player beside it was built for {}. Rebuild the \
                 runtime for the same backend (`cargo build --features player{}`) \
                 before exporting.",
                backend_label(expected),
                backend_label(found),
                feature_hint(expected),
            ),
        )),
    }
}

// Extract the shader-platform token from a runtime binary's backend stamp: the
// lowercase-ASCII run immediately after the marker prefix. Returns None when
// the marker is absent (an older, unstamped runtime).
fn find_platform_stamp(bytes: &[u8]) -> Option<String> {
    let start = find_subslice(bytes, RUNTIME_PLATFORM_MARKER)? + RUNTIME_PLATFORM_MARKER.len();
    let rest = &bytes[start..];
    let end = rest
        .iter()
        .position(|b| !b.is_ascii_lowercase())
        .unwrap_or(rest.len());
    let token = &rest[..end];
    if token.is_empty() {
        None
    } else {
        std::str::from_utf8(token).ok().map(str::to_string)
    }
}

// Index of the first occurrence of `needle` in `haystack`, or None.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// A human-facing label for a shader-platform key.
fn backend_label(platform_key: &str) -> &str {
    match platform_key {
        "metal" => "Metal (MSL)",
        "hlsl" => "DirectX (DXIL)",
        "glsl" => "Vulkan (SPIR-V)",
        other => other,
    }
}

// The cargo feature flag that reproduces a given cook backend when rebuilding
// the runtime: only the Vulkan (SPIR-V) backend is feature-gated.
// Appended to the `--features player` the rebuild hint already names, so a
// Vulkan player reads as one feature list rather than two flags.
fn feature_hint(platform_key: &str) -> &str {
    match platform_key {
        "glsl" => ",vulkan",
        _ => "",
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

// Read the app metadata from the expanded world, applying the `--name` /
// `--version` overrides and deriving anything the AppConfig asset left unset.
fn read_app_meta(
    cli_name: Option<&str>,
    cli_version: Option<&str>,
    assets: &[WorldJsonlAsset],
) -> AppMeta {
    let display_name = resolve_display_name(cli_name, assets);
    let identifier =
        string_arg(assets, "appconfig", "id").unwrap_or_else(|| derive_identifier(&display_name));
    let version = cli_version
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| string_arg(assets, "appconfig", "version"))
        .unwrap_or_else(|| "0.1.0".to_string());
    let icon = string_arg(assets, "appconfig", "icon").map(PathBuf::from);
    AppMeta {
        display_name,
        identifier,
        version,
        icon,
    }
}

// The app name, by precedence: an explicit `--name`, then the AppConfig
// asset's name, then a MainMenu title, then the engine default.
fn resolve_display_name(cli_name: Option<&str>, assets: &[WorldJsonlAsset]) -> String {
    if let Some(n) = cli_name.map(str::trim).filter(|s| !s.is_empty()) {
        return n.to_string();
    }
    if let Some(n) = string_arg(assets, "appconfig", "name") {
        return n;
    }
    if let Some(n) = string_arg(assets, "mainmenu", "title") {
        return n;
    }
    "Concinnity".to_string()
}

// A reverse-DNS bundle identifier derived from the name when the AppConfig
// asset declares none: `gg.concinnity.<name>`, the name reduced to bundle-id
// characters (ascii alphanumerics, lowercased; other runs become a single `-`).
fn derive_identifier(name: &str) -> String {
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
// (including spaces) to a single `-`. Falls back to "app" when empty.
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
        "app".to_string()
    } else {
        trimmed.to_string()
    }
}

// Short platform token for distributable artifact file names, from an OS string
// (`std::env::consts::OS`). Export is host-only, so this always reflects the
// bundle's actual platform.
fn platform_tag(os: &str) -> &str {
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
fn artifact_stem(meta: &AppMeta, platform: &str) -> String {
    format!(
        "{}-{}-{}",
        slug(&meta.display_name),
        slug(&meta.version),
        platform
    )
}

// Synthesize the macOS Info.plist. The bundle display name comes from
// CFBundleDisplayName, so the `.app` file name can stay a plain slug while
// Finder still shows the real name.
fn info_plist(meta: &AppMeta, exe_name: &str, icon_file: Option<&str>) -> String {
    let icon_entry = match icon_file {
        Some(icon) => format!(
            "\t<key>CFBundleIconFile</key>\n\t<string>{}</string>\n",
            xml_escape(icon)
        ),
        None => String::new(),
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
         \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n\
         <dict>\n\
         \t<key>CFBundleName</key>\n\t<string>{name}</string>\n\
         \t<key>CFBundleDisplayName</key>\n\t<string>{name}</string>\n\
         \t<key>CFBundleExecutable</key>\n\t<string>{exe}</string>\n\
         \t<key>CFBundleIdentifier</key>\n\t<string>{id}</string>\n\
         \t<key>CFBundleVersion</key>\n\t<string>{ver}</string>\n\
         \t<key>CFBundleShortVersionString</key>\n\t<string>{ver}</string>\n\
         \t<key>CFBundlePackageType</key>\n\t<string>APPL</string>\n\
         \t<key>CFBundleInfoDictionaryVersion</key>\n\t<string>6.0</string>\n\
         {icon}\
         \t<key>LSMinimumSystemVersion</key>\n\t<string>11.0</string>\n\
         \t<key>NSHighResolutionCapable</key>\n\t<true/>\n\
         \t<key>NSPrincipalClass</key>\n\t<string>NSApplication</string>\n\
         </dict>\n\
         </plist>\n",
        name = xml_escape(&meta.display_name),
        exe = xml_escape(exe_name),
        id = xml_escape(&meta.identifier),
        ver = xml_escape(&meta.version),
        icon = icon_entry,
    )
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// Build `<resources>/<slug>.icns` from a source PNG using the stock macOS tools
// `sips` (resize) and `iconutil` (assemble). Returns the icns file name for the
// Info.plist. Errors if the source is missing or a tool fails.
fn build_icns(src: &Path, resources: &Path, slug: &str) -> io::Result<String> {
    if !src.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("AppConfig icon not found: {}", src.display()),
        ));
    }
    // `iconutil` reads the ladder back out of a directory, and takes the whole
    // tree with it when this returns however it returns.
    let iconset = scratch::Scratch::dir(&format!("{slug}.iconset"))?;
    // The standard iconset ladder: each size at 1x and 2x.
    for size in [16u32, 32, 128, 256, 512] {
        for scale in [1u32, 2] {
            let px = size * scale;
            let suffix = if scale == 2 { "@2x" } else { "" };
            let dst = iconset
                .path()
                .join(format!("icon_{size}x{size}{suffix}.png"));
            run_tool(
                "sips",
                &[
                    "-z",
                    &px.to_string(),
                    &px.to_string(),
                    &src.to_string_lossy(),
                    "--out",
                    &dst.to_string_lossy(),
                ],
            )?;
        }
    }

    let icns_name = format!("{slug}.icns");
    let icns_path = resources.join(&icns_name);
    run_tool(
        "iconutil",
        &[
            "-c",
            "icns",
            &iconset.path().to_string_lossy(),
            "-o",
            &icns_path.to_string_lossy(),
        ],
    )?;
    Ok(icns_name)
}

// The engine's built-in fallback icon (the Concinnity mark on a dark gradient),
// used when the AppConfig asset sets no icon. Baked into the `cn` binary so a
// shipped toolchain, which has no engine source tree, still carries it.
const DEFAULT_ICON_PNG: &[u8] = include_bytes!("../assets/default-icon.png");

// Build `<resources>/<slug>.icns` from the bundled default icon. Writes the
// embedded PNG to a scratch file so it flows through the same sips/iconutil
// pipeline as a user-supplied icon.
fn build_default_icns(resources: &Path, slug: &str) -> io::Result<String> {
    let src = scratch::Scratch::file(&format!("{slug}-default-icon.png"));
    fs::write(src.path(), DEFAULT_ICON_PNG)?;
    build_icns(src.path(), resources, slug)
}

// Wrap a `.app` in a compressed `.dmg` via `hdiutil`. The `.app` is staged into
// its own folder first so it lands at the disk image's root (hdiutil's
// -srcfolder makes the folder the volume root).
fn build_dmg(app_dir: &Path, slug: &str, volume_name: &str, dmg_path: &Path) -> io::Result<()> {
    let staging = scratch::Scratch::dir(&format!("{slug}-dmg"))?;
    copy_tree(app_dir, &staging.path().join(format!("{slug}.app")))?;

    if dmg_path.exists() {
        fs::remove_file(dmg_path)?;
    }
    run_tool(
        "hdiutil",
        &[
            "create",
            "-volname",
            volume_name,
            "-srcfolder",
            &staging.path().to_string_lossy(),
            "-ov",
            "-format",
            "UDZO",
            &dmg_path.to_string_lossy(),
        ],
    )?;
    Ok(())
}

// Run an external tool, turning a non-zero exit into an io::Error carrying its
// stderr.
fn run_tool(program: &str, args: &[&str]) -> io::Result<()> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .map_err(|e| io::Error::new(e.kind(), format!("failed to run `{program}`: {e}")))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "`{program}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

// Copy the runtime player's sibling native libraries into the bundle beside the
// exe, so the bundle carries everything the player needs to launch. On Windows
// the graphics-SDK runtime DLLs (FidelityFX / XeSS / DLSS / DXC) sit beside the
// built runtime binary, and the Agility SDK D3D12 runtime lives in a `D3D12/`
// subdir that the exe's `D3D12SDKPath` export points at (`.\D3D12\`). A strict
// no-op on macOS and Linux, where the runtime links only system frameworks /
// libraries (no sibling `.dll`, no `D3D12/`).
//
// `runtime_platform` is the player's stamped shader platform. The `D3D12/` dir
// is meaningful only to a DirectX player -- its `D3D12SDKPath` export is gated
// on the DX backend, so a Vulkan or Metal player never references it. It is
// copied only for a DX (or unstamped, so unknown) runtime, keeping a Vulkan
// bundle from carrying an inert Agility runtime when built in a target dir
// shared with a DX build.
fn copy_runtime_sidecars(
    runtime: &Path,
    runtime_platform: Option<&str>,
    dest_dir: &Path,
) -> io::Result<()> {
    let Some(src_dir) = runtime.parent() else {
        return Ok(());
    };
    // Sibling DLLs (the Windows graphics-SDK runtimes). Matched by extension so
    // the set can change without export knowing backend specifics.
    for entry in fs::read_dir(src_dir)? {
        let path = entry?.path();
        let is_dll = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("dll"));
        if is_dll
            && path.is_file()
            && let Some(name) = path.file_name()
        {
            fs::copy(&path, dest_dir.join(name))?;
        }
    }
    // The Agility SDK D3D12 runtime, resolved by the exe relative to itself.
    // Only a DX player looks for it (see above).
    if runtime_wants_d3d12(runtime_platform) {
        let d3d12 = src_dir.join("D3D12");
        if d3d12.is_dir() {
            copy_tree(&d3d12, &dest_dir.join("D3D12"))?;
        }
    }
    Ok(())
}

// Whether a player with the given stamped platform references the `D3D12/`
// Agility runtime: a DX player does, a Vulkan or Metal player never does, and an
// unstamped (unknown) runtime is treated as maybe-DX so nothing it needs is
// dropped.
fn runtime_wants_d3d12(runtime_platform: Option<&str>) -> bool {
    !matches!(runtime_platform, Some("glsl") | Some("metal"))
}

// Recursively copy the directory tree at `src` to `dst`.
fn copy_tree(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_tree(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

// Zip the tree at `src_dir` under a single top-level `<top>/` folder, marking
// the player executable (`exe_rel`, relative to `src_dir`) executable so it
// stays runnable after extraction on Unix. Files are added in sorted order for
// a reproducible archive.
fn zip_tree(src_dir: &Path, top: &str, exe_rel: &str, zip_path: &Path) -> io::Result<()> {
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    let mut files = Vec::new();
    collect_files(src_dir, &mut files)?;
    files.sort();

    let file = fs::File::create(zip_path)?;
    let mut zw = zip::ZipWriter::new(file);
    for path in files {
        let rel = path
            .strip_prefix(src_dir)
            .map_err(io::Error::other)?
            .to_string_lossy()
            .replace('\\', "/");
        let mode = if rel == exe_rel { 0o755 } else { 0o644 };
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(mode);
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
        assert_eq!(slug("***"), "app");
        assert_eq!(slug(""), "app");
    }

    #[test]
    fn blob_index_matches_only_integer_files() {
        assert_eq!(blob_index("0"), Some(0));
        assert_eq!(blob_index("42"), Some(42));
        assert_eq!(blob_index(""), None);
        assert_eq!(blob_index("0.metallib"), None);
        assert_eq!(blob_index("default_vertex_shader.air"), None);
        assert_eq!(blob_index("settings"), None);
    }

    #[test]
    fn name_precedence_is_cli_then_app_config_then_menu_then_default() {
        let app = asset("app", "AppConfig", serde_json::json!({"name": "App Name"}));
        let menu = asset("m", "MainMenu", serde_json::json!({"title": "Menu Title"}));

        assert_eq!(
            resolve_display_name(Some("CLI Name"), &[app.clone(), menu.clone()]),
            "CLI Name"
        );
        assert_eq!(
            resolve_display_name(None, &[app.clone(), menu.clone()]),
            "App Name"
        );
        assert_eq!(
            resolve_display_name(None, std::slice::from_ref(&menu)),
            "Menu Title"
        );
        assert_eq!(resolve_display_name(None, &[]), "Concinnity");
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
        check_target_platform(None).unwrap();
        check_target_platform(Some(std::env::consts::OS)).unwrap();
        let foreign = if std::env::consts::OS == "windows" {
            "linux"
        } else {
            "windows"
        };
        assert!(check_target_platform(Some(foreign)).is_err());
    }

    #[test]
    fn app_meta_derives_id_and_version_defaults() {
        // No AppConfig: name falls to default, id derived, version defaulted.
        let meta = read_app_meta(Some("My Cool App"), None, &[]);
        assert_eq!(meta.display_name, "My Cool App");
        assert_eq!(meta.identifier, "gg.concinnity.my-cool-app");
        assert_eq!(meta.version, "0.1.0");
        assert!(meta.icon.is_none());

        // AppConfig supplies id / version / icon verbatim.
        let app = asset(
            "app",
            "AppConfig",
            serde_json::json!({
                "name": "Named", "id": "gg.studio.thing", "version": "2.3.4", "icon": "art/i.png"
            }),
        );
        let meta = read_app_meta(None, None, std::slice::from_ref(&app));
        assert_eq!(meta.display_name, "Named");
        assert_eq!(meta.identifier, "gg.studio.thing");
        assert_eq!(meta.version, "2.3.4");
        assert_eq!(meta.icon.as_deref(), Some(Path::new("art/i.png")));
    }

    #[test]
    fn version_precedence_is_cli_then_application_then_default() {
        let app = asset("app", "AppConfig", serde_json::json!({"version": "2.3.4"}));

        // --version overrides the AppConfig version.
        assert_eq!(
            read_app_meta(None, Some("9.9.9"), std::slice::from_ref(&app)).version,
            "9.9.9"
        );
        // A blank --version is ignored, falling back to the AppConfig version.
        assert_eq!(
            read_app_meta(None, Some("  "), std::slice::from_ref(&app)).version,
            "2.3.4"
        );
        // --version with no AppConfig still wins over the default.
        assert_eq!(read_app_meta(None, Some("3.0"), &[]).version, "3.0");
        // No override, no AppConfig: the default.
        assert_eq!(read_app_meta(None, None, &[]).version, "0.1.0");
    }

    #[test]
    fn platform_tag_maps_host_os() {
        assert_eq!(platform_tag("macos"), "mac");
        assert_eq!(platform_tag("windows"), "win");
        assert_eq!(platform_tag("linux"), "linux");
        // An unmapped OS passes through unchanged rather than being lost.
        assert_eq!(platform_tag("freebsd"), "freebsd");
    }

    #[test]
    fn artifact_stem_is_slug_version_platform() {
        let meta = AppMeta {
            display_name: "My Game".to_string(),
            identifier: "gg.studio.mg".to_string(),
            version: "1.0.0".to_string(),
            icon: None,
        };
        assert_eq!(artifact_stem(&meta, "mac"), "My-Game-1.0.0-mac");
        // The version is slugged too, so odd characters stay filesystem-safe.
        let meta = AppMeta {
            version: "1.0.0+build 7".to_string(),
            ..meta
        };
        assert_eq!(artifact_stem(&meta, "win"), "My-Game-1.0.0-build-7-win");
    }

    #[test]
    fn info_plist_has_required_keys_and_escapes() {
        let meta = AppMeta {
            display_name: "Tom & Jerry".to_string(),
            identifier: "gg.studio.tj".to_string(),
            version: "1.0".to_string(),
            icon: None,
        };
        let plist = info_plist(&meta, "tj", Some("tj.icns"));
        assert!(plist.contains("<key>CFBundleExecutable</key>\n\t<string>tj</string>"));
        assert!(plist.contains("<key>CFBundleIdentifier</key>\n\t<string>gg.studio.tj</string>"));
        assert!(plist.contains("<key>CFBundleShortVersionString</key>\n\t<string>1.0</string>"));
        assert!(plist.contains("<key>CFBundleIconFile</key>\n\t<string>tj.icns</string>"));
        // The `&` in the display name is XML-escaped.
        assert!(plist.contains("Tom &amp; Jerry"));
        assert!(!plist.contains("Tom & Jerry"));

        // Without an icon there is no CFBundleIconFile key.
        let plist = info_plist(&meta, "tj", None);
        assert!(!plist.contains("CFBundleIconFile"));
    }

    #[test]
    fn copy_runtime_sidecars_copies_dlls_and_the_d3d12_dir() {
        // Simulate a Windows target/<profile>/ dir: the runtime exe, sibling
        // DLLs, a non-lib file, and the Agility D3D12/ subdir.
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(src.join("D3D12")).unwrap();
        fs::write(src.join("concinnity-run.exe"), b"exe").unwrap();
        fs::write(src.join("amd_fidelityfx_dx12.dll"), b"x").unwrap();
        fs::write(src.join("libconcinnity_ffi.dll"), b"x").unwrap();
        fs::write(src.join("notes.txt"), b"x").unwrap();
        fs::write(src.join("D3D12").join("D3D12Core.dll"), b"x").unwrap();

        let dest = tmp.path().join("dest");
        fs::create_dir_all(&dest).unwrap();
        // A DX runtime pulls in its sibling DLLs and the Agility `D3D12/` dir.
        copy_runtime_sidecars(&src.join("concinnity-run.exe"), Some("hlsl"), &dest).unwrap();

        assert!(dest.join("amd_fidelityfx_dx12.dll").exists());
        assert!(dest.join("libconcinnity_ffi.dll").exists());
        assert!(dest.join("D3D12").join("D3D12Core.dll").exists());
        // Non-DLL siblings and the exe itself are not carried along.
        assert!(!dest.join("notes.txt").exists());
        assert!(!dest.join("concinnity-run.exe").exists());
    }

    #[test]
    fn copy_runtime_sidecars_skips_d3d12_for_a_non_dx_runtime() {
        // A Vulkan runtime built in a target dir shared with a DX build has a
        // stale `D3D12/` sibling; it must not land in the Vulkan bundle (the VK
        // exe never references it), though the sibling DLLs still copy.
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(src.join("D3D12")).unwrap();
        fs::write(src.join("concinnity-run.exe"), b"exe").unwrap();
        fs::write(src.join("amd_fidelityfx_vk.dll"), b"x").unwrap();
        fs::write(src.join("D3D12").join("D3D12Core.dll"), b"x").unwrap();

        let dest = tmp.path().join("dest");
        fs::create_dir_all(&dest).unwrap();
        copy_runtime_sidecars(&src.join("concinnity-run.exe"), Some("glsl"), &dest).unwrap();

        assert!(dest.join("amd_fidelityfx_vk.dll").exists());
        assert!(!dest.join("D3D12").exists());
    }

    #[test]
    fn runtime_wants_d3d12_only_for_dx_or_unknown() {
        assert!(runtime_wants_d3d12(Some("hlsl")));
        // Unstamped: keep everything the runtime may need.
        assert!(runtime_wants_d3d12(None));
        assert!(!runtime_wants_d3d12(Some("glsl")));
        assert!(!runtime_wants_d3d12(Some("metal")));
    }

    #[test]
    fn find_platform_stamp_reads_the_token_after_the_marker() {
        // Surround the stamp with binary noise, as it would appear in a real
        // executable, and terminate it with the stamp's NUL.
        let mut buf = vec![0xAAu8, 0x00, 0xFF, b'x'];
        buf.extend_from_slice(b"cn-runtime-platform:hlsl\0");
        buf.extend_from_slice(&[0x01, 0x02, 0x03]);
        assert_eq!(find_platform_stamp(&buf).as_deref(), Some("hlsl"));

        // Each backend token is recovered verbatim.
        for token in ["metal", "hlsl", "glsl"] {
            let stamp = format!("cn-runtime-platform:{token}\0");
            assert_eq!(
                find_platform_stamp(stamp.as_bytes()).as_deref(),
                Some(token)
            );
        }
    }

    #[test]
    fn find_platform_stamp_is_none_without_the_marker() {
        assert_eq!(find_platform_stamp(b"no stamp here"), None);
        assert_eq!(find_platform_stamp(b""), None);
        // Marker present but immediately terminated: no token.
        assert_eq!(find_platform_stamp(b"cn-runtime-platform:\0"), None);
    }

    #[test]
    fn find_subslice_locates_and_reports_absence() {
        assert_eq!(find_subslice(b"abcdef", b"cd"), Some(2));
        assert_eq!(find_subslice(b"abcdef", b"abc"), Some(0));
        assert_eq!(find_subslice(b"abcdef", b"xy"), None);
        assert_eq!(find_subslice(b"ab", b"abc"), None);
        assert_eq!(find_subslice(b"abc", b""), None);
    }

    #[test]
    fn backend_label_and_feature_hint_cover_each_platform() {
        assert_eq!(backend_label("metal"), "Metal (MSL)");
        assert_eq!(backend_label("hlsl"), "DirectX (DXIL)");
        assert_eq!(backend_label("glsl"), "Vulkan (SPIR-V)");
        assert_eq!(feature_hint("glsl"), ",vulkan");
        assert_eq!(feature_hint("hlsl"), "");
        assert_eq!(feature_hint("metal"), "");
    }

    #[test]
    fn default_icon_is_a_nonempty_png() {
        // The bundled fallback must be a real PNG so sips can rasterize it into
        // the iconset ladder; a missing/corrupt file would fail every export
        // without an AppConfig icon.
        assert!(DEFAULT_ICON_PNG.len() > 1024);
        assert_eq!(&DEFAULT_ICON_PNG[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn copy_runtime_sidecars_is_a_noop_without_dlls() {
        // The macOS / Linux case: no sibling `.dll`, no `D3D12/`.
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("concinnity-run"), b"exe").unwrap();
        fs::write(src.join("libconcinnity_ffi.dylib"), b"x").unwrap();

        let dest = tmp.path().join("dest");
        fs::create_dir_all(&dest).unwrap();
        copy_runtime_sidecars(&src.join("concinnity-run"), Some("metal"), &dest).unwrap();
        assert_eq!(fs::read_dir(&dest).unwrap().count(), 0);
    }

    #[test]
    fn export_rejects_an_unknown_format_up_front() {
        // The format check runs before any path resolution or build, so this
        // touches nothing on disk.
        let err = export(None, None, None, None, "out", "tarball", false).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("tarball"), "got: {err}");
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn export_rejects_dmg_off_macos() {
        let err = export(None, None, None, None, "out", "zip", true).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn derive_identifier_reduces_names_to_bundle_id_form() {
        assert_eq!(derive_identifier("My Game"), "gg.concinnity.my-game");
        assert_eq!(
            derive_identifier("Space:  Above!"),
            "gg.concinnity.space-above"
        );
        assert_eq!(derive_identifier("***"), "gg.concinnity.app");
        assert_eq!(derive_identifier(""), "gg.concinnity.app");
    }

    #[test]
    fn xml_escape_escapes_every_entity() {
        assert_eq!(
            xml_escape(r#"<a href="x">Tom & Jerry's</a>"#),
            "&lt;a href=&quot;x&quot;&gt;Tom &amp; Jerry&apos;s&lt;/a&gt;"
        );
        assert_eq!(xml_escape("plain"), "plain");
    }

    #[test]
    fn verify_runtime_backend_accepts_matching_or_unstamped() {
        let cooked = concinnity_cook::platform::Platform::Metal;
        verify_runtime_backend(None, cooked).unwrap();
        verify_runtime_backend(Some(cooked.key()), cooked).unwrap();

        let err = verify_runtime_backend(Some("hlsl"), cooked).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("mismatch"), "got: {err}");
    }

    #[test]
    fn read_runtime_platform_reads_a_stamp_or_warns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let stamped = tmp.path().join("stamped");
        fs::write(&stamped, b"junk cn-runtime-platform:metal\0 junk").unwrap();
        assert_eq!(
            read_runtime_platform(&stamped).unwrap().as_deref(),
            Some("metal")
        );

        let unstamped = tmp.path().join("unstamped");
        fs::write(&unstamped, b"no marker in here").unwrap();
        assert_eq!(read_runtime_platform(&unstamped).unwrap(), None);
    }

    #[test]
    fn copy_blobs_takes_only_integer_named_files() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("data");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("0"), b"blob0").unwrap();
        fs::write(src.join("12"), b"blob12").unwrap();
        fs::write(src.join("default_vert.air"), b"scratch").unwrap();
        fs::write(src.join("settings"), b"state").unwrap();

        let dst = tmp.path().join("out");
        let count = copy_blobs(&src, &dst).unwrap();
        assert_eq!(count, 2);
        assert!(dst.is_dir(), "an overflowing world ships a directory");
        assert_eq!(fs::read(dst.join("0")).unwrap(), b"blob0");
        assert_eq!(fs::read(dst.join("12")).unwrap(), b"blob12");
        assert!(!dst.join("default_vert.air").exists());
        assert!(!dst.join("settings").exists());
    }

    // A world that fits in one blob ships as a file named `data`, which is what
    // keeps a small game's bundle from carrying integer-named files at all.
    #[test]
    fn a_single_blob_world_ships_as_one_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("data");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("0"), b"blob0").unwrap();
        fs::write(src.join("default_vert.air"), b"scratch").unwrap();

        let dst = tmp.path().join("bundle").join("data");
        assert_eq!(copy_blobs(&src, &dst).unwrap(), 1);
        assert!(dst.is_file(), "one blob ships as the `data` file itself");
        assert_eq!(fs::read(&dst).unwrap(), b"blob0");
    }

    // The two forms occupy the same name, so a re-export across the boundary
    // has to clear whatever the last one left rather than write a file over a
    // directory (or leave a stale `1` beside a now-single blob).
    #[test]
    fn re_exporting_across_the_form_boundary_replaces_the_previous_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("data");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("0"), b"blob0").unwrap();
        fs::write(src.join("1"), b"blob1").unwrap();
        let dst = tmp.path().join("bundle").join("data");

        assert_eq!(copy_blobs(&src, &dst).unwrap(), 2);
        assert!(dst.is_dir());

        // The world shrinks to one blob: the directory gives way to a file.
        fs::remove_file(src.join("1")).unwrap();
        assert_eq!(copy_blobs(&src, &dst).unwrap(), 1);
        assert!(dst.is_file());
        assert_eq!(fs::read(&dst).unwrap(), b"blob0");

        // ...and back the other way.
        fs::write(src.join("1"), b"blob1").unwrap();
        assert_eq!(copy_blobs(&src, &dst).unwrap(), 2);
        assert!(dst.is_dir());
        assert_eq!(fs::read(dst.join("1")).unwrap(), b"blob1");
    }

    // A build that produced nothing is caught here rather than shipping a
    // bundle whose player has no world to open.
    #[test]
    fn copy_blobs_refuses_a_data_dir_with_no_blobs() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("data");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("default_vert.air"), b"scratch").unwrap();

        let err = copy_blobs(&src, &tmp.path().join("out")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn reset_dir_clears_previous_content() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bundle");
        fs::create_dir_all(dir.join("nested")).unwrap();
        fs::write(dir.join("nested").join("stale"), b"old").unwrap();

        reset_dir(&dir).unwrap();
        assert!(dir.exists());
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    }

    #[test]
    fn copy_tree_copies_nested_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(src.join("a").join("b")).unwrap();
        fs::write(src.join("top.txt"), b"1").unwrap();
        fs::write(src.join("a").join("b").join("deep.txt"), b"2").unwrap();

        let dst = tmp.path().join("dst");
        copy_tree(&src, &dst).unwrap();
        assert_eq!(fs::read(dst.join("top.txt")).unwrap(), b"1");
        assert_eq!(
            fs::read(dst.join("a").join("b").join("deep.txt")).unwrap(),
            b"2"
        );
    }

    #[test]
    fn collect_files_walks_the_whole_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("tree");
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("a"), b"1").unwrap();
        fs::write(dir.join("sub").join("b"), b"2").unwrap();

        let mut files = Vec::new();
        collect_files(&dir, &mut files).unwrap();
        files.sort();
        assert_eq!(files, vec![dir.join("a"), dir.join("sub").join("b")]);
    }

    #[test]
    fn zip_tree_archives_under_a_top_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("My-Game");
        fs::create_dir_all(bundle.join("data")).unwrap();
        fs::write(bundle.join("My-Game"), b"player").unwrap();
        fs::write(bundle.join("data").join("0"), b"blob").unwrap();

        let zip_path = tmp.path().join("My-Game-1.0.0-mac.zip");
        zip_tree(&bundle, "My-Game", "My-Game", &zip_path).unwrap();

        let file = fs::File::open(&zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert_eq!(archive.len(), 2);
        assert!(
            names.contains(&"My-Game/My-Game".to_string()),
            "got: {names:?}"
        );
        assert!(
            names.contains(&"My-Game/data/0".to_string()),
            "got: {names:?}"
        );
        // The player entry carries the executable bit for Unix extraction.
        let exe = archive.by_name("My-Game/My-Game").unwrap();
        assert_eq!(exe.unix_mode().map(|m| m & 0o777), Some(0o755));
    }

    #[test]
    fn build_icns_rejects_a_missing_source_before_running_tools() {
        let tmp = tempfile::tempdir().unwrap();
        let err = build_icns(&tmp.path().join("missing.png"), tmp.path(), "slug").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn run_tool_reports_success_failure_and_missing() {
        // A tool that exits 0 succeeds; a non-zero exit is surfaced as an error
        // naming the tool. `true`/`false` are not reliably on PATH on Windows
        // (they exist only under a Unix shell), so drive the exit code through
        // the platform shell, which always resolves.
        #[cfg(windows)]
        {
            run_tool("cmd", &["/C", "exit 0"]).unwrap();
            let err = run_tool("cmd", &["/C", "exit 1"]).unwrap_err();
            assert!(err.to_string().contains("cmd"), "got: {err}");
        }
        #[cfg(not(windows))]
        {
            run_tool("true", &[]).unwrap();
            let err = run_tool("false", &[]).unwrap_err();
            assert!(err.to_string().contains("false"), "got: {err}");
        }
        // A missing program is surfaced as an error rather than a panic. How
        // it is reported is platform-dependent: some spawn paths fail to spawn
        // (ErrorKind::NotFound -> "failed to run"), others run and exit non-zero
        // (127 -> "`prog` failed"). Both name the tool, so assert on that.
        let err = run_tool("cn-nonexistent-tool-xyz", &[]).unwrap_err();
        assert!(
            err.to_string().contains("cn-nonexistent-tool-xyz"),
            "got: {err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn make_executable_sets_the_exec_bit() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("player");
        fs::write(&f, b"bin").unwrap();
        // A freshly written file carries no execute bits.
        assert_eq!(fs::metadata(&f).unwrap().permissions().mode() & 0o111, 0);
        make_executable(&f).unwrap();
        assert_ne!(fs::metadata(&f).unwrap().permissions().mode() & 0o111, 0);
    }

    // A folder bundle assembles the renamed player beside the copied blobs and
    // archives them. Cross-platform: a `metal` runtime pulls in no Windows
    // sidecars, so this runs identically on macOS and Linux.
    #[test]
    fn export_portable_assembles_folder_and_zip() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        fs::create_dir_all(&data).unwrap();
        fs::write(data.join("0"), b"blob0").unwrap();
        fs::write(data.join("1"), b"blob1").unwrap();
        fs::write(data.join("scratch.air"), b"ignored").unwrap();

        let runtime = tmp.path().join(exe_file_name("concinnity-run"));
        fs::write(&runtime, b"runtime-bin").unwrap();

        let out = tmp.path().join("out");
        fs::create_dir_all(&out).unwrap();

        let meta = AppMeta {
            display_name: "My Game".to_string(),
            identifier: "gg.studio.mg".to_string(),
            version: "1.0.0".to_string(),
            icon: None,
        };
        export_portable(&meta, &runtime, Some("metal"), &out, &data, true).unwrap();

        let bundle = out.join("My-Game");
        let exe = bundle.join(exe_file_name("My-Game"));
        assert!(exe.exists(), "renamed player missing");
        assert_eq!(fs::read(&exe).unwrap(), b"runtime-bin");
        assert!(bundle.join("data").join("0").exists());
        assert!(bundle.join("data").join("1").exists());
        // Build scratch is not packaged.
        assert!(!bundle.join("data").join("scratch.air").exists());

        // The versioned archive was written beside the folder.
        let stem = artifact_stem(&meta, platform_tag(std::env::consts::OS));
        assert!(out.join(format!("{stem}.zip")).exists(), "zip missing");
    }

    // The full `.app` assembly, including the bundled default `.icns` icon
    // ladder (sips + iconutil). macOS-only: it shells out to the stock Apple
    // icon tools, which do not exist on the Linux CI runner.
    #[cfg(target_os = "macos")]
    #[test]
    fn export_macos_builds_app_bundle_with_default_icon() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        fs::create_dir_all(&data).unwrap();
        fs::write(data.join("0"), b"blob0").unwrap();

        let runtime = tmp.path().join("concinnity-run");
        fs::write(&runtime, b"runtime-bin").unwrap();

        let out = tmp.path().join("out");
        fs::create_dir_all(&out).unwrap();

        let meta = AppMeta {
            display_name: "My Game".to_string(),
            identifier: "gg.studio.mg".to_string(),
            version: "1.0.0".to_string(),
            icon: None,
        };
        // No AppConfig icon -> the bundled default flows through the icon
        // pipeline. make_zip on, make_dmg off (dmg needs a slow hdiutil).
        export_macos(&meta, &runtime, &out, &data, true, false).unwrap();

        let app = out.join("My-Game.app");
        assert!(app.join("Contents/MacOS/My-Game").exists(), "exe missing");
        // One blob, so the bundle carries `data` as a file rather than a
        // directory -- the form the player reads as blob 0 directly.
        let data_entry = app.join("Contents/Resources/data");
        assert!(data_entry.is_file(), "blob missing");
        assert_eq!(fs::read(&data_entry).unwrap(), b"blob0");
        assert!(
            app.join("Contents/Resources/My-Game.icns").exists(),
            "icns missing"
        );

        let plist = fs::read_to_string(app.join("Contents/Info.plist")).unwrap();
        assert!(plist.contains("<string>My-Game.icns</string>"));
        assert!(plist.contains("gg.studio.mg"));

        let stem = artifact_stem(&meta, platform_tag(std::env::consts::OS));
        assert!(out.join(format!("{stem}.zip")).exists(), "zip missing");
    }

    // The icon ladder rasterizes a real PNG into an `.icns` via sips + iconutil.
    // macOS-only for the same reason as the `.app` test.
    #[cfg(target_os = "macos")]
    #[test]
    fn build_icns_produces_an_icns_from_a_valid_png() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("icon.png");
        fs::write(&src, DEFAULT_ICON_PNG).unwrap();
        let resources = tmp.path().join("Resources");
        fs::create_dir_all(&resources).unwrap();

        let name = build_icns(&src, &resources, "app").unwrap();
        assert_eq!(name, "app.icns");
        assert!(resources.join("app.icns").exists());
    }

    #[test]
    fn runtime_binary_path_errors_when_the_player_is_absent() {
        // The test harness binary lives in `target/<profile>/deps/`, and the
        // `concinnity-run` player is built one level up in `target/<profile>/`,
        // so it never sits beside the test executable: the lookup must report a
        // clear NotFound rather than pointing at a nonexistent path.
        let err = runtime_binary_path().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(
            err.to_string().contains("runtime player not found"),
            "got: {err}"
        );
    }

    #[test]
    fn backend_label_passes_through_an_unknown_key() {
        // A platform key the label table does not recognise is echoed back
        // verbatim rather than dropped, so an odd stamp still reads sensibly.
        assert_eq!(backend_label("wgpu"), "wgpu");
    }
}
