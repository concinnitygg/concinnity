//! `cn export`: package one application for distribution. Builds it (reusing
//! the normal build cache), then assembles a self-contained bundle: the
//! runtime player executable beside the application's `data/`, which holds its
//! compiled blobs and its engine shaders compiled ahead of time, in a flat
//! layout the shipped runtime resolves relative to its own executable. On macOS
//! the bundle is a proper `.app` (optionally wrapped in a `.dmg`); elsewhere it
//! is a folder. The result is archived to a `.zip` by default.
//!
//! The player is the prebuilt `concinnity-player` binary that ships beside the
//! `cn`/`concinnity` executable; export copies it rather than compiling, so a
//! user needs no build toolchain and no engine source. Because the runtime is a
//! single compiled binary, a bundle targets exactly the platform this `cn` was
//! built for (host-only for now; see the --platform check).

mod files;
mod macos;
mod meta;
mod runtime;
mod shaders;
#[cfg(test)]
mod tests;

use concinnity_cook::authoring::world::WorldSource;
use concinnity_cook::paths::StateTree;
use concinnity_cook::platform::PlatformSet;
use std::path::{Path, PathBuf};
use std::{fs, io};

use crate::command::Target;

use files::{copy_blobs, make_executable, reset_dir, zip_tree};
use macos::{build_default_icns, build_dmg, build_icns, info_plist};
use meta::{artifact_stem, platform_tag, slug, world_app_meta};
use runtime::{
    check_target_platform, copy_runtime_sidecars, exe_file_name, read_runtime_platforms,
    runtime_binary_path, verify_runtime_backend, verify_runtime_shader_stamp,
};
use shaders::{compile_shipped_shaders, write_shipped_shaders};

// Resolved naming/metadata for the exported app.
struct AppMeta {
    // Display name (window title, macOS bundle display name); may contain spaces.
    display_name: String,
    // Reverse-DNS bundle identifier (macOS CFBundleIdentifier).
    identifier: String,
    // Version string (macOS CFBundle[Short]Version).
    version: String,
    // Source icon path, resolved under its file's root, if the AppConfig asset
    // set one.
    icon: Option<PathBuf>,
}

/// How an exported bundle is written to the output directory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum BundleFormat {
    /// The bundle archived to a `.zip`.
    #[default]
    Zip,
    /// The bundle left as a plain directory.
    Dir,
}

/// What [`export`] packages and where it writes the result.
#[derive(Clone, Debug, Default)]
pub(crate) struct ExportOptions<'a> {
    /// The file to build: the one `-f` names, else the application.
    pub target: Target<'a>,
    /// Overrides the application name the `AppConfig` declares.
    pub name: Option<String>,
    /// Overrides the application version the `AppConfig` declares.
    pub version: Option<String>,
    /// The target platform; must name this host when set.
    pub platform: Option<String>,
    /// The output directory.
    pub out: String,
    /// Whether the bundle is zipped or left as a directory.
    pub format: BundleFormat,
    /// Also wrap the `.app` in a `.dmg` (macOS only).
    pub dmg: bool,
}

/// Package a built application into a distributable bundle: the player binary and the
/// blob, written to `options.out` as a zip or a directory.
pub(crate) fn export(options: &ExportOptions<'_>) -> io::Result<()> {
    let ExportOptions {
        target,
        name,
        version,
        platform,
        out,
        format,
        dmg,
    } = options;
    let (name, version, platform) = (name.as_deref(), version.as_deref(), platform.as_deref());
    let make_zip = *format == BundleFormat::Zip;
    let dmg = *dmg;
    if dmg && !cfg!(target_os = "macos") {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "--dmg is only available when exporting on macOS",
        ));
    }

    // Fail fast, before building, on the things export cannot recover from: a
    // target that is not this platform, a missing runtime player, a runtime
    // rendering with a backend this `cn` cannot cook blobs for, and one that
    // would refuse the shaders this `cn` compiles.
    check_target_platform(platform)?;
    let runtime = runtime_binary_path()?;
    let runtime_bytes = fs::read(&runtime)?;
    let runtime_platforms = read_runtime_platforms(&runtime_bytes, &runtime)?;
    verify_runtime_shader_stamp(
        &runtime_bytes,
        &runtime,
        concinnity_runtime::SHIPPED_SHADER_STAMP,
    )?;
    drop(runtime_bytes);
    let targets = verify_runtime_backend(runtime_platforms, crate::cook_targets())?;

    // Build the application like `cn cook` (validates, compiles, writes the
    // blobs + cook-lock.json, reuses the build cache), for the player's backends.
    let world_path = target.resolve()?;
    crate::authoring::build_world_file_for(&world_path, targets)?;
    let shaders = compile_shipped_shaders(targets)?;

    let content = fs::read_to_string(&world_path)?;
    let root = crate::project::content_root();
    let meta = world_app_meta(
        WorldSource::file(&content, &world_path, root.as_deref()),
        name,
        version,
    )?;

    let out_dir = Path::new(out);
    fs::create_dir_all(out_dir)?;
    let data_dir = crate::project::data_dir().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no project state directory to read the built blobs from",
        )
    })?;

    if cfg!(target_os = "macos") {
        export_macos(&meta, &runtime, out_dir, &data_dir, &shaders, make_zip, dmg)
    } else {
        export_portable(
            &meta,
            &runtime,
            runtime_platforms,
            out_dir,
            &data_dir,
            &shaders,
            make_zip,
        )
    }
}

// A folder bundle (Windows / Linux): the renamed player beside `data/`, then a
// zip of that folder. `runtime_platforms` are the player's stamped shader
// platforms (None when unstamped), used to decide which native sidecars belong
// beside it.
fn export_portable(
    meta: &AppMeta,
    runtime: &Path,
    runtime_platforms: Option<PlatformSet>,
    out_dir: &Path,
    data_dir: &Path,
    shaders: &[u8],
    make_zip: bool,
) -> io::Result<()> {
    let slug = slug(&meta.display_name);
    let bundle_dir = out_dir.join(&slug);
    reset_dir(&bundle_dir)?;

    let exe_name = exe_file_name(&slug);
    let exe_dst = bundle_dir.join(&exe_name);
    fs::copy(runtime, &exe_dst)?;
    make_executable(&exe_dst)?;
    let runtime_libraries = crate::vendor::staged_runtime();
    if runtime_libraries.is_none() && cfg!(windows) {
        let searched = crate::vendor::vendor_root().map_or_else(
            || "no vendor directory".to_string(),
            |dir| dir.display().to_string(),
        );
        println!(
            "note: no runtime SDKs are staged in {searched}, so the export runs without \
             DLSS, FSR or XeSS; `concinnity vendor` fetches them"
        );
    }
    copy_runtime_sidecars(runtime_libraries.as_deref(), runtime_platforms, &bundle_dir)?;

    let blobs = copy_blobs(data_dir, &StateTree::at(&bundle_dir).data_dir())?;
    write_shipped_shaders(shaders, &bundle_dir)?;
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
// to Contents/Resources (see `concinnity_host::store::paths::tree_for_exe`).
// Optionally zipped and/or wrapped in a `.dmg`.
fn export_macos(
    meta: &AppMeta,
    runtime: &Path,
    out_dir: &Path,
    data_dir: &Path,
    shaders: &[u8],
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
    write_shipped_shaders(shaders, &resources)?;

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
