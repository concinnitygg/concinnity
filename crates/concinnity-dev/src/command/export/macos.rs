//! The macOS `.app` pieces: its Info.plist, its icon set and the optional `.dmg`.

use concinnity_host::scratch;
use std::path::Path;
use std::{fs, io};

use super::AppMeta;
use super::files::copy_tree;

// Synthesize the macOS Info.plist. The bundle display name comes from
// CFBundleDisplayName, so the `.app` file name can stay a plain slug while
// Finder still shows the real name.
pub(super) fn info_plist(meta: &AppMeta, exe_name: &str, icon_file: Option<&str>) -> String {
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

pub(super) fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// Build `<resources>/<slug>.icns` from a source PNG using the stock macOS tools
// `sips` (resize) and `iconutil` (assemble). Returns the icns file name for the
// Info.plist. Errors if the source is missing or a tool fails.
pub(super) fn build_icns(src: &Path, resources: &Path, slug: &str) -> io::Result<String> {
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
pub(super) const DEFAULT_ICON_PNG: &[u8] = include_bytes!("../../../assets/default-icon.png");

// Build `<resources>/<slug>.icns` from the bundled default icon. Writes the
// embedded PNG to a scratch file so it flows through the same sips/iconutil
// pipeline as a user-supplied icon.
pub(super) fn build_default_icns(resources: &Path, slug: &str) -> io::Result<String> {
    let src = scratch::Scratch::file(&format!("{slug}-default-icon.png"));
    fs::write(src.path(), DEFAULT_ICON_PNG)?;
    build_icns(src.path(), resources, slug)
}

// Wrap a `.app` in a compressed `.dmg` via `hdiutil`. The `.app` is staged into
// its own folder first so it lands at the disk image's root (hdiutil's
// -srcfolder makes the folder the volume root).
pub(super) fn build_dmg(
    app_dir: &Path,
    slug: &str,
    volume_name: &str,
    dmg_path: &Path,
) -> io::Result<()> {
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
pub(super) fn run_tool(program: &str, args: &[&str]) -> io::Result<()> {
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
