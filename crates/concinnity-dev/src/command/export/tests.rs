use super::files::{blob_index, collect_files, copy_tree};
use super::macos::{DEFAULT_ICON_PNG, run_tool, xml_escape};
use super::meta::{derive_identifier, read_app_meta, resolve_display_name};
use super::runtime::{
    backend_label, find_platform_stamp, find_subslice, normalize_platform, parse_runtime_platforms,
    runtime_wants_d3d12,
};
use super::*;
use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::WorldJsonlAsset;
use concinnity_cook::platform::Platform;

fn asset(name: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
    WorldJsonlAsset {
        id: name.to_string(),
        asset_type,
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
    let app = asset(
        "app",
        RegisteredType::AppConfig,
        serde_json::json!({"name": "App Name"}),
    );
    let menu = asset(
        "m",
        RegisteredType::MainMenu,
        serde_json::json!({"title": "Menu Title"}),
    );

    assert_eq!(
        resolve_display_name(
            Some("CLI Name"),
            std::slice::from_ref(&app),
            std::slice::from_ref(&menu)
        ),
        "CLI Name"
    );
    assert_eq!(
        resolve_display_name(
            None,
            std::slice::from_ref(&app),
            std::slice::from_ref(&menu)
        ),
        "App Name"
    );
    assert_eq!(
        resolve_display_name(None, &[], std::slice::from_ref(&menu)),
        "Menu Title"
    );
    assert_eq!(resolve_display_name(None, &[], &[]), "Concinnity");
    assert_eq!(resolve_display_name(Some("  "), &[app], &[]), "App Name");
}

#[test]
fn menu_title_names_an_app_after_the_menu_expands() {
    let world = r#"["MainMenu",{"$id":"pause","title":"Menu Title"}]"#;
    let meta = world_app_meta(world.into(), None, None, None).unwrap();
    assert_eq!(meta.display_name, "Menu Title");
    assert_eq!(meta.identifier, "gg.concinnity.menu-title");

    let world = format!(
        "{world}\n{}",
        r#"["AppConfig",{"$id":"app","name":"App Name"}]"#
    );
    let meta = world_app_meta((&world).into(), None, None, None).unwrap();
    assert_eq!(meta.display_name, "App Name");
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
    let meta = read_app_meta(Some("My Cool App"), None, &[], &[]);
    assert_eq!(meta.display_name, "My Cool App");
    assert_eq!(meta.identifier, "gg.concinnity.my-cool-app");
    assert_eq!(meta.version, "0.1.0");
    assert!(meta.icon.is_none());

    // AppConfig supplies id / version / icon verbatim.
    let app = asset(
        "app",
        RegisteredType::AppConfig,
        serde_json::json!({
            "name": "Named", "id": "gg.studio.thing", "version": "2.3.4", "icon": "art/i.png"
        }),
    );
    let meta = read_app_meta(None, None, std::slice::from_ref(&app), &[]);
    assert_eq!(meta.display_name, "Named");
    assert_eq!(meta.identifier, "gg.studio.thing");
    assert_eq!(meta.version, "2.3.4");
    assert_eq!(meta.icon.as_deref(), Some(Path::new("art/i.png")));
}

#[test]
fn version_precedence_is_cli_then_application_then_default() {
    let app = asset(
        "app",
        RegisteredType::AppConfig,
        serde_json::json!({"version": "2.3.4"}),
    );

    // --version overrides the AppConfig version.
    assert_eq!(
        read_app_meta(None, Some("9.9.9"), std::slice::from_ref(&app), &[]).version,
        "9.9.9"
    );
    // A blank --version is ignored, falling back to the AppConfig version.
    assert_eq!(
        read_app_meta(None, Some("  "), std::slice::from_ref(&app), &[]).version,
        "2.3.4"
    );
    // --version with no AppConfig still wins over the default.
    assert_eq!(read_app_meta(None, Some("3.0"), &[], &[]).version, "3.0");
    // No override, no AppConfig: the default.
    assert_eq!(read_app_meta(None, None, &[], &[]).version, "0.1.0");
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
    copy_runtime_sidecars(
        &src.join("concinnity-run.exe"),
        Some(Platform::DirectX.into()),
        &dest,
    )
    .unwrap();

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
    copy_runtime_sidecars(
        &src.join("concinnity-run.exe"),
        Some(Platform::Vulkan.into()),
        &dest,
    )
    .unwrap();

    assert!(dest.join("amd_fidelityfx_vk.dll").exists());
    assert!(!dest.join("D3D12").exists());
}

#[test]
fn runtime_wants_d3d12_only_for_dx_or_unknown() {
    let set = |platforms: &[Platform]| Some(platforms.iter().copied().collect());
    assert!(runtime_wants_d3d12(set(&[Platform::DirectX])));
    assert!(runtime_wants_d3d12(set(&[
        Platform::DirectX,
        Platform::Vulkan
    ])));
    // Unstamped: keep everything the runtime may need.
    assert!(runtime_wants_d3d12(None));
    assert!(!runtime_wants_d3d12(set(&[Platform::Vulkan])));
    assert!(!runtime_wants_d3d12(set(&[
        Platform::Metal,
        Platform::Vulkan
    ])));
}

#[test]
fn find_platform_stamp_reads_the_token_after_the_marker() {
    // Surround the stamp with binary noise, as it would appear in a real
    // executable, and terminate it with the stamp's NUL.
    let mut buf = vec![0xAAu8, 0x00, 0xFF, b'x'];
    buf.extend_from_slice(b"cn-runtime-platform:directx\0");
    buf.extend_from_slice(&[0x01, 0x02, 0x03]);
    assert_eq!(find_platform_stamp(&buf).as_deref(), Some("directx"));

    // Each backend token, and a list of them, is recovered verbatim.
    for token in ["metal", "directx", "vulkan", "directx,vulkan"] {
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
fn backend_label_covers_each_platform() {
    assert_eq!(backend_label("metal"), "Metal (MSL)");
    assert_eq!(backend_label("directx"), "DirectX (DXIL)");
    assert_eq!(backend_label("vulkan"), "Vulkan (SPIR-V)");
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
    copy_runtime_sidecars(
        &src.join("concinnity-run"),
        Some(Platform::Metal.into()),
        &dest,
    )
    .unwrap();
    assert_eq!(fs::read_dir(&dest).unwrap().count(), 0);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn export_rejects_dmg_off_macos() {
    let options = ExportOptions {
        out: "out".to_string(),
        dmg: true,
        ..ExportOptions::default()
    };
    let err = export(&options).unwrap_err();
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
fn verify_runtime_backend_cooks_the_players_platforms() {
    let metal = PlatformSet::single(Platform::Metal);
    let metal_vk: PlatformSet = [Platform::Metal, Platform::Vulkan].into_iter().collect();
    // Unstamped: cook everything this `cn` can.
    assert_eq!(verify_runtime_backend(None, metal_vk).unwrap(), metal_vk);
    // A player carrying fewer backends gets only its own.
    assert_eq!(
        verify_runtime_backend(Some(metal), metal_vk).unwrap(),
        metal
    );
    assert_eq!(
        verify_runtime_backend(Some(metal_vk), metal_vk).unwrap(),
        metal_vk
    );
}

#[test]
fn verify_runtime_backend_refuses_a_player_cn_cannot_cook_for() {
    let metal = PlatformSet::single(Platform::Metal);
    let metal_vk: PlatformSet = [Platform::Metal, Platform::Vulkan].into_iter().collect();
    let err = verify_runtime_backend(Some(metal_vk), metal).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    let message = err.to_string();
    assert!(message.contains("Vulkan (SPIR-V)"), "{message}");
    assert!(message.contains("`vulkan` feature"), "{message}");
}

#[test]
fn a_stamp_parses_to_its_platforms_and_refuses_unknown_keys() {
    let parsed = parse_runtime_platforms("directx,vulkan").unwrap();
    assert_eq!(
        parsed.iter().collect::<Vec<_>>(),
        [Platform::DirectX, Platform::Vulkan]
    );
    let err = parse_runtime_platforms("metal,glide").unwrap_err();
    assert!(err.to_string().contains("`glide`"), "{err}");
}

#[test]
fn read_runtime_platforms_reads_a_stamp_or_warns_none() {
    let tmp = tempfile::tempdir().unwrap();
    let stamped = tmp.path().join("stamped");
    fs::write(&stamped, b"junk cn-runtime-platform:metal,vulkan\0 junk").unwrap();
    assert_eq!(
        read_runtime_platforms(&stamped).unwrap(),
        Some([Platform::Metal, Platform::Vulkan].into_iter().collect())
    );

    let unstamped = tmp.path().join("unstamped");
    fs::write(&unstamped, b"no marker in here").unwrap();
    assert_eq!(read_runtime_platforms(&unstamped).unwrap(), None);
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
        .map(|i| archive.by_index(i).unwrap().name().unwrap().into_owned())
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
    export_portable(
        &meta,
        &runtime,
        Some(Platform::Metal.into()),
        &out,
        &data,
        true,
    )
    .unwrap();

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
    // A platform key the label table does not recognize is echoed back
    // verbatim rather than dropped, so an odd stamp still reads sensibly.
    assert_eq!(backend_label("wgpu"), "wgpu");
}
