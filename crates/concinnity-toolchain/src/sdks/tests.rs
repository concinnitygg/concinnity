use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::*;

// Create `path` (and its parents) with junk bytes standing in for a DLL.
fn touch(path: &Path) {
    touch_with(path, b"junk dll bytes");
}

fn touch_with(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

// An SdkEnv rooted in `dir`: every SDK root points at a subdirectory of the
// tempdir, every probe is enabled, and OUT_DIR mirrors the real cargo layout
// (`<target>/<profile>/build/<pkg>-<hash>/out`).
fn env_in(dir: &Path) -> SdkEnv {
    fs::create_dir_all(profile(dir)).unwrap();
    SdkEnv {
        target_os: "windows".to_string(),
        out_dir: Some(profile(dir).join("build").join("pkg-0123abcd").join("out")),
        agility_root: Some(dir.join("agility")),
        fidelityfx_root: Some(dir.join("ffx")),
        fidelityfx_vk_root: Some(dir.join("ffx-vk")),
        xess_root: Some(dir.join("xess")),
        streamline_root: Some(dir.join("streamline")),
        dxc_root: None,
        windows_sdk_bin: Some(dir.join("winkits")),
        agility_enabled: true,
        ffx_enabled: true,
        xess_enabled: true,
        dlss_enabled: true,
        dxc_enabled: true,
    }
}

fn profile(dir: &Path) -> PathBuf {
    dir.join("target").join("debug")
}

fn examples(dir: &Path) -> PathBuf {
    profile(dir).join("examples")
}

fn has(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|l| l.contains(needle))
}

fn warnings(lines: &[String]) -> Vec<&String> {
    lines
        .iter()
        .filter(|l| l.starts_with("cargo::warning="))
        .collect()
}

fn install_agility(dir: &Path) {
    let bin = dir
        .join("agility")
        .join("build")
        .join("native")
        .join("bin")
        .join("x64");
    touch(&bin.join("D3D12Core.dll"));
    touch(&bin.join("d3d12SDKLayers.dll"));
}

fn install_ffx_dx(dir: &Path) {
    touch(&dir.join("ffx").join("bin").join("amd_fidelityfx_dx12.dll"));
}

fn install_ffx_vk_sdk(dir: &Path, bytes: &[u8]) {
    touch_with(
        &dir.join("ffx").join("bin").join("amd_fidelityfx_vk.dll"),
        bytes,
    );
}

// The rebuilt runtime vendored beside the SDK, which carries the shader fix
// the SDK's own copy does not.
fn install_ffx_vk_rebuilt(dir: &Path, bytes: &[u8]) {
    touch_with(
        &dir.join("ffx-vk").join("bin").join("amd_fidelityfx_vk.dll"),
        bytes,
    );
}

fn install_xess(dir: &Path) {
    touch(&dir.join("xess").join("bin").join("libxess.dll"));
}

fn install_ngx_lib(dir: &Path) {
    touch(
        &dir.join("streamline")
            .join("external")
            .join("ngx-sdk")
            .join("lib")
            .join("Windows_x86_64")
            .join("nvsdk_ngx_d.lib"),
    );
}

fn install_ngx_dll(dir: &Path) {
    touch(
        &dir.join("streamline")
            .join("bin")
            .join("x64")
            .join("nvngx_dlss.dll"),
    );
}

fn install_winkits_dxc(dir: &Path, version: &str, bytes: &[u8]) {
    let x64 = dir.join("winkits").join(version).join("x64");
    touch_with(&x64.join("dxcompiler.dll"), bytes);
    touch_with(&x64.join("dxil.dll"), bytes);
}

#[test]
fn check_cfg_directives_declare_every_gated_cfg() {
    let lines = check_cfg_directives();
    assert_eq!(lines.len(), 8);
    for cfg in [
        "backend_metal",
        "backend_dx",
        "backend_vk",
        "agility_sdk_configured",
        "ffx_sdk_bundled",
        "xess_sdk_bundled",
        "ngx_sdk_bundled",
        "dxc_bundled",
    ] {
        assert!(lines.contains(&format!("cargo::rustc-check-cfg=cfg({cfg})")));
    }
}

#[test]
fn backend_cfg_directive_names_the_backend() {
    assert_eq!(
        backend_cfg_directive(Backend::Metal),
        "cargo::rustc-cfg=backend_metal"
    );
    assert_eq!(
        backend_cfg_directive(Backend::Dx),
        "cargo::rustc-cfg=backend_dx"
    );
    assert_eq!(
        backend_cfg_directive(Backend::Vk),
        "cargo::rustc-cfg=backend_vk"
    );
}

#[test]
fn agility_bundles_dlls_and_emits_exports() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    agility_directives(&env, BinaryTargets::Bins, &mut out);

    let d3d12 = profile(tmp.path()).join("D3D12");
    assert!(d3d12.join("D3D12Core.dll").is_file());
    assert!(d3d12.join("d3d12SDKLayers.dll").is_file());
    assert!(has(
        &out,
        "cargo::rerun-if-env-changed=CN_ENABLE_AGILITY_SDK"
    ));
    assert!(has(&out, "cargo::rerun-if-env-changed=CN_AGILITY_SDK"));
    assert!(has(&out, "/EXPORT:D3D12SDKVersion,DATA"));
    assert!(has(&out, "/EXPORT:D3D12SDKPath,DATA"));
    assert!(out.contains(&"cargo::rustc-cfg=agility_sdk_configured".to_string()));
    // A rerun-if-changed per copied source DLL.
    assert!(has(&out, "cargo::rerun-if-changed") && has(&out, "D3D12Core.dll"));
    assert!(warnings(&out).is_empty());
}

#[test]
fn agility_present_without_bundling_emits_cfg_only() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    agility_directives(&env, BinaryTargets::None, &mut out);

    assert!(out.contains(&"cargo::rustc-cfg=agility_sdk_configured".to_string()));
    assert!(!has(&out, "/EXPORT:"));
    assert!(!profile(tmp.path()).join("D3D12").exists());
}

#[test]
fn agility_missing_warns_only_when_bundling() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    let mut bundled = Vec::new();
    agility_directives(&env, BinaryTargets::Bins, &mut bundled);
    // The build asked for the SDK and it is not installed, so say so -- unlike
    // the no-opt-in case, which is silent.
    assert!(has(&bundled, "CN_ENABLE_AGILITY_SDK is set but"));
    assert!(!has(&bundled, "agility_sdk_configured"));

    let mut quiet = Vec::new();
    agility_directives(&env, BinaryTargets::None, &mut quiet);
    assert!(warnings(&quiet).is_empty());
    // Only the two rerun directives remain.
    assert_eq!(
        quiet,
        vec![
            "cargo::rerun-if-env-changed=CN_ENABLE_AGILITY_SDK".to_string(),
            "cargo::rerun-if-env-changed=CN_AGILITY_SDK".to_string(),
        ]
    );
}

// Without the opt-in nothing happens, and nothing is said about it: this is the
// ordinary build, producing an executable that runs wherever it is copied. An
// installed SDK must not change that -- a build machine that has one is exactly
// how the non-relocatable binary used to get made by accident.
#[test]
fn agility_without_the_opt_in_is_silent_and_stages_nothing() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    let env = SdkEnv {
        agility_enabled: false,
        ..env_in(tmp.path())
    };

    for targets in [
        BinaryTargets::None,
        BinaryTargets::Bins,
        BinaryTargets::Examples,
    ] {
        let mut out = Vec::new();
        agility_directives(&env, targets, &mut out);
        assert_eq!(
            out,
            vec!["cargo::rerun-if-env-changed=CN_ENABLE_AGILITY_SDK".to_string()],
            "{targets:?}"
        );
    }
    assert!(!profile(tmp.path()).join("D3D12").exists());
    assert!(!examples(tmp.path()).join("D3D12").exists());
}

#[test]
fn agility_without_out_dir_emits_no_cfg() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    let env = SdkEnv {
        out_dir: None,
        ..env_in(tmp.path())
    };

    let mut out = Vec::new();
    agility_directives(&env, BinaryTargets::Bins, &mut out);
    assert!(!has(&out, "agility_sdk_configured"));
    assert!(!has(&out, "/EXPORT:"));
}

#[test]
fn ffx_dx_bundles_dll_next_to_exe() {
    let tmp = TempDir::new().unwrap();
    install_ffx_dx(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    fidelityfx_dx_directives(&env, BinaryTargets::Bins, &mut out);

    assert!(
        profile(tmp.path())
            .join("amd_fidelityfx_dx12.dll")
            .is_file()
    );
    assert!(out.contains(&"cargo::rustc-cfg=ffx_sdk_bundled".to_string()));
    assert!(has(&out, "cargo::rerun-if-changed"));
    assert!(warnings(&out).is_empty());
}

#[test]
fn ffx_dx_present_without_bundling_emits_cfg_without_copy() {
    let tmp = TempDir::new().unwrap();
    install_ffx_dx(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    fidelityfx_dx_directives(&env, BinaryTargets::None, &mut out);

    assert!(out.contains(&"cargo::rustc-cfg=ffx_sdk_bundled".to_string()));
    assert!(!profile(tmp.path()).join("amd_fidelityfx_dx12.dll").exists());
}

#[test]
fn ffx_dx_missing_or_opted_out_emits_no_cfg() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    let mut missing = Vec::new();
    fidelityfx_dx_directives(&env, BinaryTargets::Bins, &mut missing);
    assert!(has(&missing, "FidelityFX SDK not found at"));
    assert!(!has(&missing, "ffx_sdk_bundled"));

    let disabled_env = SdkEnv {
        ffx_enabled: false,
        ..env_in(tmp.path())
    };
    let mut disabled = Vec::new();
    fidelityfx_dx_directives(&disabled_env, BinaryTargets::Bins, &mut disabled);
    assert!(has(&disabled, "CN_ENABLE_FFX_FSR3=0"));
    assert!(!has(&disabled, "ffx_sdk_bundled"));
    assert!(!has(&disabled, "CN_FIDELITYFX_SDK"));
}

// The stock SDK ships a VK runtime of its own, so the rebuilt one is only
// reached by being preferred over it, not by being the sole candidate.
#[test]
fn ffx_vk_prefers_the_rebuilt_dll() {
    let tmp = TempDir::new().unwrap();
    install_ffx_vk_rebuilt(tmp.path(), b"rebuilt");
    install_ffx_vk_sdk(tmp.path(), b"stock sdk");
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    fidelityfx_vk_directives(&env, BinaryTargets::Bins, &mut out);

    let dst = profile(tmp.path()).join("amd_fidelityfx_vk.dll");
    assert_eq!(fs::read(&dst).unwrap(), b"rebuilt");
    assert!(out.contains(&"cargo::rustc-cfg=ffx_sdk_bundled".to_string()));
    // The rerun-if-changed directive must name the source it copied.
    assert!(has(
        &out,
        &tmp.path()
            .join("ffx-vk")
            .join("bin")
            .join("amd_fidelityfx_vk.dll")
            .display()
            .to_string()
    ));
}

#[test]
fn ffx_vk_falls_back_to_the_sdk_root() {
    let tmp = TempDir::new().unwrap();
    install_ffx_vk_sdk(tmp.path(), b"stock sdk");
    let env = SdkEnv {
        fidelityfx_vk_root: None,
        ..env_in(tmp.path())
    };

    let mut out = Vec::new();
    fidelityfx_vk_directives(&env, BinaryTargets::Bins, &mut out);

    let dst = profile(tmp.path()).join("amd_fidelityfx_vk.dll");
    assert_eq!(fs::read(&dst).unwrap(), b"stock sdk");
    assert!(out.contains(&"cargo::rustc-cfg=ffx_sdk_bundled".to_string()));
}

#[test]
fn ffx_vk_missing_everywhere_warns_when_bundling() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    fidelityfx_vk_directives(&env, BinaryTargets::Bins, &mut out);
    assert!(has(&out, "FidelityFX VK runtime not found"));
    assert!(!has(&out, "ffx_sdk_bundled"));

    let mut quiet = Vec::new();
    fidelityfx_vk_directives(&env, BinaryTargets::None, &mut quiet);
    assert!(warnings(&quiet).is_empty());
}

#[test]
fn xess_bundles_dll_and_emits_cfg() {
    let tmp = TempDir::new().unwrap();
    install_xess(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    xess_directives(&env, BinaryTargets::Bins, &mut out);

    assert!(profile(tmp.path()).join("libxess.dll").is_file());
    assert!(out.contains(&"cargo::rustc-cfg=xess_sdk_bundled".to_string()));
    assert!(warnings(&out).is_empty());
}

#[test]
fn xess_missing_or_opted_out_emits_no_cfg() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    let mut missing = Vec::new();
    xess_directives(&env, BinaryTargets::Bins, &mut missing);
    assert!(has(&missing, "XeSS SDK not found at"));
    assert!(!has(&missing, "xess_sdk_bundled"));

    let disabled_env = SdkEnv {
        xess_enabled: false,
        ..env_in(tmp.path())
    };
    let mut disabled = Vec::new();
    xess_directives(&disabled_env, BinaryTargets::Bins, &mut disabled);
    assert!(has(&disabled, "CN_ENABLE_XESS=0"));
    assert!(!has(&disabled, "CN_XESS_SDK"));
}

#[test]
fn dlss_links_import_lib_without_bundling() {
    let tmp = TempDir::new().unwrap();
    install_ngx_lib(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    dlss_directives(&env, BinaryTargets::None, &mut out);

    assert!(has(&out, "cargo::rustc-link-arg="));
    assert!(has(&out, "nvsdk_ngx_d.lib"));
    assert!(out.contains(&"cargo::rustc-cfg=ngx_sdk_bundled".to_string()));
    assert!(!profile(tmp.path()).join("nvngx_dlss.dll").exists());
}

#[test]
fn dlss_bundles_the_feature_dll() {
    let tmp = TempDir::new().unwrap();
    install_ngx_lib(tmp.path());
    install_ngx_dll(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    dlss_directives(&env, BinaryTargets::Bins, &mut out);

    assert!(profile(tmp.path()).join("nvngx_dlss.dll").is_file());
    assert!(out.contains(&"cargo::rustc-cfg=ngx_sdk_bundled".to_string()));
    assert!(warnings(&out).is_empty());
}

#[test]
fn dlss_missing_feature_dll_still_links_the_lib() {
    let tmp = TempDir::new().unwrap();
    install_ngx_lib(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    dlss_directives(&env, BinaryTargets::Bins, &mut out);

    assert!(has(&out, "NGX feature DLL not found at"));
    assert!(has(&out, "cargo::rustc-link-arg="));
    assert!(out.contains(&"cargo::rustc-cfg=ngx_sdk_bundled".to_string()));
    assert!(!profile(tmp.path()).join("nvngx_dlss.dll").exists());
}

#[test]
fn dlss_missing_import_lib_emits_nothing_linkable() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    dlss_directives(&env, BinaryTargets::Bins, &mut out);
    assert!(has(&out, "NGX import lib not found at"));
    assert!(!has(&out, "cargo::rustc-link-arg="));
    assert!(!has(&out, "ngx_sdk_bundled"));
}

#[test]
fn dlss_opt_out_skips_the_probe() {
    let tmp = TempDir::new().unwrap();
    install_ngx_lib(tmp.path());
    let env = SdkEnv {
        dlss_enabled: false,
        ..env_in(tmp.path())
    };

    let mut out = Vec::new();
    dlss_directives(&env, BinaryTargets::Bins, &mut out);
    assert!(has(&out, "CN_ENABLE_DLSS=0"));
    assert!(!has(&out, "cargo::rustc-link-arg="));
    assert!(!has(&out, "CN_STREAMLINE_SDK"));
}

#[test]
fn dxc_override_root_wins_over_windows_sdk() {
    let tmp = TempDir::new().unwrap();
    let override_dir = tmp.path().join("dxc-override");
    touch_with(&override_dir.join("dxcompiler.dll"), b"override");
    touch_with(&override_dir.join("dxil.dll"), b"override");
    install_winkits_dxc(tmp.path(), "10.0.22621.0", b"winkits");
    let env = SdkEnv {
        dxc_root: Some(override_dir),
        ..env_in(tmp.path())
    };

    let mut out = Vec::new();
    dxc_directives(&env, BinaryTargets::Bins, &mut out);

    let dst = profile(tmp.path()).join("dxcompiler.dll");
    assert_eq!(fs::read(&dst).unwrap(), b"override");
    assert!(profile(tmp.path()).join("dxil.dll").is_file());
    assert!(out.contains(&"cargo::rustc-cfg=dxc_bundled".to_string()));
}

#[test]
fn dxc_falls_back_to_the_newest_complete_windows_sdk() {
    let tmp = TempDir::new().unwrap();
    install_winkits_dxc(tmp.path(), "10.0.19041.0", b"old");
    install_winkits_dxc(tmp.path(), "10.0.22621.0", b"new");
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    dxc_directives(&env, BinaryTargets::Bins, &mut out);

    let dst = profile(tmp.path()).join("dxcompiler.dll");
    assert_eq!(fs::read(&dst).unwrap(), b"new");
    assert!(out.contains(&"cargo::rustc-cfg=dxc_bundled".to_string()));
}

#[test]
fn dxc_skips_an_incomplete_newer_windows_sdk() {
    let tmp = TempDir::new().unwrap();
    install_winkits_dxc(tmp.path(), "10.0.19041.0", b"old");
    // The newer version has only one of the two DLLs.
    touch(
        &tmp.path()
            .join("winkits")
            .join("10.0.22621.0")
            .join("x64")
            .join("dxcompiler.dll"),
    );
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    dxc_directives(&env, BinaryTargets::Bins, &mut out);

    let dst = profile(tmp.path()).join("dxcompiler.dll");
    assert_eq!(fs::read(&dst).unwrap(), b"old");
}

#[test]
fn dxc_override_missing_a_dll_falls_through_to_windows_sdk() {
    let tmp = TempDir::new().unwrap();
    let override_dir = tmp.path().join("dxc-override");
    touch_with(&override_dir.join("dxcompiler.dll"), b"override");
    install_winkits_dxc(tmp.path(), "10.0.22621.0", b"winkits");
    let env = SdkEnv {
        dxc_root: Some(override_dir),
        ..env_in(tmp.path())
    };

    assert_eq!(
        find_dxc_dir(&env),
        Some(tmp.path().join("winkits").join("10.0.22621.0").join("x64"))
    );
}

#[test]
fn dxc_not_found_or_opted_out_warns_without_cfg() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    let mut missing = Vec::new();
    dxc_directives(&env, BinaryTargets::Bins, &mut missing);
    assert!(has(&missing, "dxcompiler.dll + dxil.dll not found"));
    assert!(!has(&missing, "dxc_bundled"));
    assert_eq!(find_dxc_dir(&env), None);

    let disabled_env = SdkEnv {
        dxc_enabled: false,
        ..env_in(tmp.path())
    };
    let mut disabled = Vec::new();
    dxc_directives(&disabled_env, BinaryTargets::Bins, &mut disabled);
    assert!(has(&disabled, "CN_ENABLE_DXC=0"));
    assert!(!has(&disabled, "CN_DXC_SDK"));
}

#[test]
fn copy_next_to_exe_reports_failures() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    // Missing source: warning recorded, copy reported failed.
    let mut out = Vec::new();
    let missing_src = tmp.path().join("nope.dll");
    assert!(!copy_next_to_exe(
        &env,
        BinaryTargets::Bins,
        &missing_src,
        "nope.dll",
        &mut out
    ));
    assert!(has(&out, "could not copy"));

    // No OUT_DIR: silently reported failed (matches build-script behavior).
    let no_out_env = SdkEnv {
        out_dir: None,
        ..env_in(tmp.path())
    };
    let mut quiet = Vec::new();
    let src = tmp.path().join("real.dll");
    touch(&src);
    assert!(!copy_next_to_exe(
        &no_out_env,
        BinaryTargets::Bins,
        &src,
        "real.dll",
        &mut quiet
    ));
    assert!(quiet.is_empty());
}

#[test]
fn copy_next_to_exe_skips_when_up_to_date() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());
    let src = tmp.path().join("up2date.dll");
    touch_with(&src, b"junk dll bytes");

    // First call copies the DLL into place.
    let mut out = Vec::new();
    assert!(copy_next_to_exe(
        &env,
        BinaryTargets::Bins,
        &src,
        "up2date.dll",
        &mut out
    ));
    let dst = profile(tmp.path()).join("up2date.dll");
    assert!(dst.is_file());

    // Overwrite the destination with same-length sentinel bytes; a skipped copy
    // leaves them untouched, a redundant copy would clobber them.
    touch_with(&dst, b"sentinel bytes");

    let mut second = Vec::new();
    assert!(copy_next_to_exe(
        &env,
        BinaryTargets::Bins,
        &src,
        "up2date.dll",
        &mut second
    ));
    assert_eq!(fs::read(&dst).unwrap(), b"sentinel bytes");
    // The source stays watched even when the copy is skipped.
    assert!(has(&second, "rerun-if-changed"));
}

#[test]
fn copy_next_to_exe_recopies_when_size_differs() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());
    let src = tmp.path().join("resized.dll");
    touch_with(&src, b"a longer set of dll bytes");

    // A stale destination of a different length is not up to date.
    let dst = profile(tmp.path()).join("resized.dll");
    touch_with(&dst, b"short");

    let mut out = Vec::new();
    assert!(copy_next_to_exe(
        &env,
        BinaryTargets::Bins,
        &src,
        "resized.dll",
        &mut out
    ));
    assert_eq!(fs::read(&dst).unwrap(), b"a longer set of dll bytes");
}

#[test]
fn copy_next_to_exe_recopies_when_source_newer() {
    use std::time::{Duration, SystemTime};

    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    // Same size, but the source is stamped newer than the destination.
    let src = tmp.path().join("newer.dll");
    touch_with(&src, b"fresh bytes!");
    let dst = profile(tmp.path()).join("newer.dll");
    touch_with(&dst, b"stale bytes!");

    // Setting the modified time needs a write handle on Windows.
    let stamp = |path: &Path, secs: u64| {
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
            .unwrap();
    };
    stamp(&dst, 100);
    stamp(&src, 200);

    let mut out = Vec::new();
    assert!(copy_next_to_exe(
        &env,
        BinaryTargets::Bins,
        &src,
        "newer.dll",
        &mut out
    ));
    assert_eq!(fs::read(&dst).unwrap(), b"fresh bytes!");
}

#[test]
fn profile_dir_walks_up_from_out_dir() {
    // Backslash paths only parse into components on Windows; on other hosts
    // `Path` treats the whole string as one component, so this case is
    // Windows-only. The Unix-style case below runs everywhere.
    #[cfg(windows)]
    {
        let out = Path::new("C:\\proj\\target\\release\\build\\concinnity-engine-abcd1234\\out");
        assert_eq!(
            profile_dir_from_out_dir(out),
            Some(Path::new("C:\\proj\\target\\release"))
        );
    }

    let out_debug = Path::new("/proj/target/debug/build/bistro-deadbeef/out");
    assert_eq!(
        profile_dir_from_out_dir(out_debug),
        Some(Path::new("/proj/target/debug"))
    );
}

#[test]
fn profile_dir_none_when_too_shallow() {
    assert_eq!(profile_dir_from_out_dir(Path::new("out")), None);
}

#[test]
fn version_dirs_sort_oldest_to_newest() {
    let dirs = vec![
        PathBuf::from("10.0.22621.0"),
        PathBuf::from("10.0.19041.0"),
        PathBuf::from("10.0.20348.0"),
    ];
    let sorted = sorted_version_dirs(dirs);
    assert_eq!(
        sorted,
        vec![
            PathBuf::from("10.0.19041.0"),
            PathBuf::from("10.0.20348.0"),
            PathBuf::from("10.0.22621.0"),
        ]
    );
}

#[test]
fn metal_and_non_windows_vulkan_are_noops() {
    let tmp = TempDir::new().unwrap();
    let env = SdkEnv {
        target_os: "linux".to_string(),
        ..env_in(tmp.path())
    };
    for targets in [
        &[BinaryTargets::None][..],
        &[BinaryTargets::Bins],
        &[BinaryTargets::Examples],
        &[BinaryTargets::Bins, BinaryTargets::Examples],
    ] {
        assert!(graphics_sdk_directives(Backend::Metal, targets, &env).is_empty());
        assert!(graphics_sdk_directives(Backend::Vk, targets, &env).is_empty());
    }
}

#[test]
fn windows_vulkan_sets_up_the_vulkan_sdks() {
    let tmp = TempDir::new().unwrap();
    install_ffx_vk_sdk(tmp.path(), b"stock sdk");
    install_ngx_lib(tmp.path());
    install_ngx_dll(tmp.path());
    install_xess(tmp.path());
    let env = env_in(tmp.path());

    let out = graphics_sdk_directives(Backend::Vk, &[BinaryTargets::Bins], &env);

    assert!(out.contains(&"cargo::rustc-cfg=ffx_sdk_bundled".to_string()));
    assert!(out.contains(&"cargo::rustc-cfg=ngx_sdk_bundled".to_string()));
    assert!(out.contains(&"cargo::rustc-cfg=xess_sdk_bundled".to_string()));
    // No DirectX-only setup on the Vulkan path.
    assert!(!has(&out, "CN_ENABLE_AGILITY_SDK"));
    assert!(!has(&out, "CN_ENABLE_DXC"));
}

#[test]
fn directx_bundling_runs_every_sdk() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    install_ffx_dx(tmp.path());
    install_xess(tmp.path());
    install_ngx_lib(tmp.path());
    install_ngx_dll(tmp.path());
    install_winkits_dxc(tmp.path(), "10.0.22621.0", b"winkits");
    let env = env_in(tmp.path());

    let out = graphics_sdk_directives(Backend::Dx, &[BinaryTargets::Bins], &env);

    for cfg in [
        "agility_sdk_configured",
        "ffx_sdk_bundled",
        "xess_sdk_bundled",
        "ngx_sdk_bundled",
        "dxc_bundled",
    ] {
        assert!(out.contains(&format!("cargo::rustc-cfg={cfg}")), "{cfg}");
    }
    assert!(warnings(&out).is_empty());

    let prof = profile(tmp.path());
    for file in [
        "D3D12/D3D12Core.dll",
        "D3D12/d3d12SDKLayers.dll",
        "amd_fidelityfx_dx12.dll",
        "libxess.dll",
        "nvngx_dlss.dll",
        "dxcompiler.dll",
        "dxil.dll",
    ] {
        assert!(prof.join(file).is_file(), "{file}");
    }
}

#[test]
fn directx_without_bundling_skips_dxc() {
    let tmp = TempDir::new().unwrap();
    install_winkits_dxc(tmp.path(), "10.0.22621.0", b"winkits");
    let env = env_in(tmp.path());

    let out = graphics_sdk_directives(Backend::Dx, &[BinaryTargets::None], &env);

    assert!(!has(&out, "CN_ENABLE_DXC"));
    assert!(!has(&out, "dxc_bundled"));
    assert!(!profile(tmp.path()).join("dxcompiler.dll").exists());
    // No warnings for absent SDKs when not producing a final binary.
    assert!(warnings(&out).is_empty());
}

#[test]
fn examples_take_their_dlls_and_exports_to_the_examples_directory() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    install_ffx_dx(tmp.path());
    install_xess(tmp.path());
    install_ngx_lib(tmp.path());
    install_ngx_dll(tmp.path());
    install_winkits_dxc(tmp.path(), "10.0.22621.0", b"winkits");
    let env = env_in(tmp.path());
    // Deliberately not creating `examples/` first: on a clean tree the build
    // script runs before Cargo lays it out, and the copies have to survive that.
    assert!(!examples(tmp.path()).exists());

    let out = graphics_sdk_directives(Backend::Dx, &[BinaryTargets::Examples], &env);
    assert!(warnings(&out).is_empty(), "{:?}", warnings(&out));

    // Every bundled file lands beside the example binaries, and `D3D12/` with
    // them: `D3D12SDKPath` resolves against the directory holding the .exe.
    for file in [
        "D3D12/D3D12Core.dll",
        "D3D12/d3d12SDKLayers.dll",
        "amd_fidelityfx_dx12.dll",
        "libxess.dll",
        "nvngx_dlss.dll",
        "dxcompiler.dll",
        "dxil.dll",
    ] {
        assert!(examples(tmp.path()).join(file).is_file(), "{file}");
        assert!(
            !profile(tmp.path()).join(file).exists(),
            "{file} must not land in the profile directory"
        );
    }
}

#[test]
fn the_agility_exports_are_scoped_to_the_targets_that_define_them() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    let env = env_in(tmp.path());

    // Cargo has no per-target key for examples and rejects the `-bins` key
    // outright from a package with no bin target, so the two kinds must not
    // share a directive.
    for (targets, key) in [
        (BinaryTargets::Bins, "cargo::rustc-link-arg-bins="),
        (BinaryTargets::Examples, "cargo::rustc-link-arg-examples="),
    ] {
        let mut out = Vec::new();
        agility_directives(&env, targets, &mut out);
        for symbol in ["D3D12SDKVersion", "D3D12SDKPath"] {
            assert!(
                out.contains(&format!("{key}/EXPORT:{symbol},DATA")),
                "{targets:?} {symbol}"
            );
        }
        assert_eq!(
            out.iter().filter(|l| l.contains("/EXPORT:")).count(),
            2,
            "{targets:?} emitted an export under another scope"
        );
    }
}

#[test]
fn a_package_without_binaries_emits_no_scoped_link_arg() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    let env = env_in(tmp.path());

    let mut out = Vec::new();
    agility_directives(&env, BinaryTargets::None, &mut out);
    assert!(!has(&out, "cargo::rustc-link-arg-"));
    assert_eq!(BinaryTargets::None.link_arg_key(), None);
}

#[test]
fn a_package_building_both_kinds_bundles_for_each_and_states_every_cfg_once() {
    let tmp = TempDir::new().unwrap();
    install_agility(tmp.path());
    install_ngx_lib(tmp.path());
    install_ngx_dll(tmp.path());
    let env = env_in(tmp.path());

    let out = graphics_sdk_directives(
        Backend::Dx,
        &[BinaryTargets::Bins, BinaryTargets::Examples],
        &env,
    );

    // Each kind takes its own export scope, since Cargo has no key covering
    // both.
    for key in [
        "cargo::rustc-link-arg-bins=",
        "cargo::rustc-link-arg-examples=",
    ] {
        assert!(
            out.contains(&format!("{key}/EXPORT:D3D12SDKVersion,DATA")),
            "{key}"
        );
    }

    // The DLLs land beside both sets of binaries.
    for dir in [profile(tmp.path()), examples(tmp.path())] {
        assert!(dir.join("D3D12/D3D12Core.dll").is_file(), "{dir:?}");
        assert!(dir.join("nvngx_dlss.dll").is_file(), "{dir:?}");
    }

    // Everything a kind does not scope is stated once: a cfg repeated is noise,
    // and a missing SDK warned about twice reads as two separate problems.
    for once in [
        "cargo::rustc-cfg=agility_sdk_configured",
        "cargo::rustc-cfg=ngx_sdk_bundled",
    ] {
        assert_eq!(
            out.iter().filter(|l| l.as_str() == once).count(),
            1,
            "{once}"
        );
    }
    let warned = warnings(&out);
    assert!(
        !warned.is_empty(),
        "this env installs only Agility and NGX, so the other SDKs must warn"
    );
    for warning in &warned {
        assert_eq!(
            warned.iter().filter(|w| w == &warning).count(),
            1,
            "warned twice: {warning}"
        );
    }
}

#[test]
fn only_examples_take_a_subdirectory_of_the_profile_dir() {
    let tmp = TempDir::new().unwrap();
    let env = env_in(tmp.path());

    assert_eq!(
        exe_dir(&env, BinaryTargets::None),
        Some(profile(tmp.path()))
    );
    assert_eq!(
        exe_dir(&env, BinaryTargets::Bins),
        Some(profile(tmp.path()))
    );
    assert_eq!(
        exe_dir(&env, BinaryTargets::Examples),
        Some(examples(tmp.path()))
    );

    let without_out_dir = SdkEnv {
        out_dir: None,
        ..env_in(tmp.path())
    };
    assert_eq!(exe_dir(&without_out_dir, BinaryTargets::Examples), None);
}
