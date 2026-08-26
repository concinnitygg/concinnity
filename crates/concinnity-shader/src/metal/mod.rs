// The Metal toolchain: MSL through `xcrun metal` + `xcrun metallib`, and GLSL
// through `glslc` (macOS ships no in-process GLSL compiler, and a world may
// still carry a `.glsl` stage for a sibling backend).

mod validator;

use concinnity_cook::shader::{ShaderCompileArgs, ShaderToolchain, set_shader_toolchain};

pub(crate) fn install() {
    set_shader_toolchain(Box::new(MetalToolchain));
    validator::register_shader_layout_validator();
}

struct MetalToolchain;

impl ShaderToolchain for MetalToolchain {
    fn compile_metal(
        &self,
        source: &str,
        args: &ShaderCompileArgs,
    ) -> Result<Vec<u8>, std::io::Error> {
        compile_metal(source, args)
    }

    fn compile_glsl(&self, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
        compile_glsl(args)
    }
}

// A unique temp-file stem for a shader compile's transient artifacts (`.air` /
// `.metallib` / `.spv`). Keying on the process id plus a per-process counter makes
// it unique across concurrent compiles in the same process (parallel builds, the
// test suite) and across separate processes, so they never collide on one path; and
// rooting it in the OS temp dir keeps the working directory untouched.
fn transient_stem(asset_name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "cn-shader-{}-{}-{}",
        asset_name,
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

fn compile_metal(source: &str, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    use std::fs;
    use std::io::Write;
    use std::process::Stdio;

    // Transient intermediates go to a UNIQUE path in the OS temp dir (not a shared
    // the state root's `data/<name>`): parallel compiles of
    // the same shader -- concurrent builds, or the test suite cooking several worlds
    // at once -- must not race on one path (one removing the file mid-read of
    // another), and a cook must not read or write the working directory.
    let stem = transient_stem(&args.asset_name);
    let air_path = format!("{}.air", stem.display());
    let lib_path = format!("{}.metallib", stem.display());

    // Feed the source to `xcrun metal` over stdin (`-x metal` selects the
    // language since stdin has no extension, `-` is the stdin input) so no
    // shader source file is written to disk. The .air and .metallib it emits
    // are intermediate artifacts, removed once read.
    let mut metal = std::process::Command::new("xcrun")
        .args(["metal", "-x", "metal", "-c", "-", "-o", &air_path])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    metal
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(source.as_bytes())?;
    let metal_output = metal.wait_with_output()?;

    if !metal_output.status.success() {
        let _ = fs::remove_file(&air_path);
        return Err(std::io::Error::other(format!(
            "xcrun metal failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&metal_output.stdout),
            String::from_utf8_lossy(&metal_output.stderr),
        )));
    }

    let lib_output = std::process::Command::new("xcrun")
        .args(["metallib", &air_path, "-o", &lib_path])
        .output()?;

    let _ = fs::remove_file(&air_path);

    if !lib_output.status.success() {
        let _ = fs::remove_file(&lib_path);
        return Err(std::io::Error::other(format!(
            "xcrun metallib failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&lib_output.stdout),
            String::from_utf8_lossy(&lib_output.stderr),
        )));
    }

    let bytes = fs::read(&lib_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("Failed to read metallib '{}': {}", lib_path, e),
        )
    })?;

    let _ = fs::remove_file(&lib_path);

    Ok(bytes)
}

fn compile_glsl(args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    // A unique temp path (see `transient_stem`): keeps parallel compiles from racing
    // on a shared file and leaves the working directory untouched.
    let out_path = format!("{}.spv", transient_stem(&args.asset_name).display());

    let output = std::process::Command::new("glslc")
        .args([
            "--target-env=vulkan1.0",
            "-fshader-stage",
            &args.kind,
            &args.source_path,
            "-o",
            &out_path,
        ])
        .output()?;

    if !output.status.success() {
        let _ = std::fs::remove_file(&out_path);
        return Err(std::io::Error::other(format!(
            "glslc failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )));
    }

    let bytes = std::fs::read(&out_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to read SPIR-V '{}': {}", out_path, e),
        )
    })?;

    let _ = std::fs::remove_file(&out_path);

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(asset_name: &str) -> ShaderCompileArgs {
        ShaderCompileArgs {
            source_path: "user_frag.metal".to_string(),
            asset_name: asset_name.to_string(),
            kind: "fragment".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn a_transient_stem_is_unique_per_call_and_lives_in_the_temp_dir() {
        let a = transient_stem("asset");
        let b = transient_stem("asset");
        assert_ne!(a, b, "two compiles of one asset must not share a path");
        assert!(a.starts_with(std::env::temp_dir()));
    }

    // The Metal compiler ships with the Xcode command line tools. A machine
    // without them cannot build a world at all, so the tests that need a real
    // compile skip rather than fail there.
    fn metal_compiler_available() -> bool {
        std::process::Command::new("xcrun")
            .args(["--find", "metal"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn a_source_that_does_not_compile_reports_the_asset_and_the_tool() {
        if !metal_compiler_available() {
            return;
        }
        let err = compile_metal("this is not valid MSL", &args("bad_stage")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bad_stage"), "names the asset: {msg}");
        assert!(msg.contains("xcrun metal"), "names the tool: {msg}");
    }

    #[test]
    fn a_valid_source_compiles_to_metallib_bytes() {
        if !metal_compiler_available() {
            return;
        }
        let src = "#include <metal_stdlib>\nusing namespace metal;\n\
                   fragment float4 f() { return float4(0.0); }\n";
        let bytes = compile_metal(src, &args("good_stage")).expect("valid MSL compiles");
        assert!(!bytes.is_empty(), "a metallib is never empty");
    }

    // Neither the success nor the failure path leaves an intermediate behind.
    #[test]
    fn a_compile_removes_its_transient_artifacts() {
        if !metal_compiler_available() {
            return;
        }
        let leaked = || {
            std::fs::read_dir(std::env::temp_dir())
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with(&format!("cn-shader-cleanup-{}-", std::process::id()))
                })
                .count()
        };
        let before = leaked();
        let _ = compile_metal("this is not valid MSL", &args("cleanup"));
        let src = "#include <metal_stdlib>\nusing namespace metal;\n\
                   fragment float4 f() { return float4(0.0); }\n";
        let _ = compile_metal(src, &args("cleanup"));
        assert_eq!(leaked(), before, "a compile must clean up after itself");
    }
}
