// The Metal toolchain: MSL through `xcrun metal` + `xcrun metallib`, and GLSL
// through `glslc` (macOS ships no in-process GLSL compiler, and a world may
// still carry a `.glsl` stage for a sibling backend).

mod validator;

use concinnity_cook::compile::shader::{ShaderCompileArgs, ShaderToolchain, set_shader_toolchain};
use concinnity_host::scratch::Scratch;

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

fn compile_metal(source: &str, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    use std::fs;
    use std::io::Write;
    use std::process::Stdio;

    // Transient intermediates go to scratch paths of their own: parallel
    // compiles of one shader -- concurrent builds, or the test suite cooking
    // several worlds at once -- must not race on one path, and a cook must not
    // read or write the working directory. Each removes itself, so the error
    // paths below carry no cleanup.
    let air = Scratch::file(&format!("{}.air", args.asset_name));
    let lib = Scratch::file(&format!("{}.metallib", args.asset_name));

    // Feed the source to `xcrun metal` over stdin (`-x metal` selects the
    // language since stdin has no extension, `-` is the stdin input) so no
    // shader source file is written to disk. The .air and .metallib it emits
    // are intermediate artifacts, removed once read.
    let mut metal = std::process::Command::new("xcrun")
        .args(["metal", "-x", "metal", "-c", "-"])
        .arg("-o")
        .arg(air.path())
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
        return Err(std::io::Error::other(format!(
            "xcrun metal failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&metal_output.stdout),
            String::from_utf8_lossy(&metal_output.stderr),
        )));
    }

    let lib_output = std::process::Command::new("xcrun")
        .arg("metallib")
        .arg(air.path())
        .arg("-o")
        .arg(lib.path())
        .output()?;

    if !lib_output.status.success() {
        return Err(std::io::Error::other(format!(
            "xcrun metallib failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&lib_output.stdout),
            String::from_utf8_lossy(&lib_output.stderr),
        )));
    }

    fs::read(lib.path()).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("Failed to read metallib '{}': {}", lib.path().display(), e),
        )
    })
}

fn compile_glsl(args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    // Its own scratch path, removed with the guard: parallel compiles must not
    // race on one file, and the working directory stays untouched.
    let out = Scratch::file(&format!("{}.spv", args.asset_name));

    let output = std::process::Command::new("glslc")
        .args([
            "--target-env=vulkan1.0",
            "-fshader-stage",
            &args.kind,
            &args.source_path,
        ])
        .arg("-o")
        .arg(out.path())
        .output()?;

    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "glslc failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )));
    }

    std::fs::read(out.path()).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to read SPIR-V '{}': {}", out.path().display(), e),
        )
    })
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
        // Where a scratch path lives, without naming the temporary directory:
        // `file_access_discipline` leaves that to `concinnity_host::scratch`.
        let probe = Scratch::file("probe");
        let root = probe
            .path()
            .parent()
            .expect("scratch has a parent")
            .to_path_buf();
        let leaked = || {
            std::fs::read_dir(&root)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with(&format!("cn-{}-", std::process::id()))
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
