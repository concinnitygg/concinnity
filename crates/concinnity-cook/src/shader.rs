//! Shader compilation seam. Dispatches on the source file extension to the
//! registered toolchain:
//!   .metal -> MSL   .hlsl -> HLSL   anything else -> GLSL
//!
//! This crate produces no shader bytecode itself: every backend's compiler needs
//! a platform toolchain (xcrun, the Direct3D compiler, shaderc), and the cook
//! runs on build hosts that have none of them. The concrete compilers live in
//! concinnity-shader and are installed here by the binary before a build.
//!
//! Every source this crate compiles is world-authored: the engine's own pipeline
//! shaders are compiled by the backend, not cooked into a world. Source is read
//! from disk and handed to the toolchain in memory; no shader source file is
//! written back out.

#[derive(Debug, Clone, Default)]
/// Inputs for compiling one world-authored shader stage.
pub struct ShaderCompileArgs {
    /// Path to the stage's source file.
    pub source_path: String,
    /// Name of the `Shader` asset the stage belongs to.
    pub asset_name: String,
    /// Which stage to compile, as its authored kind name.
    pub kind: String,
    /// An entry point this stage must define, when its role in the world forces
    /// one. A world declaring more than one Shader routes every draw through the
    /// GPU-driven bindless main pass, so each fragment stage has to expose
    /// `fragment_main_bindless`; checking it here fails `cn build` instead of the
    /// pipeline build, which for a scene-owned shader happens mid-session.
    pub required_entry: Option<String>,
}

/// The platform shader compilers for the backend this build targets, installed
/// by the binary through [`set_shader_toolchain`]. A toolchain implements only
/// the source languages its backend consumes; the rest report `Unsupported`
/// through the defaults, so a world authored for another backend fails the build
/// with a clear message rather than emitting bytecode nothing can load.
pub trait ShaderToolchain: Send + Sync {
    /// Compile Metal Shading Language source to a `.metallib`.
    fn compile_metal(
        &self,
        _source: &str,
        args: &ShaderCompileArgs,
    ) -> Result<Vec<u8>, std::io::Error> {
        Err(unsupported_language(".metal", args))
    }

    /// Compile HLSL source to shader bytecode.
    fn compile_hlsl(
        &self,
        _source: &str,
        args: &ShaderCompileArgs,
    ) -> Result<Vec<u8>, std::io::Error> {
        Err(unsupported_language(".hlsl", args))
    }

    /// Compile the GLSL source at `args.source_path` to SPIR-V. Takes the path
    /// rather than the source text: a GLSL source is never an engine built-in,
    /// and the GLSL compilers resolve `#include` relative to the source file.
    fn compile_glsl(&self, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
        Err(unsupported_language("GLSL", args))
    }
}

// The error a toolchain's default arm reports for a language it does not
// compile.
fn unsupported_language(language: &str, args: &ShaderCompileArgs) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!(
            "Asset '{}': {language} shaders are not supported by this build's shader toolchain",
            args.asset_name
        ),
    )
}

static SHADER_TOOLCHAIN: std::sync::OnceLock<Box<dyn ShaderToolchain>> = std::sync::OnceLock::new();

/// Register the process-wide shader toolchain. The first registration wins, so
/// build entry points can call this unconditionally.
pub fn set_shader_toolchain(toolchain: Box<dyn ShaderToolchain>) {
    let _ = SHADER_TOOLCHAIN.set(toolchain);
}

// Unwrap the toolchain at the point a compiler is actually needed. Not having
// one is a wiring mistake in the binary rather than a fault in the world being
// built, so the message names the missing step.
fn require<'a>(
    toolchain: Option<&'a dyn ShaderToolchain>,
    args: &ShaderCompileArgs,
) -> Result<&'a dyn ShaderToolchain, std::io::Error> {
    toolchain.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!(
                "Asset '{}': no shader toolchain is registered, so no shader can be compiled",
                args.asset_name
            ),
        )
    })
}

/// Backend hook the build pipeline calls after a user shader compiles, so a
/// render backend can validate that the shader's engine-provided buffer structs
/// (per-frame uniforms, object data, lights, ...) have the same memory layout
/// as the engine's `#[repr(C)]` Rust structs. Catches CPU/GPU layout mismatches
/// at `cn build` with a clear message instead of as a GPU page fault at `cn run`.
///
/// The build pipeline (this crate) is backend-agnostic and never links a GPU
/// API, so the actual reflection lives in the client's render backend, which
/// installs an implementation via [`set_shader_build_validator`]. When none is
/// registered (a core-only build, a non-macOS host, the server) the call is a
/// no-op and the build is unaffected.
pub trait ShaderBuildValidator: Send + Sync {
    /// Validate one compiled shader stage. `source` is the shader source text,
    /// `kind` the compile kind (`"vertex"` or `"fragment"`; a shadow stage
    /// compiles as `"vertex"` and is told apart by its entry-point name),
    /// `asset_name` the declaring asset. Return `Err(msg)` to fail the build.
    fn validate_metal(&self, source: &str, kind: &str, asset_name: &str) -> Result<(), String>;

    /// Confirm the source defines `entry`. Called only when the world's shader
    /// set forces an entry point (see `ShaderCompileArgs::required_entry`).
    /// Default `Ok`: a backend without reflection cannot check, and a missing
    /// entry point still fails when the pipeline is built.
    fn validate_metal_entry(
        &self,
        source: &str,
        entry: &str,
        asset_name: &str,
    ) -> Result<(), String> {
        let _ = (source, entry, asset_name);
        Ok(())
    }
}

static SHADER_BUILD_VALIDATOR: std::sync::OnceLock<Box<dyn ShaderBuildValidator>> =
    std::sync::OnceLock::new();

/// Register the process-wide shader build validator. The first registration
/// wins; later calls are ignored, so build entry points can call this
/// unconditionally (the backend installs exactly one validator).
pub fn set_shader_build_validator(validator: Box<dyn ShaderBuildValidator>) {
    let _ = SHADER_BUILD_VALIDATOR.set(validator);
}

// Run the registered validator against a just-compiled `.metal` source. A no-op
// when no validator is registered. A validation error is surfaced as an
// `InvalidData` build error so `cn build` fails.
fn validate_compiled_metal(source: &str, args: &ShaderCompileArgs) -> Result<(), std::io::Error> {
    let Some(validator) = SHADER_BUILD_VALIDATOR.get() else {
        return Ok(());
    };
    let invalid = |msg| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    validator
        .validate_metal(source, &args.kind, &args.asset_name)
        .map_err(invalid)?;
    if let Some(entry) = &args.required_entry {
        validator
            .validate_metal_entry(source, entry, &args.asset_name)
            .map_err(invalid)?;
    }
    Ok(())
}

/// Compile one world-authored shader stage to the backend's binary format.
pub fn compile_shader(args: ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    compile_with(SHADER_TOOLCHAIN.get().map(|t| t.as_ref()), &args)
}

// Route one stage to the toolchain's arm for its source language. Source
// resolution happens before the toolchain is unwrapped, so a source that cannot
// be read fails as a read error rather than as a missing compiler.
fn compile_with(
    toolchain: Option<&dyn ShaderToolchain>,
    args: &ShaderCompileArgs,
) -> Result<Vec<u8>, std::io::Error> {
    let ext = std::path::Path::new(&args.source_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // .metal/.hlsl source is resolved here and passed to the toolchain in
    // memory; .glsl is only ever caller-supplied (never a built-in), so that
    // path hands over the source path instead.
    match ext {
        "metal" => {
            let source = read_shader_source(&args.source_path)?;
            let bytes = require(toolchain, args)?.compile_metal(&source, args)?;
            // The shader compiled; now check that every engine-provided buffer
            // struct it reads matches the engine's layout. A mismatch fails the
            // build here rather than faulting the GPU at run time.
            validate_compiled_metal(&source, args)?;
            Ok(bytes)
        }
        "hlsl" => {
            let source = read_shader_source(&args.source_path)?;
            require(toolchain, args)?.compile_hlsl(&source, args)
        }
        _ => require(toolchain, args)?.compile_glsl(args),
    }
}

// Resolve a shader source path to its text.
fn read_shader_source(source_path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(source_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("Failed to read shader source '{}': {}", source_path, e),
        )
    })
}

// A toolchain for tests whose worlds pull in a Shader but assert on the
// surrounding build rather than on bytecode. Registering it keeps the cook's own
// tests off the platform compilers, so they behave identically on every host.
#[cfg(test)]
pub(crate) fn install_stub_toolchain() {
    struct StubToolchain;

    impl ShaderToolchain for StubToolchain {
        fn compile_metal(
            &self,
            _source: &str,
            _args: &ShaderCompileArgs,
        ) -> Result<Vec<u8>, std::io::Error> {
            Ok(b"stub-shader".to_vec())
        }

        fn compile_hlsl(
            &self,
            _source: &str,
            _args: &ShaderCompileArgs,
        ) -> Result<Vec<u8>, std::io::Error> {
            Ok(b"stub-shader".to_vec())
        }

        fn compile_glsl(&self, _args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
            Ok(b"stub-shader".to_vec())
        }
    }

    set_shader_toolchain(Box::new(StubToolchain));
}

#[cfg(test)]
fn args_for(asset_name: &str, source_path: &str) -> ShaderCompileArgs {
    ShaderCompileArgs {
        source_path: source_path.to_string(),
        asset_name: asset_name.to_string(),
        kind: "fragment".to_string(),
        ..Default::default()
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    use super::args_for as args;

    // A toolchain that compiles nothing but names the arm it was routed to, so
    // dispatch is assertable without any platform compiler. Passed to
    // `compile_with` directly rather than registered, so these tests neither
    // depend on nor disturb the process-wide toolchain.
    struct ArmNamingToolchain;

    impl ShaderToolchain for ArmNamingToolchain {
        fn compile_metal(
            &self,
            source: &str,
            _args: &ShaderCompileArgs,
        ) -> Result<Vec<u8>, std::io::Error> {
            Ok(format!("metal:{source}").into_bytes())
        }

        fn compile_hlsl(
            &self,
            source: &str,
            _args: &ShaderCompileArgs,
        ) -> Result<Vec<u8>, std::io::Error> {
            Ok(format!("hlsl:{source}").into_bytes())
        }

        fn compile_glsl(&self, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
            Ok(format!("glsl:{}", args.source_path).into_bytes())
        }
    }

    fn routed(source_path: &str) -> Result<String, std::io::Error> {
        compile_with(Some(&ArmNamingToolchain), &args("user", source_path))
            .map(|b| String::from_utf8(b).expect("test toolchain emits utf8"))
    }

    // Write a shader source into `dir` and return its path.
    fn source_file(dir: &std::path::Path, name: &str, text: &str) -> String {
        let path = dir.join(name);
        std::fs::write(&path, text).expect("write source");
        path.to_string_lossy().into_owned()
    }

    // A `.metal` source with no file on disk fails in `read_shader_source`
    // before the toolchain is consulted, so no compiler runs. The read happens
    // in the dispatch arm, ahead of the unwrap.
    #[test]
    fn missing_metal_source_fails_at_read_before_any_compile() {
        let err = routed("/no/such/user_frag.metal").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string().contains("Failed to read shader source"),
            "got: {err}"
        );
    }

    // A `.hlsl` source routes to the HLSL arm with its text, on every backend.
    #[test]
    fn an_hlsl_source_routes_to_the_hlsl_arm() {
        let dir = tempfile::tempdir().unwrap();
        let path = source_file(dir.path(), "user_frag.hlsl", "// hlsl");
        assert!(
            routed(&path).unwrap().starts_with("hlsl:"),
            "expected the hlsl arm"
        );
    }

    // The `.hlsl` arm reads its source in the dispatch match too, so a missing
    // file fails there rather than in the toolchain.
    #[test]
    fn missing_hlsl_source_fails_at_read_before_any_compile() {
        let err = routed("/no/such/user_frag.hlsl").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string().contains("Failed to read shader source"),
            "got: {err}"
        );
    }

    // Anything that is not `.metal` or `.hlsl` falls through to the GLSL arm,
    // which hands over the path instead of reading it here -- a path that does
    // not exist still reaches the compiler.
    #[test]
    fn other_extensions_route_to_the_glsl_arm_without_reading() {
        assert_eq!(
            routed("/no/such/user_frag.glsl").unwrap(),
            "glsl:/no/such/user_frag.glsl"
        );
    }

    // A `.metal` source routes to the MSL arm carrying the resolved source
    // text, not the path.
    #[test]
    fn a_metal_source_routes_to_the_metal_arm_with_its_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = source_file(dir.path(), "user.metal", "vertex void main() {}");
        let out = routed(&path).unwrap();
        assert!(out.starts_with("metal:"), "expected the metal arm: {out}");
        assert!(out.contains("vertex"), "expected the source text");
    }

    // With no toolchain installed, a source that resolves still fails -- and
    // says so as a wiring problem rather than a fault in the world.
    #[test]
    fn without_a_toolchain_a_resolvable_source_reports_the_missing_toolchain() {
        let dir = tempfile::tempdir().unwrap();
        let path = source_file(dir.path(), "user.metal", "// msl");
        let err = compile_with(None, &args("user", &path)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no shader toolchain is registered"),
            "got: {err}"
        );
        assert!(err.to_string().contains("user"), "names the asset: {err}");
    }

    // The read still runs first without a toolchain, so an unreadable source is
    // reported as the read error it is.
    #[test]
    fn without_a_toolchain_an_unreadable_source_still_fails_at_read() {
        let err = compile_with(None, &args("user", "/no/such/user_frag.metal")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn a_read_error_reports_the_path_and_keeps_the_io_error_kind() {
        let err = read_shader_source("/no/such/user.metal").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string()
                .starts_with("Failed to read shader source '/no/such/user.metal'"),
            "got: {err}"
        );
    }

    #[test]
    fn an_on_disk_source_is_read_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("user.metal");
        std::fs::write(&path, "fragment float4 f() { return 0; }").unwrap();
        assert_eq!(
            read_shader_source(&path.to_string_lossy()).unwrap(),
            "fragment float4 f() { return 0; }"
        );
    }
}

#[cfg(test)]
mod toolchain_tests {
    use super::args_for as args;
    use super::*;

    // A toolchain that implements no language, so every call falls to the
    // trait's default arm.
    struct NoLanguages;
    impl ShaderToolchain for NoLanguages {}

    #[test]
    fn a_language_the_toolchain_does_not_implement_is_unsupported() {
        let t = NoLanguages;
        let a = args("user", "user_frag.hlsl");
        for err in [
            t.compile_metal("src", &a).unwrap_err(),
            t.compile_hlsl("src", &a).unwrap_err(),
            t.compile_glsl(&a).unwrap_err(),
        ] {
            assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
            assert!(err.to_string().contains("user"), "names the asset: {err}");
        }
    }

    // A partial toolchain keeps its own arms and falls back to the defaults for
    // the rest, so a world authored for another backend fails on the language
    // rather than on the missing registration.
    #[test]
    fn a_partial_toolchain_keeps_its_own_arms() {
        struct MetalOnly;
        impl ShaderToolchain for MetalOnly {
            fn compile_metal(
                &self,
                _source: &str,
                _args: &ShaderCompileArgs,
            ) -> Result<Vec<u8>, std::io::Error> {
                Ok(vec![1])
            }
        }
        let a = args("user", "user_frag.metal");
        assert_eq!(MetalOnly.compile_metal("src", &a).unwrap(), vec![1]);
        assert_eq!(
            MetalOnly.compile_hlsl("src", &a).unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
    }
}

#[cfg(test)]
mod hook_tests {
    use super::args_for as args;
    use super::*;

    // A validator that only objects to one sentinel asset name, so registering
    // it process-wide (the OnceLock can only be set once per test binary) leaves
    // every other asset's build untouched.
    struct SentinelValidator;
    const SENTINEL: &str = "__layout_hook_sentinel__";

    impl ShaderBuildValidator for SentinelValidator {
        fn validate_metal(
            &self,
            _source: &str,
            _kind: &str,
            asset_name: &str,
        ) -> Result<(), String> {
            if asset_name == SENTINEL {
                Err("sentinel layout mismatch".to_string())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn validator_hook_dispatches_per_asset() {
        set_shader_build_validator(Box::new(SentinelValidator));

        // A user source for the sentinel asset surfaces the validator's error.
        let err = validate_compiled_metal("frag source", &args(SENTINEL, "user_frag.metal"))
            .expect_err("sentinel must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("sentinel layout mismatch"));

        // Any other asset passes through cleanly.
        validate_compiled_metal("frag source", &args("ok_asset", "user_frag.metal"))
            .expect("non-sentinel assets pass");
    }
}
