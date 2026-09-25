//! The HLSL shader toolchain, shared by build scripts and the renderer.
//!
//! The engine's single-source shaders
//! (`crates/concinnity-core/src/render/shaders/*.hlsl`) compile through one
//! pipeline: `dxc` emits DXIL for D3D12 and SPIR-V for Vulkan, and the Metal
//! leg takes that same SPIR-V through spirv-cross to MSL and the Metal
//! toolchain to a metallib (see [`metallib`], the only caller of that
//! toolchain in the engine). Every call site assembles the full source text first (defines
//! injected as `#define` lines), so a compile is a pure function of that text,
//! the entry point, and the target -- which is what lets the renderer's
//! content-addressed shader cache key it.
//!
//! Being needed on both sides is why this is its own crate rather than a module
//! of `concinnity-toolchain`: that crate is build-script support, consumed only
//! under `[build-dependencies]` and never linked into a shipped binary, and it
//! stays that way. This one sits below it and holds no policy beyond two
//! things no compiler can be asked for: the Metal binding table, which
//! `metal_bindings` derives from the source's own annotations rather than from
//! a hand-maintained list, and what each backend loads a cooked program as
//! ([`HlslTarget::cooked`]), which the cook and a renderer's fallback compile
//! must agree on.
//!
//! dxc resolves from `dxc/bin` beside the running executable, then the
//! checkout's pinned release under `vendor/`, then PATH, then `$VULKAN_SDK/bin`
//! (see `locate`). Everything the engine draws is compiled ahead of time, so a
//! host with none of these candidates runs a world fine and fails only when
//! something has to be compiled: an authored shader, or an edited one.
//!
//! The MSL leg and the layout reflection are behind the `spirv-cross` feature,
//! which links spirv-cross itself. A DirectX or Vulkan build needs neither and
//! builds no C++.

include!(concat!(env!("OUT_DIR"), "/source_hash.rs"));

pub mod declarations;
pub mod diagnostics;
mod dxc;
#[cfg(feature = "spirv-cross")]
pub mod layout;
mod locate;
#[cfg(feature = "spirv-cross")]
mod metal_bindings;
pub mod metallib;
#[cfg(feature = "spirv-cross")]
mod msl;
mod stage;
mod vendored;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use concinnity_core::platform::Platform;

pub use stage::{Stage, stage_of};
pub use vendored::vendored_releases;

/// The shader model every program compiles at unless it asks for 6.5: the
/// floor `NonUniformResourceIndex` needs, which the bindless pool does.
const BASE_MODEL: (u32, u32) = (6, 0);

/// The first shader model with an inline `RayQuery`.
const MODEL_6_5: (u32, u32) = (6, 5);

/// What dxc and the legs behind it should emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HlslTarget {
    /// SPIR-V for Vulkan, entry point renamed to `main`.
    Spirv,
    /// A DXIL container for D3D12, which dxc's own validator signs on every
    /// host.
    Dxil {
        /// Compile at shader model 6.5 rather than 6.0, as an entry reaching
        /// an inline `RayQuery` must. A DXIL container records its model, and
        /// dxc lowers some operations differently under it.
        shader_model_6_5: bool,
    },
    /// A Metal library. Needs the `spirv-cross` feature.
    Metallib,
    /// MSL source text. Needs the `spirv-cross` feature.
    Msl,
    /// SPIR-V under Vulkan layout rules with every declared resource kept,
    /// read or not, for reading a program's struct layouts back. Never
    /// shipped: only [`layout::struct_layouts`] reads it.
    SpirvWithVulkanLayout,
    /// SPIR-V laid out the way DXIL lays its buffers out, with every declared
    /// resource kept, for reading the DirectX leg's struct layouts back on a
    /// host with no D3D12 reflection interface. Never shipped: only
    /// [`layout::struct_layouts`] reads it.
    SpirvWithDxLayout,
}

impl HlslTarget {
    /// What `platform`'s renderer loads a program it did not build in as: MSL
    /// text for Metal's `newLibraryWithSource`, a DXIL container for D3D12,
    /// SPIR-V for Vulkan. The cook stores this for a world's shaders, and a
    /// renderer compiling one itself must emit the same, or it produces
    /// something the cook never would.
    #[must_use]
    pub fn cooked(platform: Platform) -> Self {
        match platform {
            Platform::Metal => HlslTarget::Msl,
            Platform::DirectX => HlslTarget::Dxil {
                shader_model_6_5: false,
            },
            Platform::Vulkan => HlslTarget::Spirv,
        }
    }

    /// A short name for this target, which a shader cache keys an artifact
    /// under so two targets' artifacts of one source never collide.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            HlslTarget::Spirv => "spirv",
            HlslTarget::Dxil {
                shader_model_6_5: false,
            } => "dxil",
            HlslTarget::Dxil {
                shader_model_6_5: true,
            } => "dxil-6.5",
            HlslTarget::Metallib => "metallib",
            HlslTarget::Msl => "msl",
            HlslTarget::SpirvWithVulkanLayout => "spirv-vulkan-layout",
            HlslTarget::SpirvWithDxLayout => "spirv-dx-layout",
        }
    }

    // The dxc profile an entry of `stage` compiles at for this target. SPIR-V
    // gates a ray query on a capability rather than on the profile, so every
    // leg but a 6.5 DXIL program compiles at the base model.
    fn profile(self, stage: Stage) -> String {
        let (major, minor) = match self {
            HlslTarget::Dxil {
                shader_model_6_5: true,
            } => MODEL_6_5,
            _ => BASE_MODEL,
        };
        stage.profile(major, minor)
    }
}

/// How a compile treats the warnings dxc reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Warnings {
    /// A warning fails the compile. The engine's own shaders build this way,
    /// so a misspelled attribute is a build break rather than a log line.
    Deny,
    /// A warning is handed back beside the artifact, for authored content
    /// whose author reads it as a diagnostic.
    Report,
}

/// An artifact and whatever dxc warned while producing it.
#[derive(Debug)]
pub struct Compiled {
    /// The artifact bytes.
    pub artifact: Vec<u8>,
    /// dxc's warnings, verbatim, or `None` when it printed none.
    pub warnings: Option<String>,
}

/// One compile: assembled source text in, artifact bytes out.
pub struct HlslJob<'a> {
    /// Fully assembled source (defines already injected as `#define` lines).
    pub source: &'a str,
    /// File name the source is written under in `work_dir`; also the name dxc
    /// diagnostics and `#line` directives carry.
    pub file_name: &'a str,
    /// Entry point compiled out of the source. Its `[shader("...")]`
    /// attribute states the stage.
    pub entry: &'a str,
    /// What to emit.
    pub target: HlslTarget,
}

// A usable compiler: the binary to invoke and its `compiler_id`.
struct Dxc {
    path: PathBuf,
    id: String,
}

// Resolved once per process.
fn resolved() -> &'static Result<Dxc, String> {
    static DXC: OnceLock<Result<Dxc, String>> = OnceLock::new();
    DXC.get_or_init(probe_dxc)
}

/// The dxc to invoke, or `None` when no candidate answered.
#[must_use]
pub fn dxc_path() -> Option<&'static Path> {
    resolved().as_ref().ok().map(|d| d.path.as_path())
}

/// Whether this host can compile `.hlsl` sources.
///
/// The cook asks before compiling a Shader or an SdfVolume field and fails
/// naming the asset when the answer is no, so a build never quietly produces a
/// world missing what it declared.
///
/// ```
/// // True only where a dxc resolves.
/// let _ = concinnity_shader::dxc_available();
/// ```
#[must_use]
pub fn dxc_available() -> bool {
    dxc_path().is_some()
}

/// Why no candidate answered, or `None` when one did.
#[must_use]
pub fn unavailable_reason() -> Option<&'static str> {
    resolved().as_ref().err().map(String::as_str)
}

/// Identifies the compiler for the renderer's content-addressed shader cache.
/// Two dxc releases can emit different bytes for identical source, so an
/// artifact keyed without the version outlives the toolchain that produced it
/// and gets served to a later one.
///
/// The id is the version and commit dxc reports, `dxc 1.9 0d3ee6b5`, and not
/// the build counter beside them: that counts whatever history the builder had
/// cloned, so two builds of one commit disagree on it.
#[must_use]
pub fn compiler_id() -> &'static str {
    resolved().as_ref().map_or("dxc", |d| d.id.as_str())
}

// The `<major>.<minor> <commit>` in a `--version` line such as
// `dxcompiler.dll: 1.9(5402-0d3ee6b5)(1.9.0.5402) - 1.9.0.5402 (0d3ee6b55-dirty)`,
// or `None` for a build that names no commit (`libdxcompiler.so: 1.8(dev)`).
fn release_id(line: &str) -> Option<String> {
    let rest = line.split_once(": ").map_or(line, |(_, rest)| rest);
    let (version, tail) = rest.split_once('(')?;
    let build = tail.split_once(')')?.0;
    let commit = build.rsplit('-').next()?;
    let dotted = version.split('.').count() == 2
        && version
            .split('.')
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    let hex = commit.len() >= 7 && commit.bytes().all(|b| b.is_ascii_hexdigit());
    (dotted && hex && build.contains('-')).then(|| format!("{version} {commit}"))
}

fn probe_dxc() -> Result<Dxc, String> {
    for path in locate::dxc_candidates() {
        // dxc prints its release on stdout and exits 0.
        if let Some(version) = stdout_line(Command::new(&path).arg("--version")) {
            let id = format!(
                "dxc {}",
                release_id(&version).unwrap_or_else(|| version.clone())
            );
            return Ok(Dxc { path, id });
        }
    }
    Err(format!(
        "dxc not found: the engine's single-source shaders need the DirectX \
         Shader Compiler. In an engine checkout, `scripts/vendor.py fetch dxc` \
         vendors the pinned release (`build dxc` on macOS, which Microsoft \
         publishes no binary for). Otherwise install a release \
         (https://github.com/microsoft/DirectXShaderCompiler/releases) and put \
         its `{}` first on PATH.",
        locate::EXE
    ))
}

// What `cmd` printed on stdout, or `None` when it failed to run or exited
// non-zero.
pub(crate) fn stdout(cmd: &mut Command) -> Option<String> {
    let out = cmd.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

// The first line of `stdout`, trimmed, or `None` when it is empty.
pub(crate) fn stdout_line(cmd: &mut Command) -> Option<String> {
    let text = stdout(cmd)?;
    let line = text.lines().next()?.trim();
    (!line.is_empty()).then(|| line.to_string())
}

/// Compile `job` under `work_dir`, returning the artifact bytes. A warning
/// fails the compile.
pub fn compile(job: &HlslJob<'_>, work_dir: &Path) -> Result<Vec<u8>, String> {
    with_scratch(work_dir, |dxc, scratch| {
        produce(job, dxc, scratch, Warnings::Deny)
    })
    .map(|compiled| compiled.artifact)
}

/// Compile `job` under `work_dir`, returning the artifact with any warnings
/// dxc printed. The artifact is byte-identical to what
/// [`compile`] emits for a source that warns of nothing.
pub fn compile_with_warnings(job: &HlslJob<'_>, work_dir: &Path) -> Result<Compiled, String> {
    with_scratch(work_dir, |dxc, scratch| {
        produce(job, dxc, scratch, Warnings::Report)
    })
}

// Run `f` with the resolved dxc and a scratch directory of its own under
// `work_dir`.
fn with_scratch<T>(
    work_dir: &Path,
    f: impl FnOnce(&Path, &Path) -> Result<T, String>,
) -> Result<T, String> {
    let dxc = match resolved() {
        Ok(found) => found.path.as_path(),
        Err(message) => return Err(message.clone()),
    };
    in_scratch(work_dir, |scratch| f(dxc, scratch))
}

// Run `f` in a scratch directory of its own under `work_dir`, removed
// afterwards whatever `f` returned.
pub(crate) fn in_scratch<T>(
    work_dir: &Path,
    f: impl FnOnce(&Path) -> Result<T, String>,
) -> Result<T, String> {
    let scratch = work_dir.join(scratch_name());
    std::fs::create_dir_all(&scratch)
        .map_err(|e| format!("hlsl: create {}: {e}", scratch.display()))?;
    let result = f(&scratch);
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

// Run `cmd`, returning what it printed, and folding that into the error when it
// fails. `what` names the tool and `source` the file it was working on.
pub(crate) fn run(cmd: &mut Command, what: &str, source: &str) -> Result<String, String> {
    let output = cmd
        .output()
        .map_err(|e| format!("hlsl: {what} failed to launch for {source}: {e}"))?;
    let printed = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    if output.status.success() {
        return Ok(printed);
    }
    Err(format!("hlsl: {what} failed for {source}:\n{printed}"))
}

// Every invocation gets its own subdirectory: two compiles of the same
// `file_name` can run at once (one shared source serves several programs), and
// a shared path means one deletes the artifact the other is still reading. dxc
// runs inside that directory and is given bare file names, because the `#line`
// directives of the text targets quote the path exactly as passed: an absolute
// one would stamp the process-unique directory into the artifact and make it
// differ per compile.
fn produce(
    job: &HlslJob<'_>,
    dxc: &Path,
    scratch: &Path,
    warnings: Warnings,
) -> Result<Compiled, String> {
    let src_name = job.file_name;
    let source = declarations::strip_engine_attributes(job.source)
        .map_err(|e| format!("hlsl: {src_name}: {e}"))?;
    std::fs::write(scratch.join(src_name), source.as_bytes())
        .map_err(|e| format!("hlsl: write {src_name}: {e}"))?;

    let stage = stage_of(job.source, job.entry)?;
    let profile = job.target.profile(stage);

    let emit = match job.target {
        HlslTarget::Dxil { .. } => dxc::Emit::Dxil,
        HlslTarget::Spirv => dxc::Emit::SpirvForVulkan,
        HlslTarget::SpirvWithVulkanLayout => dxc::Emit::SpirvWithVulkanLayout,
        HlslTarget::SpirvWithDxLayout => dxc::Emit::SpirvWithDxLayout,
        HlslTarget::Metallib | HlslTarget::Msl => dxc::Emit::SpirvForMsl,
    };
    let out_name = "artifact";
    let printed = dxc::compile(
        dxc,
        scratch,
        src_name,
        out_name,
        &dxc::command_args(stage, &profile, job.entry, emit, warnings),
    )?;
    let artifact = std::fs::read(scratch.join(out_name))
        .map_err(|e| format!("hlsl: read {} artifact: {e}", job.file_name))?;
    if artifact.is_empty() {
        return Err(format!(
            "hlsl: {} compiled to an empty artifact",
            job.file_name
        ));
    }

    let artifact = match job.target {
        HlslTarget::Dxil { .. }
        | HlslTarget::Spirv
        | HlslTarget::SpirvWithVulkanLayout
        | HlslTarget::SpirvWithDxLayout => artifact,
        HlslTarget::Msl | HlslTarget::Metallib => metal_leg(job, dxc, scratch, &artifact, stage)?,
    };
    let printed = printed.trim();
    Ok(Compiled {
        artifact,
        warnings: (!printed.is_empty()).then(|| printed.to_string()),
    })
}

// The SPIR-V dxc emitted, through spirv-cross to MSL and, for a library, the
// Metal toolchain.
#[cfg(feature = "spirv-cross")]
fn metal_leg(
    job: &HlslJob<'_>,
    dxc: &Path,
    scratch: &Path,
    spirv: &[u8],
    stage: Stage,
) -> Result<Vec<u8>, String> {
    let source = translate_to_msl(job, dxc, scratch, spirv, stage)?;
    if job.target == HlslTarget::Msl {
        return Ok(source.into_bytes());
    }
    metallib::build(scratch, &source)
}

#[cfg(not(feature = "spirv-cross"))]
fn metal_leg(
    job: &HlslJob<'_>,
    _dxc: &Path,
    _scratch: &Path,
    _spirv: &[u8],
    _stage: Stage,
) -> Result<Vec<u8>, String> {
    Err(format!(
        "hlsl: {}: a Metal target needs concinnity-shader's `spirv-cross` feature",
        job.file_name
    ))
}

// The Metal binding table is read off the *preprocessed* source, so a
// declaration behind an inactive `#if` never contributes a slot the compile
// cannot see. It is the job's own text, `cn::` attributes and all, rather than
// the stripped copy dxc compiled.
#[cfg(feature = "spirv-cross")]
fn translate_to_msl(
    job: &HlslJob<'_>,
    dxc: &Path,
    scratch: &Path,
    spirv: &[u8],
    stage: Stage,
) -> Result<String, String> {
    let text = preprocess(dxc, scratch, job.file_name, job.source)?;
    msl::translate(spirv, &text, stage)
        .map_err(|e| format!("{e} ({}: {})", job.file_name, job.entry))
}

/// The resource declarations `source` carries once the preprocessor has run,
/// which is the only reading of them that matches what a compile sees: a
/// declaration behind an inactive `#if` is not a declaration, and a register
/// spelled as a macro is not a register until it is expanded.
pub fn preprocessed_declarations(
    source: &str,
    file_name: &str,
    work_dir: &Path,
) -> Result<Vec<declarations::Declaration>, String> {
    with_scratch(work_dir, |dxc, scratch| {
        preprocess(dxc, scratch, file_name, source)
    })
    .map(|text| declarations::resource_declarations(&text))
}

// Write `source` under `scratch` as `file_name`, preprocess it and return the
// text.
fn preprocess(dxc: &Path, scratch: &Path, file_name: &str, source: &str) -> Result<String, String> {
    std::fs::write(scratch.join(file_name), source)
        .map_err(|e| format!("hlsl: write {file_name}: {e}"))?;
    let out_name = "preprocessed.hlsl";
    dxc::preprocess(dxc, scratch, file_name, out_name)?;
    std::fs::read_to_string(scratch.join(out_name))
        .map_err(|e| format!("hlsl: read {out_name}: {e}"))
}

/// The `[[id(n)]]` of the argument-buffer member named `member` in MSL text,
/// or `None` when no struct in `msl` declares one.
#[must_use]
pub fn msl_argument_id(msl: &str, member: &str) -> Option<u32> {
    msl.lines().find_map(|line| {
        let (decl, rest) = line.trim().split_once(" [[id(")?;
        let name = decl.rsplit(' ').next()?;
        (name == member).then(|| rest.split(')').next()?.parse().ok())?
    })
}

// SPIR-V is a stream of 32-bit words, little-endian on every target the engine
// builds for.
#[cfg(feature = "spirv-cross")]
fn words(spirv: &[u8]) -> Result<Vec<u32>, String> {
    if !spirv.len().is_multiple_of(4) || spirv.is_empty() {
        return Err(format!(
            "hlsl: {} bytes is not a SPIR-V module",
            spirv.len()
        ));
    }
    Ok(spirv
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect())
}

// A scratch directory name no concurrent compile can share: the process id
// pairs with a monotonic counter, so neither two threads nor two engine
// processes (a `cn build` alongside a running editor) collide.
fn scratch_name() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{}-{seq}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "spirv-cross")]
    #[test]
    fn a_byte_stream_that_is_not_whole_words_is_not_a_module() {
        assert!(words(&[0, 1, 2]).is_err());
        assert!(words(&[]).is_err());
        assert_eq!(words(&[1, 0, 0, 0]), Ok(vec![1]));
    }

    #[test]
    fn an_argument_buffer_member_reads_back_its_id() {
        let msl = "struct spvDescriptorSetBuffer1\n{\n    \
                   array<texture2d<float>, 8> tex_pool [[id(0)]];\n    \
                   sampler tex_sampler [[id(12)]];\n};\n";
        assert_eq!(msl_argument_id(msl, "tex_pool"), Some(0));
        assert_eq!(msl_argument_id(msl, "tex_sampler"), Some(12));
        assert_eq!(msl_argument_id(msl, "tex"), None);
        assert_eq!(msl_argument_id("float4 x [[id(q)]];", "x"), None);
    }

    // The cache folds this in, so it has to be a real value rather than the
    // zero a failed build script would leave.
    #[test]
    fn the_source_hash_is_derived() {
        assert_ne!(SOURCE_HASH, 0);
    }

    #[test]
    fn the_compiler_id_names_the_toolchain() {
        assert!(compiler_id().starts_with("dxc"));
    }

    // Microsoft's Windows and Linux archives of v1.9.2607 and a shallow source
    // build of the same tag count different histories, and must key alike.
    #[test]
    fn builds_of_one_commit_share_a_release_id() {
        let lines = [
            "dxcompiler.dll: 1.9(5402-0d3ee6b5)(1.9.0.5402) - 1.9.0.5402 (0d3ee6b55-dirty)",
            "libdxcompiler.so: 1.9(1-0d3ee6b5)(1.9.0.1)",
            "libdxcompiler.dylib: 1.9(1-0d3ee6b5)(1.9.0.1)",
        ];
        for line in lines {
            assert_eq!(release_id(line).as_deref(), Some("1.9 0d3ee6b5"), "{line}");
        }
    }

    #[test]
    fn another_commit_is_another_release_id() {
        assert_eq!(
            release_id("libdxcompiler.dylib: 1.9(5399-a107ba61)(1.9.0.5399)").as_deref(),
            Some("1.9 a107ba61")
        );
    }

    // A distribution's build names no commit, so the whole line is its id.
    #[test]
    fn a_build_naming_no_commit_has_no_release_id() {
        assert_eq!(release_id("libdxcompiler.so: 1.8(dev)"), None);
        assert_eq!(release_id("dxc"), None);
        assert_eq!(release_id("libdxcompiler.so: 1.9(1-nothex)(1.9.0.1)"), None);
    }

    // Only a DXIL program that asks for 6.5 leaves the base model; the stage
    // half of the profile is the entry's own.
    #[test]
    fn the_profile_follows_the_stage_and_only_a_6_5_dxil_leaves_6_0() {
        let dxil = HlslTarget::Dxil {
            shader_model_6_5: false,
        };
        let dxil_6_5 = HlslTarget::Dxil {
            shader_model_6_5: true,
        };
        assert_eq!(dxil.profile(Stage::Vertex), "vs_6_0");
        assert_eq!(dxil.profile(Stage::Compute), "cs_6_0");
        assert_eq!(dxil_6_5.profile(Stage::Pixel), "ps_6_5");
        assert_eq!(dxil_6_5.profile(Stage::Compute), "cs_6_5");
        for target in [
            HlslTarget::Spirv,
            HlslTarget::Msl,
            HlslTarget::SpirvWithDxLayout,
        ] {
            assert_eq!(target.profile(Stage::Pixel), "ps_6_0", "{target:?}");
        }
    }

    // Each backend loads the form its renderer consumes without a compiler of
    // its own: MSL text on Metal, a container on the two that take bytecode.
    #[test]
    fn each_platform_cooks_what_its_renderer_loads() {
        assert_eq!(HlslTarget::cooked(Platform::Metal), HlslTarget::Msl);
        assert_eq!(
            HlslTarget::cooked(Platform::DirectX),
            HlslTarget::Dxil {
                shader_model_6_5: false
            }
        );
        assert_eq!(HlslTarget::cooked(Platform::Vulkan), HlslTarget::Spirv);
    }

    // A cache keys artifacts by these names, so two targets must never share one.
    #[test]
    fn every_target_has_a_distinct_name() {
        let mut names: Vec<&str> = [
            HlslTarget::Spirv,
            HlslTarget::Dxil {
                shader_model_6_5: false,
            },
            HlslTarget::Dxil {
                shader_model_6_5: true,
            },
            HlslTarget::Metallib,
            HlslTarget::Msl,
            HlslTarget::SpirvWithVulkanLayout,
            HlslTarget::SpirvWithDxLayout,
        ]
        .into_iter()
        .map(HlslTarget::name)
        .collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    // The two answers are one fact, so a host cannot report both a compiler and
    // a reason it has none.
    #[test]
    fn availability_and_the_reason_are_exclusive() {
        assert_eq!(dxc_available(), unavailable_reason().is_none());
    }
}
