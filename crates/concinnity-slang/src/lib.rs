//! The `slangc` invocation, shared by build scripts and the renderer.
//!
//! The engine's single-source shaders (`crates/concinnity-device/src/shaders/
//! *.slang`) compile through the `slangc` binary: at build time to Metal
//! metallibs (the device build script), and at renderer init to SPIR-V for the
//! Vulkan backend (plus Metal hot-reload rebuilds). Both call sites assemble the
//! full source text first (defines injected as `#define` lines), so a compile is
//! a pure function of that text, the entry list, and the target -- which is what
//! lets the renderer's content-addressed shader cache key it.
//!
//! Being needed on both sides is why this is its own crate rather than a module
//! of `concinnity-toolchain`: that crate is build-script support, consumed only
//! under `[build-dependencies]` and never linked into a shipped binary, and it
//! stays that way. This one sits below it, holds no policy, and depends on
//! nothing but std -- a compile here is a subprocess, not a linked compiler.
//! The single invocation is the point: the build script and the runtime must
//! produce byte-identical artifacts or the content-addressed cache serves one
//! path's bytes to the other, so the flag list exists exactly once.
//!
//! slangc resolves from PATH first, then `$VULKAN_SDK/bin`, taking the first
//! candidate that meets `MIN_SLANGC`. A host without it degrades the same way a
//! missing Metal toolchain does: the build script emits a stub lookup and the
//! renderer falls back to compiling at startup, which then needs slangc at
//! runtime and reports a clear error when it is absent or too old.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// What slangc should emit. `Metal` (MSL text) and `Hlsl` exist for build-time
/// binding assertions and diagnostics; shipped artifacts are `Spirv`,
/// `Metallib`, and `Dxil`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlangTarget {
    /// SPIR-V for Vulkan.
    Spirv,
    /// A Metal library for the Metal backend.
    Metallib,
    /// MSL source text, for build-time assertions and diagnostics.
    Metal,
    /// A signed DXIL container for D3D12. The shader-model profile rides the
    /// variant because DXIL has no usable default: it sets the container's
    /// feature floor (SM 6.0 is what `NonUniformResourceIndex` needs) and a
    /// stage-correct profile is not inferable from the target alone.
    Dxil(&'static str),
    /// HLSL source text for the named profile, for build-time assertions.
    Hlsl(&'static str),
}

impl SlangTarget {
    fn flag(self) -> &'static str {
        match self {
            SlangTarget::Spirv => "spirv",
            SlangTarget::Metallib => "metallib",
            SlangTarget::Metal => "metal",
            SlangTarget::Dxil(_) => "dxil",
            SlangTarget::Hlsl(_) => "hlsl",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            SlangTarget::Spirv => "spv",
            SlangTarget::Metallib => "metallib",
            SlangTarget::Metal => "metal",
            SlangTarget::Dxil(_) => "dxil",
            SlangTarget::Hlsl(_) => "hlsl",
        }
    }

    // The shader-model profile this target compiles against, if it takes one.
    fn profile(self) -> Option<&'static str> {
        match self {
            SlangTarget::Dxil(p) | SlangTarget::Hlsl(p) => Some(p),
            _ => None,
        }
    }
}

/// One slangc compile: assembled source text in, artifact bytes out.
pub struct SlangJob<'a> {
    /// Fully assembled source (defines already injected as `#define` lines).
    pub source: &'a str,
    /// File name the source is written under in `work_dir`; also the name
    /// slangc diagnostics and `#line` directives carry.
    pub file_name: &'a str,
    /// Entry points to compile from the source.
    pub entries: &'a [&'a str],
    /// What slangc should emit.
    pub target: SlangTarget,
}

// The oldest slangc release the engine's shader output is validated against.
// 2025.x emits SPIR-V declaring `StorageImageMultisample` for an ordinary
// multisampled sampled texture, a capability the shaders never use and that
// Vulkan rejects unless the matching device feature is enabled.
const MIN_SLANGC: (u32, u32) = (2026, 1);

// A usable compiler: the binary to invoke and the release it reports.
struct Slangc {
    path: PathBuf,
    version: String,
}

// Resolved once per process. The error side carries the diagnostic, so a host
// with only an unusable candidate reports why rather than "not found".
fn resolved() -> &'static Result<Slangc, String> {
    static SLANGC: OnceLock<Result<Slangc, String>> = OnceLock::new();
    SLANGC.get_or_init(probe_slangc)
}

/// The slangc to invoke, or `None` when no candidate is usable.
pub fn slangc_path() -> Option<&'static Path> {
    resolved().as_ref().ok().map(|s| s.path.as_path())
}

/// Identifies the compiler for the renderer's content-addressed shader cache.
/// Two slangc releases can emit different bytes for identical source, so an
/// artifact keyed without the version outlives the toolchain that produced it
/// and gets served to a later one.
pub fn compiler_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| match resolved() {
        Ok(s) => format!("slang {}", s.version),
        Err(_) => "slang".to_string(),
    })
}

// Locate slangc: PATH first, then `$VULKAN_SDK/bin`, taking the first that
// meets `MIN_SLANGC`. A candidate below the floor is kept only to name it in
// the error, so an old slangc earlier on PATH does not shadow a newer SDK.
fn probe_slangc() -> Result<Slangc, String> {
    let mut candidates = vec![PathBuf::from("slangc")];
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let exe = if cfg!(windows) {
            "slangc.exe"
        } else {
            "slangc"
        };
        candidates.push(Path::new(&sdk).join("bin").join(exe));
    }
    let mut too_old: Option<Slangc> = None;
    for path in candidates {
        let Some(version) = query_version(&path) else {
            continue;
        };
        let found = Slangc { path, version };
        if parse_version(&found.version).is_some_and(|v| v >= MIN_SLANGC) {
            return Ok(found);
        }
        too_old.get_or_insert(found);
    }
    Err(match too_old {
        Some(old) => too_old_message(&old),
        None => missing_slangc_message(),
    })
}

// slangc prints its release on stderr and exits 0.
fn query_version(exe: &Path) -> Option<String> {
    let out = Command::new(exe).arg("-version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stderr);
    let line = text.lines().next()?.trim().to_string();
    (!line.is_empty()).then_some(line)
}

// Leading `<year>.<minor>` of a release string like `2026.1-52-gc8ddf20bb`.
fn parse_version(version: &str) -> Option<(u32, u32)> {
    let head = version.split(['-', ' ']).next()?;
    let mut parts = head.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

/// Compile `job` under `work_dir`, returning the artifact bytes.
pub fn compile(job: &SlangJob<'_>, work_dir: &Path) -> Result<Vec<u8>, String> {
    let produced = run(job, work_dir, false)?;
    if produced.artifact.is_empty() {
        return Err(format!(
            "slang: {} compiled to an empty artifact",
            job.file_name
        ));
    }
    Ok(produced.artifact)
}

/// The layout slangc gives `job`'s declarations, as its `-reflection-json`
/// emits it. The invocation is the one `compile` uses, so the offsets reported
/// are the ones the shipped artifact carries -- and they are per target: MSL
/// sizes a constant-buffer `float3` at 16 bytes where SPIR-V and DXIL pack a
/// scalar after it, and SPIR-V rounds an array element stride up to 16 where
/// neither of the others does. A caller comparing a `#[repr(C)]` mirror has to
/// ask each target separately.
pub fn reflect(job: &SlangJob<'_>, work_dir: &Path) -> Result<String, String> {
    let produced = run(job, work_dir, true)?;
    produced
        .reflection
        .ok_or_else(|| format!("slang: {} emitted no reflection JSON", job.file_name))
}

// What one slangc run produced.
struct Produced {
    artifact: Vec<u8>,
    reflection: Option<String>,
}

// One slangc run under `work_dir` (created if needed; source and outputs are
// cleaned up afterwards). `-matrix-layout-column-major` is mandatory because
// Slang defaults to row-major and every CPU-uploaded matrix would read
// transposed without it.
//
// Each invocation gets its own subdirectory: two compiles of the same
// `file_name` can run at once (one shared source serves several programs -- the
// fullscreen vertex is compiled by every post pass), and a shared path means
// one deletes the artifact the other is still reading. The scratch path reaches
// only slangc's diagnostics and the `#line` directives of the text targets;
// neither the metallib nor the SPIR-V embeds it, so per-invocation naming costs
// no artifact determinism.
fn run(job: &SlangJob<'_>, work_dir: &Path, reflection: bool) -> Result<Produced, String> {
    let slangc = match resolved() {
        Ok(found) => found.path.as_path(),
        Err(message) => return Err(message.clone()),
    };
    let scratch = work_dir.join(scratch_name());
    std::fs::create_dir_all(&scratch)
        .map_err(|e| format!("slang: create {}: {e}", scratch.display()))?;
    let src_path = scratch.join(job.file_name);
    let out_path = src_path.with_extension(job.target.extension());
    let refl_path = src_path.with_extension("reflection.json");
    std::fs::write(&src_path, job.source)
        .map_err(|e| format!("slang: write {}: {e}", src_path.display()))?;

    let mut cmd = Command::new(slangc);
    cmd.arg(&src_path)
        .args(command_args(job))
        .arg("-o")
        .arg(&out_path);
    if reflection {
        cmd.arg("-reflection-json").arg(&refl_path);
    }
    let output = cmd
        .output()
        .map_err(|e| format!("slang: failed to launch slangc: {e}"))?;
    let result = if output.status.success() {
        read_produced(&out_path, reflection.then_some(refl_path.as_path()))
    } else {
        Err(format!(
            "slang: {} failed:\n{}{}",
            job.file_name,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
    };
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_file(&refl_path);
    let _ = std::fs::remove_dir(&scratch);
    result
}

// The files a successful run left behind.
fn read_produced(out_path: &Path, refl_path: Option<&Path>) -> Result<Produced, String> {
    let artifact =
        std::fs::read(out_path).map_err(|e| format!("slang: read {}: {e}", out_path.display()))?;
    let reflection = match refl_path {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|e| format!("slang: read {}: {e}", path.display()))?,
        ),
        None => None,
    };
    Ok(Produced {
        artifact,
        reflection,
    })
}

// A scratch directory name no concurrent compile can share: the process id
// pairs with a monotonic counter, so neither two threads nor two engine
// processes (a `cn build` alongside a running editor) collide.
fn scratch_name() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{}-{seq}", std::process::id())
}

// The argument list after the source path and before `-o`. Split out so the
// invocation shape is unit-testable without slangc installed.
fn command_args(job: &SlangJob<'_>) -> Vec<String> {
    let mut args = vec!["-matrix-layout-column-major".to_string()];
    for entry in job.entries {
        args.push("-entry".to_string());
        args.push((*entry).to_string());
    }
    args.push("-target".to_string());
    args.push(job.target.flag().to_string());
    if let Some(profile) = job.target.profile() {
        args.push("-profile".to_string());
        args.push(profile.to_string());
    }
    args
}

/// Prepend `#define` lines for each `(name, value)` pair. The defines become
/// part of the source text on purpose: the renderer's shader cache keys on the
/// assembled source, so two pool sizes can never share an artifact.
pub fn inject_defines(source: &str, defines: &[(&str, &str)]) -> String {
    if defines.is_empty() {
        return source.to_string();
    }
    let mut out = String::with_capacity(source.len() + defines.len() * 32);
    for (name, value) in defines {
        out.push_str("#define ");
        out.push_str(name);
        out.push(' ');
        out.push_str(value);
        out.push('\n');
    }
    out.push_str(source);
    out
}

fn missing_slangc_message() -> String {
    "slangc not found: the engine's single-source shaders need the Slang \
     compiler. Install the Vulkan SDK (https://vulkan.lunarg.com) or a \
     standalone Slang release (https://github.com/shader-slang/slang/releases) \
     and make sure `slangc` is on PATH or under $VULKAN_SDK/bin."
        .to_string()
}

fn too_old_message(found: &Slangc) -> String {
    // The PATH candidate is invoked by bare name, so there is no path to name.
    let origin = if found.path == Path::new("slangc") {
        "on PATH".to_string()
    } else {
        format!("at {}", found.path.display())
    };
    format!(
        "slangc {} ({origin}) is older than {}.{}, the oldest release the \
         engine's shaders are validated against: earlier ones emit SPIR-V \
         declaring capabilities the shaders never use. Install a newer Vulkan \
         SDK (https://vulkan.lunarg.com) or Slang release \
         (https://github.com/shader-slang/slang/releases), or put one already \
         installed ahead of it on PATH.",
        found.version, MIN_SLANGC.0, MIN_SLANGC.1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Release strings carry a build suffix that must not defeat the compare.
    #[test]
    fn a_release_string_parses_to_year_and_minor() {
        assert_eq!(parse_version("2026.1-52-gc8ddf20bb"), Some((2026, 1)));
        assert_eq!(parse_version("2025.17.2"), Some((2025, 17)));
        assert_eq!(parse_version("2026"), Some((2026, 0)));
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("unknown"), None);
    }

    // The floor exists to reject the releases that miscompile the engine's
    // shaders; 2025.17.2 is the one that was caught doing it.
    #[test]
    fn the_floor_rejects_a_known_bad_release() {
        let meets = |v: &str| parse_version(v).is_some_and(|v| v >= MIN_SLANGC);
        assert!(!meets("2025.17.2"));
        assert!(meets("2026.1-52-gc8ddf20bb"));
        assert!(meets("2026.2"));
        assert!(meets("2027.0"));
    }

    #[test]
    fn an_unresolved_compiler_still_names_the_toolchain() {
        assert!(compiler_id().starts_with("slang"));
    }

    #[test]
    fn command_args_carry_layout_entries_and_target() {
        let job = SlangJob {
            source: "",
            file_name: "x.slang",
            entries: &["vmain", "fmain"],
            target: SlangTarget::Metallib,
        };
        assert_eq!(
            command_args(&job),
            [
                "-matrix-layout-column-major",
                "-entry",
                "vmain",
                "-entry",
                "fmain",
                "-target",
                "metallib",
            ]
        );
    }

    #[test]
    fn every_target_has_a_distinct_flag_and_extension() {
        let targets = [
            SlangTarget::Spirv,
            SlangTarget::Metallib,
            SlangTarget::Metal,
            SlangTarget::Dxil("ps_6_0"),
            SlangTarget::Hlsl("ps_6_0"),
        ];
        for t in targets {
            assert!(!t.flag().is_empty());
            assert!(!t.extension().is_empty());
        }
        assert_eq!(SlangTarget::Spirv.flag(), "spirv");
        assert_eq!(SlangTarget::Metallib.extension(), "metallib");
        assert_eq!(SlangTarget::Dxil("cs_6_0").flag(), "dxil");
    }

    // Only the profile-carrying targets emit `-profile`, and it lands after
    // `-target` so the flag applies to it.
    #[test]
    fn a_profile_target_passes_its_shader_model() {
        let job = SlangJob {
            source: "",
            file_name: "x.slang",
            entries: &["k"],
            target: SlangTarget::Dxil("cs_6_0"),
        };
        assert_eq!(
            command_args(&job),
            [
                "-matrix-layout-column-major",
                "-entry",
                "k",
                "-target",
                "dxil",
                "-profile",
                "cs_6_0",
            ]
        );
        assert!(
            !command_args(&SlangJob {
                target: SlangTarget::Spirv,
                ..job
            })
            .contains(&"-profile".to_string())
        );
    }

    #[test]
    fn defines_prepend_in_order_and_empty_is_identity() {
        let out = inject_defines("body\n", &[("POOL_SIZE", "8"), ("METAL_ABI", "1")]);
        assert_eq!(out, "#define POOL_SIZE 8\n#define METAL_ABI 1\nbody\n");
        assert_eq!(inject_defines("body\n", &[]), "body\n");
    }

    // Round-trip through the real compiler when it is installed; skipped
    // silently otherwise so CI hosts without slangc still pass.
    #[test]
    fn compiles_a_trivial_kernel_when_slangc_is_installed() {
        if slangc_path().is_none() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("cn_slang_test_{}", std::process::id()));
        let job = SlangJob {
            source: "RWStructuredBuffer<float> o;\n[shader(\"compute\")] [numthreads(1,1,1)]\n\
                     void k(uint3 t : SV_DispatchThreadID) { o[t.x] = 1.0; }\n",
            file_name: "trivial.slang",
            entries: &["k"],
            target: SlangTarget::Spirv,
        };
        let bytes = compile(&job, &dir).expect("trivial slang compile");
        // SPIR-V magic.
        assert_eq!(&bytes[0..4], &0x0723_0203u32.to_le_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Reflection is the layout oracle the shader-struct checks read, so it has
    // to carry names, offsets and sizes -- and report them per target, since
    // MSL sizes a constant-buffer `float3` at 16 bytes where SPIR-V packs a
    // scalar after it.
    #[test]
    fn reflection_reports_constant_buffer_offsets_per_target() {
        if slangc_path().is_none() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("cn_slang_refl_{}", std::process::id()));
        let source = "struct Hazard { float3 a; float b; };\n\
                      ConstantBuffer<Hazard> h;\n\
                      RWStructuredBuffer<float> o;\n\
                      [shader(\"compute\")] [numthreads(1,1,1)]\n\
                      void k(uint3 t : SV_DispatchThreadID) { o[t.x] = h.a.x + h.b; }\n";
        let for_target = |target| {
            reflect(
                &SlangJob {
                    source,
                    file_name: "hazard.slang",
                    entries: &["k"],
                    target,
                },
                &dir,
            )
            .expect("reflection compile")
        };
        let msl = for_target(SlangTarget::Metal);
        let spirv = for_target(SlangTarget::Spirv);
        assert!(msl.contains("\"name\": \"Hazard\""), "{msl}");
        // A float3 followed by a scalar is the divergence the engine's packed
        // types and pad fields exist to avoid: 16 + 4 on Metal, 12 + 4 elsewhere.
        assert!(msl.contains("\"offset\": 16, \"size\": 4"), "{msl}");
        assert!(spirv.contains("\"offset\": 12, \"size\": 4"), "{spirv}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_compile_error_reports_the_diagnostic() {
        if slangc_path().is_none() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("cn_slang_err_{}", std::process::id()));
        let job = SlangJob {
            source: "void broken( {",
            file_name: "broken.slang",
            entries: &["k"],
            target: SlangTarget::Spirv,
        };
        let err = compile(&job, &dir).expect_err("broken source must fail");
        assert!(
            err.contains("broken.slang"),
            "diagnostic names the file: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
