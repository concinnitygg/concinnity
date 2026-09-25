//! The compile the cook hands an assembled program to.
//!
//! A world's authored text does not compile on its own: it is spliced into an
//! engine template and the two become one translation unit, which dxc
//! compiles.
//!
//! What is emitted is what that backend's renderer loads directly
//! ([`HlslTarget::cooked`]): SPIR-V for Vulkan, a signed DXIL container for
//! D3D12, and MSL text for Metal, whose `newLibraryWithSource` is an
//! always-present OS API. Each artifact carries the digest of the source it
//! came from, so a renderer whose template has moved compiles instead of
//! loading something stale.
//!
//! The build script makes the same call in
//! `concinnity_toolchain::shader_artifacts` for the engine's own programs, and
//! the two must agree: a renderer takes a stored artifact whenever its source
//! digest matches, so the same text compiled under different flags would be a
//! different shader under the same name.
//!
//! [`HlslTarget::cooked`]: concinnity_shader::HlslTarget::cooked

use std::path::Path;

use concinnity_core::components::compiled_programs::CompiledProgram;
use concinnity_core::platform::Platform;
use concinnity_core::render::shader_source;
use concinnity_shader::diagnostics;

mod failure;

pub use concinnity_shader::diagnostics::{Diagnostic, Severity};
pub use failure::{CompileFailure, EntryFailure, ProgramError};

/// One entry point to compile out of an assembled source.
pub(crate) struct Job<'a> {
    /// The template's file name, which dxc diagnostics carry for every line
    /// outside a `#line`-fenced splice.
    pub file: &'a str,
    /// The entry point.
    pub entry: &'a str,
    /// The assembled source text.
    pub source: &'a str,
}

/// Fails naming `owner` (the asset, as a message names it) when this host has
/// no compiler.
///
/// Reaching here means the payload cache had nothing for the asset, so it has
/// to be compiled. A world whose assets are already cooked never gets this far:
/// the cache answers first and the compiler is not part of its key.
/// `have_compiler` is the host's answer, supplied so the no-compiler path is
/// reachable without uninstalling one.
pub(crate) fn require_compiler(owner: &str, have_compiler: bool) -> std::io::Result<()> {
    if have_compiler {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "{owner}: {}",
            concinnity_shader::unavailable_reason()
                .unwrap_or("no shader compiler found and this asset has no compiled payload")
        ),
    ))
}

/// What [`compile_all`] produced: the programs in job order, and every
/// warning the compiler printed on the way, each once.
#[derive(Debug)]
pub(crate) struct Compiled {
    pub programs: Vec<CompiledProgram>,
    pub warnings: Vec<Diagnostic>,
}

/// Compile every job for `platform` side by side, in a scratch directory named
/// `scratch`.
///
/// A failure names `owner` and every entry that failed, carries dxc's
/// diagnostics (each once, however many entries reported it) and its raw
/// output, and ends with whatever `hint` adds for that output. Warnings are
/// logged against `owner` and returned beside the programs.
pub(crate) fn compile_all(
    owner: &str,
    scratch: &str,
    jobs: &[Job<'_>],
    platform: Platform,
    hint: impl Fn(&str) -> &'static str,
) -> Result<Compiled, ProgramError> {
    use rayon::prelude::*;

    let work = concinnity_host::scratch::Scratch::dir(scratch).map_err(|source| {
        ProgramError::Scratch {
            owner: owner.to_string(),
            source,
        }
    })?;
    let results: Vec<_> = jobs
        .par_iter()
        .map(|job| compile(job, platform, work.path()))
        .collect();
    gather(owner, jobs, results, hint)
}

// Fold each job's result into the programs and their warnings, or into one
// failure naming every entry that failed.
fn gather(
    owner: &str,
    jobs: &[Job<'_>],
    results: Vec<Result<concinnity_shader::Compiled, String>>,
    hint: impl Fn(&str) -> &'static str,
) -> Result<Compiled, ProgramError> {
    let mut programs = Vec::with_capacity(jobs.len());
    let mut warnings = Vec::new();
    let mut failures = Vec::new();
    for (job, result) in jobs.iter().zip(results) {
        match result {
            Ok(compiled) => {
                if let Some(printed) = compiled.warnings {
                    let parsed = diagnostics::parse(&printed);
                    if parsed.is_empty() {
                        tracing::warn!("{owner}: compiling '{}':\n{printed}", job.entry);
                    }
                    warnings.extend(parsed);
                }
                programs.push(CompiledProgram {
                    entry: job.entry.to_string(),
                    source_digest: shader_source::source_digest(job.source),
                    artifact: compiled.artifact,
                });
            }
            Err(output) => failures.push(EntryFailure {
                entry: job.entry.to_string(),
                output,
            }),
        }
    }
    if !failures.is_empty() {
        return Err(CompileFailure::new(owner, failures, hint).into());
    }
    let warnings = diagnostics::dedup(warnings);
    for warning in &warnings {
        tracing::warn!("{owner}: {warning}");
    }
    Ok(Compiled { programs, warnings })
}

// One entry point of a job's source, emitted for `platform`. The text is
// authored, so a dxc warning comes back beside the artifact rather than
// failing the build.
fn compile(
    job: &Job<'_>,
    platform: Platform,
    work_dir: &Path,
) -> Result<concinnity_shader::Compiled, String> {
    concinnity_shader::compile_with_warnings(
        &concinnity_shader::HlslJob {
            source: job.source,
            file_name: job.file,
            entry: job.entry,
            target: concinnity_shader::HlslTarget::cooked(platform),
        },
        work_dir,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIXEL: &str = "[shader(\"pixel\")]\nfloat4 main_ps() : SV_Target { return 1.0; }\n";

    fn job(source: &str) -> Job<'_> {
        Job {
            file: "x.hlsl",
            entry: "main_ps",
            source,
        }
    }

    // Every host cooks every target, so a world cooked on Windows or Linux
    // still carries the MSL a Metal player loads without a compiler of its own.
    #[test]
    fn an_msl_artifact_compiles_on_every_host() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let msl = compile(&job(PIXEL), Platform::Metal, dir.path()).expect("MSL off any host");
        let text = String::from_utf8(msl.artifact).expect("MSL is text");
        assert!(text.contains("fragment "), "got: {text}");
        assert!(text.contains("main_ps"), "got: {text}");
    }

    // A host with no compiler is an error naming the asset, and one with a
    // compiler passes.
    #[test]
    fn a_missing_compiler_fails_naming_the_owner() {
        let err = require_compiler("Shader 'wall'", false).expect_err("no compiler is an error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let message = err.to_string();
        assert!(message.starts_with("Shader 'wall': "), "{message}");
        assert!(
            message.contains("compiled payload") || message.contains("dxc"),
            "{message}"
        );
        assert!(require_compiler("Shader 'wall'", true).is_ok());
    }

    // Each program carries the digest of the text it came from, in job order,
    // and a failure names every entry that failed, with the caller's hint.
    #[test]
    fn programs_come_back_in_job_order_and_a_failure_names_every_failed_entry() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let other = PIXEL.replace("1.0", "0.5");
        let jobs = [job(PIXEL), job(&other)];
        let compiled = compile_all(
            "Shader 'x'",
            "program-order",
            &jobs,
            Platform::Vulkan,
            |_| "",
        )
        .unwrap();
        let digests: Vec<u64> = compiled.programs.iter().map(|p| p.source_digest).collect();
        assert_eq!(
            digests,
            [
                shader_source::source_digest(PIXEL),
                shader_source::source_digest(&other)
            ]
        );
        assert!(compiled.warnings.is_empty());

        let broken = [
            job(PIXEL),
            Job {
                entry: "first",
                ..job("float4 first(")
            },
            Job {
                entry: "second",
                ..job("float4 second(")
            },
        ];
        let err = compile_all(
            "Shader 'x'",
            "program-fail",
            &broken,
            Platform::Vulkan,
            |_| "\nhint",
        )
        .unwrap_err();
        let ProgramError::Compile(failed) = &err else {
            panic!("a compile failure: {err}");
        };
        let entries: Vec<&str> = failed.failures.iter().map(|f| f.entry.as_str()).collect();
        assert_eq!(entries, ["first", "second"]);
        assert!(failed.errors().all(|d| d.path == "x.hlsl"), "{err}");
        let message = err.to_string();
        assert!(
            message.starts_with("Shader 'x': compiling 'first', 'second':"),
            "{message}"
        );
        assert!(message.ends_with("\nhint"), "{message}");
    }

    // A warning every entry printed comes back once, and the programs keep job
    // order.
    #[test]
    fn a_warning_from_every_entry_is_returned_once() {
        let warned = |bytes: &[u8]| {
            Ok(concinnity_shader::Compiled {
                artifact: bytes.to_vec(),
                warnings: Some(
                    "user.hlsl:3:7: warning: implicit truncation\n  x = y;\n      ^".to_string(),
                ),
            })
        };
        let jobs = [
            Job {
                entry: "a",
                ..job(PIXEL)
            },
            Job {
                entry: "b",
                ..job(PIXEL)
            },
        ];
        let compiled = gather(
            "Shader 'x'",
            &jobs,
            vec![warned(b"A"), warned(b"B")],
            |_| "",
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let artifacts: Vec<&[u8]> = compiled.programs.iter().map(|p| &p.artifact[..]).collect();
        assert_eq!(artifacts, [b"A", b"B"]);
        assert_eq!(compiled.warnings.len(), 1);
        assert_eq!(
            (
                compiled.warnings[0].path.as_str(),
                compiled.warnings[0].line
            ),
            ("user.hlsl", 3)
        );
    }
}
