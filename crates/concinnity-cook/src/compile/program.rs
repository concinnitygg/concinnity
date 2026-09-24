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

/// One entry point to compile out of an assembled source.
pub(crate) struct Job<'a> {
    /// The template's file name, which dxc diagnostics carry.
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

/// Compile every job for `platform` side by side, in a scratch directory named
/// `scratch`, and return the programs in job order.
///
/// A failure names `owner` and the entry, carries dxc's own diagnostic, and
/// ends with whatever `hint` adds for that diagnostic. When several entries
/// fail, the first in job order is reported, so every run reports the same one.
pub(crate) fn compile_all(
    owner: &str,
    scratch: &str,
    jobs: &[Job<'_>],
    platform: Platform,
    hint: impl Fn(&str) -> &'static str + Sync,
) -> std::io::Result<Vec<CompiledProgram>> {
    use rayon::prelude::*;

    let work = concinnity_host::scratch::Scratch::dir(scratch)?;
    let compiled: Vec<_> = jobs
        .par_iter()
        .map(|job| {
            let artifact = compile(owner, job, platform, work.path()).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{owner}: compiling '{}': {e}{}", job.entry, hint(&e)),
                )
            })?;
            Ok(CompiledProgram {
                entry: job.entry.to_string(),
                source_digest: shader_source::source_digest(job.source),
                artifact,
            })
        })
        .collect();
    compiled.into_iter().collect()
}

// One entry point of a job's source, emitted for `platform`. The text is
// authored, so a dxc warning is logged against `owner` rather than failing the
// build.
fn compile(
    owner: &str,
    job: &Job<'_>,
    platform: Platform,
    work_dir: &Path,
) -> Result<Vec<u8>, String> {
    let compiled = concinnity_shader::compile_with_warnings(
        &concinnity_shader::HlslJob {
            source: job.source,
            file_name: job.file,
            entry: job.entry,
            target: concinnity_shader::HlslTarget::cooked(platform),
        },
        work_dir,
    )?;
    if let Some(warnings) = compiled.warnings {
        tracing::warn!("{owner}: compiling '{}':\n{warnings}", job.entry);
    }
    Ok(compiled.artifact)
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
        let msl =
            compile("test", &job(PIXEL), Platform::Metal, dir.path()).expect("MSL off any host");
        let text = String::from_utf8(msl).expect("MSL is text");
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
    // and the first failing entry is the one reported with the caller's hint.
    #[test]
    fn programs_come_back_in_job_order_and_the_first_failure_wins() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let other = PIXEL.replace("1.0", "0.5");
        let jobs = [job(PIXEL), job(&other)];
        let programs = compile_all(
            "Shader 'x'",
            "program-order",
            &jobs,
            Platform::Vulkan,
            |_| "",
        )
        .unwrap();
        let digests: Vec<u64> = programs.iter().map(|p| p.source_digest).collect();
        assert_eq!(
            digests,
            [
                shader_source::source_digest(PIXEL),
                shader_source::source_digest(&other)
            ]
        );

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
        let message = err.to_string();
        assert!(
            message.starts_with("Shader 'x': compiling 'first': "),
            "{message}"
        );
        assert!(message.ends_with("\nhint"), "{message}");
    }
}
