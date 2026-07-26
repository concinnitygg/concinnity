// asset_impls/shader.rs

use crate::asset::BuildCtx;
use concinnity_core::assets::shader::platform_key;
use concinnity_core::assets::{Shader, ShaderKind, ShaderPayload};
use concinnity_world::source_args::{declares_only_builtin_sources, resolve_source_from_args};

// Resolve a raw per-platform source string to the on-disk path the build will
// read. A bare filename is looked up recursively under `.concinnity/assets/`
// first, then under `<artifacts_dir>` when set, then directly under
// `.concinnity/assets/<raw>`. A path with a directory component is used
// verbatim. Mirrors the resolution `compile_payload` applies; built-in shaders
// short-circuit upstream and never reach this.
pub fn resolve_source_path_for(raw: &str, ctx: &BuildCtx<'_>) -> String {
    let p = std::path::Path::new(raw);
    if p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true) {
        if let Some(path) = concinnity_core::paths::find_in_assets(raw) {
            return path;
        }
        if let Some(dir) = ctx.artifacts_dir {
            let artifact_path = format!("{dir}/{raw}");
            if std::path::Path::new(&artifact_path).exists() {
                return artifact_path;
            }
        }
        return concinnity_core::paths::assets_dir()
            .join(raw)
            .to_string_lossy()
            .into_owned();
    }
    raw.to_string()
}

// The stage slots a Shader declares, keyed by their args field.
const STAGES: &[(&str, ShaderKind)] = &[
    ("vertex", ShaderKind::Vertex),
    ("fragment", ShaderKind::Fragment),
    ("vertex_instanced", ShaderKind::VertexInstanced),
];

// Compile one declared stage to backend bytecode. `None` when the stage
// resolves no source for this platform (the Vulkan inline-GLSL carve-out).
fn compile_stage(
    stage_args: &serde_json::Value,
    kind: ShaderKind,
    ctx: &BuildCtx<'_>,
) -> std::io::Result<Option<Vec<u8>>> {
    let resolved = resolve_source_from_args(stage_args);

    // On Linux/Vulkan, missing per-platform sources are not fatal: the
    // Vulkan backend ships inline GLSL for every required stage and
    // compiles it whenever the payload carries no bytes for that stage.
    if resolved.is_none() && platform_key() == "glsl" {
        // The bundled default shader declares only metal/hlsl built-ins and
        // renders via the backend's inline GLSL by design, so stay quiet for
        // it. Only a custom stage that forgot its glsl variant (some
        // non-built-in source) is worth flagging -- it won't render as
        // authored on Vulkan.
        if !declares_only_builtin_sources(stage_args) {
            tracing::warn!(
                "Asset '{}': no shader source for platform \"glsl\", falling back to built-in GLSL",
                ctx.name
            );
        }
        return Ok(None);
    }

    let raw = resolved.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Compiled asset '{}': no shader source for platform \"{}\"",
                ctx.name,
                platform_key()
            ),
        )
    })?;

    let source_path = resolve_source_path_for(&raw, ctx);

    let compile_args = crate::shader::ShaderCompileArgs {
        source_path,
        asset_name: ctx.name.to_string(),
        kind: kind.compile_kind().to_string(),
    };
    crate::shader::compile_shader(compile_args)
        .map(Some)
        .map_err(|e| std::io::Error::other(format!("Asset '{}' compile error: {}", ctx.name, e)))
}

impl crate::asset::BuildAsset for Shader {
    // Compile every declared stage and pack them into one ShaderPayload
    // container: a Shader is one asset with one payload, so its stages load
    // and unload together.
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let mut payload = ShaderPayload::default();
        for (field, kind) in STAGES {
            let Some(stage_args) = args.get(field) else {
                continue;
            };
            if let Some(bytes) = compile_stage(stage_args, *kind, ctx)? {
                payload.stages.push((*kind, bytes));
            }
        }
        payload.encode().map_err(|e| {
            std::io::Error::other(format!("Asset '{}': shader payload encode: {e}", ctx.name))
        })
    }

    // The same source can be selected by more than one backend and still
    // compile to different bytecode, so the compile target is an input.
    const TARGET_DEPENDENT: bool = true;

    // Each stage compiles exactly one source: the current backend's. Reporting
    // those sources alone (rather than letting the cache's generic walk hash
    // every path in every `sources` map) keeps an edit to the `.glsl` variant
    // from invalidating the DirectX build's cached payload.
    //
    // A stage with no source for this backend contributes nothing: on Vulkan
    // it compiles nothing and the payload carries no bytes for it, so there is
    // no input to hash.
    fn source_files(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> crate::asset::SourceFiles {
        use crate::asset::{SourceFiles, SourceInput};
        let mut inputs = Vec::new();
        for (field, _) in STAGES {
            let Some(stage_args) = args.get(field) else {
                continue;
            };
            let Some(raw) = resolve_source_from_args(stage_args) else {
                continue;
            };
            // A built-in has no filesystem path; its embedded source is hashed
            // so editing a shipped shader and rebuilding the binary busts the
            // entry.
            if concinnity_core::build::shader::builtin_shader_source(&raw).is_some() {
                inputs.push(SourceInput::Builtin(raw));
                continue;
            }
            let path = resolve_source_path_for(&raw, ctx);
            if std::path::Path::new(&path).exists() {
                inputs.push(SourceInput::Path(path));
            }
        }
        SourceFiles::Only(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, SourceFiles, SourceInput};

    fn ctx<'a>(artifacts_dir: Option<&'a str>) -> BuildCtx<'a> {
        BuildCtx {
            name: "s",
            artifacts_dir,
            all_assets: &[],
        }
    }

    // A shader whose stages declare a source for the backend this build
    // targets, so `resolve_source_from_args` selects them on every platform.
    fn args(vertex: &str, fragment: &str) -> serde_json::Value {
        serde_json::json!({
            "vertex": {"sources": {platform_key(): vertex}},
            "fragment": {"sources": {platform_key(): fragment}},
        })
    }

    #[test]
    fn resolve_source_path_for_keeps_paths_with_a_directory_component() {
        // A path that already contains a directory is returned verbatim; the
        // `.concinnity/assets/` recursive search consults process-global asset
        // anchors and is left to integration coverage.
        assert_eq!(
            resolve_source_path_for("shaders/x.metal", &ctx(None)),
            "shaders/x.metal"
        );
    }

    #[test]
    fn resolve_source_path_for_prefers_an_artifact_over_the_assets_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("user.metal"), "// msl").unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path_for("user.metal", &ctx(Some(&artifacts))),
            format!("{artifacts}/user.metal")
        );
    }

    // The fallback reads the process-global assets anchor, so it shares the
    // build-output lock with the other tests that install a state dir.
    #[test]
    fn resolve_source_path_for_falls_back_to_the_assets_dir() {
        let _guard = crate::blob::test_output::LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let expected = concinnity_core::paths::assets_dir()
            .join("cn_no_such.metal")
            .to_string_lossy()
            .into_owned();
        // No artifacts dir at all...
        assert_eq!(
            resolve_source_path_for("cn_no_such.metal", &ctx(None)),
            expected
        );
        // ...and an artifacts dir that doesn't hold the file both land there.
        let dir = tempfile::tempdir().unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path_for("cn_no_such.metal", &ctx(Some(&artifacts))),
            expected
        );
    }

    #[test]
    fn a_compile_failure_names_the_asset() {
        // The vertex source resolves to a path that does not exist, so the
        // compile fails while reading it and never reaches a shader toolchain.
        let err =
            Shader::compile_payload(&args("cn_no_such.metal", "cn_no_such.metal"), &ctx(None))
                .unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("Asset 's' compile error:"), "got: {msg}");
        assert!(msg.contains("cn_no_such.metal"), "got: {msg}");
    }

    #[test]
    fn source_files_reports_each_stage_and_builtins_by_embedded_source() {
        // Built-ins have no filesystem path, so they are reported by name and
        // hashed from the source embedded in the binary; one entry per stage.
        assert_eq!(
            Shader::source_files(&args("default.metal", "default.metal"), &ctx(None)),
            SourceFiles::Only(vec![
                SourceInput::Builtin("default.metal".to_string()),
                SourceInput::Builtin("default.metal".to_string()),
            ])
        );
    }

    #[test]
    fn source_files_reports_a_user_shader_only_once_it_exists_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("user.metal");
        let raw = path.to_string_lossy().into_owned();
        // Nothing on disk yet: an empty set, since there is no input to hash.
        assert_eq!(
            Shader::source_files(&args(&raw, &raw), &ctx(None)),
            SourceFiles::Only(Vec::new())
        );
        std::fs::write(&path, "// msl").unwrap();
        assert_eq!(
            Shader::source_files(&args(&raw, &raw), &ctx(None)),
            SourceFiles::Only(vec![SourceInput::Path(raw.clone()), SourceInput::Path(raw)])
        );
        // A shader declaring no stage sources for this backend hashes nothing.
        assert_eq!(
            Shader::source_files(
                &serde_json::json!({"vertex": {}, "fragment": {}}),
                &ctx(None)
            ),
            SourceFiles::Only(Vec::new())
        );
        // The same source compiles to different bytecode per backend.
        const { assert!(Shader::TARGET_DEPENDENT) };
    }
}
