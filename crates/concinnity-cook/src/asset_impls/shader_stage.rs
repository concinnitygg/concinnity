// asset_impls/shader_stage.rs

use crate::asset::BuildCtx;
use concinnity_core::assets::ShaderKind;
use concinnity_core::assets::ShaderStage;
use concinnity_core::assets::shader_stage::platform_key;
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

impl crate::asset::BuildAsset for ShaderStage {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let shader_kind: ShaderKind = args
            .get("kind")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(ShaderKind::Vertex);

        let resolved = resolve_source_from_args(args);

        // On Linux/Vulkan, missing per-platform sources are not fatal: the
        // Vulkan backend ships inline GLSL for every required stage and
        // compiles it whenever the payload bytes aren't valid SPIR-V.
        if resolved.is_none() && platform_key() == "glsl" {
            // The bundled default shader set declares only metal/hlsl built-ins
            // and renders via the backend's inline GLSL by design, so stay
            // quiet for it. Only a custom stage that forgot its glsl variant
            // (some non-built-in source) is worth flagging -- it won't render
            // as authored on Vulkan.
            if !declares_only_builtin_sources(args) {
                tracing::warn!(
                    "Asset '{}': no shader source for platform \"glsl\", falling back to built-in GLSL",
                    ctx.name
                );
            }
            return Ok(vec![]);
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
            kind: shader_kind.compile_kind().to_string(),
        };
        crate::shader::compile_shader(compile_args).map_err(|e| {
            std::io::Error::other(format!("Asset '{}' compile error: {}", ctx.name, e))
        })
    }

    // The cache's generic JSON-string walk only finds bare filenames via
    // `find_in_assets` (which walks `.concinnity/assets/`). A `sources` entry
    // with a directory component, or a bare filename that lives in
    // `<artifacts_dir>` instead of `.concinnity/assets/`, is missed. Built-in
    // shader names short-circuit through the generic walk's `builtin:` path
    // so we skip them here.
    fn source_files(args: &serde_json::Value, ctx: &crate::asset::BuildCtx<'_>) -> Vec<String> {
        let Some(raw) = resolve_source_from_args(args) else {
            return Vec::new();
        };
        if concinnity_core::build::shader::builtin_shader_source(&raw).is_some() {
            return Vec::new();
        }
        let path = resolve_source_path_for(&raw, ctx);
        if std::path::Path::new(&path).exists() {
            vec![path]
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_source_path_for_keeps_paths_with_a_directory_component() {
        // A path that already contains a directory is returned verbatim; the
        // bare-filename branch consults process-global asset anchors and is left
        // to integration coverage.
        let ctx = BuildCtx {
            name: "s",
            artifacts_dir: None,
            all_assets: &[],
        };
        assert_eq!(
            resolve_source_path_for("shaders/x.metal", &ctx),
            "shaders/x.metal"
        );
    }
}
