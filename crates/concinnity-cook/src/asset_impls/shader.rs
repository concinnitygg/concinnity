use concinnity_core::components::{Shader, ShaderPrograms, ShaderStage};
use concinnity_core::render::shader_programs::surface::Sources;
use concinnity_core::render::shader_source::SourceFile;

use crate::asset::BuildCtx;
use crate::compile::shader::{compile_world_shader, read_shader_source};

// Resolve a declared source path to the on-disk path the build will read. A
// bare filename is looked up recursively under the build's asset search root,
// then falls back to `<assets>/<raw>`. A path with a directory component is
// used verbatim.
pub(super) fn resolve_source_path_for(raw: &str, ctx: &BuildCtx<'_>) -> String {
    let p = std::path::Path::new(raw);
    if p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true) {
        if let Some(path) = ctx
            .assets_dir
            .and_then(|dir| concinnity_host::store::source::find_in(dir, raw))
        {
            return path;
        }
        if let Some(assets) = ctx.assets_dir {
            return assets.join(raw).to_string_lossy().into_owned();
        }
    }
    raw.to_string()
}

// The declared path of one of the Shader's files, straight from the args.
fn declared_path(args: &serde_json::Value, stage: ShaderStage) -> Option<String> {
    let field = match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
    };
    args.get(field)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

// A Shader's compiled programs, or an error naming what this host is missing.
fn world_shader_programs(
    name: &str,
    sources: &Sources<'_>,
    platform: concinnity_core::platform::Platform,
    have_compiler: bool,
) -> std::io::Result<ShaderPrograms> {
    crate::compile::program::require_compiler(&format!("Shader '{name}'"), have_compiler)?;
    Ok(compile_world_shader(name, sources, platform)?.programs)
}

impl crate::asset::BuildAsset for Shader {
    // Read the declared files and compile every program this backend consumes
    // into one container: a Shader is one asset with one payload, so its
    // programs load and unload together.
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let fragment_raw = declared_path(args, ShaderStage::Fragment).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Shader '{}': no `fragment` file declared (set it to a `.hlsl` path \
                     defining `shade`)",
                    ctx.name
                ),
            )
        })?;
        let fragment = read_shader_source(&resolve_source_path_for(&fragment_raw, ctx))?;
        let vertex_raw = declared_path(args, ShaderStage::Vertex);
        let vertex = vertex_raw
            .as_deref()
            .map(|raw| read_shader_source(&resolve_source_path_for(raw, ctx)))
            .transpose()?;
        // Diagnostics name each file as the world declared it.
        let sources = Sources {
            vertex: vertex_raw
                .as_deref()
                .zip(vertex.as_deref())
                .map(|(path, text)| SourceFile { path, text }),
            fragment: SourceFile {
                path: &fragment_raw,
                text: &fragment,
            },
        };
        let programs = world_shader_programs(
            ctx.name,
            &sources,
            ctx.platform,
            concinnity_shader::dxc_available(),
        )?;
        programs.encode().map_err(|e| {
            std::io::Error::other(format!("Asset '{}': shader payload encode: {e}", ctx.name))
        })
    }

    // The same files compile to SPIR-V on one backend, a DXIL container on
    // another and MSL text on the third, so the target belongs in the cache key.
    const TARGET_DEPENDENT: bool = true;

    // Only the declared files are read. Reporting them covers the resolution
    // the cache's generic walk misses: a bare filename found under the asset
    // root, which the walk cannot see.
    fn source_files(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> crate::asset::SourceFiles {
        use crate::asset::SourceFiles;
        let mut inputs = Vec::new();
        for stage in [ShaderStage::Vertex, ShaderStage::Fragment] {
            let Some(raw) = declared_path(args, stage) else {
                continue;
            };
            let path = resolve_source_path_for(&raw, ctx);
            if std::path::Path::new(&path).exists() {
                inputs.push(path);
            }
        }
        SourceFiles::Only(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, SourceFiles};

    fn ctx() -> BuildCtx<'static> {
        with_assets(None)
    }

    fn with_assets(assets_dir: Option<&std::path::Path>) -> BuildCtx<'_> {
        BuildCtx {
            name: "s",
            platform: concinnity_core::platform::Platform::Metal,
            assets_dir,
            all_assets: &[],
        }
    }

    fn args(fragment: &str) -> serde_json::Value {
        serde_json::json!({ "fragment": fragment })
    }

    // A Shader that has to be compiled on a host with no compiler is an error
    // naming the Shader, not a payload that quietly renders as something else.
    // The message carries the resolver's own reason, so it says what to install.
    #[test]
    fn a_shader_needing_a_compiler_fails_when_there_is_none() {
        let sources = Sources {
            vertex: Some(SourceFile {
                path: "v.hlsl",
                text: "VertexOut transform() {}",
            }),
            fragment: SourceFile {
                path: "f.hlsl",
                text: "float4 shade() { return float4(1.0); }",
            },
        };
        let err = world_shader_programs(
            "wall",
            &sources,
            concinnity_core::platform::Platform::Metal,
            false,
        )
        .expect_err("no compiler is an error");

        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let message = err.to_string();
        assert!(message.contains("wall"), "{message}");
        assert!(
            message.contains("compiled payload") || message.contains("dxc"),
            "{message}"
        );
    }

    #[test]
    fn resolve_source_path_for_keeps_paths_with_a_directory_component() {
        // A path that already contains a directory is returned verbatim: no
        // search applies, with or without an asset root.
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            resolve_source_path_for("shaders/x.hlsl", &ctx()),
            "shaders/x.hlsl"
        );
        assert_eq!(
            resolve_source_path_for("shaders/x.hlsl", &with_assets(Some(dir.path()))),
            "shaders/x.hlsl"
        );
    }

    // A bare filename is found by recursive search under the asset root.
    #[test]
    fn resolve_source_path_for_finds_a_nested_asset() {
        let assets = tempfile::tempdir().unwrap();
        let nested = assets.path().join("shaders");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("user.hlsl"), "// hlsl").unwrap();
        assert_eq!(
            resolve_source_path_for("user.hlsl", &with_assets(Some(assets.path()))),
            nested.join("user.hlsl").to_string_lossy()
        );
    }

    #[test]
    fn resolve_source_path_for_falls_back_to_the_assets_dir() {
        let assets = tempfile::tempdir().unwrap();
        assert_eq!(
            resolve_source_path_for("cn_no_such.hlsl", &with_assets(Some(assets.path()))),
            assets.path().join("cn_no_such.hlsl").to_string_lossy()
        );
        // With no search root at all the bare name is left as it was authored.
        assert_eq!(
            resolve_source_path_for("cn_no_such.hlsl", &ctx()),
            "cn_no_such.hlsl"
        );
    }

    // The fragment file is the one required input: without it there is no
    // `shade`, and the error says which field to set.
    #[test]
    fn no_fragment_file_is_a_hard_error() {
        let err = Shader::compile_payload(&serde_json::json!({}), &ctx()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("no `fragment` file declared"),
            "got: {err}"
        );
        let err = Shader::compile_payload(&args(""), &ctx()).unwrap_err();
        assert!(
            err.to_string().contains("no `fragment` file declared"),
            "got: {err}"
        );
    }

    // A missing file fails at the read, naming the path, before any compiler
    // runs.
    #[test]
    fn a_missing_file_names_the_path() {
        let err = Shader::compile_payload(&args("/no/such/user.hlsl"), &ctx()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("/no/such/user.hlsl"), "got: {err}");

        let both = serde_json::json!({"vertex": "/no/such/v.hlsl", "fragment": "/no/such/f.hlsl"});
        let dir = tempfile::tempdir().unwrap();
        let frag = dir.path().join("f.hlsl");
        std::fs::write(&frag, "// f").unwrap();
        let mut both = both;
        both["fragment"] = serde_json::Value::String(frag.to_string_lossy().into_owned());
        let err = Shader::compile_payload(&both, &ctx()).unwrap_err();
        assert!(err.to_string().contains("/no/such/v.hlsl"), "got: {err}");
    }

    #[test]
    fn source_files_reports_the_declared_files_once_they_exist_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let frag = dir.path().join("f.hlsl");
        let vert = dir.path().join("v.hlsl");
        let raw_f = frag.to_string_lossy().into_owned();
        let raw_v = vert.to_string_lossy().into_owned();
        let both = serde_json::json!({"vertex": raw_v, "fragment": raw_f});
        // Nothing on disk yet: an empty set, since there is no input to hash.
        assert_eq!(
            Shader::source_files(&both, &ctx()),
            SourceFiles::Only(Vec::new())
        );
        std::fs::write(&frag, "// f").unwrap();
        assert_eq!(
            Shader::source_files(&both, &ctx()),
            SourceFiles::Only(vec![raw_f.clone()])
        );
        std::fs::write(&vert, "// v").unwrap();
        assert_eq!(
            Shader::source_files(&both, &ctx()),
            SourceFiles::Only(vec![raw_v, raw_f])
        );
        // A Shader declaring nothing hashes nothing.
        assert_eq!(
            Shader::source_files(&serde_json::json!({}), &ctx()),
            SourceFiles::Only(Vec::new())
        );
        // The same files compile to different artifacts per backend.
        const { assert!(Shader::TARGET_DEPENDENT) };
    }
}
