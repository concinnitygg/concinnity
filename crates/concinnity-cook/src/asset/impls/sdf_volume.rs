#[cfg(test)]
use concinnity_core::platform::Platform;
use concinnity_core::platform::PlatformSet;
use concinnity_engine::components::SdfVolume;
use concinnity_engine::render::sdf::programs::SdfPrograms;
use concinnity_engine::render::shader::source::SourceFile;

use crate::asset::BuildCtx;
use crate::authoring::source_args::sdf_volume_source_path;
use crate::compile::sdf_field::compile_sdf_field;

// The on-disk path of a raw `fragment_shader` arg, under the build's root.
pub(super) fn resolve_source_path(raw: &str, ctx: &BuildCtx<'_>) -> std::path::PathBuf {
    concinnity_host::store::source::resolve(raw, ctx.root)
}

// A volume's compiled field, or an error naming what this host is missing.
fn field_programs(
    name: &str,
    field: SourceFile<'_>,
    targets: PlatformSet,
    flags: (bool, bool),
    have_compiler: bool,
) -> std::io::Result<SdfPrograms> {
    let owner = format!("SdfVolume '{name}'");
    crate::compile::program::require_compiler(&owner, have_compiler)?;
    let (volumetric, cast_shadows) = flags;
    crate::compile::program::for_each_target(
        &owner,
        targets,
        |platform| Ok(compile_sdf_field(name, field, platform, volumetric, cast_shadows)?.programs),
        |field: &mut SdfPrograms| &mut field.programs,
    )
}

impl crate::asset::BuildAsset for SdfVolume {
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>> {
        let raw = sdf_volume_source_path(args).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "SdfVolume '{}': no distance field declared (set `fragment_shader` \
                     to a `.hlsl` path declaring map + shade, or sampleVolume)",
                    ctx.name
                ),
            )
        })?;

        let source_path = resolve_source_path(&raw, ctx);
        let field = std::fs::read_to_string(&source_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "SdfVolume '{}': failed to read distance field '{}': {}",
                    ctx.name,
                    source_path.display(),
                    e
                ),
            )
        })?;

        // The flags decide which entries exist: a medium is integrated rather
        // than surfaced, and only a caster needs the depth-only pair. Read from
        // the args rather than the validated asset because the payload is built
        // before validation runs.
        let flag = |k: &str| args.get(k).and_then(serde_json::Value::as_bool);
        let volumetric = flag("volumetric").unwrap_or(false);
        let cast_shadows = flag("cast_shadows").unwrap_or(false);

        // Diagnostics name the field as the world declared it.
        let programs = field_programs(
            ctx.name,
            SourceFile {
                path: &raw,
                text: &field,
            },
            ctx.targets,
            (volumetric, cast_shadows),
            concinnity_shader::dxc_available(),
        )?;
        programs.encode().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("SdfVolume '{}': encoding compiled field: {e}", ctx.name),
            )
        })
    }

    // The compile emits SPIR-V on one backend, a DXIL container on another and
    // MSL text on the third, so one field's bytes produce three different
    // payloads and the target belongs in the cache key.
    const TARGET_DEPENDENT: bool = true;

    // Only the declared field is read, when it is there.
    fn source_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> crate::asset::SourceFiles {
        use crate::asset::SourceFiles;
        let Some(raw) = sdf_volume_source_path(args) else {
            return SourceFiles::Only(Vec::new());
        };
        let path = resolve_source_path(&raw, ctx);
        SourceFiles::Only(path.exists().then_some(path).into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, SourceFiles};

    fn args(source: &str) -> serde_json::Value {
        serde_json::json!({ "fragment_shader": source })
    }

    fn ctx(root: Option<&std::path::Path>) -> BuildCtx<'_> {
        BuildCtx {
            name: "blob",
            targets: PlatformSet::single(Platform::Metal),
            root,
            all_assets: &[],
        }
    }

    // A minimal surface field: enough for the compiler to accept it, so a compile
    // that fails in these tests is the engine template's fault, not the field's.
    const FIELD: &str = r"
float map(float3 p, SdfParams params, float time) { return sdSphere(p, 0.5); }
SdfSurface shade(float3 p, float3 n, SdfParams params, float time, float2 uv) {
    SdfSurface s;
    s.albedo = float3(1.0, 1.0, 1.0);
    s.roughness = 0.5;
    s.metallic = 0.0;
    s.emissive = float3(0.0, 0.0, 0.0);
    s.transmitted = float3(0.0, 0.0, 0.0);
    return s;
}
";

    // A volume whose field has to be compiled on a host with no compiler is an
    // error naming the volume, not a payload that quietly draws nothing.
    #[test]
    fn a_volume_needing_a_compiler_fails_when_there_is_none() {
        let field = SourceFile {
            path: "shaders/blob.hlsl",
            text: "float map(float3 p) { return length(p) - 1.0; }",
        };
        let err = field_programs("blob", field, Platform::Metal.into(), (false, true), false)
            .expect_err("no compiler is an error");

        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let message = err.to_string();
        assert!(message.contains("blob"), "{message}");
        assert!(
            message.contains("compiled payload") || message.contains("dxc"),
            "{message}"
        );
    }

    // A relative path names a file under the root, whether or not it is there
    // yet; an absolute one names itself; nothing is searched for by name.
    #[test]
    fn a_field_path_resolves_under_the_root() {
        let root = std::path::Path::new("/p");
        assert_eq!(
            resolve_source_path("fields/chrome.hlsl", &ctx(Some(root))),
            root.join("fields/chrome.hlsl")
        );
        assert_eq!(
            resolve_source_path("chrome.hlsl", &ctx(Some(root))),
            root.join("chrome.hlsl")
        );
        assert_eq!(
            resolve_source_path("/abs/chrome.hlsl", &ctx(Some(root))),
            std::path::Path::new("/abs/chrome.hlsl")
        );
    }

    // The cache keys on the field only once it exists.
    #[test]
    fn only_an_existing_field_is_a_source_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("fields")).unwrap();
        std::fs::write(dir.path().join("fields/chrome.hlsl"), FIELD).unwrap();
        let files =
            |field: &str| match SdfVolume::source_files(&args(field), &ctx(Some(dir.path()))) {
                SourceFiles::Only(files) => files,
                SourceFiles::Extra(_) => panic!("a volume names its only input"),
            };
        assert_eq!(
            files("fields/chrome.hlsl"),
            [dir.path().join("fields/chrome.hlsl")]
        );
        assert!(files("fields/absent.hlsl").is_empty());
    }

    #[test]
    fn a_missing_source_file_names_the_asset_and_the_path() {
        let err =
            SdfVolume::compile_payload(&args("/no/such/chrome.hlsl"), &ctx(None)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string()
                .contains("SdfVolume 'blob': failed to read distance field '/no/such/chrome.hlsl'"),
            "got: {err}"
        );
    }

    #[test]
    fn no_declared_field_is_a_hard_error() {
        let err = SdfVolume::compile_payload(&serde_json::json!({}), &ctx(None)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("no distance field declared"),
            "got: {err}"
        );
    }

    #[test]
    fn source_files_reports_only_the_declared_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chrome.hlsl");
        std::fs::write(&path, FIELD).unwrap();
        let raw = path.to_string_lossy().into_owned();
        assert_eq!(
            SdfVolume::source_files(&args(&raw), &ctx(None)),
            SourceFiles::Only(vec![path])
        );
        // Nothing declared and nothing resolvable both report an empty set.
        assert_eq!(
            SdfVolume::source_files(&serde_json::json!({}), &ctx(None)),
            SourceFiles::Only(Vec::new())
        );
        assert_eq!(
            SdfVolume::source_files(&args("/no/such/chrome.hlsl"), &ctx(None)),
            SourceFiles::Only(Vec::new())
        );
        // The field compiles to a different artifact per backend, so two
        // backends must not share one cache entry for it.
        const { assert!(SdfVolume::TARGET_DEPENDENT) };
    }
}
