// asset_impls/sdf_volume.rs

use crate::asset::BuildCtx;
use crate::authoring::source_args::sdf_volume_source_path;
use concinnity_core::components::SdfVolume;

// Resolve a raw `fragment_shader` arg to an on-disk path, picking the first
// candidate that exists. `<assets>` is the build's asset search root.
// Resolution order:
//   1. `<assets>/<raw>`: runtime-fetched cache (the production
//      location once a world has been built and `cn run` fetches its
//      dependencies).
//   2. `<assets>/<bare>` recursive search: same bare-filename
//      match `Shader` does.
//   3. `<artifacts_dir>/<raw>`: LLM-written artifact under
//      `data/artifacts/<account_id>/`, matching the existing Shader stage path.
//   4. `assets/<raw>`: source-tree convenience for `cn debug` run from
//      `concinnity-engine/` against shaders authored in the repo's `assets/`
//      directory.
//   5. `<raw>` as-is: relative-to-cwd fallback (matches how other asset
//      `source` fields handle e.g. `"../concinnity-infra/assets/..."`).
// Returns `None` when nothing exists; `compile_payload` falls back to the raw
// path in that case so the read error surfaces with a useful message.
pub(super) fn resolve_source_path(raw: &str, ctx: &BuildCtx<'_>) -> Option<String> {
    let raw_path = std::path::Path::new(raw);
    let mut candidates: Vec<String> = Vec::new();
    if raw_path.is_absolute() {
        candidates.push(raw.to_string());
    } else {
        if let Some(assets) = ctx.assets_dir {
            candidates.push(assets.join(raw).to_string_lossy().into_owned());
        }
        if raw_path
            .parent()
            .map(|d| d.as_os_str().is_empty())
            .unwrap_or(true)
            && let Some(found) = ctx
                .assets_dir
                .and_then(|dir| concinnity_host::store::source::find_in(dir, raw))
        {
            candidates.push(found);
        }
        if let Some(dir) = ctx.artifacts_dir {
            candidates.push(format!("{dir}/{raw}"));
        }
        candidates.push(format!("assets/{raw}"));
        candidates.push(raw.to_string());
    }
    candidates
        .into_iter()
        .find(|p| std::path::Path::new(p).exists())
}

impl crate::asset::BuildAsset for SdfVolume {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let raw = sdf_volume_source_path(args).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "SdfVolume '{}': no distance field declared (set `fragment_shader` \
                     to a `.slang` path declaring map + shade, or sampleVolume)",
                    ctx.name
                ),
            )
        })?;

        let source_path = resolve_source_path(&raw, ctx).unwrap_or_else(|| raw.clone());
        let field = std::fs::read_to_string(&source_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "SdfVolume '{}': failed to read distance field '{}': {}",
                    ctx.name, source_path, e
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

        let programs =
            super::sdf_field::compile(ctx.name, &field, ctx.platform, volumetric, cast_shadows)?;
        postcard::to_allocvec(&programs).map_err(|e| {
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

    // Only the declared field is read. Reporting it covers the resolution the
    // cache's generic walk misses: `fragment_shader` is typically a path with a
    // directory component under the source-tree `assets/` dir, and without this
    // an edit to it would replay stale bytes forever.
    fn source_files(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> crate::asset::SourceFiles {
        use crate::asset::SourceFiles;
        let Some(raw) = sdf_volume_source_path(args) else {
            return SourceFiles::Only(Vec::new());
        };
        SourceFiles::Only(resolve_source_path(&raw, ctx).into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, SourceFiles};
    use concinnity_core::platform::Platform;

    fn args(source: &str) -> serde_json::Value {
        serde_json::json!({ "fragment_shader": source })
    }

    fn ctx<'a>(artifacts_dir: Option<&'a str>) -> BuildCtx<'a> {
        BuildCtx {
            name: "blob",
            platform: Platform::Metal,
            assets_dir: None,
            artifacts_dir,
            all_assets: &[],
        }
    }

    // A minimal surface field: enough for slangc to accept it, so a compile
    // that fails in these tests is the engine template's fault, not the field's.
    const FIELD: &str = r#"
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
"#;

    #[test]
    fn an_absolute_path_resolves_only_when_it_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chrome.slang");
        std::fs::write(&path, FIELD).unwrap();
        let raw = path.to_string_lossy().into_owned();
        assert_eq!(resolve_source_path(&raw, &ctx(None)), Some(raw.clone()));

        let missing = dir
            .path()
            .join("absent.slang")
            .to_string_lossy()
            .into_owned();
        assert_eq!(resolve_source_path(&missing, &ctx(None)), None);
    }

    #[test]
    fn a_relative_path_resolves_under_the_artifacts_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("shaders")).unwrap();
        std::fs::write(dir.path().join("shaders/chrome.slang"), FIELD).unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path("shaders/chrome.slang", &ctx(Some(&artifacts))),
            Some(format!("{artifacts}/shaders/chrome.slang"))
        );
    }

    #[test]
    fn a_bare_filename_resolves_under_the_artifacts_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("chrome.slang"), FIELD).unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path("chrome.slang", &ctx(Some(&artifacts))),
            Some(format!("{artifacts}/chrome.slang"))
        );
    }

    #[test]
    fn an_unresolvable_relative_path_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path("cn_no_such_field.slang", &ctx(Some(&artifacts))),
            None
        );
        assert_eq!(
            resolve_source_path("cn_no_such_field.slang", &ctx(None)),
            None
        );
    }

    #[test]
    fn a_missing_source_file_names_the_asset_and_the_path() {
        let err =
            SdfVolume::compile_payload(&args("/no/such/chrome.slang"), &ctx(None)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string().contains(
                "SdfVolume 'blob': failed to read distance field '/no/such/chrome.slang'"
            ),
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
        let path = dir.path().join("chrome.slang");
        std::fs::write(&path, FIELD).unwrap();
        let raw = path.to_string_lossy().into_owned();
        assert_eq!(
            SdfVolume::source_files(&args(&raw), &ctx(None)),
            SourceFiles::Only(vec![raw])
        );
        // Nothing declared and nothing resolvable both report an empty set.
        assert_eq!(
            SdfVolume::source_files(&serde_json::json!({}), &ctx(None)),
            SourceFiles::Only(Vec::new())
        );
        assert_eq!(
            SdfVolume::source_files(&args("/no/such/chrome.slang"), &ctx(None)),
            SourceFiles::Only(Vec::new())
        );
        // The field compiles to a different artifact per backend, so two
        // backends must not share one cache entry for it.
        const { assert!(SdfVolume::TARGET_DEPENDENT) };
    }
}
