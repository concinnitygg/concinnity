// asset_impls/sdf_volume.rs

use crate::asset::BuildCtx;
use concinnity_core::assets::SdfVolume;
use concinnity_world::source_args::current_platform_source_arg;

// Resolve a raw `fragment_shader` arg to an on-disk path, picking the first
// candidate that exists. Resolution order:
//   1. `.concinnity/assets/<raw>`: runtime-fetched cache (the production
//      location once a world has been built and `cn run` fetches its
//      dependencies).
//   2. `.concinnity/assets/<bare>` recursive search: same bare-filename
//      match `ShaderStage` does.
//   3. `<artifacts_dir>/<raw>`: LLM-written artifact under
//      `data/artifacts/<account_id>/`, matching the existing ShaderStage path.
//   4. `assets/<raw>`: source-tree convenience for `cn debug` run from
//      `concinnity-engine/` against shaders authored in the repo's `assets/`
//      directory.
//   5. `<raw>` as-is: relative-to-cwd fallback (matches how other asset
//      `source` fields handle e.g. `"../concinnity-infra/assets/..."`).
// Returns `None` when nothing exists; `compile_payload` falls back to the raw
// path in that case so the read error surfaces with a useful message.
pub fn resolve_source_path(raw: &str, ctx: &BuildCtx<'_>) -> Option<String> {
    let raw_path = std::path::Path::new(raw);
    let mut candidates: Vec<String> = Vec::new();
    if raw_path.is_absolute() {
        candidates.push(raw.to_string());
    } else {
        candidates.push(
            concinnity_core::paths::assets_dir()
                .join(raw)
                .to_string_lossy()
                .into_owned(),
        );
        if raw_path
            .parent()
            .map(|d| d.as_os_str().is_empty())
            .unwrap_or(true)
            && let Some(found) = concinnity_core::paths::find_in_assets(raw)
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
        // Only the current backend's shader is required: a volume that
        // declares an `.hlsl` source (or an `hlsl`-only map) contributes
        // nothing the Metal build can compile, so it is a hard error here
        // rather than an attempt to read a file the backend never needs.
        let platform_key = concinnity_core::build::Platform::current().key();
        let raw = current_platform_source_arg(args).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "SdfVolume '{}': no fragment shader source for backend \"{}\" \
                     (declare `fragment_shaders.{}` or a `fragment_shader` path \
                     with a matching extension)",
                    ctx.name, platform_key, platform_key
                ),
            )
        })?;

        let source_path = resolve_source_path(&raw, ctx).unwrap_or_else(|| raw.clone());

        // No MSL compilation here: the runtime backend prepends the
        // engine-shipped helpers + appends the template and compiles
        // via `newLibraryWithSource_options_error` (matching how every
        // other Metal feature pass loads its MSL). We just transport
        // the user source bytes through the blob so `cn run` worlds
        // don't need the file on disk.
        std::fs::read(&source_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "SdfVolume '{}': failed to read fragment shader '{}': {}",
                    ctx.name, source_path, e
                ),
            )
        })
    }

    // `TARGET_DEPENDENT` stays false: `compile_payload` transports the source
    // bytes verbatim, so identical bytes yield an identical payload and two
    // backends pointing at one file may correctly share a cache entry.

    // Only the current backend's shader is read. Reporting it alone keeps an
    // edit to a sibling backend's shader from invalidating this one, and
    // covers the resolution the cache's generic walk misses: `fragment_shader`
    // is typically a path with a directory component (e.g.
    // `"shaders/chrome_blob.metal"`) under the source-tree `assets/` dir.
    // Without it, editing that file would replay stale bytes forever.
    fn source_files(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> crate::asset::SourceFiles {
        use crate::asset::{SourceFiles, SourceInput};
        let Some(raw) = current_platform_source_arg(args) else {
            return SourceFiles::Only(Vec::new());
        };
        SourceFiles::Only(
            resolve_source_path(&raw, ctx)
                .map(SourceInput::Path)
                .into_iter()
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::{BuildAsset, SourceFiles, SourceInput};

    // A `.metal` source is the one the macOS build selects; on the other
    // backends the same shape resolves through their own extension.
    fn args(source: &str) -> serde_json::Value {
        let key = concinnity_core::build::Platform::current().key();
        serde_json::json!({"fragment_shaders": {key: source}})
    }

    fn ctx<'a>(artifacts_dir: Option<&'a str>) -> BuildCtx<'a> {
        BuildCtx {
            name: "blob",
            artifacts_dir,
            all_assets: &[],
        }
    }

    #[test]
    fn an_absolute_path_resolves_only_when_it_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chrome.metal");
        std::fs::write(&path, "// msl").unwrap();
        let raw = path.to_string_lossy().into_owned();
        assert_eq!(resolve_source_path(&raw, &ctx(None)), Some(raw.clone()));

        let missing = dir
            .path()
            .join("absent.metal")
            .to_string_lossy()
            .into_owned();
        assert_eq!(resolve_source_path(&missing, &ctx(None)), None);
    }

    #[test]
    fn a_relative_path_resolves_under_the_artifacts_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("shaders")).unwrap();
        std::fs::write(dir.path().join("shaders/chrome.metal"), "// msl").unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path("shaders/chrome.metal", &ctx(Some(&artifacts))),
            Some(format!("{artifacts}/shaders/chrome.metal"))
        );
    }

    #[test]
    fn a_bare_filename_resolves_under_the_artifacts_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("chrome.metal"), "// msl").unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path("chrome.metal", &ctx(Some(&artifacts))),
            Some(format!("{artifacts}/chrome.metal"))
        );
    }

    #[test]
    fn an_unresolvable_relative_path_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_source_path("cn_no_such_shader.metal", &ctx(Some(&artifacts))),
            None
        );
        assert_eq!(
            resolve_source_path("cn_no_such_shader.metal", &ctx(None)),
            None
        );
    }

    #[test]
    fn the_payload_is_the_shader_source_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chrome.metal");
        std::fs::write(&path, "fragment float4 f() { return 0; }").unwrap();
        let payload =
            SdfVolume::compile_payload(&args(&path.to_string_lossy()), &ctx(None)).unwrap();
        assert_eq!(payload, std::fs::read(&path).unwrap());
    }

    #[test]
    fn a_missing_source_file_names_the_asset_and_the_path() {
        let err =
            SdfVolume::compile_payload(&args("/no/such/chrome.metal"), &ctx(None)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string().contains(
                "SdfVolume 'blob': failed to read fragment shader '/no/such/chrome.metal'"
            ),
            "got: {err}"
        );
    }

    #[test]
    fn no_source_for_the_building_backend_is_a_hard_error() {
        let err = SdfVolume::compile_payload(&serde_json::json!({}), &ctx(None)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("no fragment shader source"),
            "got: {err}"
        );
    }

    #[test]
    fn source_files_reports_only_the_resolved_backend_shader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chrome.metal");
        std::fs::write(&path, "// msl").unwrap();
        let raw = path.to_string_lossy().into_owned();
        assert_eq!(
            SdfVolume::source_files(&args(&raw), &ctx(None)),
            SourceFiles::Only(vec![SourceInput::Path(raw)])
        );
        // Nothing declared and nothing resolvable both report an empty set.
        assert_eq!(
            SdfVolume::source_files(&serde_json::json!({}), &ctx(None)),
            SourceFiles::Only(Vec::new())
        );
        assert_eq!(
            SdfVolume::source_files(&args("/no/such/chrome.metal"), &ctx(None)),
            SourceFiles::Only(Vec::new())
        );
        // The source bytes pass through untouched, so two backends pointing at
        // one file share a cache entry.
        const { assert!(!SdfVolume::TARGET_DEPENDENT) };
    }
}
