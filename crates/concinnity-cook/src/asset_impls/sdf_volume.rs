// asset_impls/sdf_volume.rs

use crate::asset::BuildCtx;
use concinnity_core::assets::SdfVolume;
use concinnity_core::assets::sdf_volume::current_platform_source_arg;

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

    // The cache's generic JSON-string walk only resolves bare filenames
    // (via `find_in_assets`) and cwd-relative paths. `fragment_shader` is
    // typically a path with a directory component (e.g.
    // `"shaders/chrome_blob.metal"`) under the source-tree `assets/` dir,
    // which the generic walk misses. Without this override, editing the
    // .metal file would not invalidate the cache and stale bytes would
    // replay forever.
    fn source_files(args: &serde_json::Value, ctx: &crate::asset::BuildCtx<'_>) -> Vec<String> {
        let Some(raw) = current_platform_source_arg(args) else {
            return Vec::new();
        };
        resolve_source_path(&raw, ctx).into_iter().collect()
    }
}
