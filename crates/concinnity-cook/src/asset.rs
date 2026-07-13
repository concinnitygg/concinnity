// `BuildAsset` is the build-time counterpart to `Component`. Components whose
// `PAYLOAD = AssetPayload::Compiled` implement this trait to turn their args
// into a binary payload (mesh vertices, shader bytecode, decoded image, etc.).
// The build pipeline calls `<T as BuildAsset>::compile_payload` for each
// declared asset and packs the resulting bytes into a blob.
//
// `BuildCtx` is the build-time context handed to each impl. It lives here
// because it is build-only: the runtime never compiles a payload. `Platform`
// and the `SourceBacked` trait stay in concinnity-core, since the engine reads
// a `ShaderStage`'s current-platform source at runtime.

use crate::ecs::Component;
use crate::world::WorldJsonlAsset;

// Build-time context handed to each `BuildAsset` impl.
pub struct BuildCtx<'a> {
    // The asset's declared name (used in error messages and as a key for
    // build-time intermediates such as compiled shader filenames).
    pub name: &'a str,
    // Optional directory of user-supplied artifacts (e.g. account-uploaded
    // shader source files) consulted when resolving bare filenames.
    pub artifacts_dir: Option<&'a str>,
    // All sibling assets declared in the same world. Used by types like
    // `VoxelChunk` that need to resolve cross-asset references (palette).
    pub all_assets: &'a [WorldJsonlAsset],
}

// A component that compiles to a binary payload at build time.
//
// Only types whose `Component::PAYLOAD` is `AssetPayload::Compiled` should
// implement this. The build pipeline dispatches via a match on
// `ComponentType` in [`crate::pipeline`].
pub trait BuildAsset: Component {
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>>;

    // On-disk files this asset's `compile_payload` reads, beyond what the
    // payload cache can derive from the args JSON. The cache layer mixes the
    // contents-hash of each returned path into the per-asset cache key so an
    // edit to one of those files invalidates the cached payload.
    //
    // Default is empty: appropriate for assets whose only inputs are the
    // args themselves (or whose source paths are resolved by the cache's
    // generic JSON string walk). Override when `compile_payload` reads a
    // file at a path the generic walk would miss, e.g. `SdfVolume` reading
    // `assets/shaders/<name>.metal` from the source tree, or any asset
    // whose resolution rules differ from the cache's default lookup.
    //
    // Return only paths that exist on disk. The cache silently drops paths
    // it can't read, so returning a placeholder is safe but pointless.
    fn source_files(_args: &serde_json::Value, _ctx: &BuildCtx<'_>) -> Vec<String> {
        Vec::new()
    }
}
