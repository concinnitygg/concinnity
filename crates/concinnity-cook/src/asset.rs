// `BuildAsset` is the build-time counterpart to `Component`. Components whose
// `PAYLOAD = AssetPayload::Compiled` implement this trait to turn their args
// into a binary payload (mesh vertices, shader bytecode, decoded image, etc.).
// The build pipeline calls `<T as BuildAsset>::compile_payload` for each
// declared asset and packs the resulting bytes into a blob.
//
// `BuildCtx` is the build-time context handed to each impl. It lives here
// because it is build-only: the runtime never compiles a payload. `Platform`
// and the `SourceBacked` trait stay in concinnity-core, since the engine reads
// a Shader stage's current-platform source at runtime.

use crate::ecs::Component;
use crate::world::WorldJsonlAsset;

// The on-disk inputs an asset's `compile_payload` reads, and how they relate to
// the payload cache's generic walk of the args JSON.
//
// `Extra` adds to what the walk finds: the walk stays a conservative safety net
// that over-approximates the input set, and the asset contributes only the
// paths the walk would miss.
//
// `Only` replaces the walk entirely and is the complete input set. Use it when
// the walk over-approximates in a way that matters -- a Shader stage declaring
// both an `.hlsl` and a `.glsl` source reads exactly one of them, so hashing
// both makes an edit to the unused variant invalidate the other backend's
// cached payload. The tradeoff is that the safety net is gone: an input this
// list omits is not hashed at all, and a stale payload replays silently. Cover
// each `Only` impl with a test asserting the set it reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceFiles {
    Extra(Vec<String>),
    Only(Vec<String>),
}

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

    // True when identical source bytes compile to a different payload per
    // backend, so the compile target is itself an input to the payload and
    // belongs in the cache key. `Shader` sets this: the same file can be
    // selected by two backends (`Platform::accepts_ext` accepts any
    // unrecognised extension, and a `sources` map entry is taken without an
    // extension check at all) and compiles to DXIL on one and SPIR-V on the
    // other.
    //
    // Assets that transport their source verbatim leave this false, even when
    // they select that source per backend: `SdfVolume` reads shader bytes
    // straight through, so identical bytes yield an identical payload and two
    // backends may correctly share one cache entry.
    const TARGET_DEPENDENT: bool = false;

    // The inputs this asset's `compile_payload` reads, relative to the payload
    // cache's generic walk of the args JSON. The cache layer mixes a hash of
    // each input into the per-asset cache key so an edit to one of them
    // invalidates the cached payload.
    //
    // Default is `Extra(vec![])`: appropriate for assets whose only inputs are
    // the args themselves, or whose source paths the generic walk already
    // resolves. Override with `Extra` when `compile_payload` reads a file at a
    // path the walk would miss, or with `Only` when the walk over-approximates
    // (see `SourceFiles`).
    //
    // Return only paths that exist on disk. The cache silently drops paths it
    // can't read, so returning a placeholder is safe but pointless.
    fn source_files(_args: &serde_json::Value, _ctx: &BuildCtx<'_>) -> SourceFiles {
        SourceFiles::Extra(Vec::new())
    }
}

// An asset's contribution to its payload cache key: the inputs its compile
// reads, plus whether the compile target affects the output. Paired so the
// dispatch that builds one builds the other, keeping a new platform-dependent
// asset from silently inheriting `TARGET_DEPENDENT = false`.
pub struct CacheInputs {
    pub sources: SourceFiles,
    pub target_dependent: bool,
}

impl CacheInputs {
    // Inputs for an asset whose compile is platform-independent and whose
    // source paths the cache's generic args walk resolves on its own.
    pub fn extra(paths: Vec<String>) -> Self {
        Self {
            sources: SourceFiles::Extra(paths),
            target_dependent: false,
        }
    }
}
