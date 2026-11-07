//! `BuildAsset` turns an asset's args into its binary payload (mesh vertices,
//! shader bytecode, a decoded image, ...). Every registered type the registry
//! marks `compiled` or `data`, component or resource, implements it, and the
//! registry's generated `RegisteredType::build_asset` is the one table the
//! build pipeline reaches each impl through, so a compiled type without an impl
//! does not build.
//!
//! `BuildCtx` is the build-time context handed to each impl. It lives here
//! because it is build-only: the runtime never compiles a payload. It carries
//! the shader `Platform` the build cooks for; the enum itself stays in
//! concinnity-engine, since the engine selects a Shader stage's source with it at
//! runtime.

pub mod api;
mod impls;

use concinnity_core::platform::PlatformSet;
use std::path::Path;

use crate::authoring::world::WorldJsonlAsset;

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
pub(crate) enum SourceFiles {
    Extra(Vec<std::path::PathBuf>),
    Only(Vec<std::path::PathBuf>),
}

// Build-time context handed to each `BuildAsset` impl.
pub(crate) struct BuildCtx<'a> {
    // The asset's declared name (used in error messages and as a key for
    // build-time intermediates such as compiled shader filenames).
    pub name: &'a str,
    // The shader platforms this build cooks for: a shader-backed asset compiles
    // its programs for each, side by side in one payload.
    pub(crate) targets: PlatformSet,
    // The build's asset search root: the tree a bare `source` filename is
    // searched under. `None` leaves bare filenames unresolved.
    pub(crate) assets_dir: Option<&'a Path>,
    // All sibling assets declared in the same world. Used by types like
    // `VoxelChunk` that need to resolve cross-asset references (palette).
    pub(crate) all_assets: &'a [WorldJsonlAsset],
}

// An asset that compiles to a binary payload at build time.
pub(crate) trait BuildAsset {
    fn compile_payload(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> std::io::Result<Vec<u8>>;

    // True when identical source bytes compile to a different payload per
    // backend, so the compile target is itself an input to the payload and
    // belongs in the cache key. `Shader` and `SdfVolume` set this: one `.hlsl`
    // file compiles to MSL text on one backend, DXIL on another and SPIR-V on
    // the third.
    //
    // Assets that transport their source verbatim leave this false: identical
    // bytes yield an identical payload and two backends may correctly share one
    // cache entry.
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

    // The args of the other assets this asset's `compile_payload` reads, such
    // as a texture it decodes. The cache key folds each in, with the files its
    // args name, so editing one invalidates this asset's cached payload too.
    fn dependency_args(_args: &serde_json::Value, _ctx: &BuildCtx<'_>) -> Vec<serde_json::Value> {
        Vec::new()
    }

    // The args the payload cache key hashes: all of them, unless the payload
    // never reads some fields, which then key on the bake inputs alone so
    // editing one reuses the payload.
    fn payload_key_args(args: &serde_json::Value) -> std::borrow::Cow<'_, serde_json::Value> {
        std::borrow::Cow::Borrowed(args)
    }

    // Runtime data baked beside the payload into the record's own bytes, for a
    // resource whose record carries fields the payload does not. `name` is the
    // asset's declared name.
    fn compile_data(_name: &str, _args: &serde_json::Value) -> std::io::Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

type CompileFn = fn(&serde_json::Value, &BuildCtx<'_>) -> std::io::Result<Vec<u8>>;
type CacheInputsFn = fn(&serde_json::Value, &BuildCtx<'_>) -> CacheInputs;
type KeyArgsFn = for<'a> fn(&'a serde_json::Value) -> std::borrow::Cow<'a, serde_json::Value>;
type DataFn = fn(&str, &serde_json::Value) -> std::io::Result<Option<Vec<u8>>>;

// One type's `BuildAsset` impl, as the registry's dispatch table holds it.
#[derive(Clone, Copy)]
pub(crate) struct BuildAssetEntry {
    pub(crate) compile: CompileFn,
    pub(crate) cache_inputs: CacheInputsFn,
    pub(crate) payload_key_args: KeyArgsFn,
    pub(crate) compile_data: DataFn,
}

impl BuildAssetEntry {
    pub(crate) fn of<T: BuildAsset>() -> Self {
        Self {
            compile: T::compile_payload,
            cache_inputs: cache_inputs_of::<T>,
            payload_key_args: T::payload_key_args,
            compile_data: T::compile_data,
        }
    }
}

// `source_files` and `TARGET_DEPENDENT` are read together, so a type whose
// payload differs per backend cannot report one without the other.
fn cache_inputs_of<T: BuildAsset>(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> CacheInputs {
    CacheInputs {
        sources: T::source_files(args, ctx),
        target_dependent: T::TARGET_DEPENDENT,
        dependencies: T::dependency_args(args, ctx),
    }
}

// An asset's contribution to its payload cache key: the inputs its compile
// reads, plus whether the compile target affects the output. Paired so the
// dispatch that builds one builds the other, keeping a new platform-dependent
// asset from silently inheriting `TARGET_DEPENDENT = false`.
pub(crate) struct CacheInputs {
    pub sources: SourceFiles,
    pub(crate) target_dependent: bool,
    // The args of the other assets the compile reads.
    pub(crate) dependencies: Vec<serde_json::Value>,
}

#[cfg(test)]
impl CacheInputs {
    // Inputs for an asset whose compile is platform-independent and whose
    // source paths the cache's generic args walk resolves on its own.
    pub(crate) fn extra(paths: Vec<std::path::PathBuf>) -> Self {
        Self {
            sources: SourceFiles::Extra(paths),
            target_dependent: false,
            dependencies: Vec::new(),
        }
    }
}

// A type's dispatch-table entry, reached the way the build reaches it, for the
// tests that compile one asset with no surrounding world.
#[cfg(test)]
impl crate::authoring::registry::RegisteredType {
    fn test_ctx(assets_dir: Option<&Path>) -> BuildCtx<'_> {
        BuildCtx {
            name: "test",
            targets: PlatformSet::single(concinnity_core::platform::Platform::Metal),
            assets_dir,
            all_assets: &[],
        }
    }

    fn entry(self) -> BuildAssetEntry {
        self.build_asset()
            .unwrap_or_else(|| panic!("{} compiles no payload", self.as_str()))
    }

    pub(crate) fn compile_payload(
        self,
        args: &serde_json::Value,
        assets_dir: Option<&Path>,
    ) -> std::io::Result<Vec<u8>> {
        (self.entry().compile)(args, &Self::test_ctx(assets_dir))
    }

    pub(crate) fn source_files(
        self,
        args: &serde_json::Value,
        assets_dir: Option<&Path>,
    ) -> Vec<std::path::PathBuf> {
        match (self.entry().cache_inputs)(args, &Self::test_ctx(assets_dir)).sources {
            SourceFiles::Extra(paths) | SourceFiles::Only(paths) => paths,
        }
    }

    pub(crate) fn payload_key_args(
        self,
        args: &serde_json::Value,
    ) -> std::borrow::Cow<'_, serde_json::Value> {
        (self.entry().payload_key_args)(args)
    }

    pub(crate) fn compile_data(
        self,
        name: &str,
        args: &serde_json::Value,
    ) -> std::io::Result<Option<Vec<u8>>> {
        (self.entry().compile_data)(name, args)
    }
}
