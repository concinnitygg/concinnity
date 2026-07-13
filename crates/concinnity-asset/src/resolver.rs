// Name -> id resolution seam.
//
// A reference deserializes either from an already-resolved integer id (the
// compiled-args / runtime form) or from a name string (the authoring form).
// Turning a name into a dense id is engine policy -- the build assigns ids in
// world declaration order -- so this data crate does not own it. concinnity-core
// installs a resolver here, backed by its build-time interner, before it
// deserializes named references. A name seen with no resolver installed is a
// configuration error, surfaced as a deserialization failure (the resolver is
// always installed during a build; only an out-of-engine tool reading authoring
// JSON would hit the unset case).
//
// The resolver is a plain function pointer held in an atomic, so this stays
// `no_std` and thread-safe: the pointer is written once (install) and only read
// afterward, and the installed function keeps its own (per-thread) state in
// concinnity-core.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A name -> dense id resolver.
pub type ResolveFn = fn(&str) -> u32;

// 0 means "no resolver installed". Any other value is a `ResolveFn` address.
static RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> id resolver. Called once by concinnity-core, backed by
/// its build-time interner. Idempotent; the last writer wins.
pub fn set_name_resolver(f: ResolveFn) {
    RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve a name to a dense id via the installed resolver, or `None` if none is
/// installed (only expected outside a build).
pub(crate) fn resolve_name(name: &str) -> Option<u32> {
    let v = RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `ResolveFn` address stored by
        // `set_name_resolver`; the transmute reverses that exact `fn as usize`.
        let f: ResolveFn = unsafe { core::mem::transmute::<usize, ResolveFn>(v) };
        Some(f(name))
    }
}

/// A name -> per-kind resource-handle resolver. Returns the resource's dense
/// handle, or `None` when the name is not a known resource of that kind in the
/// current build (or no build map is installed). Unlike the name interner a
/// handle is not assignable on demand: it is a position in the build's
/// declaration-ordered resource table, so a name with no matching resource has
/// no handle.
pub type HandleResolveFn = fn(&str) -> Option<u32>;

// 0 means "no resolver installed". Any other value is a `HandleResolveFn`.
static TEXTURE_HANDLE_RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> texture-handle resolver. Called by concinnity-cook,
/// backed by the current build's declaration-ordered texture handle map.
/// Idempotent; the last writer wins.
pub fn set_texture_handle_resolver(f: HandleResolveFn) {
    TEXTURE_HANDLE_RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve a texture reference name to its dense `TextureHandle` value via the
/// installed resolver. `None` means either no resolver is installed or the name
/// is not a declared texture; the caller decides whether to fall back (a
/// validation context) or to fail (a real build).
pub(crate) fn resolve_texture_handle(name: &str) -> Option<u32> {
    let v = TEXTURE_HANDLE_RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address
        // stored by `set_texture_handle_resolver`; the transmute reverses that
        // exact `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(name)
    }
}

// 0 means "no resolver installed". Any other value is a `HandleResolveFn`.
static AUDIO_CLIP_HANDLE_RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> audio-clip-handle resolver. Called by concinnity-cook,
/// backed by the current build's declaration-ordered audio-clip handle map.
/// Idempotent; the last writer wins. Mirrors [`set_texture_handle_resolver`].
pub fn set_audio_clip_handle_resolver(f: HandleResolveFn) {
    AUDIO_CLIP_HANDLE_RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve an audio-clip reference name to its dense `AudioClipHandle` value via
/// the installed resolver. `None` means either no resolver is installed or the
/// name is not a declared audio clip; the caller decides whether to fall back
/// (a validation context) or to fail (a real build).
pub(crate) fn resolve_audio_clip_handle(name: &str) -> Option<u32> {
    let v = AUDIO_CLIP_HANDLE_RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address
        // stored by `set_audio_clip_handle_resolver`; the transmute reverses that
        // exact `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(name)
    }
}

// 0 means "no resolver installed". Any other value is a `HandleResolveFn`.
static FONT_HANDLE_RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> font-handle resolver. Called by concinnity-cook, backed by
/// the current build's declaration-ordered font handle map. Idempotent; the last
/// writer wins. Mirrors [`set_texture_handle_resolver`].
pub fn set_font_handle_resolver(f: HandleResolveFn) {
    FONT_HANDLE_RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve a font reference name to its dense `FontHandle` value via the installed
/// resolver. `None` means either no resolver is installed or the name is not a
/// declared font; the caller decides whether to fall back (a validation context)
/// or to fail (a real build).
pub(crate) fn resolve_font_handle(name: &str) -> Option<u32> {
    let v = FONT_HANDLE_RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address stored
        // by `set_font_handle_resolver`; the transmute reverses that exact
        // `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(name)
    }
}

// 0 means "no resolver installed". Any other value is a `HandleResolveFn`.
static MESH_HANDLE_RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> mesh-handle resolver. Called by concinnity-cook, backed by
/// the current build's mesh-source handle map. Idempotent; the last writer wins.
/// Mirrors [`set_texture_handle_resolver`]. The mesh-source handle space is shared
/// across every geometry-producing kind (Mesh, ProceduralMesh, VoxelChunk, and
/// mesh-kind File), so one resolver serves them all.
pub fn set_mesh_handle_resolver(f: HandleResolveFn) {
    MESH_HANDLE_RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve a mesh reference name to its dense `MeshHandle` value via the installed
/// resolver. `None` means either no resolver is installed or the name is not a
/// declared mesh source; the caller decides whether to fall back (a validation
/// context) or to fail (a real build).
pub(crate) fn resolve_mesh_handle(name: &str) -> Option<u32> {
    let v = MESH_HANDLE_RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address stored
        // by `set_mesh_handle_resolver`; the transmute reverses that exact
        // `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(name)
    }
}

// 0 means "no resolver installed". Any other value is a `HandleResolveFn`.
static MATERIAL_HANDLE_RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> material-handle resolver. Called by concinnity-cook, backed
/// by the current build's Material handle map. Idempotent; the last writer wins.
/// Mirrors [`set_texture_handle_resolver`].
pub fn set_material_handle_resolver(f: HandleResolveFn) {
    MATERIAL_HANDLE_RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve a material reference name to its dense `MaterialHandle` value via the
/// installed resolver. `None` means either no resolver is installed or the name is
/// not a declared material; the caller decides whether to fall back (a validation
/// context) or to fail (a real build).
pub(crate) fn resolve_material_handle(name: &str) -> Option<u32> {
    let v = MATERIAL_HANDLE_RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address stored
        // by `set_material_handle_resolver`; the transmute reverses that exact
        // `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(name)
    }
}

// 0 means "no resolver installed". Any other value is a `HandleResolveFn`.
static SKINNED_MESH_HANDLE_RESOLVER: AtomicUsize = AtomicUsize::new(0);

/// Install the name -> skinned-mesh-handle resolver. Called by concinnity-cook,
/// backed by the current build's SkinnedMesh handle map. Idempotent; the last
/// writer wins. Mirrors [`set_texture_handle_resolver`]. A SkinnedMesh stays an
/// ECS component, but its authored references (`Animation.target`,
/// `AnimGraph.target`, `FollowController.target`) resolve to its dense handle so
/// they no longer carry an interned id.
pub fn set_skinned_mesh_handle_resolver(f: HandleResolveFn) {
    SKINNED_MESH_HANDLE_RESOLVER.store(f as usize, Ordering::Release);
}

/// Resolve a skinned-mesh reference name to its dense `SkinnedMeshHandle` value via
/// the installed resolver. `None` means either no resolver is installed or the name
/// is not a declared SkinnedMesh; the caller decides whether to fall back (a
/// validation context) or to fail (a real build).
pub(crate) fn resolve_skinned_mesh_handle(name: &str) -> Option<u32> {
    let v = SKINNED_MESH_HANDLE_RESOLVER.load(Ordering::Acquire);
    if v == 0 {
        None
    } else {
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address stored
        // by `set_skinned_mesh_handle_resolver`; the transmute reverses that exact
        // `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(name)
    }
}

#[cfg(test)]
mod tests {
    // These tests own the process-global resolver: each installs the same
    // deterministic stand-in first, so they stay correct regardless of the order
    // the test harness runs them in (installs are idempotent, last-writer-wins).
    use super::*;
    use crate::{AssetId, AssetRef, de_opt_asset_ref, de_opt_asset_ref_typed};

    // A name resolves to its byte length: a simple, order-independent mapping.
    fn len_resolver(name: &str) -> u32 {
        name.len() as u32
    }

    struct Clip;

    #[test]
    fn installed_resolver_is_used() {
        set_name_resolver(len_resolver);
        assert_eq!(resolve_name("abcd"), Some(4));
    }

    #[test]
    fn asset_id_resolves_a_name_through_the_seam() {
        set_name_resolver(len_resolver);
        let id: AssetId = serde_json::from_str("\"floor\"").unwrap();
        assert_eq!(id, AssetId(5));
    }

    #[test]
    fn asset_ref_resolves_a_name_through_the_seam() {
        set_name_resolver(len_resolver);
        let r: AssetRef<Clip> = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(r.id(), Some(AssetId(4)));
        assert!(r.is_resolved());
    }

    #[test]
    fn opt_helpers_resolve_a_name_and_pass_through_an_id() {
        set_name_resolver(len_resolver);

        #[derive(serde::Deserialize)]
        struct Bare {
            #[serde(default, deserialize_with = "de_opt_asset_ref")]
            r: Option<AssetId>,
        }
        #[derive(serde::Deserialize)]
        struct Typed {
            #[serde(default, deserialize_with = "de_opt_asset_ref_typed")]
            r: Option<AssetRef<Clip>>,
        }

        assert_eq!(
            serde_json::from_str::<Bare>("{\"r\":\"mesh_a\"}")
                .unwrap()
                .r,
            Some(AssetId(6))
        );
        assert_eq!(
            serde_json::from_str::<Typed>("{\"r\":\"abc\"}")
                .unwrap()
                .r
                .unwrap()
                .id(),
            Some(AssetId(3))
        );
    }
}
