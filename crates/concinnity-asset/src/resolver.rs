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
// Each resolver is a plain function pointer held in an atomic, so this stays
// `no_std` and thread-safe: the pointer is written once (install) and only read
// afterward, and the installed function keeps its own (per-thread) state in
// concinnity-core. The two slot types below centralize the single unavoidable
// piece of unsafe -- `core` has no atomic function-pointer type, so reading a
// `fn` back out of a `usize` requires a `transmute` -- into one audited place
// per function-pointer shape.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A name -> dense id resolver.
pub type ResolveFn = fn(&str) -> u32;

/// A name -> per-kind resource-handle resolver. Returns the resource's dense
/// handle, or `None` when the name is not a known resource of that kind in the
/// current build (or no build map is installed). Unlike the name interner a
/// handle is not assignable on demand: it is a position in the build's
/// declaration-ordered resource table, so a name with no matching resource has
/// no handle.
pub type HandleResolveFn = fn(&str) -> Option<u32>;

// An atomically-installable `ResolveFn` slot. Holds the function pointer as a
// `usize` (0 = unset): written once at install, only read afterward.
struct NameResolverSlot(AtomicUsize);

impl NameResolverSlot {
    const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    fn set(&self, f: ResolveFn) {
        self.0.store(f as usize, Ordering::Release);
    }

    fn resolve(&self, name: &str) -> Option<u32> {
        let v = self.0.load(Ordering::Acquire);
        if v == 0 {
            return None;
        }
        // SAFETY: `v` is non-zero here, so it is a `ResolveFn` address stored by
        // `set`; the transmute reverses that exact `fn as usize`.
        let f: ResolveFn = unsafe { core::mem::transmute::<usize, ResolveFn>(v) };
        Some(f(name))
    }
}

// An atomically-installable `HandleResolveFn` slot. Same install-once /
// read-many discipline as `NameResolverSlot`; one instance backs each per-kind
// handle resolver.
struct HandleResolverSlot(AtomicUsize);

impl HandleResolverSlot {
    const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    fn set(&self, f: HandleResolveFn) {
        self.0.store(f as usize, Ordering::Release);
    }

    fn resolve(&self, name: &str) -> Option<u32> {
        let v = self.0.load(Ordering::Acquire);
        if v == 0 {
            return None;
        }
        // SAFETY: `v` is non-zero here, so it is a `HandleResolveFn` address
        // stored by `set`; the transmute reverses that exact `fn as usize`.
        let f: HandleResolveFn = unsafe { core::mem::transmute::<usize, HandleResolveFn>(v) };
        f(name)
    }
}

static RESOLVER: NameResolverSlot = NameResolverSlot::new();

/// Install the name -> id resolver. Called once by concinnity-core, backed by
/// its build-time interner. Idempotent; the last writer wins.
pub fn set_name_resolver(f: ResolveFn) {
    RESOLVER.set(f);
}

/// Resolve a name to a dense id via the installed resolver, or `None` if none is
/// installed (only expected outside a build).
pub(crate) fn resolve_name(name: &str) -> Option<u32> {
    RESOLVER.resolve(name)
}

static TEXTURE_HANDLE_RESOLVER: HandleResolverSlot = HandleResolverSlot::new();

/// Install the name -> texture-handle resolver. Called by concinnity-cook,
/// backed by the current build's declaration-ordered texture handle map.
/// Idempotent; the last writer wins.
pub fn set_texture_handle_resolver(f: HandleResolveFn) {
    TEXTURE_HANDLE_RESOLVER.set(f);
}

/// Resolve a texture reference name to its dense `TextureHandle` value via the
/// installed resolver. `None` means either no resolver is installed or the name
/// is not a declared texture; the caller decides whether to fall back (a
/// validation context) or to fail (a real build).
pub(crate) fn resolve_texture_handle(name: &str) -> Option<u32> {
    TEXTURE_HANDLE_RESOLVER.resolve(name)
}

static AUDIO_CLIP_HANDLE_RESOLVER: HandleResolverSlot = HandleResolverSlot::new();

/// Install the name -> audio-clip-handle resolver. Called by concinnity-cook,
/// backed by the current build's declaration-ordered audio-clip handle map.
/// Idempotent; the last writer wins. Mirrors [`set_texture_handle_resolver`].
pub fn set_audio_clip_handle_resolver(f: HandleResolveFn) {
    AUDIO_CLIP_HANDLE_RESOLVER.set(f);
}

/// Resolve an audio-clip reference name to its dense `AudioClipHandle` value via
/// the installed resolver. `None` means either no resolver is installed or the
/// name is not a declared audio clip; the caller decides whether to fall back
/// (a validation context) or to fail (a real build).
pub(crate) fn resolve_audio_clip_handle(name: &str) -> Option<u32> {
    AUDIO_CLIP_HANDLE_RESOLVER.resolve(name)
}

static FONT_HANDLE_RESOLVER: HandleResolverSlot = HandleResolverSlot::new();

/// Install the name -> font-handle resolver. Called by concinnity-cook, backed by
/// the current build's declaration-ordered font handle map. Idempotent; the last
/// writer wins. Mirrors [`set_texture_handle_resolver`].
pub fn set_font_handle_resolver(f: HandleResolveFn) {
    FONT_HANDLE_RESOLVER.set(f);
}

/// Resolve a font reference name to its dense `FontHandle` value via the installed
/// resolver. `None` means either no resolver is installed or the name is not a
/// declared font; the caller decides whether to fall back (a validation context)
/// or to fail (a real build).
pub(crate) fn resolve_font_handle(name: &str) -> Option<u32> {
    FONT_HANDLE_RESOLVER.resolve(name)
}

static MESH_HANDLE_RESOLVER: HandleResolverSlot = HandleResolverSlot::new();

/// Install the name -> mesh-handle resolver. Called by concinnity-cook, backed by
/// the current build's mesh-source handle map. Idempotent; the last writer wins.
/// Mirrors [`set_texture_handle_resolver`]. The mesh-source handle space is shared
/// across every geometry-producing kind (Mesh, ProceduralMesh, VoxelChunk, and
/// mesh-kind File), so one resolver serves them all.
pub fn set_mesh_handle_resolver(f: HandleResolveFn) {
    MESH_HANDLE_RESOLVER.set(f);
}

/// Resolve a mesh reference name to its dense `MeshHandle` value via the installed
/// resolver. `None` means either no resolver is installed or the name is not a
/// declared mesh source; the caller decides whether to fall back (a validation
/// context) or to fail (a real build).
pub(crate) fn resolve_mesh_handle(name: &str) -> Option<u32> {
    MESH_HANDLE_RESOLVER.resolve(name)
}

static MATERIAL_HANDLE_RESOLVER: HandleResolverSlot = HandleResolverSlot::new();

/// Install the name -> material-handle resolver. Called by concinnity-cook, backed
/// by the current build's Material handle map. Idempotent; the last writer wins.
/// Mirrors [`set_texture_handle_resolver`].
pub fn set_material_handle_resolver(f: HandleResolveFn) {
    MATERIAL_HANDLE_RESOLVER.set(f);
}

/// Resolve a material reference name to its dense `MaterialHandle` value via the
/// installed resolver. `None` means either no resolver is installed or the name is
/// not a declared material; the caller decides whether to fall back (a validation
/// context) or to fail (a real build).
pub(crate) fn resolve_material_handle(name: &str) -> Option<u32> {
    MATERIAL_HANDLE_RESOLVER.resolve(name)
}

static SKINNED_MESH_HANDLE_RESOLVER: HandleResolverSlot = HandleResolverSlot::new();

/// Install the name -> skinned-mesh-handle resolver. Called by concinnity-cook,
/// backed by the current build's SkinnedMesh handle map. Idempotent; the last
/// writer wins. Mirrors [`set_texture_handle_resolver`]. A SkinnedMesh stays an
/// ECS component, but its authored references (`Animation.target`,
/// `AnimGraph.target`, `FollowController.target`) resolve to its dense handle so
/// they no longer carry an interned id.
pub fn set_skinned_mesh_handle_resolver(f: HandleResolveFn) {
    SKINNED_MESH_HANDLE_RESOLVER.set(f);
}

/// Resolve a skinned-mesh reference name to its dense `SkinnedMeshHandle` value via
/// the installed resolver. `None` means either no resolver is installed or the name
/// is not a declared SkinnedMesh; the caller decides whether to fall back (a
/// validation context) or to fail (a real build).
pub(crate) fn resolve_skinned_mesh_handle(name: &str) -> Option<u32> {
    SKINNED_MESH_HANDLE_RESOLVER.resolve(name)
}

static SHADER_HANDLE_RESOLVER: HandleResolverSlot = HandleResolverSlot::new();

/// Install the name -> shader-handle resolver. Called by concinnity-cook,
/// backed by the current build's declaration-ordered Shader handle map.
/// Idempotent; the last writer wins. Mirrors [`set_texture_handle_resolver`].
/// A Shader stays an ECS component, but a Material's authored `shader`
/// reference resolves to its dense handle so the runtime never scans by name.
pub fn set_shader_handle_resolver(f: HandleResolveFn) {
    SHADER_HANDLE_RESOLVER.set(f);
}

/// Resolve a shader reference name to its dense `ShaderHandle` value via the
/// installed resolver. `None` means either no resolver is installed or the name
/// is not a declared Shader; the caller decides whether to fall back (a
/// validation context) or to fail (a real build).
pub(crate) fn resolve_shader_handle(name: &str) -> Option<u32> {
    SHADER_HANDLE_RESOLVER.resolve(name)
}

#[cfg(test)]
mod tests {
    // These tests own the process-global resolver: each installs the same
    // deterministic stand-in first, so they stay correct regardless of the order
    // the test harness runs them in (installs are idempotent, last-writer-wins).
    use super::*;
    use crate::test_support::{install_resolvers, len_handle_resolver, len_name_resolver};
    use crate::{AssetId, AssetRef, de_opt_asset_ref, de_opt_asset_ref_typed};

    struct Clip;

    #[test]
    fn a_slot_reads_back_the_function_pointer_it_was_given() {
        // The slots hold their function pointer as a `usize` and transmute it
        // back, the one piece of unsafe here. Exercising a fresh slot rather
        // than the process-global statics is the only way to see the unset
        // state, which a test cannot restore once something has installed.
        let name_slot = NameResolverSlot::new();
        assert_eq!(name_slot.resolve("floor"), None);
        name_slot.set(len_name_resolver);
        assert_eq!(name_slot.resolve("floor"), Some(5));

        let handle_slot = HandleResolverSlot::new();
        assert_eq!(handle_slot.resolve("floor"), None);
        handle_slot.set(len_handle_resolver);
        assert_eq!(handle_slot.resolve("floor"), Some(5));
        // A handle resolver may also answer "no such resource of this kind",
        // which the name interner slot has no way to express.
        assert_eq!(handle_slot.resolve("unknown_x"), None);
    }

    #[test]
    fn installed_resolver_is_used() {
        set_name_resolver(len_name_resolver);
        assert_eq!(resolve_name("abcd"), Some(4));
    }

    #[test]
    fn asset_id_resolves_a_name_through_the_seam() {
        set_name_resolver(len_name_resolver);
        let id: AssetId = serde_json::from_str("\"floor\"").unwrap();
        assert_eq!(id, AssetId(5));
    }

    #[test]
    fn asset_ref_resolves_a_name_through_the_seam() {
        set_name_resolver(len_name_resolver);
        let r: AssetRef<Clip> = serde_json::from_str("\"wall\"").unwrap();
        assert_eq!(r.id(), Some(AssetId(4)));
        assert!(r.is_resolved());
    }

    #[test]
    fn every_handle_seam_resolves_through_its_own_slot() {
        // One slot per kind: a name is a position in that kind's declaration-
        // ordered table, so the kinds never share an answer by accident.
        install_resolvers();
        set_texture_handle_resolver(len_handle_resolver);
        set_audio_clip_handle_resolver(len_handle_resolver);
        set_font_handle_resolver(len_handle_resolver);
        set_mesh_handle_resolver(len_handle_resolver);
        set_material_handle_resolver(len_handle_resolver);
        set_skinned_mesh_handle_resolver(len_handle_resolver);
        set_shader_handle_resolver(len_handle_resolver);

        assert_eq!(resolve_texture_handle("floor"), Some(5));
        assert_eq!(resolve_audio_clip_handle("floor"), Some(5));
        assert_eq!(resolve_font_handle("floor"), Some(5));
        assert_eq!(resolve_mesh_handle("floor"), Some(5));
        assert_eq!(resolve_material_handle("floor"), Some(5));
        assert_eq!(resolve_skinned_mesh_handle("floor"), Some(5));
        assert_eq!(resolve_shader_handle("floor"), Some(5));

        // A handle is not assignable on demand: a name the build declares no
        // resource of that kind for has none, even with a resolver installed.
        assert_eq!(resolve_texture_handle("unknown_x"), None);
        assert_eq!(resolve_shader_handle("unknown_x"), None);
    }

    #[test]
    fn opt_helpers_resolve_a_name_and_pass_through_an_id() {
        set_name_resolver(len_name_resolver);

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
