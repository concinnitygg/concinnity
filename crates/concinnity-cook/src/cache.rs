// Content-addressed cache for compiled asset payloads.
//
// Some assets are expensive to compile -- the EnvironmentMap IBL convolution
// alone is hundreds of millions of float ops per build. The compiled payload
// is, however, a deterministic function of a small set of inputs: the cache
// format version, the component discriminant, the asset's args JSON, and the
// contents of any source files the args reference. This module hashes those
// inputs into a key and stores the compiled bytes under `.concinnity/cache/`.
// A later build that produces the same key reuses the cached payload instead
// of recompiling.
//
// Every operation here is best-effort: a cache miss, a read error, or a write
// error all fall back to a normal compile, so the cache can never break or
// corrupt a build.

use crate::asset::{BuildCtx, CacheInputs, SourceFiles, SourceInput};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

// SHA-256 a source file's contents, memoized by path within the process. A
// single build can reference one large source file from hundreds of assets
// (e.g. every Mesh imported from one `.fbx`), and hashing it once per asset
// dominates the build; the memo reads + hashes each unique file once. Keyed by
// (mtime, len) so a file edited between in-process rebuilds (the `cn debug`
// hot-reload path) is re-hashed rather than served stale.
fn file_content_hash(path: &str) -> Option<[u8; 32]> {
    // path -> (mtime_nanos, len, content hash)
    type HashMemo = Mutex<HashMap<String, (u64, u64, [u8; 32])>>;
    static MEMO: OnceLock<HashMemo> = OnceLock::new();
    let meta = std::fs::metadata(path).ok()?;
    let len = meta.len();
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&(m, l, h)) = memo.lock().unwrap().get(path)
        && m == mtime
        && l == len
    {
        return Some(h);
    }
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hash: [u8; 32] = hasher.finalize().into();
    memo.lock()
        .unwrap()
        .insert(path.to_string(), (mtime, len, hash));
    Some(hash)
}

// Bump this whenever a compile path's output changes without a corresponding
// change to asset args -- e.g. a convolution algorithm tweak, a payload format
// revision, or a change to a default sample count. A bump changes every key
// and so invalidates every existing cache entry.
//
// 4: font payload gained a supersample factor in its header (build::font).
// 5: EnvironmentMap default irradiance_face_size changed 32 -> 8
//    (build::environment_map), so worlds that omit it bake a different cube.
//    (The counter was later reset to 1 with the postcard/blob migration.)
// 2: EnvironmentMap glossy reflection mips gained a firefly clamp
//    (prefilter_clamp, default 12); worlds that omit the arg still bake dimmer
//    hot texels, so every cached envmap must rebake (build::environment_map).
// 3: Font payload header gained the rasterisation size (size_px) after
//    supersample, shifting the atlas offset 12 -> 16 (build::font).
// 4: baked resource data (`Material` data_bytes, the SkinnedMesh data tuple)
//    switched JSON -> postcard alongside BLOB_VERSION 3; cached JSON bytes
//    must not be replayed into a postcard blob.
// 6: the SKMV skinned payload gained the optional MRPH morph-target block, so
//    a mesh whose source carries morph targets compiles different bytes from
//    unchanged args.
// 7: every 2D texture payload switched to the tagged format (magic + format_id
//    + per-mip records) so KTX2 / DDS can ship block-compressed mip chains; the
//    old headerless RGBA8 bytes no longer parse (build::texture).
// 8: the Material data resource gained `alpha_cutoff`, so its postcard bytes
//    grew a field and cached records from before it no longer decode.
// 9: BC5 sources decode to RGBA8 instead of shipping blocks (the shaders need a
//    reconstructed Z in blue), and a block-compressed chain now honours
//    `max_size` by dropping leading mips (build::texture).
// 10: BC5 sources ship their blocks again now that the shaders reconstruct a
//     normal map's Z from X and Y, so cached RGBA8 payloads must recompile
//     (build::texture).
const CACHE_FORMAT_VERSION: u32 = 10;

// Compute the cache key for one compiled asset. The key folds in the cache
// format version, the component discriminant, the args JSON, a hash of every
// input the compile reads, and -- for assets that compile rather than
// transport their source -- the active backend's shader platform.
//
// The key is content-addressed with no namespacing prefix: an asset that
// compiles identically on every backend (a mesh, a texture, a font) produces
// one entry shared by a DirectX and a Vulkan cook rather than a copy each.
// Assets whose payload really does differ per backend separate themselves
// through their inputs -- a differing source file, or the compile target when
// `CacheInputs::target_dependent` is set.
pub fn payload_key(
    discriminant: u8,
    args: &serde_json::Value,
    ctx: &BuildCtx<'_>,
    inputs: &CacheInputs,
) -> String {
    let files = match &inputs.sources {
        SourceFiles::Extra(extra) => {
            let mut files = referenced_files(args, ctx);
            for path in extra {
                if let Some(h) = file_content_hash(path) {
                    files.push((path.clone(), h));
                }
            }
            files
        }
        SourceFiles::Only(inputs) => inputs.iter().filter_map(hash_source_input).collect(),
    };
    let target = inputs
        .target_dependent
        .then(|| concinnity_core::build::Platform::current().key());
    key_from_parts(discriminant, args, &files, target)
}

// Hash one declared input into the (key, content-hash) pair `key_from_parts`
// consumes. Mirrors how `referenced_files` treats each kind, so an asset that
// reports its inputs explicitly hashes them the same way the generic walk
// would have.
fn hash_source_input(input: &SourceInput) -> Option<(String, [u8; 32])> {
    match input {
        SourceInput::Path(path) => file_content_hash(path).map(|h| (path.clone(), h)),
        SourceInput::Builtin(name) => {
            let src = concinnity_core::build::shader::builtin_shader_source(name)?;
            let bare = Path::new(name)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(name);
            let mut h = Sha256::new();
            h.update(src.as_bytes());
            Some((format!("builtin:{bare}"), h.finalize().into()))
        }
    }
}

// Bump when the SceneImport expansion output shape changes (a new generated
// asset field, a renamed arg, a different naming scheme) so existing cached
// entry lists are invalidated. v2: glass materials are detected (by FBX
// transparency / name) and emitted smooth + translucent. v3: a skinned node
// expands to SkinnedMesh + Animation entries instead of a static
// Mesh / Model / Prop. v4: glTF materials carry their packed
// metallic-roughness + emissive textures and an alpha-cutout threshold.
const EXPAND_FORMAT_VERSION: u32 = 4;

// Cache key for a SceneImport expansion. The generated asset-entry list is a
// deterministic function of the source file's contents, the import options,
// and the expansion format version, so editing the source file or changing an
// option busts the entry. Like `payload_key` this is platform-independent: the
// entries are plain JSON with no per-backend branching. `load` / `store` are
// shared with the payload cache (same `.concinnity/cache/` directory); the two
// key spaces stay distinct because they hash structurally different inputs.
pub fn expand_key(source: &str, args: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(EXPAND_FORMAT_VERSION.to_le_bytes());
    if let Some(h) = file_content_hash(source) {
        hasher.update(h);
    }
    // A text `.gltf` pulls geometry and images from sibling files the source
    // hash alone cannot see; fold their contents in so editing a referenced
    // `.bin` or image re-expands the import.
    if source.to_lowercase().ends_with(".gltf") {
        for path in crate::gltf_source::referenced_files(source) {
            if let Some(h) = file_content_hash(&path) {
                hasher.update(h);
            }
        }
    }
    let args_bytes = serde_json::to_vec(args).unwrap_or_default();
    hasher.update((args_bytes.len() as u64).to_le_bytes());
    hasher.update(&args_bytes);
    format!("{:x}", hasher.finalize())
}

// Read a cached payload for `key`, if one is present.
pub fn load(key: &str) -> Option<Vec<u8>> {
    // Disabled under `cargo test` so the suite neither creates stray cache
    // directories nor lets a stale entry mask a change to a compile path.
    if cfg!(test) {
        return None;
    }
    std::fs::read(crate::paths::cache_dir().join(key)).ok()
}

// Store a compiled payload under `key`. Best-effort: any error is ignored.
// The bytes are written to a temp file and renamed into place so a concurrent
// reader never observes a half-written entry.
pub fn store(key: &str, bytes: &[u8]) {
    if cfg!(test) {
        return;
    }
    store_in(&crate::paths::cache_dir(), key, bytes);
}

// Hash the fixed parts of a key. Split out from `payload_key` so tests can
// supply file contents directly without touching the filesystem.
fn key_from_parts(
    discriminant: u8,
    args: &serde_json::Value,
    files: &[(String, [u8; 32])],
    target: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_FORMAT_VERSION.to_le_bytes());
    hasher.update([discriminant]);

    let args_bytes = serde_json::to_vec(args).unwrap_or_default();
    hasher.update((args_bytes.len() as u64).to_le_bytes());
    hasher.update(&args_bytes);

    // Absent and present-but-empty must not hash alike, so the discriminating
    // byte goes in either way.
    match target {
        Some(t) => {
            hasher.update([1u8]);
            hasher.update((t.len() as u64).to_le_bytes());
            hasher.update(t.as_bytes());
        }
        None => hasher.update([0u8]),
    }

    // Sort so the key does not depend on JSON traversal order.
    let mut files = files.to_vec();
    files.sort();
    for (path, content_hash) in &files {
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update(content_hash);
    }
    format!("{:x}", hasher.finalize())
}

// Collect (path, content-hash) for every source file the args reference.
// Walks the args JSON for string leaves and resolves each one to a file using
// the same lookup rules the asset compilers use: a bare filename is searched
// under `.concinnity/assets/`, a relative or absolute path is used directly,
// and `artifacts_dir` is consulted when set. Strings that do not resolve to a
// file (asset names, generator keywords, colors) contribute nothing.
//
// A string that names a built-in shader is a special case: the source is
// embedded in the binary rather than living at a filesystem path, and built-ins
// always win over a disk copy at compile time (see shader::read_shader_source).
// Such a string is hashed from its embedded source under a `builtin:` key so
// that editing a shipped shader and rebuilding the binary busts the cache.
fn referenced_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> Vec<(String, [u8; 32])> {
    let mut strings = Vec::new();
    collect_strings(args, &mut strings);

    let mut out = Vec::new();
    for s in strings {
        if let Some(src) = concinnity_core::build::shader::builtin_shader_source(&s) {
            let bare = Path::new(&s)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&s);
            let mut h = Sha256::new();
            h.update(src.as_bytes());
            out.push((format!("builtin:{bare}"), h.finalize().into()));
            continue;
        }
        let Some(path) = resolve_source(&s, ctx) else {
            continue;
        };
        if let Some(h) = file_content_hash(&path) {
            out.push((path, h));
        }
    }
    out
}

fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => a.iter().for_each(|e| collect_strings(e, out)),
        serde_json::Value::Object(m) => m.values().for_each(|e| collect_strings(e, out)),
        _ => {}
    }
}

// Resolve a single args string to an existing file path, or None. Only strings
// that look like filenames (have an extension or a path separator) are probed,
// so the common case of short keyword args costs nothing.
fn resolve_source(s: &str, ctx: &BuildCtx<'_>) -> Option<String> {
    let looks_like_file = s.contains('/') || s.contains('\\') || Path::new(s).extension().is_some();
    if !looks_like_file {
        return None;
    }
    // Direct path (absolute, or relative to the build working directory).
    if Path::new(s).is_file() {
        return Some(s.to_string());
    }
    // Bare filename searched recursively under .concinnity/assets/.
    if let Some(p) = crate::paths::find_in_assets(s) {
        return Some(p);
    }
    // Account artifact directory, when the build supplied one.
    if let Some(dir) = ctx.artifacts_dir {
        let p = format!("{dir}/{s}");
        if Path::new(&p).is_file() {
            return Some(p);
        }
    }
    None
}

fn store_in(dir: &Path, key: &str, bytes: &[u8]) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = dir.join(format!("{key}.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, dir.join(key));
    }
}

#[cfg(test)]
fn load_in(dir: &Path, key: &str) -> Option<Vec<u8>> {
    std::fs::read(dir.join(key)).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> BuildCtx<'static> {
        BuildCtx {
            name: "test",
            artifacts_dir: None,
            all_assets: &[],
        }
    }

    #[test]
    fn key_is_stable_for_same_inputs() {
        let a = json!({"generator": "box", "half_extents": [1, 2, 3]});
        assert_eq!(
            key_from_parts(7, &a, &[], None),
            key_from_parts(7, &a, &[], None)
        );
    }

    #[test]
    fn key_changes_with_args_discriminant_and_files() {
        let a = json!({"generator": "box"});
        let b = json!({"generator": "sphere"});
        let base = key_from_parts(1, &a, &[], None);
        assert_ne!(
            base,
            key_from_parts(1, &b, &[], None),
            "args must affect the key"
        );
        assert_ne!(
            base,
            key_from_parts(2, &a, &[], None),
            "discriminant must affect the key"
        );
        assert_ne!(
            base,
            key_from_parts(1, &a, &[("x.hdr".into(), [9u8; 32])], None),
            "a referenced file must affect the key"
        );
    }

    #[test]
    fn key_ignores_referenced_file_order() {
        let a = json!({});
        let f1 = ("a.hdr".to_string(), [1u8; 32]);
        let f2 = ("b.hdr".to_string(), [2u8; 32]);
        assert_eq!(
            key_from_parts(0, &a, &[f1.clone(), f2.clone()], None),
            key_from_parts(0, &a, &[f2, f1], None),
        );
    }

    // The compile target separates payloads that share every other input --
    // the case that made the old `hlsl-` / `glsl-` filename prefixes
    // load-bearing.
    #[test]
    fn key_changes_with_the_compile_target() {
        let a = json!({"sources": {"hlsl": "shared.inc", "glsl": "shared.inc"}});
        let hlsl = key_from_parts(1, &a, &[], Some("hlsl"));
        let glsl = key_from_parts(1, &a, &[], Some("glsl"));
        assert_ne!(hlsl, glsl, "the compile target must affect the key");
        assert_ne!(
            hlsl,
            key_from_parts(1, &a, &[], None),
            "a target-dependent key must differ from a target-independent one"
        );
        assert_ne!(
            key_from_parts(1, &a, &[], Some("")),
            key_from_parts(1, &a, &[], None),
            "an empty target must not hash as no target"
        );
    }

    #[test]
    fn key_tracks_referenced_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("env.hdr");
        std::fs::write(&file, b"first").unwrap();
        let args = json!({ "source": file.to_str().unwrap() });

        let before = payload_key(3, &args, &ctx(), &CacheInputs::extra(vec![]));
        std::fs::write(&file, b"second").unwrap();
        let after = payload_key(3, &args, &ctx(), &CacheInputs::extra(vec![]));
        assert_ne!(
            before, after,
            "key must change when a referenced file changes"
        );
    }

    #[test]
    fn key_tracks_extra_source_file_contents() {
        // Files whose paths the generic JSON-string walk can't resolve (e.g.
        // an SdfVolume `fragment_shader` resolved through the source-tree
        // `assets/` dir) must still bust the cache when their contents change.
        // The asset's `BuildAsset::source_files` override hands those paths to
        // `payload_key` via `extra_source_files`.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("shader.metal");
        std::fs::write(&file, b"void shade() {}").unwrap();
        let path = file.to_str().unwrap().to_string();
        // Args reference the file by a bare token the cache cannot resolve on
        // its own (no extension, no separator), so only `extra_source_files`
        // can contribute the content hash.
        let args = json!({ "fragment_shader": "chrome" });

        let before = payload_key(11, &args, &ctx(), &CacheInputs::extra(vec![path.clone()]));
        std::fs::write(&file, b"void shade(float) {}").unwrap();
        let after = payload_key(11, &args, &ctx(), &CacheInputs::extra(vec![path.clone()]));
        assert_ne!(
            before, after,
            "an extra source file's contents must affect the key"
        );
    }

    #[test]
    fn key_ignores_unreadable_extra_source_file() {
        // A path that does not exist is silently dropped (best-effort, matching
        // the rest of the cache layer). A missing file produces the same key
        // as no extra files at all.
        let args = json!({ "fragment_shader": "chrome" });
        let missing = "/definitely/not/a/real/path.metal".to_string();
        assert_eq!(
            payload_key(11, &args, &ctx(), &CacheInputs::extra(vec![])),
            payload_key(11, &args, &ctx(), &CacheInputs::extra(vec![missing])),
        );
    }

    #[test]
    fn non_file_strings_are_not_resolved() {
        // "box" has neither an extension nor a separator -> never probed.
        assert!(referenced_files(&json!({"generator": "box"}), &ctx()).is_empty());
    }

    #[test]
    fn builtin_shader_content_is_folded_into_key() {
        use concinnity_core::build::shader::builtin_shader_source;

        // A built-in shader referenced by bare filename has no filesystem path,
        // but its embedded source must still contribute to the key.
        let args = json!({ "sources": { "metal": "default.metal" } });
        let files = referenced_files(&args, &ctx());

        let src = builtin_shader_source("default.metal").expect("default.metal is built in");
        let mut h = Sha256::new();
        h.update(src.as_bytes());
        let expected: [u8; 32] = h.finalize().into();
        assert_eq!(
            files,
            vec![("builtin:default.metal".to_string(), expected)],
            "a built-in shader reference must contribute its embedded source hash",
        );

        // The key is a function of that hash, so any edit to the shader source
        // changes the key. A perturbed hash stands in for an edited shader.
        let real_key = key_from_parts(5, &args, &files, None);
        let edited = vec![("builtin:default.metal".to_string(), [0u8; 32])];
        assert_ne!(
            real_key,
            key_from_parts(5, &args, &edited, None),
            "editing a built-in shader source must change the key",
        );
    }

    // An asset reporting `Only` must hash a built-in exactly as the generic
    // walk would, so editing a shipped shader still busts its entry.
    #[test]
    fn only_inputs_hash_a_builtin_like_the_generic_walk() {
        let name = "default.metal";
        let walked = referenced_files(&json!(name), &ctx());
        let reported = hash_source_input(&SourceInput::Builtin(name.to_string()))
            .expect("default.metal is built in");
        assert_eq!(walked, vec![reported]);
        assert!(
            hash_source_input(&SourceInput::Builtin("not_a_builtin.metal".into())).is_none(),
            "an unregistered name contributes nothing"
        );
    }

    // `Only` replaces the generic args walk rather than adding to it: a path
    // sitting in the args that the asset does not report is not hashed. This
    // is what keeps an edit to the unused backend's shader from invalidating
    // this backend's payload -- and what makes each `Only` impl load-bearing.
    #[test]
    fn only_inputs_replace_the_generic_args_walk() {
        let dir = tempfile::tempdir().unwrap();
        let used = dir.path().join("used.hlsl");
        let unused = dir.path().join("unused.glsl");
        std::fs::write(&used, b"used").unwrap();
        std::fs::write(&unused, b"unused").unwrap();
        let args = json!({"sources": {
            "hlsl": used.to_str().unwrap(),
            "glsl": unused.to_str().unwrap(),
        }});
        let only = |inputs: &CacheInputs| payload_key(9, &args, &ctx(), inputs);
        let reported = CacheInputs {
            sources: SourceFiles::Only(vec![SourceInput::Path(used.to_str().unwrap().to_string())]),
            target_dependent: false,
        };

        let before = only(&reported);
        std::fs::write(&unused, b"edited").unwrap();
        assert_eq!(
            before,
            only(&reported),
            "editing an unreported file must not affect the key"
        );

        std::fs::write(&used, b"edited").unwrap();
        assert_ne!(
            before,
            only(&reported),
            "editing a reported file must affect the key"
        );

        // The same args under the generic walk do pick the unused file up:
        // `Only` is what narrows the input set, not the args themselves.
        assert_eq!(referenced_files(&args, &ctx()).len(), 2);
    }

    #[test]
    fn builtin_shader_directory_prefix_resolves_to_bare_key() {
        // Built-ins match by bare filename, so a leading directory must not
        // produce a distinct key entry.
        let bare = referenced_files(&json!("default.metal"), &ctx());
        let prefixed = referenced_files(&json!("default_shader/default.metal"), &ctx());
        assert_eq!(bare, prefixed);
        assert_eq!(bare.len(), 1);
    }

    #[test]
    fn expand_key_tracks_source_contents_and_args() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("scene.fbx");
        std::fs::write(&file, b"first").unwrap();
        let src = file.to_str().unwrap();
        let args = json!({ "prefix": "scn", "texture_max_size": 512 });

        let base = expand_key(src, &args);
        // Stable for identical inputs.
        assert_eq!(base, expand_key(src, &args));
        // Changing an option busts the key.
        assert_ne!(
            base,
            expand_key(src, &json!({ "prefix": "scn", "texture_max_size": 256 }))
        );
        // Editing the source file busts the key.
        std::fs::write(&file, b"second").unwrap();
        assert_ne!(base, expand_key(src, &args));
    }

    // A text `.gltf` SceneImport must re-expand when a referenced sibling file
    // changes, even though the `.gltf` itself is untouched.
    #[test]
    fn expand_key_tracks_gltf_sibling_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("geo.bin"), b"first").unwrap();
        let gltf = dir.path().join("scene.gltf");
        std::fs::write(
            &gltf,
            serde_json::to_vec(&json!({
                "asset": {"version": "2.0"},
                "buffers": [{"byteLength": 5, "uri": "geo.bin"}]
            }))
            .unwrap(),
        )
        .unwrap();
        let src = gltf.to_str().unwrap();
        let args = json!({ "prefix": "scn" });

        let before = expand_key(src, &args);
        assert_eq!(before, expand_key(src, &args));
        std::fs::write(dir.path().join("geo.bin"), b"second").unwrap();
        assert_ne!(
            before,
            expand_key(src, &args),
            "editing a referenced .bin must bust the expansion key"
        );
    }

    // Both key spaces share one directory and neither carries a prefix any
    // more, so they stay distinct purely by hashing different preimages.
    #[test]
    fn expansion_and_payload_key_spaces_stay_distinct() {
        let args = json!({ "prefix": "scn" });
        assert_ne!(
            expand_key("/no/such/scene.glb", &args),
            payload_key(0, &args, &ctx(), &CacheInputs::extra(vec![])),
        );
    }

    #[test]
    fn store_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        store_in(dir.path(), "abc123", b"payload bytes");
        assert_eq!(
            load_in(dir.path(), "abc123").as_deref(),
            Some(&b"payload bytes"[..])
        );
        assert_eq!(load_in(dir.path(), "missing"), None);
    }

    #[test]
    fn store_in_creates_the_directory_and_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("cache").join("deep");

        store_in(&nested, "k", b"one");
        assert_eq!(load_in(&nested, "k").as_deref(), Some(&b"one"[..]));

        // A second store for the same key replaces the entry in place.
        store_in(&nested, "k", b"two");
        assert_eq!(load_in(&nested, "k").as_deref(), Some(&b"two"[..]));
    }

    // Keys are bare content hashes. A platform-independent asset must produce
    // the same filename on every backend so one entry is shared across a
    // DirectX and a Vulkan cook instead of duplicated per backend.
    #[test]
    fn payload_keys_are_bare_hashes_with_no_prefix() {
        let key = payload_key(1, &json!({}), &ctx(), &CacheInputs::extra(vec![]));
        assert_eq!(
            key.len(),
            64,
            "key '{key}' must be a bare sha256 hex digest"
        );
        assert!(
            key.chars().all(|c| c.is_ascii_hexdigit()),
            "key '{key}' must contain no namespacing prefix"
        );
        assert_eq!(
            key,
            key_from_parts(1, &json!({}), &[], None),
            "a target-independent asset must not fold the platform into its key"
        );
    }

    // A target-dependent asset does fold the platform in, so the two backends
    // separate even when every other input matches.
    #[test]
    fn target_dependent_payload_keys_fold_in_the_platform() {
        let args = json!({});
        let dependent = CacheInputs {
            sources: SourceFiles::Only(Vec::new()),
            target_dependent: true,
        };
        let platform = concinnity_core::build::Platform::current().key();
        assert_eq!(
            payload_key(1, &args, &ctx(), &dependent),
            key_from_parts(1, &args, &[], Some(platform)),
        );
        assert_ne!(
            payload_key(1, &args, &ctx(), &dependent),
            payload_key(
                1,
                &args,
                &ctx(),
                &CacheInputs {
                    sources: SourceFiles::Only(Vec::new()),
                    target_dependent: false,
                }
            ),
        );
    }

    #[test]
    fn collect_strings_walks_nested_arrays_and_objects() {
        let mut out = Vec::new();
        collect_strings(
            &json!({"a": ["x", {"b": "y"}], "n": 5, "f": true, "z": null}),
            &mut out,
        );
        out.sort();
        assert_eq!(out, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn referenced_files_resolve_through_the_artifacts_dir() {
        // A bare filename that exists neither directly nor under
        // .concinnity/assets/ still resolves when the build supplied an
        // account artifacts directory.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("fx.hdr"), b"pixels").unwrap();
        let artifacts = dir.path().to_str().unwrap().to_string();
        let artifact_ctx = BuildCtx {
            name: "test",
            artifacts_dir: Some(&artifacts),
            all_assets: &[],
        };

        // Nested placement also exercises the recursive JSON string walk.
        let args = json!({"maps": [{"source": "fx.hdr"}]});
        let files = referenced_files(&args, &artifact_ctx);
        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("fx.hdr"));

        // The same reference without an artifacts dir resolves nothing.
        assert!(referenced_files(&args, &ctx()).is_empty());
    }

    // A bare filename that exists neither directly nor in the artifacts dir is
    // searched recursively under the state root's `assets/` tree. The lookup
    // runs against the process-global anchor, so it shares the build-output
    // lock with the other tests that install one.
    #[test]
    fn resolve_source_finds_a_bare_filename_under_the_assets_tree() {
        let _guard = crate::blob::test_output::LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let state = crate::blob::test_output::StateDir::new();
        let nested = state.assets_dir().join("hdri");
        std::fs::create_dir_all(&nested).expect("assets tree");
        std::fs::write(nested.join("sky.hdr"), b"radiance").expect("write source");

        let found = resolve_source("sky.hdr", &ctx()).expect("bare filename resolves");
        assert!(found.ends_with("sky.hdr"), "got: {found}");
        assert_eq!(resolve_source("missing.hdr", &ctx()), None);
    }

    // An artifacts dir that does not hold the file contributes nothing, so the
    // lookup falls through to no resolution rather than a bogus path.
    #[test]
    fn resolve_source_falls_through_an_artifacts_dir_without_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = dir.path().to_str().unwrap().to_string();
        let artifact_ctx = BuildCtx {
            name: "test",
            artifacts_dir: Some(&artifacts),
            all_assets: &[],
        };
        assert_eq!(resolve_source("absent.hdr", &artifact_ctx), None);
    }

    // Storing is best-effort: a directory that cannot be created drops the
    // entry rather than failing the build.
    #[test]
    fn store_in_gives_up_when_the_directory_cannot_be_created() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"a file, not a directory").unwrap();

        let unusable = blocker.join("cache");
        store_in(&unusable, "k", b"bytes");
        assert_eq!(load_in(&unusable, "k"), None);
        assert!(!unusable.exists());
    }

    #[test]
    fn resolve_source_skips_non_file_looking_strings() {
        // No extension and no separator: never probed, even when a file of
        // that exact name exists in the artifacts dir.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("chrome"), b"x").unwrap();
        let artifacts = dir.path().to_str().unwrap().to_string();
        let artifact_ctx = BuildCtx {
            name: "test",
            artifacts_dir: Some(&artifacts),
            all_assets: &[],
        };
        assert_eq!(resolve_source("chrome", &artifact_ctx), None);
    }

    #[test]
    fn expand_key_is_stable_when_the_source_is_missing() {
        // A missing source file contributes no hash; the key is still a
        // deterministic function of the remaining inputs.
        let args = json!({ "prefix": "scn" });
        let a = expand_key("/no/such/scene.glb", &args);
        let b = expand_key("/no/such/scene.glb", &args);
        assert_eq!(a, b);
        assert_ne!(
            a,
            expand_key("/no/such/scene.glb", &json!({ "prefix": "x" }))
        );
    }
}
