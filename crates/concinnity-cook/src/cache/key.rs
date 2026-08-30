// The key half of the cache: what an entry is a function of.
//
// Every key is a SHA-256 over the inputs the stored bytes were produced from --
// the args JSON, the contents of every source file those args reach, and the
// compile target where one applies. Content addressing is what makes a hit
// safe: an entry is served only to a compile whose inputs hash to the same
// value.
//
// What produced the bytes is not hashed here. That is the identity of the cook
// binary, which the segment's header carries once for every entry it holds
// (see `identity`), so a key covers the inputs and the header covers the code.

use crate::asset::{BuildCtx, CacheInputs, SourceFiles};
use crate::file_stamp::FileStamp;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

// SHA-256 a source file's contents, memoized by path within the process. A
// single build can reference one large source file from hundreds of assets
// (e.g. every Mesh imported from one `.fbx`), and hashing it once per asset
// dominates the build; the memo reads + hashes each unique file once. Keyed by
// `FileStamp` so a file edited between in-process rebuilds (the `cn debug`
// hot-reload path) is re-hashed rather than served stale.
fn file_content_hash(path: &str) -> Option<[u8; 32]> {
    type HashMemo = Mutex<HashMap<String, (FileStamp, [u8; 32])>>;
    static MEMO: OnceLock<HashMemo> = OnceLock::new();
    let stamp = FileStamp::read(path)?;
    // Decided before the read, so a write racing it lands on a later mtime and
    // misses this entry rather than matching it.
    let memoizable = stamp.settled();
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&(s, h)) = memo
        .lock()
        .expect("file-stamp memo lock is not poisoned")
        .get(path)
        && s == stamp
    {
        return Some(h);
    }
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hash: [u8; 32] = hasher.finalize().into();
    if memoizable {
        memo.lock()
            .expect("file-stamp memo lock is not poisoned")
            .insert(path.to_string(), (stamp, hash));
    }
    Some(hash)
}

// Compute the cache key for one compiled asset. The key folds in the asset
// schema the baked records encode against, the component discriminant, the args
// JSON, a hash of every input the compile reads, and -- for assets that compile
// rather than transport their source -- the active backend's shader platform.
//
// The key is content-addressed with no namespacing prefix: an asset that
// compiles identically on every backend (a mesh, a texture, a font) produces
// one entry shared by a DirectX and a Vulkan cook rather than a copy each.
// Assets whose payload really does differ per backend separate themselves
// through their inputs -- a differing source file, or the compile target when
// `CacheInputs::target_dependent` is set.
pub(crate) fn payload_key(
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
        SourceFiles::Only(paths) => paths
            .iter()
            .filter_map(|p| file_content_hash(p).map(|h| (p.clone(), h)))
            .collect(),
    };
    let target = inputs
        .target_dependent
        .then(|| concinnity_core::platform::Platform::current().key());
    key_from_parts(discriminant, args, &files, target)
}

// Cache key for a SceneImport expansion. The generated asset-entry list is a
// deterministic function of the source file's contents and the import options,
// so editing the source file or changing an option busts the entry. Like
// `payload_key` this is platform-independent: the entries are plain JSON with
// no per-backend branching. The two share one segment, and it is the entry
// kind that keeps their key spaces apart rather than anything in the preimage.
pub(crate) fn expand_key(
    source: &str,
    args: &serde_json::Value,
    assets_dir: Option<&Path>,
) -> String {
    let mut hasher = Sha256::new();
    if let Some(h) = file_content_hash(source) {
        hasher.update(h);
    }
    // A text `.gltf` pulls geometry and images from sibling files the source
    // hash alone cannot see; fold their contents in so editing a referenced
    // `.bin` or image re-expands the import.
    if source.to_lowercase().ends_with(".gltf") {
        for path in crate::import::gltf_source::referenced_files(source, assets_dir) {
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

// Key for what a bake derived from geometry its target's payload replaces: the
// capsule scaling, which is read off the pre-bake bind pose a stored payload no
// longer carries. Namespaced off the payload key so the two are written on the
// same miss and invalidated by the same inputs.
pub(crate) fn bake_key(payload_key: &str) -> String {
    format!("{payload_key}-bake")
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
    // Covers the runtime half of the pipeline the binary identity in the
    // segment header does not: the payload serialisers in
    // `concinnity_core::bake`, and the asset schema the baked records inside a
    // payload encode against.
    hasher.update(concinnity_core::SCHEMA_VERSION.to_le_bytes());
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
// under the build's asset search root, a relative or absolute path is used
// directly, and `artifacts_dir` is consulted when set. Strings that do not
// resolve to a file (asset names, generator keywords, colors) contribute
// nothing.
fn referenced_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> Vec<(String, [u8; 32])> {
    let mut strings = Vec::new();
    collect_strings(args, &mut strings);

    let mut out = Vec::new();
    for s in strings {
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
    // Bare filename searched recursively under the asset search root.
    if let Some(p) = ctx
        .assets_dir
        .and_then(|dir| crate::source::find_in(dir, s))
    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> BuildCtx<'static> {
        BuildCtx {
            name: "test",
            assets_dir: None,
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

    // The stat memo behind `file_content_hash` cannot lean on length alone: an
    // edit that preserves it (a retargeted URI, a flipped flag) leaves the two
    // writes distinguishable only by mtime, and back-to-back writes routinely
    // share one filesystem tick. `FileStamp::settled` is what keeps this key
    // moving; without it this assert fails whenever the writes land together.
    #[test]
    fn key_tracks_an_equal_length_edit() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("env.hdr");
        std::fs::write(&file, b"aaaaa").unwrap();
        let args = json!({ "source": file.to_str().unwrap() });

        let before = payload_key(3, &args, &ctx(), &CacheInputs::extra(vec![]));
        std::fs::write(&file, b"bbbbb").unwrap();
        let after = payload_key(3, &args, &ctx(), &CacheInputs::extra(vec![]));
        assert_ne!(
            before, after,
            "an equal-length edit in the same mtime tick must still bust the key"
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
            sources: SourceFiles::Only(vec![used.to_str().unwrap().to_string()]),
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
    fn expand_key_tracks_source_contents_and_args() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("scene.fbx");
        std::fs::write(&file, b"first").unwrap();
        let src = file.to_str().unwrap();
        let args = json!({ "prefix": "scn", "texture_max_size": 512 });

        let base = expand_key(src, &args, None);
        // Stable for identical inputs.
        assert_eq!(base, expand_key(src, &args, None));
        // Changing an option busts the key.
        assert_ne!(
            base,
            expand_key(
                src,
                &json!({ "prefix": "scn", "texture_max_size": 256 }),
                None
            )
        );
        // Editing the source file busts the key.
        std::fs::write(&file, b"second").unwrap();
        assert_ne!(base, expand_key(src, &args, None));
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

        let before = expand_key(src, &args, None);
        assert_eq!(before, expand_key(src, &args, None));
        std::fs::write(dir.path().join("geo.bin"), b"second").unwrap();
        assert_ne!(
            before,
            expand_key(src, &args, None),
            "editing a referenced .bin must bust the expansion key"
        );
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
        let platform = concinnity_core::platform::Platform::current().key();
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
        // The assets dir still resolves when the build supplied an
        // account artifacts directory.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("fx.hdr"), b"pixels").unwrap();
        let artifacts = dir.path().to_str().unwrap().to_string();
        let artifact_ctx = BuildCtx {
            name: "test",
            assets_dir: None,
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
    // searched recursively under the build's asset search root.
    #[test]
    fn resolve_source_finds_a_bare_filename_under_the_assets_tree() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("hdri");
        std::fs::create_dir_all(&nested).expect("assets tree");
        std::fs::write(nested.join("sky.hdr"), b"radiance").expect("write source");
        let assets_ctx = BuildCtx {
            name: "test",
            assets_dir: Some(dir.path()),
            artifacts_dir: None,
            all_assets: &[],
        };

        let found = resolve_source("sky.hdr", &assets_ctx).expect("bare filename resolves");
        assert!(found.ends_with("sky.hdr"), "got: {found}");
        assert_eq!(resolve_source("missing.hdr", &assets_ctx), None);
        // Without a search root the same bare filename resolves nothing.
        assert_eq!(resolve_source("sky.hdr", &ctx()), None);
    }

    // An artifacts dir that does not hold the file contributes nothing, so the
    // lookup falls through to no resolution rather than a bogus path.
    #[test]
    fn resolve_source_falls_through_an_artifacts_dir_without_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = dir.path().to_str().unwrap().to_string();
        let artifact_ctx = BuildCtx {
            name: "test",
            assets_dir: None,
            artifacts_dir: Some(&artifacts),
            all_assets: &[],
        };
        assert_eq!(resolve_source("absent.hdr", &artifact_ctx), None);
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
            assets_dir: None,
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
        let a = expand_key("/no/such/scene.glb", &args, None);
        let b = expand_key("/no/such/scene.glb", &args, None);
        assert_eq!(a, b);
        assert_ne!(
            a,
            expand_key("/no/such/scene.glb", &json!({ "prefix": "x" }), None)
        );
    }
}
