//! Build-time compilation of the single-source engine shaders into bytes the
//! binary carries.
//!
//! The Metal backend has always done this (`metal_shaders`); this is the same
//! idea for the targets whose artifacts are plain byte blobs rather than
//! metallibs. What it buys is not startup time but self-containment: a binary
//! that embeds its shaders needs no compiler on the host that runs it, which is
//! otherwise a hard runtime dependency of the DirectX and Vulkan backends.
//!
//! The declarations come from `concinnity_core::render::shader_programs`, the
//! same table the renderer iterates, so the text compiled here is the text the
//! renderer would have compiled.

use std::path::{Path, PathBuf};

// The directory under `OUT_DIR` the compiled artifacts land in.
const ARTIFACT_DIR: &str = "shader_artifacts";

/// One program to precompile: the name the renderer looks it up by, the
/// assembled source text, the entry point, and what to emit.
pub struct ShaderArtifact<'a> {
    /// Lookup key. Must match what the renderer asks for.
    pub name: String,
    /// Fully assembled source, defines and fragments already spliced.
    pub source: String,
    /// Entry point compiled out of `source`.
    pub entry: &'a str,
    /// File name the compiler reports diagnostics against.
    pub file_name: &'a str,
    /// What to emit.
    pub target: concinnity_shader::HlslTarget,
}

/// Compile every artifact into `OUT_DIR` and generate `generated` beside it: a
/// `fn <lookup>(name: &str) -> Option<(u64, &'static [u8])>` mapping each name
/// to the digest of the source it was built from and its embedded bytes.
///
/// The digest is what makes the artifact safe to use: the caller compares it
/// against the source it just assembled and compiles instead on a mismatch, so
/// an edited shader is never shadowed by a stale build-time copy.
///
/// A program that fails to compile is a hard error: a broken shader must fail
/// the build rather than surface at renderer init. So is a missing dxc (see
/// `embedded_shaders::require_dxc`). An empty `artifacts` emits a stub lookup
/// answering `None` for every name, for a host that has nothing to embed.
pub fn precompile_shader_artifacts(
    artifacts: &[ShaderArtifact<'_>],
    generated: &str,
    lookup: &str,
) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let generated = out_dir.join(generated);

    // Before the compiler is demanded: a leg with nothing to compile embeds
    // nothing by design, which is what the DirectX one does off Windows.
    if artifacts.is_empty() {
        std::fs::write(&generated, stub_lookup_source(lookup)).expect("write shader lookup");
        return;
    }

    crate::embedded_shaders::require_dxc();

    let art_dir = out_dir.join(ARTIFACT_DIR);
    std::fs::create_dir_all(&art_dir).expect("create shader artifact dir");
    embed(
        artifacts,
        |artifact| {
            (
                artifact.name.clone(),
                concinnity_core::render::shader_source::source_digest(&artifact.source),
                art_dir.join(&artifact.name),
            )
        },
        |artifact| {
            compile(artifact, &art_dir)
                .map(|bytes| (bytes, ()))
                .map_err(|e| format!("precompiling {}: {e}", artifact.name))
        },
        &generated,
        lookup,
    );
}

/// Compile `jobs` side by side, write each one's bytes to the path `entry`
/// names, and generate `lookup` at `generated` over them in job order.
///
/// `entry` gives a job's lookup name, source digest and artifact path.
/// `compile` gives its bytes and whatever else the caller keeps from it, or
/// the message the build fails with. Returns the kept values in job order.
pub(crate) fn embed<J: Sync, K: Send>(
    jobs: &[J],
    entry: impl Fn(&J) -> (String, u64, PathBuf),
    compile: impl Fn(&J) -> Result<(Vec<u8>, K), String> + Sync,
    generated: &Path,
    lookup: &str,
) -> Vec<K> {
    let compiled = crate::parallel_map(jobs, compile);
    let mut entries = Vec::with_capacity(jobs.len());
    let mut kept = Vec::with_capacity(jobs.len());
    for (job, result) in jobs.iter().zip(compiled) {
        let (bytes, keep) = result.unwrap_or_else(|e| panic!("{e}"));
        let (name, digest, path) = entry(job);
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        entries.push((name, digest, path));
        kept.push(keep);
    }
    std::fs::write(generated, lookup_source(&entries, lookup))
        .unwrap_or_else(|e| panic!("write {}: {e}", generated.display()));
    kept
}

// One artifact through dxc.
fn compile(artifact: &ShaderArtifact<'_>, work_dir: &Path) -> Result<Vec<u8>, String> {
    let job = concinnity_shader::HlslJob {
        source: &artifact.source,
        file_name: artifact.file_name,
        entry: artifact.entry,
        target: artifact.target,
    };
    concinnity_shader::compile(&job, work_dir)
}

// Generated lookup mapping a program name to the digest of the source it was
// built from and its embedded bytes, shared with the Metal precompile. Kept
// pure (names and paths in, source out) for unit testing. No entries means the
// stub: a `match` with only the fallthrough arm fails clippy in the including
// crate.
fn lookup_source(entries: &[(String, u64, PathBuf)], lookup: &str) -> String {
    if entries.is_empty() {
        return stub_lookup_source(lookup);
    }
    let mut src = format!(
        "// @generated by concinnity-toolchain\n\
         pub(crate) fn {lookup}(name: &str) -> Option<(u64, &'static [u8])> {{\n\
         \x20   match name {{\n"
    );
    for (name, digest, path) in entries {
        src.push_str(&format!(
            "        {name:?} => Some(({digest}u64, include_bytes!({:?}))),\n",
            path.display().to_string()
        ));
    }
    src.push_str("        _ => None,\n    }\n}\n");
    src
}

// The lookup for a build that embeds nothing.
pub(crate) fn stub_lookup_source(lookup: &str) -> String {
    format!(
        "// @generated by concinnity-toolchain (nothing embedded)\n\
         pub(crate) fn {lookup}(_name: &str) -> Option<(u64, &'static [u8])> {{\n\
         \x20   None\n\
         }}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_source_maps_each_entry_and_falls_through() {
        let entries = vec![
            (
                "vert.dxil".to_string(),
                7u64,
                PathBuf::from("/out/vert.dxil"),
            ),
            (
                "frag.dxil".to_string(),
                9u64,
                PathBuf::from("/out/frag.dxil"),
            ),
        ];
        let src = lookup_source(&entries, "embedded_dxil");
        assert!(src.contains("fn embedded_dxil"));
        assert!(src.contains("\"vert.dxil\" => Some((7u64, include_bytes!(\"/out/vert.dxil\")))"));
        assert!(src.contains("\"frag.dxil\" => Some((9u64, include_bytes!(\"/out/frag.dxil\")))"));
        assert!(src.contains("_ => None"));
    }

    // Without a compiler the lookup must still compile and answer for every name,
    // so the renderer falls back to compiling rather than failing to build.
    #[test]
    fn stub_lookup_returns_none_for_everything() {
        let src = stub_lookup_source("embedded_dxil");
        assert!(src.contains("fn embedded_dxil"));
        assert!(src.contains("None"));
        assert!(!src.contains("include_bytes!"));
    }

    // The two share a signature, so a build with a compiler and one without are
    // drop-in replacements for each other.
    #[test]
    fn the_stub_and_the_real_lookup_share_a_signature() {
        let sig = "(name: &str) -> Option<(u64, &'static [u8])>";
        let entries = vec![("v.spv".to_string(), 1u64, PathBuf::from("/out/v.spv"))];
        assert!(lookup_source(&entries, "f").contains(&format!("fn f{sig}")));
        assert!(
            stub_lookup_source("f").contains("fn f(_name: &str) -> Option<(u64, &'static [u8])>")
        );
    }

    // Every job's bytes land at its path and in the lookup, and what each
    // compile kept comes back in job order.
    #[test]
    fn embed_writes_each_artifact_and_its_lookup_in_job_order() {
        let tree = concinnity_testing::TempTree::new();
        let generated = tree.join("lookup.rs");
        let jobs = ["a", "b", "c"];
        let kept = embed(
            &jobs,
            |name| (name.to_string(), 1, tree.join(name)),
            |name| Ok((name.as_bytes().to_vec(), name.to_uppercase())),
            &generated,
            "f",
        );
        assert_eq!(kept, ["A", "B", "C"]);
        for name in jobs {
            assert_eq!(std::fs::read(tree.join(name)).unwrap(), name.as_bytes());
        }
        let lookup = std::fs::read_to_string(&generated).unwrap();
        let a = lookup.find("\"a\" =>").unwrap();
        let c = lookup.find("\"c\" =>").unwrap();
        assert!(a < c, "{lookup}");
    }

    // A failed compile fails the build with the caller's message.
    #[test]
    #[should_panic(expected = "b broke")]
    fn embed_fails_the_build_on_a_failed_compile() {
        let tree = concinnity_testing::TempTree::new();
        embed(
            &["a", "b"],
            |name| (name.to_string(), 1, tree.join(name)),
            |name| {
                if *name == "b" {
                    Err("b broke".to_string())
                } else {
                    Ok((Vec::new(), ()))
                }
            },
            &tree.join("lookup.rs"),
            "f",
        );
    }

    // A host that compiles nothing gets the stub, not a `match` with one arm,
    // which clippy rejects in the crate that includes it.
    #[test]
    fn an_empty_lookup_is_the_stub() {
        assert_eq!(lookup_source(&[], "f"), stub_lookup_source("f"));
        assert!(!lookup_source(&[], "f").contains("match"));
    }
}
