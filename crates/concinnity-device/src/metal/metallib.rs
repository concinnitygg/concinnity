// src/metal/metallib.rs
//
// Precompiled engine shader libraries. The build script compiles every static
// `.metal` under src/metal/shaders/ to a metallib and generates the
// `embedded_metallib` lookup included here, pairing each name with the digest
// of the source it was built from; `shader_library` in pipeline.rs takes these
// bytes whenever that digest matches the source it assembled. When the build
// host lacked the Metal toolchain the generated lookup returns `None` for every
// name and the source path takes over.

include!(concat!(env!("OUT_DIR"), "/engine_metallibs.rs"));

#[cfg(test)]
mod tests {
    use super::embedded_metallib;

    // A shader the build script always compiles when the Metal toolchain is
    // present, so its absence means the stub lookup and nothing else. It has to
    // be one that still exists: a name that has been ported away skips both
    // coverage tests silently, which is what `post.metal` did after the
    // composite pass moved to single source.
    const TOOLCHAIN_SENTINEL: &str = "cull_encode.metal";

    #[test]
    fn precompiled_coverage_is_all_or_nothing() {
        // The build script either compiles every eligible shader or emits the
        // stub lookup; per-shader gaps would mean a silent slow path. Skip when
        // the build host had no Metal toolchain (stub lookup).
        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal/shaders");
        for entry in std::fs::read_dir(dir).expect("read shaders dir") {
            let file_name = entry.expect("dir entry").file_name();
            let name = file_name.to_str().expect("utf8 shader filename");
            if !name.ends_with(".metal") || name.starts_with("raymarch_") {
                continue;
            }
            let (_, bytes) = embedded_metallib(name)
                .unwrap_or_else(|| panic!("{name}: no precompiled metallib embedded"));
            assert!(!bytes.is_empty(), "{name}: embedded metallib is empty");
        }
    }

    // Every registered single-source variant must be precompiled too, unless
    // the build host lacked slangc (then all of them miss together and the
    // runtime compile path takes over, warned at build time).
    #[test]
    fn slang_precompiled_coverage_is_all_or_nothing() {
        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        let present: Vec<bool> = crate::metal::slang_shaders::ALL
            .iter()
            .map(|lib| embedded_metallib(lib.name).is_some_and(|(_, b)| !b.is_empty()))
            .collect();
        assert!(
            present.iter().all(|&p| p) || present.iter().all(|&p| !p),
            "partial slang metallib coverage: {present:?}"
        );
    }

    #[test]
    fn unregistered_names_return_none() {
        assert!(embedded_metallib("nope.metal").is_none());
    }

    // Every registered name's embedded digest must equal the digest of the
    // source the renderer assembles for it, or `shader_library` would miss and
    // compile a shader the binary already carries -- which is what made an
    // installed `cn editor` demand slangc at startup.
    #[test]
    fn every_embedded_digest_matches_the_assembled_source() {
        use concinnity_core::render::slang_source::source_digest;

        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        for lib in crate::metal::slang_shaders::ALL {
            let (digest, _) = embedded_metallib(lib.name)
                .unwrap_or_else(|| panic!("{}: no precompiled metallib embedded", lib.name));
            let source = concinnity_core::render::slang_source::assemble(lib.file, lib.defines);
            assert_eq!(digest, source_digest(&source), "{}", lib.name);
        }
        let source = crate::metal::pipeline::shader_source(false, TOOLCHAIN_SENTINEL);
        let (digest, _) = embedded_metallib(TOOLCHAIN_SENTINEL).expect("sentinel embedded");
        assert_eq!(digest, source_digest(&source), "{TOOLCHAIN_SENTINEL}");
    }

    // Hot-reload must not cost a recompile when nothing was edited: the digest
    // of the disk-preferred assembly equals the embedded one for an unedited
    // checkout, and equals it trivially for an install with no checkout at all.
    #[test]
    fn an_unedited_hot_reload_assembly_still_matches_the_embedded_digest() {
        use concinnity_core::render::slang_source::source_digest;

        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        for lib in crate::metal::slang_shaders::ALL {
            let (digest, _) = embedded_metallib(lib.name).expect("embedded");
            let source = crate::slang_source::assemble(true, lib.file, lib.defines, &[]);
            assert_eq!(digest, source_digest(&source), "{}", lib.name);
        }
        let source = crate::metal::pipeline::shader_source(true, TOOLCHAIN_SENTINEL);
        let (digest, _) = embedded_metallib(TOOLCHAIN_SENTINEL).expect("sentinel embedded");
        assert_eq!(digest, source_digest(&source), "{TOOLCHAIN_SENTINEL}");
    }

    // An edited shader must miss, or hot-reload would silently keep serving the
    // build-time copy. Same comparison the renderer makes, over changed text.
    #[test]
    fn an_edited_source_does_not_match_the_embedded_digest() {
        use concinnity_core::render::slang_source::source_digest;

        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        let lib = crate::metal::slang_shaders::ALL[0];
        let (digest, _) = embedded_metallib(lib.name).expect("embedded");
        let edited = format!(
            "{}\n// an edit\n",
            concinnity_core::render::slang_source::assemble(lib.file, lib.defines)
        );
        assert_ne!(digest, source_digest(&edited), "{}", lib.name);
    }
}
