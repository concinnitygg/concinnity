//! Precompiled engine shader libraries. The build script compiles every
//! single-source program and every static `.metal` under src/metal/shaders/ to
//! a metallib and generates the `embedded_metallib` lookup included here,
//! pairing each name with the digest of the source it was built from;
//! `shader::builtin::fetch` takes these bytes whenever that digest matches the
//! source the renderer assembled. When the build host lacked the Metal
//! toolchain the generated lookup returns `None` for every name and the source
//! path takes over.

include!(concat!(env!("OUT_DIR"), "/engine_metallibs.rs"));

#[cfg(test)]
mod tests {
    use super::embedded_metallib;
    use concinnity_core::platform::Platform;
    use concinnity_core::render::shader_programs::metal::TABLE;

    // A shader the build script always compiles when the Metal toolchain is
    // present, so its absence means the stub lookup and nothing else.
    const TOOLCHAIN_SENTINEL: &str = crate::metal::pipeline::CULL_ENCODE;

    // Every registered single-source variant must be precompiled, unless
    // the build host lacked the Metal toolchain (then all of them miss together
    // and the runtime compile path takes over, warned at build time).
    #[test]
    fn single_source_precompiled_coverage_is_all_or_nothing() {
        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        let present: Vec<bool> = TABLE
            .variants()
            .map(|v| embedded_metallib(&v.artifact_name()).is_some_and(|(_, b)| !b.is_empty()))
            .collect();
        assert!(
            present.iter().all(|&p| p) || present.iter().all(|&p| !p),
            "partial single-source metallib coverage: {present:?}"
        );
    }

    #[test]
    fn unregistered_names_return_none() {
        assert!(embedded_metallib("nope.metal").is_none());
    }

    // Every registered name's embedded digest must equal the digest of the
    // source the renderer assembles for it, or `fetch` would miss and
    // compile a shader the binary already carries -- which is what made an
    // installed `cn editor` demand a shader compiler at startup.
    #[test]
    fn every_embedded_digest_matches_the_assembled_source() {
        use concinnity_core::render::shader_source::source_digest;

        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        for v in TABLE.variants() {
            let name = v.artifact_name();
            let (digest, _) = embedded_metallib(&name)
                .unwrap_or_else(|| panic!("{name}: no precompiled metallib embedded"));
            let source = v.assemble(Platform::Metal);
            assert_eq!(digest, source_digest(&source), "{name}");
        }
        let source = crate::metal::pipeline::cull_encode_source(false);
        let (digest, _) = embedded_metallib(TOOLCHAIN_SENTINEL).expect("sentinel embedded");
        assert_eq!(digest, source_digest(&source), "{TOOLCHAIN_SENTINEL}");
    }

    // Hot-reload must not cost a recompile when nothing was edited: the digest
    // of the disk-preferred assembly equals the embedded one for an unedited
    // checkout, and equals it trivially for an install with no checkout at all.
    #[test]
    fn an_unedited_hot_reload_assembly_still_matches_the_embedded_digest() {
        use concinnity_core::render::shader_source::source_digest;

        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        for v in TABLE.variants() {
            let name = v.artifact_name();
            let (digest, _) = embedded_metallib(&name).expect("embedded");
            let source = crate::shader::source::assemble_variant(true, Platform::Metal, v);
            assert_eq!(digest, source_digest(&source), "{name}");
        }
        let source = crate::metal::pipeline::cull_encode_source(true);
        let (digest, _) = embedded_metallib(TOOLCHAIN_SENTINEL).expect("sentinel embedded");
        assert_eq!(digest, source_digest(&source), "{TOOLCHAIN_SENTINEL}");
    }

    // An edited shader must miss, or hot-reload would silently keep serving the
    // build-time copy. Same comparison the renderer makes, over changed text.
    #[test]
    fn an_edited_source_does_not_match_the_embedded_digest() {
        use concinnity_core::render::shader_source::source_digest;

        if embedded_metallib(TOOLCHAIN_SENTINEL).is_none() {
            return;
        }
        let v = TABLE.variants().next().expect("a Metal program");
        let name = v.artifact_name();
        let (digest, _) = embedded_metallib(&name).expect("embedded");
        let edited = format!("{}\n// an edit\n", v.assemble(Platform::Metal));
        assert_ne!(digest, source_digest(&edited), "{name}");
    }
}
