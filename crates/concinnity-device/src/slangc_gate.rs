// Whether this host can compile the single-source `.slang` shaders.
//
// The backends' shader compile tests check the shaders, not the toolchain:
// slangc resolves from PATH and is absent on a build-only host (a CI image, a
// container), where every one of them would fail for the same reason and say
// nothing about the source. They return early there; the hosts that carry a
// compiler keep the coverage.
pub(crate) fn slangc_available() -> bool {
    concinnity_slang::slangc_path().is_some()
}
