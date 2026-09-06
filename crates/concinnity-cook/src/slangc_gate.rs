// Whether this host can run the compile-backed tests.
//
// The tests that cook a Shader check the compile, not the toolchain: slangc is
// absent on a build-only host (a CI image, a container), where every one of
// them would fail for the same reason and say nothing about the cook. They
// return early there; the hosts that carry a compiler keep the coverage.
pub(crate) fn slangc_available() -> bool {
    concinnity_slang::slangc_path().is_some()
}
