// Whether this host can compile the `.slang` files a world authors.
//
// Two callers. The cook asks before compiling a Shader or an SdfVolume field,
// and fails naming the asset when the answer is no, so a build never quietly
// produces a world missing what it declared. And the tests that cook a Shader
// check the compile rather than the toolchain: slangc is absent on a build-only
// host (a CI image, a container), where every one of them would fail for the
// same reason and say nothing about the cook. They return early there; the hosts
// that carry a compiler keep the coverage.
pub(crate) fn slangc_available() -> bool {
    concinnity_slang::slangc_path().is_some()
}
