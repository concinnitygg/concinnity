// The Agility SDK handshake: two symbols Windows' system `d3d12.dll` reads
// from the running EXE's PE export table at process start. When both are
// present and the named path resolves to a directory holding `D3D12Core.dll`,
// it loads that copy in place of the OS-bundled (older) D3D12 runtime. FFX FSR3
// needs the newer one -- see `post::upscale::fsr` -- and without it
// `ffxCreateContext` throws a C++ exception that aborts the process.
//
// An EXE only exports what its linker was told to export, so the statics alone
// are not enough: each package owning a final binary emits
// `/EXPORT:<name>,DATA` from its build script (concinnity-toolchain's
// `BinaryTargets`), and that argument is also what pulls these definitions out
// of the rlib. `,DATA` is critical -- without it the linker inserts a code thunk
// that `d3d12.dll` would dereference as a pointer.
//
// This is the only mechanism a shipped binary can use. The runtime alternative,
// `ID3D12SDKConfiguration::SetSDKVersion`, is documented as usable "only in
// Windows Developer Mode" -- it exists for tools like PIX, not for applications
// -- and returns DXGI_ERROR_INVALID_CALL everywhere else.
//
// The handshake has no fallback: if the directory is absent, D3D12 device
// creation fails outright rather than reverting to the OS runtime, and the
// renderer reports "no suitable D3D12 adapter found" (see `init::window`, which
// says so). A binary carrying these exports is therefore only portable together
// with the directory beside it.
//
// The version must match the Agility NuGet package the build script bundles:
// the directory name is `microsoft.direct3d.d3d12.1.<VERSION>.<PATCH>`, so
// `microsoft.direct3d.d3d12.1.619.3` is 619. `DEFAULT_CN_AGILITY_SDK` in
// concinnity-toolchain names the tree these bytes have to agree with.

// `pub` even though nothing in the crate names them: the visibility is what
// declares them as exported symbols alongside `#[unsafe(no_mangle)]`.
#[unsafe(no_mangle)]
#[used]
#[expect(
    unreachable_pub,
    reason = "the pub visibility is what exports the symbol; nothing in the crate names it"
)]
pub static D3D12SDKVersion: u32 = 619;

#[unsafe(no_mangle)]
#[used]
#[expect(
    unreachable_pub,
    reason = "the pub visibility is what exports the symbol; nothing in the crate names it"
)]
pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";

#[cfg(test)]
mod tests {
    use super::*;

    // `d3d12.dll` appends `D3D12Core.dll` to this path verbatim and resolves it
    // against the directory holding the EXE, so it has to stay a NUL-terminated
    // relative directory with a trailing separator -- naming the same
    // subdirectory the build script copies the SDK into.
    #[test]
    fn the_sdk_path_names_the_bundled_subdirectory() {
        let path = std::str::from_utf8(&D3D12SDKPath[..]).expect("ascii");
        let dir = path
            .strip_suffix('\0')
            .unwrap_or_else(|| panic!("{path:?} must be NUL terminated"));
        assert!(
            dir.starts_with(".\\"),
            "{dir:?} must be relative to the exe"
        );
        assert!(dir.ends_with('\\'), "{dir:?} must end in a separator");
        assert_eq!(&dir[2..dir.len() - 1], "D3D12");
    }
}
