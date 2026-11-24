// The Agility SDK handshake: two symbols the system `d3d12.dll` reads from the
// EXE's export table at process start, loading `D3D12Core.dll` from the named
// path instead of the OS runtime. FSR3 needs the newer runtime.
//
// Each binary's build script emits `/EXPORT:<name>,DATA`, which also pulls
// these out of the rlib; without `,DATA` the linker adds a code thunk.
//
// With the directory missing, device creation fails outright, so a binary is
// portable only with it. The version must match the bundled Agility package
// (`microsoft.direct3d.d3d12.1.619.3` is 619).

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
