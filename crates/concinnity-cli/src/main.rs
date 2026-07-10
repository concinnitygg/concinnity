// concinnity-cli: the `concinnity` dev CLI binary.
//
// A thin frontend over the concinnity-editor library: it parses argv and
// dispatches through `concinnity_editor::run`. The dev tooling (world authoring,
// the in-engine editor HUD, the localhost debug server) all lives in that
// library; this crate is the executable entry point plus the Windows Agility SDK
// export statics the final binary must carry.

// Microsoft Agility SDK opt-in.
//
// Windows' system `d3d12.dll` reads these two symbols from the host EXE's PE
// export table at process start: when both are present and the named SDK path
// resolves to a directory containing `D3D12Core.dll`, it loads that copy in
// place of the OS-bundled (older) D3D12 runtime. Modern FidelityFX FSR3 (and any
// other feature requiring a recent D3D12 capability bit) needs the Agility SDK;
// without these exports `ffxCreateContext` throws a C++ exception that aborts the
// process.
//
// The companion `build.rs` setup copies `D3D12Core.dll` + `d3d12SDKLayers.dll`
// from the NuGet package into `target/{profile}/D3D12/` so this relative path
// resolves. Setting the `D3D12SDKVersion` value here to match the NuGet package
// version is critical; the directory name is
// `microsoft.direct3d.d3d12.1.<VER>.<PATCH>`. Mirrors concinnity-runtime; keep
// the version in sync when bumping the Agility SDK.
//
// `#[used]` forces the linker to keep the symbols around even though nothing in
// Rust references them; `#[no_mangle]` keeps the exact case-sensitive name
// `d3d12.dll` looks up.
#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKVersion: u32 = 619;

#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";

fn main() -> std::io::Result<()> {
    concinnity_editor::run()
}
