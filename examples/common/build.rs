// build.rs
//
// This package is a lib with no binary, but its lib test target is still a
// final link and it pulls in the runtime's DLSS modules. Those are
// `#[cfg(ngx_sdk_bundled)]`, so when that cfg is on their NGX symbols have to
// resolve here too -- and the NGX link directive is scoped to the package that
// emits it (see `dlss_directives` in concinnity-toolchain), so every package
// linking the DLSS code emits its own. Nothing here owns a final artifact, so
// no runtime DLLs are bundled: `SdkOptions { bundle_dlls: false }`.
//
// The backend is resolved without emitting its cfg: this package's source gates
// on none of the renderer cfgs, it only needs the backend to pick the SDK set.

use concinnity_toolchain::{SdkOptions, backend_from_cargo, setup_graphics_sdks};

fn main() {
    setup_graphics_sdks(backend_from_cargo(), SdkOptions { bundle_dlls: false });
}
