// concinnity-ffi: the C-ABI embedding surface.
//
// The full extern "C" surface a host application links (`ffi`), built as a
// cdylib (`libconcinnity_ffi.dylib` on macOS / `concinnity_ffi.dll` on
// Windows). Sits on top of the dev tooling library (concinnity-editor, for the
// authoring / in-memory build API), the runtime (concinnity-engine), the device
// backends (concinnity-device, for the embedded-preview hooks), and the
// compiler (concinnity-cook). A shipped runtime links none of these.

// Bridge: re-export the runtime/device modules the ffi code names under crate::*
// so its `crate::<module>` import paths resolve.
#[allow(unused_imports)]
pub(crate) use concinnity_engine::ecs;
// The Metal backend (the embedded-preview hooks the host app drives) lives in
// concinnity-device.
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub(crate) use concinnity_device::metal;

pub mod ffi;

// Shared process-global test lock; test builds only.
#[cfg(test)]
pub(crate) mod test_support;
