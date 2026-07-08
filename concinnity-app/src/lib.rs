// concinnity-app/src/lib.rs
//
// The embedding + authoring library. Holds the full C-ABI surface the Swift app
// links (`ffi`) and the world authoring / in-memory build code (add / rm / check
// / build) shared with the `concinnity` dev CLI. Sits on top of concinnity-client
// (the runtime), concinnity-cook (the compiler), and concinnity-core; a shipped
// runtime links none of these, so it sheds the build dependencies.

// Bridge: re-export the runtime/core modules the authoring + ffi code names
// under crate::* so their `crate::<module>` import paths resolve.
#[allow(unused_imports)]
pub(crate) use concinnity_client::{blob, ecs, gfx};
pub(crate) use concinnity_core::world;
// The Metal backend (for shader-layout reflection + the embedded-preview FFI
// hooks) now lives in concinnity-device.
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub(crate) use concinnity_device::metal;

// Authoring / in-memory build: private modules with a flat public API.
mod add;
mod build;
mod check;
mod rm;
mod sources;

pub mod ffi;
#[cfg(backend_metal)]
mod shader_reflect;

// Shared process-global test lock; test builds only.
#[cfg(test)]
pub(crate) mod test_support;

// The authoring API: add / remove / check assets in a world JSONL, and build a
// world into a runnable in-memory World. Consumed as `concinnity_app::<fn>`.
pub use add::add_to_path;
pub use build::{build_world_from_path, build_world_to_disk};
pub use check::{check_at_path, check_from_str};
pub use rm::rm_at_path;

// Build API (the published Rust build + validate surface).
pub use concinnity_cook::{build_pipeline_from_str, validate_asset, validate_world_jsonl};
pub use concinnity_core::world::{parse_world_jsonl, write_world_jsonl};
