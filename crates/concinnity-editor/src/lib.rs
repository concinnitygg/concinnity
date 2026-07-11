// concinnity-editor: the dev tooling library.
//
// Holds the world authoring / in-memory build code (add / rm / check / build),
// the in-engine editor HUD, the localhost debug server, and the `concinnity`
// CLI dispatch. The thin `concinnity` binary (concinnity-cli) and the C-ABI
// embedding surface (concinnity-ffi) both build on this crate. Sits on top of
// the runtime (concinnity-engine) and the compiler (concinnity-cook); a shipped
// runtime links none of this.

// Bridge: re-export the runtime/core modules the authoring, editor, and debug
// code names under crate::* so their `crate::<module>` import paths resolve.
// world.jsonl I/O lives in the compiler (concinnity-cook), not core.
#[allow(unused_imports)]
pub(crate) use concinnity_cook::world;
#[allow(unused_imports)]
pub(crate) use concinnity_core::{build, geometry, result};
#[allow(unused_imports)]
pub(crate) use concinnity_engine::{app, assets, blob, config, ecs, gfx, jobs};

// Authoring / in-memory build, shared with the FFI embedding surface.
mod authoring;
// The dev CLI: clap command tree + dispatch.
mod entry;

// CLI subcommands, the in-engine editor HUD, the localhost debug server, the
// interpreted run loop, and animation clip hot-reload.
mod anim_reload;
mod cli;
mod debug;
mod debug_hook;
mod editor;
mod run;

// The Metal shader-layout validator bridge, installed before any build. Only
// the Metal backend ships one; other backends leave the hook unregistered.
#[cfg(backend_metal)]
mod shader_reflect;

// Process-global test serialization lock; test builds only.
#[cfg(test)]
mod test_support;

// The dev CLI entry point: parse argv and dispatch. concinnity-cli calls this.
pub use entry::run;

// The authoring API, shared with the FFI embedding surface (concinnity-ffi) and
// published as the Rust build + validate surface.
pub use authoring::{
    add_to_path, arg_value_to_json, build_world_from_path, build_world_from_str,
    build_world_to_disk, check_at_path, check_from_str, rm_at_path, spec_args, spec_to_value,
    world_from_loaded, world_template_entries,
};
pub use concinnity_cook::world::{parse_world_jsonl, write_world_jsonl};
pub use concinnity_cook::{build_pipeline_from_str, validate_asset, validate_world_jsonl};
