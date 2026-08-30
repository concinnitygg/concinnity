//! The runtime crate (rlib). Two jobs, both delegated to the shared
//! `concinnity-toolchain` build helper:
//!
//! 1. Resolve the rendering backend once and expose it as a single cfg the crate
//!    gates on (`backend_metal` / `backend_dx` / `backend_vk`).
//!
//! 2. Detect the optional upscaler SDKs and emit the cfgs the renderer gates on.
//!    This crate produces only an rlib (consumed by the editor and the examples)
//!    plus its own test binaries, so it does NOT bundle runtime DLLs next to a
//!    binary (that belongs to whichever package owns the final artifact). The one
//!    link directive kept is the NGX import lib: the DLSS modules are
//!    `#[cfg(ngx_sdk_bundled)]`, so when that cfg is on they compile into the lib
//!    and must resolve their NGX symbols when this crate's test binaries link.

// 3. Bake the bundled face into an SDF atlas the crate embeds, so text can draw
//    with no compiled world data behind it: the startup error screen, which runs
//    when loading that data is what failed, and any TextLabel or TextInput naming
//    no Font.

use concinnity_toolchain::{emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

// Native size of the built-in face: what text naming no Font lays out at before
// its own `scale`. The field is signed-distance and the atlas supersamples, so
// one size covers the range both callers need, and sits near the `Font` asset's
// own 20px default so a font-less label reads like an authored one.
const BUILTIN_FONT_PX: u32 = 24;

fn main() {
    emit_check_cfgs();
    setup_graphics_sdks(emit_backend_cfg());
    bake_builtin_font();
}

// Compile the bundled face into `OUT_DIR`, where `gfx::builtin_font` embeds it.
fn bake_builtin_font() {
    let payload = concinnity_core::bake::font::compile(
        concinnity_core::bake::font::BUILTIN_FONT_BYTES,
        BUILTIN_FONT_PX,
        concinnity_core::bake::font::BUILTIN_FONT_FILE,
    )
    .expect("bundled face compiles");

    let out = std::path::Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR is set"))
        .join("builtin_font.bin");
    std::fs::write(&out, payload).expect("write baked font atlas");
    println!("cargo:rerun-if-changed=build.rs");
}
