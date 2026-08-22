// build.rs
//
// The runtime crate (rlib). Two jobs, both delegated to the shared
// `concinnity-toolchain` build helper:
//
// 1. Resolve the rendering backend once and expose it as a single cfg the crate
//    gates on (`backend_metal` / `backend_dx` / `backend_vk`).
//
// 2. Detect the optional upscaler SDKs and emit the cfgs the renderer gates on.
//    This crate produces only an rlib (consumed by the editor and the examples)
//    plus its own test binaries, so it does NOT bundle runtime DLLs next to a
//    binary (that belongs to whichever package owns the final artifact). The one
//    link directive kept is the NGX import lib: the DLSS modules are
//    `#[cfg(ngx_sdk_bundled)]`, so when that cfg is on they compile into the lib
//    and must resolve their NGX symbols when this crate's test binaries link.
//    That is `BinaryTargets::None`.

// 3. Bake the bundled default font into an SDF atlas the crate embeds, so the
//    startup error screen can draw text with no compiled world data present --
//    the font asset it would normally use lives in the blob that failed to load.

use concinnity_toolchain::{BinaryTargets, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

// Rasterisation size of the embedded atlas. The field is signed-distance, so a
// single size scales cleanly over the range the error screen needs.
const ERROR_SCREEN_FONT_PX: u32 = 24;

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, BinaryTargets::None);
    bake_error_screen_font();
}

// Compile the bundled face into `OUT_DIR`, where `error_screen::font` embeds it.
fn bake_error_screen_font() {
    let payload = concinnity_font::compile(
        concinnity_font::BUILTIN_FONT_BYTES,
        ERROR_SCREEN_FONT_PX,
        concinnity_font::BUILTIN_FONT_FILE,
    )
    .expect("bundled face compiles");

    let out = std::path::Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR is set"))
        .join("error_screen_font.bin");
    std::fs::write(&out, payload).expect("write baked font atlas");
    println!("cargo:rerun-if-changed=build.rs");
}
