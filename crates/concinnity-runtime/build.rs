//! The runtime crate (rlib). Resolves the rendering backends through the shared
//! `concinnity-toolchain` helper's `setup_graphics_backends`, exposing one cfg
//! per backend the crate gates on (`backend_metal` / `backend_dx` /
//! `backend_vk`).

// Also bake the bundled face into an SDF atlas the crate embeds, so text can draw
//    with no compiled world data behind it: the startup error screen, which runs
//    when loading that data is what failed, and any TextLabel or TextField naming
//    no Font.

use concinnity_toolchain::setup_graphics_backends;

// Native size of the built-in face: what text naming no Font lays out at before
// its own `scale`. The field is signed-distance and the atlas supersamples, so
// one size covers the range both callers need, and sits near the `Font` asset's
// own 20px default so a font-less label reads like an authored one.
const BUILTIN_FONT_PX: u32 = 24;

fn main() {
    setup_graphics_backends();
    bake_builtin_font();
}

// Compile the bundled face into `OUT_DIR`, where `gfx::builtin_font` embeds it.
#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
fn bake_builtin_font() {
    let payload = concinnity_engine::bake::font::compile(
        concinnity_engine::bake::font::BUILTIN_FONT_BYTES,
        BUILTIN_FONT_PX,
        concinnity_engine::bake::font::BUILTIN_FONT_FILE,
    )
    .expect("bundled face compiles");

    let out = std::path::Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR is set"))
        .join("builtin_font.bin");
    std::fs::write(&out, payload).expect("write baked font atlas");
    println!("cargo:rerun-if-changed=build.rs");
}
