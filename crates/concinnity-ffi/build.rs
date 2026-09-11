//! Two jobs: resolve the rendering backend into the single cfg the crate gates
//! on, and generate the C header from the extern "C" surface.
//!
//! This crate produces libraries a host links, not an executable, so the
//! toolchain helper's SDK setup bundles no runtime DLLs; that belongs to
//! whichever package owns the final binary.

use std::path::Path;

fn main() {
    concinnity_toolchain::setup_graphics_backend();
    generate_header();
}

// The header is the public API, so it lives beside the libraries an SDK ships
// rather than in OUT_DIR where a packaging step cannot find it.
fn generate_header() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let source = Path::new(&crate_dir).join("src/ffi.rs");
    let out_dir = Path::new(&crate_dir).join("include");
    let header = out_dir.join("concinnity.h");

    println!("cargo::rerun-if-changed=src/ffi.rs");
    println!("cargo::rerun-if-changed=cbindgen.toml");

    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        println!("cargo::warning=cbindgen: cannot create {out_dir:?}: {e}");
        return;
    }
    let config = cbindgen::Config::from_root_or_default(&crate_dir);
    match cbindgen::Builder::new()
        .with_src(&source)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header);
        }
        Err(e) => println!("cargo::warning=cbindgen: failed to generate {header:?}: {e}"),
    }
}
