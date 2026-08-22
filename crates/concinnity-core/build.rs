//! Resolves the rendering backend cfg the same way the client crate's build.rs
//! does. `Platform::current` is the only consumer: it reports which shader
//! source language the target backend consumes.
//!
//!   backend_metal  macOS, default
//!   backend_dx     Windows, default
//!   backend_vk     Linux (always), or macOS / Windows with the `vulkan` feature
//! The choice must stay in lockstep with concinnity-engine/build.rs.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(backend_metal)");
    println!("cargo::rustc-check-cfg=cfg(backend_dx)");
    println!("cargo::rustc-check-cfg=cfg(backend_vk)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();

    let backend = match (target_os.as_str(), vulkan) {
        ("macos", false) => "backend_metal",
        ("windows", false) => "backend_dx",
        _ => "backend_vk",
    };
    println!("cargo::rustc-cfg={backend}");
}
