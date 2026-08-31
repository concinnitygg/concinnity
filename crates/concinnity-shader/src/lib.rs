//! The build-time shader toolchain: the platform compilers that turn a
//! Shader stage's authored source into the bytecode one backend loads -- Metal
//! (`xcrun metal` + `xcrun metallib`), DirectX (the Direct3D `Fxc` compiler),
//! Vulkan (`shaderc`). At most one compiles per build, resolved by build.rs from
//! the backend features into a single backend_* cfg; a build that names none
//! registers no compiler.
//!
//! This is the build-side twin of concinnity-device: the device crate owns
//! runtime GPU submission, this one owns build-time shader production. Both are
//! backend-specific and neither is on the other's path. Keeping them apart is
//! what lets concinnity-cook -- which must run on build hosts with no GPU and no
//! platform compiler -- stay free of backend cfgs and native dependencies.
//!
//! The cook declares the `ShaderToolchain` seam and calls whatever is registered;
//! the binary calls [`install`] once at startup to fill it in.

#[cfg(backend_dx)]
mod directx;
#[cfg(backend_metal)]
mod metal;
#[cfg(backend_vk)]
mod vulkan;

/// Register this build's shader toolchain (and, where the backend ships one, its
/// shader-layout validator) with the cook's build pipeline. Idempotent and
/// first-wins, so every build entry point can call it unconditionally.
///
/// A binary that compiles worlds MUST call this before any build: without it the
/// cook has no compiler and every Shader stage fails with an "no shader toolchain
/// is registered" error.
pub fn install() {
    #[cfg(backend_metal)]
    metal::install();
    #[cfg(backend_dx)]
    directx::install();
    #[cfg(backend_vk)]
    vulkan::install();
}
