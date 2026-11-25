//! Shader source assembly and the compilation caches shared by the device
//! backends.

// What a compiled artifact depends on beyond its source, and the stamp a
// shipped shader file carries.
pub(crate) mod stamp;

// The runtime cache segment both caches below write into, and the checkpoints
// at which it reaches disk.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod runtime_cache;

// Disk cache for shader binaries compiled at runtime: the built-ins every
// backend compiles on first use, and the Metal raymarch libraries assembled
// from world-authored SdfVolume fragments.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod cache;

// Scratch directory for the shader compilers that work on files.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod compiler_work;

// The compile call for each backend's target: SPIR-V, DXIL, MSL or a metallib.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod compile;

// The engine shader artifacts an export ships, and compiling them.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod shipped;

// The shipped-else-cached-else-compiled fetch every backend's built-in
// programs go through.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod builtin;

// Shared source assembly for the single-source shaders every backend draws
// from.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod raymarch_source;
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod source;
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod surface_source;

// Disk persistence for driver pipeline blobs (VkPipelineCache, D3D12 pipeline
// library). Metal needs none: its libraries are shipped or cached above,
// and the OS maintains the per-app pipeline binary cache.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) mod pipeline_cache;

// The host binding layouts checked against what the shared sources declare.
#[cfg(test)]
mod abi_tests;
