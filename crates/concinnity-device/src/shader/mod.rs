//! Shader source assembly and the compilation caches shared by the device
//! backends.

// The runtime cache segment both caches below write into, and the checkpoints
// at which it reaches disk.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod runtime_cache;

// Disk cache for shader binaries compiled after build time: the built-ins the
// DirectX and Vulkan backends compile at init, and the Metal raymarch
// libraries assembled from world-authored SdfVolume fragments (the rest of
// Metal precompiles into the binary via the toolchain crate).
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod cache;

// Scratch directory for the shader compilers that work on files.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod compiler_work;

// The compile call for each backend's target: SPIR-V, DXIL, MSL or a metallib.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod compile;

// The embedded-else-cached-else-compiled fetch every backend's built-in
// programs go through.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod builtin;

// Shared source assembly for the single-source shaders every backend draws
// from.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod raymarch_source;
pub(crate) mod source;
pub(crate) mod surface_source;

// Disk persistence for driver pipeline blobs (VkPipelineCache, D3D12 pipeline
// library). Metal needs none: its libraries are precompiled or cached above,
// and the OS maintains the per-app pipeline binary cache.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) mod pipeline_cache;
