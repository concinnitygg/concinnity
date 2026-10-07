//! The graphics-API-independent half of the vendor temporal upscalers the
//! DirectX and Vulkan backends drive: AMD FidelityFX FSR (`ffx_api`), NVIDIA
//! DLSS (NGX) and Intel XeSS. Holds the SDK declarations and constants the two
//! APIs share byte for byte, the context lifetimes, the quality and depth
//! mappings, and the backend fallback order. Each backend supplies what is its
//! own: how the SDK library is loaded, the API-typed resource descriptions and
//! entry points, and the command recording around the dispatch.

mod camera;
#[cfg(ngx_sdk_bundled)]
pub(crate) mod dlss;
mod extent;
pub(crate) mod fsr;
mod library;
mod select;
pub(crate) mod xess;

pub(crate) use camera::UpscaleCamera;
pub(crate) use extent::UpscaleExtent;
pub(crate) use library::{SdkLibrary, entry_point};
#[cfg(backend_vk)]
pub(crate) use select::preferred;
pub(crate) use select::{Availability, ResolvedBackend, UpscaleRequest};
