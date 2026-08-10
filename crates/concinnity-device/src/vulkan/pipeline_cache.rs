// The device's VkPipelineCache, persisted across launches.
//
// One cache exists per logical device, seeded from `pipeline-cache/vk-<uuid>.bin`
// at init and handed to every pipeline creation in the backend, so a warm
// launch skips the driver's back-end compile of each pipeline. The handle
// lives in a process global rather than on `VkContext` because every path
// that builds pipelines (init, shader hot-reload, lazy wireframe twins, world
// shader buckets) bottoms out in the same builder functions, and a live
// editor `reload_world` inherits the device: the cache simply stays installed
// across the successor context.
//
// The blob is machine code for one GPU and driver. The file name carries the
// device's `pipeline_cache_uuid` and the blob's own header is checked against
// the device before use; a mismatched or unreadable blob is deleted and the
// launch proceeds cold. Nothing here can fail init.

use ash::vk::Handle;
use ash::{Device, vk};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// The installed cache handle, read by every pipeline creation. Zero (null)
// until `install`, so creation sites degrade to uncached before init or after
// shutdown.
static CURRENT: AtomicU64 = AtomicU64::new(0);

static STATE: Mutex<Option<Persisted>> = Mutex::new(None);

struct Persisted {
    file: String,
    // The bytes currently on disk (loaded at install, updated per serialize),
    // so an unchanged cache never rewrites the file.
    disk: Option<Vec<u8>>,
    warm: bool,
}

// The cache handle pipeline creation passes to `create_*_pipelines`.
fn current() -> vk::PipelineCache {
    vk::PipelineCache::from_raw(CURRENT.load(Ordering::Relaxed))
}

// `vkCreateGraphicsPipelines` through the installed cache, timed for the init
// tally. Signature mirrors ash so call sites keep their error mapping.
//
// # Safety
// Same contract as `ash::Device::create_graphics_pipelines`.
pub(in crate::vulkan) unsafe fn create_graphics_pipelines(
    device: &Device,
    infos: &[vk::GraphicsPipelineCreateInfo],
) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)> {
    let started = std::time::Instant::now();
    let result = unsafe { device.create_graphics_pipelines(current(), infos, None) };
    crate::pipeline_cache::note_creation(started.elapsed().as_micros() as u64);
    result
}

// `vkCreateComputePipelines` counterpart of [`create_graphics_pipelines`].
//
// # Safety
// Same contract as `ash::Device::create_compute_pipelines`.
pub(in crate::vulkan) unsafe fn create_compute_pipelines(
    device: &Device,
    infos: &[vk::ComputePipelineCreateInfo],
) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)> {
    let started = std::time::Instant::now();
    let result = unsafe { device.create_compute_pipelines(current(), infos, None) };
    crate::pipeline_cache::note_creation(started.elapsed().as_micros() as u64);
    result
}

// The raw handle for callers that pass a cache to foreign pipeline builders
// (the XeSS SDK). Null before `install`.
pub(in crate::vulkan) fn handle() -> vk::PipelineCache {
    current()
}

// Create the device's pipeline cache, seeded from disk when a blob for this
// device exists and its header matches. No-op when already installed (the
// `reload_world` path, whose successor context reuses the device).
pub(in crate::vulkan) fn install(device: &Device, props: &vk::PhysicalDeviceProperties) {
    let mut state = STATE.lock().expect("pipeline cache state");
    if state.is_some() {
        return;
    }
    let file = file_name(&props.pipeline_cache_uuid);
    let disk = crate::pipeline_cache::load(&file).filter(|blob| {
        let ok = header_matches(
            blob,
            props.vendor_id,
            props.device_id,
            &props.pipeline_cache_uuid,
        );
        if !ok {
            tracing::warn!("pipeline cache: {file} does not match this device, rebuilding cold");
            crate::pipeline_cache::delete(&file);
        }
        ok
    });
    let info = vk::PipelineCacheCreateInfo::default().initial_data(disk.as_deref().unwrap_or(&[]));
    let created = unsafe { device.create_pipeline_cache(&info, None) }.or_else(|e| {
        // A header can match while the driver still rejects the payload (a
        // truncated tail, a driver update with an unchanged UUID). Drop the
        // blob and retry empty.
        tracing::warn!("pipeline cache: driver rejected {file} ({e}), rebuilding cold");
        crate::pipeline_cache::delete(&file);
        let empty = vk::PipelineCacheCreateInfo::default();
        unsafe { device.create_pipeline_cache(&empty, None) }
    });
    match created {
        Ok(cache) => {
            let warm = disk.is_some();
            CURRENT.store(cache.as_raw(), Ordering::Relaxed);
            *state = Some(Persisted { file, disk, warm });
        }
        Err(e) => {
            // Every creation site tolerates a null cache, so init proceeds
            // uncached rather than failing.
            tracing::warn!("pipeline cache: create failed ({e}), pipelines build uncached");
        }
    }
}

// Whether `install` seeded the cache from a valid disk blob; names the init
// tally's disk state.
pub(in crate::vulkan) fn disk_state() -> &'static str {
    match &*STATE.lock().expect("pipeline cache state") {
        Some(p) if p.warm => "warm",
        Some(_) => "cold",
        None => "absent",
    }
}

// Persist the cache if it accumulated new content since the last write. Called at
// the end of init (so a crash later in the session cannot lose the warm-up)
// and again at clean shutdown.
pub(in crate::vulkan) fn serialize(device: &Device) {
    let mut state = STATE.lock().expect("pipeline cache state");
    let Some(persisted) = state.as_mut() else {
        return;
    };
    // On a lost device this returns an error and the write is skipped; the
    // previous blob on disk stays valid.
    let Ok(data) = (unsafe { device.get_pipeline_cache_data(current()) }) else {
        return;
    };
    if crate::pipeline_cache::store_if_grown(&persisted.file, &data, persisted.disk.as_deref()) {
        persisted.disk = Some(data);
    }
}

// Serialize, then destroy the cache and clear the global. Called from the
// final context's Drop (never the outgoing context of a `reload_world`).
pub(in crate::vulkan) fn shutdown(device: &Device) {
    serialize(device);
    let mut state = STATE.lock().expect("pipeline cache state");
    if state.take().is_some() {
        let cache = current();
        CURRENT.store(0, Ordering::Relaxed);
        if cache != vk::PipelineCache::null() {
            unsafe { device.destroy_pipeline_cache(cache, None) };
        }
    }
}

fn file_name(uuid: &[u8; vk::UUID_SIZE]) -> String {
    let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    format!("vk-{hex}.bin")
}

// Validate the blob's spec-defined header (all fields little-endian): u32
// header length, u32 header version (ONE = 1), u32 vendorID, u32 deviceID,
// then the 16-byte pipelineCacheUUID.
fn header_matches(blob: &[u8], vendor_id: u32, device_id: u32, uuid: &[u8; 16]) -> bool {
    if blob.len() < 32 {
        return false;
    }
    let word = |at: usize| u32::from_le_bytes(blob[at..at + 4].try_into().expect("4 bytes"));
    word(0) >= 32
        && word(4) == vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32
        && word(8) == vendor_id
        && word(12) == device_id
        && blob[16..32] == uuid[..]
}

#[cfg(test)]
mod tests {
    use super::*;

    const UUID: [u8; 16] = *b"0123456789abcdef";

    fn blob(length: u32, version: u32, vendor: u32, device: u32, uuid: [u8; 16]) -> Vec<u8> {
        let mut b = Vec::new();
        for word in [length, version, vendor, device] {
            b.extend_from_slice(&word.to_le_bytes());
        }
        b.extend_from_slice(&uuid);
        b.extend_from_slice(&[0xEE; 8]);
        b
    }

    #[test]
    fn a_matching_header_is_accepted() {
        assert!(header_matches(
            &blob(32, 1, 0x106b, 0xf01, UUID),
            0x106b,
            0xf01,
            &UUID
        ));
    }

    #[test]
    fn every_header_field_is_checked() {
        let good = blob(32, 1, 7, 9, UUID);
        assert!(header_matches(&good, 7, 9, &UUID));
        assert!(
            !header_matches(&blob(16, 1, 7, 9, UUID), 7, 9, &UUID),
            "length"
        );
        assert!(
            !header_matches(&blob(32, 2, 7, 9, UUID), 7, 9, &UUID),
            "version"
        );
        assert!(!header_matches(&good, 8, 9, &UUID), "vendor");
        assert!(!header_matches(&good, 7, 10, &UUID), "device");
        assert!(!header_matches(&good, 7, 9, &[0u8; 16]), "uuid");
    }

    #[test]
    fn a_truncated_blob_is_rejected() {
        assert!(!header_matches(&[0u8; 31], 0, 0, &[0u8; 16]));
        assert!(!header_matches(&[], 0, 0, &[0u8; 16]));
    }

    #[test]
    fn the_file_name_is_per_device() {
        assert_eq!(file_name(&UUID), "vk-30313233343536373839616263646566.bin");
        assert_ne!(file_name(&UUID), file_name(&[0u8; 16]));
    }
}
