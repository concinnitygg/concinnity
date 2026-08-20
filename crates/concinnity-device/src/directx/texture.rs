// D3D12 resource creation helpers.
// All texture uploads use an upload heap (CPU-visible) that is copied to a
// default heap (GPU-local) via CopyTextureRegion on a one-shot command list.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::allocator::{DeviceAllocator, PooledBuffer, PooledTexture};

// GPU resource handle

// A D3D12 texture plus the descriptors it binds through. `R` is how the texture
// is held: pooled textures carry their placement lease, while the GPU-written
// depth arrays that stay committed carry a bare resource.
#[allow(dead_code)]
pub(super) struct GpuResource<R = PooledTexture> {
    pub resource: R,
    // CPU descriptor handle for the SRV (zero/invalid for buffers that don't need one).
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    // GPU descriptor handle for the SRV.
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

// One-shot command list helper

// Record and submit a freshly allocated command list without waiting for it.
// Returns the still-executing list and its allocator; the caller must keep
// both alive (and any resource the list references) until the GPU provably
// retired the work -- either by a fence wait, or because a later frame fence
// on the same in-order queue signalled.
pub(super) fn one_shot_submit_nowait<F>(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    f: F,
) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList), String>
where
    F: FnOnce(&ID3D12GraphicsCommandList),
{
    let allocator: ID3D12CommandAllocator =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| format!("one_shot allocator: {e}"))?;

    let cmd: ID3D12GraphicsCommandList =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
            .map_err(|e| format!("one_shot cmd list: {e}"))?;

    f(&cmd);

    // SAFETY: the command list is live and in the recording state, which is what `Close` requires.
    unsafe { cmd.Close() }.map_err(|e| format!("one_shot close: {e}"))?;

    let cmd_list: ID3D12CommandList = cmd.cast().map_err(|e| format!("one_shot cast: {e}"))?;
    // SAFETY: every command list in the submission is live and closed, and the slice outlives the
    // call.
    unsafe { queue.ExecuteCommandLists(&[Some(cmd_list)]) };
    Ok((allocator, cmd))
}

// Execute f on a freshly allocated command list, submit, and wait for idle.
pub(super) fn one_shot_submit<F>(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    f: F,
) -> Result<(), String>
where
    F: FnOnce(&ID3D12GraphicsCommandList),
{
    let _keep_alive = one_shot_submit_nowait(device, queue, f)?;

    // Fence-wait for completion.
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| format!("one_shot fence: {e}"))?;
    let event =
        // SAFETY: an auto-reset, initially unsignalled event with no name and no security
        // attributes; the call borrows nothing.
        unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
            .map_err(|e| format!("one_shot event: {e}"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    unsafe { queue.Signal(&fence, 1) }.map_err(|e| format!("one_shot signal: {e}"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    if unsafe { fence.GetCompletedValue() } < 1 {
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| format!("one_shot set event: {e}"))?;
        // SAFETY: `event` is the handle created above and is still open.
        unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, u32::MAX) };
    }
    // SAFETY: `event` was created above, every wait on it has returned, and it is closed exactly
    // once.
    unsafe { windows::Win32::Foundation::CloseHandle(event) }.ok();
    Ok(())
}

// Buffer helpers

// Place a buffer of the given heap type inside a pooled heap. Buffers are never
// GPU-written through this path (`create_uav_buffer` is the compute-writable
// one), so they suballocate; see `directx/allocator.rs`.
pub(super) fn create_buffer(
    alloc: &DeviceAllocator,
    size: u64,
    heap_type: D3D12_HEAP_TYPE,
    initial_state: D3D12_RESOURCE_STATES,
) -> Result<PooledBuffer, String> {
    alloc.alloc_buffer(size, heap_type, initial_state)
}

// Create a default-heap buffer with `ALLOW_UNORDERED_ACCESS`, suitable for a
// compute shader to write through a UAV. Used by the compute-cull
// pass for the per-frame indirect-command buffers.
pub(super) fn create_uav_buffer(
    device: &ID3D12Device,
    size: u64,
    initial_state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initial_state,
            None,
            &mut resource,
        )
    }
    .map_err(|e| format!("create_uav_buffer: {e}"))?;
    resource.ok_or_else(|| "create_uav_buffer returned None".to_string())
}

// Upload raw bytes to a GPU-local buffer via a temporary upload heap.
// Returns the device-local buffer.
pub(super) fn upload_buffer(
    alloc: &DeviceAllocator,
    data: &[u8],
    usage_state: D3D12_RESOURCE_STATES,
) -> Result<PooledBuffer, String> {
    upload_buffer_padded(alloc, data, data.len() as u64, usage_state)
}

// As `upload_buffer`, with the destination grown to `size` bytes when that is
// larger than the data. For a buffer a shader addresses in wider units than the
// data's own element type, so its last load reaches past the data's end: the
// skinned u16 index buffer, which the ray-traced hit path reads as packed u32
// words. The pad is zeroed rather than left as whatever the upload heap held.
pub(super) fn upload_buffer_padded(
    alloc: &DeviceAllocator,
    data: &[u8],
    size: u64,
    usage_state: D3D12_RESOURCE_STATES,
) -> Result<PooledBuffer, String> {
    let size = size.max(data.len() as u64).max(4);

    let upload = create_buffer(
        alloc,
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    // Map and copy.
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("upload map: {e}"))?;
    // SAFETY: `Map` returned a CPU-visible mapping of the whole `size`-byte
    // upload buffer, `data` is a distinct live allocation of `data.len()` bytes,
    // and `data.len() <= size` holds by the clamp above, so both the copy and
    // the pad write stay inside the mapping.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        let pad = size as usize - data.len();
        if pad > 0 {
            std::ptr::write_bytes((ptr as *mut u8).add(data.len()), 0, pad);
        }
        upload.Unmap(0, None);
    }

    // Buffers are always created in COMMON regardless of requested state, so
    // pass COMMON explicitly to avoid the debug layer warning.
    let dest = create_buffer(
        alloc,
        size,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
    )?;

    // SAFETY: the command list is in the recording state, and every resource, descriptor and slice
    // these commands name is live for the call.
    one_shot_submit(alloc.device(), alloc.queue(), |cmd| unsafe {
        cmd.CopyBufferRegion(&*dest, 0, &*upload, 0, size);
        // CopyBufferRegion implicitly promotes the buffer COMMON -> COPY_DEST,
        // so the transition barrier's before-state must be COPY_DEST.
        let barrier = transition_barrier(&dest, D3D12_RESOURCE_STATE_COPY_DEST, usage_state);
        cmd.ResourceBarrier(&[barrier]);
    })?;

    Ok(dest)
}

// Texture helpers

// A streamed texture swap's GPU debris, released once `DxContext::stream_frame`
// reaches `retire_at`: the replaced pool resource (pending lists may still
// sample it, and the per-frame flat-pool copies re-point over the next FRAMES
// ticks) plus the upload's staging buffer and one-shot allocator + list (still
// executing when parked; covered by the first frame fence signalled after the
// upload's submission). The handles are held only so dropping the entry
// releases them (COM refcounts), hence never read.
pub(super) struct StreamedUploadRetire {
    #[allow(dead_code)]
    pub texture: PooledTexture,
    #[allow(dead_code)]
    pub upload: PooledBuffer,
    #[allow(dead_code)]
    pub allocator: ID3D12CommandAllocator,
    #[allow(dead_code)]
    pub cmd: ID3D12GraphicsCommandList,
    pub retire_at: u64,
}

// The transient resources a deferred texture upload leaves in flight.
pub(super) struct UploadInFlight {
    pub upload: PooledBuffer,
    pub allocator: ID3D12CommandAllocator,
    pub cmd: ID3D12GraphicsCommandList,
}

// Upload RGBA pixel data to a GPU-local RGBA8_UNORM texture without waiting
// for the copy: the command list is submitted and left executing. The final
// transition to PIXEL_SHADER_RESOURCE orders every later submission on the
// same in-order queue after the copy, so the texture is safe to sample from
// any subsequently submitted frame; only releasing the returned in-flight
// resources needs GPU retirement.
// Returns just the resource; call `write_texture_srv` to bind it into a slot.
// Multiple SRVs may reference the same resource (one per object using it).
pub(super) fn upload_texture_resource_deferred(
    alloc: &DeviceAllocator,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(PooledTexture, UploadInFlight), String> {
    let base = (width as usize) * (height as usize) * 4;
    if pixels.len() < base {
        return Err(format!(
            "pixel data too short for {}x{} texture ({} bytes, need {})",
            width,
            height,
            pixels.len(),
            base
        ));
    }

    // Box-filtered mip chain so the texture minifies through hardware trilinear /
    // aniso selection instead of aliasing from a single mip-0 sample.
    let chain = crate::gfx::mipmap::generate_mip_chain(width, height, pixels);
    let levels: Vec<TextureLevel<'_>> = chain
        .iter()
        .map(|m| TextureLevel {
            width: m.width,
            height: m.height,
            data: &m.pixels,
        })
        .collect();
    upload_texture_levels_deferred(alloc, DXGI_FORMAT_R8G8B8A8_UNORM, &levels)
}

// One mip level handed to `upload_texture_levels_deferred`: its texel
// dimensions plus the tightly packed level bytes (RGBA8 pixels or 4x4 blocks,
// per the upload's DXGI format).
pub(super) struct TextureLevel<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

// DXGI equivalent of a compiled texture payload format.
fn dxgi_texture_format(format: concinnity_cpu::build::texture::TextureFormat) -> DXGI_FORMAT {
    use concinnity_cpu::build::texture::TextureFormat;
    match format {
        TextureFormat::Rgba8 => DXGI_FORMAT_R8G8B8A8_UNORM,
        TextureFormat::Bc1 => DXGI_FORMAT_BC1_UNORM,
        TextureFormat::Bc3 => DXGI_FORMAT_BC3_UNORM,
        TextureFormat::Bc5 => DXGI_FORMAT_BC5_UNORM,
        TextureFormat::Bc7 => DXGI_FORMAT_BC7_UNORM,
    }
}

// Upload a decoded texture into a 2-D resource. RGBA8 images take the CPU
// mip-generation path above; block-compressed images (BC1/BC3/BC5/BC7) upload
// their container mip chain verbatim. Call `write_texture_srv` to bind the
// result, which picks the view format back off the resource.
pub(super) fn upload_texture_image_deferred(
    alloc: &DeviceAllocator,
    image: &concinnity_cpu::build::texture::TextureImage,
) -> Result<(PooledTexture, UploadInFlight), String> {
    use concinnity_cpu::build::texture::TextureFormat;
    if image.format == TextureFormat::Rgba8 {
        let mip = image
            .mips
            .first()
            .ok_or("RGBA8 texture image has no mip level")?;
        return upload_texture_resource_deferred(alloc, mip.width, mip.height, &mip.data);
    }
    let levels: Vec<TextureLevel<'_>> = image
        .mips
        .iter()
        .map(|m| TextureLevel {
            width: m.width,
            height: m.height,
            data: &m.data,
        })
        .collect();
    upload_texture_levels_deferred(alloc, dxgi_texture_format(image.format), &levels)
}

// Synchronous `upload_texture_image_deferred`.
pub(super) fn upload_texture_image(
    alloc: &DeviceAllocator,
    image: &concinnity_cpu::build::texture::TextureImage,
) -> Result<PooledTexture, String> {
    let (texture, in_flight) = upload_texture_image_deferred(alloc, image)?;
    wait_for_upload(alloc.device(), alloc.queue())?;
    drop(in_flight);
    Ok(texture)
}

// Upload pre-built mip levels of any DXGI format into a default-heap texture
// without waiting for the copy. `GetCopyableFootprints` sizes each subresource,
// so the per-row copy below is format-agnostic: for block-compressed formats a
// "row" is a row of 4x4 blocks and the row count is the block-row count.
fn upload_texture_levels_deferred(
    alloc: &DeviceAllocator,
    format: DXGI_FORMAT,
    levels: &[TextureLevel<'_>],
) -> Result<(PooledTexture, UploadInFlight), String> {
    let device = alloc.device();
    let base = levels.first().ok_or("texture upload has no mip level")?;
    let (width, height) = (base.width, base.height);
    let mip_count = levels.len() as u32;

    // Texture resource (pooled default heap, copy-dest initially), full mip chain.
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: mip_count as u16,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let texture = alloc.alloc_texture(
        &desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    // Footprints for every subresource (one per mip).
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); mip_count as usize];
    let mut row_counts = vec![0u32; mip_count as usize];
    let mut row_sizes = vec![0u64; mip_count as usize];
    let mut total_size: u64 = 0;
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            mip_count,
            0,
            Some(layouts.as_mut_ptr()),
            Some(row_counts.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total_size),
        );
    }

    // Upload heap holding every mip packed at its footprint offset.
    let upload = create_buffer(
        alloc,
        total_size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("upload tex map: {e}"))?;
    for (m, level) in levels.iter().enumerate() {
        let src_row = row_sizes[m] as usize;
        let rows = row_counts[m] as usize;
        let needed = src_row * rows;
        if level.data.len() < needed {
            // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping
            // past this call.
            unsafe { upload.Unmap(0, None) };
            return Err(format!(
                "texture mip {} ({}x{}) is {} bytes, need {}",
                m,
                level.width,
                level.height,
                level.data.len(),
                needed
            ));
        }
        let dst_pitch = layouts[m].Footprint.RowPitch as usize;
        let base_off = layouts[m].Offset as usize;
        for row in 0..rows {
            let src = &level.data[row * src_row..(row + 1) * src_row];
            // SAFETY: `map_ptr` is the base of an UPLOAD buffer sized by `GetCopyableFootprints`
            // for every mip, and `base_off + row * dst_pitch` is the start of a row inside this
            // mip's footprint, so the offset stays in bounds.
            let dst = unsafe { (map_ptr as *mut u8).add(base_off + row * dst_pitch) };
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
        }
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    // Copy each mip subresource, then transition all subresources to shader-read.
    // The copy-location structs are created once and reused across the loop (only
    // the footprint / subresource index changes). `pResource` borrows the upload /
    // texture pointer without an AddRef: the field is a `ManuallyDrop`, so a
    // `clone()` would never be released and would leak a reference to the transient
    // upload buffer (a real memory leak) and the destination texture (a VRAM leak
    // under streaming eviction) on every upload. Both outlive the recorded
    // `CopyTextureRegion` calls (the upload buffer rides the returned in-flight
    // handle until the GPU retires the copy).
    let (allocator, cmd) = one_shot_submit_nowait(device, alloc.queue(), |cmd| {
        let mut src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&*upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layouts[0],
            },
        };
        let mut dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&*texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        for m in 0..mip_count {
            src.Anonymous = D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layouts[m as usize],
            };
            dst.Anonymous = D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: m,
            };
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        }
        let barrier = transition_barrier(
            &texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[barrier]) };
    })?;

    Ok((
        texture,
        UploadInFlight {
            upload,
            allocator,
            cmd,
        },
    ))
}

// Synchronous `upload_texture_resource_deferred`: waits for the copy, then
// drops the transient upload resources. The init-time upload paths use this.
pub(super) fn upload_texture_resource(
    alloc: &DeviceAllocator,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<PooledTexture, String> {
    let (texture, in_flight) = upload_texture_resource_deferred(alloc, width, height, pixels)?;
    wait_for_upload(alloc.device(), alloc.queue())?;
    drop(in_flight);
    Ok(texture)
}

// Block until the upload queue drains, so a synchronous upload's transient
// staging resources can be released.
fn wait_for_upload(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Result<(), String> {
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| format!("upload fence: {e}"))?;
    let event =
        // SAFETY: an auto-reset, initially unsignalled event with no name and no security
        // attributes; the call borrows nothing.
        unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
            .map_err(|e| format!("upload event: {e}"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    unsafe { queue.Signal(&fence, 1) }.map_err(|e| format!("upload signal: {e}"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    if unsafe { fence.GetCompletedValue() } < 1 {
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| format!("upload set event: {e}"))?;
        // SAFETY: `event` is the handle created above and is still open.
        unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, u32::MAX) };
    }
    // SAFETY: `event` was created above, every wait on it has returned, and it is closed exactly
    // once.
    unsafe { windows::Win32::Foundation::CloseHandle(event) }.ok();
    Ok(())
}

// Write a Texture2D SRV at the given heap slot, exposing the resource's full
// mip chain so minified samples trilinear-select down it. The view format is
// taken from the resource, so block-compressed pool textures bind through the
// same call as RGBA8 ones.
pub(super) fn write_texture_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    // SAFETY: a property query on a live COM object; it only reads.
    let desc = unsafe { resource.GetDesc() };
    let mip_levels = desc.MipLevels as u32;
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: desc.Format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: mip_levels,
                ..Default::default()
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Upload RGBA pixel data to a GPU-local RGBA8_UNORM texture and write its SRV
// at the given heap slot. Used for resources that bind to a single slot
// (text atlases, etc.). Per-object scene textures use `upload_texture_resource`
// + `write_texture_srv` directly so one resource can feed multiple per-object slots.
pub(super) fn upload_texture(
    alloc: &DeviceAllocator,
    width: u32,
    height: u32,
    pixels: &[u8],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let texture = upload_texture_resource(alloc, width, height, pixels)?;
    write_texture_srv(alloc.device(), &texture, srv_cpu);
    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Create a 1×1 opaque white RGBA texture (no SRV write; caller binds it).
pub(super) fn create_fallback_white_resource(
    alloc: &DeviceAllocator,
) -> Result<PooledTexture, String> {
    upload_texture_resource(alloc, 1, 1, &[255u8, 255, 255, 255])
}

// Create a 1×1 flat-normal RGBA texture (tangent-space no-op 128,128,255,255).
pub(super) fn create_fallback_flat_normal_resource(
    alloc: &DeviceAllocator,
) -> Result<PooledTexture, String> {
    upload_texture_resource(alloc, 1, 1, &[128u8, 128, 255, 255])
}

// Create a 1×1×1 R32_FLOAT Texture2DArray fallback for when no shadow pass is
// configured. Value 0.0 ensures SampleCmpLevelZero (LESS_EQUAL) always passes,
// returning 1.0 (fully lit). R32_FLOAT is required for comparison sampling.
// The SRV is declared as Texture2DArray (ArraySize=1) so the fragment shader's
// binding type stays identical between the disabled and CSM-enabled cases.
pub(super) fn create_fallback_shadow_array(
    alloc: &DeviceAllocator,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource<ID3D12Resource>, String> {
    let device = alloc.device();
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 1,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create fallback shadow array: {e}"))?;
    let texture =
        tex_opt.ok_or_else(|| "create fallback shadow array returned None".to_string())?;

    let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    unsafe {
        device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut layout), None, None, None);
    }

    let upload = create_buffer(
        alloc,
        layout.Footprint.RowPitch as u64,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("map fallback shadow array: {e}"))?;
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe {
        *(map_ptr as *mut f32) = 0.0f32;
        upload.Unmap(0, None);
    }

    // `pResource` borrows the upload / texture pointer without an AddRef: the
    // field is a `ManuallyDrop`, so a `clone()` would never be released and would
    // leak a reference to the transient upload buffer and the destination texture
    // on every upload. Both outlive the synchronous `CopyTextureRegion` call.
    one_shot_submit(device, alloc.queue(), |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&*upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            let barrier = transition_barrier(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            cmd.ResourceBarrier(&[barrier]);
        }
    })?;

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                FirstArraySlice: 0,
                ArraySize: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(&texture, Some(&srv_desc), srv_cpu) };

    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Create the main pass depth buffer. DEPTH_STENCIL only, no SRV.
// `shader_readable` drops the `DENY_SHADER_RESOURCE` flag so the resource
// can also be bound as a `Texture2D[MS]<float>` SRV, needed by the
// projected-decal pass, which samples scene depth to reconstruct world
// positions. The HiZ cost is acceptable for the cases that opt in; the
// SSR / SSAO pre-pass depth buffers leave the flag set (depth-only).
pub(super) fn create_main_depth_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    sample_count: u32,
    shader_readable: bool,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let mut flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if !shader_readable {
        flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_TYPELESS,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&clear_value),
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create main depth texture: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create main depth texture returned None".to_string())?;

    let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        ViewDimension: if sample_count > 1 {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2D
        },
        Flags: D3D12_DSV_FLAG_NONE,
        Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateDepthStencilView(&texture, Some(&dsv_desc), dsv_cpu) };

    Ok(texture)
}

// Create a `layers`-slice Texture2DArray shadow map. Returns the resource
// (initial state DEPTH_WRITE), one DSV cpu handle per slice (written at
// `dsv_cpu_base + i * dsv_stride`), and an SRV pointing at the whole array
// suitable for sampling as a `Texture2DArray<float>` with SampleCmpLevelZero.
pub(super) fn create_shadow_map_array(
    device: &ID3D12Device,
    size: u32,
    layers: u32,
    dsv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    dsv_stride: usize,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<
    (
        GpuResource<ID3D12Resource>,
        Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    ),
    String,
> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: layers as u16,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_TYPELESS,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            // Rest in the sampled state. The graph's Shadow producer barrier
            // transitions this to DEPTH_WRITE before each shadow pass and Main's
            // consumer returns it here, so the cross-frame reset is the graph's
            // producer barrier, not an inline end-of-frame transition. Creating
            // it sampled makes frame 0's producer barrier (sampled -> DEPTH_WRITE)
            // start from the resource's real state.
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&clear_value),
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create shadow map array: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create shadow map array returned None".to_string())?;

    let mut dsvs = Vec::with_capacity(layers as usize);
    for i in 0..layers {
        let dsv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: dsv_cpu_base.ptr + (i as usize) * dsv_stride,
        };
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
            Flags: D3D12_DSV_FLAG_NONE,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: 0,
                    FirstArraySlice: i,
                    ArraySize: 1,
                },
            },
        };
        // SAFETY: the view descriptor and the resource it names are live for the call, and the
        // destination handle addresses a slot this context reserved for the view in a heap it owns.
        unsafe { device.CreateDepthStencilView(&texture, Some(&dsv_desc), dsv_cpu) };
        dsvs.push(dsv_cpu);
    }

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                FirstArraySlice: 0,
                ArraySize: layers,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(&texture, Some(&srv_desc), srv_cpu) };

    Ok((
        GpuResource {
            resource: texture,
            srv_cpu,
            srv_gpu,
        },
        dsvs,
    ))
}

// Off-screen HDR colour format. The main + instanced passes render
// linear-light HDR into a target of this format; the composite pass tonemaps
// it down to the swapchain backbuffer.
pub(super) const HDR_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Create the off-screen HDR colour render target the main pass draws into.
// `sample_count` matches the depth buffer's MSAA; with MSAA off this target
// is single-sample and the composite pass samples it directly. Created in the
// RENDER_TARGET state, with an RTV written at `rtv_cpu`.
pub(super) fn create_hdr_color_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    sample_count: u32,
    rtv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    clear_color: [f32; 4],
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: HDR_FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear_color },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: HDR_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut res_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&clear_value),
            &mut res_opt,
        )
    }
    .map_err(|e| format!("create hdr color target: {e}"))?;
    let res = res_opt.ok_or_else(|| "create hdr color returned None".to_string())?;

    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: if sample_count > 1 {
            D3D12_RTV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_RTV_DIMENSION_TEXTURE2D
        },
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateRenderTargetView(&res, Some(&rtv_desc), rtv_cpu) };

    Ok(res)
}

// Create the single-sample HDR resolve target. The MSAA `create_hdr_color_target`
// resolves into this each frame; the composite pass then samples it. Created in
// the PIXEL_SHADER_RESOURCE state (the per-frame cycle flips it to RESOLVE_DEST
// and back). Only needed when MSAA is on.
pub(super) fn create_hdr_resolve_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    // `ALLOW_RENDER_TARGET` so the projected-decal pass can flip the
    // resolved target back to RENDER_TARGET to stamp decals onto the
    // scene; the resolve copy still works as before.
    let clear_value = D3D12_CLEAR_VALUE {
        Format: HDR_FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: HDR_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut res_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&clear_value),
            &mut res_opt,
        )
    }
    .map_err(|e| format!("create hdr resolve target: {e}"))?;
    res_opt.ok_or_else(|| "create hdr resolve returned None".to_string())
}

// Write an `HDR_FORMAT` Texture2D SRV at the given heap slot so the composite
// pass can sample the HDR scene target.
pub(super) fn write_hdr_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Single-sample colour render targets
//
// Used by the TAA velocity + history images and the SSAO G-buffer / occlusion
// targets. The bloom mip chain is its own family (see post/bloom.rs).

// Create a single-sample colour render target usable as both a render target
// and a sampled texture. Created in the PIXEL_SHADER_RESOURCE state so the
// first frame can bind it before it has been rendered (the TAA velocity
// buffer and the ping-pong history images). The per-frame cycle flips it to
// RENDER_TARGET for the draw and back. Caller writes the RTV + SRV.
// Cleared to transparent black; see `create_rt_target_with_clear` for targets
// whose per-frame clear is a different value.
pub(super) fn create_rt_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource, String> {
    create_rt_target_with_clear(device, width, height, format, [0.0; 4])
}

// As `create_rt_target`, but bakes `clear_color` as the resource's optimized
// clear value. This must match the colour the caller passes to
// ClearRenderTargetView every frame, else D3D12 falls back to a slower clear
// path and warns. Defaulting to transparent black covers most targets;
// non-zero backgrounds (e.g. roughness 1.0) pass their value here.
pub(super) fn create_rt_target_with_clear(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    clear_color: [f32; 4],
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: format,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear_color },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width.max(1) as u64,
        Height: height.max(1),
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut res_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&clear_value),
            &mut res_opt,
        )
    }
    .map_err(|e| format!("create rt target: {e}"))?;
    res_opt.ok_or_else(|| "create rt target returned None".to_string())
}

// Write a single-sample Texture2D render-target view of the given format.
pub(super) fn write_format_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    rtv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
) {
    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateRenderTargetView(resource, Some(&rtv_desc), rtv_cpu) };
}

// Write a single-sample Texture2D shader-resource view of the given format.
pub(super) fn write_format_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Resource barriers

pub(super) fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // Borrow the resource pointer into the barrier without an
                // AddRef. `pResource` is wrapped in `ManuallyDrop`, so a
                // `clone()` here is never released and leaks one reference to
                // the resource on every barrier; against the swapchain back
                // buffers that accumulates until `ResizeBuffers` rejects the
                // resize ("outstanding buffer references"). The caller's
                // `&resource` outlives the `ResourceBarrier` call, so copying
                // the raw pointer (no refcount change) is sound, and the
                // `ManuallyDrop` guarantees it is never released.
                // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object
                // outlives the call, and the `ManuallyDrop` field never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                StateBefore: before,
                StateAfter: after,
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            }),
        },
    }
}

// Unordered-access barrier ordering one UAV write against the next on the same
// resource. Unlike a render-target write, which the pipeline orders by itself,
// consecutive shader writes through a UAV have no implied ordering.
pub(super) fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: `ID3D12Resource` is a transparent wrapper over its COM
                // pointer, so `transmute_copy` yields that pointer and nothing
                // else. It is borrowed rather than cloned because `pResource`
                // sits in a `ManuallyDrop` that is never dropped: an AddRef here
                // would leak one reference on every barrier. The caller's
                // `&resource` outlives the `ResourceBarrier` call that consumes
                // the returned struct, so the raw pointer stays valid. Same
                // contract as `transition_barrier` above.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

// Aliasing barrier announcing that `after` is about to use heap memory another
// placed resource may have just occupied. `pResourceBefore` is left NULL ("any
// resource could have aliased here"), which is the conservative form: it makes
// no assumption about which resource last owned the memory, so it is correct on
// the first frame (nothing has used it yet) and across the single-buffered
// cyclic reuse without tracking the live occupant. After it, `after`'s contents
// are undefined and `after` must be re-initialized (a Clear/Discard/Copy) before
// any non-overwriting use. Mirrors the Vulkan executor's `UNDEFINED -> ...`
// aliasing transition.
pub(super) fn aliasing_barrier(after: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                // Borrow the resource pointer without an AddRef, same rationale
                // as `transition_barrier`: the caller's `&after` outlives the
                // `ResourceBarrier` call and the `ManuallyDrop` never releases it.
                // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object
                // outlives the call, and the `ManuallyDrop` field never releases it.
                pResourceAfter: unsafe { std::mem::transmute_copy(after) },
            }),
        },
    }
}

// IBL textures produced by a single `EnvironmentMap` asset. Mirrors the Metal
// `EnvironmentMapTextures` shape so the fragment-shader code stays portable.
// `prefilter_mip_count == 0` is the runtime signal for "IBL disabled"; the
// fragment shader keys off it and falls back to the legacy ambient path.
// The `irradiance` / `prefilter` fields hold the COM resources alive while
// their SRVs are referenced via the shader-visible descriptor heap; the SRV
// GPU handles are read by `draw.rs`, not the GpuResource itself.
#[allow(dead_code)]
pub(super) struct EnvironmentMapTextures {
    pub irradiance: GpuResource,
    pub prefilter: GpuResource,
    pub prefilter_mip_count: u32,
}

// Write a TextureCube SRV (1 mip) at the given heap slot.
fn write_cube_srv_single_mip(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCube: D3D12_TEXCUBE_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Write a multi-mip TextureCube SRV at the given heap slot. `pub(super)` so the
// reflection-probe init fill + install (`directx/probe.rs`) can point a probe cube
// array slot at the sky prefilter (init) or a baked probe cube (install).
pub(super) fn write_cube_srv_mips(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    mip_count: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCube: D3D12_TEXCUBE_SRV {
                MostDetailedMip: 0,
                MipLevels: mip_count,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Create a 1×1 RGBA32F cube of `value` for every face. Used as the IBL
// fallback when no `EnvironmentMap` is bound; the fragment shader keys off
// `prefilter_mip_count == 0` and skips IBL math, but the cube SRV must still
// resolve to a valid texture.
pub(super) fn create_fallback_cubemap(
    alloc: &DeviceAllocator,
    value: [f32; 4],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let face_bytes = [value; 1]; // 16 bytes = one RGBA32F pixel per face
    let mut all_faces = Vec::with_capacity(6 * 16);
    for _ in 0..6 {
        for v in &face_bytes {
            all_faces.extend_from_slice(&v[0].to_le_bytes());
            all_faces.extend_from_slice(&v[1].to_le_bytes());
            all_faces.extend_from_slice(&v[2].to_le_bytes());
            all_faces.extend_from_slice(&v[3].to_le_bytes());
        }
    }
    let resource = upload_cube_resource(alloc, 1, 1, &all_faces)?;
    write_cube_srv_single_mip(alloc.device(), &resource, srv_cpu);
    Ok(GpuResource {
        resource,
        srv_cpu,
        srv_gpu,
    })
}

// Upload a six-face HDR cubemap from a CubemapTexture payload. `bytes` is the
// raw RGBA32F face-major data emitted by build/cubemap.rs::compile_cubemap_payload:
// 6 * face_size² * 16 bytes in face order +X, -X, +Y, -Y, +Z, -Z. Single-mip.
#[allow(dead_code)]
pub(super) fn upload_cubemap(
    alloc: &DeviceAllocator,
    face_size: u32,
    bytes: &[u8],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let resource = upload_cube_resource(alloc, face_size, 1, bytes)?;
    write_cube_srv_single_mip(alloc.device(), &resource, srv_cpu);
    Ok(GpuResource {
        resource,
        srv_cpu,
        srv_gpu,
    })
}

// Upload an EnvironmentMap payload into two cube textures: a single-mip
// irradiance cube and a multi-mip prefiltered radiance cube. Both are
// RGBA32F TextureCube SRVs.
//
// `irradiance_face` / `prefilter_face` are the mip-0 face sizes. `mip_bytes`
// is one slice per mip in order 0..mip_count; `mip_count` must equal
// `mip_bytes.len()`.

// The two IBL cubes for an EnvironmentMap upload: the irradiance cube and the
// multi-mip prefiltered radiance cube (both RGBA32F).
pub(super) struct EnvironmentMapPayload<'a> {
    // Mip-0 face size of the irradiance cube.
    pub irradiance_face: u32,
    // RGBA32F irradiance cube bytes (6 faces).
    pub irradiance_bytes: &'a [u8],
    // Mip-0 face size of the prefilter cube.
    pub prefilter_face: u32,
    // One slice per prefilter mip, in order 0..mip_count.
    pub mip_bytes: &'a [&'a [u8]],
}

// Descriptor heap slots for the irradiance + prefilter cube SRVs.
#[derive(Clone, Copy)]
pub(super) struct EnvironmentMapDescriptors {
    pub irr_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub irr_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub pre_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub pre_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

pub(super) fn upload_environment_map(
    alloc: &DeviceAllocator,
    payload: EnvironmentMapPayload,
    descriptors: EnvironmentMapDescriptors,
) -> Result<EnvironmentMapTextures, String> {
    let device = alloc.device();
    let EnvironmentMapPayload {
        irradiance_face,
        irradiance_bytes,
        prefilter_face,
        mip_bytes,
    } = payload;
    let EnvironmentMapDescriptors {
        irr_srv_cpu,
        irr_srv_gpu,
        pre_srv_cpu,
        pre_srv_gpu,
    } = descriptors;
    if mip_bytes.is_empty() {
        return Err("envmap upload: prefilter mip_bytes must not be empty".into());
    }
    let irradiance_res = upload_cube_resource(alloc, irradiance_face, 1, irradiance_bytes)
        .map_err(|e| format!("envmap irradiance: {e}"))?;
    write_cube_srv_single_mip(device, &irradiance_res, irr_srv_cpu);

    let prefilter_res = upload_prefilter_cube_resource(alloc, prefilter_face, mip_bytes)
        .map_err(|e| format!("envmap prefilter: {e}"))?;
    write_cube_srv_mips(device, &prefilter_res, mip_bytes.len() as u32, pre_srv_cpu);

    Ok(EnvironmentMapTextures {
        irradiance: GpuResource {
            resource: irradiance_res,
            srv_cpu: irr_srv_cpu,
            srv_gpu: irr_srv_gpu,
        },
        prefilter: GpuResource {
            resource: prefilter_res,
            srv_cpu: pre_srv_cpu,
            srv_gpu: pre_srv_gpu,
        },
        prefilter_mip_count: mip_bytes.len() as u32,
    })
}

// Create a multi-mip prefiltered radiance cube from a reflection-probe ENVM
// payload's prefilter mips and return the bare resource (no SRV). The probe
// capture (`directx/probe.rs`) stores these per probe; the SRVs into the probe
// cube array are written separately when the array is bound to the shaders.
// `mip_bytes[m]` is `6 * (face_size >> m)² * 16` bytes in face-major order.
pub(super) fn upload_probe_prefilter_cube(
    alloc: &DeviceAllocator,
    face_size: u32,
    mip_bytes: &[&[u8]],
) -> Result<PooledTexture, String> {
    upload_prefilter_cube_resource(alloc, face_size, mip_bytes)
}

// Create a single-mip RGBA32F TextureCube resource and upload `bytes` (six
// faces in +X,-X,+Y,-Y,+Z,-Z order). Transitions to PIXEL_SHADER_RESOURCE.
fn upload_cube_resource(
    alloc: &DeviceAllocator,
    face_size: u32,
    mip_count: u32,
    bytes: &[u8],
) -> Result<PooledTexture, String> {
    let face_bytes_mip0 = (face_size as usize) * (face_size as usize) * 16;
    let needed = 6 * face_bytes_mip0 * mip_count as usize;
    if mip_count == 1 && bytes.len() < needed {
        return Err(format!(
            "cubemap data too short for face_size {}: {} bytes, need {}",
            face_size,
            bytes.len(),
            needed
        ));
    }

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: face_size as u64,
        Height: face_size,
        DepthOrArraySize: 6,
        MipLevels: mip_count as u16,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let texture = alloc.alloc_texture(
        &desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    upload_face_major_into_cube(alloc, &texture, &desc, face_size, mip_count, &[bytes])?;
    Ok(texture)
}

// Upload a multi-mip prefilter cube. `mip_bytes[m]` is 6 * (face_size >> m)² * 16 bytes
// in face-major order. Mip 0 corresponds to `face_size`; each subsequent mip halves.
fn upload_prefilter_cube_resource(
    alloc: &DeviceAllocator,
    face_size: u32,
    mip_bytes: &[&[u8]],
) -> Result<PooledTexture, String> {
    let mip_count = mip_bytes.len() as u32;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: face_size as u64,
        Height: face_size,
        DepthOrArraySize: 6,
        MipLevels: mip_count as u16,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let texture = alloc.alloc_texture(
        &desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    upload_face_major_into_cube(alloc, &texture, &desc, face_size, mip_count, mip_bytes)?;
    Ok(texture)
}

// Copy face-major RGBA32F bytes into a 6-slice cube `texture`. For each mip
// `m` (0..mip_count), `mip_bytes[m]` is expected to be
// `6 * (face_size >> m)² * 16` bytes in face order +X,-X,+Y,-Y,+Z,-Z.
// Transitions the resource to PIXEL_SHADER_RESOURCE at the end.
fn upload_face_major_into_cube(
    alloc: &DeviceAllocator,
    texture: &ID3D12Resource,
    desc: &D3D12_RESOURCE_DESC,
    face_size: u32,
    mip_count: u32,
    mip_bytes: &[&[u8]],
) -> Result<(), String> {
    let device = alloc.device();
    let num_subresources = 6 * mip_count;
    let mut layouts: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> =
        vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); num_subresources as usize];
    let mut row_counts: Vec<u32> = vec![0; num_subresources as usize];
    let mut row_sizes: Vec<u64> = vec![0; num_subresources as usize];
    let mut total_bytes: u64 = 0;
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    unsafe {
        device.GetCopyableFootprints(
            desc,
            0,
            num_subresources,
            0,
            Some(layouts.as_mut_ptr()),
            Some(row_counts.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total_bytes),
        );
    }

    let upload = create_buffer(
        alloc,
        total_bytes.max(4),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("cube upload map: {e}"))?;

    // Layout in D3D12: subresource index = mip + face * MipLevels.
    // Source data layout: per-mip slab `mip_bytes[m]`, face-major within each.
    for mip in 0..mip_count {
        let mip_face_size = (face_size >> mip).max(1);
        let face_bytes = (mip_face_size as usize) * (mip_face_size as usize) * 16;
        let slab = mip_bytes[mip as usize];
        if slab.len() < 6 * face_bytes {
            // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping
            // past this call.
            unsafe { upload.Unmap(0, None) };
            return Err(format!(
                "cube upload mip {} too short: {} bytes, need {}",
                mip,
                slab.len(),
                6 * face_bytes
            ));
        }
        for face in 0..6u32 {
            let subres = mip + face * mip_count;
            let layout = &layouts[subres as usize];
            let row_pitch = layout.Footprint.RowPitch as usize;
            let src_row = (mip_face_size as usize) * 16;
            let face_src_offset = (face as usize) * face_bytes;
            for row in 0..mip_face_size as usize {
                let src =
                    &slab[face_src_offset + row * src_row..face_src_offset + (row + 1) * src_row];
                let dst =
                    // SAFETY: `map_ptr` is the base of an UPLOAD buffer sized by
                    // `GetCopyableFootprints` for every cube subresource, and `layout.Offset + row
                    // * row_pitch` is the start of a row inside this face's footprint, so the
                    // offset stays in bounds.
                    unsafe { (map_ptr as *mut u8).add(layout.Offset as usize + row * row_pitch) };
                // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload,
                // and the source is a separate allocation, so the ranges cannot overlap.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
            }
        }
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    // `pResource` borrows the upload / texture pointer without an AddRef: the
    // field is a `ManuallyDrop`, so a `clone()` would never be released and would
    // leak a reference to the transient upload buffer and the destination texture
    // on every subresource copy. Both outlive the synchronous `CopyTextureRegion`
    // calls (`texture` is borrowed from the caller).
    one_shot_submit(device, alloc.queue(), |cmd| {
        for subres in 0..num_subresources {
            let src = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object
                // outlives the call, and the `ManuallyDrop` field never releases it.
                pResource: unsafe { std::mem::transmute_copy(&*upload) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: layouts[subres as usize],
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object
                // outlives the call, and the `ManuallyDrop` field never releases it.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: subres,
                },
            };
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        }
        let barrier = transition_barrier(
            texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[barrier]) };
    })?;

    Ok(())
}

// Colour-grading LUT (3D texture)

// Write a Texture3D R8G8B8A8_UNORM SRV at the given heap slot.
fn write_lut_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Upload a deserialised `ColorLut` payload into a 3D R8G8B8A8_UNORM texture and
// write its Texture3D SRV at the given heap slot. `data` is `size³ * 4` bytes
// in red-fastest, then green, then blue order, the same texel order the
// composite shader samples with the display-referred `(r, g, b)` colour as the
// coordinate. Mirrors `vulkan/texture.rs::upload_color_lut`.
pub(super) fn upload_color_lut(
    alloc: &DeviceAllocator,
    size: u32,
    data: &[u8],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let device = alloc.device();
    let n = size as usize;
    let needed = n * n * n * 4;
    if data.len() < needed {
        return Err(format!(
            "color LUT data too short for size {}: {} bytes, need {}",
            size,
            data.len(),
            needed
        ));
    }

    // 3D texture resource (pooled default heap, copy-dest initially).
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: size as u16,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let texture = alloc.alloc_texture(
        &desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    // Query upload size/layout. A 3D texture is one subresource.
    let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total_size: u64 = 0;
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut layout),
            None,
            None,
            Some(&mut total_size),
        );
    }

    let upload = create_buffer(
        alloc,
        total_size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    // Map and copy row-by-row to match D3D12's row pitch alignment. The placed
    // 3D footprint is `n` depth slices, each `n` rows of `RowPitch` bytes, so
    // the slice pitch is `RowPitch * n`.
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("color LUT upload map: {e}"))?;
    let src_row = n * 4;
    let dst_pitch = layout.Footprint.RowPitch as usize;
    let slice_pitch = dst_pitch * n;
    for z in 0..n {
        for y in 0..n {
            let src_off = (z * n + y) * src_row;
            let src = &data[src_off..src_off + src_row];
            // SAFETY: `map_ptr` is the base of an UPLOAD buffer sized by `GetCopyableFootprints`
            // for the whole volume, and `layout.Offset + z * slice_pitch + y * dst_pitch` is the
            // start of a row inside its footprint, so the offset stays in bounds.
            let dst = unsafe {
                (map_ptr as *mut u8).add(layout.Offset as usize + z * slice_pitch + y * dst_pitch)
            };
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
        }
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    // Copy upload → texture, then transition to shader-read. `pResource` borrows
    // the upload / texture pointer without an AddRef: the field is a `ManuallyDrop`,
    // so a `clone()` would never be released and would leak a reference to the
    // transient upload buffer and the destination texture on every upload. Both
    // outlive the synchronous `CopyTextureRegion` call.
    one_shot_submit(device, alloc.queue(), |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&*upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&*texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            let barrier = transition_barrier(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            cmd.ResourceBarrier(&[barrier]);
        }
    })?;

    write_lut_srv(device, &texture, srv_cpu);
    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Upload a square float lookup table as a 2D texture. `components` selects the
// format: 4 -> RGBA32F, 2 -> RG32F. Used for the two area-light LTC tables,
// which are scene-independent (fitted at build time) and uploaded once at init.
pub(super) fn upload_float_lut(
    alloc: &DeviceAllocator,
    size: u32,
    components: u32,
    texels: &[f32],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let device = alloc.device();
    let n = size as usize;
    let comp = components as usize;
    let needed = n * n * comp;
    if texels.len() < needed {
        return Err(format!(
            "float LUT data too short for {size}x{size}x{components}: {} floats, need {needed}",
            texels.len()
        ));
    }
    let format = match components {
        4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        2 => DXGI_FORMAT_R32G32_FLOAT,
        other => return Err(format!("unsupported float LUT component count {other}")),
    };

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let texture = alloc.alloc_texture(
        &desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;

    let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total_size: u64 = 0;
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut layout),
            None,
            None,
            Some(&mut total_size),
        );
    }

    let upload = create_buffer(
        alloc,
        total_size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    // Row-by-row to honour D3D12's row-pitch alignment.
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("float LUT upload map: {e}"))?;
    let src_row = n * comp;
    let dst_pitch = layout.Footprint.RowPitch as usize;
    for y in 0..n {
        let src = &texels[y * src_row..y * src_row + src_row];
        let dst =
            // SAFETY: `map_ptr` is the base of an UPLOAD buffer sized by `GetCopyableFootprints`
            // for the whole LUT, and `layout.Offset + y * dst_pitch` is the start of a row inside
            // its footprint. D3D12 hands back a page-aligned mapping and both the footprint offset
            // and the row pitch are multiples of four, so the `f32` cast is aligned.
            unsafe { (map_ptr as *mut u8).add(layout.Offset as usize + y * dst_pitch) as *mut f32 };
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    // `pResource` borrows without an AddRef; both resources outlive the
    // synchronous copy (see `upload_color_lut`).
    one_shot_submit(device, alloc.queue(), |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&*upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives
            // the call, and the `ManuallyDrop` field never releases it.
            pResource: unsafe { std::mem::transmute_copy(&*texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            let barrier = transition_barrier(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            cmd.ResourceBarrier(&[barrier]);
        }
    })?;

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(&*texture, Some(&srv_desc), srv_cpu) };
    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Build a 2×2×2 identity colour LUT so the composite pass always binds a valid
// Texture3D even when the world declares no `ColorLut`. With the identity LUT
// the grade is a no-op at any `lut_strength`. Mirrors
// `vulkan/texture.rs::create_fallback_color_lut`.
pub(super) fn create_fallback_color_lut(
    alloc: &DeviceAllocator,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    // Red-fastest, then green, then blue, matching the payload texel order.
    let mut data = Vec::with_capacity(2 * 2 * 2 * 4);
    for b in 0..2u8 {
        for g in 0..2u8 {
            for r in 0..2u8 {
                data.extend_from_slice(&[r * 255, g * 255, b * 255, 255]);
            }
        }
    }
    upload_color_lut(alloc, 2, &data, srv_cpu, srv_gpu)
}
