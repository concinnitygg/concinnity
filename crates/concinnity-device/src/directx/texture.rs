// D3D12 resource creation helpers.
// All texture uploads use an upload heap (CPU-visible) that is copied to a
// default heap (GPU-local) via CopyTextureRegion on a one-shot command list.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::mipmap;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::depth::DEPTH_CLEAR;
use concinnity_engine::render::output::color_lut::{IDENTITY_LUT, IDENTITY_LUT_SIZE};
use concinnity_engine::render::render_graph::FAR_SHADOW_MAP_RESTS;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::allocator::{DeviceAllocator, PooledBuffer, PooledTexture};
use super::com;
use super::error::map_hresult;
use crate::directx::depth::{optimized_clear, optimized_clear_of};
use crate::directx::descriptor_slot::SrvSlot;

// GPU resource handle

// A D3D12 texture plus the descriptors it binds through. `R` is how the texture
// is held: pooled textures carry their placement lease, while the GPU-written
// depth arrays that stay committed carry a bare resource.
pub(super) struct GpuResource<R = PooledTexture> {
    pub resource: R,
    // CPU descriptor handle for the SRV (zero/invalid for buffers that don't need one).
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    // GPU descriptor handle for the SRV.
    pub srv_gpu: SrvSlot,
}

// One-shot command list helper

// Submit a freshly allocated command list without waiting. The caller keeps the
// list, its allocator and what it references alive until a fence proves the
// GPU retired it.
pub(super) fn one_shot_submit_nowait<F>(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    f: F,
) -> RenderResult<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>
where
    F: FnOnce(&ID3D12GraphicsCommandList),
{
    let allocator: ID3D12CommandAllocator =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| map_hresult(e.code(), "one_shot allocator"))?;

    let cmd: ID3D12GraphicsCommandList =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
            .map_err(|e| map_hresult(e.code(), "one_shot cmd list"))?;

    f(&cmd);

    // SAFETY: the command list is live and in the recording state, which is what `Close` requires.
    unsafe { cmd.Close() }.map_err(|e| map_hresult(e.code(), "one_shot close"))?;

    let cmd_list: ID3D12CommandList = cmd
        .cast()
        .map_err(|e| map_hresult(e.code(), "one_shot cast"))?;
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
) -> RenderResult<()>
where
    F: FnOnce(&ID3D12GraphicsCommandList),
{
    let _keep_alive = one_shot_submit_nowait(device, queue, f)?;
    wait_for_queue(device, queue)
}

// Buffer helpers

// Create a default-heap buffer with `ALLOW_UNORDERED_ACCESS`, suitable for a
// compute shader to write through a UAV. It starts in `COMMON`, as every
// buffer does, and promotes on first use.
pub(super) fn create_uav_buffer(device: &ID3D12Device, size: u64) -> RenderResult<ID3D12Resource> {
    committed_uav_buffer(device, size, D3D12_RESOURCE_STATE_COMMON)
}

// Create an acceleration-structure backing buffer, which alone among buffers
// must start, and stays, in `RAYTRACING_ACCELERATION_STRUCTURE`.
pub(super) fn create_acceleration_structure_buffer(
    device: &ID3D12Device,
    size: u64,
) -> RenderResult<ID3D12Resource> {
    committed_uav_buffer(
        device,
        size,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    )
}

fn committed_uav_buffer(
    device: &ID3D12Device,
    size: u64,
    initial_state: D3D12_RESOURCE_STATES,
) -> RenderResult<ID3D12Resource> {
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            initial_state,
            None,
            &raw mut resource,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create_uav_buffer"))?;
    resource.ok_or_else(|| RenderError::Other("create_uav_buffer returned None".into()))
}

// Upload raw bytes to a GPU-local buffer via a temporary upload heap.
// Returns the device-local buffer.
pub(super) fn upload_buffer(
    alloc: &DeviceAllocator,
    data: &[u8],
    usage_state: D3D12_RESOURCE_STATES,
) -> RenderResult<PooledBuffer> {
    upload_buffer_padded(alloc, data, data.len() as u64, usage_state)
}

// `upload_buffer` with the destination grown to `size` and the pad zeroed, for
// a buffer read in wider units than its data (the skinned index buffer, read
// as u32 words by the RT hit path).
pub(super) fn upload_buffer_padded(
    alloc: &DeviceAllocator,
    data: &[u8],
    size: u64,
    usage_state: D3D12_RESOURCE_STATES,
) -> RenderResult<PooledBuffer> {
    let size = size.max(data.len() as u64).max(4);

    let upload = alloc.alloc_buffer(
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    // Map and copy.
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), "upload map"))?;
    // SAFETY: `Map` returned a CPU-visible mapping of the whole `size`-byte upload buffer, `data`
    // is a distinct live allocation, and `data.len() <= size` holds by the clamp above.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len()) };
    let pad = usize::saturating_from(size) - data.len();
    if pad > 0 {
        // SAFETY: `data.len() < size`, so the offset stays inside the `size`-byte mapping.
        let tail = unsafe { ptr.cast::<u8>().add(data.len()) };
        // SAFETY: `tail` plus `pad` bytes ends exactly at the end of the mapping.
        unsafe { std::ptr::write_bytes(tail, 0, pad) };
    }
    // SAFETY: unmaps subresource 0 of `upload`, mapped above; the writes are done.
    unsafe { upload.Unmap(0, None) };

    // Buffers are always created in COMMON regardless of requested state, so
    // pass COMMON explicitly to avoid the debug layer warning.
    let dest = alloc.alloc_buffer(size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON)?;

    one_shot_submit(alloc.device(), alloc.queue(), |cmd| {
        // SAFETY: the one-shot list is recording, both buffers are `size` bytes, and both outlive
        // the fence-waited submit.
        unsafe { cmd.CopyBufferRegion(&*dest, 0, &*upload, 0, size) };
        // CopyBufferRegion implicitly promotes the buffer COMMON -> COPY_DEST,
        // so the transition barrier's before-state must be COPY_DEST.
        let barrier = transition_barrier(&dest, D3D12_RESOURCE_STATE_COPY_DEST, usage_state);
        // SAFETY: the list is recording, and `dest` is a live local.
        unsafe { cmd.ResourceBarrier(&[barrier]) };
    })?;

    Ok(dest)
}

// Texture helpers

// A streamed texture swap's debris, parked until pending lists stop sampling
// the replaced resource and the upload has retired. Held only to release.
pub(super) struct StreamedUploadRetire {
    #[expect(
        dead_code,
        reason = "held only so dropping the entry releases the COM reference"
    )]
    pub texture: PooledTexture,
    #[expect(
        dead_code,
        reason = "held only so dropping the entry releases the COM reference"
    )]
    pub upload: PooledBuffer,
    #[expect(
        dead_code,
        reason = "held only so dropping the entry releases the COM reference"
    )]
    pub allocator: ID3D12CommandAllocator,
    #[expect(
        dead_code,
        reason = "held only so dropping the entry releases the COM reference"
    )]
    pub cmd: ID3D12GraphicsCommandList,
}

// The transient resources a deferred texture upload leaves in flight.
pub(super) struct UploadInFlight {
    pub upload: PooledBuffer,
    pub allocator: ID3D12CommandAllocator,
    pub cmd: ID3D12GraphicsCommandList,
}

// Where an uploaded texture rests once its copy lands: readable by pixel and
// non-pixel shaders alike, so a compute kernel samples it as a draw does.
pub(super) const SAMPLED_STATE: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;

// Upload RGBA pixels to an RGBA8 texture without waiting: the final transition
// orders every later submission after the copy. Returns the resource; bind it
// with `write_texture_srv`.
pub(super) fn upload_texture_resource_deferred(
    alloc: &DeviceAllocator,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> RenderResult<(PooledTexture, UploadInFlight)> {
    let base = (width as usize) * (height as usize) * 4;
    if pixels.len() < base {
        return Err(RenderError::Other(format!(
            "pixel data too short for {}x{} texture ({} bytes, need {})",
            width,
            height,
            pixels.len(),
            base
        )));
    }

    // Box-filtered mip chain for the backend's own init-time textures (fallbacks,
    // text atlases); scene textures arrive with their chain cooked.
    let chain = mipmap::generate_mip_chain(width, height, pixels);
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
fn dxgi_texture_format(format: concinnity_engine::bake::texture::TextureFormat) -> DXGI_FORMAT {
    use concinnity_engine::bake::texture::TextureFormat;
    match format {
        TextureFormat::Rgba8 => DXGI_FORMAT_R8G8B8A8_UNORM,
        TextureFormat::Bc1 => DXGI_FORMAT_BC1_UNORM,
        TextureFormat::Bc3 => DXGI_FORMAT_BC3_UNORM,
        TextureFormat::Bc5 => DXGI_FORMAT_BC5_UNORM,
        TextureFormat::Bc7 => DXGI_FORMAT_BC7_UNORM,
    }
}

// Upload a decoded texture into a 2-D resource, every level exactly as the image
// carries them: RGBA8 with its cooked chain, or a block-compressed
// (BC1/BC3/BC5/BC7) container chain. Call `write_texture_srv` to bind the
// result, which picks the view format back off the resource.
pub(super) fn upload_texture_image_deferred(
    alloc: &DeviceAllocator,
    image: &concinnity_engine::bake::texture::TextureImage,
) -> RenderResult<(PooledTexture, UploadInFlight)> {
    image.validate().map_err(RenderError::Other)?;
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
    image: &concinnity_engine::bake::texture::TextureImage,
) -> RenderResult<PooledTexture> {
    let (texture, in_flight) = upload_texture_image_deferred(alloc, image)?;
    wait_for_queue(alloc.device(), alloc.queue())?;
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
) -> RenderResult<(PooledTexture, UploadInFlight)> {
    let device = alloc.device();
    let base = levels
        .first()
        .ok_or_else(|| RenderError::Other("texture upload has no mip level".into()))?;
    let (width, height) = (base.width, base.height);
    let mip_count = u32::saturating_from(levels.len());

    // Texture resource (pooled default heap, copy-dest initially), full mip chain.
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width),
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: u16::saturating_from(mip_count),
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
            &raw const desc,
            0,
            mip_count,
            0,
            Some(layouts.as_mut_ptr()),
            Some(row_counts.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&raw mut total_size),
        );
    }

    // Upload heap holding every mip packed at its footprint offset.
    let upload = alloc.alloc_buffer(
        total_size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&raw mut map_ptr)) }
        .map_err(|e| map_hresult(e.code(), "upload tex map"))?;
    for (m, level) in levels.iter().enumerate() {
        let src_row = usize::saturating_from(row_sizes[m]);
        let rows = row_counts[m] as usize;
        let needed = src_row * rows;
        if level.data.len() < needed {
            // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping
            // past this call.
            unsafe { upload.Unmap(0, None) };
            return Err(RenderError::Other(format!(
                "texture mip {} ({}x{}) is {} bytes, need {}",
                m,
                level.width,
                level.height,
                level.data.len(),
                needed
            )));
        }
        let dst_pitch = layouts[m].Footprint.RowPitch as usize;
        let base_off = usize::saturating_from(layouts[m].Offset);
        for row in 0..rows {
            let src = &level.data[row * src_row..(row + 1) * src_row];
            // SAFETY: `map_ptr` is the base of an UPLOAD buffer sized by `GetCopyableFootprints`
            // for every mip, and `base_off + row * dst_pitch` is the start of a row inside this
            // mip's footprint, so the offset stays in bounds.
            let dst = unsafe { map_ptr.cast::<u8>().add(base_off + row * dst_pitch) };
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
        }
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    // Copy each mip, then transition everything to shader-read.
    let (allocator, cmd) = one_shot_submit_nowait(device, alloc.queue(), |cmd| {
        let mut src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&*upload),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layouts[0],
            },
        };
        let mut dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&*texture),
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
            unsafe { cmd.CopyTextureRegion(&raw const dst, 0, 0, 0, &raw const src, None) };
        }
        let barrier = transition_barrier(&texture, D3D12_RESOURCE_STATE_COPY_DEST, SAMPLED_STATE);
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
) -> RenderResult<PooledTexture> {
    let (texture, in_flight) = upload_texture_resource_deferred(alloc, width, height, pixels)?;
    wait_for_queue(alloc.device(), alloc.queue())?;
    drop(in_flight);
    Ok(texture)
}

// Block until every list submitted to `queue` so far has executed.
fn wait_for_queue(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> RenderResult<()> {
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| map_hresult(e.code(), "upload fence"))?;
    let event =
        // SAFETY: an auto-reset, initially unsignaled event with no name and no security
        // attributes; the call borrows nothing.
        unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
            .map_err(|e| map_hresult(e.code(), "upload event"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    unsafe { queue.Signal(&fence, 1) }.map_err(|e| map_hresult(e.code(), "upload signal"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    if unsafe { fence.GetCompletedValue() } < 1 {
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| map_hresult(e.code(), "upload set event"))?;
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
    let mip_levels = u32::from(desc.MipLevels);
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
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv_desc), srv_cpu) };
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
    srv_gpu: SrvSlot,
) -> RenderResult<GpuResource> {
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
) -> RenderResult<PooledTexture> {
    upload_texture_resource(alloc, 1, 1, &[255u8, 255, 255, 255])
}

// Create a 1×1 flat-normal RGBA texture (tangent-space no-op 128,128,255,255).
pub(super) fn create_fallback_flat_normal_resource(
    alloc: &DeviceAllocator,
) -> RenderResult<PooledTexture> {
    upload_texture_resource(alloc, 1, 1, &[128u8, 128, 255, 255])
}

// A 1x1x1 R32_FLOAT Texture2DArray holding the far plane, so every shadow
// compare is lit; the same binding type as a real shadow map.
pub(super) fn create_fallback_shadow_array(
    alloc: &DeviceAllocator,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: SrvSlot,
) -> RenderResult<GpuResource<ID3D12Resource>> {
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &raw mut tex_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create fallback shadow array"))?;
    let texture = tex_opt
        .ok_or_else(|| RenderError::Other("create fallback shadow array returned None".into()))?;

    let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    unsafe {
        device.GetCopyableFootprints(
            &raw const desc,
            0,
            1,
            0,
            Some(&raw mut layout),
            None,
            None,
            None,
        );
    }

    let upload = alloc.alloc_buffer(
        u64::from(layout.Footprint.RowPitch),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&raw mut map_ptr)) }
        .map_err(|e| map_hresult(e.code(), "map fallback shadow array"))?;
    // SAFETY: `map_ptr` is the mapping from the call above of a buffer at least one row long, and
    // D3D12 mappings are page-aligned, so the `f32` write is in bounds and aligned.
    unsafe { *map_ptr.cast::<f32>() = DEPTH_CLEAR };
    // SAFETY: unmaps subresource 0 of `upload`, mapped above; nothing keeps the mapping.
    unsafe { upload.Unmap(0, None) };

    one_shot_submit(device, alloc.queue(), |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&*upload),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: the one-shot list is recording, the texture was created in COPY_DEST, and both
        // locations borrow resources that outlive the fence-waited submit.
        unsafe { cmd.CopyTextureRegion(&raw const dst, 0, 0, 0, &raw const src, None) };
        let barrier = transition_barrier(
            &texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the list is recording, and the copy above is ordered before the transition.
        unsafe { cmd.ResourceBarrier(&[barrier]) };
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
    unsafe { device.CreateShaderResourceView(&texture, Some(&raw const srv_desc), srv_cpu) };

    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Create the main depth buffer. `shader_readable` keeps it bindable as an SRV
// for the decal pass's world reconstruction, at a Hi-Z cost.
pub(super) fn create_main_depth_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    sample_count: u32,
    shader_readable: bool,
) -> RenderResult<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = optimized_clear();
    let mut flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if !shader_readable {
        flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width),
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&raw const clear_value),
            &raw mut tex_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create main depth texture"))?;
    let texture = tex_opt
        .ok_or_else(|| RenderError::Other("create main depth texture returned None".into()))?;

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
    unsafe { device.CreateDepthStencilView(&texture, Some(&raw const dsv_desc), dsv_cpu) };

    Ok(texture)
}

// Create the far shadow's `size`-square D16 map with its DSV written at
// `dsv_cpu`. Created in the state it rests readable in between frames
// (`FAR_SHADOW_MAP_RESTS`), where the graph opens every frame.
pub(super) fn create_far_shadow_map(
    device: &ID3D12Device,
    size: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> RenderResult<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = optimized_clear_of(DXGI_FORMAT_D16_UNORM);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size),
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R16_TYPELESS,
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            super::barrier_translate::resting_read_state(FAR_SHADOW_MAP_RESTS),
            Some(&raw const clear_value),
            &raw mut tex_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create far shadow map"))?;
    let texture =
        tex_opt.ok_or_else(|| RenderError::Other("create far shadow map returned None".into()))?;
    let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D16_UNORM,
        ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
        Flags: D3D12_DSV_FLAG_NONE,
        Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateDepthStencilView(&texture, Some(&raw const dsv_desc), dsv_cpu) };
    Ok(texture)
}

// Write the far shadow map's SRV at `cpu`: its D16 map read as R16_UNORM, or
// without one a null descriptor of the same shape, which nothing samples since
// the far lanes are off.
pub(super) fn write_far_shadow_srv(
    device: &ID3D12Device,
    map: Option<&ID3D12Resource>,
    cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R16_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names, if any, are live for the call, and
    // the destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(map, Some(&raw const desc), cpu) };
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
    srv_gpu: SrvSlot,
) -> RenderResult<(
    GpuResource<ID3D12Resource>,
    Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
)> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = optimized_clear();
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size),
        Height: size,
        DepthOrArraySize: u16::saturating_from(layers),
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            // Rest sampled, as the graph's producer barrier expects on frame 0.
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&raw const clear_value),
            &raw mut tex_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create shadow map array"))?;
    let texture = tex_opt
        .ok_or_else(|| RenderError::Other("create shadow map array returned None".into()))?;

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
        unsafe { device.CreateDepthStencilView(&texture, Some(&raw const dsv_desc), dsv_cpu) };
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
    unsafe { device.CreateShaderResourceView(&texture, Some(&raw const srv_desc), srv_cpu) };

    Ok((
        GpuResource {
            resource: texture,
            srv_cpu,
            srv_gpu,
        },
        dsvs,
    ))
}

// Off-screen HDR color format. The main + instanced passes render
// linear-light HDR into a target of this format; the composite pass tonemaps
// it down to the swapchain backbuffer.
pub(super) const HDR_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Create the off-screen HDR color render target the main pass draws into.
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
) -> RenderResult<ID3D12Resource> {
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
        Width: u64::from(width),
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&raw const clear_value),
            &raw mut res_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create hdr color target"))?;
    let res = res_opt.ok_or_else(|| RenderError::Other("create hdr color returned None".into()))?;

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
    unsafe { device.CreateRenderTargetView(&res, Some(&raw const rtv_desc), rtv_cpu) };

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
) -> RenderResult<ID3D12Resource> {
    create_hdr_sampled_target(device, width, height, [0.0; 4])
}

// Create a single-sample HDR target that is both rendered into and sampled,
// committed with the `clear_color` its pass clears to so the clear takes the
// fast path (and the debug layer sees no mismatch). `create_hdr_resolve_target`
// is the same texture for a resolve destination, which is never cleared.
pub(super) fn create_hdr_sampled_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    clear_color: [f32; 4],
) -> RenderResult<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    // `ALLOW_RENDER_TARGET` so the projected-decal pass can flip the
    // resolved target back to RENDER_TARGET to stamp decals onto the
    // scene, and so a single-sampled planar mirror can render straight into it.
    let clear_value = D3D12_CLEAR_VALUE {
        Format: HDR_FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear_color },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width),
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&raw const clear_value),
            &raw mut res_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create hdr sampled target"))?;
    res_opt.ok_or_else(|| RenderError::Other("create hdr sampled target returned None".into()))
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
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv_desc), srv_cpu) };
}

// Single-sample color render targets
//
// Used by the transparent pass's layers and the RT reflection output; the
// shared post passes create theirs through `post/post_device.rs`.

// A single-sample color render target created in PIXEL_SHADER_RESOURCE so it
// can be bound before its first render, cleared to transparent black. The
// caller writes the RTV and SRV.
pub(super) fn create_rt_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
) -> RenderResult<ID3D12Resource> {
    create_rt_target_with_clear(device, width, height, format, [0.0; 4])
}

// `create_rt_target` with `clear_color` as the optimized clear value, which
// must match the per-frame clear or D3D12 takes a slower path.
pub(super) fn create_rt_target_with_clear(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    clear_color: [f32; 4],
) -> RenderResult<ID3D12Resource> {
    create_rt_resource(
        device,
        (width, height, 1),
        format,
        clear_color,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    )
}

// As `create_rt_target`, but created in RENDER_TARGET for a target that is
// drawn into and never read, so it never leaves that state.
pub(super) fn create_rt_sink(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
) -> RenderResult<ID3D12Resource> {
    create_rt_resource(
        device,
        (width, height, 1),
        format,
        [0.0; 4],
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    )
}

// A single-level texture compute kernels write through a UAV and later passes
// sample, created in PIXEL_SHADER_RESOURCE, where it rests between frames.
pub(super) fn create_storage_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
) -> RenderResult<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width.max(1)),
        Height: height.max(1),
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut res_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            None,
            &raw mut res_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create storage target"))?;
    res_opt.ok_or_else(|| RenderError::Other("create storage target returned None".into()))
}

// As `create_rt_target`, with `levels` mip levels, each of which a draw can
// render to through its own RTV.
pub(super) fn create_rt_chain(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    levels: u32,
) -> RenderResult<ID3D12Resource> {
    create_rt_resource(
        device,
        (width, height, levels),
        format,
        [0.0; 4],
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    )
}

// A color render target of `width` x `height` with `levels` mips, created in
// `initial_state`.
fn create_rt_resource(
    device: &ID3D12Device,
    (width, height, levels): (u32, u32, u32),
    format: DXGI_FORMAT,
    clear_color: [f32; 4],
    initial_state: D3D12_RESOURCE_STATES,
) -> RenderResult<ID3D12Resource> {
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
        Width: u64::from(width.max(1)),
        Height: height.max(1),
        DepthOrArraySize: 1,
        MipLevels: u16::saturating_from(levels.clamp(1, u32::from(u16::MAX))),
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
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            initial_state,
            Some(&raw const clear_value),
            &raw mut res_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create rt target"))?;
    res_opt.ok_or_else(|| RenderError::Other("create rt target returned None".into()))
}

// Write a single-sample Texture2D render-target view of the given format.
pub(super) fn write_format_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    rtv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
) {
    write_level_rtv(device, resource, rtv_cpu, format, 0);
}

// Write a render-target view of mip `level` of a Texture2D.
pub(super) fn write_level_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    rtv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
    level: u32,
) {
    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_RTV {
                MipSlice: level,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateRenderTargetView(resource, Some(&raw const rtv_desc), rtv_cpu) };
}

// Write a single-sample Texture2D shader-resource view of the given format.
pub(super) fn write_format_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
) {
    write_levels_srv(device, resource, srv_cpu, format, (0, 1));
}

// Write a single-level Texture2D unordered-access view of the given format.
pub(super) fn write_format_uav(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
) {
    let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateUnorderedAccessView(resource, None, Some(&raw const uav_desc), uav_cpu) };
}

// Write a Texture2D shader-resource view of `count` mip levels from `first`.
pub(super) fn write_levels_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
    (first, count): (u32, u32),
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: first,
                MipLevels: count,
                ..Default::default()
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv_desc), srv_cpu) };
}

// Resource barriers

pub(super) fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    subresource_transition_barrier(
        resource,
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        before,
        after,
    )
}

// As `transition_barrier`, for one subresource (a mip level of a single-slice
// texture) or `D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES`.
pub(super) fn subresource_transition_barrier(
    resource: &ID3D12Resource,
    subresource: u32,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // Leaked back-buffer references accumulate until `ResizeBuffers`
                // rejects the resize ("outstanding buffer references").
                pResource: com::borrowed(resource),
                StateBefore: before,
                StateAfter: after,
                Subresource: subresource,
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
                pResource: com::borrowed(resource),
            }),
        },
    }
}

// Aliasing barrier for `after` with a null before-resource, which assumes
// nothing about the last occupant. `after` must be re-initialized (Clear,
// Discard, Copy) before any non-overwriting use.
pub(super) fn aliasing_barrier(after: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                pResourceAfter: com::borrowed(after),
            }),
        },
    }
}

// IBL textures from one `EnvironmentMap`; `prefilter_mip_count == 0` means none
// and the shader falls back to flat ambient. The resources stay alive while
// their SRVs are referenced.
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
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv_desc), srv_cpu) };
}

// Write a multi-mip RGBA32F TextureCube SRV at the given heap slot.
fn write_cube_srv_mips(
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
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv_desc), srv_cpu) };
}

// Create a 1×1 RGBA32F cube of `value` for every face. Used as the IBL
// fallback when no `EnvironmentMap` is bound; the fragment shader keys off
// `prefilter_mip_count == 0` and skips IBL math, but the cube SRV must still
// resolve to a valid texture.
pub(super) fn create_fallback_cubemap(
    alloc: &DeviceAllocator,
    value: [f32; 4],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: SrvSlot,
) -> RenderResult<GpuResource> {
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

// The two IBL cubes for an EnvironmentMap upload: the irradiance cube and the
// multi-mip prefiltered radiance cube (both RGBA32F).
#[derive(Clone, Copy)]
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
    pub irr_srv_gpu: SrvSlot,
    pub pre_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub pre_srv_gpu: SrvSlot,
}

// Upload an EnvironmentMap payload into a single-mip irradiance cube and a
// multi-mip prefiltered cube, both RGBA32F. `mip_bytes` holds one slice per mip.
pub(super) fn upload_environment_map(
    alloc: &DeviceAllocator,
    payload: EnvironmentMapPayload<'_>,
    descriptors: EnvironmentMapDescriptors,
) -> RenderResult<EnvironmentMapTextures> {
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
        return Err(RenderError::Other(
            "envmap upload: prefilter mip_bytes must not be empty".into(),
        ));
    }
    let irradiance_res = upload_cube_resource(alloc, irradiance_face, 1, irradiance_bytes)
        .map_err(|e| e.context("envmap irradiance"))?;
    write_cube_srv_single_mip(device, &irradiance_res, irr_srv_cpu);

    let prefilter_res = upload_prefilter_cube_resource(alloc, prefilter_face, mip_bytes)
        .map_err(|e| e.context("envmap prefilter"))?;
    write_cube_srv_mips(
        device,
        &prefilter_res,
        u32::saturating_from(mip_bytes.len()),
        pre_srv_cpu,
    );

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
        prefilter_mip_count: u32::saturating_from(mip_bytes.len()),
    })
}

// Create a single-mip RGBA32F TextureCube resource and upload `bytes` (six
// faces in +X,-X,+Y,-Y,+Z,-Z order). Transitions to `SAMPLED_STATE`.
fn upload_cube_resource(
    alloc: &DeviceAllocator,
    face_size: u32,
    mip_count: u32,
    bytes: &[u8],
) -> RenderResult<PooledTexture> {
    let needed = concinnity_engine::bake::environment_map::cube_mip_byte_len(face_size, 0)?;
    if mip_count == 1 && bytes.len() < needed {
        return Err(RenderError::Other(format!(
            "cubemap data too short for face_size {}: {} bytes, need {}",
            face_size,
            bytes.len(),
            needed
        )));
    }

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(face_size),
        Height: face_size,
        DepthOrArraySize: 6,
        MipLevels: u16::saturating_from(mip_count),
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
) -> RenderResult<PooledTexture> {
    let mip_count = u32::saturating_from(mip_bytes.len());
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(face_size),
        Height: face_size,
        DepthOrArraySize: 6,
        MipLevels: u16::saturating_from(mip_count),
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
// Transitions the resource to `SAMPLED_STATE` at the end.
fn upload_face_major_into_cube(
    alloc: &DeviceAllocator,
    texture: &ID3D12Resource,
    desc: &D3D12_RESOURCE_DESC,
    face_size: u32,
    mip_count: u32,
    mip_bytes: &[&[u8]],
) -> RenderResult<()> {
    if mip_bytes.len() < mip_count as usize {
        return Err(RenderError::Other(format!(
            "cube upload has {} mip slabs, need {mip_count}",
            mip_bytes.len()
        )));
    }
    for (mip, slab) in mip_bytes.iter().enumerate().take(mip_count as usize) {
        let needed = concinnity_engine::bake::environment_map::cube_mip_byte_len(face_size, mip)?;
        if slab.len() < needed {
            return Err(RenderError::Other(format!(
                "cube upload mip {mip} too short: {} bytes, need {needed}",
                slab.len()
            )));
        }
    }
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
            Some(&raw mut total_bytes),
        );
    }

    let upload = alloc.alloc_buffer(
        total_bytes.max(4),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&raw mut map_ptr)) }
        .map_err(|e| map_hresult(e.code(), "cube upload map"))?;

    // Layout in D3D12: subresource index = mip + face * MipLevels.
    // Source data layout: per-mip slab `mip_bytes[m]`, face-major within each.
    for mip in 0..mip_count {
        let mip_face_size = face_size >> mip;
        let face_bytes = (mip_face_size as usize) * (mip_face_size as usize) * 16;
        let slab = mip_bytes[mip as usize];
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
                    unsafe { map_ptr.cast::<u8>().add(usize::saturating_from(layout.Offset) + row * row_pitch) };
                // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload,
                // and the source is a separate allocation, so the ranges cannot overlap.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
            }
        }
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    one_shot_submit(device, alloc.queue(), |cmd| {
        for subres in 0..num_subresources {
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: com::borrowed(&*upload),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: layouts[subres as usize],
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: com::borrowed(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: subres,
                },
            };
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.CopyTextureRegion(&raw const dst, 0, 0, 0, &raw const src, None) };
        }
        let barrier = transition_barrier(texture, D3D12_RESOURCE_STATE_COPY_DEST, SAMPLED_STATE);
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[barrier]) };
    })?;

    Ok(())
}

// Color-grading LUT (3D texture)

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
    unsafe { device.CreateShaderResourceView(resource, Some(&raw const srv_desc), srv_cpu) };
}

// Upload a 3D color LUT and write its SRV: `size³ * 4` bytes in red-fastest
// order.
pub(super) fn upload_color_lut(
    alloc: &DeviceAllocator,
    size: u32,
    data: &[u8],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: SrvSlot,
) -> RenderResult<GpuResource> {
    let device = alloc.device();
    let n = size as usize;
    let needed = n * n * n * 4;
    if data.len() < needed {
        return Err(RenderError::Other(format!(
            "color LUT data too short for size {}: {} bytes, need {}",
            size,
            data.len(),
            needed
        )));
    }

    // 3D texture resource (pooled default heap, copy-dest initially).
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: u64::from(size),
        Height: size,
        DepthOrArraySize: u16::saturating_from(size),
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
            &raw const desc,
            0,
            1,
            0,
            Some(&raw mut layout),
            None,
            None,
            Some(&raw mut total_size),
        );
    }

    let upload = alloc.alloc_buffer(
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
    unsafe { upload.Map(0, None, Some(&raw mut map_ptr)) }
        .map_err(|e| map_hresult(e.code(), "color LUT upload map"))?;
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
                map_ptr
                    .cast::<u8>()
                    .add(usize::saturating_from(layout.Offset) + z * slice_pitch + y * dst_pitch)
            };
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
        }
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    // Copy upload → texture, then transition to shader-read.
    one_shot_submit(device, alloc.queue(), |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&*upload),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&*texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: the one-shot list is recording, the new texture rests in COPY_DEST, and the
        // upload buffer and texture both outlive the fence-waited submit.
        unsafe { cmd.CopyTextureRegion(&raw const dst, 0, 0, 0, &raw const src, None) };
        let barrier = transition_barrier(
            &texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the list is recording, and the texture is a live local of this function.
        unsafe { cmd.ResourceBarrier(&[barrier]) };
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
#[expect(
    clippy::cast_ptr_alignment,
    reason = "D3D12 aligns a placed subresource footprint to 512 bytes"
)]
pub(super) fn upload_float_lut(
    alloc: &DeviceAllocator,
    size: u32,
    components: u32,
    texels: &[f32],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: SrvSlot,
) -> RenderResult<GpuResource> {
    let device = alloc.device();
    let n = size as usize;
    let comp = components as usize;
    let needed = n * n * comp;
    if texels.len() < needed {
        return Err(RenderError::Other(format!(
            "float LUT data too short for {size}x{size}x{components}: {} floats, need {needed}",
            texels.len()
        )));
    }
    let format = match components {
        4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        2 => DXGI_FORMAT_R32G32_FLOAT,
        other => {
            return Err(RenderError::Other(format!(
                "unsupported float LUT component count {other}"
            )));
        }
    };

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size),
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
            &raw const desc,
            0,
            1,
            0,
            Some(&raw mut layout),
            None,
            None,
            Some(&raw mut total_size),
        );
    }

    let upload = alloc.alloc_buffer(
        total_size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    // Row-by-row to honor D3D12's row-pitch alignment.
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { upload.Map(0, None, Some(&raw mut map_ptr)) }
        .map_err(|e| map_hresult(e.code(), "float LUT upload map"))?;
    let src_row = n * comp;
    let dst_pitch = layout.Footprint.RowPitch as usize;
    for y in 0..n {
        let src = &texels[y * src_row..y * src_row + src_row];
        let dst =
            // SAFETY: `map_ptr` is the base of an UPLOAD buffer sized by `GetCopyableFootprints`
            // for the whole LUT, and `layout.Offset + y * dst_pitch` is the start of a row inside
            // its footprint. D3D12 hands back a page-aligned mapping and both the footprint offset
            // and the row pitch are multiples of four, so the `f32` cast is aligned.
            unsafe { map_ptr.cast::<u8>().add(usize::saturating_from(layout.Offset) + y * dst_pitch).cast::<f32>() };
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
    }
    // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past this
    // call.
    unsafe { upload.Unmap(0, None) };

    one_shot_submit(device, alloc.queue(), |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&*upload),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&*texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: the one-shot list is recording, the texture starts in COPY_DEST, and `upload`
        // holds its rows in the placed footprint until the fence wait returns.
        unsafe { cmd.CopyTextureRegion(&raw const dst, 0, 0, 0, &raw const src, None) };
        let barrier = transition_barrier(
            &texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the list is recording, and `texture` stays owned by this function.
        unsafe { cmd.ResourceBarrier(&[barrier]) };
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
    unsafe { device.CreateShaderResourceView(&*texture, Some(&raw const srv_desc), srv_cpu) };
    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Upload the identity color LUT, so the composite pass always binds a valid
// Texture3D even when the world declares no `ColorLut`.
pub(super) fn create_fallback_color_lut(
    alloc: &DeviceAllocator,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: SrvSlot,
) -> RenderResult<GpuResource> {
    upload_color_lut(alloc, IDENTITY_LUT_SIZE, &IDENTITY_LUT, srv_cpu, srv_gpu)
}
