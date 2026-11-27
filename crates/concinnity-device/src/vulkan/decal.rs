//! Projected (deferred) decals for the Vulkan backend. Each decal is drawn
//! as a unit cube (positions in `[-0.5, 0.5]^3`) transformed by its world
//! model matrix and the camera VP; the fragment shader samples the main
//! pass's depth attachment to reconstruct the world-space sample point at
//! each pixel and tests it against the decal's local bounding box,
//! stamping the texture onto whatever fills the box.
//!
//! Runs after the main HDR resolve and before SSR resolve / TAA, so
//! decals are reflected and tracked by the temporal history just like
//! the rest of the scene. Mirrors `src/directx/decal.rs` and
//! `src/metal/decal.rs`.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::geometry::proxy_cube::{CUBE_INDICES, cube_corners};
use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{DecalPass, RenderBackend};
use concinnity_engine::render::decal::DecalRecord;
use concinnity_engine::transform::mat4_inverse;
use std::cell::Cell;
// `DecalView` (per-frame, 144 bytes) is the layout struct shared with the other
// backends; the per-decal `DecalParams` (160 bytes, inside the 256-byte stride
// slot) rides the set's slots. Both mirror `shaders/decal.hlsl`.
use concinnity_engine::render::uniforms::DecalView;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::{Blend, GraphicsPipelineDesc, Raster};
use super::rendering::PassFormats;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout, source_set_bindings};
use super::set_writes::SetWrites;
use super::texture::GpuImage;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

// The descriptor pool reserves a fixed block of `MAX_DECALS` per-decal albedo
// sets at init.
pub(in crate::vulkan) const MAX_DECALS: usize = concinnity_engine::render::decal::MAX_DECALS;

// The unit cube in `[-0.5, 0.5]^3`, the local space a decal's `inv_model` maps into.
const CUBE_VERTS: [[f32; 3]; 8] = cube_corners(0.5);

// Stride of the per-decal params ring: `minUniformBufferOffsetAlignment` is at
// most 256 by spec, so 256 always aligns.
const PARAMS_STRIDE: u64 = 256;

// The decal pipeline and its resources. Set 0 is per frame: the `DecalView`
// UBO (0), the dynamic `DecalParams` ring (1), and the main depth (2). Set 1 is
// per decal: its albedo (0) and sampler (1).
pub(in crate::vulkan) struct DecalResources {
    pub(in crate::vulkan) pipeline: OwnedPipeline,
    pub(in crate::vulkan) pipeline_layout: OwnedPipelineLayout,
    pub(in crate::vulkan) _view_set_layout: OwnedSetLayout,
    pub(in crate::vulkan) _albedo_set_layout: OwnedSetLayout,
    pub(in crate::vulkan) _descriptor_pool: OwnedDescriptorPool,

    // Unit-cube vertex + index buffers (shared across frames).
    pub(in crate::vulkan) vertex_buffer: PooledBuffer,
    pub(in crate::vulkan) index_buffer: PooledBuffer,

    // Per-frame view UBO (DecalView, 144 bytes). Persistently mapped.
    pub(in crate::vulkan) view_ubos: Vec<PooledBuffer>,

    // Per-frame per-decal params ring (PARAMS_STRIDE * MAX_DECALS bytes).
    // Persistently mapped; bound as UNIFORM_BUFFER_DYNAMIC with a
    // per-draw offset of `decal_id * PARAMS_STRIDE`.
    pub(in crate::vulkan) params_ubos: Vec<PooledBuffer>,

    // Per-frame view sets (binding 0 view UBO, 1 params dynamic, 2 depth).
    pub(in crate::vulkan) view_sets: Vec<vk::DescriptorSet>,
    // Per-decal albedo sets (binding 0 albedo, 1 its sampler). Indexed by
    // `decal_id`; tombstoned slots keep their last write, but the
    // executor only binds the set for visible records.
    pub(in crate::vulkan) albedo_sets: Vec<vk::DescriptorSet>,

    // Last-uploaded texture-pool slot per decal id. Used by
    // `rewrite_texture_slot` to detect which decal albedo sets need
    // re-writing when a streamed texture pool entry swaps.
    pub(in crate::vulkan) decal_texture_slots: Cell<[usize; MAX_DECALS]>,
}

// Vulkan handles needed to create the decal pass's GPU resources
// (buffers, one-shot transfer submits). Borrowed for the duration of
// `DecalResources::new`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct DecalDeviceContext<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) device: &'a VkDevice,
    pub(in crate::vulkan) command_pool: vk::CommandPool,
    pub(in crate::vulkan) queue: vk::Queue,
}

// Render-target inputs the decal pass writes into / samples from: the
// resolved HDR color format, the main depth views and the albedo sampler.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct DecalPassTargets<'a> {
    pub(in crate::vulkan) hdr_format: vk::Format,
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
    pub(in crate::vulkan) sampler: vk::Sampler,
}

impl DecalResources {
    // Build the decal pipeline + its dependent resources. Called
    // unconditionally from `VkContext::new` so runtime `add_decal`
    // works from a world that started empty; the cost is one pipeline
    // + small buffers.
    pub(in crate::vulkan) fn new(
        ctx: DecalDeviceContext<'_>,
        targets: DecalPassTargets<'_>,
        frames: usize,
        msaa: bool,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let DecalDeviceContext {
            alloc,
            device,
            command_pool,
            queue,
        } = ctx;
        let DecalPassTargets {
            hdr_format,
            depth_views,
            sampler,
        } = targets;
        let view_set_layout = create_descriptor_set_layout(device, &view_set_bindings())?;
        let albedo_set_layout = create_descriptor_set_layout(device, &source_set_bindings(1))?;
        let pipeline_layout = create_decal_pipeline_layout(
            device,
            view_set_layout.handle(),
            albedo_set_layout.handle(),
        )?;

        let (vert_spv, frag_spv) = compile_decal_shaders(hot_reload, msaa)?;
        let pipeline = create_decal_pipeline(
            device,
            hdr_format,
            pipeline_layout.handle(),
            &vert_spv,
            &frag_spv,
        )?;

        // Unit-cube vertex + index buffers (single device-local upload).
        let vertex_buffer = upload_static_buffer(
            alloc,
            device,
            command_pool,
            queue,
            bytemuck_cast(&CUBE_VERTS),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let index_buffer = upload_static_buffer(
            alloc,
            device,
            command_pool,
            queue,
            bytemuck_cast(&CUBE_INDICES),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        // Per-frame view UBOs (HOST_VISIBLE | HOST_COHERENT, persistently
        // mapped).
        let mut view_ubos = Vec::with_capacity(frames);
        for _ in 0..frames {
            view_ubos.push(alloc.create_buffer(
                size_of::<DecalView>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        // Per-frame per-decal params ring.
        let params_total = PARAMS_STRIDE * MAX_DECALS as u64;
        let mut params_ubos = Vec::with_capacity(frames);
        for _ in 0..frames {
            params_ubos.push(alloc.create_buffer(
                params_total,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        let descriptor_pool = create_decal_descriptor_pool(device, frames)?;

        // Per-frame view sets (one per frame slot).
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout.handle()).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            // The dynamic params binding windows one PARAMS_STRIDE slot at the
            // offset given at bind time.
            SetWrites::new(set)
                .uniform_buffer(0, view_ubos[i].buffer(), size_of::<DecalView>() as u64)
                .buffer(
                    1,
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    params_ubos[i].buffer(),
                    0,
                    PARAMS_STRIDE,
                )
                .sampled_image(2, depth_views[i.min(depth_views.len().saturating_sub(1))])
                .apply(device);
        }

        // Per-decal albedo sets (MAX_DECALS sets, pre-allocated).
        let albedo_layouts: Vec<_> = (0..MAX_DECALS)
            .map(|_| albedo_set_layout.handle())
            .collect();
        let albedo_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &albedo_layouts)?;
        // The sampler never changes, so it is written once here; a decal's
        // albedo image is written when the decal takes the slot.
        for &set in &albedo_sets {
            SetWrites::new(set).sampler(1, sampler).apply(device);
        }

        Ok(Self {
            pipeline,
            pipeline_layout,
            _view_set_layout: view_set_layout,
            _albedo_set_layout: albedo_set_layout,
            _descriptor_pool: descriptor_pool,
            vertex_buffer,
            index_buffer,
            view_ubos,
            params_ubos,
            view_sets,
            albedo_sets,
            decal_texture_slots: Cell::new([usize::MAX; MAX_DECALS]),
        })
    }

    // Re-point each per-frame view set's depth binding (binding 2) at the
    // rebuilt depth view after a resize.
    pub(in crate::vulkan) fn repoint_depth(
        &self,
        device: &VkDevice,
        depth_views: &[vk::ImageView],
    ) {
        for (i, &set) in self.view_sets.iter().enumerate() {
            SetWrites::new(set)
                .sampled_image(2, depth_views[i.min(depth_views.len().saturating_sub(1))])
                .apply(device);
        }
    }

    // Destroy every GPU resource. Called from `VkContext::destroy` after
    // `wait_idle`; the pooled buffers retire through the allocator as their
    // fields clear.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.view_ubos.clear();
        self.params_ubos.clear();
        self.vertex_buffer = PooledBuffer::null();
        self.index_buffer = PooledBuffer::null();
    }
}

// Pipeline construction

// Set 0, per frame: the view UBO, the per-decal params dynamic UBO and the
// main depth.
fn view_set_bindings() -> [Binding; 3] {
    use vk::DescriptorType as T;
    let vert_frag = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, vert_frag),
        (1, T::UNIFORM_BUFFER_DYNAMIC, vert_frag),
        (2, T::SAMPLED_IMAGE, vk::ShaderStageFlags::FRAGMENT),
    ]
}

fn create_decal_pipeline_layout(
    device: &VkDevice,
    view_set_layout: vk::DescriptorSetLayout,
    albedo_set_layout: vk::DescriptorSetLayout,
) -> RenderResult<OwnedPipelineLayout> {
    let set_layouts = [view_set_layout, albedo_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    device
        .create_pipeline_layout(&info)
        .map_err(|e| super::error::map_vk_result(e, "decal pipeline layout"))
}

fn create_decal_descriptor_pool(
    device: &VkDevice,
    frames: usize,
) -> RenderResult<OwnedDescriptorPool> {
    let frames = u32::saturating_from(frames);
    let max_decals = u32::saturating_from(MAX_DECALS);
    let sizes = PoolSizes::default()
        .sets(&view_set_bindings(), frames)
        .sets(&source_set_bindings(1), max_decals)
        .build();
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frames + max_decals)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| super::error::map_vk_result(e, "decal descriptor pool"))
}

fn compile_decal_shaders(hot_reload: bool, msaa: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::DECAL_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::DECAL_FRAG
        .at(msaa)
        .compile(hot_reload)?;
    Ok((vert, frag))
}

// Rebuild the decal graphics pipeline against the existing layout. Used by the
// Vulkan shader hot-reload path. The caller is responsible for destroying the
// previous pipeline only after this call succeeds.
pub(in crate::vulkan) fn rebuild_decal_pipeline(
    device: &VkDevice,
    decals: &DecalResources,
    msaa: bool,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let (vert_spv, frag_spv) = compile_decal_shaders(hot_reload, msaa)?;
    create_decal_pipeline(
        device,
        super::context::HDR_FORMAT,
        decals.pipeline_layout.handle(),
        &vert_spv,
        &frag_spv,
    )
}

// A unit cube with front faces culled, so the back faces still rasterize when
// the camera is inside the decal volume, alpha-blended into the SINGLE-SAMPLE
// resolved HDR regardless of the main pass's MSAA.
fn create_decal_pipeline(
    device: &VkDevice,
    hdr_format: vk::Format,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(12) // vec3 position
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attrs = [vk::VertexInputAttributeDescription::default()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0)];
    GraphicsPipelineDesc {
        raster: Raster {
            cull: vk::CullModeFlags::FRONT,
            ..Raster::default()
        },
        vertex_bindings: &bindings,
        vertex_attributes: &attrs,
        ..GraphicsPipelineDesc::fullscreen(
            vert_spv,
            frag_spv,
            layout,
            PassFormats::color(hdr_format),
            &[Blend::AlphaOver],
        )
    }
    .build(device, "decal")
}

// Helpers for upload + casting

pub(in crate::vulkan) fn upload_static_buffer(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> RenderResult<PooledBuffer> {
    let size = data.len() as vk::DeviceSize;
    let staging = alloc.create_buffer(
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging.write_bytes(0, data);
    let buf = alloc.create_buffer(
        size,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    super::texture::one_shot_submit(device, command_pool, queue, |rec| {
        let region = vk::BufferCopy::default().size(size);
        rec.copy_buffer(
            staging.buffer(),
            buf.buffer(),
            std::slice::from_ref(&region),
        );
    })?;
    Ok(buf)
}

fn bytemuck_cast<T: bytemuck::NoUninit>(s: &[T]) -> &[u8] {
    bytemuck::cast_slice(s)
}

// Encoder

impl VkContext {
    // Encode the projected-decal pass between the main resolve and SSR, so decals
    // are reflected and accumulated like the scene. `vp` is the main pass's
    // jittered view-projection.
    pub(in crate::vulkan) fn encode_decals(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        vp: [[f32; 4]; 4],
        frustum: &Frustum,
    ) {
        let Some(decals) = &self.decal.resources else {
            return;
        };
        // Frustum-cull first so a frame where every live decal lands
        // off-screen skips the pass, including the depth-transition
        // barriers. Peeking answers that without testing any decal twice.
        let mut visible = self.state.decals.visible(frustum).peekable();
        if visible.peek().is_none() {
            return;
        }

        let extent = self.targets.render_extent;

        // Upload this frame's view UBO.
        let inv_vp = mat4_inverse(vp);
        let viewport_pix = [extent.width as f32, extent.height as f32];
        let view_uni = DecalView {
            vp,
            inv_vp,
            viewport: viewport_pix,
            pre_exposure: self.pre_exposure.value(),
            _pad: 0.0,
        };
        decals.view_ubos[frame_idx].write_val(0, &view_uni);

        // Main depth is already in SHADER_READ_ONLY for the fragment's sample:
        // the graph declares this pass's depth read and the executor emits the
        // transition ahead of this command buffer.
        let area = vk::Rect2D::default().extent(extent);

        // Negative-height viewport matches the main pass so the
        // rasterized pixel grid lines up with the depth attachment we're
        // sampling. Without this, the cube would be drawn Y-flipped
        // relative to the scene depth.
        let vp_state = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        rec.render_over_sampled(&self.targets.hdr_resolve_images[frame_idx], extent, |rec| {
            rec.set_viewport(&vp_state);
            rec.set_scissor(&area);
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &decals.pipeline);
            rec.bind_vertex_buffers(0, &[decals.vertex_buffer.buffer()], &[0]);
            rec.bind_index_buffer(decals.index_buffer.buffer(), 0, vk::IndexType::UINT16);

            for decal in visible {
                // This frame's ring slot keeps what an earlier frame wrote, so a
                // decal whose record has not changed since is already uploaded.
                if decal.take_upload(frame_idx) {
                    decals.params_ubos[frame_idx].write_val(
                        decal.id * usize::saturating_from(PARAMS_STRIDE),
                        decal.params,
                    );
                }
                // Selects this decal's record inside the params UBO written above.
                let dynamic_offset = u32::saturating_from(decal.id as u64 * PARAMS_STRIDE);
                rec.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    &decals.pipeline_layout,
                    0,
                    std::slice::from_ref(&decals.view_sets[frame_idx]),
                    std::slice::from_ref(&dynamic_offset),
                );
                rec.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    &decals.pipeline_layout,
                    1,
                    std::slice::from_ref(&decals.albedo_sets[decal.id]),
                    &[],
                );
                // 36 indices is the cube's full index count.
                rec.draw_indexed(36, 1, 0, 0, 0);
                self.inc_draw_calls(1);
            }
        });
    }
}

// Runtime decals (`RenderBackend::add_decal` / `remove_decal`)

impl VkContext {
    // Whether any live decal samples texture-pool `slot`. Decal sets are
    // single-copy and always bound, so a swap of such a slot must drain first.
    pub(in crate::vulkan) fn decal_samples_slot(&self, slot: usize) -> bool {
        match &self.decal.resources {
            Some(s) => s.decal_texture_slots.get().contains(&slot),
            None => false,
        }
    }

    // Re-point every decal albedo set that samples pool `slot` at its new view.
    pub(in crate::vulkan) fn rewrite_decal_albedo_slot(&self, slot: usize) {
        let Some(decals) = &self.decal.resources else {
            return;
        };
        let slots = decals.decal_texture_slots.get();
        let last_tex = self.scene.textures.len().saturating_sub(1);
        for (id, &tex_slot) in slots.iter().enumerate() {
            if tex_slot == slot {
                let view = self.scene.textures[tex_slot.min(last_tex)].view;
                write_albedo_image(&self.hw.device, decals.albedo_sets[id], view);
            }
        }
    }
}

// Point a decal's albedo set at `view`. Its sampler was written with the set.
fn write_albedo_image(device: &VkDevice, set: vk::DescriptorSet, view: vk::ImageView) {
    SetWrites::new(set).sampled_image(0, view).apply(device);
}

// Wire authored decals into the runtime state at init.

impl VkContext {
    // Push every world-authored `DecalRecord` through `add_decal` so its
    // albedo descriptor lands in the reserved slot. Called once from
    // `VkContext::new` after `decal.resources` is built.
    pub(in crate::vulkan) fn upload_initial_decals(
        &mut self,
        records: Vec<DecalRecord>,
    ) -> RenderResult<()> {
        if records.len() > MAX_DECALS {
            return Err(RenderError::Other(format!(
                "decals: {} authored decals exceed MAX_DECALS ({})",
                records.len(),
                MAX_DECALS
            )));
        }
        for rec in records {
            self.add_decal(rec)?;
        }
        Ok(())
    }
}

// Re-export GpuImage so the textures module compiles cleanly when the
// decal module is the only consumer of a few of its helpers.
type _Marker = GpuImage;

impl DecalPass for VkContext {
    // The decal pipeline is built at init, so a world without one cannot take
    // a runtime decal.
    fn ensure_decal_pass(&mut self) -> RenderResult<()> {
        if self.decal.resources.is_none() {
            return Err(RenderError::Other(
                "add_decal: decal pipeline unavailable".to_string(),
            ));
        }
        Ok(())
    }

    // Write the albedo descriptor for slot `id`; the encoder reads it next
    // frame. The texture pool entry is referenced live; a future eviction
    // routes through `rewrite_texture_slot` to re-point.
    fn bind_decal_texture(&mut self, id: usize, texture_slot: usize) {
        let Some(decals) = self.decal.resources.as_ref() else {
            return;
        };
        let Some(&set) = decals.albedo_sets.get(id) else {
            debug_assert!(
                false,
                "decal id {id} is past the {} reserved albedo sets",
                decals.albedo_sets.len()
            );
            return;
        };
        let tex_idx = texture_slot.min(self.scene.textures.len().saturating_sub(1));
        write_albedo_image(&self.hw.device, set, self.scene.textures[tex_idx].view);
        let mut slots = decals.decal_texture_slots.get();
        slots[id] = tex_idx;
        decals.decal_texture_slots.set(slots);
    }

    fn release_decal_texture(&mut self, id: usize) {
        if let Some(decals) = &self.decal.resources {
            let mut slots = decals.decal_texture_slots.get();
            slots[id] = usize::MAX;
            decals.decal_texture_slots.set(slots);
        }
    }
}
