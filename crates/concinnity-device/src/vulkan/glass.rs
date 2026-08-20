// src/vulkan/glass.rs
//
// GlassPanel: the generic producer for the engine's transparent pass on the
// Vulkan backend. Each panel is a flat world-space quad (built once at init)
// drawn in the `PassId::Transparent` slot after SSR resolve and before TAA. The
// pass snapshots the pre-transparent scene, sorts the panels back-to-front by
// camera distance, and draws each one; the fragment shader refracts the
// snapshot, tints it, and adds a Fresnel rim (see shaders/glass.slang, the
// single source all three backends compile).
//
// Same uniform layouts, back-to-front ordering and manual depth-occlusion test
// as the DirectX and Metal hosts. The pass writes
// into the post-SSR scene image (the same image the post stack samples:
// `SsrResources::output` when SSR is on, else `hdr_resolve_images[frame]`),
// alpha-blending over it; downstream TAA / bloom / composite pick the
// translucent geometry up unchanged. Water is a separate (Metal-only) producer
// and is not ported here; the transparent slot on Vulkan is glass-only.

use ash::{Device, vk};

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::assets::GlassPanel;
use crate::geometry::glass_quad::build_glass_quad;
use crate::gfx::mesh_payload::Vertex;

use crate::gfx::render_types::RtParams;
use crate::gfx::rt_reflections::RtParamsInputs;

use super::context::{HDR_FORMAT, VkContext};
use super::pipeline::spv_module;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::texture::{
    GpuImage, ImageSpec, LayoutTransition, SubresourceRange, create_image, create_image_view,
    one_shot_submit, transition_image_layout_range,
};

// The live acceleration-structure handles wired into the glass RT descriptor ring.
// Passed once at init (`None` when RT is not live at launch) and re-pointed every
// frame thereafter by `VkContext::rt_dynamic_update`, so the ring tracks dynamic
// TLAS / geometry-table / deformed-buffer rebuilds. Mirrors the per-frame inputs
// `post::rt_reflections::wire_dynamic` takes.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassRtInputs {
    pub tlas: vk::AccelerationStructureKHR,
    pub geom_buffer: vk::Buffer,
    pub geom_size: vk::DeviceSize,
    pub deformed_verts: vk::Buffer,
    pub skinned_indices: vk::Buffer,
}

// The live acceleration-structure handles re-pointed into one frame's glass RT
// descriptor set every frame by `wire_dynamic` / `wire_rt_dynamic`. Same handles
// the RT-reflection pass rewires; the deformed buffer is always valid while
// `skinned_indices` is null until the first skinned rebuild.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassRtDynamic {
    pub tlas: vk::AccelerationStructureKHR,
    pub geom_buffer: vk::Buffer,
    pub geom_size: vk::DeviceSize,
    pub deformed: vk::Buffer,
    pub skinned_indices: vk::Buffer,
}

// `TransparentView` (per-frame glass view UBO) and `GlassParams` (per-panel
// glass UBO) are GPU-free layout structs that live in concinnity-render;
// re-export them so `crate::vulkan::glass::{TransparentView,GlassParams}` are
// unchanged for the encode + `glass_params_from` paths.
pub(in crate::vulkan) use concinnity_render::uniforms::GlassParams;
pub(in crate::vulkan) use concinnity_render::uniforms::TransparentView;

// Build the per-panel `GlassParams` from an authored panel. `planar` is 1.0 when
// the pane has a planar reflection slot, else 0.0. Pure; unit tested. Mirrors
// `directx::glass::glass_params_from`.
fn glass_params_from(panel: &GlassPanel, planar: f32) -> GlassParams {
    let n = panel.normal; // already unit-length from GlassPanel::from_args
    GlassParams {
        centre: [panel.centre[0], panel.centre[1], panel.centre[2], 0.0],
        normal: [n[0], n[1], n[2], 0.0],
        tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
        opacity: panel.opacity,
        refraction_strength: panel.refraction_strength,
        fresnel_power: panel.fresnel_power,
        planar,
    }
}

// World-space distance from the camera to a panel centre. Larger = farther =
// drawn first. Pure; unit tested.
fn sort_distance(centre: [f32; 3], cam: [f32; 3]) -> f32 {
    let dx = centre[0] - cam[0];
    let dy = centre[1] - cam[1];
    let dz = centre[2] - cam[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// Indices of the visible panels, ordered farthest-camera-distance first. Pure;
// unit tested. Invisible panels are excluded; the visible set is sorted via the
// shared `gfx::transparent::back_to_front_order`.
fn ordered_visible(centres: &[[f32; 3]], visible: &[bool], cam: [f32; 3]) -> Vec<usize> {
    let live: Vec<usize> = (0..centres.len()).filter(|&i| visible[i]).collect();
    let dists: Vec<f32> = live
        .iter()
        .map(|&i| sort_distance(centres[i], cam))
        .collect();
    crate::gfx::transparent::back_to_front_order(&dists)
        .into_iter()
        .map(|oi| live[oi])
        .collect()
}

// Compile the glass vertex + fragment shaders, injecting the MSAA define so the
// depth sampler type matches the main-depth resource's sample count. The
// fragment's shared reflection-probe sampling ({PROBE_DESC_SET} = 2, the global
// set carrying the probe set/cubes here) is substituted by the builtins
// assembly. Mirrors compile_ssr_shaders.
fn compile_glass_shaders(
    hot_reload: bool,
    msaa: bool,
    probe_cube_count: u32,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = super::builtins::Ctx {
        hot_reload,
        msaa,
        pool_size: 0,
        probe_count: probe_cube_count as usize,
    };
    let vert = super::slang_builtins::GLASS_VERT.compile(&ctx)?;
    let frag = super::slang_builtins::GLASS_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

// SPIR-V blobs for the ray-traced glass pipelines: the shared vertex stage (the
// same one the base pass uses -- the trace is entirely in the fragment), the
// flat fragment, and the textured fragment (`None` when the bindless pool is
// absent). Mirrors `post::rt_reflections::RtShaders`.
struct GlassRtShaders {
    vs: Vec<u8>,
    flat_fs: Vec<u8>,
    textured_fs: Option<Vec<u8>>,
}

// Compile the glass vertex shader + the ray-traced glass fragment (flat, plus
// the textured variant when `pool_size > 0`). slangc emits `SPV_KHR_ray_query`
// for the traversal, which the device already advertises wherever these
// pipelines are built.
fn compile_glass_rt_shaders(
    hot_reload: bool,
    msaa: bool,
    pool_size: usize,
    probe_cube_count: u32,
) -> Result<GlassRtShaders, String> {
    // The pool declaration needs at least one slot even when the bindless pool
    // is absent (the textured variant is then skipped).
    let ctx = super::builtins::Ctx {
        hot_reload,
        msaa,
        pool_size: pool_size.max(1),
        probe_count: probe_cube_count as usize,
    };
    let vs = super::slang_builtins::GLASS_VERT.compile(&ctx)?;
    let flat_fs = super::slang_builtins::GLASS_FRAG_RT.compile(&ctx)?;
    let textured_fs = if pool_size > 0 {
        Some(super::slang_builtins::GLASS_FRAG_RT_TEXTURED.compile(&ctx)?)
    } else {
        None
    };
    Ok(GlassRtShaders {
        vs,
        flat_fs,
        textured_fs,
    })
}

// The glass RT descriptor set (set 3): RtParams UBO (0), scene TLAS (1), the
// per-instance geometry table (2), the shared static verts (3) + u32 indices (4),
// and the deformed skinned verts (5) + u16 skinned indices (6). Mirrors
// `post::rt_reflections`'s set 0, minus the fullscreen pass's screen-space scene /
// gbuffer / roughness inputs (glass traces off the pane surface point).
fn create_rt_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let frag = vk::ShaderStageFlags::FRAGMENT;
    create_descriptor_set_layout(
        device,
        &[
            (0, vk::DescriptorType::UNIFORM_BUFFER, frag),
            (1, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, frag),
            (2, vk::DescriptorType::STORAGE_BUFFER, frag),
            (3, vk::DescriptorType::STORAGE_BUFFER, frag),
            (4, vk::DescriptorType::STORAGE_BUFFER, frag),
            (5, vk::DescriptorType::STORAGE_BUFFER, frag),
            (6, vk::DescriptorType::STORAGE_BUFFER, frag),
        ],
    )
}

impl GlassRt {
    // Write the per-frame static RT bindings: the RtParams UBO (0) + the shared
    // static verts (3) + u32 indices (4). The TLAS / geom table / skinned buffers
    // (1/2/5/6) are filled by `wire_dynamic`. Called once at init.
    fn wire_static(&self, device: &Device, vertex_buffer: vk::Buffer, index_buffer: vk::Buffer) {
        for (i, &set) in self.sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(self.params_buffers[i].buffer())
                .offset(0)
                .range(std::mem::size_of::<RtParams>() as vk::DeviceSize);
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info))];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        self.rewire_geometry(device, vertex_buffer, index_buffer);
    }

    // Re-point every frame's shared static verts (3) + u32 indices (4) at the
    // given buffers. Called by `wire_static`, and again on its own when an asset
    // hot-reload replaces the shared geometry buffers under the pass.
    fn rewire_geometry(
        &self,
        device: &Device,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) {
        let verts_info = vk::DescriptorBufferInfo::default()
            .buffer(vertex_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let indices_info = vk::DescriptorBufferInfo::default()
            .buffer(index_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        for &set in &self.sets {
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&verts_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&indices_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Re-point one frame's TLAS (1), geometry table (2), deformed skinned verts
    // (5), and skinned indices (6) at the live handles. Called every frame from
    // `VkContext::rt_dynamic_update` (the current frame's set is fence-gated). The
    // deformed buffer is always a valid handle (the accel data holds a 1-element
    // dummy when there is no skinned geometry); `skinned_indices` is null until the
    // first skinned rebuild, in which case the 1-element dummy SSBO binds so the
    // descriptor stays valid. Mirrors `post::rt_reflections::wire_dynamic`.
    fn wire_dynamic(&self, device: &Device, frame_idx: usize, dynamic: GlassRtDynamic) {
        let GlassRtDynamic {
            tlas,
            geom_buffer,
            geom_size,
            deformed,
            skinned_indices,
        } = dynamic;
        let set = self.sets[frame_idx];
        let accels = [tlas];
        let mut accel_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accels);
        let mut tlas_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut accel_write);
        // `push_next` does not set the count for an acceleration-structure write.
        tlas_write.descriptor_count = 1;
        let geom_info = vk::DescriptorBufferInfo::default()
            .buffer(geom_buffer)
            .offset(0)
            .range(geom_size);
        let geom_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&geom_info));
        let deformed_info = vk::DescriptorBufferInfo::default()
            .buffer(deformed)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let deformed_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&deformed_info));
        let sidx_buffer = if skinned_indices != vk::Buffer::null() {
            skinned_indices
        } else {
            self.dummy_ssbo.buffer()
        };
        let sidx_info = vk::DescriptorBufferInfo::default()
            .buffer(sidx_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let sidx_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&sidx_info));
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe {
            device
                .update_descriptor_sets(&[tlas_write, geom_write, deformed_write, sidx_write], &[])
        };
    }

    fn destroy(&mut self, device: &Device) {
        // SAFETY: every handle here was created from this device and is destroyed exactly once; the
        // caller has already waited for the device to go idle, so no submission still references
        // them.
        unsafe {
            device.destroy_pipeline(self.flat_pso, None);
            if let Some(p) = self.textured_pso.take() {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout_flat, None);
            if let Some(l) = self.layout_textured.take() {
                device.destroy_pipeline_layout(l, None);
            }
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_descriptor_pool(self.pool, None);
        }
        self.params_buffers.clear();
        self.dummy_ssbo = PooledBuffer::null();
    }
}

// The render-pass + pipeline build config for the glass RT pipelines: the target
// render pass, the per-frame ring depth, the MSAA depth-sampler flavour, and the
// hot-reload shader source toggle.
#[derive(Clone, Copy)]
struct GlassRtPipelineConfig {
    render_pass: vk::RenderPass,
    frames: usize,
    msaa: bool,
    hot_reload: bool,
}

// The descriptor set layouts the glass RT pipeline layouts reference: the shared
// glass view / params / global sets (0/1/2) plus the bindless texture pool set
// (with its pool size) that gates the textured hit-shading variant.
// `probe_cube_count` is the global set layout's binding-8 descriptor count, which
// sizes the fragment's probe cube array.
#[derive(Clone, Copy)]
struct GlassRtSetLayouts {
    view: vk::DescriptorSetLayout,
    params: vk::DescriptorSetLayout,
    global: vk::DescriptorSetLayout,
    probe_cube_count: u32,
    bindless: Option<vk::DescriptorSetLayout>,
    bindless_pool_size: usize,
}

// The shared static geometry the trace reads plus the initial acceleration-
// structure handles. `rt_inputs` wires the initial accel handles when RT is live
// at launch; otherwise the first `rt_dynamic_update` fills them.
#[derive(Clone, Copy)]
struct GlassRtGeometry {
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    rt_inputs: Option<GlassRtInputs>,
}

// Build the glass RT pipelines + descriptor ring. Called from `GlassResources::new`
// when the device is RT-capable. Returns `Err` on a shader-compile failure (the
// caller then leaves `rt` `None` and the probe / planar glass path runs). The two
// pipeline layouts share the glass view / params / global set layouts (sets 0/1/2)
// so the same descriptor sets the base pass binds carry over; the RT geometry rides
// a dedicated set 3 (bindless pool on set 4 for the textured variant). `rt_inputs`
// wires the initial accel handles when RT is live at launch; otherwise the first
// `rt_dynamic_update` fills them before the RT path is taken.
fn build_glass_rt(
    alloc: &DeviceAllocator,
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    config: GlassRtPipelineConfig,
    layouts: GlassRtSetLayouts,
    geometry: GlassRtGeometry,
) -> Result<GlassRt, String> {
    let GlassRtPipelineConfig {
        render_pass,
        frames,
        msaa,
        hot_reload,
    } = config;
    let GlassRtSetLayouts {
        view: view_set_layout,
        params: params_set_layout,
        global: global_set_layout,
        probe_cube_count,
        bindless: bindless_set_layout,
        bindless_pool_size,
    } = layouts;
    let GlassRtGeometry {
        vertex_buffer,
        index_buffer,
        rt_inputs,
    } = geometry;
    let shaders = compile_glass_rt_shaders(hot_reload, msaa, bindless_pool_size, probe_cube_count)?;
    let set_layout = create_rt_set_layout(device)?;

    let flat_layouts = [
        view_set_layout,
        params_set_layout,
        global_set_layout,
        set_layout,
    ];
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let layout_flat = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&flat_layouts),
            None,
        )
    }
    .map_err(|e| format!("glass rt flat pipeline layout: {e}"))?;
    // The textured variant binds 5 sets (view / params / global / rt-geom / bindless
    // pool); the flat variant binds 4. The Vulkan spec only guarantees
    // `maxBoundDescriptorSets >= 4`, so on a device that reports exactly 4 fall back
    // to the flat trace (the bindless pool is unreachable there). Every RT-capable
    // desktop GPU reports >= 8; this mirrors the `rt_capable -> flat -> base`
    // degradation ladder.
    // SAFETY: a property query on a live handle; it only reads.
    let max_bound_sets = unsafe { instance.get_physical_device_properties(physical_device) }
        .limits
        .max_bound_descriptor_sets;
    let layout_textured = match bindless_set_layout {
        Some(bsl) if max_bound_sets >= 5 => {
            let layouts = [
                view_set_layout,
                params_set_layout,
                global_set_layout,
                set_layout,
                bsl,
            ];
            Some(
                // SAFETY: the create-info and every slice it borrows are live for the call, and
                // each handle it names belongs to this device.
                unsafe {
                    device.create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                        None,
                    )
                }
                .map_err(|e| format!("glass rt textured pipeline layout: {e}"))?,
            )
        }
        _ => None,
    };

    let flat_pso = create_pipeline(
        device,
        render_pass,
        layout_flat,
        &shaders.vs,
        &shaders.flat_fs,
    )?;
    let textured_pso = match (layout_textured, &shaders.textured_fs) {
        (Some(layout), Some(fs)) => Some(create_pipeline(
            device,
            render_pass,
            layout,
            &shaders.vs,
            fs,
        )?),
        _ => None,
    };

    // Per-frame RtParams UBO ring (host-mapped).
    let params_size = std::mem::size_of::<RtParams>() as vk::DeviceSize;
    let mut params_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        params_buffers.push(alloc.create_buffer(
            params_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }

    // Pool: per-frame sets, each 1 UBO + 1 TLAS + 5 SSBO (geom, verts, indices,
    // deformed verts, skinned indices).
    let f = frames as u32;
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(f),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(f),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(f * 5),
    ];
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(f),
            None,
        )
    }
    .map_err(|e| format!("glass rt descriptor pool: {e}"))?;
    let layouts: Vec<_> = (0..frames).map(|_| set_layout).collect();
    let sets = alloc_descriptor_sets(device, pool, &layouts)?;

    // 1-element dummy SSBO for the skinned-index binding when there is no skinned
    // geometry.
    let dummy_ssbo = alloc.create_buffer(
        16,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let rt = GlassRt {
        set_layout,
        layout_flat,
        layout_textured,
        flat_pso,
        textured_pso,
        params_buffers,
        sets,
        pool,
        dummy_ssbo,
    };
    rt.wire_static(device, vertex_buffer, index_buffer);
    if let Some(inputs) = rt_inputs {
        for i in 0..frames {
            rt.wire_dynamic(
                device,
                i,
                GlassRtDynamic {
                    tlas: inputs.tlas,
                    geom_buffer: inputs.geom_buffer,
                    geom_size: inputs.geom_size,
                    deformed: inputs.deformed_verts,
                    skinned_indices: inputs.skinned_indices,
                },
            );
        }
    }
    Ok(rt)
}

// Per-panel GPU state: the static world-space quad VB + IB, the per-panel
// `GlassParams` UBO + its descriptor set, and the visibility flag.
struct GlassPanelRecord {
    vertex_buffer: PooledBuffer,
    index_buffer: PooledBuffer,
    index_count: u32,
    params_ubo: PooledBuffer,
    params_set: vk::DescriptorSet,
    visible: bool,
    // World-space centre, used for the back-to-front camera-distance sort.
    centre: [f32; 3],
    // The pane's planar reflection slot (its mirror render's target), or `None`
    // when it falls back to the probe cube. Drives the resize re-point of the
    // planar binding (binding 1 of `params_set`).
    planar_slot: Option<usize>,
}

// Engine-side glass resources. Built only when the world declared at least one
// `GlassPanel`; `VkContext::glass` stays `None` otherwise and the Transparent
// pass is omitted from the frame graph.
pub(in crate::vulkan) struct GlassResources {
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    view_set_layout: vk::DescriptorSetLayout,
    params_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,

    // Per-frame `TransparentView` UBO ring. Persistently mapped; the encoder
    // memcpys this frame's view into `view_ubo_buffers[frame_idx].mapped_ptr()` before binding.
    view_ubos: Vec<PooledBuffer>,
    view_sets: Vec<vk::DescriptorSet>,

    // Per-frame scene target the pass blends into: `SsrResources::output`
    // (repeated for every frame slot) when SSR is on, else this slot's
    // `hdr_resolve_images[i]`. The framebuffer targets the view; the snapshot
    // copy reads the image.
    scene_images: Vec<vk::Image>,
    framebuffers: Vec<vk::Framebuffer>,

    // Pre-transparent HDR scene snapshot for the refraction tap. The encoder
    // copies the scene image into this at the head of the pass; sized to render
    // dims, recreated by `rebuild` on resize. Single image shared across frames
    // (the same single-shared-snapshot pattern as the raymarch pass).
    snapshot: GpuImage,
    // Linear sampler bound alongside the snapshot (binding 1) and the main
    // depth (binding 2). Borrowed from `VkContext`; not owned, never destroyed
    // here.
    sampler: vk::Sampler,

    panels: Vec<GlassPanelRecord>,

    // Per-pixel ray-traced reflection resources. `Some` whenever the device is
    // RT-capable (so a live quality toggle can bring RT up), independent of
    // whether RT is on at launch; the encoder uses them only when
    // `VkContext::rt_glass_active()`. `None` on a non-RT GPU (the probe / planar
    // glass path then always runs). Mirrors the RT half of
    // `directx::glass::GlassResources`.
    rt: Option<GlassRt>,
}

// Per-pixel ray-traced reflection state for the glass pass: the two RT pipelines
// (flat material-tint + textured bindless), their layouts, the per-frame RtParams
// UBO ring, and the per-frame RT descriptor ring (set 3: TLAS + geometry table +
// the static + skinned vertex/index buffers). Built together (`flat_pso` present
// implies the rest), so `GlassResources::rt_pipelines_ready` gates on the outer
// `Option`. Mirrors the RT fields of `directx::glass::GlassResources`.
struct GlassRt {
    set_layout: vk::DescriptorSetLayout,
    layout_flat: vk::PipelineLayout,
    // The textured layout / PSO are `Some` only when the bindless texture pool is
    // live (the same gate the bindless static + RT-reflection passes use).
    layout_textured: Option<vk::PipelineLayout>,
    flat_pso: vk::Pipeline,
    textured_pso: Option<vk::Pipeline>,

    // Per-frame RtParams UBO ring (144 B, host-mapped). The encoder fills this
    // frame's slot (sun + ray tunables) before binding, mirroring
    // `encode_rt_reflections`.
    params_buffers: Vec<PooledBuffer>,

    // Per-frame RT descriptor ring (set 3). Static bindings (RtParams UBO, the
    // shared static verts / indices) are written once; the TLAS / geom table /
    // deformed verts / skinned indices (bindings 1/2/5/6) are re-pointed every
    // frame by `wire_dynamic` because a dynamic rebuild fresh-allocates them.
    sets: Vec<vk::DescriptorSet>,
    pool: vk::DescriptorPool,

    // 1-element dummy SSBO bound to the skinned vertex/index bindings (5/6) when
    // the scene carries no skinned geometry (the accel data's skinned-index handle
    // is then `vk::Buffer::null()`), so the descriptor stays valid. Mirrors the
    // RT-reflection pass's dummy.
    dummy_ssbo: PooledBuffer,
}

// The transparent render pass: load + store the single-sample scene image (the
// post-SSR scene rests in SHADER_READ_ONLY) with no depth attachment (the
// fragment shader does the manual occlusion test). Mirrors the decal render
// pass shape.
fn create_glass_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, String> {
    let color = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    // The encoder's explicit barrier (scene back to SHADER_READ_ONLY after the
    // snapshot copy) makes the load available; this dependency orders the load
    // after it.
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::TRANSFER,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        );
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_render_pass(&info, None) }.map_err(|e| format!("glass render pass: {e}"))
}

fn create_view_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let frag = vk::ShaderStageFlags::FRAGMENT;
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(frag),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(frag),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("glass view set layout: {e}"))
}

fn create_params_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [
        // 0: the per-panel GlassParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        // 1: this pane's planar reflection target (or the snapshot stand-in).
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("glass params set layout: {e}"))
}

fn create_descriptor_pool(
    device: &Device,
    frames: usize,
    panels: usize,
) -> Result<vk::DescriptorPool, String> {
    let f = frames as u32;
    let p = panels as u32;
    let sizes = [
        // view UBO per frame + params UBO per panel.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: f + p,
        },
        // snapshot + depth per per-frame view set, plus one planar target per pane.
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 2 * f + p,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(f + p)
        .pool_sizes(&sizes);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(|e| format!("glass descriptor pool: {e}"))
}

fn alloc_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>, String> {
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.allocate_descriptor_sets(&info) }
        .map_err(|e| format!("glass descriptor sets: {e}"))
}

// Write one per-frame view set: the view UBO (binding 0), the shared scene
// snapshot (binding 1), and this frame's main-depth view (binding 2).
fn write_view_set(
    device: &Device,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    snapshot_view: vk::ImageView,
    depth_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let view_info = vk::DescriptorBufferInfo::default()
        .buffer(view_ubo)
        .offset(0)
        .range(std::mem::size_of::<TransparentView>() as u64);
    let img = |view: vk::ImageView| {
        vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)
    };
    let snapshot_info = img(snapshot_view);
    let depth_info = img(depth_view);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&view_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&snapshot_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Write a pane's params set: the GlassParams UBO (binding 0) and the planar
// reflection target it samples (binding 1) -- its slot's mirror render, or the
// snapshot stand-in for a slotless pane (the shader gates on the `planar` flag).
fn write_params_set(
    device: &Device,
    set: vk::DescriptorSet,
    params_ubo: vk::Buffer,
    planar_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(0)
        .range(std::mem::size_of::<GlassParams>() as u64);
    let planar_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(planar_view)
        .sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&planar_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Build the glass graphics pipeline. No face culling (the shader is two-sided),
// no depth attachment / test (the fragment does the manual occlusion test), and
// SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending into the single-sample scene target.
// The standard engine `Vertex` stride is bound with only the position attribute
// (location 0) fetched. Negative-height viewport applied dynamically at encode.
fn create_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(&entry),
    ];

    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attribute = vk::VertexInputAttributeDescription::default()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&binding))
        .vertex_attribute_descriptions(std::slice::from_ref(&attribute));

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    // The scene target is single-sample regardless of the main pass's MSAA.
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // No depth attachment: the fragment shader does the manual occlusion test.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend_state)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass);
    // SAFETY: the create-infos and every slice they borrow are live for the call, and each handle
    // they name belongs to this device.
    let pipeline = unsafe {
        crate::vulkan::pipeline_cache::create_graphics_pipelines(
            device,
            std::slice::from_ref(&info),
        )
    }
    .map_err(|(_, e)| format!("create glass pipeline: {e}"))?[0];
    // SAFETY: the shader module was created from this device, and a module may be destroyed as soon
    // as the pipelines that consumed it exist.
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(pipeline)
}

// Create the pre-transparent HDR scene snapshot (SAMPLED | TRANSFER_DST,
// GPU-local) and rest it in SHADER_READ_ONLY so the first frame's snapshot
// barrier (SHADER_READ_ONLY -> TRANSFER_DST) matches. Mirrors the raymarch
// scene snapshot.
fn create_snapshot(
    alloc: &DeviceAllocator,
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
) -> Result<GpuImage, String> {
    let pooled = create_image(
        alloc,
        &ImageSpec {
            width: width.max(1),
            height: height.max(1),
            format: HDR_FORMAT,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples: vk::SampleCountFlags::TYPE_1,
        },
    )?;
    let image = pooled.image();
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout_range(
            device,
            cmd,
            image,
            LayoutTransition {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                aspect: vk::ImageAspectFlags::COLOR,
            },
            SubresourceRange {
                base_layer: 0,
                layer_count: 1,
                base_mip: 0,
                mip_count: 1,
            },
        );
    })?;
    let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage::from_pooled(pooled, view))
}

// Upload one panel's static quad VB + IB (host-visible, written once) and its
// per-panel `GlassParams` UBO; allocate + write the panel's descriptor set.
type PanelBuffers = (PooledBuffer, PooledBuffer, u32);
fn build_panel_buffers(
    alloc: &DeviceAllocator,
    panel: &GlassPanel,
) -> Result<PanelBuffers, String> {
    let (verts, idxs) = build_glass_quad(panel.centre, panel.normal, panel.half_size);

    // Flatten into the standard engine `Vertex` layout. Tangent is a
    // placeholder (the glass shader rebuilds its frame from the panel normal)
    // and per-vertex colour is unused.
    let mut packed: Vec<Vertex> = Vec::with_capacity(verts.len());
    for (pos, normal, color, uv) in verts {
        packed.push(Vertex {
            pos,
            normal,
            tangent: [1.0, 0.0, 0.0],
            color,
            uv,
        });
    }

    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let vb_bytes = std::mem::size_of_val(packed.as_slice()) as u64;
    let ib_bytes = std::mem::size_of_val(idxs.as_slice()) as u64;
    let vb = alloc.create_buffer(vb_bytes, vk::BufferUsageFlags::VERTEX_BUFFER, host)?;
    let ib = alloc.create_buffer(ib_bytes, vk::BufferUsageFlags::INDEX_BUFFER, host)?;
    // SAFETY: the staging buffer was created HOST_VISIBLE | HOST_COHERENT and sized to `size`,
    // which is at least the source length, so `mapped_ptr()` is a live mapping of that many bytes;
    // the source is a separate live allocation, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            packed.as_ptr() as *const u8,
            vb.mapped_ptr(),
            vb_bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            idxs.as_ptr() as *const u8,
            ib.mapped_ptr(),
            ib_bytes as usize,
        );
    }
    Ok((vb, ib, idxs.len() as u32))
}

// The Vulkan device handles the glass build + rebuild need: the instance,
// logical + physical device, and the transient command pool + queue used for the
// one-shot snapshot layout transition.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassDeviceCtx<'a> {
    pub alloc: &'a DeviceAllocator,
    pub instance: &'a ash::Instance,
    pub device: &'a Device,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// The non-resource build config for `GlassResources::new`: the render dims + ring
// depth + MSAA sample count, the per-frame global descriptor set layout bound as
// glass set 2, and the hot-reload shader source toggle.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassBuildConfig {
    pub frames: usize,
    pub msaa_samples: vk::SampleCountFlags,
    pub width: u32,
    pub height: u32,
    // The per-frame global descriptor set layout (ViewUniforms, IBL cubes, probe
    // set + cube array). Bound as glass set 2 so the fragment shader reflects the
    // probe set / sky prefilter cube; the pipeline layout must reference it even
    // though glass only samples bindings 5 / 7 / 8. `probe_cube_count` is that
    // layout's binding-8 descriptor count, sizing the fragment's cube array.
    pub global_set_layout: vk::DescriptorSetLayout,
    pub probe_cube_count: u32,
    pub hot_reload: bool,
}

// The post-SSR scene target per frame slot plus the per-frame main-depth views.
// `scene_views` / `scene_images` are the post-SSR scene target per frame slot
// (SSR output repeated, or `hdr_resolve_images[i]`); `depth_views` are the main-
// depth views the manual occlusion test samples. `sampler` is the linear sampler
// bound alongside the snapshot + depth.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassSceneTargets<'a> {
    pub scene_views: &'a [vk::ImageView],
    pub scene_images: &'a [vk::Image],
    pub depth_views: &'a [vk::ImageView],
    pub sampler: vk::Sampler,
}

// Per-pane planar reflection slot (`None` falls back to the probe cube) and the
// per-distinct-plane mirror target views the assigned panes sample. A slotless
// pane (or an empty `target_views`) binds the snapshot as a valid stand-in and
// never samples it (the shader gates on the flag).
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassPlanarTargets<'a> {
    pub slots: &'a [Option<usize>],
    pub target_views: &'a [vk::ImageView],
}

// Per-pixel RT reflection inputs, built whenever the device is RT-capable (so a
// live quality toggle can bring RT up), independent of whether RT is on at launch.
// `vertex_buffer` / `index_buffer` are the shared static geometry the trace reads;
// `rt_inputs` is the initial acceleration-structure handles (`None` when RT is off
// at launch, then filled per frame by `rt_dynamic_update`); `bindless_set_layout` +
// pool size enable the textured hit-shading variant.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassRtSetup {
    pub rt_capable: bool,
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    pub rt_inputs: Option<GlassRtInputs>,
    pub bindless_set_layout: Option<vk::DescriptorSetLayout>,
    pub bindless_pool_size: usize,
}

// The resized post-SSR scene target + per-frame depth views a `rebuild` re-points
// into. `planar_target_views` are the resized per-distinct-plane mirror target
// views (the planar set is rebuilt just before glass), re-pointed into each pane's
// binding 1. The sampler is borrowed from `VkContext` and survives on the resource.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlassRebuildTargets<'a> {
    pub scene_views: &'a [vk::ImageView],
    pub scene_images: &'a [vk::Image],
    pub depth_views: &'a [vk::ImageView],
    pub planar_target_views: &'a [vk::ImageView],
}

impl GlassResources {
    // Build the glass pipeline + per-panel quad buffers + per-panel uniform
    // UBOs + the per-frame view ring + the scene snapshot + the per-frame
    // framebuffers. Called from `VkContext::new` when the world declares any
    // `GlassPanel`. `scene_views` / `scene_images` are the post-SSR scene
    // target per frame slot (SSR output repeated, or `hdr_resolve_images[i]`);
    // `depth_views` are the per-frame main-depth views the manual occlusion
    // test samples.
    pub(in crate::vulkan) fn new(
        ctx: GlassDeviceCtx,
        config: GlassBuildConfig,
        scene: GlassSceneTargets,
        planar: GlassPlanarTargets,
        rt_setup: GlassRtSetup,
        panels: &[GlassPanel],
    ) -> Result<Self, String> {
        let GlassDeviceCtx {
            alloc,
            instance,
            device,
            physical_device,
            command_pool,
            queue,
        } = ctx;
        let GlassBuildConfig {
            frames,
            msaa_samples,
            width,
            height,
            global_set_layout,
            probe_cube_count,
            hot_reload,
        } = config;
        let GlassSceneTargets {
            scene_views,
            scene_images,
            depth_views,
            sampler,
        } = scene;
        let GlassPlanarTargets {
            slots: planar_slots,
            target_views: planar_target_views,
        } = planar;
        let GlassRtSetup {
            rt_capable,
            vertex_buffer,
            index_buffer,
            rt_inputs,
            bindless_set_layout,
            bindless_pool_size,
        } = rt_setup;
        let msaa = msaa_samples != vk::SampleCountFlags::TYPE_1;
        let render_pass = create_glass_render_pass(device, HDR_FORMAT)?;
        let view_set_layout = create_view_set_layout(device)?;
        let params_set_layout = create_params_set_layout(device)?;
        let set_layouts = [view_set_layout, params_set_layout, global_set_layout];
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            // SAFETY: the create-info and every slice it borrows are live for the call, and each
            // handle it names belongs to this device.
            unsafe { device.create_pipeline_layout(&info, None) }
                .map_err(|e| format!("glass pipeline layout: {e}"))?
        };

        let (vert_spv, frag_spv) = compile_glass_shaders(hot_reload, msaa, probe_cube_count)?;
        let pipeline = create_pipeline(device, render_pass, pipeline_layout, &vert_spv, &frag_spv)?;

        // Per-pixel RT glass pipelines, when the device is RT-capable. A compile /
        // build failure leaves `rt` `None` and the probe / planar glass path runs
        // (mirrors DirectX's `build_glass_rt` graceful fallback).
        let rt = if rt_capable {
            match build_glass_rt(
                alloc,
                instance,
                device,
                physical_device,
                GlassRtPipelineConfig {
                    render_pass,
                    frames,
                    msaa,
                    hot_reload,
                },
                GlassRtSetLayouts {
                    view: view_set_layout,
                    params: params_set_layout,
                    global: global_set_layout,
                    probe_cube_count,
                    bindless: bindless_set_layout,
                    bindless_pool_size,
                },
                GlassRtGeometry {
                    vertex_buffer,
                    index_buffer,
                    rt_inputs,
                },
            ) {
                Ok(rt) => Some(rt),
                Err(e) => {
                    tracing::warn!(
                        "glass RT pipelines failed to build ({e}); using the probe / planar glass path"
                    );
                    None
                }
            }
        } else {
            None
        };

        let snapshot = create_snapshot(alloc, device, command_pool, queue, width, height)?;

        // Per-frame view UBO ring (HOST_VISIBLE | HOST_COHERENT, mapped).
        let view_size = std::mem::size_of::<TransparentView>() as u64;
        let mut view_ubos = Vec::with_capacity(frames);
        for _ in 0..frames {
            view_ubos.push(alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        let descriptor_pool = create_descriptor_pool(device, frames, panels.len())?;
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout).collect();
        let view_sets = alloc_sets(device, descriptor_pool, &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                view_ubos[i].buffer(),
                snapshot.view,
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                sampler,
            );
        }

        // Per-frame framebuffers targeting the scene image for that slot.
        let framebuffers = create_framebuffers(device, render_pass, scene_views, width, height)?;

        // Per-panel records: quad buffers + static params UBO + descriptor set.
        let mut records: Vec<GlassPanelRecord> = Vec::with_capacity(panels.len());
        for (i, panel) in panels.iter().enumerate() {
            let (vertex_buffer, index_buffer, index_count) = build_panel_buffers(alloc, panel)?;

            let planar_slot = planar_slots.get(i).copied().flatten();
            let planar = if planar_slot.is_some() { 1.0 } else { 0.0 };
            let params = glass_params_from(panel, planar);
            let params_ubo = alloc.create_buffer(
                std::mem::size_of::<GlassParams>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            // SAFETY: the destination buffer was created HOST_VISIBLE | HOST_COHERENT and sized to
            // hold a `GlassParams`, so `mapped_ptr()` is a live mapping of at least
            // `size_of::<GlassParams>()` bytes; the source is a separate live borrow, so the ranges
            // cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &params as *const GlassParams as *const u8,
                    params_ubo.mapped_ptr(),
                    std::mem::size_of::<GlassParams>(),
                );
            }
            let planar_view = planar_slot
                .and_then(|s| planar_target_views.get(s).copied())
                .unwrap_or(snapshot.view);
            let params_set = alloc_sets(device, descriptor_pool, &[params_set_layout])?[0];
            write_params_set(
                device,
                params_set,
                params_ubo.buffer(),
                planar_view,
                sampler,
            );

            records.push(GlassPanelRecord {
                vertex_buffer,
                index_buffer,
                index_count,
                params_ubo,
                params_set,
                visible: panel.visible,
                centre: panel.centre,
                planar_slot,
            });
        }

        Ok(Self {
            render_pass,
            pipeline,
            pipeline_layout,
            view_set_layout,
            params_set_layout,
            descriptor_pool,
            view_ubos,
            view_sets,
            scene_images: scene_images.to_vec(),
            framebuffers,
            snapshot,
            sampler,
            panels: records,
            rt,
        })
    }

    // True when the per-pixel RT glass pipelines are built (RT-capable device + the
    // shader compile + descriptor setup succeeded). Single-sources the "glass can
    // trace" half of `VkContext::rt_glass_active`. Mirrors DirectX's
    // `rt_pipelines_ready`.
    pub(in crate::vulkan) fn rt_pipelines_ready(&self) -> bool {
        self.rt.is_some()
    }

    // Re-point this frame's glass RT descriptor set at the live TLAS + geometry
    // handles. A no-op when the RT pipelines are absent. Called from
    // `VkContext::rt_dynamic_update` alongside the RT-reflection pass's re-point, so
    // the glass trace samples the same per-frame acceleration structure.
    pub(in crate::vulkan) fn wire_rt_dynamic(
        &self,
        device: &Device,
        frame_idx: usize,
        dynamic: GlassRtDynamic,
    ) {
        if let Some(rt) = self.rt.as_ref() {
            rt.wire_dynamic(device, frame_idx, dynamic);
        }
    }

    // Re-point the glass RT set's shared static verts + indices at new buffers,
    // after an asset hot-reload replaced the shared geometry buffers. A no-op
    // when the RT pipelines are absent.
    pub(in crate::vulkan) fn wire_rt_geometry(
        &self,
        device: &Device,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) {
        if let Some(rt) = self.rt.as_ref() {
            rt.rewire_geometry(device, vertex_buffer, index_buffer);
        }
    }

    // True when any panel is currently visible. Drives
    // `FrameGraphInputs::transparent_enabled` and the encoder early-out.
    pub(in crate::vulkan) fn any_visible(&self) -> bool {
        self.panels.iter().any(|p| p.visible)
    }

    // Recreate the scene snapshot + per-frame framebuffers at new render dims +
    // re-point the snapshot (binding 1) and per-frame depth (binding 2) of every
    // view set. The pipeline, layouts, UBOs, panel buffers, and render pass all
    // survive. Called from the swapchain-resize handler after the SSR / HDR
    // resolve targets have been rebuilt (so `scene_views` / `scene_images` carry
    // the new handles).
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        ctx: GlassDeviceCtx,
        width: u32,
        height: u32,
        targets: GlassRebuildTargets,
    ) -> Result<(), String> {
        let GlassDeviceCtx {
            alloc,
            device,
            command_pool,
            queue,
            ..
        } = ctx;
        let GlassRebuildTargets {
            scene_views,
            scene_images,
            depth_views,
            planar_target_views,
        } = targets;
        let old = std::mem::replace(
            &mut self.snapshot,
            create_snapshot(alloc, device, command_pool, queue, width, height)?,
        );
        drop(old);

        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
        }
        self.framebuffers =
            create_framebuffers(device, self.render_pass, scene_views, width, height)?;
        self.scene_images = scene_images.to_vec();

        for (i, &set) in self.view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                self.view_ubos[i].buffer(),
                self.snapshot.view,
                depth_views[i.min(depth_views.len().saturating_sub(1))],
                self.sampler,
            );
        }

        // Re-point each pane's planar binding (binding 1) at its slot's resized
        // target, or the new snapshot for a slotless pane (the moved snapshot view
        // must be refreshed there too, even though the shader never samples it).
        for p in &self.panels {
            let planar_view = p
                .planar_slot
                .and_then(|s| planar_target_views.get(s).copied())
                .unwrap_or(self.snapshot.view);
            write_params_set(
                device,
                p.params_set,
                p.params_ubo.buffer(),
                planar_view,
                self.sampler,
            );
        }
        Ok(())
    }

    // Destroy every owned GPU resource. The `sampler` is borrowed from
    // `VkContext` and is not destroyed here.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        if let Some(mut rt) = self.rt.take() {
            rt.destroy(device);
        }
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.view_set_layout, None);
            device.destroy_descriptor_set_layout(self.params_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
        }
        self.panels.clear();
        self.view_ubos.clear();
        self.snapshot = GpuImage::null();
        self.framebuffers.clear();
        self.scene_images.clear();
    }
}

// One framebuffer per frame slot, each binding that slot's scene image view as
// the sole colour attachment.
fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    scene_views: &[vk::ImageView],
    width: u32,
    height: u32,
) -> Result<Vec<vk::Framebuffer>, String> {
    let mut out = Vec::with_capacity(scene_views.len());
    for &view in scene_views {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&view))
            .width(width.max(1))
            .height(height.max(1))
            .layers(1);
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let fb = unsafe { device.create_framebuffer(&info, None) }
            .map_err(|e| format!("glass framebuffer: {e}"))?;
        out.push(fb);
    }
    Ok(out)
}

impl VkContext {
    // Assemble the per-frame transparent view from the frame's jittered VP (the
    // matrix the main pass rasterised the depth buffer with, so the glass quad's
    // clip-space depth matches the stored main-depth) + camera position. Mirrors
    // `directx::graph_exec::build_transparent_view`.
    pub(in crate::vulkan) fn build_transparent_view(
        &self,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
    ) -> TransparentView {
        TransparentView {
            vp,
            inv_vp: super::math::mat4_inverse(vp),
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            viewport: [
                self.render_extent.width as f32,
                self.render_extent.height as f32,
            ],
            time,
            prefilter_mip_count: self.prefilter_mip_count as f32,
        }
    }

    // Encode the transparent (glass) pass. Runs after `SsrResolve` and before
    // `TaaResolve` / `Upscale`. Snapshots the post-SSR scene into `snapshot`
    // for refractive taps, then draws every visible panel back-to-front into the
    // scene image with SRC_ALPHA blending; the manual occlusion test samples the
    // main depth. No-op when no glass / no visible panels. Leaves the scene image
    // SHADER_READ_ONLY and the main depth DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the
    // downstream stack.
    pub(in crate::vulkan) fn encode_transparent(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        view: &TransparentView,
        // Projection inputs for the per-pixel RT reflection trace's RtParams (the
        // same values the RT-reflection resolve uses); only consumed on the RT path.
        fov_y_radians: f32,
        aspect: f32,
    ) -> Result<(), String> {
        let Some(glass) = self.glass.as_ref() else {
            return Ok(());
        };
        let cam = [view.camera_pos[0], view.camera_pos[1], view.camera_pos[2]];
        let centres: Vec<[f32; 3]> = glass.panels.iter().map(|p| p.centre).collect();
        let visible: Vec<bool> = glass.panels.iter().map(|p| p.visible).collect();
        let order = ordered_visible(&centres, &visible, cam);
        if order.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let extent = self.render_extent;
        let scene_image = *glass
            .scene_images
            .get(frame_idx)
            .ok_or("glass: scene image index OOB")?;
        let snapshot = glass.snapshot.image;

        // Upload this frame's view UBO.
        let view_ptr = glass
            .view_ubos
            .get(frame_idx)
            .map(|b| b.mapped_ptr())
            .ok_or("glass: view_ubos index OOB")?;
        // SAFETY: the destination UBO was created HOST_VISIBLE | HOST_COHERENT and sized to hold a
        // `TransparentView`, so the mapped pointer is a live mapping of at least that many bytes;
        // the source is a separate live borrow, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                view as *const TransparentView as *const u8,
                view_ptr,
                std::mem::size_of::<TransparentView>(),
            );
        }

        // Per-pixel RT reflection is selected over the probe / planar path when RT
        // is live (the scene TLAS is built) AND the glass RT pipelines compiled --
        // single-sourced via `rt_glass_active`, the same predicate `graph_exec`
        // uses to skip the planar mirror re-render, so the two always agree. The
        // textured variant additionally needs the bindless albedo/normal pool the
        // GPU-cull path populates; without it the flat-tint trace runs. Mirrors
        // DirectX's selection.
        let rt_live = self.rt_glass_active();
        let textured = rt_live
            && self.cull.bindless_pipeline.is_some()
            && glass.rt.as_ref().is_some_and(|r| r.textured_pso.is_some());

        // On the RT path, upload this frame's RtParams (sun + ray tunables) into the
        // glass RtParams ring, mirroring `encode_rt_reflections`'s build. The
        // settings come from the RT-reflection pass (always present when `rt_live`).
        if rt_live {
            let rtres = self
                .rt_reflections
                .as_ref()
                .ok_or("glass rt_live but rt_reflections missing")?;
            let glass_rt = glass
                .rt
                .as_ref()
                .ok_or("glass rt_live but rt pipelines missing")?;
            let v = self.view_matrix;
            let inv_view_rot = [
                [v[0][0], v[1][0], v[2][0], 0.0],
                [v[0][1], v[1][1], v[2][1], 0.0],
                [v[0][2], v[1][2], v[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let params = rtres.settings.params(RtParamsInputs {
                fov_y_radians,
                aspect,
                inv_view_rot,
                cam_pos: cam,
                sun_dir: self.fog_sun_dir,
                sun_color: self.fog_sun_color,
                prefilter_mip_count: self.prefilter_mip_count as f32,
            });
            // SAFETY: the destination buffer was created HOST_VISIBLE | HOST_COHERENT and sized to
            // hold a `RtParams`, so `mapped_ptr()` is a live mapping of at least
            // `size_of::<RtParams>()` bytes; the source is a separate live borrow, so the ranges
            // cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &params as *const RtParams as *const u8,
                    glass_rt.params_buffers[frame_idx].mapped_ptr(),
                    std::mem::size_of::<RtParams>(),
                );
            }
        }

        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let color_barrier = |image: vk::Image,
                             old: vk::ImageLayout,
                             new: vk::ImageLayout,
                             src: vk::AccessFlags,
                             dst: vk::AccessFlags| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range)
        };

        // 1) Open the scene image + snapshot for the refraction snapshot copy.
        // The src scopes order the scene's last writer (SSR resolve / particles
        // colour write) and the prior frame's snapshot read ahead of the
        // transfer.
        let scene_to_src = color_barrier(
            scene_image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_READ,
        );
        let snapshot_to_dst = color_barrier(
            snapshot,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[scene_to_src, snapshot_to_dst],
            );
            let region = vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            device.cmd_copy_image(
                cmd,
                scene_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                snapshot,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }

        // 2) Close the snapshot for the fragment read and restore the scene
        // image to SHADER_READ_ONLY, so the render pass's colour LOAD matches
        // its declared initial layout. Main depth is already sampled here: the
        // graph transitions it once for the whole decoration run.
        let snapshot_to_read = color_barrier(
            snapshot,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        let scene_to_read = color_barrier(
            scene_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[snapshot_to_read, scene_to_read],
            );
        }

        // 3) The render pass: LOAD the scene colour, draw each visible panel
        // back-to-front, STORE. The negative-height viewport matches the main
        // pass so the manual depth test + refraction taps line up at pixel
        // coordinates.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(glass.render_pass)
            .framebuffer(glass.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);

        // Select the RT pipeline (sharp per-pixel trace) when live, else the base
        // probe / planar pipeline. The first three set layouts (view / params /
        // global) are shared between the two pipeline layouts, so the same view /
        // global / per-pane params sets bind unchanged; the RT layout adds the RT
        // geometry (set 3) and, for the textured variant, the bindless pool (set 4).
        let (pipeline, layout) = match (rt_live, glass.rt.as_ref()) {
            (true, Some(r)) if textured => (
                r.textured_pso.expect("textured implies a textured pso"),
                r.layout_textured
                    .expect("textured implies a textured layout"),
            ),
            (true, Some(r)) => (r.flat_pso, r.layout_flat),
            _ => (glass.pipeline, glass.pipeline_layout),
        };
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                std::slice::from_ref(&glass.view_sets[frame_idx]),
                &[],
            );
            // The per-frame global set (set 2): the fragment shader reflects its
            // probe set / cube array (bindings 7 / 8) + sky prefilter cube
            // (binding 5). Bound once per frame; stable across the panel loop.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                2,
                std::slice::from_ref(&self.descriptors.global_sets[frame_idx]),
                &[],
            );
            if rt_live {
                let r = glass.rt.as_ref().expect("rt_live implies the RT pipelines");
                // set 3: this frame's RT geometry (TLAS + geom table + the static +
                // skinned vertex/index buffers). Bound once; stable across the loop.
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    3,
                    std::slice::from_ref(&r.sets[frame_idx]),
                    &[],
                );
                if textured {
                    // set 4: the bindless albedo/normal pool for textured hit shading
                    // (the same set the main bindless pass binds).
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
                        4,
                        std::slice::from_ref(&self.cull.bindless_sets[frame_idx]),
                        &[],
                    );
                }
            }
            for &i in &order {
                let p = &glass.panels[i];
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    1,
                    std::slice::from_ref(&p.params_set),
                    &[],
                );
                device.cmd_bind_vertex_buffers(cmd, 0, &[p.vertex_buffer.buffer()], &[0]);
                device.cmd_bind_index_buffer(
                    cmd,
                    p.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT16,
                );
                device.cmd_draw_indexed(cmd, p.index_count, 1, 0, 0, 0);
                self.inc_draw_calls(1);
            }
            device.cmd_end_render_pass(cmd);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `TransparentView` / `GlassParams` layout tests live with the structs
    // in `concinnity_render::vulkan::uniforms`.

    #[test]
    fn glass_params_from_maps_fields() {
        let panel = GlassPanel {
            centre: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            tint: [0.6, 0.85, 0.9],
            opacity: 0.45,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            ..Default::default()
        };
        let p = glass_params_from(&panel, 1.0);
        assert_eq!(p.centre, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.normal, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.tint, [0.6, 0.85, 0.9, 0.0]);
        assert_eq!(p.opacity, 0.45);
        assert_eq!(p.refraction_strength, 0.04);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.planar, 1.0);
        // A slotless pane gets planar = 0.0 (probe/sky fallback path).
        assert_eq!(glass_params_from(&panel, 0.0).planar, 0.0);
    }

    #[test]
    fn sort_distance_is_euclidean_and_monotone() {
        let cam = [0.0, 0.0, 0.0];
        let near = sort_distance([0.0, 0.0, 1.0], cam);
        let far = sort_distance([0.0, 0.0, 5.0], cam);
        assert!((near - 1.0).abs() < 1e-5);
        assert!((far - 5.0).abs() < 1e-5);
        assert!(far > near);
    }

    #[test]
    fn ordered_visible_excludes_hidden_and_sorts_back_to_front() {
        // Panel 1 is hidden; 0 (dist 5) and 2 (dist 3) are visible. Farthest
        // first => [0, 2]; the hidden panel never appears.
        let centres = [[0.0, 0.0, 5.0], [0.0, 0.0, 9.0], [0.0, 0.0, 3.0]];
        let visible = [true, false, true];
        let order = ordered_visible(&centres, &visible, [0.0, 0.0, 0.0]);
        assert_eq!(order, vec![0, 2]);
    }

    // Compile the glass vertex + fragment shaders (both MSAA variants) so a GLSL
    // regression fails the suite without a GPU. Mirrors the decal / fog compile
    // guards.
    #[test]
    fn glass_shaders_compile() {
        // Both the ceiling and a device-shortened probe cube array must compile.
        for probes in [1, concinnity_render::uniforms::MAX_PROBES as u32] {
            super::compile_glass_shaders(false, true, probes).expect("glass compiles (msaa)");
            super::compile_glass_shaders(false, false, probes).expect("glass compiles (no msaa)");
        }
    }

    // Compile the ray-traced glass shaders (both MSAA variants, both flat +
    // textured) so a regression in glass.slang's `GLASS_RT` arm (the shared
    // `{RT_TRACE}` traversal + the probe `{PROBE_COMMON}` injection + the
    // `RT_TEXTURED` split) fails the suite without a GPU. Mirrors `rt_reflections_shaders_compile`. The
    // CPU<->GPU `RtParams` / `RtGeomEntry` layouts are guarded by the
    // `rt_params_layout_*` / `rt_geom_entry_*` tests in gfx::render_types.
    #[test]
    fn glass_rt_shaders_compile() {
        for &msaa in &[true, false] {
            let shaders = super::compile_glass_rt_shaders(false, msaa, 4, 4)
                .expect("glass rt shaders compile");
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.vs));
            assert!(crate::vulkan::pipeline::is_spirv(&shaders.flat_fs));
            assert!(
                shaders.textured_fs.is_some(),
                "pool_size>0 builds the textured variant"
            );
        }
        // pool_size 0 builds only the flat variant.
        let flat_only =
            super::compile_glass_rt_shaders(false, false, 0, 4).expect("glass rt flat compiles");
        assert!(flat_only.textured_fs.is_none());
    }
}
