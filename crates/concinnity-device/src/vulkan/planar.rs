// src/vulkan/planar.rs
//
// Planar reflection for flat glass panes on the Vulkan backend. Each frame the
// scene is rendered a second time from the camera reflected across each pane's
// plane (mirror view + oblique near-plane clip so geometry behind the plane
// never leaks in) into a render-resolution target; the pane's fragment shader
// then samples that target projectively for a sharp, scene-correct reflection
// instead of the box-projected probe cube.
//
// GLSL/Vulkan port of src/directx/planar.rs (itself a port of src/metal/planar.rs),
// One mirror render per DISTINCT
// plane: near-coplanar panes (one wall of windows) share a render, and panes past
// the budget (MAX_PLANAR_PLANES) fall back to the probe cube. The plane -> slot
// grouping + the mirror matrices come from the pure, unit-tested
// gfx::planar_reflection.
//
// Each plane gets a DEDICATED reflected-frustum cull (the shared probe-bake
// encode_probe_cull): the GPU cull re-runs against the reflected-camera frustum
// into that plane's own indirect buffer, reading the FRAME's camera-independent
// object + draw-args SSBOs. So geometry visible only in the reflection (behind /
// beside the main camera) is captured, not just the main camera's visible set; the
// reflected view-proj's oblique near-plane clip also rejects geometry behind the
// reflector. The face render then draws that indirect. Like the probe capture, the
// skinned tail is not drawn into a mirror (static + instance + chunk only).

use ash::vk;
use concinnity_core::gfx::transform::mat4_inverse;

use crate::vulkan::owned::{OwnedDescriptorPool, OwnedFramebuffer, OwnedRenderPass, VkDevice};

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::{HDR_FORMAT, VkContext};
use super::draw::ViewUniforms;
use super::resources::alloc_descriptor_sets;
use super::texture::{GpuImage, ImageSpec, create_image, create_image_view};
use concinnity_core::gfx::transform::mat4_mul;

// The engine capacity ceiling for distinct reflection planes: the count the
// reserved planar targets are sized to. Single-sourced from `gfx::planar_reflection`
// so the three backends stay in lockstep by construction. The per-frame budget
// passed to `assign_planar_slots` at init can be lower under a quality preset / GPU
// tier, never higher; panes past it fall back to the box-projected probe cube.
pub(in crate::vulkan) const MAX_PLANAR_PLANES: usize =
    crate::gfx::planar_reflection::MAX_PLANAR_PLANES;

// Clip the reflection a hair toward the kept (camera) side of the plane so
// geometry exactly on the surface is not lost to near-plane precision. Matches
// the other backends' PLANAR_CLIP_BIAS.
const PLANAR_CLIP_BIAS: f32 = 0.02;
const PLANAR_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// World-space plane [nx, ny, nz, d] (unit normal, n . p + d = 0 on the surface)
// for a glass pane with unit normal through centre. Pure; unit tested. The init
// path feeds these to assign_planar_slots.
pub(in crate::vulkan) fn pane_plane(normal: [f32; 3], centre: [f32; 3]) -> [f32; 4] {
    [
        normal[0],
        normal[1],
        normal[2],
        -(normal[0] * centre[0] + normal[1] * centre[1] + normal[2] * centre[2]),
    ]
}

// The set of distinct reflection planes for the world, each rendering its mirror
// into the shared colour + depth then resolving into its own shader-readable
// target. A pane samples the target of the slot it was assigned at init (see
// gfx::planar_reflection::assign_planar_slots). Recreated on resize alongside the
// HDR targets; the planes + slot assignment are fixed at init.
pub(in crate::vulkan) struct PlanarReflectionSet {
    planes: Vec<[f32; 4]>,
    frames: usize,
    sample_count: vk::SampleCountFlags,
    width: u32,
    height: u32,
    // Borrowed from VkContext (render-pass-compatible with the bindless main
    // pipeline). Not owned, never destroyed here.
    main_render_pass: vk::RenderPass,

    // Shared MSAA colour (Some only when MSAA) + shared depth, reused across
    // planes (rendered one plane at a time on the frame's cmd buffer) and across
    // frames (the single graphics queue executes submissions in order). Recreated
    // on resize.
    color: Option<GpuImage>,
    depth: GpuImage,
    // Per-plane shader-readable target: the MSAA resolve when MSAA, else the
    // single-sample colour attachment itself. The glass pass samples it. Recreated
    // on resize.
    targets: Vec<GpuImage>,
    framebuffers: Vec<OwnedFramebuffer>,

    // Per-(plane, frame) reflected ViewUniforms UBO ring (HOST_VISIBLE, mapped),
    // indexed plane * frames + frame, so the CPU writes this frame's slot without
    // racing the GPU reading a prior frame's. Bound at binding 0 of the matching
    // planar global set.
    view_bufs: Vec<PooledBuffer>,
    // Per-(plane, frame) global set (the bindless main set): binding 0 = that
    // (plane, frame) reflected view, 1/2 = the shared light/shadow UBOs, 3..6 =
    // the static shadow/env/ssao images, 7 = an EMPTY ProbeSet (the mirror render
    // reflects only sky -- no recursion), 8 = the sky-filled probe cube array.
    global_sets: Vec<vk::DescriptorSet>,
    // EMPTY ProbeSet UBO (count 0), shared by every planar global set.
    probeset_buf: PooledBuffer,

    // Per-(plane, frame) reflected-frustum mirror cull: a DEVICE_LOCAL indirect +
    // status SSBO each (indexed plane * frames + frame), and a cull set that reads
    // the FRAME's object + draw-args SSBOs (camera-independent, so the reflected
    // cull sees every object) and writes this plane's indirect + status. Sized by
    // the build-time object count, so resize never touches them.
    cull_indirect_bufs: Vec<PooledBuffer>,
    cull_status_bufs: Vec<PooledBuffer>,
    cull_sets: Vec<vk::DescriptorSet>,
    // A bake-style Hi-Z read set (cull set 1) with hiz_enabled = 0 so the mirror
    // cull is frustum-only -- the main camera's pyramid is meaningless for a
    // reflected frustum. `Some` only when the world runs Hi-Z. Shared across planes.
    hiz_set: Option<vk::DescriptorSet>,
    hiz_ubo: Option<PooledBuffer>,
    _pool: OwnedDescriptorPool,
}

// The frame-side handles the planar reflected-frustum cull needs: the per-frame
// object + draw-args SSBOs it reads (camera-independent, so the reflected cull sees
// every object, not just the main camera's visible set), the cull descriptor-set
// layout, the build-time object count, and -- when the world runs Hi-Z -- the Hi-Z
// read-set layout + pyramid (view, sampler) so a hiz_enabled = 0 set can be bound
// (the cull pipeline layout statically references set 1).
pub(in crate::vulkan) struct PlanarCullSources<'a> {
    pub(in crate::vulkan) frame_object_buffers: &'a [PooledBuffer],
    pub(in crate::vulkan) frame_draw_args_buffers: &'a [PooledBuffer],
    pub(in crate::vulkan) cull_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) cull_count: usize,
    pub(in crate::vulkan) hiz: Option<(vk::DescriptorSetLayout, vk::ImageView, vk::Sampler)>,
}

// SAFETY: The mapped view-ring pointers are POD raw pointers; the upload buffers stay
// alive through the struct fields and the pointers are written on the render
// thread only. Mirrors GlassResources.
unsafe impl Send for PlanarReflectionSet {}
// SAFETY: as for `Send` above.
unsafe impl Sync for PlanarReflectionSet {}

// The GPU allocation context threaded through every planar create call: the
// instance + logical device + physical device create_image / create_buffer need.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PlanarDevice<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) device: &'a VkDevice,
}

// Render dimensions for the shared colour + depth + per-plane targets: the MSAA
// sample count, pixel dimensions, and how many per-plane targets to create.
#[derive(Clone, Copy)]
struct PlanarTargetDims {
    sample_count: vk::SampleCountFlags,
    width: u32,
    height: u32,
    plane_count: usize,
}

// Create the shared colour (MSAA only) + shared depth + per-plane targets at the
// given render dimensions.
fn create_targets(
    gpu: PlanarDevice<'_>,
    dims: PlanarTargetDims,
) -> Result<(Option<GpuImage>, GpuImage, Vec<GpuImage>), String> {
    let PlanarDevice { alloc, device } = gpu;
    let PlanarTargetDims {
        sample_count,
        width,
        height,
        plane_count,
    } = dims;
    let msaa = sample_count != vk::SampleCountFlags::TYPE_1;
    let w = width.max(1);
    let h = height.max(1);

    let color = if msaa {
        let pooled = create_image(
            alloc,
            &ImageSpec {
                width: w,
                height: h,
                format: HDR_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: sample_count,
            },
        )?;
        let img = pooled.image();
        let view = create_image_view(device, img, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        Some(GpuImage::from_pooled(pooled, view))
    } else {
        None
    };

    let pooled = create_image(
        alloc,
        &ImageSpec {
            width: w,
            height: h,
            format: PLANAR_DEPTH_FORMAT,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples: sample_count,
        },
    )?;
    let depth_img = pooled.image();
    let depth_view = create_image_view(
        device,
        depth_img,
        PLANAR_DEPTH_FORMAT,
        vk::ImageAspectFlags::DEPTH,
    )?;
    let depth = GpuImage::from_pooled(pooled, depth_view);

    let mut targets = Vec::with_capacity(plane_count);
    for _ in 0..plane_count {
        let pooled = create_image(
            alloc,
            &ImageSpec {
                width: w,
                height: h,
                format: HDR_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            },
        )?;
        let img = pooled.image();
        let view = create_image_view(device, img, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        targets.push(GpuImage::from_pooled(pooled, view));
    }
    Ok((color, depth, targets))
}

// The attachments + geometry for the per-plane framebuffers: the compatible main
// pass, the MSAA sample count, the shared colour (MSAA only) + shared depth reused
// across planes, the per-plane targets (one framebuffer each), and the pixel
// dimensions.
struct PlanarFramebufferInputs<'a> {
    main_render_pass: vk::RenderPass,
    sample_count: vk::SampleCountFlags,
    color: Option<&'a GpuImage>,
    depth: &'a GpuImage,
    targets: &'a [GpuImage],
    width: u32,
    height: u32,
}

// One framebuffer per plane, render-pass-compatible with the bindless main pass:
// MSAA -> [shared colour, shared depth, plane target (resolve)], single-sample ->
// [plane target (colour), shared depth].
fn create_framebuffers(
    device: &VkDevice,
    inputs: PlanarFramebufferInputs<'_>,
) -> Result<Vec<OwnedFramebuffer>, String> {
    let PlanarFramebufferInputs {
        main_render_pass,
        sample_count,
        color,
        depth,
        targets,
        width,
        height,
    } = inputs;
    let msaa = sample_count != vk::SampleCountFlags::TYPE_1;
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let attachments: Vec<vk::ImageView> = if msaa {
            vec![
                color
                    .expect("a multisampled planar target has a colour image")
                    .view,
                depth.view,
                target.view,
            ]
        } else {
            vec![target.view, depth.view]
        };
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(main_render_pass)
            .attachments(&attachments)
            .width(width.max(1))
            .height(height.max(1))
            .layers(1);
        let fb = device
            .create_framebuffer(&info)
            .map_err(|e| format!("planar framebuffer: {e}"))?;
        out.push(fb);
    }
    Ok(out)
}

// The frame-independent render config for a planar set: how many frames the ring
// buffers double-buffer over, the MSAA sample count, and the render dimensions.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PlanarConfig {
    pub(in crate::vulkan) frames: usize,
    pub(in crate::vulkan) sample_count: vk::SampleCountFlags,
    pub(in crate::vulkan) width: u32,
    pub(in crate::vulkan) height: u32,
}

// The forward global set every planar re-render binds: its layout and the
// binding-8 reflection-probe cube array's descriptor count, which sizes the
// sky-filled writes each per-(plane, frame) set makes.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PlanarGlobalSet {
    pub(in crate::vulkan) layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) probe_cube_count: u32,
    // Whether `layout` is update-after-bind, which the pool allocating from it
    // must declare in turn.
    pub(in crate::vulkan) update_after_bind: bool,
}

// The shared lighting + environment bindings every planar global set carries: the
// light + shadow UBOs (buffer + size), the shadow map, the IBL irradiance +
// prefilter cubes (+ their sampler), and the SSAO white fallback (+ its sampler).
// All Copy vk handles, shared unchanged across every (plane, frame) global set.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PlanarLightingBindings<'a> {
    pub(in crate::vulkan) light_ubo: vk::Buffer,
    pub(in crate::vulkan) light_size: u64,
    // Per-scene local-light SSBO (global set 0 binding 9) + its byte size; the
    // shared static buffer, bound unchanged into every planar global set.
    pub(in crate::vulkan) local_light_buffer: vk::Buffer,
    pub(in crate::vulkan) local_light_size: u64,
    // Clustered lighting (global set 0 bindings 10 + 11). A reflected view does
    // not match the main camera's cluster grid, so these sets bind the static
    // `use_clusters = 0` ClusterParams and fall back to iterating every light;
    // the list SSBO is still bound because the shader references it.
    pub(in crate::vulkan) cluster_params_ubo: vk::Buffer,
    pub(in crate::vulkan) cluster_list_buffer: vk::Buffer,
    // Spot shadows (global set 0 bindings 12 + 13). Bound unchanged: a shadowed
    // spot occludes a reflected view exactly as it does the main camera.
    pub(in crate::vulkan) spot_shadow_map_view: vk::ImageView,
    pub(in crate::vulkan) spot_shadow_data_buffer: vk::Buffer,
    // Area lights (global set 0 bindings 14..16). Bound unchanged: a panel lights
    // a reflected view exactly as it does the main camera.
    pub(in crate::vulkan) area_light_buffer: vk::Buffer,
    pub(in crate::vulkan) ltc_matrix_view: vk::ImageView,
    pub(in crate::vulkan) ltc_magnitude_view: vk::ImageView,
    pub(in crate::vulkan) ltc_sampler: vk::Sampler,
    // Per-frame-in-flight ShadowUniforms ring; set `i` covers plane
    // `i / frames`, frame `i % frames`.
    pub(in crate::vulkan) shadow_ubos: &'a [PooledBuffer],
    pub(in crate::vulkan) shadow_size: u64,
    pub(in crate::vulkan) shadow_map_view: vk::ImageView,
    pub(in crate::vulkan) shadow_sampler: vk::Sampler,
    pub(in crate::vulkan) irradiance_view: vk::ImageView,
    pub(in crate::vulkan) prefilter_view: vk::ImageView,
    pub(in crate::vulkan) cube_sampler: vk::Sampler,
    pub(in crate::vulkan) ssao_white_view: vk::ImageView,
    pub(in crate::vulkan) linear_sampler: vk::Sampler,
}

impl PlanarReflectionSet {
    // Build the planar set: shared colour + depth + per-plane targets at render
    // dimensions, per-plane framebuffers, the per-(plane, frame) reflected-view
    // UBO ring, and the per-(plane, frame) global sets (each carrying its reflected
    // view + the shared lighting / env bindings + an EMPTY ProbeSet so the mirror
    // render samples only sky) + the per-(plane, frame) reflected-frustum cull
    // resources (indirect + status + cull set reading the frame's object/draw-args).
    // The bindless object SSBO + texture pool (the bindless set) is the FRAME's,
    // bound at encode time.
    pub(in crate::vulkan) fn new(
        gpu: PlanarDevice<'_>,
        config: PlanarConfig,
        planes: &[[f32; 4]],
        main_render_pass: &OwnedRenderPass,
        global_set: PlanarGlobalSet,
        lighting: PlanarLightingBindings,
        cull: PlanarCullSources<'_>,
    ) -> Result<Self, String> {
        use concinnity_core::render::uniforms::ProbeSet;

        let PlanarGlobalSet {
            layout: global_set_layout,
            probe_cube_count,
            update_after_bind: global_update_after_bind,
        } = global_set;

        let PlanarDevice { alloc, device } = gpu;
        let PlanarConfig {
            frames,
            sample_count,
            width,
            height,
        } = config;
        let PlanarLightingBindings {
            light_ubo,
            light_size,
            local_light_buffer,
            local_light_size,
            cluster_params_ubo,
            cluster_list_buffer,
            spot_shadow_map_view,
            spot_shadow_data_buffer,
            area_light_buffer,
            ltc_matrix_view,
            ltc_magnitude_view,
            ltc_sampler,
            shadow_ubos,
            shadow_size,
            shadow_map_view,
            shadow_sampler,
            irradiance_view,
            prefilter_view,
            cube_sampler,
            ssao_white_view,
            linear_sampler,
        } = lighting;

        let plane_count = planes.len();
        let (color, depth, targets) = create_targets(
            gpu,
            PlanarTargetDims {
                sample_count,
                width,
                height,
                plane_count,
            },
        )?;
        let framebuffers = create_framebuffers(
            device,
            PlanarFramebufferInputs {
                main_render_pass: main_render_pass.handle(),
                sample_count,
                color: color.as_ref(),
                depth: &depth,
                targets: &targets,
                width,
                height,
            },
        )?;

        // EMPTY ProbeSet UBO (count 0): every mirror face reflects only the sky, so
        // the planar render never recurses into the probe set it is feeding.
        let empty = ProbeSet::EMPTY;
        let probeset_size = std::mem::size_of::<ProbeSet>() as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let probeset_buf =
            alloc.create_buffer(probeset_size, vk::BufferUsageFlags::UNIFORM_BUFFER, host)?;
        probeset_buf.write_val(0, &empty);

        // Per-(plane, frame) reflected-view UBO ring.
        let view_size = std::mem::size_of::<ViewUniforms>() as u64;
        let ring = plane_count * frames;
        let mut view_bufs = Vec::with_capacity(ring);
        for _ in 0..ring {
            view_bufs.push(alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                host,
            )?);
        }

        // Per-(plane, frame) reflected-frustum cull output: a DEVICE_LOCAL indirect +
        // status SSBO each, sized by the build-time object count (resize never
        // touches them).
        use crate::gfx::render_types::{GpuDrawArgs, GpuObjectData};
        let object_range = (cull.cull_count * std::mem::size_of::<GpuObjectData>()).max(4) as u64;
        let args_range = (cull.cull_count * std::mem::size_of::<GpuDrawArgs>()).max(4) as u64;
        let indirect_size =
            (cull.cull_count * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()).max(4) as u64;
        let status_size = (cull.cull_count * std::mem::size_of::<u32>()).max(4) as u64;
        let mut cull_indirect_bufs = Vec::with_capacity(ring);
        let mut cull_status_bufs = Vec::with_capacity(ring);
        for _ in 0..ring {
            cull_indirect_bufs.push(alloc.create_buffer(
                indirect_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);
            cull_status_bufs.push(alloc.create_buffer(
                status_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);
        }

        // One pool: the per-(plane, frame) global sets (4 UBO + 4 sampler + the cube
        // array + the local-light SSBO each) + the per-(plane, frame) cull sets (4
        // storage each) + one Hi-Z set (1 sampler + 1 UBO) when the world runs Hi-Z.
        let has_hiz = cull.hiz.is_some();
        let pool_sizes = [
            // ring * (view + light + shadow + ProbeSet + ClusterParams) UBOs.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count((ring * 5 + usize::from(has_hiz)).max(1) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(
                    (ring * (7 + probe_cube_count as usize) + usize::from(has_hiz)).max(1) as u32,
                ),
            // ring * 4 cull-set SSBOs + the binding-9 local-light, binding-11
            // cluster-list, binding-13 spot-shadow and binding-14 area-light
            // SSBOs, one of each per global set.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((ring * 4 + ring + ring + ring + ring).max(1) as u32),
        ];
        // The per-(plane, frame) global sets come from the forward global set
        // layout, so this pool has to declare update-after-bind whenever that
        // layout does.
        let mut pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets((ring * 2 + usize::from(has_hiz)).max(1) as u32);
        if global_update_after_bind {
            pool_info = pool_info.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        }
        let pool = device
            .create_descriptor_pool(&pool_info)
            .map_err(|e| format!("planar descriptor pool: {e}"))?;

        let layouts: Vec<_> = (0..ring).map(|_| global_set_layout).collect();
        let global_sets = alloc_descriptor_sets(device, pool.handle(), &layouts)?;

        let probe_cube_sky: Vec<vk::DescriptorImageInfo> = (0..probe_cube_count)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(prefilter_view)
                    .sampler(cube_sampler)
            })
            .collect();
        for (i, &set) in global_sets.iter().enumerate() {
            let view_info = buf_info(view_bufs[i].buffer(), view_size);
            let light_info = buf_info(light_ubo, light_size);
            let shadow_info = buf_info(shadow_ubos[i % frames].buffer(), shadow_size);
            let probeset_info = buf_info(probeset_buf.buffer(), probeset_size);
            let shadow_img = img_info(shadow_map_view, shadow_sampler);
            let irr_img = img_info(irradiance_view, cube_sampler);
            let pre_img = img_info(prefilter_view, cube_sampler);
            let ssao_img = img_info(ssao_white_view, linear_sampler);
            let writes = [
                ubo_write(set, 0, &view_info),
                ubo_write(set, 1, &light_info),
                ubo_write(set, 2, &shadow_info),
                sampler_write(set, 3, &shadow_img),
                sampler_write(set, 4, &irr_img),
                sampler_write(set, 5, &pre_img),
                sampler_write(set, 6, &ssao_img),
                ubo_write(set, 7, &probeset_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(super::descriptor_layout::PROBE_CUBE_ARRAY_BINDING)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&probe_cube_sky),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            // Binding 9: the shared per-scene local-light SSBO.
            write_storage(
                device,
                set,
                super::descriptor_layout::LOCAL_LIGHT_SSBO_BINDING,
                local_light_buffer,
                local_light_size,
            );
            // Bindings 10 + 11: the `use_clusters = 0` ClusterParams (a reflected
            // view does not match the main camera's grid) + the cluster lists,
            // bound because the forward shader references them unconditionally.
            let cluster_params_info = vk::DescriptorBufferInfo::default()
                .buffer(cluster_params_ubo)
                .offset(0)
                .range(std::mem::size_of::<crate::gfx::render_types::ClusterParams>() as u64);
            let cluster_write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(super::descriptor_layout::CLUSTER_PARAMS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&cluster_params_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&cluster_write), &[]) };
            write_storage(
                device,
                set,
                super::descriptor_layout::CLUSTER_LIGHT_LIST_SSBO_BINDING,
                cluster_list_buffer,
                super::light_cull::cluster_list_size(),
            );
            // Bindings 12 + 13: the spot shadow depth array + its per-slice
            // projections, bound exactly as the main camera binds them.
            let spot_img = img_info(spot_shadow_map_view, shadow_sampler);
            let spot_write = sampler_write(
                set,
                super::descriptor_layout::SPOT_SHADOW_MAP_BINDING,
                &spot_img,
            );
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&spot_write), &[]) };
            write_storage(
                device,
                set,
                super::descriptor_layout::SPOT_SHADOW_DATA_SSBO_BINDING,
                spot_shadow_data_buffer,
                vk::WHOLE_SIZE,
            );
            // Bindings 14..16: the area-light table and its two LTC lookups.
            write_storage(
                device,
                set,
                super::descriptor_layout::AREA_LIGHT_SSBO_BINDING,
                area_light_buffer,
                vk::WHOLE_SIZE,
            );
            let ltc_m = img_info(ltc_matrix_view, ltc_sampler);
            let ltc_g = img_info(ltc_magnitude_view, ltc_sampler);
            let ltc_writes = [
                sampler_write(set, super::descriptor_layout::LTC_MATRIX_BINDING, &ltc_m),
                sampler_write(set, super::descriptor_layout::LTC_MAGNITUDE_BINDING, &ltc_g),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&ltc_writes, &[]) };
        }

        // Per-(plane, frame) cull sets: read the frame's object + draw-args SSBOs
        // (b0 / b1), write this plane's indirect + status (b2 / b3). Ring index
        // slot * frames + frame, so `i % frames` selects the frame's buffers.
        let cull_layouts: Vec<_> = (0..ring).map(|_| cull.cull_set_layout).collect();
        let cull_sets = alloc_descriptor_sets(device, pool.handle(), &cull_layouts)?;
        for (i, &set) in cull_sets.iter().enumerate() {
            let frame = i % frames;
            write_storage(
                device,
                set,
                0,
                cull.frame_object_buffers[frame].buffer(),
                object_range,
            );
            write_storage(
                device,
                set,
                1,
                cull.frame_draw_args_buffers[frame].buffer(),
                args_range,
            );
            write_storage(
                device,
                set,
                2,
                cull_indirect_bufs[i].buffer(),
                indirect_size,
            );
            write_storage(device, set, 3, cull_status_bufs[i].buffer(), status_size);
        }

        // The Hi-Z set (cull set 1) with hiz_enabled = 0: a frustum-only reflected
        // cull never samples the main camera's pyramid. Only when Hi-Z runs (the
        // cull pipeline layout statically references set 1 then). Shared across planes.
        let (hiz_set, hiz_ubo) = if let Some((hiz_layout, hiz_view, hiz_sampler)) = cull.hiz {
            use super::hiz::CullHizParams;
            let params = CullHizParams {
                prev_view_proj: [[0.0; 4]; 4],
                hiz_size: [1.0, 1.0],
                hiz_mip_count: 1,
                hiz_enabled: 0,
            };
            let params_size = std::mem::size_of::<CullHizParams>() as u64;
            let ubo =
                alloc.create_buffer(params_size, vk::BufferUsageFlags::UNIFORM_BUFFER, host)?;
            ubo.write_val(0, &params);
            let set =
                alloc_descriptor_sets(device, pool.handle(), std::slice::from_ref(&hiz_layout))?[0];
            let img = img_info(hiz_view, hiz_sampler);
            let ubo_info = buf_info(ubo.buffer(), params_size);
            let writes = [sampler_write(set, 0, &img), ubo_write(set, 1, &ubo_info)];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            (Some(set), Some(ubo))
        } else {
            (None, None)
        };

        Ok(Self {
            planes: planes.to_vec(),
            frames,
            sample_count,
            width,
            height,
            main_render_pass: main_render_pass.handle(),
            color,
            depth,
            targets,
            framebuffers,
            view_bufs,
            global_sets,
            probeset_buf,
            cull_indirect_bufs,
            cull_status_bufs,
            cull_sets,
            hiz_set,
            hiz_ubo,
            _pool: pool,
        })
    }

    // Number of distinct reflector planes (mirror renders per frame).
    pub(in crate::vulkan) fn plane_count(&self) -> usize {
        self.planes.len()
    }

    // The shader-readable target view for plane `slot` (what the glass pass binds
    // for a pane assigned to that slot).
    pub(in crate::vulkan) fn target_view(&self, slot: usize) -> vk::ImageView {
        self.targets[slot].view
    }

    // Re-point the reflected-frustum cull's Hi-Z set (binding 0) at a fresh pyramid
    // view + sampler after a resize. The Hi-Z resource recreates its pyramid image
    // on resize, destroying the view this set captured at `new`; the planar set
    // persists, so its set 1 would otherwise dangle a freed view (the cull binds set
    // 1 unconditionally, even though hiz_enabled = 0 keeps it unsampled). Called
    // after `hiz.resize_to`, with the device idle. A no-op when the world runs no
    // Hi-Z (`hiz_set` is None).
    pub(in crate::vulkan) fn rewrite_hiz_view(
        &self,
        device: &VkDevice,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        let Some(set) = self.hiz_set else {
            return;
        };
        let img = img_info(view, sampler);
        let write = sampler_write(set, 0, &img);
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }

    // Recreate the shared colour + depth + per-plane targets + framebuffers at new
    // render dimensions. The view UBO ring + global sets + pool survive (the global
    // sets reference only the unchanged shared lighting / env bindings + the
    // per-(plane, frame) view UBOs). The targets move, so the caller must re-point
    // the glass pass's per-pane planar binding afterward.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        // Build the new targets + framebuffers first, then retire the old ones, so
        // a failure leaves the existing set intact.
        let (color, depth, targets) = create_targets(
            PlanarDevice { alloc, device },
            PlanarTargetDims {
                sample_count: self.sample_count,
                width,
                height,
                plane_count: self.planes.len(),
            },
        )?;
        let framebuffers = create_framebuffers(
            device,
            PlanarFramebufferInputs {
                main_render_pass: self.main_render_pass,
                sample_count: self.sample_count,
                color: color.as_ref(),
                depth: &depth,
                targets: &targets,
                width,
                height,
            },
        )?;

        // The replaced targets retire through the allocator as they drop.
        self.color = color;
        self.depth = depth;
        self.targets = targets;
        self.framebuffers = framebuffers;
        self.width = width;
        self.height = height;
        Ok(())
    }

    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        // The pool frees every global / cull / Hi-Z set allocated from it.
        self.color = None;
        self.depth = GpuImage::null();
        self.framebuffers.clear();
        self.targets.clear();
        self.view_bufs.clear();
        self.cull_indirect_bufs.clear();
        self.cull_status_bufs.clear();
        self.hiz_ubo = None;
        self.probeset_buf = PooledBuffer::null();
        self.global_sets.clear();
        self.cull_sets.clear();
    }
}

fn buf_info(buffer: vk::Buffer, range: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(range)
}

fn write_storage(
    device: &VkDevice,
    set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    range: u64,
) {
    let info = buf_info(buffer, range);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn img_info(view: vk::ImageView, sampler: vk::Sampler) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler)
}

fn ubo_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}

fn sampler_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a vk::DescriptorImageInfo,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(info))
}

impl VkContext {
    // Render the scene reflected across each plane in the planar set into that
    // plane's target. A no-op when no set exists. For each plane: write the
    // reflected ViewUniforms into this (plane, frame) ring slot, run the dedicated
    // reflected-frustum cull into the plane's indirect, then render the culled set
    // from the reflected view through the shared bindless encode_main_into_face into
    // the plane's framebuffer. Encoded on `cmd` at the
    // head of the transparent pass, before the glass draws sample the targets;
    // same-cmd-buffer ordering retires each target before its glass sample. Each
    // plane is oriented toward the camera so the oblique near-plane clip keeps the
    // camera's side.
    pub(in crate::vulkan) fn encode_planar_reflections(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        vp_mat: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        elapsed: f32,
    ) -> Result<(), String> {
        let Some(set) = self.planar_reflection.as_ref() else {
            return Ok(());
        };
        let Some(&bindless_set) = self.cull.bindless_sets.get(frame_idx) else {
            return Ok(());
        };

        // Recover the (jittered) projection from this frame's view-projection so the
        // mirror render shares the main camera's projection + jitter, keeping the
        // reflection aligned with the reflective fragment's screen-space sample.
        let proj = mat4_mul(vp_mat, mat4_inverse(self.view.matrix));
        let prefilter_mip_count = self.prefilter_mip_count as f32;
        let extent = vk::Extent2D {
            width: set.width,
            height: set.height,
        };

        for slot in 0..set.plane_count() {
            let oriented =
                crate::gfx::planar_reflection::orient_plane_toward(set.planes[slot], cam_pos);
            let m = crate::gfx::planar_reflection::planar_matrices(
                self.view.matrix,
                proj,
                cam_pos,
                oriented,
                PLANAR_CLIP_BIAS,
            );
            let view = ViewUniforms {
                vp: m.view_proj,
                view: m.view,
                elapsed,
                // No reflection composite runs over the mirror render, so the
                // forward probe specular is its only reflection source; the EMPTY
                // ProbeSet then leaves it on the sky path.
                reflections_enabled: 0.0,
                cam_pos: [m.eye[0], m.eye[1], m.eye[2]],
                prefilter_mip_count,
                // A mirror render is always lit, whatever the viewport shows.
                shade_mode: 0.0,
                _end_pad: 0.0,
            };
            let ring = slot * set.frames + frame_idx;
            set.view_bufs[ring].write_val(0, &view);
            // Reflected-frustum cull (compute, outside any render pass) into this
            // plane's indirect, reading the frame's camera-independent object set so
            // geometry visible only in the reflection is captured. The oblique clip
            // already rides the view-proj, so the extracted frustum also rejects
            // geometry behind the reflector.
            let frustum = crate::gfx::frustum::Frustum::from_view_projection(m.view_proj);
            self.encode_probe_cull(cmd, set.cull_sets[ring], set.hiz_set, &frustum, m.eye);
            // Order the previous mirror render's attachment writes before this one's
            // layout transition. `main_render_pass` declares both attachments
            // `initial_layout = UNDEFINED`, so every `vkCmdBeginRenderPass` here
            // write-after-writes the last render that touched them: the depth is
            // shared by every plane, and each plane's colour target is the one its
            // own render wrote last frame. The render pass's external dependency
            // declares an empty src access mask -- an execution dependency with no
            // availability operation -- so nothing else covers it. Needed on the
            // first plane too, where the prior write is the previous frame's submit;
            // the single graphics queue's submission order carries it across.
            let attachment_waw = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );
            // SAFETY: `cmd` is the frame's recording command buffer, inside a
            // recording scope and outside a render pass, which is where
            // `vkCmdPipelineBarrier` is legal; the barrier owns no resource
            // handles (a global `VkMemoryBarrier`, no buffer or image references
            // to outlive), and `from_ref` gives the one-element slice the count
            // implies.
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&attachment_waw),
                    &[],
                    &[],
                );
            }
            self.encode_main_into_face(
                cmd,
                set.framebuffers[slot].handle(),
                extent,
                set.global_sets[ring],
                bindless_set,
                set.cull_indirect_bufs[ring].buffer(),
            );
        }

        // Make every freshly rendered target visible to the glass fragment read.
        // The main render pass leaves them in SHADER_READ_ONLY (final layout) but
        // adds no output-side dependency, so order the colour writes before the
        // sample explicitly. Layout is unchanged (SHADER_READ_ONLY -> same).
        let barriers: Vec<vk::ImageMemoryBarrier> = set
            .targets
            .iter()
            .map(|t| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(t.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
            })
            .collect();
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pane_plane_passes_through_centre_with_unit_normal() {
        // A pane facing +z through (1, 2, 3): the plane constant places the centre
        // on the surface (n . c + d == 0), and the normal is carried unchanged.
        let p = pane_plane([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);
        assert_eq!([p[0], p[1], p[2]], [0.0, 0.0, 1.0]);
        let signed = p[0] * 1.0 + p[1] * 2.0 + p[2] * 3.0 + p[3];
        assert!(signed.abs() < 1e-5, "centre lies on the plane");
    }

    #[test]
    fn pane_plane_offset_is_negative_normal_dot_centre() {
        // Tilted normal: d == -(n . c).
        let n = [0.6, 0.0, 0.8];
        let c = [2.0, 5.0, -1.0];
        let p = pane_plane(n, c);
        let expect_d = -(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]);
        assert!((p[3] - expect_d).abs() < 1e-5);
    }

    #[test]
    fn planar_capacity_is_four() {
        // The reserved planar targets are sized off this. It now aliases the single
        // `gfx::planar_reflection` source, so this guards that the shared capacity
        // the allocation assumes is still 4.
        assert_eq!(MAX_PLANAR_PLANES, 4);
        assert_eq!(
            MAX_PLANAR_PLANES,
            crate::gfx::planar_reflection::MAX_PLANAR_PLANES
        );
    }
}
