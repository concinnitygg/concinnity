//! Planar reflection for flat glass panes on the Vulkan backend. Each frame the
//! scene is rendered a second time from the camera reflected across each pane's
//! plane (mirror view + oblique near-plane clip so geometry behind the plane
//! never leaks in) into a render-resolution target; the pane's fragment shader
//! then samples that target projectively for a sharp, scene-correct reflection
//! instead of the box-projected probe cube.
//!
//! GLSL/Vulkan port of src/directx/planar.rs (itself a port of src/metal/planar.rs),
//! One mirror render per DISTINCT
//! plane: near-coplanar panes (one wall of windows) share a render, and panes past
//! the budget (MAX_PLANAR_PLANES) fall back to the probe cube. The plane -> slot
//! grouping + the mirror matrices come from the pure, unit-tested
//! gfx::planar_reflection.
//!
//! Each plane gets a DEDICATED reflected-frustum cull (the shared probe-bake
//! encode_probe_cull): the GPU cull re-runs against the reflected-camera frustum
//! into that plane's own indirect buffer, reading the FRAME's camera-independent
//! object + draw-args SSBOs. So geometry visible only in the reflection (behind /
//! beside the main camera) is captured, not just the main camera's visible set; the
//! reflected view-proj's oblique near-plane clip also rejects geometry behind the
//! reflector. The face render then draws that indirect. Like the probe capture, the
//! skinned tail is not drawn into a mirror (static + instance + chunk only).

use ash::vk;
use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::planar_reflection;
use concinnity_core::transform::mat4_inverse;
use concinnity_core::transform::mat4_mul;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::{HDR_FORMAT, VkContext};
use super::descriptor_layout::{PoolSizes, global_set};
use super::draw::ViewUniforms;
use super::global_set::{GlobalBindings, GlobalSetContents};
use super::resources::{alloc_descriptor_sets, write_storage_buffer};
use super::texture::{GpuImage, ImageSpec, create_image, create_image_view};
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedFramebuffer, OwnedRenderPass, VkDevice};

// The engine capacity ceiling for distinct reflection planes: the count the
// reserved planar targets are sized to. Single-sourced from `gfx::planar_reflection`
// so the three backends stay in lockstep by construction. The per-frame budget
// passed to `assign_planar_slots` at init can be lower under a quality preset / GPU
// tier, never higher; panes past it fall back to the box-projected probe cube.
pub(in crate::vulkan) const MAX_PLANAR_PLANES: usize = planar_reflection::MAX_PLANAR_PLANES;

// Clip the reflection a hair toward the kept (camera) side of the plane so
// geometry exactly on the surface is not lost to near-plane precision. Matches
// the other backends' PLANAR_CLIP_BIAS.
const PLANAR_CLIP_BIAS: f32 = 0.02;
const PLANAR_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// World-space plane [nx, ny, nz, d] (unit normal, n . p + d = 0 on the surface)
// for a glass pane with unit normal through center. Pure; unit tested. The init
// path feeds these to assign_planar_slots.
pub(in crate::vulkan) fn pane_plane(normal: [f32; 3], center: [f32; 3]) -> [f32; 4] {
    [
        normal[0],
        normal[1],
        normal[2],
        -(normal[0] * center[0] + normal[1] * center[1] + normal[2] * center[2]),
    ]
}

// The set of distinct reflection planes for the world, each rendering its mirror
// into the shared color + depth then resolving into its own shader-readable
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

    // Shared MSAA color (Some only when MSAA) + shared depth, reused across
    // planes (rendered one plane at a time on the frame's cmd buffer) and across
    // frames (the single graphics queue executes submissions in order). Recreated
    // on resize.
    color: Option<GpuImage>,
    depth: GpuImage,
    // Per-plane shader-readable target: the MSAA resolve when MSAA, else the
    // single-sample color attachment itself. The glass pass samples it. Recreated
    // on resize.
    targets: Vec<GpuImage>,
    framebuffers: Vec<OwnedFramebuffer>,

    // Per-(plane, frame) reflected ViewUniforms UBO ring (HOST_VISIBLE, mapped),
    // indexed plane * frames + frame, so the CPU writes this frame's slot without
    // racing the GPU reading a prior frame's. Bound at binding 0 of the matching
    // planar global set.
    view_bufs: Vec<PooledBuffer>,
    // Per-(plane, frame) global set (the bindless main set): that (plane,
    // frame)'s reflected view and no probe set, so the mirror render reflects
    // only the sky and never recurses into the probes it feeds.
    global_sets: Vec<vk::DescriptorSet>,

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
// read-set layout + pyramid view so a hiz_enabled = 0 set can be bound
// (the cull pipeline layout statically references set 1).
pub(in crate::vulkan) struct PlanarCullSources<'a> {
    pub(in crate::vulkan) frame_object_buffers: &'a [PooledBuffer],
    pub(in crate::vulkan) frame_draw_args_buffers: &'a [PooledBuffer],
    pub(in crate::vulkan) cull_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) cull_count: usize,
    pub(in crate::vulkan) hiz: Option<(vk::DescriptorSetLayout, vk::ImageView)>,
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

// Render dimensions for the shared color + depth + per-plane targets: the MSAA
// sample count, pixel dimensions, and how many per-plane targets to create.
#[derive(Clone, Copy)]
struct PlanarTargetDims {
    sample_count: vk::SampleCountFlags,
    width: u32,
    height: u32,
    plane_count: usize,
}

// Create the shared color (MSAA only) + shared depth + per-plane targets at the
// given render dimensions.
fn create_targets(
    gpu: PlanarDevice<'_>,
    dims: PlanarTargetDims,
) -> RenderResult<(Option<GpuImage>, GpuImage, Vec<GpuImage>)> {
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
// pass, the MSAA sample count, the shared color (MSAA only) + shared depth reused
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
// MSAA -> [shared color, shared depth, plane target (resolve)], single-sample ->
// [plane target (color), shared depth].
fn create_framebuffers(
    device: &VkDevice,
    inputs: PlanarFramebufferInputs<'_>,
) -> RenderResult<Vec<OwnedFramebuffer>> {
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
                    .expect("a multisampled planar target has a color image")
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
            .map_err(|e| super::error::map_vk_result(e, "planar framebuffer"))?;
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

// The forward global set every planar re-render binds: its layout, and the
// resources its per-(plane, frame) sets are built from.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PlanarGlobalSet<'a> {
    pub(in crate::vulkan) layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) bindings: GlobalBindings<'a>,
}

// What a planar global set holds: its (plane, frame) reflected view, and that
// frame's light and shadow UBOs.
fn global_contents(
    bindings: &GlobalBindings<'_>,
    view: vk::Buffer,
    frame: usize,
) -> GlobalSetContents {
    bindings.off_camera(
        view,
        bindings.uniforms.light_ubo_buffers[frame].buffer(),
        bindings.shadow.ubos[frame].buffer(),
    )
}

impl PlanarReflectionSet {
    // Build the planar set: shared color + depth + per-plane targets at render
    // dimensions, per-plane framebuffers, the per-(plane, frame) reflected-view
    // UBO ring, and the per-(plane, frame) global sets (each carrying its reflected
    // view and no probe set, so the mirror render samples only sky) + the
    // per-(plane, frame) reflected-frustum cull
    // resources (indirect + status + cull set reading the frame's object/draw-args).
    // The bindless object SSBO + texture pool (the bindless set) is the FRAME's,
    // bound at encode time.
    pub(in crate::vulkan) fn new(
        gpu: PlanarDevice<'_>,
        config: PlanarConfig,
        planes: &[[f32; 4]],
        main_render_pass: &OwnedRenderPass,
        globals: PlanarGlobalSet<'_>,
        cull: PlanarCullSources<'_>,
    ) -> RenderResult<Self> {
        let PlanarDevice { alloc, device } = gpu;
        let PlanarConfig {
            frames,
            sample_count,
            width,
            height,
        } = config;
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

        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
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
        use concinnity_core::gfx::render_types::{GpuDrawArgs, GpuObjectData};
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

        // One pool: the per-(plane, frame) global and cull sets, and one Hi-Z
        // set (an image and a UBO) when the world runs Hi-Z.
        let has_hiz = u32::from(cull.hiz.is_some());
        let ring_sets = ring as u32;
        let pool_sizes = PoolSizes::default()
            .sets(&global_set(), ring_sets)
            .add(vk::DescriptorType::STORAGE_BUFFER, ring_sets * 4)
            .add(vk::DescriptorType::UNIFORM_BUFFER, has_hiz)
            .add(vk::DescriptorType::SAMPLED_IMAGE, has_hiz)
            .build();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(ring_sets * 2 + has_hiz);
        let pool = device
            .create_descriptor_pool(&pool_info)
            .map_err(|e| super::error::map_vk_result(e, "planar descriptor pool"))?;

        // Per-(plane, frame) global sets: set `i` covers plane `i / frames`,
        // frame `i % frames`, through that frame's light and shadow UBOs.
        let PlanarGlobalSet { layout, bindings } = globals;
        let global_sets = alloc_descriptor_sets(device, pool.handle(), &vec![layout; ring])?;
        for (i, &set) in global_sets.iter().enumerate() {
            global_contents(&bindings, view_bufs[i].buffer(), i % frames).write(device, set);
        }

        // Per-(plane, frame) cull sets: read the frame's object + draw-args SSBOs
        // (b0 / b1), write this plane's indirect + status (b2 / b3). Ring index
        // slot * frames + frame, so `i % frames` selects the frame's buffers.
        let cull_layouts: Vec<_> = (0..ring).map(|_| cull.cull_set_layout).collect();
        let cull_sets = alloc_descriptor_sets(device, pool.handle(), &cull_layouts)?;
        for (i, &set) in cull_sets.iter().enumerate() {
            let frame = i % frames;
            write_storage_buffer(
                device,
                set,
                0,
                cull.frame_object_buffers[frame].buffer(),
                object_range,
            );
            write_storage_buffer(
                device,
                set,
                1,
                cull.frame_draw_args_buffers[frame].buffer(),
                args_range,
            );
            write_storage_buffer(
                device,
                set,
                2,
                cull_indirect_bufs[i].buffer(),
                indirect_size,
            );
            write_storage_buffer(device, set, 3, cull_status_bufs[i].buffer(), status_size);
        }

        // The Hi-Z set (cull set 1) with hiz_enabled = 0: a frustum-only reflected
        // cull never samples the main camera's pyramid. Only when Hi-Z runs (the
        // cull pipeline layout statically references set 1 then). Shared across planes.
        let (hiz_set, hiz_ubo) = match cull.hiz {
            Some((layout, view)) => {
                let (set, ubo) =
                    super::hiz::off_camera_read_set(alloc, device, pool.handle(), layout, view)?;
                (Some(set), Some(ubo))
            }
            None => (None, None),
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

    // Rewrite binding `binding` of every (plane, frame) global set from what it
    // is built with now. The caller has idled the device.
    pub(in crate::vulkan) fn rewrite_global_binding(
        &self,
        device: &VkDevice,
        bindings: &GlobalBindings<'_>,
        binding: u32,
    ) {
        for (i, &set) in self.global_sets.iter().enumerate() {
            global_contents(bindings, self.view_bufs[i].buffer(), i % self.frames)
                .write_binding(device, set, binding);
        }
    }

    // The shader-readable target view for plane `slot` (what the glass pass binds
    // for a pane assigned to that slot).
    pub(in crate::vulkan) fn target_view(&self, slot: usize) -> vk::ImageView {
        self.targets[slot].view
    }

    // Re-point the reflected-frustum cull's Hi-Z set (binding 0) at a fresh pyramid
    // view after a resize. The Hi-Z resource recreates its pyramid image
    // on resize, destroying the view this set captured at `new`; the planar set
    // persists, so its set 1 would otherwise dangle a freed view (the cull binds set
    // 1 unconditionally, even though hiz_enabled = 0 keeps it unsampled). Called
    // after `hiz.resize_to`, with the device idle. A no-op when the world runs no
    // Hi-Z (`hiz_set` is None).
    pub(in crate::vulkan) fn rewrite_hiz_view(&self, device: &VkDevice, view: vk::ImageView) {
        if let Some(set) = self.hiz_set {
            super::hiz::rewrite_read_set_view(device, set, view);
        }
    }

    // Recreate the shared color + depth + per-plane targets + framebuffers at new
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
    ) -> RenderResult<()> {
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
        self.global_sets.clear();
        self.cull_sets.clear();
    }
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
    ) -> RenderResult<()> {
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
        let prefilter_mip_count = self.scene.prefilter_mip_count as f32;
        let extent = vk::Extent2D {
            width: set.width,
            height: set.height,
        };

        for slot in 0..set.plane_count() {
            let oriented = planar_reflection::orient_plane_toward(set.planes[slot], cam_pos);
            let m = planar_reflection::planar_matrices(
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
                sky_rot: self.view.sky_rot,
            };
            let ring = slot * set.frames + frame_idx;
            set.view_bufs[ring].write_val(0, &view);
            // Reflected-frustum cull (compute, outside any render pass) into this
            // plane's indirect, reading the frame's camera-independent object set so
            // geometry visible only in the reflection is captured. The oblique clip
            // already rides the view-proj, so the extracted frustum also rejects
            // geometry behind the reflector.
            let frustum = Frustum::from_view_projection(m.view_proj);
            self.encode_probe_cull(cmd, set.cull_sets[ring], set.hiz_set, &frustum, m.eye);
            // Order the previous mirror render's attachment writes before this one's
            // layout transition. `main_render_pass` declares both attachments
            // `initial_layout = UNDEFINED`, so every `vkCmdBeginRenderPass` here
            // write-after-writes the last render that touched them: the depth is
            // shared by every plane, and each plane's color target is the one its
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
                self.hw.device.cmd_pipeline_barrier(
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
        // adds no output-side dependency, so order the color writes before the
        // sample explicitly. Layout is unchanged (SHADER_READ_ONLY -> same).
        // One barrier per plane, and the plane count is capped at
        // `MAX_PLANAR_PLANES` where the set is built, so this fits on the stack.
        let mut barriers = [vk::ImageMemoryBarrier::default(); MAX_PLANAR_PLANES];
        debug_assert!(set.targets.len() <= MAX_PLANAR_PLANES);
        let n = set.targets.len().min(MAX_PLANAR_PLANES);
        for (slot, t) in set.targets.iter().take(n).enumerate() {
            barriers[slot] = vk::ImageMemoryBarrier::default()
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
                });
        }
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            self.hw.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers[..n],
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pane_plane_passes_through_center_with_unit_normal() {
        // A pane facing +z through (1, 2, 3): the plane constant places the center
        // on the surface (n . c + d == 0), and the normal is carried unchanged.
        let p = pane_plane([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);
        assert_eq!([p[0], p[1], p[2]], [0.0, 0.0, 1.0]);
        let signed = p[0] * 1.0 + p[1] * 2.0 + p[2] * 3.0 + p[3];
        assert!(signed.abs() < 1e-5, "center lies on the plane");
    }

    #[test]
    fn pane_plane_offset_is_negative_normal_dot_center() {
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
        assert_eq!(MAX_PLANAR_PLANES, planar_reflection::MAX_PLANAR_PLANES);
    }
}
