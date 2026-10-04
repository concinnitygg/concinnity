//! Scene-captured reflection probes on Vulkan. Each declared `ReflectionProbe`
//! (or an auto-seeded grid when a world declares none) describes a cube to bake
//! DISTINCT from `env_map`: the specular reflection term box-projects against the
//! probe's influence box and samples its cube, so glossy surfaces reflect the
//! actual surrounding geometry instead of the imported HDR sky, while the skybox +
//! diffuse irradiance keep sampling `env_map` so the visible sky is never replaced.
//!
//! The cube math + the staggered-bake state machine are backend-agnostic
//! (`concinnity_core::render::reflection_probe`); this module drives the placement intake + the
//! GPU capture, mirroring `crate::directx::probe` / `crate::metal::probe`.
//!
//! `set_reflection_probes` converts the graphics-system placements (auto-seeding a
//! grid from the scene bounds when a world declares none) into the stored placement
//! list + an EMPTY `ProbeSet`, then enqueues them. `bake_pending_probes` (driven each
//! frame from `draw_frame`) advances the shared `ProbeBake` sequencing, which this
//! module serves as a `ProbeBakeDevice`:
//! it renders one cube face per frame into a bake-owned target on a per-face fence
//! and copies it into a cube layer, convolves that capture into the probe cube with
//! the compute kernels in `probe_prefilter.hlsl` (the source pyramid in one frame,
//! then one GGX mip per frame), and installs the finished cube into the forward /
//! SSR / RT cube array -- all without blocking the render loop (the sky reflection
//! covers a probe until its cube installs). Nothing is read back and no convolution
//! runs on the CPU. The forward / SSR / RT sampling lives in the main / resolve
//! shaders (see the reflection_probes.md DX/VK port checklist).

use ash::vk;
use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::gfx::render_types;
use concinnity_core::render::depth::DepthConvention;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::probe_bake::{CAPTURE_FACES, ProbeBake, ProbeBakeDevice};
use concinnity_core::render::probe_book::ProbeBook;
use concinnity_core::render::reflection_probe::{self, PrefilterPlan, ProbePlacement};

use super::allocator::PooledBuffer;
use super::context::{HDR_FORMAT, VkContext};
use super::descriptor_layout::{PoolSizes, global_set};
use super::draw::ViewUniforms;
use super::global_set::{GlobalBindings, GlobalSetContents};
use super::material_params::MATERIAL_PARAMS_BINDING;
use super::probe_prefilter::PrefilterGpu;
use super::resources::alloc_descriptor_sets;
use super::set_writes::SetWrites;
use super::texture::{GpuImage, ImageSpec, create_image, create_image_view};
use crate::vulkan::depth;
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedFramebuffer, VkDevice};
use concinnity_core::render::uniforms::vulkan::CullParams;

// What a runtime capture bakes: face size, mip count, GGX sample count and firefly
// clamp, shared with the DirectX and Metal backends (and with the build-time CPU
// convolution's roughness ramp) so a probe looks the same whichever backend
// captured it.
pub(super) const PLAN: PrefilterPlan = PrefilterPlan::RUNTIME;
// Captured cube-face resolution (mip 0 of the prefilter chain).
const PROBE_FACE_SIZE: u32 = PLAN.face_size();
// Depth format of the probe-face target (matches the main pass's DSV).
const PROBE_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// The cull push constant for an off-camera capture (a probe face or a planar
// mirror plane), which differs from the main camera's in one way: `bucket_count`
// is 1, so every record is routed into region 0 whatever shader bucket it belongs
// to. The capture callers allocate a single-region indirect buffer and draw it
// with the one default bindless pipeline, so a bucketed record must land in
// region 0 to appear at all -- with default shading, which is the documented
// trade the DirectX and Metal capture paths make too.
fn capture_cull_params(frustum: &Frustum, cam_pos: [f32; 3], n_cull: u32) -> CullParams {
    let mut params = CullParams {
        planes: [[0.0; 4]; 6],
        cam_pos,
        object_count: n_cull,
        bucket_count: 1,
        // Never indexed with `bucket_count == 1` (region 0 starts at 0), but it
        // names the region capacity the caller sized its buffer with.
        bucket_stride: n_cull,
    };
    for (i, p) in frustum.planes.iter().enumerate().take(6) {
        params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
    }
    params
}

// The bake's two slots: a capture rendering its faces and a capture convolving
// into its cube.
pub(super) type VkProbeBake = ProbeBake<RenderingBake, PrefilteringBake>;

impl VkContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. The cube array grows to hold every
    // placement; a world whose array cannot grow keeps the sky. Pushed once
    // after construction; the cube capture that fills the probe set runs across
    // later frames.
    pub(super) fn set_reflection_probes(&mut self, declared: &[ProbePlacement]) {
        let placements = reflection_probe::resolve_placements(
            declared,
            self.state.draw.objects.iter().map(|o| (o.bb_min, o.bb_max)),
        );
        // Idle first when probes are installed: the frames in flight may sample
        // the cubes the next bake overwrites.
        if self.probe.book.count() > 0 {
            self.wait_idle();
        }
        let placed = self.with_probe_bake(|bake, ctx| bake.place(ctx, placements));
        crate::probe_report::report_probe_placement(placed);
    }

    // Advance the staggered reflection-probe bake one frame. Called every frame
    // from `draw_frame` after this frame's slot fence wait; cheap once the queue
    // drains. A failure abandons the remaining bakes, keeping what installed.
    //
    // Static + streamed-chunk geometry only: instanced + skinned draws are left
    // disabled in the bake cull buffers (the kernel skips them). They still
    // RECEIVE probe reflections. Lighting is cold, so shadows may be unpopulated
    // on the first frames, like the DX / Metal first-frame bake.
    pub(super) fn bake_pending_probes(&mut self) {
        let report = self.with_probe_bake(|bake, ctx| bake.advance(ctx, &()));
        crate::probe_report::report_probe_bake(report);
    }

    // Run `f` over the bake with this context as its device. The slots are lent
    // to `f`, so the context reads them as empty for the call.
    fn with_probe_bake<R>(&mut self, f: impl FnOnce(&mut VkProbeBake, &mut Self) -> R) -> R {
        let mut bake = std::mem::take(&mut self.probe.bake);
        let out = f(&mut bake, self);
        self.probe.bake = bake;
        out
    }

    // Rewrite the in-flight capture's face view uniforms from the live scene,
    // after an environment reload changed the prefilter mip count they carry. The
    // caller has idled the device.
    pub(super) fn rewrite_probe_capture_views(&self) {
        if let Some(rendering) = self.probe.bake.capture() {
            rendering.write_face_views(self.scene.prefilter_mip_count, self.state.view.sky_rot);
        }
    }

    // Write the whole live texture pool into a bake face's bindless set
    // (binding 1). Called right before the face records, so the face samples
    // the pool as it stands this frame.
    fn write_probe_face_pool(&self, set: vk::DescriptorSet) {
        // Every slot the layout declares, padded with the last reserved fallback
        // across the unused tail exactly as init fills the frame's own sets.
        let mut pool_infos: Vec<vk::DescriptorImageInfo> = self
            .scene
            .textures
            .iter()
            .chain(self.scene.fallback_textures.iter())
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(img.view)
            })
            .collect();
        if let Some(&tail) = pool_infos.last() {
            pool_infos.resize(self.cull.bindless_pool_size, tail);
        }
        SetWrites::new(set)
            .images(1, vk::DescriptorType::SAMPLED_IMAGE, &pool_infos)
            .apply(&self.hw.device);
    }

    // Submit cube face `face` of `capture`: a fresh command buffer that culls for
    // this face's frustum, draws the bindless main into the bake target, and
    // copies the resolved face into its cube layer, on a per-face fence (polled,
    // never waited). The command buffer + fence are held in the `RenderingBake`
    // until the convolution starts, so the last face's fence retiring means the
    // whole capture is done.
    fn record_probe_face(&self, capture: &mut RenderingBake, face: usize) -> RenderResult<()> {
        let device = self.hw.device.clone();
        let extent = vk::Extent2D {
            width: PROBE_FACE_SIZE,
            height: PROBE_FACE_SIZE,
        };
        let eye = capture.eye;
        let b = &capture.bake;
        let (cull_set, hiz_set, framebuffer, global_set, bindless_set, indirect, copy_src) = (
            b.cull_set,
            b.hiz_set,
            b.framebuffer.handle(),
            b.global_sets[face],
            b.bindless_sets[face],
            b.indirect_buf.buffer(),
            b.copy_source(),
        );
        let capture_image = capture.prefilter.capture_image();

        // Snapshot the live texture pool into this face's set. The set has
        // never been bound in a submitted command buffer (each face uses its
        // own), so the write is legal without a drain, and a texture streamed
        // in since the bake started is picked up here.
        self.write_probe_face_pool(bindless_set);

        // A fresh command buffer + fence for this face, from the one-shot pool.
        // Register both in the `RenderingBake` the instant they exist so a later
        // record / submit error still reclaims them when the failed bake is
        // abandoned, which idles the device before `RenderingBake::destroy`.
        let cmd = {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.commands.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the create-info and every slice it borrows are live for the call, and each
            // handle it names belongs to this device.
            unsafe { device.allocate_command_buffers(&info) }
                .map_err(|e| super::error::map_vk_result(e, "probe face cmd alloc"))?[0]
        };
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f,
            Err(e) => {
                // The command buffer is allocated but not yet tracked; free it before
                // bailing so it does not leak.
                // SAFETY: the handle was created from this device moments ago and never submitted,
                // so this cleanup is its only remaining use.
                unsafe {
                    device.free_command_buffers(
                        self.commands.command_pool,
                        std::slice::from_ref(&cmd),
                    );
                }
                return Err(super::error::map_vk_result(e, "probe face fence"));
            }
        };
        capture.face_cmds.push(cmd);
        capture.face_fences.push(fence);

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` was allocated from this device's pool and is not in flight (its face fence
        // was waited on), so it is in the initial state that `begin` requires.
        unsafe { device.begin_command_buffer(cmd, &begin) }
            .map_err(|e| super::error::map_vk_result(e, "probe face begin"))?;
        // Order the previous face's cube copy + indirect-draw read (a prior
        // frame's submit) before this face's cull (rewrites the shared indirect
        // buffer) and resolve (rewrites the shared color). Intra-queue, so the
        // queue's submission order preserves it across the separate submits.
        //
        // The attachment writes are here for a second reason: all six faces share
        // one framebuffer, and `main_render_pass` declares `initial_layout =
        // UNDEFINED`, so this face's `vkCmdBeginRenderPass` performs a layout
        // transition that write-after-writes the previous face's storeOp. The
        // render pass's own external dependency declares an empty src access mask,
        // an execution dependency with no availability operation, so nothing else
        // covers it.
        if face > 0 {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::INDIRECT_COMMAND_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .dst_access_mask(
                    vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER
                        | vk::PipelineStageFlags::DRAW_INDIRECT
                        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&barrier),
                    &[],
                    &[],
                );
            }
        }
        let vp = reflection_probe::face_view_projection(eye, face);
        let frustum = Frustum::from_camera(vp);
        self.encode_probe_cull(cmd, cull_set, hiz_set, &frustum, eye);
        self.encode_main_into_face(
            cmd,
            framebuffer,
            FaceArea::whole(extent),
            global_set,
            bindless_set,
            indirect,
        );
        // The face color rests in SHADER_READ_ONLY_OPTIMAL after the render pass;
        // flip it to TRANSFER_SRC for the copy into the capture cube. This exact
        // transition is the one the shared layout-transition table omits.
        let to_src = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(copy_src)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // The capture cube is created UNDEFINED; the first face is what puts it in
        // TRANSFER_DST, and it stays there until the convolution starts.
        let capture_to_dst = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(capture_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: PLAN.mips(),
                base_array_layer: 0,
                layer_count: 6,
            });
        // This face's color into the cube's matching layer, at mip 0. Face order
        // is the hardware cube order (`gfx::cubemap`), so layer `face` is the face
        // a sampler finds looking that way.
        let copy = vk::ImageCopy::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: face as u32,
                layer_count: 1,
            })
            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .extent(vk::Extent3D {
                width: PROBE_FACE_SIZE,
                height: PROBE_FACE_SIZE,
                depth: 1,
            });
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            let barriers = if face == 0 {
                vec![to_src, capture_to_dst]
            } else {
                vec![to_src]
            };
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
            device.cmd_copy_image(
                cmd,
                copy_src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                capture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy),
            );
            device
                .end_command_buffer(cmd)
                .map_err(|e| super::error::map_vk_result(e, "probe face end"))?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device
                .queue_submit(self.hw.graphics_queue, std::slice::from_ref(&submit), fence)
                .map_err(|e| super::error::map_vk_result(e, "probe face submit"))?;
        }

        Ok(())
    }

    // Allocate a command buffer + fence for one convolution step and register both
    // on the bake the instant they exist, so a later record / submit failure still
    // reclaims them when the failed bake is abandoned.
    fn begin_prefilter_command(
        &self,
        bake: &mut PrefilteringBake,
    ) -> RenderResult<(vk::CommandBuffer, vk::Fence)> {
        let device = &self.hw.device;
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.commands.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let cmd = unsafe { device.allocate_command_buffers(&info) }
            .map_err(|e| super::error::map_vk_result(e, "probe convolve cmd alloc"))?[0];
        // SAFETY: the create-info is live for the call and names only this device.
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f,
            Err(e) => {
                // Allocated but not yet tracked; free it before bailing.
                // SAFETY: the handle was allocated from this device's pool moments ago and never
                // submitted, so this cleanup is its only remaining use.
                unsafe {
                    device.free_command_buffers(
                        self.commands.command_pool,
                        std::slice::from_ref(&cmd),
                    );
                }
                return Err(super::error::map_vk_result(e, "probe convolve fence"));
            }
        };
        bake.cmds.push(cmd);
        bake.fences.push(fence);
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` was allocated from this device's pool moments ago and has never been
        // submitted, so it is in the initial state that `begin` requires.
        unsafe { device.begin_command_buffer(cmd, &begin) }
            .map_err(|e| super::error::map_vk_result(e, "probe convolve begin"))?;
        Ok((cmd, fence))
    }

    fn submit_prefilter_command(
        &self,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> RenderResult<()> {
        // SAFETY: `cmd` is in the recording state and every handle these calls name belongs to this
        // device; the fence is unsignaled and not already in use.
        unsafe {
            self.hw
                .device
                .end_command_buffer(cmd)
                .map_err(|e| super::error::map_vk_result(e, "probe convolve end"))?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.hw
                .device
                .queue_submit(self.hw.graphics_queue, std::slice::from_ref(&submit), fence)
                .map_err(|e| super::error::map_vk_result(e, "probe convolve submit"))
        }
    }

    // Dispatch the compute cull for one probe face (or one planar mirror plane)
    // into the caller's indirect buffer. A thin sibling of `encode_cull`: it binds
    // the given cull set (set 0) and -- when the world runs Hi-Z -- a Hi-Z set
    // (set 1, written with `hiz_enabled = 0` so the frustum-only cull never samples
    // the pyramid; the cull layout statically references set 1, so it must be
    // bound), pushes the face/plane frustum + eye, dispatches one invocation per
    // record, and orders the writes before the indirect draw's read. Shared by the
    // probe bake + the planar reflection's reflected-frustum cull.
    //
    // `bucket_count = 1` routes every record into region 0 whatever shader bucket
    // it belongs to, matching the single indirect region these callers allocate and
    // the one bindless pipeline `encode_main_into_face` draws it with: a bucketed
    // draw appears in the capture with default shading rather than not at all.
    pub(in crate::vulkan) fn encode_probe_cull(
        &self,
        cmd: vk::CommandBuffer,
        cull_set: vk::DescriptorSet,
        hiz_set: Option<vk::DescriptorSet>,
        frustum: &Frustum,
        cam_pos: [f32; 3],
    ) {
        let Some(kernels) = self.cull.cull_kernels.as_ref() else {
            return;
        };
        let (pipeline, layout) = (&kernels.pipeline, &kernels.pipeline_layout);
        let device = &self.hw.device;
        let params = capture_cull_params(frustum, cam_pos, self.cull_count() as u32);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout.handle(),
                0,
                std::slice::from_ref(&cull_set),
                &[],
            );
            if let Some(hs) = hiz_set {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout.handle(),
                    1,
                    std::slice::from_ref(&hs),
                    &[],
                );
            }
            crate::vulkan::record::cmd_push_constants(
                device,
                cmd,
                layout.handle(),
                vk::ShaderStageFlags::COMPUTE,
                &params,
            );
            device.cmd_dispatch(cmd, (self.cull_count() as u32).div_ceil(64), 1, 1);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    // Render the bindless static + instance + chunk prefix into a probe face (or a
    // planar mirror plane), limited to `area.render_area`. A thin sibling of `encode_main_pass`'s bindless branch:
    // begins the render pass (reusing `main_render_pass`, render-pass-compatible
    // with the bindless pipeline), binds the caller's face/plane global set (set 0)
    // + bindless set (set 1), and issues one indirect draw of
    // `[0, skinned_record_base())` from the given indirect buffer. The skinned tail
    // is omitted (V1). Shared by the probe bake + the planar reflection render.
    pub(in crate::vulkan) fn encode_main_into_face(
        &self,
        cmd: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        area: FaceArea,
        global_set: vk::DescriptorSet,
        bindless_set: vk::DescriptorSet,
        indirect: vk::Buffer,
    ) {
        let (Some(pipeline), Some(layout)) = (
            self.cull.bindless_pipeline.as_ref(),
            self.cull.bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let device = &self.hw.device;
        let [r, g, b, a] = self.state.view.clear_color;
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [r, g, b, a],
            },
        };
        let clear_depth = depth::clear_value(DepthConvention::Camera);
        let clears: &[vk::ClearValue] = if self.targets.msaa_samples != vk::SampleCountFlags::TYPE_1
        {
            &[clear_color, clear_depth, vk::ClearValue::default()]
        } else {
            &[clear_color, clear_depth]
        };
        let FaceArea {
            extent,
            render_area,
        } = area;
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.targets.main_render_pass.handle())
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(clears);
        // Negative-height viewport (Y flip), matching the main pass so the captured
        // faces share the cube convention `face_view_projection` was built against.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&render_area));
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                0,
                &[global_set, bindless_set],
                &[],
            );
            device.cmd_draw_indexed_indirect(
                cmd,
                indirect,
                0,
                self.skinned_record_base() as u32,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            device.cmd_end_render_pass(cmd);
        }
    }
}

impl ProbeBakeDevice for VkContext {
    type Capture = RenderingBake;
    type Prefilter = PrefilteringBake;
    type Frame<'f> = ();

    fn book(&mut self) -> &mut ProbeBook {
        &mut self.probe.book
    }

    // The capture renders through the bindless GPU cull, which never comes or
    // goes after init.
    fn capture_supported(&self) -> bool {
        self.cull.cull_kernels.is_some()
            && self.cull.bindless_pipeline.is_some()
            && self.probe.prefilter.is_some()
    }

    // Geometry may still be streaming: a zero cull would bake an empty cube.
    fn capture_ready(&self, _prefilter_in_flight: bool) -> bool {
        self.cull_count() > 0
    }

    fn reserve_cubes(&mut self, count: usize) -> RenderResult<()> {
        self.reserve_probe_cubes(&PLAN, count)
    }

    // Build the bake-owned capture resources (target + cull ring + per-face view
    // UBOs + both cubes) and fill the cull buffers + the six per-face view
    // uniforms ONCE (frustum-independent; each face re-runs only the cull with
    // its own frustum).
    fn start_capture(
        &mut self,
        _frame: &(),
        index: usize,
        placement: ProbePlacement,
    ) -> RenderResult<RenderingBake> {
        let eye = placement.position;
        let bake = BakeResources::new(self)?;

        // Bake-owned cull buffers, zeroed first so the untouched instance tail reads
        // as disabled (a probe omits instanced geometry), then filled with this
        // probe's static + chunk + skinned records (LOD by probe eye).
        let object_size = self.cull_count() * std::mem::size_of::<render_types::GpuObjectData>();
        let args_size = self.cull_count() * std::mem::size_of::<render_types::GpuDrawArgs>();
        bake.object_buf.zero_bytes(0, object_size);
        bake.draw_args_buf.zero_bytes(0, args_size);
        self.build_object_records_into(&bake.object_buf);
        self.build_draw_args_records_into(
            &bake.draw_args_buf,
            eye,
            concinnity_core::render::model_history::HistoryMode::Untracked,
        );

        // The capture cube each face copies into, and the probe cube the
        // convolution writes. Allocated with the capture rather than at the
        // convolution's start: face 0 copies into the cube, so it has to exist
        // before the first face records.
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or_else(|| RenderError::Other("probe: prefilter pipelines missing".into()))?;
        let cubes = self
            .probe
            .gpu
            .cubes
            .as_ref()
            .ok_or_else(|| RenderError::Other("probe: no cube array for a placement".into()))?;
        let prefilter = PrefilterGpu::new(
            &self.hw.device,
            &self.hw.alloc,
            pipelines,
            &PLAN,
            super::probe_prefilter::ProbeSlice { cubes, index },
        )?;

        let rendering = RenderingBake {
            eye,
            bake,
            prefilter,
            face_cmds: Vec::with_capacity(CAPTURE_FACES),
            face_fences: Vec::with_capacity(CAPTURE_FACES),
        };
        rendering.write_face_views(self.scene.prefilter_mip_count, self.state.view.sky_rot);
        Ok(rendering)
    }

    fn render_face(
        &mut self,
        _frame: &(),
        capture: &mut RenderingBake,
        face: usize,
    ) -> RenderResult<()> {
        self.record_probe_face(capture, face)
    }

    // The single graphics queue retires the faces in order, so the last face's
    // fence covers them all.
    fn capture_retired(&self, capture: &RenderingBake) -> bool {
        capture.face_fences.last().is_some_and(|&fence| {
            // SAFETY: the fence was created from this device; the query only reads.
            unsafe { self.hw.device.get_fence_status(fence) }.unwrap_or(false)
        })
    }

    // Free the capture's draw resources (the last face's fence signaled, so the
    // GPU is done with all of them); the two cubes carry on.
    fn begin_prefilter(
        &mut self,
        _index: usize,
        capture: RenderingBake,
    ) -> RenderResult<PrefilteringBake> {
        let RenderingBake {
            bake,
            prefilter,
            face_cmds,
            face_fences,
            ..
        } = capture;
        free_face_recordings(
            &self.hw.device,
            self.commands.command_pool,
            &face_cmds,
            &face_fences,
        );
        drop(bake);
        Ok(PrefilteringBake {
            gpu: prefilter,
            cmds: Vec::with_capacity(PLAN.mips() as usize),
            fences: Vec::with_capacity(PLAN.mips() as usize),
        })
    }

    // Mip 0 is the firefly-clamped mirror mip plus the capture's source pyramid;
    // each later mip one GGX convolution reading the finished pyramid and writing
    // a mip nothing else touches, so consecutive mips need no barrier. The last
    // mip also makes the cube's writes visible to the fragment reads, so the
    // install has nothing left to submit.
    fn prefilter_mip(&mut self, prefilter: &mut PrefilteringBake, mip: u32) -> RenderResult<()> {
        let (cmd, fence) = self.begin_prefilter_command(prefilter)?;
        if mip == 0 {
            self.encode_probe_pyramid(cmd, &prefilter.gpu, &PLAN)?;
        } else {
            self.encode_probe_ggx_mip(cmd, &prefilter.gpu, &PLAN, mip)?;
            if mip + 1 == PLAN.mips() {
                self.encode_probe_cube_readable(cmd, &prefilter.gpu);
            }
        }
        self.submit_prefilter_command(cmd, fence)
    }

    // The install frees each dispatch's command buffer and fence, so it waits for
    // the GPU to retire them, not just for them to be submitted.
    fn prefilter_retired(&self, prefilter: &PrefilteringBake) -> bool {
        prefilter.dispatches_retired(&self.hw.device)
    }

    // Nothing is uploaded at install -- the cube was written in place -- and no
    // descriptor moves: the frame's records upload carries the new count.
    fn finish_prefilter(&mut self, prefilter: PrefilteringBake) {
        prefilter.destroy(&self.hw.device, self.commands.command_pool);
    }

    // Idle the device before dropping either slot: their command buffers may
    // still be executing, and every payload owns images, views and descriptor
    // sets a submission could still name.
    fn abandon(&mut self, capture: Option<RenderingBake>, prefilter: Option<PrefilteringBake>) {
        self.wait_idle();
        let device = self.hw.device.clone();
        if let Some(rendering) = capture {
            rendering.destroy(&device, self.commands.command_pool);
        }
        if let Some(prefiltering) = prefilter {
            prefiltering.destroy(&device, self.commands.command_pool);
        }
    }
}

// Where an off-camera render draws in its framebuffer: the full `extent` the
// viewport maps the projection onto, and the `render_area` (inside it) that is
// cleared, drawn and stored. Texels outside the area keep whatever they held.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct FaceArea {
    pub(in crate::vulkan) extent: vk::Extent2D,
    pub(in crate::vulkan) render_area: vk::Rect2D,
}

impl FaceArea {
    // The whole framebuffer.
    pub(in crate::vulkan) fn whole(extent: vk::Extent2D) -> Self {
        Self {
            extent,
            render_area: vk::Rect2D::default().extent(extent),
        }
    }
}

// One in-flight probe's GPU capture state, held in the bake's capture slot while
// its six faces submit one per frame. Reuses one `BakeResources` (built in
// `start_capture`, freed in `begin_prefilter`) across the faces; the per-face
// command buffers + fences accumulate until the convolution starts, when the
// last face's fence retiring guarantees the GPU is done with all of them.
pub(crate) struct RenderingBake {
    eye: [f32; 3],
    bake: BakeResources,
    // The capture cube each face copies into, and the probe cube the convolution
    // will write. Allocated with the capture because face 0 copies into it, and
    // handed to the prefiltering slot once every face has landed.
    prefilter: PrefilterGpu,
    face_cmds: Vec<vk::CommandBuffer>,
    face_fences: Vec<vk::Fence>,
}

impl RenderingBake {
    // Write the six face view uniforms: each face's view from the probe eye, and
    // the sky the capture lights with. reflections_enabled stays 0: no resolve
    // runs over a probe face, so the bake captures the full forward probe
    // specular -- here the sky, since the bake binds an EMPTY ProbeSet.
    fn write_face_views(&self, prefilter_mip_count: u32, sky_rot: [[f32; 4]; 3]) {
        let eye = self.eye;
        for (face, buf) in self.bake.view_bufs.iter().enumerate() {
            let view = ViewUniforms {
                vp: reflection_probe::face_view_projection(eye, face),
                view: reflection_probe::face_view_matrix(eye, face),
                elapsed: 0.0,
                reflections_enabled: 0.0,
                cam_pos: eye,
                prefilter_mip_count: prefilter_mip_count as f32,
                // A probe capture is always lit, whatever the viewport shows.
                shade_mode: 0.0,
                ambient_occlusion: 0.0,
                sky_rot,
            };
            buf.write_val(0, &view);
        }
    }

    // Rewrite binding `binding` of every face's global set from what it is built
    // with now. The caller has idled the device.
    pub(super) fn rewrite_global_binding(
        &self,
        device: &VkDevice,
        bindings: &GlobalBindings<'_>,
        binding: u32,
    ) {
        for (face, &set) in self.bake.global_sets.iter().enumerate() {
            self.bake
                .global_contents(bindings, face)
                .write_binding(device, set, binding);
        }
    }

    // Re-point this bake's Hi-Z set at a rebuilt pyramid view. Called by
    // `rebuild_swapchain` after `hiz.resize_to` retired the view this set
    // captured at bake start; `wait_idle` gated the in-flight faces, and
    // hiz_enabled = 0 keeps the binding unsampled, but it must not dangle.
    // Mirrors the planar cull set's treatment.
    pub(super) fn rewrite_hiz_view(&self, device: &VkDevice, view: vk::ImageView) {
        if let Some(set) = self.bake.hiz_set {
            super::hiz::rewrite_read_set_view(device, set, view);
        }
    }

    // Free every owned GPU resource: the per-face command buffers (back to the
    // one-shot pool), the per-face fences, the bake target / cull / sets, and both
    // cubes. The caller has ensured the GPU retired them (the last face's fence is
    // signaled, or the device is idle).
    pub(super) fn destroy(self, device: &VkDevice, command_pool: vk::CommandPool) {
        free_face_recordings(device, command_pool, &self.face_cmds, &self.face_fences);
    }
}

// The prior probe whose capture is convolving into its cube on the GPU, one
// destination mip per frame. Holds both cubes plus the command buffer and fence of
// every dispatch it has submitted, which install frees once they retire.
pub(crate) struct PrefilteringBake {
    gpu: PrefilterGpu,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
}

impl PrefilteringBake {
    // Whether every convolution dispatch has retired. Only the last fence is
    // polled: one graphics queue retires the rest ahead of it.
    fn dispatches_retired(&self, device: &VkDevice) -> bool {
        match self.fences.last() {
            // SAFETY: the fence was created from this device; the query only reads.
            Some(&fence) => unsafe { device.get_fence_status(fence) }.unwrap_or(false),
            None => false,
        }
    }

    // Free the dispatch recordings and both cubes. The caller has idled the device.
    pub(super) fn destroy(self, device: &VkDevice, command_pool: vk::CommandPool) {
        free_face_recordings(device, command_pool, &self.cmds, &self.fences);
    }
}

// Return a bake step's command buffers to the one-shot pool and destroy its
// fences. The caller has proved the GPU retired them (a signaled fence, or an
// idle device).
fn free_face_recordings(
    device: &VkDevice,
    command_pool: vk::CommandPool,
    cmds: &[vk::CommandBuffer],
    fences: &[vk::Fence],
) {
    // SAFETY: every handle was created from this device and is destroyed exactly once; the caller
    // has already waited for the GPU to retire them, so no submission still references one.
    unsafe {
        if !cmds.is_empty() {
            device.free_command_buffers(command_pool, cmds);
        }
        for &fence in fences {
            device.destroy_fence(fence, None);
        }
    }
}

// The GPU resources for ONE reflection-probe capture: the 512x512 color/depth
// (/resolve) target + framebuffer, a bake-owned cull ring + its descriptor sets,
// and six per-face global sets carrying the face view + snapshot lighting. One
// per in-flight probe (held in `RenderingBake`); it drops when the capture hands
// its cube to the convolution.
struct BakeResources {
    color: GpuImage,
    // Held for the bake's lifetime; the framebuffer and sets alias them.
    _depth: GpuImage,
    resolve: Option<GpuImage>,
    framebuffer: OwnedFramebuffer,
    object_buf: PooledBuffer,
    draw_args_buf: PooledBuffer,
    indirect_buf: PooledBuffer,
    _status_buf: PooledBuffer,
    // The material parameter table as the bake started; every face's set binds it.
    _params_buf: PooledBuffer,
    _pool: OwnedDescriptorPool,
    cull_set: vk::DescriptorSet,
    // One texture-pool set per face, written from the live pool right before
    // that face records. A face's set is never touched after its submit, so a
    // streamed texture swap mid-bake needs no rewrite of pending sets (and no
    // device drain): the next face simply snapshots the current pool.
    bindless_sets: Vec<vk::DescriptorSet>,
    hiz_set: Option<vk::DescriptorSet>,
    _hiz_ubo: Option<PooledBuffer>,
    global_sets: Vec<vk::DescriptorSet>,
    view_bufs: Vec<PooledBuffer>,
    light: PooledBuffer,
    shadow: PooledBuffer,
}

impl BakeResources {
    // The image the capture-cube copy reads: the single-sample resolve when MSAA is on,
    // else the (single-sample) color attachment. Both rest in SHADER_READ_ONLY
    // after the render pass.
    fn copy_source(&self) -> vk::Image {
        match &self.resolve {
            Some(r) => r.image,
            None => self.color.image,
        }
    }

    fn new(ctx: &VkContext) -> RenderResult<BakeResources> {
        use concinnity_core::gfx::render_types::{GpuDrawArgs, GpuObjectData};
        let device = &ctx.hw.device;
        let alloc = &ctx.hw.alloc;
        let msaa = ctx.targets.msaa_samples != vk::SampleCountFlags::TYPE_1;
        let size = PROBE_FACE_SIZE;

        // Color + depth (+ single-sample resolve when MSAA), then a framebuffer
        // compatible with `main_render_pass`.
        let color_pooled = create_image(
            alloc,
            &ImageSpec {
                width: size,
                height: size,
                format: HDR_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: ctx.targets.msaa_samples,
            },
        )?;
        let color_view = create_image_view(
            device,
            color_pooled.image(),
            HDR_FORMAT,
            vk::ImageAspectFlags::COLOR,
        )?;
        let color = GpuImage::from_pooled(color_pooled, color_view);
        let depth_pooled = create_image(
            alloc,
            &ImageSpec {
                width: size,
                height: size,
                format: PROBE_DEPTH_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: ctx.targets.msaa_samples,
            },
        )?;
        let depth_view = create_image_view(
            device,
            depth_pooled.image(),
            PROBE_DEPTH_FORMAT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let depth = GpuImage::from_pooled(depth_pooled, depth_view);
        let resolve = if msaa {
            let resolve_pooled = create_image(
                alloc,
                &ImageSpec {
                    width: size,
                    height: size,
                    format: HDR_FORMAT,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::SAMPLED,
                    mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
            )?;
            let view = create_image_view(
                device,
                resolve_pooled.image(),
                HDR_FORMAT,
                vk::ImageAspectFlags::COLOR,
            )?;
            Some(GpuImage::from_pooled(resolve_pooled, view))
        } else {
            None
        };
        let fb_attachments: Vec<vk::ImageView> = if msaa {
            vec![
                color.view,
                depth.view,
                resolve
                    .as_ref()
                    .expect("a multisampled probe target has a resolve image")
                    .view,
            ]
        } else {
            vec![color.view, depth.view]
        };
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(ctx.targets.main_render_pass.handle())
            .attachments(&fb_attachments)
            .width(size)
            .height(size)
            .layers(1);
        let framebuffer = device
            .create_framebuffer(&fb_info)
            .map_err(|e| super::error::map_vk_result(e, "probe framebuffer"))?;

        // Bake-owned cull ring, sized like the per-frame rings.
        let n = ctx.cull_count();
        let object_size = (n * std::mem::size_of::<GpuObjectData>()) as u64;
        let args_size = (n * std::mem::size_of::<GpuDrawArgs>()) as u64;
        let indirect_size = (n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()) as u64;
        let status_size = (n * std::mem::size_of::<u32>()) as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let object_buf =
            alloc.create_buffer(object_size, vk::BufferUsageFlags::STORAGE_BUFFER, host)?;
        let draw_args_buf =
            alloc.create_buffer(args_size, vk::BufferUsageFlags::STORAGE_BUFFER, host)?;
        let indirect_buf = alloc.create_buffer(
            indirect_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let status_buf = alloc.create_buffer(
            status_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Snapshot lighting (so all faces share one set) and six per-face view UBOs.
        let light = make_ubo_bytes(alloc, bytemuck::bytes_of(&ctx.uniforms.light_uniforms))?;
        let shadow = make_ubo_bytes(alloc, bytemuck::bytes_of(&ctx.shadow.uniforms))?;
        let view_size = std::mem::size_of::<ViewUniforms>() as u64;
        let mut view_bufs = Vec::with_capacity(CAPTURE_FACES);
        for _ in 0..CAPTURE_FACES {
            view_bufs.push(alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                host,
            )?);
        }

        // One dedicated descriptor pool for the bake's cull + per-face bindless +
        // global + Hi-Z sets.
        // The pool binding's declared length, not the world's image count: the
        // bake allocates the same bindless set layout the main pass does, so it
        // has to budget for every slot that layout declares.
        let tex_pool = ctx.cull.bindless_pool_size as u32;
        let has_hiz = u32::from(ctx.cull.hiz.is_some());
        let faces = CAPTURE_FACES as u32;
        // The six per-face global sets, the four cull SSBOs, the object SSBO,
        // texture pool and parameter table of each face's bindless set, and a
        // Hi-Z set (an image and a UBO) when the world runs Hi-Z.
        let pool_sizes = PoolSizes::default()
            .sets(&global_set(), faces)
            .add(vk::DescriptorType::STORAGE_BUFFER, 4 + 2 * faces)
            .add(
                vk::DescriptorType::SAMPLED_IMAGE,
                faces * tex_pool + has_hiz,
            )
            .add(vk::DescriptorType::UNIFORM_BUFFER, has_hiz)
            .build();
        let max_sets = 1 + 2 * faces + has_hiz;
        // The per-face bindless sets below come from `cull.bindless_set_layout`,
        // so this pool has to declare update-after-bind whenever that layout
        // does.
        let mut pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);
        if ctx.cull.bindless_update_after_bind {
            pool_info = pool_info.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        }
        let pool = device
            .create_descriptor_pool(&pool_info)
            .map_err(|e| super::error::map_vk_result(e, "probe descriptor pool"))?;

        // Cull set (set 0): object / draw-args / indirect / status SSBOs.
        let cull_kernels =
            ctx.cull.cull_kernels.as_ref().ok_or_else(|| {
                RenderError::Other("probe: the GPU cull is not initialized".into())
            })?;
        let cull_set = alloc_descriptor_sets(
            device,
            pool.handle(),
            std::slice::from_ref(&cull_kernels.set_layout.handle()),
        )?[0];
        SetWrites::new(cull_set)
            .storage_buffer(0, object_buf.buffer(), object_size)
            .storage_buffer(1, draw_args_buf.buffer(), args_size)
            .storage_buffer(2, indirect_buf.buffer(), indirect_size)
            .storage_buffer(3, status_buf.buffer(), status_size)
            .apply(device);

        // The material parameter table as the bake starts, for every face.
        let params_buf = ctx
            .cull
            .material_params
            .as_ref()
            .ok_or_else(|| RenderError::Other("probe: no material parameter table".into()))?
            .snapshot(alloc)?;

        // Per-face bindless sets (set 1): object SSBO + the shared texture pool
        // array + the parameter table. Only the SSBOs are written here; each face's pool array is
        // written from the live pool right before that face records
        // (`write_face_pool`), so a mid-bake streamed swap needs no rewrite of
        // a pending set.
        let bindless_layouts = vec![
            ctx.cull
                .bindless_set_layout
                .as_ref()
                .expect("bindless descriptor set layout exists once culling is initialized")
                .handle();
            CAPTURE_FACES
        ];
        let bindless_sets = alloc_descriptor_sets(device, pool.handle(), &bindless_layouts)?;
        for &set in &bindless_sets {
            SetWrites::new(set)
                .storage_buffer(0, object_buf.buffer(), object_size)
                .storage_buffer(MATERIAL_PARAMS_BINDING, params_buf.buffer(), vk::WHOLE_SIZE)
                .apply(device);
        }

        // Bake Hi-Z set (cull set 1), hiz_enabled = 0.
        let (hiz_set, hiz_ubo) = match ctx.cull.hiz.as_ref() {
            Some(hiz) => {
                let (set, ubo) = super::hiz::off_camera_read_set(
                    alloc,
                    device,
                    pool.handle(),
                    hiz.read_set_layout.handle(),
                    hiz.read_set_view(),
                )?;
                (Some(set), Some(ubo))
            }
            None => (None, None),
        };

        // Six per-face global sets (set 0 of the bindless main pass): the face view
        // and the snapshot lighting, reading no probe so a face reflects only the
        // sky.
        let layouts = vec![ctx.descriptors.global_set_layout.handle(); CAPTURE_FACES];
        let global_sets = alloc_descriptor_sets(device, pool.handle(), &layouts)?;

        let bake = BakeResources {
            color,
            _depth: depth,
            resolve,
            framebuffer,
            object_buf,
            draw_args_buf,
            indirect_buf,
            _status_buf: status_buf,
            _params_buf: params_buf,
            _pool: pool,
            cull_set,
            bindless_sets,
            hiz_set,
            _hiz_ubo: hiz_ubo,
            global_sets,
            view_bufs,
            light,
            shadow,
        };
        let bindings = ctx.global_bindings();
        for (face, &set) in bake.global_sets.iter().enumerate() {
            bake.global_contents(&bindings, face).write(device, set);
        }
        Ok(bake)
    }

    // What face `face`'s global set holds: its view, and the lighting snapshot
    // every face shares.
    fn global_contents(&self, bindings: &GlobalBindings<'_>, face: usize) -> GlobalSetContents {
        bindings.off_camera(
            self.view_bufs[face].buffer(),
            self.light.buffer(),
            self.shadow.buffer(),
        )
    }
}

// Create a HOST_VISIBLE uniform buffer holding `bytes`, persistently mapped.
fn make_ubo_bytes(
    alloc: &super::allocator::DeviceAllocator,
    bytes: &[u8],
) -> RenderResult<PooledBuffer> {
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let buf = alloc.create_buffer(
        bytes.len() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        host,
    )?;
    buf.write_bytes(0, bytes);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A capture routes every record into region 0. Getting this wrong is
    // invisible in a screenshot of most worlds -- it only drops the records whose
    // material carries a world shader -- so it is pinned rather than eyeballed.
    #[test]
    fn a_capture_cull_routes_every_record_into_one_region() {
        let p = capture_cull_params(&Frustum::from_camera(IDENTITY), [0.0; 3], 12);
        assert_eq!(p.bucket_count, 1, "one region, whatever the world declares");
        assert_eq!(p.object_count, 12);
        assert_eq!(p.bucket_stride, 12, "stride names the region capacity");
    }

    // Every byte the shader reads must be written. `cmd_push_constants` takes a
    // slice, so a short one leaves the tail undefined -- and push constants do not
    // carry across command buffers, so the capture cull (which runs on a later
    // pass's buffer than the main cull) reads whatever the driver left there.
    // That is how the mirror render lost its draws.
    #[test]
    fn the_capture_push_covers_the_whole_shader_block() {
        let p = capture_cull_params(&Frustum::from_camera(IDENTITY), [1.0, 2.0, 3.0], 4);
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 120, "cull.hlsl's push_constant block is 120 B");
        // The two routing fields live in the last 8 bytes: the exact span a
        // 112-byte push left undefined.
        assert_eq!(
            &bytes[112..116],
            &1u32.to_le_bytes(),
            "bucket_count written"
        );
        assert_eq!(
            &bytes[116..120],
            &4u32.to_le_bytes(),
            "bucket_stride written"
        );
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
}
