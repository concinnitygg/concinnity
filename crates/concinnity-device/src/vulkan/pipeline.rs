// Vulkan pipeline creation for the main, shadow, and text render passes, over
// the single-source programs in `super::slang_builtins` and a world Shader's
// cooked artifacts.

use ash::vk;

use crate::gfx::shadow_bias;
use crate::vulkan::owned::{OwnedPipeline, VkDevice};

use super::builtins;
use crate::vulkan::slang_builtins::SlangCompile;

// The uniform and push-constant layouts are the `.slang` sources' own, held
// to the `#[repr(C)]` mirrors by `crate::shader_layout`.

//  Shader compilation

#[cfg(test)]
pub(super) fn is_spirv(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == 0x07230203
}

// Compile the engine's bindless static-pass pair. `pool_size` is the bindless
// texture-pool length, injected into the fragment source's `tex_pool[]` array
// declaration; `probe_cube_count` is the global set layout's binding-8
// descriptor count, injected into the probe cube array. A bucket whose Shader
// is the world's compiles the same file through `world_entry` instead.
pub(super) fn compile_bindless_shaders(
    hot_reload: bool,
    pool_size: usize,
    probe_cube_count: u32,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = builtins::Ctx {
        hot_reload,
        msaa: false,
        pool_size,
        probe_count: probe_cube_count as usize,
    };
    let vert = super::slang_builtins::MAIN_BINDLESS_VERT.compile(&ctx)?;
    let frag = super::slang_builtins::MAIN_BINDLESS_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

// Compute cull compute kernel. One invocation per build-time `DrawObject`
// frustum/distance-tests the object's `GpuObjectData` AABB against the six
// CPU-extracted frustum planes and writes one `VkDrawIndexedIndirectCommand`
// into the per-frame indirect buffer: survivors get `instance_count = 1`,
// culled or disabled objects get `instance_count = 0` (a no-op draw). The main
// bindless pass then issues the whole buffer with a single
// `cmd_draw_indexed_indirect`, so the CPU never walks the static draw list.
//
// The frustum and distance maths mirror `gfx::frustum` exactly (the six
// planes are extracted CPU-side already normalised) so the GPU path culls
// identically to the CPU BVH path it replaces. `GpuObjectData` / `GpuDrawArgs`
// mirror `gfx::render_types` under std430; the command struct mirrors
// `VkDrawIndexedIndirectCommand`. The object id rides `first_instance` (the
// bindless vertex shader reads it as `gl_InstanceIndex`).

// Byte size of the cull kernel's `CullParams` push-constant block: six
// `vec4` planes (96) + `vec3 cam_pos` + `uint object_count` (the trailing
// scalar shares the camera position's 16-byte std430 slot) + the shader-bucket
// routing pair (8). Within the 128-byte minimum guaranteed push-constant range.
pub(super) const CULL_PUSH_CONSTANT_BYTES: u32 = 120;

// Compile the Compute cull compute kernel to SPIR-V.
pub(super) fn compile_cull_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    super::slang_builtins::CULL.compile(&builtins::Ctx::plain(hot_reload))
}

// Compile the phase-2 (two-pass occlusion) variant of the cull kernel. Same
// source as `compile_cull_shader`, with a `CULL_PHASE2` define selecting the
// re-test of phase 1's Hi-Z-occluded objects against the rebuilt pyramid.
// Mirrors the `#define` split the Hi-Z init kernel uses.
pub(super) fn compile_cull_shader_phase2(hot_reload: bool) -> Result<Vec<u8>, String> {
    super::slang_builtins::CULL_PHASE2.compile(&builtins::Ctx::plain(hot_reload))
}

// Compile the GPU-driven shadow cull kernel: the same cull source with a
// `SHADOW_CULL` define, which drops the Hi-Z (set 1) + status (binding 3)
// bindings and tests each cascade's light frustum only. Paired with the lean
// 3-SSBO shadow cull set layout.
pub(super) fn compile_shadow_cull_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    super::slang_builtins::CULL_SHADOW.compile(&builtins::Ctx::plain(hot_reload))
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader.
pub(super) fn compile_shadow_bindless_vs(hot_reload: bool) -> Result<Vec<u8>, String> {
    super::slang_builtins::SHADOW_BINDLESS_VERT.compile(&builtins::Ctx::plain(hot_reload))
}

// Create the GPU-cull compute pipeline. `layout` must include the cull
// descriptor set (set 0: object SSBO, draw-args SSBO, indirect-command SSBO)
// and the `CullParams` push-constant range.
pub(super) fn create_cull_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let module = spv_module(device, spv)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.handle())
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = crate::vulkan::pipeline_cache::create_compute_pipeline(device, &info)
        .map_err(|e| format!("create cull pipeline: {e}"))?;
    Ok(pipeline)
}

// A shader module scoped to pipeline creation: destroyed on drop, so the
// early-return error paths between module and pipeline creation cannot leak it.
pub(in crate::vulkan) struct SpvModule<'d> {
    device: &'d VkDevice,
    module: vk::ShaderModule,
}

impl SpvModule<'_> {
    pub(in crate::vulkan) fn handle(&self) -> vk::ShaderModule {
        self.module
    }
}

impl Drop for SpvModule<'_> {
    fn drop(&mut self) {
        // SAFETY: the module was created from this device and is destroyed exactly once here. A
        // module may be destroyed as soon as the pipelines that consumed it exist, and a module
        // dropped on an error path has no consumers at all.
        unsafe { self.device.destroy_shader_module(self.module, None) };
    }
}

// SPIR-V is a stream of 32-bit words and ash requires it 4-byte aligned, so
// copy the bytes into an aligned `Vec<u32>`. A length that is not a whole
// number of words means a truncated or corrupt module, so reject it here
// rather than rounding it down.
fn spirv_words(spv: &[u8]) -> Result<Vec<u32>, String> {
    if !spv.len().is_multiple_of(4) {
        return Err(format!(
            "SPIR-V length {} is not a whole number of words",
            spv.len()
        ));
    }
    Ok(spv
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes([w[0], w[1], w[2], w[3]]))
        .collect())
}

pub(in crate::vulkan) fn spv_module<'d>(
    device: &'d VkDevice,
    spv: &[u8],
) -> Result<SpvModule<'d>, String> {
    let code = spirv_words(spv).map_err(|e| format!("shader module: {e}"))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let module = unsafe { device.create_shader_module(&info, None) }
        .map_err(|e| format!("shader module: {e}"))?;
    Ok(SpvModule { device, module })
}

// The world Shader's program for `entry`, as SPIR-V: the cook's artifact when
// the engine template still matches, else a compile here. `pool_size` and
// `probe_count` are the bindless pool and probe cube array lengths the host
// declares.
pub(super) fn world_entry(
    world: &concinnity_core::components::ShaderPrograms,
    entry: &str,
    hot_reload: bool,
    pool_size: usize,
    probe_count: usize,
) -> Result<Vec<u8>, String> {
    let req = crate::surface_source::Request {
        platform: concinnity_core::platform::Platform::Glsl,
        pool_size,
        probe_count,
        hot_reload,
    };
    crate::surface_source::artifact(world, entry, &req).map(|c| c.into_owned())
}

// The depth-only skinned shadow vertex, the engine's own: skinned main-pass
// draws ride the GPU-driven pass through the skin fold.
pub(super) fn compile_skinned_shadow_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    super::slang_builtins::SKINNED_SHADOW_VERT.compile(&builtins::Ctx::plain(hot_reload))
}

// The shadow vertex shader is engine-internal. Whether the shadow pass runs at
// all is gated by `effective_shadow_size` at the call site, not here.
pub(super) fn resolve_shadow_shader(hot_reload: bool) -> Result<Option<Vec<u8>>, String> {
    let spv = super::slang_builtins::SHADOW_VERT.compile(&builtins::Ctx::plain(hot_reload))?;
    Ok(Some(spv))
}

pub(super) fn compile_text_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = builtins::Ctx::plain(hot_reload);
    let vert = super::slang_builtins::TEXT_VERT.compile(&ctx)?;
    let frag = super::slang_builtins::TEXT_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

pub(super) fn compile_composite_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = builtins::Ctx::plain(hot_reload);
    let vert = super::slang_builtins::FULLSCREEN_VERT.compile(&ctx)?;
    let frag = super::slang_builtins::COMPOSITE_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

//  Pipeline creation

// Vertex binding and attribute descriptions for the full Vertex struct (56 bytes).
fn main_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 5],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(56)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(24),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(36),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(4)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(48),
    ];
    ([binding], attrs)
}

// Reduced vertex input for the depth-only skinned shadow pipeline: only the
// position + joint indices + blend weights the skinned shadow VS consumes
// (binding stride stays 80, the same SkinnedVertex buffer is bound).
fn skinned_shadow_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 3],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(80)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(5)
            .format(vk::Format::R16G16B16A16_UINT)
            .offset(56),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(6)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(64),
    ];
    ([binding], attrs)
}

// TextVertex binding (32 bytes): pos(vec2) + uv(vec2) + color(vec3) + mode(float).
fn text_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 4],
) {
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(16),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32_SFLOAT)
            .offset(28),
    ];
    ([binding], attrs)
}

// Render pass, pipeline layout, and the vertex + fragment SPIR-V a mesh
// pipeline (main / instanced / skinned) is built against. Borrows the shader
// byte slices for the duration of the build.
pub(super) struct MeshPipelineTargets<'a> {
    pub render_pass: vk::RenderPass,
    pub layout: vk::PipelineLayout,
    pub vert_spv: &'a [u8],
    pub frag_spv: &'a [u8],
}

// The main-pass targets a material-referenced world shader's bucket pipeline is
// built against. Every bucket shares the bindless pipeline layout and render
// pass; only the stage SPIR-V differs.
#[derive(Copy, Clone)]
pub(super) struct BucketPipelineTargets {
    pub render_pass: vk::RenderPass,
    pub layout: vk::PipelineLayout,
    pub msaa_samples: vk::SampleCountFlags,
    pub swapchain_format: vk::Format,
    pub hot_reload: bool,
    // The pool and probe cube counts the bindless set layout declares, which a
    // world's bindless pair compiles against.
    pub pool_size: usize,
    pub probe_count: usize,
}

// Build one shader bucket's bindless main-pass pipeline. `bucket` is the
// `DrawObject::shader_bucket` value (1-based; bucket 0 is the world default
// program) and names the bucket in error messages.
//
// A bucket with no programs is one the world declared no Shader for, so the
// engine's own bindless pair renders it.
pub(super) fn build_bucket_pipeline(
    device: &VkDevice,
    targets: BucketPipelineTargets,
    bucket: usize,
    shader: crate::gfx::backend_init::WorldShader<'_>,
    engine_default: &(Vec<u8>, Vec<u8>),
) -> Result<OwnedPipeline, String> {
    let (vert_spv, frag_spv) = match shader.programs {
        Some(programs) => (
            world_entry(
                programs,
                "vertex_main_bindless",
                targets.hot_reload,
                targets.pool_size,
                targets.probe_count,
            )?,
            world_entry(
                programs,
                "fragment_main_bindless",
                targets.hot_reload,
                targets.pool_size,
                targets.probe_count,
            )?,
        ),
        None => (engine_default.0.clone(), engine_default.1.clone()),
    };
    if vert_spv.is_empty() || frag_spv.is_empty() {
        return Err(format!("shader bucket {bucket} carries no SPIR-V stages"));
    }
    create_main_pipeline(
        device,
        MeshPipelineTargets {
            render_pass: targets.render_pass,
            layout: targets.layout,
            vert_spv: &vert_spv,
            frag_spv: &frag_spv,
        },
        targets.msaa_samples,
        targets.swapchain_format,
    )
    .map_err(|e| format!("shader bucket {bucket}: {e}"))
}

// Build the per-bucket pipeline table from the world's material-referenced
// shaders. Index `b` holds bucket `b + 1`'s pipeline; `None` marks a bucket the
// streaming pump installs later (its Shader is owned by a scene that has not
// pinned, so `decode_shaders` deferred its payload).
pub(super) fn build_world_pipeline_table(
    device: &VkDevice,
    targets: BucketPipelineTargets,
    bucket_shaders: &[crate::gfx::backend_init::WorldShader<'_>],
    engine_default: &(Vec<u8>, Vec<u8>),
) -> Result<Vec<Option<OwnedPipeline>>, String> {
    let mut table = Vec::with_capacity(bucket_shaders.len());
    for (i, shader) in bucket_shaders.iter().enumerate() {
        if shader.deferred {
            table.push(None);
            continue;
        }
        table.push(Some(build_bucket_pipeline(
            device,
            targets,
            i + 1,
            *shader,
            engine_default,
        )?));
    }
    Ok(table)
}

pub(super) fn create_main_pipeline(
    device: &VkDevice,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    surface_format: vk::Format,
) -> Result<OwnedPipeline, String> {
    create_main_pipeline_filled(device, targets, msaa, surface_format, vk::PolygonMode::FILL)
}

// The Wireframe view mode's variant of `create_main_pipeline`. Vulkan polygon
// mode is pipeline state without `VK_EXT_extended_dynamic_state3`, so the mode
// needs its own pipeline per main-pass path; see [`super::wireframe`]. Requires
// the `fillModeNonSolid` device feature.
pub(super) fn create_main_pipeline_wireframe(
    device: &VkDevice,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    surface_format: vk::Format,
) -> Result<OwnedPipeline, String> {
    create_main_pipeline_filled(device, targets, msaa, surface_format, vk::PolygonMode::LINE)
}

fn create_main_pipeline_filled(
    device: &VkDevice,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    _surface_format: vk::Format,
    polygon_mode: vk::PolygonMode,
) -> Result<OwnedPipeline, String> {
    let MeshPipelineTargets {
        render_pass,
        layout,
        vert_spv,
        frag_spv,
    } = targets;
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod.handle())
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod.handle())
            .name(&entry),
    ];

    let (bindings, attrs) = main_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(polygon_mode)
        .line_width(1.0)
        // Match Metal's default + DirectX (no back-face culling) so meshes
        // with mixed winding (particularly procedural floor / ceiling planes
        // whose triangles have a -Y normal under the unsigned plane order)
        // render from both sides. Vulkan's pipeline-default was BACK, which
        // hid the showcase floor while leaving every solid mesh visible.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let color_blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&color_blend_attach));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &pipeline_info)
        .map_err(|e| format!("create main pipeline: {e}"))?;

    Ok(pipeline)
}

pub(super) fn create_shadow_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_mod.handle())
        .name(&entry)];

    // `shadow.vert` only reads position (it writes depth-only NDC), so the
    // optimizer strips the other attributes from its interface. Bind just
    // location 0 so the pipeline matches the shader and the validation layer
    // does not warn about unconsumed attributes. The binding keeps the full
    // 56-byte `Vertex` stride; the omitted attributes are simply not fetched.
    let (bindings, attrs) = main_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs[..1]);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        // Match Metal's default + DirectX (no back-face culling) so meshes
        // with mixed winding (particularly procedural floor / ceiling planes
        // whose triangles have a -Y normal under the unsigned plane order)
        // render from both sides. Vulkan's pipeline-default was BACK, which
        // hid the showcase floor while leaving every solid mesh visible.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(true)
        .depth_bias_constant_factor(shadow_bias::RASTER_CONSTANT)
        .depth_bias_slope_factor(shadow_bias::RASTER_SLOPE)
        // The clamp needs the optional depthBiasClamp feature; `bias_clamp` is
        // 0.0 (unclamped) on a device without it.
        .depth_bias_clamp(device.depth_bias_clamp());

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &pipeline_info)
        .map_err(|e| format!("create shadow pipeline: {e}"))?;

    Ok(pipeline)
}

// Shadow-pass pipeline for skinned geometry: the skinned shadow vertex shader
// (80-byte layout, depth-only).
pub(super) fn create_skinned_shadow_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_mod.handle())
        .name(&entry)];

    let (bindings, attrs) = skinned_shadow_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        // Match Metal's default + DirectX (no back-face culling) so meshes
        // with mixed winding (particularly procedural floor / ceiling planes
        // whose triangles have a -Y normal under the unsigned plane order)
        // render from both sides. Vulkan's pipeline-default was BACK, which
        // hid the showcase floor while leaving every solid mesh visible.
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(true)
        .depth_bias_constant_factor(shadow_bias::RASTER_CONSTANT)
        .depth_bias_slope_factor(shadow_bias::RASTER_SLOPE)
        .depth_bias_clamp(device.depth_bias_clamp());

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &pipeline_info)
        .map_err(|e| format!("create skinned shadow pipeline: {e}"))?;

    Ok(pipeline)
}

pub(super) fn create_text_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    msaa: vk::SampleCountFlags,
) -> Result<OwnedPipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod.handle())
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod.handle())
            .name(&entry),
    ];

    let (bindings, attrs) = text_vertex_input();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(msaa);

    // No depth test for text overlay; always draws on top.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);

    // Standard over-compositing alpha blend.
    let blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&blend_attach));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &pipeline_info)
        .map_err(|e| format!("create text pipeline: {e}"))?;

    Ok(pipeline)
}

// Build the composite (post-process) pipeline: a vertex-buffer-less fullscreen
// triangle that samples the resolved HDR target and applies ACES + gamma +
// FXAA. Targets the single-sample swapchain backbuffer; no depth attachment.
pub(super) fn create_composite_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod.handle())
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod.handle())
            .name(&entry),
    ];

    // No vertex input: the fullscreen triangle is generated from gl_VertexIndex.
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    // The composite pass always renders to the single-sample swapchain image.
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);

    let color_blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&color_blend_attach));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &pipeline_info)
        .map_err(|e| format!("create composite pipeline: {e}"))?;

    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::{
        SlangCompile, builtins, compile_bindless_shaders, compile_cull_shader,
        compile_cull_shader_phase2, compile_shadow_bindless_vs, compile_shadow_cull_shader,
        compile_skinned_shadow_shader, is_spirv, spirv_words, world_entry,
    };

    // Whole words become native-endian u32s, matching the raw reinterpretation
    // the driver does of the byte stream.
    #[test]
    fn spirv_words_reads_whole_words() {
        let bytes = [0x03, 0x02, 0x23, 0x07, 0x00, 0x01, 0x00, 0x00];
        let words = spirv_words(&bytes).expect("a two-word blob converts");
        assert_eq!(
            words,
            vec![
                u32::from_ne_bytes([0x03, 0x02, 0x23, 0x07]),
                u32::from_ne_bytes([0x00, 0x01, 0x00, 0x00]),
            ]
        );
        assert_eq!(spirv_words(&[]).expect("empty converts"), Vec::<u32>::new());
    }

    // A trailing partial word is a truncated module. It used to be copied past
    // the end of the destination allocation; it must be rejected instead.
    #[test]
    fn spirv_words_rejects_a_partial_word() {
        for len in [1usize, 2, 3, 5, 7] {
            let bytes = vec![0xFFu8; len];
            assert!(
                spirv_words(&bytes).is_err(),
                "length {len} is not a whole number of words"
            );
        }
    }

    // The phase-1 cull kernel, its two-pass `CULL_PHASE2` variant, and the
    // GPU-driven shadow `SHADOW_CULL` variant all compile to valid SPIR-V from
    // the embedded source. Guards the `#ifdef` split in `cull.slang`, which the
    // Vulkan-on-Windows runtime cannot currently exercise.
    #[test]
    fn cull_shaders_compile_both_phases() {
        let phase1 = compile_cull_shader(false).expect("phase-1 cull compiles");
        let phase2 = compile_cull_shader_phase2(false).expect("phase-2 cull compiles");
        let shadow = compile_shadow_cull_shader(false).expect("shadow cull compiles");
        assert!(is_spirv(&phase1), "phase-1 cull is valid SPIR-V");
        assert!(is_spirv(&phase2), "phase-2 cull is valid SPIR-V");
        assert!(is_spirv(&shadow), "shadow cull is valid SPIR-V");
        // Each define selects a different kernel body, so the modules differ.
        assert_ne!(phase1, phase2);
        assert_ne!(phase1, shadow);
    }

    // The GPU-driven shadow pass's depth-only bindless vertex shader compiles to
    // valid SPIR-V from the embedded source.
    #[test]
    fn shadow_bindless_vs_compiles() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        let vs = compile_shadow_bindless_vs(false).expect("shadow bindless VS compiles");
        assert!(is_spirv(&vs), "shadow bindless VS is valid SPIR-V");
    }

    // The bindless main shaders compile to valid SPIR-V from the embedded
    // single-source program, across the probe-array lengths a device may bind
    // (the array length is a runtime value on a sampler-starved driver, so the
    // shortest and the ceiling forms both have to survive).
    #[test]
    fn bindless_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        for probes in [1, 7, concinnity_core::render::uniforms::MAX_PROBES as u32] {
            let (vs, fs) =
                compile_bindless_shaders(false, 4, probes).expect("bindless shaders compile");
            assert!(is_spirv(&vs), "bindless vertex is valid SPIR-V");
            assert!(is_spirv(&fs), "bindless fragment is valid SPIR-V");
        }
        // The runtime pool / probe counts ride the assembled source as
        // `#define` lines, which is what the shader cache keys.
        let frag_src = crate::vulkan::slang_builtins::MAIN_BINDLESS_FRAG.source(&builtins::Ctx {
            hot_reload: false,
            msaa: false,
            pool_size: 4,
            probe_count: 4,
        });
        assert!(frag_src.contains("#define POOL_SIZE 4"));
        assert!(frag_src.contains("#define MAX_PROBES 4"));
    }

    // A world Shader's bindless pair compiles from its programs, and is its own
    // program rather than the engine's. No device is needed, so this guards the
    // world-shader path the Vulkan-on-Windows runtime cannot unit-test end to
    // end. The payload carries no cooked artifacts, so both entries take the
    // compile branch of `surface_source`, which is also what a stale cook does.
    #[test]
    fn a_world_shader_compiles_its_own_bindless_pair() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        let programs = concinnity_core::components::ShaderPrograms {
            name: "wall".to_string(),
            vertex: None,
            fragment: "float4 shade(VertexOut in, GpuObjectData od) { return float4(1.0); }"
                .to_string(),
            programs: Vec::new(),
        };
        let pool = 4;
        let probes = concinnity_core::render::uniforms::MAX_PROBES;
        let vs = world_entry(&programs, "vertex_main_bindless", false, pool, probes).unwrap();
        let fs = world_entry(&programs, "fragment_main_bindless", false, pool, probes).unwrap();
        assert!(is_spirv(&vs) && is_spirv(&fs), "the world's pair compiles");
        let (_, engine_fs) = compile_bindless_shaders(false, pool, probes as u32).unwrap();
        assert_ne!(fs, engine_fs, "the world's fragment is its own program");
        // The depth-only skinned shadow vertex stays the engine's.
        assert!(is_spirv(&compile_skinned_shadow_shader(false).unwrap()));
    }
}
