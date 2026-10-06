// Vulkan pipeline creation for the main, shadow, and text render passes, over
// the single-source programs in `super::builtin_shaders` and a world Shader's
// cooked artifacts.

use ash::vk;
use concinnity_core::render::backend_init;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shadow_bias;

use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{OwnedPipeline, VkDevice};
use crate::vulkan::pipeline_desc::{Blend, Depth, DepthBias, GraphicsPipelineDesc, Raster};

// The uniform and push-constant layouts are the `.hlsl` sources' own, held
// to the `#[repr(C)]` mirrors by `crate::shader_layout`.

#[cfg(test)]
pub(super) fn is_spirv(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == 0x07230203
}

// Compile the engine's bindless static-pass pair. No count reaches it: the
// fragment declares `tex_pool[]` unsized and reads whatever the set layout
// holds, and the probe set is a single cube array. A bucket whose Shader is the
// world's compiles the same file through `world_entry` instead.
pub(super) fn compile_bindless_shaders(hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::MAIN_BINDLESS_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::MAIN_BINDLESS_FRAG.compile(hot_reload)?;
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
// planes are extracted CPU-side already normalized) so the GPU path culls
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
pub(super) fn compile_cull_shader(hot_reload: bool) -> RenderResult<Vec<u8>> {
    super::builtin_shaders::CULL_PHASE1.compile(hot_reload)
}

// Compile the phase-2 (two-pass occlusion) variant of the cull kernel. Same
// source as `compile_cull_shader`, with a `CULL_PHASE2` define selecting the
// re-test of phase 1's Hi-Z-occluded objects against the rebuilt pyramid.
// Mirrors the `#define` split the Hi-Z init kernel uses.
pub(super) fn compile_cull_shader_phase2(hot_reload: bool) -> RenderResult<Vec<u8>> {
    super::builtin_shaders::CULL_PHASE2.compile(hot_reload)
}

// Compile the GPU-driven shadow cull kernel: the same cull source with a
// `SHADOW_CULL` define, which drops the Hi-Z (set 1) + status (binding 3)
// bindings and tests each cascade's light frustum only. Paired with the lean
// 3-SSBO shadow cull set layout.
pub(super) fn compile_shadow_cull_shader(hot_reload: bool) -> RenderResult<Vec<u8>> {
    super::builtin_shaders::CULL_SHADOW.compile(hot_reload)
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader.
pub(super) fn compile_shadow_bindless_vs(hot_reload: bool) -> RenderResult<Vec<u8>> {
    super::builtin_shaders::SHADOW_VERT_BINDLESS.compile(hot_reload)
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
fn spirv_words(spv: &[u8]) -> RenderResult<Vec<u32>> {
    if !spv.len().is_multiple_of(4) {
        return Err(RenderError::Other(format!(
            "SPIR-V length {} is not a whole number of words",
            spv.len()
        )));
    }
    Ok(spv
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes([w[0], w[1], w[2], w[3]]))
        .collect())
}

pub(in crate::vulkan) fn spv_module<'d>(
    device: &'d VkDevice,
    spv: &[u8],
) -> RenderResult<SpvModule<'d>> {
    let code = spirv_words(spv).map_err(|e| e.context("shader module"))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let module = unsafe { device.create_shader_module(&info, None) }
        .map_err(|e| super::error::map_vk_result(e, "shader module"))?;
    Ok(SpvModule { device, module })
}

// The entry point every SPIR-V module the cook emits declares. dxc renames the
// stage function to `main` on the Vulkan leg, so one name covers every stage on
// this backend.
pub(in crate::vulkan) const SHADER_ENTRY: &std::ffi::CStr = c"main";

// The vertex + fragment modules a graphics pipeline is built from, held together
// so they outlive the create call and are destroyed once it returns.
pub(in crate::vulkan) struct GraphicsStages<'d> {
    vert: SpvModule<'d>,
    frag: SpvModule<'d>,
}

impl<'d> GraphicsStages<'d> {
    pub(in crate::vulkan) fn new(
        device: &'d VkDevice,
        vert_spv: &[u8],
        frag_spv: &[u8],
    ) -> RenderResult<Self> {
        Ok(Self {
            vert: spv_module(device, vert_spv)?,
            frag: spv_module(device, frag_spv)?,
        })
    }

    // The stage array a `GraphicsPipelineCreateInfo` borrows.
    pub(in crate::vulkan) fn infos(&self) -> [vk::PipelineShaderStageCreateInfo<'_>; 2] {
        [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vert.handle())
                .name(SHADER_ENTRY),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.frag.handle())
                .name(SHADER_ENTRY),
        ]
    }
}

// The world Shader's program for `entry`, as SPIR-V: the cook's artifact when
// the engine template still matches, else a compile here.
pub(super) fn world_entry(
    world: &concinnity_core::components::ShaderPrograms,
    entry: &str,
    hot_reload: bool,
) -> RenderResult<Vec<u8>> {
    let req = crate::shader::surface_source::Request {
        platform: concinnity_core::platform::Platform::Vulkan,
        hot_reload,
    };
    crate::shader::surface_source::artifact(world, entry, &req, crate::shader::compile::cooked)
        .map(|c| c.into_owned())
}

pub(super) fn compile_text_shaders(hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::TEXT_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::TEXT_FRAG.compile(hot_reload)?;
    Ok((vert, frag))
}

pub(super) fn compile_composite_shaders(hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::FULLSCREEN_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::COMPOSITE_FRAG.compile(hot_reload)?;
    Ok((vert, frag))
}

const fn vertex_binding(stride: u32) -> [vk::VertexInputBindingDescription; 1] {
    [vk::VertexInputBindingDescription {
        binding: 0,
        stride,
        input_rate: vk::VertexInputRate::VERTEX,
    }]
}

const fn attr(
    location: u32,
    format: vk::Format,
    offset: u32,
) -> vk::VertexInputAttributeDescription {
    vk::VertexInputAttributeDescription {
        location,
        binding: 0,
        format,
        offset,
    }
}

// The full Vertex struct (56 bytes).
const MAIN_VERTEX_BINDING: [vk::VertexInputBindingDescription; 1] = vertex_binding(56);
const MAIN_VERTEX_ATTRS: [vk::VertexInputAttributeDescription; 5] = [
    attr(0, vk::Format::R32G32B32_SFLOAT, 0),
    attr(1, vk::Format::R32G32B32_SFLOAT, 12),
    attr(2, vk::Format::R32G32B32_SFLOAT, 24),
    attr(3, vk::Format::R32G32B32_SFLOAT, 36),
    attr(4, vk::Format::R32G32_SFLOAT, 48),
];

// The shadow pass reads only position, so the optimizer strips the other
// attributes from its interface. Binding just that one keeps the validation
// layer from warning about unconsumed attributes; the binding keeps the full
// 56-byte `Vertex` stride.
const SHADOW_VERTEX_ATTRS: [vk::VertexInputAttributeDescription; 1] = [MAIN_VERTEX_ATTRS[0]];

// TextVertex (32 bytes): pos(vec2) + uv(vec2) + color(vec3) + mode(float).
const TEXT_VERTEX_BINDING: [vk::VertexInputBindingDescription; 1] = vertex_binding(32);
const TEXT_VERTEX_ATTRS: [vk::VertexInputAttributeDescription; 4] = [
    attr(0, vk::Format::R32G32_SFLOAT, 0),
    attr(1, vk::Format::R32G32_SFLOAT, 8),
    attr(2, vk::Format::R32G32B32_SFLOAT, 16),
    attr(3, vk::Format::R32_SFLOAT, 28),
];

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
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) struct BucketPipelineTargets {
    pub render_pass: vk::RenderPass,
    pub layout: vk::PipelineLayout,
    pub msaa_samples: vk::SampleCountFlags,
    pub swapchain_format: vk::Format,
    pub hot_reload: bool,
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
    shader: backend_init::WorldShader<'_>,
    engine_default: &(Vec<u8>, Vec<u8>),
) -> RenderResult<OwnedPipeline> {
    match shader.programs {
        Some(programs) => build_world_shader_pipeline(device, targets, bucket, programs),
        None => create_bucket_pipeline(
            device,
            targets,
            bucket,
            &engine_default.0,
            &engine_default.1,
        ),
    }
}

// Build a world Shader's bindless main-pass pipeline for bucket `bucket` from
// its own compiled stages.
pub(super) fn build_world_shader_pipeline(
    device: &VkDevice,
    targets: BucketPipelineTargets,
    bucket: usize,
    programs: &concinnity_core::components::ShaderPrograms,
) -> RenderResult<OwnedPipeline> {
    let vert_spv = world_entry(programs, "vertex_main_bindless", targets.hot_reload)?;
    let frag_spv = world_entry(programs, "fragment_main_bindless", targets.hot_reload)?;
    create_bucket_pipeline(device, targets, bucket, &vert_spv, &frag_spv)
}

fn create_bucket_pipeline(
    device: &VkDevice,
    targets: BucketPipelineTargets,
    bucket: usize,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    if vert_spv.is_empty() || frag_spv.is_empty() {
        return Err(RenderError::Other(format!(
            "shader bucket {bucket} carries no SPIR-V stages"
        )));
    }
    create_main_pipeline(
        device,
        MeshPipelineTargets {
            render_pass: targets.render_pass,
            layout: targets.layout,
            vert_spv,
            frag_spv,
        },
        targets.msaa_samples,
        targets.swapchain_format,
    )
    .map_err(|e| e.context(format_args!("shader bucket {bucket}")))
}

// Build the per-bucket pipeline table from the world's material-referenced
// shaders. Index `b` holds bucket `b + 1`'s pipeline; `None` marks a bucket the
// streaming pump installs later (its Shader is owned by a scene that has not
// pinned, so `decode_shaders` deferred its payload).
pub(super) fn build_world_pipeline_table(
    device: &VkDevice,
    targets: BucketPipelineTargets,
    bucket_shaders: &[backend_init::WorldShader<'_>],
    engine_default: &(Vec<u8>, Vec<u8>),
) -> RenderResult<Vec<Option<OwnedPipeline>>> {
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
) -> RenderResult<OwnedPipeline> {
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
) -> RenderResult<OwnedPipeline> {
    create_main_pipeline_filled(device, targets, msaa, surface_format, vk::PolygonMode::LINE)
}

fn create_main_pipeline_filled(
    device: &VkDevice,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    _surface_format: vk::Format,
    polygon_mode: vk::PolygonMode,
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        depth: Depth::write(),
        // No back-face culling, matching Metal's default and DirectX, so meshes
        // with mixed winding (procedural floor / ceiling planes) render from
        // both sides.
        raster: Raster {
            polygon_mode,
            ..Raster::default()
        },
        samples: msaa,
        vertex_bindings: &MAIN_VERTEX_BINDING,
        vertex_attributes: &MAIN_VERTEX_ATTRS,
        ..GraphicsPipelineDesc::fullscreen(
            targets.vert_spv,
            targets.frag_spv,
            targets.layout,
            targets.render_pass,
            &[Blend::Opaque],
        )
    }
    .build(device, "main")
}

pub(super) fn create_shadow_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        frag: None,
        color_targets: &[],
        depth: Depth::write(),
        raster: Raster {
            // The clamp needs the optional depthBiasClamp feature; it is 0.0
            // (unclamped) on a device without it.
            bias: Some(DepthBias {
                constant: shadow_bias::RASTER_CONSTANT,
                clamp: device.depth_bias_clamp(),
                slope: shadow_bias::RASTER_SLOPE,
            }),
            ..Raster::default()
        },
        vertex_bindings: &MAIN_VERTEX_BINDING,
        vertex_attributes: &SHADOW_VERTEX_ATTRS,
        ..GraphicsPipelineDesc::fullscreen(vert_spv, &[], layout, render_pass, &[])
    }
    .build(device, "shadow")
}

pub(super) fn create_text_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    msaa: vk::SampleCountFlags,
) -> RenderResult<OwnedPipeline> {
    // No depth test: the text overlay always draws on top.
    GraphicsPipelineDesc {
        samples: msaa,
        vertex_bindings: &TEXT_VERTEX_BINDING,
        vertex_attributes: &TEXT_VERTEX_ATTRS,
        ..GraphicsPipelineDesc::fullscreen(
            vert_spv,
            frag_spv,
            layout,
            render_pass,
            &[Blend::AlphaOver],
        )
    }
    .build(device, "text")
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
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc::fullscreen(vert_spv, frag_spv, layout, render_pass, &[Blend::Opaque])
        .build(device, "composite")
}

#[cfg(test)]
mod tests {
    use super::{
        CompileProgram, SHADOW_VERTEX_ATTRS, compile_bindless_shaders, compile_cull_shader,
        compile_cull_shader_phase2, compile_shadow_bindless_vs, compile_shadow_cull_shader,
        is_spirv, spirv_words, world_entry,
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
    // the embedded source. Guards the `#ifdef` split in `cull.hlsl`, which the
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
        concinnity_shader::require_dxc!();
        let vs = compile_shadow_bindless_vs(false).expect("shadow bindless VS compiles");
        assert!(is_spirv(&vs), "shadow bindless VS is valid SPIR-V");
    }

    // The `Location` of every `Input` variable a SPIR-V module declares, sorted.
    fn input_locations(words: &[u32]) -> Vec<u32> {
        const OP_DECORATE: u32 = 71;
        const OP_VARIABLE: u32 = 59;
        const DECORATION_LOCATION: u32 = 30;
        const STORAGE_INPUT: u32 = 1;
        let mut locations = std::collections::HashMap::new();
        let mut inputs = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            let op = words.get(at + 1..at + count.max(1)).unwrap_or(&[]);
            match words[at] & 0xFFFF {
                OP_DECORATE if op.len() >= 3 && op[1] == DECORATION_LOCATION => {
                    locations.insert(op[0], op[2]);
                }
                OP_VARIABLE if op.len() >= 3 && op[2] == STORAGE_INPUT => inputs.push(op[1]),
                _ => {}
            }
            at += count.max(1);
        }
        let mut found: Vec<u32> = inputs
            .iter()
            .filter_map(|id| locations.get(id).copied())
            .collect();
        found.sort_unstable();
        found
    }

    // The shadow pipeline binds exactly the vertex attributes its shader reads:
    // a missing one is a validation error at pipeline creation, an extra one a
    // warning.
    #[test]
    fn the_shadow_pipeline_binds_what_its_shader_reads() {
        concinnity_shader::require_dxc!();
        let vs = compile_shadow_bindless_vs(false).expect("shadow bindless VS compiles");
        let words = spirv_words(&vs).expect("whole words");
        let mut bound: Vec<u32> = SHADOW_VERTEX_ATTRS.iter().map(|a| a.location).collect();
        bound.sort_unstable();
        assert_eq!(input_locations(&words), bound);
    }

    // The bindless main shaders compile to valid SPIR-V from the embedded
    // single-source program, with nothing but the backend define ahead of it:
    // neither the pool nor the probe set takes a count.
    #[test]
    fn bindless_shaders_compile() {
        concinnity_shader::require_dxc!();
        let (vs, fs) = compile_bindless_shaders(false).expect("bindless shaders compile");
        assert!(is_spirv(&vs), "bindless vertex is valid SPIR-V");
        assert!(is_spirv(&fs), "bindless fragment is valid SPIR-V");
        let frag_src = crate::vulkan::builtin_shaders::MAIN_BINDLESS_FRAG.source(false);
        let injected: Vec<&str> = frag_src
            .lines()
            .take_while(|l| l.starts_with("#define "))
            .collect();
        assert_eq!(injected, ["#define CN_BACKEND_VULKAN 1"]);
    }

    // A world Shader's bindless pair compiles from its programs, and is its own
    // program rather than the engine's. No device is needed, so this guards the
    // world-shader path the Vulkan-on-Windows runtime cannot unit-test end to
    // end. The payload carries no cooked artifacts, so both entries take the
    // compile branch of `surface_source`, which is also what a stale cook does.
    #[test]
    fn a_world_shader_compiles_its_own_bindless_pair() {
        concinnity_shader::require_dxc!();
        let programs = concinnity_core::components::ShaderPrograms {
            name: "wall".to_string(),
            vertex: None,
            fragment: concinnity_core::components::ShaderSource {
                path: "shaders/wall.hlsl".to_string(),
                text: "float4 shade(VertexOut v, GpuObjectData od) { return (float4)(1.0); }"
                    .to_string(),
            },
            programs: Vec::new(),
        };
        let vs = world_entry(&programs, "vertex_main_bindless", false).unwrap();
        let fs = world_entry(&programs, "fragment_main_bindless", false).unwrap();
        assert!(is_spirv(&vs) && is_spirv(&fs), "the world's pair compiles");
        let (_, engine_fs) = compile_bindless_shaders(false).unwrap();
        assert_ne!(fs, engine_fs, "the world's fragment is its own program");
    }
}
