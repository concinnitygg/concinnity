// Vulkan pipeline creation for the main, shadow, and text render passes.
// Built-in GLSL programs are declared in `super::builtins` and compiled to
// SPIR-V at context init time via shaderc, unless the caller supplies valid
// SPIR-V bytes directly.

use ash::{Device, vk};

use super::builtins;

//  GLSL source strings

// Uniform and push-constant layouts are designed to match the #[repr(C)] Rust
// structs in gfx::render_types byte-for-byte under std140/std430 rules:
//
//  - ViewUniforms (160 bytes, std140 UBO): mat4 vp, mat4 view, float elapsed,
//    float _pad0, then cam_pos as 3 individual floats + 3 pad floats.
//  - LightUniforms (400 bytes, std140 UBO): DirLight and PointLight each
//    represented as two vec4s so their size is 32 bytes (matching Rust [f32;3]+f32).
//  - ShadowUniforms (272 bytes, std140 UBO): mat4 light_vps[4] (256) +
//    vec4 cascade_splits (16). Holds the cascaded shadow map VPs and the
//    view-space far-depth threshold for each cascade.
//  - Push constants (112 bytes, std430): mat4 model (64) + MaterialUniforms (48).
//    MaterialUniforms uses vec3 tint/emissive which in std430 have alignment 16;
//    the Rust struct places them at offsets 16 and 32 (both 16-byte aligned) ✓.

// Shared reflection-probe sampling (box-parallax partition-of-unity blend),
// substituted into probe-consuming fragment shaders at their `{PROBE_COMMON}`
// marker (shaderc has no #include). `{MAX_PROBES}` inside it is replaced with
// the bind count so the GLSL array sizes stay locked to
// `probe_uniforms::MAX_PROBES`. Applied by `builtins::GlslProgram::source`.
pub(in crate::vulkan) const PROBE_COMMON_GLSL: &str = include_str!("shaders/probe_common.glsl");

//  Shader compilation

pub(super) fn is_spirv(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == 0x07230203
}

// Resolve a built-in shader's source. With `hot_reload` off (production
// `cn run`) the `include_str!`-baked GLSL passed in `embedded` is returned
// directly. With it on (`cn debug`), the matching `<crate>/src/vulkan/shaders/`
// file is read from disk first, so dev-loop edits take effect on the next
// pipeline build. A missing or unreadable disk file falls back to the
// embedded source: a typo in the path can never crash the running session.
// Mirrors `directx::pipeline::shader_source` and `metal::pipeline::shader_source`.
pub(in crate::vulkan) fn shader_source(
    hot_reload: bool,
    name: &str,
    embedded: &'static str,
) -> std::borrow::Cow<'static, str> {
    if hot_reload {
        let path = format!("{}/src/vulkan/shaders/{}", env!("CARGO_MANIFEST_DIR"), name);
        match std::fs::read_to_string(&path) {
            Ok(s) => return std::borrow::Cow::Owned(s),
            Err(e) => {
                tracing::debug!(
                    "hot-reload: falling back to embedded source for {} ({})",
                    name,
                    e
                );
            }
        }
    }
    std::borrow::Cow::Borrowed(embedded)
}

// Compile the bindless static-pass shaders (bindless). `pool_size` is
// the bindless texture-pool length, substituted into the fragment source's
// `sampler2D tex_pool[]` array declaration; `probe_cube_count` is the global set
// layout's binding-8 descriptor count, substituted into the probe cube array.
// Always built from the built-in GLSL: the bindless path only drives the
// built-in shader.
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
    let vert = builtins::MAIN_BINDLESS_VERT.compile(&ctx)?;
    let frag = builtins::MAIN_BINDLESS_FRAG.compile(&ctx)?;
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
    builtins::CULL.compile(&builtins::Ctx::plain(hot_reload))
}

// Compile the phase-2 (two-pass occlusion) variant of the cull kernel. Same
// source as `compile_cull_shader`, with a `CULL_PHASE2` define injected after
// `#version` to select the `main_phase2` body (re-test the phase-1
// Hi-Z-occluded objects against the rebuilt pyramid). Mirrors the MSAA
// `#define` split the Hi-Z init kernel uses.
pub(super) fn compile_cull_shader_phase2(hot_reload: bool) -> Result<Vec<u8>, String> {
    builtins::CULL_PHASE2.compile(&builtins::Ctx::plain(hot_reload))
}

// Compile the GPU-driven shadow cull kernel: the same cull source with a
// `SHADOW_CULL` define, which drops the Hi-Z (set 1) + status (binding 3)
// bindings and does a frustum + distance test against each cascade's light
// frustum. Paired with the lean 3-SSBO shadow cull set layout.
pub(super) fn compile_shadow_cull_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    builtins::CULL_SHADOW.compile(&builtins::Ctx::plain(hot_reload))
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader.
pub(super) fn compile_shadow_bindless_vs(hot_reload: bool) -> Result<Vec<u8>, String> {
    builtins::SHADOW_BINDLESS_VERT.compile(&builtins::Ctx::plain(hot_reload))
}

// Inject a `#define` line immediately after the `#version` directive.
pub(in crate::vulkan) fn inject_define(src: &str, define: &str) -> String {
    if let Some(pos) = src.find('\n') {
        let (head, tail) = src.split_at(pos + 1);
        format!("{head}{define}{tail}")
    } else {
        format!("{define}{src}")
    }
}

// Create the GPU-cull compute pipeline. `layout` must include the cull
// descriptor set (set 0: object SSBO, draw-args SSBO, indirect-command SSBO)
// and the `CullParams` push-constant range.
pub(super) fn create_cull_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let module = spv_module(device, spv)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create cull pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline)
}

// One shaderc compiler per thread, created on first use. `Compiler::new` builds
// glslang's builtin symbol tables, which cost ~50 ms -- paid per call, that was
// 2.6 of the 3.0 seconds the 53 built-in shaders took at init. The handle is not
// thread-safe, hence thread-local rather than a shared static.
thread_local! {
    static SHADERC: std::cell::OnceCell<shaderc::Compiler> = const { std::cell::OnceCell::new() };
}

fn with_compiler<R>(f: impl FnOnce(&shaderc::Compiler) -> Result<R, String>) -> Result<R, String> {
    SHADERC.with(|cell| {
        if cell.get().is_none() {
            let compiler =
                shaderc::Compiler::new().map_err(|e| format!("shaderc init failed: {e}"))?;
            let _ = cell.set(compiler);
        }
        f(cell.get().expect("shaderc compiler present"))
    })
}

// Cache key for a default-target (Vulkan 1.0) shaderc compile. Shared by the
// runtime compile path and the export-time precompile so the two can never
// key the same inputs differently.
pub(in crate::vulkan) fn glsl_cache_key<'a>(
    source: &'a str,
    kind: shaderc::ShaderKind,
) -> crate::shader_cache::Key<'a> {
    crate::shader_cache::Key {
        compiler: "shaderc",
        source,
        entry: "main",
        target: "vulkan1.0",
        options: kind as u64,
    }
}

// `glsl_cache_key` for the ray-query target (`compile_glsl_rt`).
pub(in crate::vulkan) fn glsl_rt_cache_key<'a>(
    source: &'a str,
    kind: shaderc::ShaderKind,
) -> crate::shader_cache::Key<'a> {
    crate::shader_cache::Key {
        compiler: "shaderc",
        source,
        entry: "main",
        target: "vulkan1.2/spv1.4",
        options: kind as u64,
    }
}

// Compile GLSL to SPIR-V, reusing a cached artifact when this exact source has
// been compiled for the same target before. See `crate::shader_cache`.
pub(in crate::vulkan) fn compile_glsl(
    source: &str,
    kind: shaderc::ShaderKind,
    label: &str,
) -> Result<Vec<u8>, String> {
    let key = glsl_cache_key(source, kind);
    crate::shader_cache::cached(&key, label, || compile_glsl_uncached(source, kind, label))
}

fn compile_glsl_uncached(
    source: &str,
    kind: shaderc::ShaderKind,
    label: &str,
) -> Result<Vec<u8>, String> {
    with_compiler(|compiler| {
        let mut opts =
            shaderc::CompileOptions::new().map_err(|e| format!("shaderc options failed: {e}"))?;
        opts.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_0 as u32,
        );
        opts.set_optimization_level(shaderc::OptimizationLevel::Performance);
        let artifact = compiler
            .compile_into_spirv(source, kind, label, "main", Some(&opts))
            .map_err(|e| format!("compile {label}: {e}"))?;
        Ok(artifact.as_binary_u8().to_vec())
    })
}

// Compile GLSL that uses `GL_EXT_ray_query` (the hardware ray-traced reflection
// fragment shader). Ray query needs SPIR-V 1.4 + the Vulkan-1.2 target
// environment (the `RayQueryKHR` capability is invalid under the default
// Vulkan-1.0 target `compile_glsl` uses); the engine's instance is already 1.2,
// so the resulting module loads fine. Kept separate from `compile_glsl` so every
// other built-in shader keeps the conservative 1.0 target.
pub(in crate::vulkan) fn compile_glsl_rt(
    source: &str,
    kind: shaderc::ShaderKind,
    label: &str,
) -> Result<Vec<u8>, String> {
    let key = glsl_rt_cache_key(source, kind);
    crate::shader_cache::cached(&key, label, || {
        compile_glsl_rt_uncached(source, kind, label)
    })
}

fn compile_glsl_rt_uncached(
    source: &str,
    kind: shaderc::ShaderKind,
    label: &str,
) -> Result<Vec<u8>, String> {
    with_compiler(|compiler| {
        let mut opts =
            shaderc::CompileOptions::new().map_err(|e| format!("shaderc options failed: {e}"))?;
        opts.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        opts.set_target_spirv(shaderc::SpirvVersion::V1_4);
        opts.set_optimization_level(shaderc::OptimizationLevel::Performance);
        let artifact = compiler
            .compile_into_spirv(source, kind, label, "main", Some(&opts))
            .map_err(|e| format!("compile {label}: {e}"))?;
        Ok(artifact.as_binary_u8().to_vec())
    })
}

pub(in crate::vulkan) fn spv_module(
    device: &Device,
    spv: &[u8],
) -> Result<vk::ShaderModule, String> {
    // ash requires 4-byte aligned SPIR-V; copy into aligned Vec<u32>.
    let len = spv.len() / 4;
    let mut code = vec![0u32; len];
    unsafe { std::ptr::copy_nonoverlapping(spv.as_ptr(), code.as_mut_ptr() as *mut u8, spv.len()) };
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&info, None) }.map_err(|e| format!("shader module: {e}"))
}

// Resolve vertex/fragment/shadow SPIR-V bytes: use caller bytes if they are
// valid SPIR-V, otherwise compile the built-in GLSL fallback.
pub(super) fn resolve_main_shaders(
    hot_reload: bool,
    vert_bytes: &[u8],
    frag_bytes: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vert = if is_spirv(vert_bytes) {
        vert_bytes.to_vec()
    } else {
        builtins::MAIN_VERT.compile(&builtins::Ctx::plain(hot_reload))?
    };
    let frag = if is_spirv(frag_bytes) {
        frag_bytes.to_vec()
    } else {
        builtins::MAIN_FRAG.compile(&builtins::Ctx::plain(hot_reload))?
    };
    Ok((vert, frag))
}

// Resolve the GPU-instanced vertex shader bytes. Returns None when no
// instancing was requested AND no caller bytes are present.
pub(super) fn resolve_instanced_shader(
    hot_reload: bool,
    vert_instanced_bytes: &[u8],
    need_instanced: bool,
) -> Result<Option<Vec<u8>>, String> {
    if !need_instanced && !is_spirv(vert_instanced_bytes) {
        return Ok(None);
    }
    let spv = if is_spirv(vert_instanced_bytes) {
        vert_instanced_bytes.to_vec()
    } else {
        builtins::MAIN_VERT_INSTANCED.compile(&builtins::Ctx::plain(hot_reload))?
    };
    Ok(Some(spv))
}

// SPIR-V for the skinned-mesh shader stages, in order: the main skinned VS, the
// depth-only skinned shadow VS, and the fragment shader.
type SkinnedShaderSpirv = (Vec<u8>, Vec<u8>, Vec<u8>);

// (Fragment is shared with the static path; `frag_bytes`, when valid SPIR-V, is
// used directly, otherwise the built-in `main.frag` is compiled.)
pub(super) fn compile_skinned_shaders(
    hot_reload: bool,
    frag_bytes: &[u8],
) -> Result<SkinnedShaderSpirv, String> {
    let ctx = builtins::Ctx::plain(hot_reload);
    let main_vs = builtins::SKINNED_VERT.compile(&ctx)?;
    let shadow_vs = builtins::SKINNED_SHADOW_VERT.compile(&ctx)?;
    let frag = if is_spirv(frag_bytes) {
        frag_bytes.to_vec()
    } else {
        builtins::MAIN_FRAG.compile(&ctx)?
    };
    Ok((main_vs, shadow_vs, frag))
}

pub(super) fn resolve_shadow_shader(
    hot_reload: bool,
    shadow_bytes: &[u8],
) -> Result<Option<Vec<u8>>, String> {
    // The shadow vertex shader is engine-internal: a non-SPIR-V or empty
    // `shadow_bytes` selects the baked shadow.vert; only a real SPIR-V
    // override is used verbatim. Whether the shadow pass runs at all is gated by
    // `effective_shadow_size` at the call site, not by this function.
    let spv = if is_spirv(shadow_bytes) {
        shadow_bytes.to_vec()
    } else {
        builtins::SHADOW_VERT.compile(&builtins::Ctx::plain(hot_reload))?
    };
    Ok(Some(spv))
}

pub(super) fn compile_text_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = builtins::Ctx::plain(hot_reload);
    let vert = builtins::TEXT_VERT.compile(&ctx)?;
    let frag = builtins::TEXT_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

pub(super) fn compile_composite_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = builtins::Ctx::plain(hot_reload);
    let vert = builtins::COMPOSITE_VERT.compile(&ctx)?;
    let frag = builtins::COMPOSITE_FRAG.compile(&ctx)?;
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

// Vertex binding + attributes for the SkinnedVertex struct (80 bytes): the
// 56-byte static attributes plus uvec4 joint indices (offset 56) and vec4
// blend weights (offset 64).
fn skinned_vertex_input() -> (
    [vk::VertexInputBindingDescription; 1],
    [vk::VertexInputAttributeDescription; 7],
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
}

// Build one shader bucket's bindless main-pass pipeline. `bucket` is the
// `DrawObject::shader_bucket` value (1-based; bucket 0 is the world default
// program) and names the bucket in error messages.
//
// A bucket whose Shader resolves to the engine's built-in default renders the
// engine's own bindless program. On Vulkan the cook compiles nothing for a
// built-in-only Shader (the inline-GLSL carve-out), so the bucket carries no
// bytes and the engine's already-compiled bindless SPIR-V stands in -- the same
// substitution bucket 0 makes for a built-in world.
pub(super) fn build_bucket_pipeline(
    device: &Device,
    targets: BucketPipelineTargets,
    bucket: usize,
    shader: crate::gfx::backend_init::ShaderBytes<'_>,
    engine_default: &(Vec<u8>, Vec<u8>),
) -> Result<vk::Pipeline, String> {
    let use_default = shader.main_is_engine_default || shader.vert.is_empty();
    let (vert_spv, frag_spv) = if use_default {
        (engine_default.0.as_slice(), engine_default.1.as_slice())
    } else {
        (shader.vert, shader.frag)
    };
    if vert_spv.is_empty() || frag_spv.is_empty() {
        return Err(format!("shader bucket {bucket} carries no SPIR-V stages"));
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
    .map_err(|e| format!("shader bucket {bucket}: {e}"))
}

// Build the per-bucket pipeline table from the world's material-referenced
// shaders. Index `b` holds bucket `b + 1`'s pipeline; `None` marks a bucket the
// streaming pump installs later (its Shader is owned by a scene that has not
// pinned, so `decode_shaders` deferred its payload).
pub(super) fn build_world_pipeline_table(
    device: &Device,
    targets: BucketPipelineTargets,
    bucket_shaders: &[crate::gfx::backend_init::ShaderBytes<'_>],
    engine_default: &(Vec<u8>, Vec<u8>),
) -> Result<Vec<Option<vk::Pipeline>>, String> {
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
    device: &Device,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    surface_format: vk::Format,
) -> Result<vk::Pipeline, String> {
    create_main_pipeline_filled(device, targets, msaa, surface_format, vk::PolygonMode::FILL)
}

// The Wireframe view mode's variant of `create_main_pipeline`. Vulkan polygon
// mode is pipeline state without `VK_EXT_extended_dynamic_state3`, so the mode
// needs its own pipeline per main-pass path; see [`super::wireframe`]. Requires
// the `fillModeNonSolid` device feature.
pub(super) fn create_main_pipeline_wireframe(
    device: &Device,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    surface_format: vk::Format,
) -> Result<vk::Pipeline, String> {
    create_main_pipeline_filled(device, targets, msaa, surface_format, vk::PolygonMode::LINE)
}

fn create_main_pipeline_filled(
    device: &Device,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    _surface_format: vk::Format,
    polygon_mode: vk::PolygonMode,
) -> Result<vk::Pipeline, String> {
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
            .module(vert_mod)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
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

    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create main pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Same as `create_main_pipeline` but takes an instanced vertex shader. The
// caller is responsible for using a pipeline layout that includes the
// per-instance storage buffer descriptor set (set=2).
pub(super) fn create_instanced_pipeline(
    device: &Device,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    surface_format: vk::Format,
) -> Result<vk::Pipeline, String> {
    create_main_pipeline(device, targets, msaa, surface_format)
}

pub(super) fn create_shadow_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_mod)
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
        .depth_bias_constant_factor(0.005)
        .depth_bias_slope_factor(1.0)
        // A non-zero clamp needs the optional depthBiasClamp device feature;
        // 0.0 (unclamped) keeps the constant + slope bias without it.
        .depth_bias_clamp(0.0);

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

    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create shadow pipeline: {e}"))?[0];

    unsafe { device.destroy_shader_module(vert_mod, None) };
    Ok(pipeline)
}

// Main-pass pipeline for skinned geometry: the skinned vertex shader (80-byte
// layout) paired with the standard fragment shader. The caller passes a
// pipeline layout that includes the joint storage-buffer descriptor set.
pub(super) fn create_skinned_pipeline(
    device: &Device,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
) -> Result<vk::Pipeline, String> {
    create_skinned_pipeline_filled(device, targets, msaa, vk::PolygonMode::FILL)
}

// The Wireframe view mode's variant of `create_skinned_pipeline`; see
// [`super::wireframe`]. Requires the `fillModeNonSolid` device feature.
pub(super) fn create_skinned_pipeline_wireframe(
    device: &Device,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
) -> Result<vk::Pipeline, String> {
    create_skinned_pipeline_filled(device, targets, msaa, vk::PolygonMode::LINE)
}

fn create_skinned_pipeline_filled(
    device: &Device,
    targets: MeshPipelineTargets<'_>,
    msaa: vk::SampleCountFlags,
    polygon_mode: vk::PolygonMode,
) -> Result<vk::Pipeline, String> {
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
            .module(vert_mod)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
            .name(&entry),
    ];

    let (bindings, attrs) = skinned_vertex_input();
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

    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create skinned pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Shadow-pass pipeline for skinned geometry: the skinned shadow vertex shader
// (80-byte layout, depth-only).
pub(super) fn create_skinned_shadow_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_mod)
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
        .depth_bias_constant_factor(0.005)
        .depth_bias_slope_factor(1.0)
        .depth_bias_clamp(0.0);

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

    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create skinned shadow pipeline: {e}"))?[0];

    unsafe { device.destroy_shader_module(vert_mod, None) };
    Ok(pipeline)
}

pub(super) fn create_text_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    msaa: vk::SampleCountFlags,
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
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

    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create text pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Build the composite (post-process) pipeline: a vertex-buffer-less fullscreen
// triangle that samples the resolved HDR target and applies ACES + gamma +
// FXAA. Targets the single-sample swapchain backbuffer; no depth attachment.
pub(super) fn create_composite_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
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

    let pipeline = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create composite pipeline: {e}"))?[0];

    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::{
        builtins, compile_bindless_shaders, compile_cull_shader, compile_cull_shader_phase2,
        compile_shadow_bindless_vs, compile_shadow_cull_shader, compile_skinned_shaders, is_spirv,
        resolve_instanced_shader, resolve_main_shaders,
    };

    // The phase-1 cull kernel, its two-pass `CULL_PHASE2` variant, and the
    // GPU-driven shadow `SHADOW_CULL` variant all compile to valid SPIR-V from
    // the embedded source. Guards the `#ifdef` split in `cull.comp`, which the
    // Vulkan-on-Windows runtime cannot currently smoke-test.
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
        let vs = compile_shadow_bindless_vs(false).expect("shadow bindless VS compiles");
        assert!(is_spirv(&vs), "shadow bindless VS is valid SPIR-V");
    }

    // The bindless main shaders compile to valid SPIR-V from the embedded source,
    // including the reflection-probe sampling injected from `probe_common.glsl` at
    // the `{PROBE_COMMON}` marker + the `{MAX_PROBES}` / `{POOL_SIZE}` substitutions.
    // Guards the probe forward path (the box-parallax partition-of-unity blend +
    // the ProbeSet UBO / probe cube array declarations) offline: a GLSL error in
    // the injection fails here without needing a GPU.
    // A device-shortened probe cube array is compiled too: the array length is a
    // runtime value on a sampler-starved driver, so the shortest and the ceiling
    // forms both have to survive the GLSL injection.
    #[test]
    fn bindless_shaders_compile() {
        for probes in [1, 7, crate::vulkan::probe_uniforms::MAX_PROBES as u32] {
            let (vs, fs) =
                compile_bindless_shaders(false, 4, probes).expect("bindless shaders compile");
            assert!(is_spirv(&vs), "bindless vertex is valid SPIR-V");
            assert!(is_spirv(&fs), "bindless fragment is valid SPIR-V");
        }
        // The probe markers must be fully substituted (no literal token survives).
        let frag_src = builtins::MAIN_BINDLESS_FRAG.source(&builtins::Ctx {
            hot_reload: false,
            msaa: false,
            pool_size: 4,
            probe_count: 4,
        });
        assert!(!frag_src.contains("{PROBE_COMMON}"));
        assert!(!frag_src.contains("{MAX_PROBES}"));
        assert!(!frag_src.contains("{PROBE_DESC_SET}"));
        assert!(!frag_src.contains("{POOL_SIZE}"));
    }

    // The shader-resolution helpers that `update_world_shader_pipelines`
    // composes when hot-swapping a world's Shader pipelines: valid
    // SPIR-V (the bytes the hot-reload recompile always produces) is passed
    // through verbatim, while non-SPIR-V selects the built-in GLSL fallback.
    // No device is needed, so this guards the world-shader hot-swap path the
    // Vulkan-on-Windows runtime cannot unit-test end to end.
    #[test]
    fn world_shader_resolution_passes_spirv_and_falls_back_to_glsl() {
        // Build real SPIR-V from the bundled GLSL, then confirm
        // `resolve_main_shaders` returns it unchanged (the hot-swap's main
        // pipeline reuses these bytes directly).
        let ctx = builtins::Ctx::plain(false);
        let vert_spv = builtins::MAIN_VERT.compile(&ctx).unwrap();
        let frag_spv = builtins::MAIN_FRAG.compile(&ctx).unwrap();
        let (v, f) = resolve_main_shaders(false, &vert_spv, &frag_spv).unwrap();
        assert_eq!(v, vert_spv, "SPIR-V vertex bytes pass through unchanged");
        assert_eq!(f, frag_spv, "SPIR-V fragment bytes pass through unchanged");

        // Non-SPIR-V bytes fall back to the engine GLSL, which compiles to
        // valid SPIR-V.
        let (v2, f2) = resolve_main_shaders(false, b"not spirv", b"still not spirv").unwrap();
        assert!(is_spirv(&v2), "GLSL fallback vertex compiles to SPIR-V");
        assert!(is_spirv(&f2), "GLSL fallback fragment compiles to SPIR-V");

        // The instanced helper, forced on (as the hot-swap does when an
        // instanced pipeline is live), yields valid SPIR-V from the fallback.
        let inst = resolve_instanced_shader(false, b"not spirv", true)
            .unwrap()
            .expect("forced instanced resolve yields Some");
        assert!(is_spirv(&inst), "instanced fallback compiles to SPIR-V");

        // The skinned helper compiles its engine-internal vertex + shadow
        // stages from inline GLSL and passes the supplied SPIR-V fragment
        // through, matching what the hot-swap feeds the skinned pipeline.
        let (skinned_vs, skinned_shadow_vs, skinned_frag) =
            compile_skinned_shaders(false, &frag_spv).unwrap();
        assert!(is_spirv(&skinned_vs), "skinned VS compiles to SPIR-V");
        assert!(is_spirv(&skinned_shadow_vs), "skinned shadow VS compiles");
        assert_eq!(skinned_frag, frag_spv, "skinned fragment passes through");
    }
}
