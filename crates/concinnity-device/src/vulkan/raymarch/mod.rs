//! Raymarched SDF volume pass for the Vulkan backend. Runs at `PassId::Raymarch`
//! between `AutoExposure` and `Decals` on the `hdr_resolve` RMW chain. Each
//! `SdfVolume` rasterizes the back faces of its world-space bounding box and runs
//! the single-source `raymarch.hlsl` fragment that sphere-traces the SDF inside
//! the box, with the same depth-compositing rules as the other backends.
//!
//! MSAA depth write-back. MSAA is on by default, so the pass renders the proxy
//! into the multisampled HDR color + the writable scene depth (the shader
//! writes hit depth through a conservative `SV_Depth` output), then resolves
//! the combined color into `hdr_resolve` so the single-sample post stack picks
//! up the raymarched pixels and the raymarched-surface depth. The main pass
//! keeps its multisampled color for this pass to draw over whenever raymarch is
//! active (see `VkContext::main_close`). When single-sampled the main pass
//! already leaves the scene in `hdr_resolve`, so the pass draws over it
//! directly (no resolve).
//!
//! Every `SdfVolume` builds here from its one distance-field payload, the same
//! payload the other backends consume. `try_new` returns `Ok(None)` only when the
//! world declares no volumes; a payload decode or pipeline compile failure aborts
//! init.

use ash::vk;
use concinnity_core::math::{SaturatingFrom, saturating_u32};
use concinnity_core::platform::Platform;
use concinnity_engine::geometry::proxy_cube::{CUBE_INDICES, cube_corners};
use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::gfx::mesh_payload::Vertex;
use concinnity_engine::gfx::render_types::{LightUniforms, ShadowUniforms};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::init::SdfVolumeSource;
use concinnity_engine::render::sdf::programs::SdfPrograms;
use concinnity_engine::render::shader::programs::raymarch::{Family, ProxyFaces, VolumeFlags};
use concinnity_engine::transform::mat4_inverse;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::{Blend, Depth, GraphicsPipelineDesc, Raster};
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::{GpuImage, create_hdr_snapshot};
use crate::shader::raymarch_source::{face_artifacts, family_artifacts};
use crate::vulkan::main_pass::{self, MAIN_FORMATS, MainClose, MainOpen};
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::post::gbuffer::{GbufferPrepassView, PREPASS_FORMATS};
use crate::vulkan::rendering::PassFormats;

// 36 indices for the unit-cube proxy: one face set is culled so each pixel
// inside the bounding box gets exactly one fragment.
const CUBE_INDEX_COUNT: u32 = 36;

// One declaration for all three backends, in `core::render::uniforms`.
// Re-exported so `crate::vulkan::raymarch::{RaymarchView, RaymarchVolumeUniforms}`
// stay the paths the encode sites use.
pub(in crate::vulkan) use concinnity_engine::render::uniforms::{
    RaymarchView, RaymarchVolumeUniforms,
};

// Which cascade a shadow-caster draw targets, pushed to both stages. The whole
// 16-byte block is declared, not just the live `u32`: the shared source spells
// it out to the padding the other two hosts allocate, and a range narrower than
// what the shader declares is a validation finding.
use crate::vulkan::record::Recorder;
use crate::vulkan::rendering::{ImageUse, Transition};
use concinnity_engine::render::uniforms::{
    GBufferView, PassCamera, PassLighting, RaymarchShadowCascade,
};

mod swap;

// Copy the resolved scene into the refraction snapshot, leaving the snapshot
// sampled and the scene a copy source, where the pass's scope opens it from.
// Only a frame drawing a volume that taps the scene runs it.
fn copy_scene_snapshot(
    rec: &Recorder<'_>,
    hdr_resolve: vk::Image,
    snapshot: vk::Image,
    extent: vk::Extent2D,
) {
    let layers = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let region = vk::ImageCopy::default()
        .src_subresource(layers)
        .dst_subresource(layers)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        });
    rec.transition(&[
        Transition::color(hdr_resolve, ImageUse::Sampled, ImageUse::TransferSrc),
        Transition::color(snapshot, ImageUse::Sampled, ImageUse::TransferDst),
    ]);
    rec.copy_image(
        hdr_resolve,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        snapshot,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        std::slice::from_ref(&region),
    );
    rec.transition(&[Transition::color(
        snapshot,
        ImageUse::TransferDst,
        ImageUse::Sampled,
    )]);
}

// Bind the shared proxy cube's vertex and index buffers on a recording `cmd`.
fn bind_cube(rec: &Recorder<'_>, rm: &RaymarchResources) {
    let vertex_buffer = rm.cube_vb.buffer();
    rec.bind_vertex_buffers(0, &[vertex_buffer], &[0]);
    rec.bind_index_buffer(rm.cube_ib.buffer(), 0, vk::IndexType::UINT16);
}

// Draw one volume's proxy cube under `pipeline` with its set bound at set 1. The
// caller has bound the cube buffers, set 0 and the push constants on a recording `cmd`.
fn draw_volume(
    rec: &Recorder<'_>,
    layout: &OwnedPipelineLayout,
    pipeline: &OwnedPipeline,
    volume_set: vk::DescriptorSet,
) {
    rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
    rec.bind_descriptor_sets(
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        1,
        std::slice::from_ref(&volume_set),
        &[],
    );
    rec.draw_indexed(CUBE_INDEX_COUNT, 1, 0, 0, 0);
}

// Per-`SdfVolume` GPU state: the compiled render pipeline, the static per-volume
// UBO (uploaded once at init), its descriptor set, and the visibility flag the
// encoder + `visible_bounds` read.
struct RaymarchVolumeRecord {
    // The volume's asset name, for pipeline errors.
    label: String,
    // The flags the volume's pipelines were built for.
    flags: VolumeFlags,
    // Owned, not a raw handle: the record outlives the local the pipeline was
    // built into, so storing the bare `vk::Pipeline` destroyed it at the end of
    // the loop iteration and left every draw binding a dangling one.
    // A surface's is the back-face variant; `front_pipeline` is the other.
    pipeline: OwnedPipeline,
    // The surface's front-face pipeline; `None` for a medium.
    front_pipeline: Option<OwnedPipeline>,
    // Depth-only shadow-caster pipeline. `Some` when the asset's `cast_shadows`
    // is set; the shadow encoder iterates only the volumes where this is `Some`
    // and `visible`.
    shadow_pipeline: Option<OwnedPipeline>,
    // The surface's G-buffer pre-pass pipelines; `None` for a medium.
    prepass_pipelines: Option<FacePipelines>,
    // The world-space bounding box, which decides the faces the proxy draws.
    center: [f32; 3],
    extent: [f32; 3],
    // Held for the volume's lifetime; `volume_set` aliases it.
    _volume_ubo: PooledBuffer,
    volume_set: vk::DescriptorSet,
    visible: bool,
    // Whether the authored field samples the scene behind the surface. The
    // pass copies hdr_resolve into `snapshot` only when some visible volume
    // does; the descriptor at set 0 binding 6 stands either way, so this
    // selects no pipeline variant.
    refractive: bool,
}

impl RaymarchVolumeRecord {
    // True when the volume is shown and its box reaches into `frustum`.
    fn drawn(&self, frustum: &Frustum) -> bool {
        self.visible && frustum.intersects_box(self.center, self.extent)
    }

    // The faces this volume's proxy draws with from `view`: a medium always
    // takes the back faces.
    fn faces(&self, view: &RaymarchView) -> ProxyFaces {
        if self.flags.volumetric {
            ProxyFaces::Back
        } else {
            ProxyFaces::for_box(view, self.center, self.extent)
        }
    }

    // The pipeline the volume's own draw takes from `view`.
    fn draw_pipeline(&self, view: &RaymarchView) -> &OwnedPipeline {
        match (self.faces(view), &self.front_pipeline) {
            (ProxyFaces::Front, Some(front)) => front,
            _ => &self.pipeline,
        }
    }
}

// Engine-side raymarch resources. Built only when at least one `SdfVolume`
// landed at init; `VkContext::raymarch` stays `None` otherwise and
// the pass is omitted from the frame graph.
pub(in crate::vulkan) struct RaymarchResources {
    pipeline_layout: OwnedPipelineLayout,
    _view_set_layout: OwnedSetLayout,
    _volume_set_layout: OwnedSetLayout,
    _descriptor_pool: OwnedDescriptorPool,

    // Per-frame `RaymarchView` UBO ring. Persistently mapped; the encoder
    // memcpys this frame's view into `view_ubo_buffers[frame_idx].mapped_ptr()` before binding.
    view_ubos: Vec<PooledBuffer>,
    view_sets: Vec<vk::DescriptorSet>,

    // Shared unit-cube proxy geometry (positions at +/-1; the vertex shader
    // scales by `vol_extent` + offsets by `vol_center`).
    cube_vb: PooledBuffer,
    cube_ib: PooledBuffer,

    // Pre-raymarch HDR scene snapshot for the refraction tap (`scene_color`).
    // The encoder copies hdr_resolve into this at the head of the pass; sized to
    // render dims, recreated by `rebuild` on resize.
    snapshot: GpuImage,

    // Shadow casters, built when a volume casts shadows: a 3-UBO view set with its
    // own per-frame view ring, so the concurrently recorded Shadow pass never
    // touches the main ring, and a `cascade_idx` push constant.
    shadow_pipeline_layout: OwnedPipelineLayout,
    _shadow_view_set_layout: OwnedSetLayout,
    shadow_view_ubos: Vec<PooledBuffer>,
    shadow_view_sets: Vec<vk::DescriptorSet>,

    // G-buffer pre-pass resources, built when a surface volume exists: its own
    // view ring and sets.
    prepass_view_ubos: Vec<PooledBuffer>,
    prepass_view_sets: Vec<vk::DescriptorSet>,

    volumes: Vec<RaymarchVolumeRecord>,
}

// One corner of the proxy cube. Only the position is fetched (location 0); the
// rest of the 56-byte engine `Vertex` is zeroed.
fn cube_vertex(pos: [f32; 3]) -> Vertex {
    Vertex {
        pos,
        normal: [0.0; 3],
        tangent: [0.0; 3],
        color: [0.0; 3],
        uv: [0.0; 2],
    }
}

// Build the shared unit-cube proxy VB + IB. 8 corners at +/-1; 36 indices (the
// pipeline culls front faces so only back faces fire). Host-visible buffers,
// written once. Mirrors `directx::raymarch::build_cube_buffers`.
type CubeBuffers = (PooledBuffer, PooledBuffer);
fn build_cube_buffers(alloc: &DeviceAllocator) -> RenderResult<CubeBuffers> {
    let corners: [Vertex; 8] = cube_corners(1.0).map(cube_vertex);
    let indices = CUBE_INDICES;

    let vb_bytes = size_of_val(&corners) as u64;
    let ib_bytes = size_of_val(&indices) as u64;
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let vb = alloc.create_buffer(vb_bytes, vk::BufferUsageFlags::VERTEX_BUFFER, host)?;
    let ib = alloc.create_buffer(ib_bytes, vk::BufferUsageFlags::INDEX_BUFFER, host)?;
    vb.write_slice(0, &corners);
    ib.write_slice(0, &indices);
    Ok((vb, ib))
}

// The per-frame view set: three UBOs, the four sampled images, the far shadow
// map and their three samplers.
fn view_set_bindings() -> [Binding; 11] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    let vert_frag = vk::ShaderStageFlags::VERTEX | frag;
    [
        (0, T::UNIFORM_BUFFER, vert_frag), // RaymarchView
        (1, T::UNIFORM_BUFFER, frag),      // RaymarchLights
        (2, T::UNIFORM_BUFFER, frag),      // RaymarchShadow
        (3, T::SAMPLED_IMAGE, frag),       // shadow_map
        (4, T::SAMPLED_IMAGE, frag),       // irradiance cube
        (5, T::SAMPLED_IMAGE, frag),       // prefilter cube
        (6, T::SAMPLED_IMAGE, frag),       // scene_color snapshot
        (7, T::SAMPLED_IMAGE, frag),       // far_shadow_map
        (8, T::SAMPLER, frag),             // the shadow map's compare sampler
        (9, T::SAMPLER, frag),             // the cubes' sampler
        (10, T::SAMPLER, frag),            // the snapshot's sampler
    ]
}

// The per-volume set: its `RaymarchVolumeUniforms`.
fn volume_set_bindings() -> [Binding; 1] {
    [(
        0,
        vk::DescriptorType::UNIFORM_BUFFER,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
    )]
}

// Which of the three-UBO view rings the world's volumes need.
#[derive(Clone, Copy)]
struct ViewRings {
    // Some volume casts a shadow.
    shadow: bool,
    // Some volume is a surface, which draws into the G-buffer pre-pass.
    prepass: bool,
}

impl ViewRings {
    fn of(volumes: &[SdfVolumeSource]) -> Self {
        Self {
            shadow: volumes.iter().any(|s| VolumeFlags::of(&s.volume).casts()),
            prepass: volumes.iter().any(|s| !s.volume.volumetric),
        }
    }
}

// Room for `frames` view sets, one set per volume, and `frames` three-UBO view
// sets for each ring in `rings`.
fn create_descriptor_pool(
    device: &VkDevice,
    frames: usize,
    volumes: usize,
    rings: ViewRings,
) -> RenderResult<OwnedDescriptorPool> {
    let f = u32::saturating_from(frames);
    let v = u32::saturating_from(volumes);
    let shadow_sets = f * (u32::from(rings.shadow) + u32::from(rings.prepass));
    let sizes = PoolSizes::default()
        .sets(&view_set_bindings(), f)
        .sets(&volume_set_bindings(), v)
        .sets(&shadow_view_set_bindings(), shadow_sets)
        .build();
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(f + v + shadow_sets)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| super::error::map_vk_result(e, "raymarch descriptor pool"))
}

// The 3-UBO set for the shadow and pre-pass marches: view, lights and cascade
// VPs. No textures, so the shadow map being written is never also bound.
fn shadow_view_set_bindings() -> [Binding; 3] {
    let frag = vk::ShaderStageFlags::FRAGMENT;
    let vert_frag = vk::ShaderStageFlags::VERTEX | frag;
    let ubo = vk::DescriptorType::UNIFORM_BUFFER;
    [
        (0, ubo, vert_frag), // RaymarchView
        (1, ubo, frag),      // RaymarchLights (sun direction)
        (2, ubo, vert_frag), // RaymarchShadow (light VPs)
    ]
}

fn write_shadow_view_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    view_ubo: vk::Buffer,
    light_ubo: vk::Buffer,
    shadow_ubo: vk::Buffer,
) {
    SetWrites::new(set)
        .uniform_buffer(0, view_ubo, size_of::<RaymarchView>() as u64)
        .uniform_buffer(1, light_ubo, size_of::<LightUniforms>() as u64)
        .uniform_buffer(2, shadow_ubo, size_of::<ShadowUniforms>() as u64)
        .apply(device);
}

// The three UBOs bound into one per-frame view set: the per-frame RaymarchView,
// the shared lights, and the shared shadow VPs.
#[derive(Clone, Copy)]
struct RaymarchViewSetBuffers {
    view: vk::Buffer,
    light: vk::Buffer,
    shadow: vk::Buffer,
}

// The image/sampler pairs bound into one per-frame view set: the CSM shadow
// map, the IBL irradiance + prefilter cubes (sharing the cube sampler), and the
// per-frame-independent scene snapshot tap.
#[derive(Clone, Copy)]
struct RaymarchViewSetTextures {
    shadow_map_view: vk::ImageView,
    far_shadow_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    irradiance_view: vk::ImageView,
    prefilter_view: vk::ImageView,
    cube_sampler: vk::Sampler,
    snapshot_view: vk::ImageView,
    scene_sampler: vk::Sampler,
}

// Write every binding of one per-frame view set. The shadow / IBL / light /
// shadow-UBO bindings are shared engine resources; the snapshot is the
// per-frame-independent scene tap (its contents are refreshed by the encoder).
fn write_view_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    buffers: RaymarchViewSetBuffers,
    textures: RaymarchViewSetTextures,
) {
    let RaymarchViewSetBuffers {
        view: view_ubo,
        light: light_ubo,
        shadow: shadow_ubo,
    } = buffers;
    let RaymarchViewSetTextures {
        shadow_map_view,
        far_shadow_view,
        shadow_sampler,
        irradiance_view,
        prefilter_view,
        cube_sampler,
        snapshot_view,
        scene_sampler,
    } = textures;
    SetWrites::new(set)
        .uniform_buffer(0, view_ubo, size_of::<RaymarchView>() as u64)
        .uniform_buffer(1, light_ubo, size_of::<LightUniforms>() as u64)
        .uniform_buffer(2, shadow_ubo, size_of::<ShadowUniforms>() as u64)
        .sampled_image(3, shadow_map_view)
        .sampled_image(4, irradiance_view)
        .sampled_image(5, prefilter_view)
        .sampled_image(6, snapshot_view)
        .sampled_image(7, far_shadow_view)
        .sampler(8, shadow_sampler)
        .sampler(9, cube_sampler)
        .sampler(10, scene_sampler)
        .apply(device);
}

fn write_volume_set(device: &VkDevice, set: vk::DescriptorSet, volume_ubo: vk::Buffer) {
    SetWrites::new(set)
        .uniform_buffer(0, volume_ubo, size_of::<RaymarchVolumeUniforms>() as u64)
        .apply(device);
}

// Cube proxy VB: the 56-byte engine `Vertex`, position (location 0) only.
const CUBE_VERTEX_BINDINGS: [vk::VertexInputBindingDescription; 1] =
    [vk::VertexInputBindingDescription {
        binding: 0,
        stride: saturating_u32(size_of::<Vertex>()),
        input_rate: vk::VertexInputRate::VERTEX,
    }];
const CUBE_VERTEX_ATTRIBUTES: [vk::VertexInputAttributeDescription; 1] =
    [vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    }];

// The proxy rasterizer drawing `faces`. Every pass renders with a
// negative-height (Y-flipped) viewport, under which the proxy's near faces wind
// CCW, so culling the front face leaves the back faces (matching the DirectX
// path).
const fn cube_raster(faces: ProxyFaces) -> Raster {
    Raster {
        cull: match faces {
            ProxyFaces::Front => vk::CullModeFlags::BACK,
            ProxyFaces::Back => vk::CullModeFlags::FRONT,
        },
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        polygon_mode: vk::PolygonMode::FILL,
        bias: None,
    }
}

// A pipeline drawing the cube proxy's `faces` into `color_targets`.
fn cube_proxy<'a>(
    vert_spv: &'a [u8],
    frag_spv: &'a [u8],
    faces: ProxyFaces,
    layout: vk::PipelineLayout,
    formats: PassFormats,
    color_targets: &'a [Blend],
) -> GraphicsPipelineDesc<'a> {
    GraphicsPipelineDesc {
        raster: cube_raster(faces),
        vertex_bindings: &CUBE_VERTEX_BINDINGS,
        vertex_attributes: &CUBE_VERTEX_ATTRIBUTES,
        ..GraphicsPipelineDesc::fullscreen(vert_spv, frag_spv, layout, formats, color_targets)
    }
}

// A per-volume pipeline over the proxy's `faces`: opaque, inclusive depth test
// with write, since the fragment writes `SV_Depth`.
fn create_pipeline(
    device: &VkDevice,
    t: &VolumePipelineTargets,
    vert_spv: &[u8],
    frag_spv: &[u8],
    faces: ProxyFaces,
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        depth: Depth::write_inclusive(),
        samples: t.msaa_samples,
        ..cube_proxy(
            vert_spv,
            frag_spv,
            faces,
            t.layout,
            MAIN_FORMATS,
            &[Blend::Opaque],
        )
    }
    .build(device, "raymarch")
}

// The volumetric variant: alpha-blended over the scene and depth-tested
// without writing, since the medium is translucent and occludes neither itself
// nor what reads the scene depth after it.
fn create_volumetric_pipeline(
    device: &VkDevice,
    t: &VolumePipelineTargets,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        depth: Depth::read_only(),
        samples: t.msaa_samples,
        ..cube_proxy(
            vert_spv,
            frag_spv,
            ProxyFaces::Back,
            t.layout,
            MAIN_FORMATS,
            &[Blend::CoverageOver],
        )
    }
    .build(device, "raymarch volumetric")
}

// Build a per-volume depth-only shadow-caster pipeline. Same cube proxy + front
// cull as the main pass, but no color attachment (single-sample shadow map),
// the shadow casters' depth test (matching the rasterized CSM casters) with depth write, and
// the fragment writes hit depth via `SV_Depth`, into a cascade slice.
fn create_shadow_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        depth: Depth::write(),
        ..cube_proxy(
            vert_spv,
            frag_spv,
            ProxyFaces::Back,
            layout,
            crate::vulkan::context::SHADOW_FORMATS,
            &[],
        )
    }
    .build(device, "raymarch shadow")
}

// Build a surface volume's G-buffer pre-pass pipeline drawing the proxy's
// `faces`: the pre-pass's three color targets over its single-sample depth,
// depth-tested inclusively with depth write like the main draw.
fn create_prepass_pipeline(
    device: &VkDevice,
    t: &VolumePipelineTargets,
    vert_spv: &[u8],
    frag_spv: &[u8],
    faces: ProxyFaces,
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        depth: Depth::write_inclusive(),
        ..cube_proxy(
            vert_spv,
            frag_spv,
            faces,
            t.shadow_layout,
            PREPASS_FORMATS,
            &crate::vulkan::post::gbuffer::PREPASS_TARGETS,
        )
    }
    .build(device, "raymarch prepass")
}

// What a volume's pipelines are built against: the pass's layouts and the
// target configuration the pass was built with.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::vulkan) struct VolumePipelineTargets {
    layout: vk::PipelineLayout,
    shadow_layout: vk::PipelineLayout,
    msaa_samples: vk::SampleCountFlags,
    hot_reload: bool,
}

// A surface family's pipeline per set of proxy faces.
pub(in crate::vulkan) struct FacePipelines {
    front: OwnedPipeline,
    back: OwnedPipeline,
}

impl FacePipelines {
    fn get(&self, faces: ProxyFaces) -> &OwnedPipeline {
        match faces {
            ProxyFaces::Front => &self.front,
            ProxyFaces::Back => &self.back,
        }
    }
}

// Every pipeline one volume draws with: its own (both face sets for a
// surface), its shadow caster when it casts one, and its G-buffer pre-pass
// share when it is a surface.
pub(in crate::vulkan) struct VolumePipelines {
    pipeline: OwnedPipeline,
    front_pipeline: Option<OwnedPipeline>,
    shadow_pipeline: Option<OwnedPipeline>,
    prepass_pipelines: Option<FacePipelines>,
}

// Build the pipelines of a volume with these flags from its compiled field. A
// medium authors `sampleVolume` and renders alpha-blended without a depth
// write; a surface volume authors `map` and `shade` and sphere-traces an opaque
// surface.
pub(in crate::vulkan) fn build_volume_pipelines(
    device: &VkDevice,
    t: &VolumePipelineTargets,
    programs: &SdfPrograms,
    flags: VolumeFlags,
    label: &str,
) -> RenderResult<VolumePipelines> {
    let artifacts = |family, faces| {
        face_artifacts(
            programs,
            family,
            faces,
            Platform::Vulkan,
            t.hot_reload,
            label,
        )
    };
    let surface = |faces| {
        let (vert, frag) = artifacts(Family::Surface, faces)?;
        create_pipeline(device, t, &vert, &frag, faces)
    };
    let (pipeline, front_pipeline) = if flags.volumetric {
        let (vert, frag) = family_artifacts(
            programs,
            Family::Volumetric,
            Platform::Vulkan,
            t.hot_reload,
            label,
        )?;
        let medium = create_volumetric_pipeline(device, t, &vert, &frag)?;
        (medium, None)
    } else {
        (
            surface(ProxyFaces::Back)?,
            Some(surface(ProxyFaces::Front)?),
        )
    };
    let shadow_pipeline = if flags.casts() {
        let (sh_vert, sh_frag) = family_artifacts(
            programs,
            Family::Shadow,
            Platform::Vulkan,
            t.hot_reload,
            label,
        )?;
        Some(create_shadow_pipeline(
            device,
            t.shadow_layout,
            &sh_vert,
            &sh_frag,
        )?)
    } else {
        None
    };
    let prepass = |faces| {
        let (vert, frag) = artifacts(Family::Prepass, faces)?;
        create_prepass_pipeline(device, t, &vert, &frag, faces)
    };
    let prepass_pipelines = if flags.volumetric {
        None
    } else {
        Some(FacePipelines {
            front: prepass(ProxyFaces::Front)?,
            back: prepass(ProxyFaces::Back)?,
        })
    };
    Ok(VolumePipelines {
        pipeline,
        front_pipeline,
        shadow_pipeline,
        prepass_pipelines,
    })
}

// Vulkan device/instance handles used to create + rebuild raymarch GPU
// resources. Shared by `try_new` and `rebuild`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RaymarchDeviceContext<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) device: &'a VkDevice,
    pub(in crate::vulkan) command_pool: vk::CommandPool,
    pub(in crate::vulkan) queue: vk::Queue,
}

// Render-target configuration for the raymarch pass: the per-frame ring depth,
// the MSAA sample count, and the render dims the snapshot is sized to.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RaymarchTargetConfig {
    pub(in crate::vulkan) frames: usize,
    pub(in crate::vulkan) msaa_samples: vk::SampleCountFlags,
    pub(in crate::vulkan) width: u32,
    pub(in crate::vulkan) height: u32,
}

// Shared engine resources bound into the view sets (and the shadow view sets):
// the CSM shadow map, the IBL cubes + samplers, the scene-snapshot sampler, the
// light + shadow UBO rings.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RaymarchSharedBindings<'a> {
    pub(in crate::vulkan) shadow_map_view: vk::ImageView,
    pub(in crate::vulkan) far_shadow_view: vk::ImageView,
    pub(in crate::vulkan) shadow_sampler: vk::Sampler,
    pub(in crate::vulkan) irradiance_view: vk::ImageView,
    pub(in crate::vulkan) prefilter_view: vk::ImageView,
    pub(in crate::vulkan) cube_sampler: vk::Sampler,
    pub(in crate::vulkan) linear_sampler: vk::Sampler,
    // Per-frame-in-flight LightUniforms ring, indexed by frame slot.
    pub(in crate::vulkan) light_ubos: &'a [PooledBuffer],
    // Per-frame-in-flight ShadowUniforms ring, indexed by frame slot.
    pub(in crate::vulkan) shadow_ubos: &'a [PooledBuffer],
}

impl RaymarchResources {
    // What this pass's volume pipelines are built against.
    pub(in crate::vulkan) fn volume_targets(
        &self,
        msaa_samples: vk::SampleCountFlags,
        hot_reload: bool,
    ) -> VolumePipelineTargets {
        VolumePipelineTargets {
            layout: self.pipeline_layout.handle(),
            shadow_layout: self.shadow_pipeline_layout.handle(),
            msaa_samples,
            hot_reload,
        }
    }

    // Build every raymarch resource + the per-volume records. Returns
    // `Ok(None)` when `sdf_volumes` is empty so the engine omits the pass.
    pub(in crate::vulkan) fn try_new(
        ctx: RaymarchDeviceContext<'_>,
        target: RaymarchTargetConfig,
        bindings: RaymarchSharedBindings<'_>,
        sdf_volumes: &[SdfVolumeSource],
        hot_reload: bool,
    ) -> RenderResult<Option<Self>> {
        let RaymarchDeviceContext {
            alloc,
            device,
            command_pool,
            queue,
        } = ctx;
        let RaymarchTargetConfig {
            frames,
            msaa_samples,
            width,
            height,
        } = target;
        let RaymarchSharedBindings {
            shadow_map_view,
            far_shadow_view,
            shadow_sampler,
            irradiance_view,
            prefilter_view,
            cube_sampler,
            linear_sampler,
            light_ubos,
            shadow_ubos,
        } = bindings;
        if sdf_volumes.is_empty() {
            return Ok(None);
        }

        let view_set_layout = create_descriptor_set_layout(device, &view_set_bindings())?;
        let volume_set_layout = create_descriptor_set_layout(device, &volume_set_bindings())?;
        let set_layouts = [view_set_layout.handle(), volume_set_layout.handle()];
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            device
                .create_pipeline_layout(&info)
                .map_err(|e| super::error::map_vk_result(e, "raymarch pipeline layout"))?
        };

        let (cube_vb, cube_ib) = build_cube_buffers(alloc)?;

        let snapshot = create_hdr_snapshot(alloc, device, command_pool, queue, width, height)?;

        // Per-frame view UBO ring (HOST_VISIBLE | HOST_COHERENT, mapped).
        let view_size = size_of::<RaymarchView>() as u64;
        let mut view_ubos = Vec::with_capacity(frames);
        for _ in 0..frames {
            view_ubos.push(alloc.create_buffer(
                view_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        let rings = ViewRings::of(sdf_volumes);
        let descriptor_pool = create_descriptor_pool(device, frames, sdf_volumes.len(), rings)?;
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout.handle()).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &view_layouts)?;
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                RaymarchViewSetBuffers {
                    view: view_ubos[i].buffer(),
                    light: light_ubos[i].buffer(),
                    shadow: shadow_ubos[i].buffer(),
                },
                RaymarchViewSetTextures {
                    shadow_map_view,
                    far_shadow_view,
                    shadow_sampler,
                    irradiance_view,
                    prefilter_view,
                    cube_sampler,
                    snapshot_view: snapshot.view,
                    scene_sampler: linear_sampler,
                },
            );
        }

        // Shadow and pre-pass infrastructure, built when a volume needs it: the 3-UBO
        // layout, a pipeline layout with `cascade_idx`, and a view ring per pass.
        let mut shadow_pipeline_layout = OwnedPipelineLayout::null();
        let mut shadow_view_set_layout = OwnedSetLayout::null();
        let mut shadow_view_ubos: Vec<PooledBuffer> = Vec::new();
        let mut shadow_view_sets: Vec<vk::DescriptorSet> = Vec::new();
        let mut prepass_view_ubos: Vec<PooledBuffer> = Vec::new();
        let mut prepass_view_sets: Vec<vk::DescriptorSet> = Vec::new();
        if rings.shadow || rings.prepass {
            shadow_view_set_layout =
                create_descriptor_set_layout(device, &shadow_view_set_bindings())?;
            let set_layouts = [shadow_view_set_layout.handle(), volume_set_layout.handle()];
            let push = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(u32::saturating_from(size_of::<RaymarchShadowCascade>()));
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push));
            shadow_pipeline_layout = device
                .create_pipeline_layout(&info)
                .map_err(|e| super::error::map_vk_result(e, "raymarch shadow pipeline layout"))?;
        }
        let three_ubo_ring =
            |ubos: &mut Vec<PooledBuffer>| -> RenderResult<Vec<vk::DescriptorSet>> {
                for _ in 0..frames {
                    ubos.push(alloc.create_buffer(
                        view_size,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?);
                }
                let layouts: Vec<_> = (0..frames)
                    .map(|_| shadow_view_set_layout.handle())
                    .collect();
                let sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &layouts)?;
                for (i, &set) in sets.iter().enumerate() {
                    write_shadow_view_set(
                        device,
                        set,
                        ubos[i].buffer(),
                        light_ubos[i].buffer(),
                        shadow_ubos[i].buffer(),
                    );
                }
                Ok(sets)
            };
        if rings.shadow {
            shadow_view_sets = three_ubo_ring(&mut shadow_view_ubos)?;
        }
        if rings.prepass {
            prepass_view_sets = three_ubo_ring(&mut prepass_view_ubos)?;
        }

        // Build per-volume records. A compile error in an active volume is a
        // developer-time bug, so it aborts init.
        let mut volumes: Vec<RaymarchVolumeRecord> = Vec::with_capacity(sdf_volumes.len());
        for SdfVolumeSource {
            volume: vol,
            fragment_source: payload,
            label,
        } in sdf_volumes
        {
            let programs = crate::shader::raymarch_source::decode(payload, label)
                .map_err(RenderError::Other)?;
            let flags = VolumeFlags::of(vol);
            let VolumePipelines {
                pipeline,
                front_pipeline,
                shadow_pipeline,
                prepass_pipelines,
            } = build_volume_pipelines(
                device,
                &VolumePipelineTargets {
                    layout: pipeline_layout.handle(),
                    shadow_layout: shadow_pipeline_layout.handle(),
                    msaa_samples,
                    hot_reload,
                },
                &programs,
                flags,
                label,
            )?;

            let uniforms = RaymarchVolumeUniforms::from(vol);
            let volume_ubo = alloc.create_buffer(
                size_of::<RaymarchVolumeUniforms>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            volume_ubo.write_val(0, &uniforms);
            let volume_set = alloc_descriptor_sets(
                device,
                descriptor_pool.handle(),
                &[volume_set_layout.handle()],
            )?[0];
            write_volume_set(device, volume_set, volume_ubo.buffer());

            volumes.push(RaymarchVolumeRecord {
                label: label.clone(),
                flags,
                pipeline,
                front_pipeline,
                shadow_pipeline,
                prepass_pipelines,
                center: vol.center,
                extent: vol.extent,
                _volume_ubo: volume_ubo,
                volume_set,
                visible: vol.visible,
                refractive: crate::shader::raymarch_source::taps_scene(&programs),
            });
        }

        Ok(Some(Self {
            pipeline_layout,
            _view_set_layout: view_set_layout,
            _volume_set_layout: volume_set_layout,
            _descriptor_pool: descriptor_pool,
            view_ubos,
            view_sets,
            cube_vb,
            cube_ib,
            snapshot,
            shadow_pipeline_layout,
            _shadow_view_set_layout: shadow_view_set_layout,
            shadow_view_ubos,
            shadow_view_sets,
            prepass_view_ubos,
            prepass_view_sets,
            volumes,
        }))
    }

    // The world box around every visible volume; `None` when none is visible.
    pub(in crate::vulkan) fn visible_bounds(
        &self,
    ) -> Option<concinnity_engine::gfx::frustum::Aabb> {
        concinnity_engine::render::sdf::volume::volume_bounds(
            self.volumes
                .iter()
                .filter(|v| v.visible)
                .map(|v| (v.center, v.extent)),
        )
    }

    // True when a drawn volume reads the scene behind its surface, which is
    // the only thing the per-frame snapshot copy feeds. A world of opaque
    // volumes skips the copy and the layout transitions that bracket it.
    // Whether any volume draws in `frustum`.
    pub(in crate::vulkan) fn draws_any(&self, frustum: &Frustum) -> bool {
        self.volumes.iter().any(|v| v.drawn(frustum))
    }

    fn any_refractive_drawn(&self, frustum: &Frustum) -> bool {
        self.volumes
            .iter()
            .any(|v| v.refractive && v.drawn(frustum))
    }

    // True when at least one visible volume opted into shadow casting (so its
    // `shadow_pipeline` was built). Gates the shadow-pass injection.
    fn any_shadow_casters(&self) -> bool {
        self.volumes
            .iter()
            .any(|v| v.visible && v.shadow_pipeline.is_some())
    }

    // Recreate the scene snapshot at new render dims + re-point the `scene_color`
    // binding of every view set. Called from the swapchain-resize handler; the
    // pipelines, layouts, UBOs and cube buffers all survive.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        ctx: RaymarchDeviceContext<'_>,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        let RaymarchDeviceContext {
            alloc,
            device,
            command_pool,
            queue,
        } = ctx;
        let old = std::mem::replace(
            &mut self.snapshot,
            create_hdr_snapshot(alloc, device, command_pool, queue, width, height)?,
        );
        drop(old);
        for &set in &self.view_sets {
            SetWrites::new(set)
                .sampled_image(6, self.snapshot.view)
                .apply(device);
        }
        Ok(())
    }

    // Re-point every view set's irradiance and prefilter cubes (bindings 4 and 5)
    // after an EnvironmentMap hot reload.
    pub(in crate::vulkan) fn rewire_ibl_cubes(
        &self,
        device: &VkDevice,
        irradiance_view: vk::ImageView,
        prefilter_view: vk::ImageView,
    ) {
        for &set in &self.view_sets {
            SetWrites::new(set)
                .sampled_image(4, irradiance_view)
                .sampled_image(5, prefilter_view)
                .apply(device);
        }
    }

    // Destroy every owned GPU resource.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.volumes.clear();
        self.view_ubos.clear();
        self.shadow_view_ubos.clear();
        self.prepass_view_ubos.clear();
        self.snapshot = GpuImage::null();
        self.cube_vb = PooledBuffer::null();
        self.cube_ib = PooledBuffer::null();
    }
}

impl VkContext {
    // Assemble the per-frame raymarch view from the frame's jittered VP (the
    // matrix the main pass rasterized the depth buffer with) + camera position.
    pub(in crate::vulkan) fn build_raymarch_view(
        &self,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
    ) -> RaymarchView {
        RaymarchView::new(&self.pass_camera(vp, cam_pos, time))
    }

    // The camera a screen-space pass reconstructs from: `vp` at the render
    // extent, with this frame's IBL and sky rotation.
    pub(in crate::vulkan) fn pass_camera(
        &self,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
    ) -> PassCamera {
        PassCamera::new(
            vp,
            mat4_inverse(vp),
            cam_pos,
            [
                self.targets.render_extent.width as f32,
                self.targets.render_extent.height as f32,
            ],
            time,
            self.pass_lighting(),
        )
    }

    // The frame's IBL and exposure, the same for every pass camera.
    fn pass_lighting(&self) -> PassLighting {
        PassLighting {
            prefilter_mip_count: self.scene.prefilter_mip_count as f32,
            irradiance: self.irradiance_lane(),
            sky_rot: self.state.view.sky_rot,
            pre_exposure: self.pre_exposure.value(),
        }
    }

    // Upload this frame's `view_time` into the shadow view ring so the shadow march
    // animates in step. A no-op without shadow casters.
    pub(in crate::vulkan) fn upload_raymarch_shadow_view(&self, frame_idx: usize, elapsed: f32) {
        let Some(rm) = self.raymarch.as_ref() else {
            return;
        };
        if !rm.any_shadow_casters() {
            return;
        }
        let Some(ubo) = rm.shadow_view_ubos.get(frame_idx) else {
            return;
        };
        // Only `time` is read by the shadow shaders; the rest is inert padding.
        let view = RaymarchView {
            vp: [[0.0; 4]; 4],
            inv_vp: [[0.0; 4]; 4],
            cam_pos: [0.0; 4],
            viewport: [0.0, 0.0],
            time: elapsed,
            prefilter_mip_count: 0.0,
            sky_rot: concinnity_engine::environment::sky::SkyOrientation::IDENTITY_ROWS,
            cur_vp: [[0.0; 4]; 4],
            prev_vp: [[0.0; 4]; 4],
            view_mat: [[0.0; 4]; 4],
            irradiance: 0.0,
            pre_exposure: 1.0,
            _pad: [0.0; 2],
        };
        ubo.write_val(0, &view);
    }

    // Draw the visible SDF shadow casters into one cascade, inside its depth-only
    // scope after the rasterized casters. A no-op without shadow casters.
    pub(in crate::vulkan) fn encode_sdf_shadow_cascade(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        cascade_idx: usize,
    ) {
        let Some(rm) = self.raymarch.as_ref() else {
            return;
        };
        if !rm.any_shadow_casters() || rm.shadow_view_sets.is_empty() {
            return;
        }
        let push = RaymarchShadowCascade {
            cascade_idx: u32::saturating_from(cascade_idx),
            _pad: [0; 3],
        };
        let layout = &rm.shadow_pipeline_layout;
        bind_cube(rec, rm);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            0,
            std::slice::from_ref(&rm.shadow_view_sets[frame_idx]),
            &[],
        );
        rec.push_constants(
            layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            &push,
        );
        for vol in &rm.volumes {
            let Some(shadow_pipeline) = vol.shadow_pipeline.as_ref() else {
                continue;
            };
            if !vol.visible {
                continue;
            }
            draw_volume(rec, layout, shadow_pipeline, vol.volume_set);
            self.inc_draw_calls(1);
        }
    }

    // Draw every visible surface volume into the G-buffer pre-pass the caller
    // began, under the same depth test as the meshes.
    pub(in crate::vulkan) fn encode_raymarch_prepass(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        prepass: &GbufferPrepassView<'_>,
        gbuffer: &GBufferView,
    ) {
        let Some(rm) = self.raymarch.as_ref() else {
            return;
        };
        let (Some(ubo), Some(&set)) = (
            rm.prepass_view_ubos.get(frame_idx),
            rm.prepass_view_sets.get(frame_idx),
        ) else {
            return;
        };
        let surfaces = || {
            rm.volumes
                .iter()
                .filter(|v| v.drawn(prepass.frustum))
                .filter_map(|v| v.prepass_pipelines.as_ref().map(|p| (v, p)))
        };
        if surfaces().next().is_none() {
            return;
        }
        let camera = self.pass_camera(prepass.jittered_vp, prepass.cam_pos, prepass.elapsed);
        let view = RaymarchView::for_gbuffer(&camera, gbuffer);
        ubo.write_val(0, &view);
        let draws = surfaces().map(|(v, p)| (p.get(v.faces(&view)), v.volume_set));
        let layout = &rm.shadow_pipeline_layout;
        let push = RaymarchShadowCascade {
            cascade_idx: 0,
            _pad: [0; 3],
        };
        bind_cube(rec, rm);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            0,
            std::slice::from_ref(&set),
            &[],
        );
        rec.push_constants(
            layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            &push,
        );
        for (pipeline, volume_set) in draws {
            draw_volume(rec, layout, pipeline, volume_set);
            self.inc_draw_calls(1);
        }
    }

    // Encode the raymarch pass between `AutoExposure` and `Decals`: snapshot the
    // scene for refraction, draw each volume's proxy back faces, and resolve into
    // hdr_resolve, left SHADER_READ_ONLY.
    pub(in crate::vulkan) fn encode_raymarch(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        view: &RaymarchView,
        frustum: &Frustum,
    ) -> RenderResult<()> {
        let Some(rm) = self.raymarch.as_ref() else {
            return Ok(());
        };
        if !rm.draws_any(frustum) {
            return Ok(());
        }
        let extent = self.targets.render_extent;
        let hdr_resolve = self
            .targets
            .hdr_resolve_images
            .get(frame_idx)
            .ok_or_else(|| RenderError::Other("raymarch: hdr_resolve index OOB".to_string()))?
            .image;
        let snapshot = rm.snapshot.image;
        // The snapshot feeds the scene tap and nothing else. Skipping it leaves
        // `snapshot` resting in SHADER_READ_ONLY, which is where its descriptor
        // already expects it, and leaves hdr_resolve untouched.
        let refract = rm.any_refractive_drawn(frustum);

        // Upload this frame's view.
        rm.view_ubos
            .get(frame_idx)
            .ok_or_else(|| RenderError::Other("raymarch: view_ubos index OOB".to_string()))?
            .write_val(0, view);

        // 1) Snapshot the resolved scene for the refraction tap.
        if refract {
            copy_scene_snapshot(rec, hdr_resolve, snapshot, extent);
        }

        // 2) Draw each visible volume over the scene color and depth, resolving
        // (MSAA) or storing (single-sample) into hdr_resolve, left sampled.
        // Negative-height viewport: matches the main pass so the proxy rasterizes
        // into identical pixels and the reprojected hit depth shares its space.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        // The scene was last sampled, or copied into the snapshot.
        let scene_from = if refract {
            ImageUse::TransferSrc
        } else {
            ImageUse::Sampled
        };
        let at = self.main_attachments(frame_idx, scene_from);
        main_pass::begin(rec, &at, MainOpen::Resume, MainClose::Sampled);
        rec.set_viewport(&vp);
        rec.set_scissor(&scissor);
        bind_cube(rec, rm);
        let layout = &rm.pipeline_layout;
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            0,
            std::slice::from_ref(&rm.view_sets[frame_idx]),
            &[],
        );
        for vol in &rm.volumes {
            if !vol.drawn(frustum) {
                continue;
            }
            draw_volume(rec, layout, vol.draw_pipeline(view), vol.volume_set);
            self.inc_draw_calls(1);
        }
        main_pass::end(rec, &at, MainClose::Sampled);
        Ok(())
    }
}
