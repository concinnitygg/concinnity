// The shapes of the graph's resources.

use super::inputs::FrameGraphInputs;
use crate::gfx::mipmap::full_mip_levels;
use crate::gfx::render_types::NUM_SHADOW_CASCADES;
use crate::render::render_graph::{
    BufferDesc, BufferUsage, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};
use concinnity_core::math::SaturatingFrom;

pub(super) fn atmosphere_luts_desc() -> TextureDesc {
    // Stands for the atmosphere's whole table set, whose largest member is the
    // aerial-perspective volume; the backend owns and transitions the tables.
    use crate::render::environment::atmosphere::AERIAL_SIZE;
    TextureDesc::volume_3d(
        TextureSize::Absolute(AERIAL_SIZE),
        TextureSize::Absolute(AERIAL_SIZE),
        AERIAL_SIZE,
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn cloud_target_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F hazed cloud light + transmittance at half the HDR dims;
    // backend-owned, so the graph only tracks the dependency.
    use crate::render::environment::clouds::cloud_target_size;
    let [w, h] = cloud_target_size(inputs.hdr_width, inputs.hdr_height);
    TextureDesc::texture_2d(
        TextureSize::Absolute(w),
        TextureSize::Absolute(h),
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn water_cascades_desc() -> TextureDesc {
    // Stands for both cascade arrays of every simulation, at the finest
    // quality; the backend owns them and transitions them itself.
    use crate::environment::water::ocean::OceanQuality;
    use crate::render::uniforms::ocean::ocean_mip_levels;
    let size = OceanQuality::FULL.size;
    TextureDesc::texture_2d(
        TextureSize::Absolute(size),
        TextureSize::Absolute(size),
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
    .with_array_layers(OceanQuality::FULL.cascades)
    .with_mip_levels(ocean_mip_levels(size))
}

pub(super) fn froxel_volume_desc() -> TextureDesc {
    // The volumetric-fog froxel volume: a 3D texture the fog kernel writes and
    // the fog pass samples. Its Z extent also rides in `FogFroxelParams.
    // froxel_dims` so shaders can map indices to volume UVs.
    TextureDesc::volume_3d(
        TextureSize::Absolute(FOG_FROXEL_X),
        TextureSize::Absolute(FOG_FROXEL_Y),
        FOG_FROXEL_Z,
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
}

/// X/Y/Z dimensions of the volumetric-fog froxel volume. Sized to keep the
/// per-frame compute cost modest (~230 k threads per dispatch) while
/// preserving enough screen-space detail for shaft-of-light shadowing.
/// Backends that implement the froxel path read these constants directly;
/// the values also ride in `FogFroxelParams.froxel_dims` so shaders can map
/// between absolute indices and normalized volume UVs without recompiling.
pub const FOG_FROXEL_X: u32 = 80;
/// Fog froxels down the screen. See [`FOG_FROXEL_X`].
pub const FOG_FROXEL_Y: u32 = 45;
/// Fog froxel depth slices. See [`FOG_FROXEL_X`].
pub const FOG_FROXEL_Z: u32 = 64;

pub(super) fn shadow_map_desc(size: u32) -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Absolute(size.max(1)),
        TextureSize::Absolute(size.max(1)),
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
    .with_array_layers(u32::saturating_from(NUM_SHADOW_CASCADES))
}

pub(super) fn far_shadow_map_desc(size: u32) -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Absolute(size.max(1)),
        TextureSize::Absolute(size.max(1)),
        PixelFormat::Depth16Unorm,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn spot_shadow_map_desc(slice_size: u32, slices: u32) -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Absolute(slice_size.max(1)),
        TextureSize::Absolute(slice_size.max(1)),
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
    .with_array_layers(slices.max(1))
}

pub(super) fn hdr_color_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET,
    )
    .with_sample_count(inputs.hdr_sample_count.max(1))
}

pub(super) fn hdr_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    render_res_2d(
        inputs,
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
    .with_sample_count(inputs.hdr_sample_count.max(1))
}

// A single-sample 2D target at the HDR render resolution, which is what most of
// the frame's off-screen targets are. Render resolution is not the drawable
// extent under temporal upscaling, so these are `Absolute` off the inputs
// rather than `Drawable`.
fn render_res_2d(
    inputs: &FrameGraphInputs,
    format: PixelFormat,
    usage: TextureUsage,
) -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Absolute(inputs.hdr_width.max(1)),
        TextureSize::Absolute(inputs.hdr_height.max(1)),
        format,
        usage,
    )
}

pub(super) fn draw_args_desc() -> BufferDesc {
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE.union(BufferUsage::INDIRECT),
    }
}

pub(super) fn grass_blades_desc() -> BufferDesc {
    // One `GpuGrassBlade` per visible blade. `STORAGE` alone, like the particle
    // pools: the kernel appends through an unordered-access binding and the
    // draws read it as a shader resource in their vertex stage. The executor
    // owns the allocation (sized to the field's blade capacity).
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

pub(super) fn grass_bend_desc() -> BufferDesc {
    // Two halves of packed bends, one per frame parity: the bend pass reads one
    // and writes the other through the same read-write binding, and the
    // placement kernels read both. The executor owns the allocation.
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

pub(super) fn cull_status_desc() -> BufferDesc {
    // One u32 per draw object: early cull writes drawn / hi-z-candidate /
    // culled, CullLate reads it. Both phases bind it the same read-write way, so it
    // never transitions to a read state and its ordering comes from an execution
    // barrier; `UNORDERED` is what says so. The executor owns the allocation
    // (sized to the live draw-object count).
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE.union(BufferUsage::UNORDERED),
    }
}

pub(super) fn particle_pool_desc() -> BufferDesc {
    // One `GpuParticle` slot per particle, per emitter. `STORAGE` alone, so the
    // class is `StorageBuffer`: the sim reads and writes the pool through a
    // storage / unordered-access binding, but the draw reads it as a plain
    // shader resource in its vertex stage, so the resource really does leave the
    // unordered-access state between the two nodes. `UnorderedBuffer` is for the
    // buffer both sides bind the same read-write way (`cull_status`), which
    // never transitions and orders through a UAV barrier instead; that would
    // leave DirectX's vertex-stage SRV read unsynchronized. The executor owns
    // the allocation (one buffer per live emitter, sized to its pool).
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

pub(super) fn cluster_lists_desc() -> BufferDesc {
    // Per-cluster light lists and probe masks LightCull writes and Main, the
    // reflection resolves and Transparent read. An identity stub: the backend
    // owns the real (persistent) buffer, so the graph only tracks the read/write
    // dependency, not the allocation.
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

pub(super) fn hiz_pyramid_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R32Float depth-mip pyramid rebuilt mid-frame from phase-1 depth, MAX
    // reduction. The cull kernel samples the coarse levels, so the chain is
    // most of the footprint and the desc carries its real length -- the same
    // `floor(log2(max)) + 1` each backend's Hi-Z build derives.
    render_res_2d(
        inputs,
        PixelFormat::R32Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
    .with_mip_levels(full_mip_levels(
        inputs.hdr_width.max(1),
        inputs.hdr_height.max(1),
    ))
}

// The unified G-buffer pre-pass writes four separate targets in one draw. They
// are four graph resources rather than one handle because their shapes differ
// (three color formats and a depth target) and so do their consumers, and a
// resource the aliaser may place has to name the memory it actually needs.
pub(super) fn gbuffer_normal_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F view-space normal + linear depth, read by SSR / SSAO / SSGI / RT.
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn gbuffer_roughness_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA8 perceptual roughness (r) + head-on specular reflectance (gba), read
    // by the reflection resolves for their gloss, blur radius and Fresnel.
    // Clears to roughness 1.0 (fully rough), so a pixel the pre-pass never
    // rasterizes reflects nothing -- the one graph target whose cleared
    // background carries meaning, and the reason `TextureDesc` models a clear
    // value at all.
    render_res_2d(
        inputs,
        PixelFormat::Rgba8Unorm,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
    .with_clear_color([1.0, 0.0, 0.0, 0.0])
}

pub(super) fn gbuffer_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The pre-pass's own depth attachment. Single-sample regardless of the
    // main pass's MSAA: the pre-pass rasterizes once.
    render_res_2d(
        inputs,
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn reactive_mask_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R8 at HDR dims: 0 keeps the temporal history, 1 discards it.
    render_res_2d(
        inputs,
        PixelFormat::R8Unorm,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn ao_output_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R8 occlusion at HDR dims; sampled by Main's ambient term.
    render_res_2d(
        inputs,
        PixelFormat::R8Unorm,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn ssgi_indirect_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F visibility + bounce at HDR dims; sampled by Main's ambient term.
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn velocity_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RG16F motion-vector buffer at HDR dims, sampled by TaaResolve.
    render_res_2d(
        inputs,
        PixelFormat::Rg16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(crate) fn bloom_top_desc() -> TextureDesc {
    // bloom_top is the bloom chain's half-resolution top octave; the prefilter
    // pass writes into it and the upsample chain accumulates back into it for
    // Composite to sample.
    //
    // Half the *drawable* extent, not half the render resolution: every backend
    // builds its bloom chain from the output extent, so under temporal
    // upscaling (where render resolution is smaller) an `hdr_width >> 1` desc
    // names a texture no backend creates.
    TextureDesc::texture_2d(
        TextureSize::DrawableScaled(0.5),
        TextureSize::DrawableScaled(0.5),
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn hdr_resolve_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn prop_probe_desc() -> TextureDesc {
    // Stands for the probe's cube face the pass renders, at the largest face
    // it may capture; the backend owns the capture and the cube.
    use crate::render::probe::prop_reflection_probe::MAX_PROP_PROBE_RESOLUTION;
    TextureDesc::texture_2d(
        TextureSize::Absolute(MAX_PROP_PROBE_RESOLUTION),
        TextureSize::Absolute(MAX_PROP_PROBE_RESOLUTION),
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn planar_mirrors_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // Every plane's mirror under one label, described at render resolution.
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(super) fn scene_color_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The engine-owned scene_color texture the post stack consumes is
    // single-sample at HDR dims regardless of whether the per-frame
    // resolution lands on the TAA history / the reflection composite's output /
    // hdr_resolve.
    render_res_2d(inputs, PixelFormat::Rgba16Float, TextureUsage::SHADER_READ)
}
