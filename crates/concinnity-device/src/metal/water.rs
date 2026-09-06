// src/metal/water.rs
//
// Water is a producer for the engine's transparent pass (`PassId::Transparent`:
// after SsrResolve, before TaaResolve / Upscale). It contributes one
// `TransparentDraw` per `WaterSurface`; the shared `encode_transparent`
// encoder owns the render pass, the scene snapshot, and back-to-front sorting.
//
// For each surface the vertex shader displaces a flat tessellated quad by a sum
// of Gerstner waves; the fragment shader composites:
//   * Refraction: sample the pre-transparent scene snapshot at a
//     normal-perturbed screen UV.
//   * Tint: shallow to deep colour mix by water-column thickness derived from
//     the difference between the main depth and the water surface depth.
//   * Foam: a soft mask where the seabed is just below the surface.
//   * Reflection: the sharp planar reflection where the surface has one, else
//     the box-projected reflection-probe set, else the IBL prefilter cubemap,
//     else a hand-tuned sky gradient.
//   * Fresnel: Schlick-power mix of refraction-tinted vs. reflected colour.
// Output blends with SRC_ALPHA / ONE_MINUS_SRC_ALPHA into `scene_pre_taa`.
//
// The shaders are the shared `shaders/water.slang`, the single source all three
// backends compile; the pipeline state matches the glass panes exactly, because
// the same transparent encoder feeds both.
//
// Refraction samples `hdr_targets.transparent_scene_copy` (the snapshot the
// transparent encoder blits from the current scene-pre-taa before drawing) so
// water renders correctly whether or not SSR produced a distinct scene texture
// (with SSR off, scene-pre-taa aliases `hdr_resolve`, and sampling it directly
// would be reading the attachment being written).

#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLRenderPipelineState, MTLResourceOptions};

use crate::components::{MAX_WATER_WAVES, WaterSurface, WaterWave};
use crate::geometry::water_grid::build_water_grid;
use crate::gfx::mesh_payload::Vertex;

use super::context::MtlContext;
use super::glass::build_transparent_pipeline_stages;
use super::slang_shaders;
use super::transparent::{TransparentDraw, bytes_of};
use concinnity_core::render::uniforms::TransparentView;
use concinnity_core::render::uniforms::{WATER_MAX_WAVES, WaterParams, WaterWaveGpu};

// Per-surface GPU state: a static tessellated grid VB + IB.
pub(in crate::metal) struct WaterSurfaceRecord {
    pub(in crate::metal) vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_count: u32,
    pub(in crate::metal) params: WaterParams,
    pub(in crate::metal) visible: bool,
    // World-space centre, for the back-to-front camera-distance sort.
    pub(in crate::metal) centre: [f32; 3],
    // Planar reflection slot this surface samples (index into the
    // `PlanarReflectionSet`). `None` when the world has no planar set or this
    // surface's plane overflowed the budget; the shader then keeps the probe/sky
    // path. Assigned at init by `assign_planar_slots`.
    pub(in crate::metal) planar_slot: Option<usize>,
}

// Build a [`WaterSurfaceRecord`] for one `WaterSurface` asset. Calls the
// shared `geometry::water_grid` to produce the tessellated mesh and uploads
// it once; per-frame uniforms (time, view) come in through the encoder.
pub(in crate::metal) fn build_water_surface_record(
    device: &ProtocolObject<dyn MTLDevice>,
    surface: &WaterSurface,
) -> Result<WaterSurfaceRecord, String> {
    let (verts, idxs) =
        build_water_grid(surface.extent[0], surface.extent[1], surface.subdivisions)?;

    // Flatten into the standard Vertex layout. Tangent + colour are filled
    // with placeholders since the water shader rebuilds the normal frame
    // analytically and the fragment ignores per-vertex colour.
    let packed: Vec<Vertex> = verts
        .into_iter()
        .map(|(pos, normal, color, uv)| Vertex {
            pos,
            normal,
            tangent: [1.0, 0.0, 0.0],
            color,
            uv,
        })
        .collect();
    let vb_bytes = packed.len() * std::mem::size_of::<Vertex>();
    let ib_bytes = idxs.len() * std::mem::size_of::<u16>();

    // SAFETY: the pointer and length describe the live `packed` allocation, and Metal copies those
    // bytes into the new buffer before the call returns.
    let vb = unsafe {
        let ptr = std::ptr::NonNull::new(packed.as_ptr() as *mut _)
            .ok_or("water vertex buffer: source pointer is null")?;
        device
            .newBufferWithBytes_length_options(ptr, vb_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate water vertex buffer")?
    };
    // SAFETY: the pointer and length describe the live `idxs` allocation, and Metal copies those
    // bytes into the new buffer before the call returns.
    let ib = unsafe {
        let ptr = std::ptr::NonNull::new(idxs.as_ptr() as *mut _)
            .ok_or("water index buffer: source pointer is null")?;
        device
            .newBufferWithBytes_length_options(ptr, ib_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate water index buffer")?
    };

    Ok(WaterSurfaceRecord {
        vertex_buffer: vb,
        index_buffer: ib,
        index_count: idxs.len() as u32,
        params: water_params_from(surface),
        visible: surface.visible,
        centre: surface.centre,
        // Patched after `assign_planar_slots` runs over all reflectors in init.
        planar_slot: None,
    })
}

// Build the per-surface `WaterParams` from an authored surface. `planar` starts
// zeroed and `collect_water_transparent_draws` patches it on the frames the
// planar pass ran. Pure; unit tested. Mirrors `directx::water::water_params_from`.
fn water_params_from(surface: &WaterSurface) -> WaterParams {
    let mut waves = [WaterWaveGpu::default(); WATER_MAX_WAVES];
    for (slot, src) in waves.iter_mut().zip(surface.waves.iter()) {
        *slot = wave_to_gpu(src);
    }
    WaterParams {
        centre: [surface.centre[0], surface.centre[1], surface.centre[2], 0.0],
        deep_colour: [
            surface.deep_colour[0],
            surface.deep_colour[1],
            surface.deep_colour[2],
            0.0,
        ],
        shallow_colour: [
            surface.shallow_colour[0],
            surface.shallow_colour[1],
            surface.shallow_colour[2],
            0.0,
        ],
        depth_falloff: surface.depth_falloff_metres,
        foam_width: surface.foam_width_metres,
        foam_intensity: surface.foam_intensity,
        fresnel_power: surface.fresnel_power,
        roughness: surface.roughness,
        refraction_strength: surface.refraction_strength,
        wave_count: surface.waves.len().min(MAX_WATER_WAVES) as u32,
        _pad: 0.0,
        waves,
        planar: [0.0; 4],
    }
}

// The shader-side wave lane for one authored wave. Pure; unit tested.
fn wave_to_gpu(w: &WaterWave) -> WaterWaveGpu {
    WaterWaveGpu {
        dir_amp_wave: [w.direction[0], w.direction[1], w.amplitude, w.wavelength],
        speed_steep_pad: [w.speed, w.steepness, 0.0, 0.0],
    }
}

impl MtlContext {
    // True when a visible water surface holds a planar slot, so the mirror
    // re-render has a consumer this frame even while the trace is live. Water
    // takes the mirror over its own trace (see `water.slang`), so this is what
    // the planar gate reads; glass is deliberately not counted.
    pub(in crate::metal) fn water_planar_slot_live(&self) -> bool {
        self.water
            .surfaces
            .iter()
            .any(|s| s.visible && s.planar_slot.is_some())
    }

    // Contribute one [`TransparentDraw`] per visible water surface to the
    // transparent pass. The shared `encode_transparent` encoder owns the render
    // pass, the scene snapshot, back-to-front sorting, and the shared reflection
    // bindings (prefilter cube + probe cubes + probe set + cube sampler). Each
    // draw binds the snapshot (refraction source) at texture(0) and the resolved
    // main depth at texture(1). Sampling the snapshot rather than `hdr_resolve` is
    // what lets water render with SSR off.
    pub(in crate::metal) fn collect_water_transparent_draws(
        &self,
        view: &TransparentView,
        bindless: bool,
        planar_live: bool,
        out: &mut Vec<TransparentDraw>,
    ) {
        // Pipeline selection (matched by `encode_transparent`'s binding):
        //   RT on + bindless world  -> textured RT trace (bindless albedo)
        //   RT on                   -> flat RT trace (per-object tint)
        //   RT off                  -> box-projected probe cube / sky prefilter
        // `rt.accel` live means RT is on; `bindless` means the texture pool
        // exists. Falls back through to the probe pipeline. Either RT pipeline
        // still takes the planar mirror over its own trace where the surface has
        // a slot, so this picks the fragment, not the reflection source.
        let rt_on = self.rt.accel.is_some();
        let pipeline = match (
            rt_on && bindless,
            &self.water.pipeline_rt_textured,
            rt_on,
            &self.water.pipeline_rt,
        ) {
            (true, Some(p), _, _) => p,
            (_, _, true, Some(p)) => p,
            _ => match &self.water.pipeline {
                Some(p) => p,
                None => return,
            },
        };
        let cam = view.camera_pos;
        let planar_set = self.planar_reflection.as_ref();
        for surface in &self.water.surfaces {
            if !surface.visible {
                continue;
            }
            // Everything but the planar flag below is asset-side-static.
            let mut params = surface.params;
            let mut fragment_textures = vec![
                // The refraction snapshot (texture 0) + resolved main depth
                // (texture 1). The IBL prefilter cube (texture 2), the probe cube
                // argument buffer, cube sampler (sampler 1) and probe set are
                // bound globally by `encode_transparent` (shared with glass).
                (0, self.hdr_targets.transparent_scene_copy.clone()),
                (1, self.hdr_targets.depth_resolve.clone()),
            ];
            // Select the sharp planar reflection when the planar pass ran this
            // frame and this surface was assigned a slot; bind that slot's resolve
            // at the planar slot. Both fragments honour the flag, so this outranks the
            // trace as well. Otherwise the shader keeps the trace / probe / sky path.
            if planar_live
                && let Some(targets) = surface
                    .planar_slot
                    .and_then(|s| planar_set.and_then(|set| set.targets.get(s)))
            {
                params.planar = WaterParams::planar_lane(surface.params.roughness, true);
                fragment_textures.push((
                    super::transparent::GLASS_PLANAR_TEXTURE_INDEX,
                    targets.resolve.clone(),
                ));
            }
            let c = surface.centre;
            let sort_distance =
                ((c[0] - cam[0]).powi(2) + (c[1] - cam[1]).powi(2) + (c[2] - cam[2]).powi(2))
                    .sqrt();
            out.push(TransparentDraw {
                pipeline: pipeline.clone(),
                vertex_buffer: surface.vertex_buffer.clone(),
                index_buffer: surface.index_buffer.clone(),
                index_count: surface.index_count,
                index_type: objc2_metal::MTLIndexType::UInt16,
                index_offset_bytes: 0,
                base_vertex: 0,
                params: bytes_of(&params),
                fragment_textures,
                fragment_samplers: vec![(0, self.post_sampler.clone())],
                sort_distance,
            });
        }
    }
}

// Build the water render pipeline. Standard 5-attribute vertex layout at
// buffer(1); the same descriptor the glass panes and the main pass use, so any
// `ProceduralMesh::water_grid` mesh can bind directly. Output target is
// `scene_pre_taa` (RGBA16Float single-sample); SRC_ALPHA blend writes the
// transparent water on top of whatever the SsrResolve pass produced.
pub(super) fn build_water_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_water_pipeline_slang(device, hot_reload, &slang_shaders::WATER_FRAG)
}

// Build the ray-traced water pipeline: the same vertex layout + blend, but the
// `water_rt_fragment` variant traces a sharp reflection ray against the scene
// acceleration structure for surfaces with no mirror plane, instead of sampling
// a probe cube (one with a plane samples it either way). Built only on
// RT-capable devices (its metallib carries a real ray query); selected per-frame
// only while `self.rt.accel` is live, the probe pipeline otherwise. This is the
// FLAT variant (per-object material tint as albedo).
pub(super) fn build_water_pipeline_rt(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_water_pipeline_slang(device, hot_reload, &slang_shaders::WATER_FRAG_RT)
}

// Build the textured ray-traced water pipeline: the same trace as the flat RT
// variant, but the reflected hit's albedo / normal / emissive are sampled from
// the bindless texture pool (buffer 10) instead of a flat per-object tint.
// Selected over the flat variant only in a bindless world.
pub(super) fn build_water_pipeline_rt_textured(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_water_pipeline_slang(device, hot_reload, &slang_shaders::WATER_FRAG_RT_TEXTURED)
}

// The water pipelines, whose stages come from the single-source `water.slang`.
// Each fragment variant declares only the resources it binds, so each is its own
// metallib while the vertex is compiled once for all of them.
fn build_water_pipeline_slang(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
    fragment: &slang_shaders::SlangLib,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let vert_fn = slang_shaders::entry_function(device, &slang_shaders::WATER_VERT, hot_reload)?;
    let frag_fn = slang_shaders::entry_function(device, fragment, hot_reload)?;
    build_transparent_pipeline_stages(device, &vert_fn, &frag_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `WaterParams` / `WaterWaveGpu` layout tests live with the structs in
    // `concinnity_core::render::uniforms`, and are checked against the compiled shader
    // by `shader_layout`.

    #[test]
    fn wave_to_gpu_packs_the_lanes() {
        let w = WaterWave {
            amplitude: 0.25,
            wavelength: 3.0,
            speed: 1.5,
            direction: [0.6, -0.8],
            steepness: 0.4,
        };
        let g = wave_to_gpu(&w);
        assert_eq!(g.dir_amp_wave, [0.6, -0.8, 0.25, 3.0]);
        assert_eq!(g.speed_steep_pad, [1.5, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn water_params_from_maps_fields() {
        let surface = WaterSurface {
            centre: [1.0, 2.0, 3.0],
            deep_colour: [0.02, 0.05, 0.12],
            shallow_colour: [0.1, 0.3, 0.4],
            depth_falloff_metres: 3.0,
            foam_width_metres: 0.2,
            foam_intensity: 0.5,
            fresnel_power: 4.0,
            roughness: 0.08,
            refraction_strength: 0.05,
            waves: vec![WaterWave::default(), WaterWave::default()],
            ..Default::default()
        };
        let p = water_params_from(&surface);
        assert_eq!(p.centre, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.deep_colour, [0.02, 0.05, 0.12, 0.0]);
        assert_eq!(p.shallow_colour, [0.1, 0.3, 0.4, 0.0]);
        assert_eq!(p.depth_falloff, 3.0);
        assert_eq!(p.foam_width, 0.2);
        assert_eq!(p.foam_intensity, 0.5);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.roughness, 0.08);
        assert_eq!(p.refraction_strength, 0.05);
        assert_eq!(p.wave_count, 2);
        // Planar is off until the encoder sees a live planar pass.
        assert_eq!(p.planar, [0.0; 4]);
    }

    // More authored waves than the shader's array can hold must clamp rather
    // than overflow the fixed lane count.
    #[test]
    fn water_params_clamps_the_wave_count() {
        let surface = WaterSurface {
            waves: vec![WaterWave::default(); MAX_WATER_WAVES + 3],
            ..Default::default()
        };
        assert_eq!(
            water_params_from(&surface).wave_count,
            MAX_WATER_WAVES as u32
        );
    }
}
