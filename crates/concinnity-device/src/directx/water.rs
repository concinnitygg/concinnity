//! `WaterSurface`: one producer of the engine's transparent pass on the D3D12
//! backend (`transparent.rs` owns the pass itself, the scene snapshot, the shared
//! root signatures and the combined back-to-front draw order; `glass.rs` is the
//! other producer). Each surface is a flat tessellated XZ grid built once at init
//! and displaced per frame by the vertex stage's Gerstner sum; the fragment
//! refracts the pass's scene snapshot, tints and foams it by the water-column
//! thickness the main depth gives, and mixes a reflection over it by a Schlick
//! Fresnel term.
//!
//! The shaders are the shared `shaders/water.hlsl`, compiled through
//! `builtin_shaders`; the ray-traced fragment needs shader model 6.5 for its
//! inline ray query, the base pair 6.0.

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::water::frame::WaterMotion;
use concinnity_engine::water::WaterBody;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::DeviceAllocator;
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::dump_on_err;
use crate::directx::transparent::{
    RecordUpload, ShapePsos, TransparentProducer, create_transparent_pso, upload_record,
};
use concinnity_engine::render::shader::programs::ShaderProgram;

// The programs one water shape draws with: a rectangle's, or a planet sea's.
struct WaterPrograms {
    vert: &'static ShaderProgram,
    frag: &'static ShaderProgram,
    frag_rt: &'static ShaderProgram,
    frag_rt_textured: &'static ShaderProgram,
}

const RECTANGLE: WaterPrograms = WaterPrograms {
    vert: &builtin_shaders::WATER_VERT,
    frag: &builtin_shaders::WATER_FRAG,
    frag_rt: &builtin_shaders::WATER_FRAG_RT,
    frag_rt_textured: &builtin_shaders::WATER_FRAG_RT_TEXTURED,
};

const SEA: WaterPrograms = WaterPrograms {
    vert: &builtin_shaders::WATER_SEA_VERT,
    frag: &builtin_shaders::WATER_SEA_FRAG,
    frag_rt: &builtin_shaders::WATER_SEA_FRAG_RT,
    frag_rt_textured: &builtin_shaders::WATER_SEA_FRAG_RT_TEXTURED,
};

// Compile a shape's vertex + fragment shaders. The fragment comes in an MSAA
// pair, which keeps its depth SRV declaration in sync with the resource's sample
// count; the vertex reads no depth and serves both pipelines. Used at init and
// by shader hot-reload.
fn compile_water_shaders(
    programs: &WaterPrograms,
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vs = programs.vert.compile(hot_reload)?;
    let ps = programs.frag.at(msaa_samples > 1).compile(hot_reload)?;
    Ok((vs, ps))
}

// Rebuild the water PSO against fresh shader source. Called from the DirectX
// shader hot-reload pass; the root signature is reused.
pub(in crate::directx) fn rebuild_water_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> RenderResult<ID3D12PipelineState> {
    let (vs, ps) = compile_water_shaders(&RECTANGLE, msaa_samples, hot_reload)?;
    dump_on_err(
        info_queue,
        create_transparent_pso(device, root_sig, &vs, &ps),
    )
}

// DXIL for the two RT water fragments, plus the vertex stage they share with the
// base pass (both root signatures put the transparent view CBV at b0 and the
// per-record params at b1).
struct WaterRtShaders {
    vs: Vec<u8>,
    flat_ps: Vec<u8>,
    textured_ps: Vec<u8>,
}

// Compile a shape's flat + textured ray-traced fragments (SM 6.5, for the
// inline ray query). Returns an `Err` (which the caller turns into a None RT
// pipeline + the base path) when dxc is unavailable or the shader fails to
// compile.
fn compile_water_rt_shaders(
    programs: &WaterPrograms,
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<WaterRtShaders> {
    let msaa = msaa_samples > 1;
    Ok(WaterRtShaders {
        vs: programs.vert.compile(hot_reload)?,
        flat_ps: programs.frag_rt.at(msaa).compile(hot_reload)?,
        textured_ps: programs.frag_rt_textured.at(msaa).compile(hot_reload)?,
    })
}

// What building the water producer needs from the pass that owns it: the
// allocator, the two shared root signatures (the RT one is `None` on a non-DXR
// GPU), and the render-state / hot-reload toggles.
#[derive(Clone, Copy)]
pub(in crate::directx) struct WaterBuild<'a> {
    pub alloc: &'a DeviceAllocator,
    pub root_sig: &'a ID3D12RootSignature,
    pub rt_root_sig: Option<&'a ID3D12RootSignature>,
    pub msaa_samples: u32,
    pub hot_reload: bool,
    pub info_queue: Option<&'a ID3D12InfoQueue>,
}

// Build the water pipelines and one record per authored surface. The RT pair is
// built whenever the pass has an RT root signature (regardless of whether RT is
// on at launch, so a live `quality-set ray_traced_reflections` selects it with no
// pipeline rebuild); a compile failure leaves it absent and the base
// probe/planar path runs.
pub(in crate::directx) fn build_water_producer(
    build: WaterBuild<'_>,
    surfaces: &[WaterBody],
    // Per-surface planar resolve slot (aligned with `surfaces`); `None` surfaces
    // keep the probe/sky reflection. From `assign_planar_slots`.
    planar_slots: &[Option<usize>],
) -> RenderResult<TransparentProducer> {
    let alloc = build.alloc;
    let rectangle = build.shape(&RECTANGLE)?;
    let sea = surfaces
        .iter()
        .any(|b| b.sea.is_some())
        .then(|| build.shape(&SEA))
        .transpose()?;

    let mut records = Vec::with_capacity(surfaces.len());
    let mut water = Vec::with_capacity(surfaces.len());
    for (i, body) in surfaces.iter().enumerate() {
        let surface = &body.surface;
        let planar_slot = planar_slots.get(i).copied().flatten();
        let (vertices, indices) = body.mesh().map_err(RenderError::Other)?;
        let motion = WaterMotion::new(body, planar_slot.is_some());
        records.push(upload_record(
            alloc,
            RecordUpload {
                vertices: &vertices,
                indices: &indices,
                params: bytemuck::bytes_of(motion.params()),
                visible: surface.visible,
                center: body.sort_center(),
                planar_slot,
            },
        )?);
        water.push(motion);
    }

    Ok(TransparentProducer {
        shape: rectangle,
        reflection_flat: None,
        reflection_textured: None,
        sea,
        records,
        water,
    })
}

impl WaterBuild<'_> {
    // A shape's PSOs. The RT pair is built whenever the pass has an RT root
    // signature; a compile failure leaves it absent and the base probe /
    // planar path runs.
    fn shape(&self, programs: &WaterPrograms) -> RenderResult<ShapePsos> {
        let (vs, ps) = compile_water_shaders(programs, self.msaa_samples, self.hot_reload)?;
        let pso = dump_on_err(
            self.info_queue,
            create_transparent_pso(self.alloc.device(), self.root_sig, &vs, &ps),
        )?;
        let (flat_rt_pso, textured_rt_pso) = match self.rt_root_sig {
            Some(sig) => match self.shape_rt(programs, sig) {
                Ok(pair) => (Some(pair.0), Some(pair.1)),
                Err(e) => {
                    tracing::warn!(
                        "water RT reflection pipeline build failed ({e}); \
                         using the probe/planar water path"
                    );
                    (None, None)
                }
            },
            None => (None, None),
        };
        Ok(ShapePsos {
            base: pso,
            flat_rt: flat_rt_pso,
            textured_rt: textured_rt_pso,
        })
    }

    // Compile and build a shape's flat + textured RT PSOs against the pass's
    // RT root signature. Both use the same render state as the base PSO.
    fn shape_rt(
        &self,
        programs: &WaterPrograms,
        rt_root_sig: &ID3D12RootSignature,
    ) -> RenderResult<(ID3D12PipelineState, ID3D12PipelineState)> {
        let shaders = compile_water_rt_shaders(programs, self.msaa_samples, self.hot_reload)?;
        let flat = dump_on_err(
            self.info_queue,
            create_transparent_pso(
                self.alloc.device(),
                rt_root_sig,
                &shaders.vs,
                &shaders.flat_ps,
            ),
        )?;
        let textured = dump_on_err(
            self.info_queue,
            create_transparent_pso(
                self.alloc.device(),
                rt_root_sig,
                &shaders.vs,
                &shaders.textured_ps,
            ),
        )?;
        Ok((flat, textured))
    }
}

#[cfg(test)]
mod tests {
    // The `WaterParams` / `WaterWaveGpu` layout tests live with the structs in
    // `concinnity_engine::render::uniforms`, and are checked against the compiled shader
    // by `shader_layout`.

    // The water shaders compile at runtime from the shared single source, so a
    // syntax or register error in either MSAA variant would otherwise surface
    // only as an init failure on a GPU host.
    #[test]
    fn water_shaders_compile() {
        concinnity_shader::require_dxc!();
        for programs in [&super::RECTANGLE, &super::SEA] {
            for msaa in [1u32, 4] {
                super::compile_water_shaders(programs, msaa, false)
                    .unwrap_or_else(|e| panic!("water shaders (msaa={msaa}) must compile: {e}"));
            }
        }
    }

    // The same for the ray-traced pair, which additionally exercises the shared
    // traversal fragment and the shader model 6.5 the ray query needs.
    #[test]
    fn water_rt_shaders_compile() {
        concinnity_shader::require_dxc!();
        for programs in [&super::RECTANGLE, &super::SEA] {
            for msaa in [1u32, 4] {
                super::compile_water_rt_shaders(programs, msaa, false)
                    .unwrap_or_else(|e| panic!("water_rt shaders (msaa={msaa}) must compile: {e}"));
            }
        }
    }
}
