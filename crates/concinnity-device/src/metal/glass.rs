//! GlassPanel: the simplest producer for the engine's transparent pass. Each
//! panel is a flat world-space quad (built once at init) that contributes one
//! [`TransparentDraw`] per frame. The shared `encode_transparent` encoder sorts
//! it back-to-front against water + other panels and draws it; the fragment
//! shader refracts the pre-transparent scene snapshot, tints it, and adds a
//! Fresnel rim (see shaders/glass.hlsl).

#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::components::GlassPanel;
use concinnity_core::geometry::glass_quad::build_glass_quad;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::transparent;
use concinnity_core::render::uniforms::{GlassMeshParams, GlassParams, TransparentView};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLBuffer, MTLDepthStencilState, MTLDevice, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLTexture,
    MTLTextureUsage, MTLVertexFormat, MTLVertexStepFunction,
};

use super::allocator::{DeviceAllocator, PooledTexture};
use super::builtin_shaders;
use super::context::MtlContext;
use super::descriptors::{TextureDesc, VertexAttr, VertexLayout, vertex_descriptor};
use super::error::allocation_failed;
use super::init::pipelines::make_depth_state;
use super::texture::upload_texture;
use super::transparent::{TransparentDraw, bytes_of};

// Refraction offset + Fresnel falloff for a transparent glass MESH. A `Material`
// carries no glass-specific tunables (unlike a `GlassPanel`), so these match the
// GlassPanel defaults: a gentle screen-space refraction and a fresnel power of 1
// (subtle reflection head-on, full mirror at grazing).
const GLASS_MESH_REFRACTION: f32 = 0.02;
const GLASS_MESH_FRESNEL_POWER: f32 = 1.0;

// Per-panel GPU state: the static world-space quad VB + IB plus the per-panel
// uniform block. The quad is pre-transformed at build time, so there is no
// per-frame vertex work beyond projection.
pub(in crate::metal) struct GlassPanelRecord {
    pub(in crate::metal) vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_count: u32,
    pub(in crate::metal) params: GlassParams,
    pub(in crate::metal) visible: bool,
    // World-space center, used for the back-to-front camera-distance sort.
    pub(in crate::metal) center: [f32; 3],
    // Planar reflection slot this pane samples (index into the
    // `PlanarReflectionSet`). `None` when the world has no planar set or this
    // pane's plane overflowed the budget; the shader then keeps the probe/sky
    // path. Assigned at init by `assign_planar_slots`.
    pub(in crate::metal) planar_slot: Option<usize>,
}

fn glass_params_from(panel: &GlassPanel) -> GlassParams {
    let n = panel.normal; // already unit-length from GlassPanel::from_args
    GlassParams {
        center: [panel.center[0], panel.center[1], panel.center[2], 0.0],
        normal: [n[0], n[1], n[2], 0.0],
        tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
        opacity: panel.opacity,
        refraction_strength: panel.refraction_strength,
        fresnel_power: panel.fresnel_power,
        // Off by default; `collect_glass_transparent_draws` sets it when the
        // planar pass ran this frame and the pane has a slot.
        planar: 0.0,
    }
}

// Build the GPU record for one `GlassPanel`: generate the quad, upload it, and
// snapshot the per-panel uniforms.
pub(in crate::metal) fn build_glass_panel_record(
    device: &ProtocolObject<dyn MTLDevice>,
    panel: &GlassPanel,
) -> RenderResult<GlassPanelRecord> {
    let (verts, idxs) = build_glass_quad(panel.center, panel.normal, panel.half_size);

    // Flatten into the standard Vertex layout. Tangent is a placeholder (the
    // glass shader rebuilds its frame from the panel normal) and per-vertex
    // color is unused.
    let mut packed: Vec<Vertex> = Vec::with_capacity(verts.len());
    for (pos, normal, color, uv) in verts {
        packed.push(Vertex {
            pos,
            normal,
            tangent: [1.0, 0.0, 0.0],
            color,
            uv,
        });
    }
    let vb_bytes = packed.len() * std::mem::size_of::<Vertex>();
    let ib_bytes = idxs.len() * std::mem::size_of::<u16>();

    // SAFETY: the pointer and length describe the live `packed` allocation, and Metal copies those
    // bytes into the new buffer before the call returns.
    let vb = unsafe {
        let ptr = std::ptr::NonNull::new(packed.as_ptr() as *mut _).ok_or_else(|| {
            RenderError::Other("glass vertex buffer: source pointer is null".into())
        })?;
        device
            .newBufferWithBytes_length_options(ptr, vb_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| allocation_failed("the glass vertex buffer"))?
    };
    // SAFETY: the pointer and length describe the live `idxs` allocation, and Metal copies those
    // bytes into the new buffer before the call returns.
    let ib = unsafe {
        let ptr = std::ptr::NonNull::new(idxs.as_ptr() as *mut _).ok_or_else(|| {
            RenderError::Other("glass index buffer: source pointer is null".into())
        })?;
        device
            .newBufferWithBytes_length_options(ptr, ib_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| allocation_failed("the glass index buffer"))?
    };

    Ok(GlassPanelRecord {
        vertex_buffer: vb,
        index_buffer: ib,
        index_count: idxs.len() as u32,
        params: glass_params_from(panel),
        visible: panel.visible,
        center: panel.center,
        // Patched after `assign_planar_slots` runs over all reflectors in init.
        planar_slot: None,
    })
}

// Build the shared glass render pipeline. Standard 5-attribute vertex layout
// at buffer(1) (same as water + the main pass); SRC_ALPHA blend into the
// RGBA16Float scene-pre-taa target, no depth attachment.
pub(super) fn build_glass_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    build_glass_pipeline_with(device, hot_reload, &builtin_shaders::GLASS_FRAG)
}

// A ray-traced glass pipeline and its reduced reflection pre-pass: `shade`
// draws the glass into the scene, and `reflection` traces into the reduced
// target `shade` reads its reflection back from when the trace is scaled down.
pub(in crate::metal) struct TracedGlassPipelines {
    pub(in crate::metal) shade: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(in crate::metal) reflection: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

// Build the ray-traced glass pipelines: the same vertex layout + blend, but the
// `glass_rt_fragment` variant traces a sharp reflection ray against the scene
// acceleration structure instead of sampling a probe cube. Built only on
// RT-capable devices (its metallib carries a real ray query); selected
// per-frame only while `self.rt.accel` is live, the probe pipeline otherwise.
pub(super) fn build_glass_pipeline_rt(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<TracedGlassPipelines> {
    build_traced_glass_pipelines(
        device,
        hot_reload,
        &builtin_shaders::GLASS_VERT,
        &builtin_shaders::GLASS_FRAG_RT,
        &builtin_shaders::GLASS_REFLECTION_FRAG,
    )
}

// Build the textured ray-traced glass pipelines: the same trace as the flat RT
// variant, but the reflected hit's albedo / normal / emissive are sampled from
// the bindless texture pool (buffer 10) instead of a flat per-object tint.
// Selected over the flat variant only in a bindless world.
pub(super) fn build_glass_pipeline_rt_textured(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<TracedGlassPipelines> {
    build_traced_glass_pipelines(
        device,
        hot_reload,
        &builtin_shaders::GLASS_VERT,
        &builtin_shaders::GLASS_FRAG_RT_TEXTURED,
        &builtin_shaders::GLASS_REFLECTION_FRAG_TEXTURED,
    )
}

// Build the ray-traced see-through glass MESH pipelines: the same 5-attribute
// vertex layout + blend as the pane pipelines, but the `glass_mesh_vertex` stage
// applies a per-draw model matrix and the fragment shades off the interpolated
// mesh normal. Compiled only on RT-capable devices. Drives the FLAT trace
// (reflected-hit material tint as albedo).
pub(super) fn build_glass_mesh_pipeline_rt(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<TracedGlassPipelines> {
    build_traced_glass_pipelines(
        device,
        hot_reload,
        &builtin_shaders::GLASS_MESH_VERT,
        &builtin_shaders::GLASS_MESH_FRAG_RT,
        &builtin_shaders::GLASS_MESH_REFLECTION_FRAG,
    )
}

// The textured see-through glass MESH variant: reflected hits sample the bindless
// pool (buffer 10). Selected over the flat variant only in a bindless world.
pub(super) fn build_glass_mesh_pipeline_rt_textured(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<TracedGlassPipelines> {
    build_traced_glass_pipelines(
        device,
        hot_reload,
        &builtin_shaders::GLASS_MESH_VERT,
        &builtin_shaders::GLASS_MESH_FRAG_RT_TEXTURED,
        &builtin_shaders::GLASS_MESH_REFLECTION_FRAG_TEXTURED,
    )
}

// The probe-path pane pipeline, whose stages come from the single-source
// `glass.hlsl`. Each fragment variant declares only the resources it binds, so
// each is its own metallib while the vertex is compiled once for all of them.
fn build_glass_pipeline_with(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
    fragment: &builtin_shaders::ShaderProgram,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert_fn =
        builtin_shaders::entry_function(device, &builtin_shaders::GLASS_VERT, hot_reload)?;
    let frag_fn = builtin_shaders::entry_function(device, fragment, hot_reload)?;
    build_transparent_pipeline_stages(device, &vert_fn, &frag_fn)
}

// A traced pair over one vertex stage: `shade` blends into the scene, and
// `reflection` overwrites the reduced reflection target.
fn build_traced_glass_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
    vertex: &builtin_shaders::ShaderProgram,
    shade: &builtin_shaders::ShaderProgram,
    reflection: &builtin_shaders::ShaderProgram,
) -> RenderResult<TracedGlassPipelines> {
    let vert_fn = builtin_shaders::entry_function(device, vertex, hot_reload)?;
    let shade_fn = builtin_shaders::entry_function(device, shade, hot_reload)?;
    let reflection_fn = builtin_shaders::entry_function(device, reflection, hot_reload)?;
    Ok(TracedGlassPipelines {
        shade: build_transparent_pipeline_stages(device, &vert_fn, &shade_fn)?,
        reflection: transparent_pipeline(
            device,
            &vert_fn,
            &reflection_fn,
            TransparentOutput::ReflectionLayer,
        )?,
    })
}

// The reduced glass reflection pre-pass targets: two layers (rgb the traced
// reflection, a the surface's distance from the camera) over one depth
// attachment each layer clears and tests against, plus the empty layer the
// first one peels behind.
pub(in crate::metal) struct GlassReflectionTargets {
    pub(in crate::metal) layers: [Retained<ProtocolObject<dyn MTLTexture>>; 2],
    pub(in crate::metal) depth: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(in crate::metal) empty: PooledTexture,
    pub(in crate::metal) depth_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
}

impl GlassReflectionTargets {
    fn new(alloc: &DeviceAllocator, width: u32, height: u32) -> RenderResult<Self> {
        let device = alloc.device();
        let layer = || {
            reduced_target(device, MTLPixelFormat::RGBA16Float, width, height)
                .ok_or_else(|| allocation_failed("a glass reflection layer"))
        };
        Ok(Self {
            layers: [layer()?, layer()?],
            depth: reduced_target(device, MTLPixelFormat::Depth32Float, width, height)
                .ok_or_else(|| allocation_failed("the glass reflection depth"))?,
            empty: upload_texture(alloc, 1, 1, &[0u8; 4])?,
            depth_state: make_depth_state(device)?,
        })
    }

    fn extent(&self) -> (u32, u32) {
        (self.depth.width() as u32, self.depth.height() as u32)
    }
}

fn reduced_target(
    device: &ProtocolObject<dyn MTLDevice>,
    format: MTLPixelFormat,
    width: u32,
    height: u32,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = TextureDesc {
        format,
        width: width as usize,
        height: height as usize,
        usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0),
        ..Default::default()
    }
    .build();
    device.newTextureWithDescriptor(&desc)
}

// Shared descriptor for every transparent-pass pipeline (glass panes, glass
// meshes, water surfaces): the standard 5-attribute vertex layout at buffer(1)
// and straight-alpha blending into the RGBA16Float scene target, with no depth
// attachment.
pub(in crate::metal) fn build_transparent_pipeline_stages(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_fn: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    frag_fn: &ProtocolObject<dyn objc2_metal::MTLFunction>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    transparent_pipeline(device, vert_fn, frag_fn, TransparentOutput::Scene)
}

// What a transparent-pass pipeline draws into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransparentOutput {
    // Straight-alpha blended over the scene, with no depth attachment.
    Scene,
    // Overwriting a glass reflection layer, depth-tested against its
    // `Depth32Float` attachment.
    ReflectionLayer,
}

// The transparent-pass pipeline over the shared vertex layout.
fn transparent_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_fn: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    frag_fn: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    output: TransparentOutput,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert_desc = vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: 1,
            },
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float3,
                offset: 12,
                buffer_index: 1,
            },
            VertexAttr {
                index: 2,
                format: MTLVertexFormat::Float3,
                offset: 24,
                buffer_index: 1,
            },
            VertexAttr {
                index: 3,
                format: MTLVertexFormat::Float3,
                offset: 36,
                buffer_index: 1,
            },
            VertexAttr {
                index: 4,
                format: MTLVertexFormat::Float2,
                offset: 48,
                buffer_index: 1,
            },
        ],
        &[VertexLayout {
            buffer_index: 1,
            stride: std::mem::size_of::<Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    );

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(vert_fn));
    desc.setFragmentFunction(Some(frag_fn));
    desc.setRasterSampleCount(1);
    let blend = output == TransparentOutput::Scene;
    if !blend {
        desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    }
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca.setBlendingEnabled(blend);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("transparent pipeline state: {e:?}")))
}

impl MtlContext {
    // Contribute one [`TransparentDraw`] per visible glass panel. The shared
    // transparent encoder owns sorting + the scene-copy snapshot; each draw
    // binds the snapshot (refraction source) at texture(0) and the resolved
    // depth at texture(1). When `planar_live` and the pane has a planar slot, the
    // draw also binds its slot's resolve at texture(11) and flips `params.planar`
    // so the shader samples the sharp planar reflection instead of the probe cube;
    // slotless panes (budget overflow, logged at init) keep the probe path.
    pub(in crate::metal) fn collect_glass_transparent_draws(
        &self,
        view: &TransparentView,
        bindless: bool,
        planar_live: bool,
        out: &mut Vec<TransparentDraw>,
    ) {
        // Pipeline selection (matched by `encode_transparent`'s binding):
        //   RT on + bindless world  -> textured RT trace (bindless albedo)
        //   RT on                   -> flat RT trace (per-object tint)
        //   RT off                  -> box-projected probe cube
        // `rt.accel` live means RT is on; `bindless` means the texture pool
        // exists. Falls back through to the probe pipeline.
        let rt_on = self.rt.accel.is_some();
        let traced = match (
            rt_on && bindless,
            &self.glass.pipeline_rt_textured,
            rt_on,
            &self.glass.pipeline_rt,
        ) {
            (true, Some(p), _, _) => Some(p),
            (_, _, true, Some(p)) => Some(p),
            _ => None,
        };
        let (pipeline, reflection_pipeline) = match traced {
            Some(t) => (&t.shade, self.glass_reflection_pipeline(t)),
            None => match &self.glass.pipeline {
                Some(p) => (p, None),
                None => return,
            },
        };
        let cam = view.camera_pos;
        let planar_set = self.planar_reflection.as_ref();
        for panel in &self.glass.panels {
            if !panel.visible {
                continue;
            }
            // The live prefilter mip count (0 = no env map -> white rim) rides
            // the shared view; the reflection-probe cubes + set are bound
            // globally by `encode_transparent`.
            let mut params = panel.params;
            let mut fragment_textures = vec![
                (0, self.targets.hdr.transparent_scene_copy.clone()),
                (1, self.targets.hdr.depth_resolve.clone()),
            ];
            // Select the sharp planar reflection when the planar pass ran this
            // frame and this pane was assigned a slot; bind that slot's resolve at
            // the planar slot (overriding the global default). Otherwise the shader
            // keeps the probe / sky path.
            if planar_live
                && let Some(targets) = panel
                    .planar_slot
                    .and_then(|s| planar_set.and_then(|set| set.targets.get(s)))
            {
                params.planar = 1.0;
                fragment_textures.push((
                    super::transparent::GLASS_PLANAR_TEXTURE_INDEX,
                    targets.resolve.clone(),
                ));
            }
            let c = panel.center;
            let sort_distance = transparent::sort_distance(c, [cam[0], cam[1], cam[2]]);
            out.push(TransparentDraw {
                pipeline: pipeline.clone(),
                reflection_pipeline: reflection_pipeline.clone(),
                vertex_buffer: panel.vertex_buffer.clone(),
                index_buffer: panel.index_buffer.clone(),
                index_count: panel.index_count,
                index_type: objc2_metal::MTLIndexType::UInt16,
                index_offset_bytes: 0,
                base_vertex: 0,
                params: bytes_of(&params),
                fragment_textures,
                fragment_samplers: vec![(0, self.composite.sampler.clone())],
                sort_distance,
            });
        }
    }

    // Keep the reduced glass reflection targets matched to the RT trace divisor
    // and the render size: present only while glass can trace (RT live, a
    // traced glass pipeline built) at a divisor above 1. Cheap when nothing
    // changed, so it runs every frame and follows resizes and live quality
    // changes alike.
    pub(in crate::metal) fn sync_glass_reflection_target(
        &mut self,
        render_w: u32,
        render_h: u32,
    ) -> RenderResult<()> {
        let traced = self.glass.pipeline_rt.is_some() || self.glass.mesh_pipeline_rt.is_some();
        let extent = self
            .rt
            .settings
            .filter(|s| s.divisor > 1 && traced && self.rt.accel.is_some())
            .map(|s| s.trace_extent(render_w, render_h));
        let current = self
            .glass
            .reflection_targets
            .as_ref()
            .map(GlassReflectionTargets::extent);
        if extent == current {
            return Ok(());
        }
        self.glass.reflection_targets = match extent {
            Some((w, h)) => Some(GlassReflectionTargets::new(&self.hw.allocator, w, h)?),
            None => None,
        };
        Ok(())
    }

    // The reduced reflection pre-pass pipeline for a traced glass draw, or
    // `None` while glass traces in place (no reduced targets this frame).
    fn glass_reflection_pipeline(
        &self,
        traced: &TracedGlassPipelines,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        self.glass
            .reflection_targets
            .as_ref()
            .map(|_| traced.reflection.clone())
    }

    // Whether any material opted into Layer 2 see-through glass AND the device can
    // drive it (the mesh pipeline built). Independent of `rt.accel`, so it answers
    // "would the see-through path run if RT is on" -- used at the RT-BLAS build,
    // which must exclude the meshes it will reroute before `rt.accel` itself is
    // assigned. Data-driven: see-through is opt-in per `Material::see_through`, so
    // a scene with no see-through material (e.g. Bistro) never engages Layer 2 and
    // its transparent glass stays Layer 1 (opaque, low roughness, reflective).
    pub(in crate::metal) fn seethrough_meshes_enabled(&self) -> bool {
        !self.glass.seethrough_mesh_indices.is_empty() && self.glass.mesh_pipeline_rt.is_some()
    }

    // Whether the transparent-mesh (Layer 2) path is live this frame: a material
    // opted into see-through, the mesh pipeline was built, AND RT is on
    // (`rt.accel`, the per-pixel trace needs the BVH). When false, those meshes
    // render opaque + reflective in the main pass (Layer 1) and the producer /
    // opaque-skip / BLAS-exclude all stay inert.
    pub(in crate::metal) fn mesh_glass_active(&self) -> bool {
        self.seethrough_meshes_enabled() && self.rt.accel.is_some()
    }

    // Whether a see-through mesh actually draws this frame. Panes and water
    // answer this from their static records; a mesh's visibility lives in
    // `draw.objects`, so the graph gate has to ask separately or a world whose
    // only translucent producer is a mesh gets no transparent pass and the mesh
    // disappears (it is already skipped in the opaque pass and the BLAS).
    // Mirrors `DxContext::mesh_glass_visible`.
    pub(in crate::metal) fn mesh_glass_visible(&self) -> bool {
        self.mesh_glass_active()
            && self.glass.seethrough_mesh_indices.iter().any(|&i| {
                self.draw
                    .objects
                    .get(i)
                    .is_some_and(|o| o.visible && o.resident)
            })
    }

    // Contribute one [`TransparentDraw`] per visible see-through glass MESH (Layer
    // 2): a `Material` flagged `see_through` (which implies `transparent`) on an
    // RT-capable device. Each mesh draws from the SHARED scene vertex/index buffers
    // via its `DrawObject` offsets + model matrix; `glass_mesh.hlsl` traces
    // a per-pixel reflection off the interpolated mesh normal. A no-op unless RT is
    // live (`mesh_glass_active`); when inactive the meshes render opaque (Layer 1)
    // in the main pass. The same gate skips them in the opaque pass + the RT BLAS,
    // so a mesh neither double-draws nor self-reflects.
    pub(in crate::metal) fn collect_mesh_transparent_draws(
        &self,
        view: &TransparentView,
        bindless: bool,
        out: &mut Vec<TransparentDraw>,
    ) {
        if !self.mesh_glass_active() {
            return;
        }
        // Textured trace in a bindless world (reflected hits carry their textures),
        // else the flat trace (reflected-hit material tint). `mesh_glass_active`
        // guarantees the flat pipeline exists.
        let traced = match (bindless, &self.glass.mesh_pipeline_rt_textured) {
            (true, Some(p)) => p,
            _ => match &self.glass.mesh_pipeline_rt {
                Some(p) => p,
                None => return,
            },
        };
        let reflection_pipeline = self.glass_reflection_pipeline(traced);
        let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;
        let cam = view.camera_pos;
        for &idx in &self.glass.seethrough_mesh_indices {
            let Some(obj) = self.draw.objects.get(idx) else {
                continue;
            };
            if !obj.visible || !obj.resident {
                continue;
            }
            let center = [
                0.5 * (obj.bb_min[0] + obj.bb_max[0]),
                0.5 * (obj.bb_min[1] + obj.bb_max[1]),
                0.5 * (obj.bb_min[2] + obj.bb_max[2]),
            ];
            let d = ((center[0] - cam[0]).powi(2)
                + (center[1] - cam[1]).powi(2)
                + (center[2] - cam[2]).powi(2))
            .sqrt();
            let (index_offset, index_count) = obj.active_lod(d);
            let t = obj.material.tint;
            let params = GlassMeshParams {
                model: obj.model,
                tint: [t[0], t[1], t[2], 0.0],
                opacity: obj.material.opacity,
                refraction_strength: GLASS_MESH_REFRACTION,
                fresnel_power: GLASS_MESH_FRESNEL_POWER,
                prefilter_mip_count,
            };
            out.push(TransparentDraw {
                pipeline: traced.shade.clone(),
                reflection_pipeline: reflection_pipeline.clone(),
                vertex_buffer: self.scene.vertex_buffer.retained(),
                index_buffer: self.scene.index_buffer.retained(),
                index_count: index_count as u32,
                index_type: objc2_metal::MTLIndexType::UInt32,
                index_offset_bytes: index_offset * std::mem::size_of::<u32>(),
                base_vertex: obj.base_vertex,
                params: bytes_of(&params),
                fragment_textures: vec![
                    (0, self.targets.hdr.transparent_scene_copy.clone()),
                    (1, self.targets.hdr.depth_resolve.clone()),
                ],
                fragment_samplers: vec![(0, self.composite.sampler.clone())],
                sort_distance: d,
            });
        }
    }
}
