//! The grass field on DirectX 12 (see `concinnity_engine::render::environment::grass`): the
//! bend pass that relaxes and stamps the trampling field, the kernel that
//! places this frame's visible blades and, a second time, the nearest shadow
//! cascade's, the indirect draws, one per detail level, that render them at the
//! tail of the G-buffer pre-pass and the main pass, and the cascade's depth-only
//! draw.
//!
//! The view's draws run under their pass's own root signature, which carries
//! the grass block and the blade buffer as two extra root parameters, so the
//! lit blades shade on the lights, shadows and environment the surfaces bound.
//! The bend pass and the cascade draw have small root signatures of their own.
//! The blade, draw-argument and bend buffers rest in `UNORDERED_ACCESS`; the
//! graph moves them to the states their readers need. On a planet the kernel
//! also reads the frame's drawn tiles, from an upload-heap buffer per frame in
//! flight, and their meshes, from the shared vertex buffer, which rests in
//! `GENERIC_READ`.

use concinnity_core::math::SaturatingFrom;
use std::cell::RefCell;

use concinnity_engine::planet::Rebase;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::grass::bend::GRASS_BEND_WORDS;
use concinnity_engine::render::environment::grass::lod::{
    GRASS_BAND_REGION, GRASS_LOD_COUNT, GRASS_LOD_VERTEX_STRIDE, GRASS_REGION_COUNT,
};
use concinnity_engine::render::environment::grass::{
    FRAME_PLANET_TILE_RECORDS, FRAME_TILE_RECORDS, GrassBender, GrassCamera, GrassField,
    GrassFrameState, GrassPass, GrassShadowView, GrassView,
};
use concinnity_engine::render::render_graph::PassId;
use concinnity_engine::render::uniforms::grass::{
    GRASS_ARGS_BYTES, GRASS_ARGS_STRIDE, GpuGrassBlade, GrassBendParams, GrassParams, GrassTileGpu,
    PlanetGrassTileGpu, grass_args_offset,
};
use concinnity_engine::render::{backend::pass_timing, shadow::bias};
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::builtin_shaders::{self, CompileProgram};
use super::com;
use super::context::DxContext;
use super::descriptor_slot::DescriptorTables;
use super::error::map_hresult;
use super::mapped_blocks::MappedBlocks;
use super::pso::{Blend, Depth, DepthBias, GraphicsPso, Raster, compute_pso};
use super::root_constants::RootConstants;
use super::root_sig::{RootSig, Visibility};
use super::texture::{
    HDR_FORMAT, create_uav_buffer, one_shot_submit_nowait, transition_barrier, upload_buffer,
};

/// Where the main pass's root signature carries the grass block (b7) and the
/// visible blades (t23).
pub(in crate::directx) const MAIN_GRASS_PARAMS_PARAM: u32 = 21;
pub(in crate::directx) const MAIN_GRASS_BLADES_PARAM: u32 = 22;
/// The b0 root constant both pass root signatures lead with, which carries each
/// grass draw's first vertex: D3D's vertex ids ignore the draw's own.
const FIRST_VERTEX_PARAM: u32 = 0;
/// The same pair in the G-buffer pre-pass's root signature.
pub(in crate::directx) const PREPASS_GRASS_PARAMS_PARAM: u32 = 9;
pub(in crate::directx) const PREPASS_GRASS_BLADES_PARAM: u32 = 10;

// The kernel's root signature: the block at b0, the blades it appends at u1,
// the draw arguments at u2, the terrain heights and mask texels it reads at t3
// and t4, last frame's depth pyramid at t5, the bend field at t6, the planet
// tiles at t7, the shared vertex stream at t8 and the tile records at t9, as
// `grass.hlsl` declares them.
fn create_generate_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .uav(1, Visibility::All)
        .uav(2, Visibility::All)
        .srv(3, Visibility::All)
        .srv(4, Visibility::All)
        .srv_table(5, 1, Visibility::All)
        .srv(6, Visibility::All)
        .srv(7, Visibility::All)
        .srv(8, Visibility::All)
        .srv(9, Visibility::All)
        .build(device, "grass generate root sig")
}

// The kernel's root parameters holding the pyramid's descriptor table, the
// bend field, the planet tiles, the shared vertex stream and the tile records.
const KERNEL_HIZ_PARAM: u32 = 5;
const KERNEL_BEND_PARAM: u32 = 6;
const KERNEL_PLANET_TILES_PARAM: u32 = 7;
const KERNEL_VERTICES_PARAM: u32 = 8;
const KERNEL_TILES_PARAM: u32 = 9;

// Planet tile and tile records one frame holds: the view's run and the
// cascade's.

// The bend pass's root signature: its block at b0 and the field at u1.
fn create_bend_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .uav(1, Visibility::All)
        .build(device, "grass bend root sig")
}

// The cascade draw's root signature: the shadow block at b0 and the cascade's
// blades at t1.
fn create_shadow_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::Vertex)
        .srv(1, Visibility::Vertex)
        .build(device, "grass shadow root sig")
}

// One non-indexed draw per command, matching the records the kernel fills, one
// per detail level. It sets no root argument, so it needs no root signature.
fn create_draw_signature(device: &ID3D12Device) -> RenderResult<ID3D12CommandSignature> {
    let arg_descs = [D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
        ..Default::default()
    }];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: u32::saturating_from(GRASS_ARGS_STRIDE),
        NumArgumentDescs: u32::saturating_from(arg_descs.len()),
        pArgumentDescs: arg_descs.as_ptr(),
        NodeMask: 0,
    };
    let mut sig: Option<ID3D12CommandSignature> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommandSignature(&raw const desc, None::<&ID3D12RootSignature>, &raw mut sig)
    }
    .map_err(|e| map_hresult(e.code(), "create grass command signature"))?;
    sig.ok_or_else(|| {
        RenderError::Other("create grass command signature: returned None".to_string())
    })
}

// The two root signatures the grass draws run under.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GrassRootSigs<'a> {
    pub main: &'a ID3D12RootSignature,
    pub prepass: &'a ID3D12RootSignature,
}

// The kernels and the draw pipelines; rebuilt as a set on a shader reload.
pub(in crate::directx) struct GrassPipelines {
    generate_root_sig: ID3D12RootSignature,
    generate: ID3D12PipelineState,
    bend_root_sig: ID3D12RootSignature,
    bend: ID3D12PipelineState,
    prepass: GrassDrawPipelines,
    main: GrassDrawPipelines,
    shadow_root_sig: ID3D12RootSignature,
    shadow: ID3D12PipelineState,
}

// One pass's blade draws: every detail level's, and the band's, whose blades
// hand the field to the canopy pixel by pixel and so discard.
pub(in crate::directx) struct GrassDrawPipelines {
    levels: ID3D12PipelineState,
    band: ID3D12PipelineState,
}

impl GrassPipelines {
    pub(in crate::directx) fn build(
        device: &ID3D12Device,
        roots: GrassRootSigs<'_>,
        msaa_samples: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let generate_root_sig = create_generate_root_signature(device)?;
        let cs = builtin_shaders::GRASS_GENERATE.compile(hot_reload)?;
        let generate = compute_pso(device, &generate_root_sig, &cs, "grass generate")?;
        let bend_root_sig = create_bend_root_signature(device)?;
        let cs = builtin_shaders::GRASS_BEND.compile(hot_reload)?;
        let bend = compute_pso(device, &bend_root_sig, &cs, "grass bend")?;
        let shadow_root_sig = create_shadow_root_signature(device)?;
        let vs = builtin_shaders::GRASS_SHADOW_VERT.compile(hot_reload)?;
        let shadow = GraphicsPso::new(&shadow_root_sig, &vs, &[])
            .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
            .raster(Raster {
                bias: DepthBias {
                    constant: i32::saturating_from(bias::RASTER_CONSTANT),
                    clamp: bias::RASTER_CLAMP,
                    slope: bias::RASTER_SLOPE,
                },
                ..Raster::default()
            })
            .build(device, "grass shadow")?;

        let vs = builtin_shaders::GRASS_PREPASS_VERT.compile(hot_reload)?;
        let prepass_pso = |ps: &[u8], label: &str| {
            super::post::gbuffer::gbuffer_targets(GraphicsPso::new(roots.prepass, &vs, ps))
                .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
                .build(device, label)
        };
        let prepass = GrassDrawPipelines {
            levels: prepass_pso(
                &builtin_shaders::GRASS_PREPASS_FRAG.compile(hot_reload)?,
                "grass prepass",
            )?,
            band: prepass_pso(
                &builtin_shaders::GRASS_PREPASS_CROSSFADE_FRAG.compile(hot_reload)?,
                "grass band prepass",
            )?,
        };

        let vs = builtin_shaders::GRASS_VERT.compile(hot_reload)?;
        let main_pso = |ps: &[u8], label: &str| {
            GraphicsPso::new(roots.main, &vs, ps)
                .target(HDR_FORMAT, Blend::Opaque)
                .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
                .samples(msaa_samples)
                .build(device, label)
        };
        let main = GrassDrawPipelines {
            levels: main_pso(&builtin_shaders::GRASS_FRAG.compile(hot_reload)?, "grass")?,
            band: main_pso(
                &builtin_shaders::GRASS_CROSSFADE_FRAG.compile(hot_reload)?,
                "grass band",
            )?,
        };
        Ok(Self {
            generate_root_sig,
            generate,
            bend_root_sig,
            bend,
            prepass,
            main,
            shadow_root_sig,
            shadow,
        })
    }
}

// The grass field and what draws it: built once at init when the world grows
// one and the GPU-driven main pass exists to draw it in.
pub(in crate::directx) struct GrassResources {
    pub(in crate::directx) field: GrassField,
    pub(in crate::directx) pipelines: GrassPipelines,
    draw_signature: ID3D12CommandSignature,
    // The visible blades the kernel appends, `field.capacity()` of them, each
    // span after the last.
    pub(in crate::directx) blades: ID3D12Resource,
    // Two slots of one draw per detail level; see `GRASS_ARGS_SLOTS`.
    pub(in crate::directx) args: ID3D12Resource,
    // The nearest shadow cascade's blades and draw arguments, laid out like
    // the view's.
    pub(in crate::directx) shadow_blades: ID3D12Resource,
    pub(in crate::directx) shadow_args: ID3D12Resource,
    // The bend field's two halves.
    pub(in crate::directx) bend_field: ID3D12Resource,
    // Every terrain's heights and every layer's mask texels, which the kernel
    // reads to root and thin the blades.
    heights: PooledBuffer,
    masks: PooledBuffer,
    // Per frame in flight, persistently mapped: the view's `GrassParams`, the
    // cascade's, and the bend pass's block.
    params: MappedBlocks,
    shadow_params: MappedBlocks,
    bend_params: MappedBlocks,
    // Per frame in flight, persistently mapped: the planet tile records and
    // the tile records both runs read.
    planet_tiles: MappedBlocks,
    tiles: MappedBlocks,
    // What the field's frames carry from one to the next. Advanced from
    // `&self` on the render thread before the fan-out.
    state: RefCell<GrassFrameState>,
}

// A buffer of `count` blades, at least one, resting in UNORDERED_ACCESS
// once a list the queue runs ahead of every later submission moves it there.
// The kernel writes every blade a draw reads, so it starts unset.
fn blade_buffer(alloc: &DeviceAllocator, count: u32) -> RenderResult<ID3D12Resource> {
    let bytes = u64::from(count.max(1)) * size_of::<GpuGrassBlade>() as u64;
    let buffer = create_uav_buffer(alloc.device(), bytes, D3D12_RESOURCE_STATE_COMMON)?;
    let to_uav = transition_barrier(
        &buffer,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let (list_alloc, list) = one_shot_submit_nowait(alloc.device(), alloc.queue(), |cmd| {
        // SAFETY: the command list is in the recording state, and the buffer the barrier names
        // is live for the call.
        unsafe { cmd.ResourceBarrier(&[to_uav]) }
    })?;
    // The allocator holds the list until the GPU has retired it.
    alloc.retire(list_alloc);
    alloc.retire(list);
    Ok(buffer)
}

impl GrassResources {
    // The phase of the next frame's per-pixel hand-off to the canopy, which
    // the canopy block carries beside the blades' own block.
    pub(in crate::directx) fn crossfade_phase(&self) -> f32 {
        self.state.borrow().history.crossfade_phase()
    }

    pub(in crate::directx) fn build(
        alloc: &DeviceAllocator,
        field: GrassField,
        roots: GrassRootSigs<'_>,
        msaa_samples: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let pipelines = GrassPipelines::build(device, roots, msaa_samples, hot_reload)?;
        let draw_signature = create_draw_signature(device)?;
        let blades = blade_buffer(alloc, field.capacity().total())?;
        let args = create_uav_buffer(device, GRASS_ARGS_BYTES as u64, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &args, GRASS_ARGS_BYTES as u64)?;
        let shadow_blades = blade_buffer(alloc, field.shadow_capacity())?;
        let shadow_args =
            create_uav_buffer(device, GRASS_ARGS_BYTES as u64, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &shadow_args, GRASS_ARGS_BYTES as u64)?;
        let bend_bytes = (GRASS_BEND_WORDS * 4) as u64;
        let bend_field = create_uav_buffer(device, bend_bytes, D3D12_RESOURCE_STATE_COMMON)?;
        super::particle::zero_default_buffer(alloc, &bend_field, bend_bytes)?;
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let heights = upload_buffer(
            alloc,
            bytemuck::cast_slice(field.buffers.bound_heights()),
            read,
        )?;
        let masks = upload_buffer(
            alloc,
            bytemuck::cast_slice(field.buffers.bound_mask_words()),
            read,
        )?;

        let params_size = size_of::<GrassParams>();
        Ok(Self {
            field,
            pipelines,
            draw_signature,
            blades,
            args,
            shadow_blades,
            shadow_args,
            bend_field,
            heights,
            masks,
            params: MappedBlocks::new(alloc, params_size, "map grass params")?,
            shadow_params: MappedBlocks::new(alloc, params_size, "map grass shadow params")?,
            bend_params: MappedBlocks::new(
                alloc,
                size_of::<GrassBendParams>(),
                "map grass bend params",
            )?,
            planet_tiles: MappedBlocks::new(
                alloc,
                FRAME_PLANET_TILE_RECORDS * size_of::<PlanetGrassTileGpu>(),
                "map grass planet tiles",
            )?,
            tiles: MappedBlocks::new(
                alloc,
                FRAME_TILE_RECORDS * size_of::<GrassTileGpu>(),
                "map grass tiles",
            )?,
            state: RefCell::new(GrassFrameState::default()),
        })
    }
}

// One run of the blade kernel, as the frame's passes read it.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GrassPlacement {
    params_gva: u64,
    args_offset: u64,
    dispatch: [u32; 3],
}

impl GrassPlacement {
    fn of(params_gva: u64, pass: &GrassPass) -> Self {
        Self {
            params_gva,
            args_offset: grass_args_offset(pass.params.args_slot) as u64,
            dispatch: pass.dispatch,
        }
    }
}

// One frame's grass inputs, prepared before the fan-out so every pass that
// draws or dispatches grass reads the same blocks.
pub(in crate::directx) struct GrassFrame {
    view: GrassPlacement,
    // The cascade's run, when the grass casts this frame.
    shadow: Option<GrassPlacement>,
    bend_gva: u64,
    bend_dispatch: [u32; 3],
    planet_tiles_gva: u64,
    tiles_gva: u64,
}

// What a frame's grass is prepared from.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GrassRequest<'a> {
    pub frame_idx: usize,
    pub cam_pos: [f32; 3],
    // The unjittered view-projection.
    pub vp: [[f32; 4]; 4],
    pub elapsed: f32,
    // The grass casts into the nearest cascade this frame.
    pub cast: bool,
    pub benders: &'a [GrassBender],
}

impl DxContext {
    // Size the grass's visible-blade buffers for this frame's render targets,
    // making them again when the blades they hold changed. The field takes
    // its new sizing only once both are made; the old buffers are retired
    // through the allocator, which holds them until no list in flight can
    // name them.
    pub(in crate::directx) fn size_grass(&mut self) -> RenderResult<()> {
        let size = [
            self.targets.extent.render_width.max(1),
            self.targets.extent.render_height.max(1),
        ];
        let Some(grass) = self.grass.as_mut() else {
            return Ok(());
        };
        let Some(resize) = grass.field.resize_for(size) else {
            return Ok(());
        };
        if resize.rebuild {
            let alloc = &self.hw.alloc;
            let blades = blade_buffer(alloc, resize.sizing.capacity.total())?;
            let shadow_blades = blade_buffer(alloc, resize.sizing.shadow_capacity)?;
            alloc.retire(std::mem::replace(&mut grass.blades, blades));
            alloc.retire(std::mem::replace(&mut grass.shadow_blades, shadow_blades));
        }
        grass.field.apply(resize);
        Ok(())
    }

    // Carry the grass into the frame the world moved to.
    pub(in crate::directx) fn rebase_grass(&self, rebase: &Rebase) {
        if let Some(grass) = &self.grass {
            grass.state.borrow_mut().history.rebase(rebase);
        }
    }

    // Write this frame's grass blocks for `request`, advancing the field's
    // history. `None` when the world grows no grass.
    pub(in crate::directx) fn prepare_grass_frame(
        &self,
        request: GrassRequest<'_>,
    ) -> Option<GrassFrame> {
        let grass = self.grass.as_ref()?;
        let GrassRequest {
            frame_idx,
            cam_pos,
            vp,
            elapsed,
            cast,
            benders,
        } = request;
        // The pyramid holds last frame's depth once one has been built, tested
        // through the view-projection the draw cull tests through.
        let hiz = self.cull.hiz.as_ref().and_then(|h| {
            self.state
                .planner
                .temporal()
                .occlusion()
                .grass_hiz([h.width as f32, h.height as f32], h.mip_count)
        });
        let view = GrassView {
            camera: GrassCamera {
                position: cam_pos,
                vp,
                focal: self.grass_lod.get().map_or(1.0, |l| l.focal),
                hiz,
            },
            elapsed,
            shadow: cast.then(|| GrassShadowView {
                vp: self.state.planner.shadow.uniforms.light_vps[0],
                to_light: self.uniforms.lights.sun().direction,
            }),
            benders,
        };
        let mut state = grass.state.borrow_mut();
        let frame = state.frame(&grass.field, &self.state, &view);
        let view_gva = grass.params.write(frame_idx, &frame.view.params)?;
        let shadow = match frame.shadow {
            Some(pass) => Some(GrassPlacement::of(
                grass.shadow_params.write(frame_idx, &pass.params)?,
                &pass,
            )),
            None => None,
        };
        Some(GrassFrame {
            view: GrassPlacement::of(view_gva, &frame.view),
            shadow,
            bend_gva: grass.bend_params.write(frame_idx, &frame.bend.params)?,
            bend_dispatch: frame.bend.dispatch,
            planet_tiles_gva: grass
                .planet_tiles
                .write_bytes(frame_idx, bytemuck::cast_slice(frame.planet_tile_records()))?,
            tiles_gva: grass
                .tiles
                .write_bytes(frame_idx, bytemuck::cast_slice(frame.tile_records()))?,
        })
    }

    // Encode the `GrassBend` node: relax the bend field and stamp this frame's
    // footprints into it. The field is already in UNORDERED_ACCESS.
    pub(in crate::directx) fn encode_grass_bend(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        let [x, y, z] = frame.bend_dispatch;
        // SAFETY: the command list is recording, and the bend root signature is owned by `grass`.
        unsafe { cmd.SetComputeRootSignature(&grass.pipelines.bend_root_sig) };
        // SAFETY: the bend PSO is owned by `grass` and matches the root signature above.
        unsafe { cmd.SetPipelineState(&grass.pipelines.bend) };
        // SAFETY: `frame.bend_gva` addresses this frame's live bend constants.
        unsafe { cmd.SetComputeRootConstantBufferView(0, frame.bend_gva) };
        let bend_field_gva = com::gpu_va(&grass.bend_field);
        // SAFETY: the bend field is owned by `grass` and already in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(1, bend_field_gva) };
        // SAFETY: the bend pipeline and both root arguments it reads are bound above.
        unsafe { cmd.Dispatch(x, y, z) };
    }

    // Encode the `GrassShadow` node: place the blades the nearest cascade
    // draws.
    pub(in crate::directx) fn encode_grass_shadow(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(shadow)) = (&self.grass, frame.shadow.as_ref()) else {
            return;
        };
        self.place_blades(
            cmd,
            grass,
            (shadow, [frame.planet_tiles_gva, frame.tiles_gva]),
            (&grass.shadow_blades, &grass.shadow_args),
        );
    }

    // Draw the cascade's blades into the nearest cascade's slice, `dsv`, with
    // the coarsest strip, under their own root signature.
    pub(in crate::directx) fn encode_grass_shadow_draw(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(shadow)) = (&self.grass, frame.shadow.as_ref()) else {
            return;
        };
        let coarsest = shadow.args_offset + ((GRASS_LOD_COUNT - 1) * GRASS_ARGS_STRIDE) as u64;
        self.timed(cmd, frame_idx, PassId::GrassShadowDraw, || {
            // SAFETY: the command list is recording, and `dsv` names the nearest cascade's live
            // slice.
            unsafe { cmd.OMSetRenderTargets(0, None, false, Some(&raw const dsv)) };
            // SAFETY: the shadow root signature is owned by `grass`.
            unsafe { cmd.SetGraphicsRootSignature(&grass.pipelines.shadow_root_sig) };
            // SAFETY: the shadow PSO is owned by `grass` and matches the root signature above.
            unsafe { cmd.SetPipelineState(&grass.pipelines.shadow) };
            // SAFETY: `shadow.params_gva` addresses the cascade's live placement constants.
            unsafe { cmd.SetGraphicsRootConstantBufferView(0, shadow.params_gva) };
            let blades_gva = com::gpu_va(&grass.shadow_blades);
            // SAFETY: the shadow blade buffer is owned by `grass` and was filled by its placement.
            unsafe { cmd.SetGraphicsRootShaderResourceView(1, blades_gva) };
            // SAFETY: setting the topology only needs a recording command list.
            unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP) };
            // SAFETY: the args buffer is in INDIRECT_ARGUMENT for this pass and `coarsest` names
            // one whole record of it.
            unsafe {
                cmd.ExecuteIndirect(
                    &grass.draw_signature,
                    1,
                    &grass.shadow_args,
                    coarsest,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            // SAFETY: restoring the list topology only needs a recording command list.
            unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        });
        self.inc_draw_calls(1);
    }

    // Encode the `Grass` node: place, cull and append this frame's blades. Both
    // buffers are already in UNORDERED_ACCESS for the write.
    pub(in crate::directx) fn encode_grass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        self.place_blades(
            cmd,
            grass,
            (&frame.view, [frame.planet_tiles_gva, frame.tiles_gva]),
            (&grass.blades, &grass.args),
        );
    }

    // One run of the blade kernel for `placement`, reading the planet tile
    // and tile records at `planet_tiles_gva` and `tiles_gva`, appending to
    // `blades` and filling `args`.
    fn place_blades(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        grass: &GrassResources,
        (placement, [planet_tiles_gva, tiles_gva]): (&GrassPlacement, [u64; 2]),
        (blades, args): (&ID3D12Resource, &ID3D12Resource),
    ) {
        let [x, y, z] = placement.dispatch;
        // SAFETY: the command list is recording, and the generate root signature is owned by
        // `grass`.
        unsafe { cmd.SetComputeRootSignature(&grass.pipelines.generate_root_sig) };
        // SAFETY: the generate PSO is owned by `grass` and matches the root signature above.
        unsafe { cmd.SetPipelineState(&grass.pipelines.generate) };
        // SAFETY: `placement.params_gva` addresses this placement's live constants.
        unsafe { cmd.SetComputeRootConstantBufferView(0, placement.params_gva) };
        let blades_gva = com::gpu_va(blades);
        // SAFETY: the caller's blade buffer is owned by `grass` and outlives the dispatch.
        unsafe { cmd.SetComputeRootUnorderedAccessView(1, blades_gva) };
        let args_gva = com::gpu_va(args);
        // SAFETY: the caller's args buffer is owned by `grass` and outlives the dispatch.
        unsafe { cmd.SetComputeRootUnorderedAccessView(2, args_gva) };
        let heights_gva = com::gpu_va(&grass.heights);
        // SAFETY: the height field is owned by `grass`.
        unsafe { cmd.SetComputeRootShaderResourceView(3, heights_gva) };
        let masks_gva = com::gpu_va(&grass.masks);
        // SAFETY: the placement masks are owned by `grass`.
        unsafe { cmd.SetComputeRootShaderResourceView(4, masks_gva) };
        let bend_field_gva = com::gpu_va(&grass.bend_field);
        // SAFETY: the bend field is owned by `grass`, written by the bend pass ahead of this one.
        unsafe { cmd.SetComputeRootShaderResourceView(KERNEL_BEND_PARAM, bend_field_gva) };
        // SAFETY: `planet_tiles_gva` addresses this frame's live planet tile records.
        unsafe {
            cmd.SetComputeRootShaderResourceView(KERNEL_PLANET_TILES_PARAM, planet_tiles_gva);
        }
        // SAFETY: `tiles_gva` addresses this frame's live tile records.
        unsafe { cmd.SetComputeRootShaderResourceView(KERNEL_TILES_PARAM, tiles_gva) };
        let vertices_gva = com::gpu_va(&self.scene.geometry.vertex_buffer);
        // SAFETY: the shared vertex stream is owned by `self.scene` and outlives the frame.
        unsafe { cmd.SetComputeRootShaderResourceView(KERNEL_VERTICES_PARAM, vertices_gva) };
        // The pyramid's descriptor lives in the shader-visible SRV heap.
        // Bound whenever it exists so the table always points at a live
        // descriptor; the block's `hiz_enabled` gates the reads.
        if let Some(hiz) = &self.cull.hiz {
            // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
            unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
            // SAFETY: the Hi-Z pyramid's SRV is a live slot in the heap set above.
            unsafe { cmd.set_compute_srv_table(KERNEL_HIZ_PARAM, hiz.srv_gpu) };
        }
        // SAFETY: the generate pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(x, y, z) };
    }

    // Draw the blades into the G-buffer pre-pass bound on `cmd`, under the
    // pre-pass root signature: the main view at b1 rasterizes them, the
    // pre-pass view at `gb_view_gva` gives their motion.
    pub(in crate::directx) fn encode_grass_prepass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        gb_view_gva: u64,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(root_sig)) = (&self.grass, self.cull.prepass_root_sig.as_ref())
        else {
            return;
        };
        let main_view_gva = com::gpu_va(&self.uniforms.view_ubo_resources[frame_idx]);
        self.timed(cmd, frame_idx, PassId::GrassPrepass, || {
            // SAFETY: the command list is recording, and the pre-pass root signature is owned by
            // `self.cull`.
            unsafe { cmd.SetGraphicsRootSignature(root_sig) };
            // SAFETY: `main_view_gva` addresses this frame's live main view constants.
            unsafe { cmd.SetGraphicsRootConstantBufferView(1, main_view_gva) };
            // SAFETY: `gb_view_gva` is the caller's live pre-pass view block.
            unsafe { cmd.SetGraphicsRootConstantBufferView(2, gb_view_gva) };
            let params_gva = frame.view.params_gva;
            // SAFETY: the root signature declares the grass params CBV, and the block is live.
            unsafe {
                cmd.SetGraphicsRootConstantBufferView(PREPASS_GRASS_PARAMS_PARAM, params_gva);
            }
            let blades_gva = com::gpu_va(&grass.blades);
            // SAFETY: the root signature declares the blade SRV, and `grass` owns the buffer.
            unsafe {
                cmd.SetGraphicsRootShaderResourceView(PREPASS_GRASS_BLADES_PARAM, blades_gva);
            }
            self.draw_grass(cmd, grass, &grass.pipelines.prepass, frame);
        });
    }

    // Draw the lit blades into the main pass bound on `cmd`, whose root
    // signature and every parameter the surfaces set are still in place.
    pub(in crate::directx) fn encode_grass_main(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        self.timed(cmd, frame_idx, PassId::GrassDraw, || {
            let params_gva = frame.view.params_gva;
            // SAFETY: the main pass is recording on `cmd`, and its bound root signature declares
            // the grass params CBV.
            unsafe { cmd.SetGraphicsRootConstantBufferView(MAIN_GRASS_PARAMS_PARAM, params_gva) };
            let blades_gva = com::gpu_va(&grass.blades);
            // SAFETY: the bound root signature declares the blade SRV, and `grass` owns the buffer.
            unsafe { cmd.SetGraphicsRootShaderResourceView(MAIN_GRASS_BLADES_PARAM, blades_gva) };
            self.draw_grass(cmd, grass, &grass.pipelines.main, frame);
        });
    }

    // The draws both passes issue, one per region: a strip per visible blade,
    // the band's under the pipeline that discards the pixels the canopy shows.
    // Each region's first vertex rides the b0 root constant, which the pass's
    // surfaces set per command, so it is free here. The pass's list topology is
    // put back for whatever draws after it.
    fn draw_grass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        grass: &GrassResources,
        pipelines: &GrassDrawPipelines,
        frame: &GrassFrame,
    ) {
        // SAFETY: the command list is recording, and the levels PSO is owned by `grass`.
        unsafe { cmd.SetPipelineState(&pipelines.levels) };
        // SAFETY: setting the topology only needs a recording command list.
        unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP) };
        for region in 0..GRASS_REGION_COUNT {
            if region == GRASS_BAND_REGION {
                // SAFETY: the band PSO is owned by `grass` and shares the levels' root signature.
                unsafe { cmd.SetPipelineState(&pipelines.band) };
            }
            let first_vertex = u32::saturating_from(region) * GRASS_LOD_VERTEX_STRIDE;
            // SAFETY: parameter 0 of the bound root signature is one root-constant DWORD.
            unsafe { cmd.set_graphics_root_constants(FIRST_VERTEX_PARAM, &first_vertex) };
            let args_offset = frame.view.args_offset + (region * GRASS_ARGS_STRIDE) as u64;
            // SAFETY: the args buffer is in INDIRECT_ARGUMENT for this pass, and `args_offset`
            // names one whole record of it.
            unsafe {
                cmd.ExecuteIndirect(
                    &grass.draw_signature,
                    1,
                    &grass.args,
                    args_offset,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
        }
        // SAFETY: restoring the list topology only needs a recording command list.
        unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        self.inc_draw_calls(u32::saturating_from(GRASS_REGION_COUNT));
    }

    // `record` bracketed by `pass`'s timestamps inside the enclosing pass's
    // command list.
    fn timed(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        pass: PassId,
        record: impl FnOnce(),
    ) {
        let (start, end) = pass_timing::pass_pair(frame_idx, pass);
        let heap = self.timestamps.query_heap.as_ref();
        if let Some(heap) = heap {
            // SAFETY: the command list is in the recording state and the query heap is live.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start) };
        }
        record();
        if let Some(heap) = heap {
            // SAFETY: as above.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end) };
        }
    }
}
