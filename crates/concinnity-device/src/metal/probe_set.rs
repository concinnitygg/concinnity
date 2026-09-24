// The reflection-probe set as Metal binds it: one cube-map array holding every
// probe's prefiltered radiance, a cube per probe, and this frame's parallax
// records in a ring buffer, one `ProbeUniforms` per installed probe.
//
// `ssr`, `rt_reflections` and the transparent pass (glass, glass_mesh, water)
// bind the array as a texture and the records as a buffer. The bindless main
// pass reaches the array through its texture argument buffer instead, since
// the draws an indirect command buffer executes cannot see encoder-bound
// textures; its records ride a buffer the ICB draws inherit.

#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::render_types::ClusterParams;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::reflection_probe::PrefilterPlan;
use concinnity_core::render::uniforms::{ProbeSet, ProbeUniforms, grown_probe_capacity};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLRenderCommandEncoder, MTLTexture, MTLTextureType, MTLTextureUsage,
};

use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::encode::RenderEncode;
use super::error::allocation_failed;
use super::probe_prefilter::PROBE_CUBE_FORMAT;

// The cube array every probe bakes a slice of. `capacity` cubes at the bake's
// face size and mip count, or a one-texel stand-in before any placement asks for
// room: the shaders read only the first `ProbeSet::count` cubes, so a world with
// no probe never samples it.
//
// Created unpooled: it is replaced only when a placement outgrows it, and the
// old one is parked behind the frames-in-flight fence rather than freed.
pub(in crate::metal) struct ProbeCubeArray {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    capacity: usize,
}

impl ProbeCubeArray {
    pub(in crate::metal) fn placeholder(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> RenderResult<ProbeCubeArray> {
        Ok(ProbeCubeArray {
            texture: create(device, 1, 1, 1)?,
            capacity: 0,
        })
    }

    pub(in crate::metal) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        plan: &PrefilterPlan,
        capacity: usize,
    ) -> RenderResult<ProbeCubeArray> {
        Ok(ProbeCubeArray {
            texture: create(
                device,
                plan.face_size() as usize,
                plan.mips() as usize,
                capacity,
            )?,
            capacity,
        })
    }

    pub(in crate::metal) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(in crate::metal) fn texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.texture
    }
}

fn create(
    device: &ProtocolObject<dyn MTLDevice>,
    face_size: usize,
    mips: usize,
    cubes: usize,
) -> RenderResult<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = TextureDesc {
        kind: MTLTextureType::TypeCubeArray,
        format: PROBE_CUBE_FORMAT,
        width: face_size,
        height: face_size,
        mip_count: mips,
        array_length: cubes.max(1),
        usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0),
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("probe cube array"))
}

// Where a render pass takes each binding of the set. `cubes` is `None` for a
// pass whose cube array rides an argument buffer, and `cluster` holds the
// buffer slots of the params and lists of the main camera's cluster grid for a
// pass that bins the probes it blends.
#[derive(Clone, Copy)]
pub(in crate::metal) struct ProbeSlots {
    pub set: usize,
    pub records: usize,
    pub cubes: Option<usize>,
    pub cluster: Option<(usize, usize)>,
}

// The world's reflection-probe set as a render pass binds it.
#[derive(Clone, Copy)]
pub(in crate::metal) struct ProbeBindings<'a> {
    // The live count.
    pub set: ProbeSet,
    // This frame's per-probe influence boxes. `None` before the first frame
    // builds them, which no pass reaches.
    pub records: Option<&'a ProtocolObject<dyn MTLBuffer>>,
    // One cube per record.
    pub cubes: &'a ProtocolObject<dyn MTLTexture>,
    // The main camera's cluster grid and its per-cluster lists, which bin the
    // probes a fragment blends.
    pub cluster: ClusterParams,
    pub cluster_list: &'a ProtocolObject<dyn MTLBuffer>,
}

impl ProbeBindings<'_> {
    // Bind the set at `slots`. The cube array's sampler is the caller's, since
    // each pass seats it beside its own.
    pub(in crate::metal) fn bind(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        slots: ProbeSlots,
    ) {
        enc.set_fragment_value(&self.set, slots.set);
        if let Some(records) = self.records {
            enc.set_fragment_buffer(records, 0, slots.records);
        }
        if let Some(cubes) = slots.cubes {
            enc.set_fragment_texture(self.cubes, cubes);
        }
        if let Some((params, list)) = slots.cluster {
            enc.set_fragment_value(&self.cluster, params);
            enc.set_fragment_buffer(self.cluster_list, 0, list);
        }
    }
}

impl MtlContext {
    // Make room for `placements` cubes. A placement list the array already
    // holds keeps it; a larger one replaces it, parking the old array behind
    // the fence since a frame in flight may still sample it.
    pub(in crate::metal) fn reserve_probe_cubes(
        &mut self,
        plan: &PrefilterPlan,
        placements: usize,
    ) -> RenderResult<()> {
        let Some(capacity) = grown_probe_capacity(self.probe.cubes.capacity(), placements, 1)
        else {
            return Ok(());
        };
        let grown = ProbeCubeArray::new(&self.hw.device, plan, capacity)?;
        let old = core::mem::replace(&mut self.probe.cubes, grown);
        self.probe.retire_pool.push(
            self.frame_ring_index,
            super::probe::RetiredBake::CubeArray(old),
        );
        self.arg_buffers.texture_epoch += 1;
        tracing::debug!("reflection probes: cube array grown to {capacity}");
        Ok(())
    }

    // The header the shaders read the live count from.
    pub(in crate::metal) fn probe_set(&self) -> ProbeSet {
        self.probe.book.header()
    }

    // Write this frame's parallax records into a ring slot. The slot always
    // holds at least one record, so the binding is valid with no probe baked.
    pub(in crate::metal) fn build_probe_records(
        &mut self,
        ring_slot: usize,
    ) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let empty = [<ProbeUniforms as bytemuck::Zeroable>::zeroed()];
        let records: &[ProbeUniforms] = if self.probe.book.count() == 0 {
            &empty
        } else {
            self.probe.book.records()
        };
        self.rings
            .probe_records
            .write(&self.hw.device, ring_slot, bytemuck::cast_slice(records))
    }

    // This frame's probe set, for a render pass to bind.
    pub(in crate::metal) fn probe_bindings(&self) -> ProbeBindings<'_> {
        ProbeBindings {
            set: self.probe_set(),
            records: self.probe.records_buf.as_deref(),
            cubes: self.probe.cubes.texture(),
            cluster: self.cluster_params,
            cluster_list: &self.light_cull.cluster_buffer,
        }
    }
}
