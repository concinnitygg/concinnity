// The reflection-probe cube array as a Metal argument buffer.
//
// Five fragment shaders sample the same set (`ssr`, `rt_reflections`, `glass`,
// `glass_mesh`, `water`). Each declares it as a `ParameterBlock<ProbeCubes>`
// rather than a global-scope array, because slangc emits a global-scope
// resource array with no `[[texture(n)]]` and the Metal compiler then places it
// at whatever slot happens to be unused -- a placement nothing can read back
// from the emitted MSL. A parameter block is one pinned buffer slot instead,
// which the build script's Metal ABI table asserts.
//
// The buffer holds `MAX_PROBES` texture handles and nothing else, so one
// per-frame build serves every pass that reads it.

#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLDevice, MTLFunction as _, MTLRenderCommandEncoder,
    MTLRenderStages, MTLResourceUsage,
};

use concinnity_core::render::uniforms::MAX_PROBES;

use super::context::MtlContext;
use super::encode::RenderEncode;

// Buffer slot the five shaders pin their `ParameterBlock<ProbeCubes>` to.
pub(super) const PROBE_CUBE_ARG_BUFFER_INDEX: usize = 11;

// The argument encoder describing that block. All five declare the same one, so
// a single encoder serves them all; it comes from the SSR resolve fragment
// because that is an engine metallib rather than a world compile, and so is
// available whether or not the world enables SSR.
pub(super) fn probe_cube_arg_encoder(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLArgumentEncoder>>, String> {
    let frag = super::slang_shaders::entry_function(
        device,
        &super::slang_shaders::SSR_RESOLVE,
        hot_reload,
    )?;
    // SAFETY: the buffer index the five probe-sampling fragments pin their
    // parameter block to, locked by the build script's Metal ABI table.
    Ok(unsafe { frag.newArgumentEncoderWithBufferIndex(PROBE_CUBE_ARG_BUFFER_INDEX) })
}

impl MtlContext {
    // Write this frame's probe cube handles into a ring slot.
    // `probe_cube_or_sky` returns the sky prefilter for unbaked slots, so every
    // entry is always a valid cube and the shaders' `ProbeSet.count` alone
    // decides how many are read.
    pub(super) fn build_probe_cube_args(
        &mut self,
        ring_slot: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        // Cloned so no borrow of `self` outlives the mutable ring borrow below.
        let enc = self.probe_cube_arg_encoder.clone();
        let len = enc.encodedLength().max(16);
        let buf = self.rings.probe_cube.slot(&self.device, ring_slot, len)?;
        // SAFETY: `buf` is sized to the encoder's own `encodedLength()`, and the
        // argument ids below are the `MAX_PROBES` entries the block declares.
        unsafe {
            enc.setArgumentBuffer_offset(Some(&buf), 0);
            for i in 0..MAX_PROBES {
                enc.setTexture_atIndex(Some(self.probe_cube_or_sky(i)), i);
            }
        }
        Ok(buf)
    }

    // Bind the probe cube block for a fragment stage and declare every cube it
    // names resident. An argument buffer's contents are not tracked, so a cube
    // reached only through it reads garbage without the `useResource`. A no-op
    // before the first `build_probe_cube_args`, which leaves the shader's
    // probe path unbound -- the same state a world with no probe set is in.
    pub(super) fn bind_probe_cubes(&self, enc: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        let Some(args) = self.probe.cube_args.as_ref() else {
            return;
        };
        enc.set_fragment_buffer(args.as_ref(), 0, PROBE_CUBE_ARG_BUFFER_INDEX);
        for i in 0..MAX_PROBES {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(self.probe_cube_or_sky(i)),
                MTLResourceUsage::Read,
                MTLRenderStages::Fragment,
            );
        }
    }
}
