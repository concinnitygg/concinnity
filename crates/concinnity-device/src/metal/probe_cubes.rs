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
// build serves every pass that reads it. It lives in a per-frame ring slot,
// re-encoded only when a bake (or an env-map swap) changes a handle.

#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLDevice, MTLFunction as _, MTLRenderCommandEncoder,
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
    // The identity of the cubes the block would name. `probe_cube_or_sky`
    // returns the sky prefilter for unbaked slots, so a bake that installs a
    // cube -- or an env-map swap that changes the fallback -- moves this and
    // nothing else does.
    fn probe_cube_signature(&self) -> u64 {
        let mut sig = super::bindless_args::Signature::new();
        sig.push_u64(self.texture_epoch);
        for i in 0..MAX_PROBES {
            sig.push_texture(self.probe_cube_or_sky(i));
        }
        sig.finish()
    }

    // Write this frame's probe cube handles into a ring slot, when they have
    // changed since that slot was last written. `probe_cube_or_sky` returns the
    // sky prefilter for unbaked slots, so every entry is always a valid cube and
    // the shaders' `ProbeSet.count` alone decides how many are read.
    pub(super) fn build_probe_cube_args(
        &mut self,
        ring_slot: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let sig = self.probe_cube_signature();
        // Cloned so no borrow of `self` outlives the mutable ring borrow below.
        let enc = self.probe_cube_arg_encoder.clone();
        let len = enc.encodedLength().max(16);
        let (buf, allocated) = self
            .rings
            .probe_cube
            .slot_fresh(&self.device, ring_slot, len)?;
        if allocated {
            self.probe.cube_arg_gates.invalidate(ring_slot);
        }
        if !self.probe.cube_arg_gates.stale(ring_slot, sig) {
            return Ok(buf);
        }
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

    // Rebuild the probe-cube residency set when the cubes change. Called once
    // per frame alongside `build_probe_cube_args`.
    pub(super) fn refresh_probe_cube_residency(&mut self) {
        let sig = self.probe_cube_signature();
        // Taken out so the iterator below can borrow the rest of `self`.
        let mut set = core::mem::replace(
            &mut self.probe.cube_residency,
            super::bindless_args::ResidencySet::new(),
        );
        set.refresh(sig, (0..MAX_PROBES).map(|i| self.probe_cube_or_sky(i)));
        self.probe.cube_residency = set;
    }

    // Bind the probe cube block for a fragment stage and declare every cube it
    // names resident. An argument buffer's contents are not tracked, so a cube
    // reached only through it reads garbage without the residency declaration.
    // A no-op before the first `build_probe_cube_args`, which leaves the
    // shader's probe path unbound -- the same state a world with no probe set
    // is in.
    pub(super) fn bind_probe_cubes(&self, enc: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        let Some(args) = self.probe.cube_args.as_ref() else {
            return;
        };
        enc.set_fragment_buffer(args.as_ref(), 0, PROBE_CUBE_ARG_BUFFER_INDEX);
        self.probe.cube_residency.declare_fragment(enc);
    }
}
