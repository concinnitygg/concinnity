//! The prop reflection probe on Metal: the six cube faces of the scene seen
//! from the prop, rendered every frame from the frame's own records like a
//! planar mirror, and their convolution into the probe's cube of the probe
//! array. The record and the plan are the shared `probe::prop`; this module
//! owns the capture targets and encodes the `PropProbe` graph node.
//!
//! Each face culls into one more mirror slot with the prop's own records left
//! out, and draws under the mirror's single stream. Everything the node writes
//! is read later on the same queue (the next face's cull reuses the slot, the
//! convolution reads the capture, the main pass reads the cube), which Metal's
//! hazard tracking orders.

#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::math::narrow_f32;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::probe::capture::{self, PrefilterPlan};
use concinnity_engine::render::probe::prop::{
    FaceCycle, FaceSlice, PropProbeFrame, prop_probe_plan,
};
use concinnity_engine::render::probe::radiance::FaceStorage;
use concinnity_engine::render::render_graph::PassId;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{MTLComputePassDescriptor, MTLTexture};

use super::context::MtlContext;
use super::cull::MirrorCull;
use super::draw::main::{FacePass, FaceTargets, GpuFrameBuffers, MainPassCamera};
use super::pass_timing::PassTimer;
use super::probe::{make_face_color, make_face_depth};
use super::probe_prefilter::{PrefilterGpu, create_capture_cube, encode_ggx_mip, encode_pyramid};

// Encoders the node records for `slice`: a cull and a face per face it
// renders, then the convolution when it closes the cycle.
fn prop_probe_encoders(slice: &FaceSlice) -> usize {
    2 * slice.count + usize::from(slice.convolve)
}

// The world's prop probe: where it captures this frame, and the capture it
// renders into. `None` capture until the first frame that rides one.
#[derive(Default)]
pub(in crate::metal) struct PropProbe {
    // This frame's capture and the faces it renders.
    frame: Option<(PropProbeFrame, FaceSlice)>,
    cycle: FaceCycle,
    capture: Option<PropCapture>,
    // Set once building the capture failed, which is not retried.
    failed: bool,
}

// The targets the faces render into and the convolution from the capture into
// the probe's cube. Rebuilt when the probe's size or the cube array changes.
struct PropCapture {
    resolution: u32,
    plan: PrefilterPlan,
    msaa_color: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    depth: Retained<ProtocolObject<dyn MTLTexture>>,
    capture: Retained<ProtocolObject<dyn MTLTexture>>,
    gpu: PrefilterGpu,
    // The cube array `gpu` writes into.
    cubes: Retained<ProtocolObject<dyn MTLTexture>>,
}

impl PropProbe {
    // Whether this frame captures the probe.
    pub(in crate::metal) fn capturing(&self) -> bool {
        self.frame.is_some()
    }
}

impl MtlContext {
    // Whether this backend can capture the world's prop probe: the bindless
    // cull it draws through, the convolution kernels, and a cube for it.
    pub(in crate::metal) fn prop_probe_supported(&self) -> bool {
        self.cull.bindless
            && self.probe.prefilter.is_some()
            && self.probe.book.prop_cube().is_some()
            && !self.probe.prop.failed
    }

    // The mirror slot the prop probe culls into: the one after the planar
    // planes'.
    pub(in crate::metal) fn prop_probe_cull_slot(&self) -> usize {
        self.planar_reflection
            .as_ref()
            .map_or(0, |s| s.layout.planes().len())
    }

    // This frame's prop probe, ahead of the probe records and the graph: where
    // the `PropProbe` node captures, and the probe's record, which the shaders
    // read once a frame before has filled the cube. A frame whose graph does
    // not run the node withdraws the record.
    pub(in crate::metal) fn prepare_prop_probe(
        &mut self,
        frame: Option<PropProbeFrame>,
        enabled: bool,
    ) {
        let Some(frame) = frame.filter(|_| enabled && self.prop_probe_supported()) else {
            self.probe.prop.frame = None;
            self.probe.prop.cycle.reset();
            self.probe.book.set_prop_probe(None);
            return;
        };
        if let Err(e) = self.ensure_prop_capture(frame.resolution) {
            tracing::error!("reflection probes: the prop probe could not be built: {e}");
            self.probe.prop = PropProbe {
                failed: true,
                ..PropProbe::default()
            };
            self.probe.book.set_prop_probe(None);
            return;
        }
        let slice = self.probe.prop.cycle.advance(frame.faces_per_frame);
        self.probe.prop.frame = Some((frame, slice));
        self.probe
            .book
            .set_prop_probe(slice.filled.then_some(&frame));
    }

    // Build the capture for faces of `resolution` texels into the current
    // cube array, unless the one held already is. A new capture has not
    // filled the cube yet.
    fn ensure_prop_capture(&mut self, resolution: u32) -> RenderResult<()> {
        let cubes = self.probe.cubes.texture();
        let current = self.probe.prop.capture.as_ref().is_some_and(|c| {
            c.resolution == resolution && Retained::as_ptr(&c.cubes) == std::ptr::from_ref(cubes)
        });
        if current {
            return Ok(());
        }
        let cube = self
            .probe
            .book
            .prop_cube()
            .ok_or_else(|| RenderError::Other("probe: no cube for the prop probe".into()))?;
        let plan = prop_probe_plan(resolution);
        let samples = self.targets.hdr.sample_count;
        let msaa_color = (samples > 1)
            .then(|| make_face_color(&self.hw.device, plan.face_size(), samples))
            .transpose()?;
        let depth = make_face_depth(&self.hw.device, plan.face_size(), samples)?;
        let capture = create_capture_cube(&self.hw.device, &plan)?;
        let gpu = PrefilterGpu::new(capture.clone(), cubes, cube, &plan)?;
        self.probe.prop.capture = Some(PropCapture {
            resolution,
            plan,
            msaa_color,
            depth,
            capture,
            gpu,
            cubes: objc2::Message::retain(cubes),
        });
        self.probe.prop.frame = None;
        self.probe.prop.cycle.reset();
        Ok(())
    }

    // The `PropProbe` node: this frame's faces of the prop probe, each culled
    // against the face's frustum with the prop left out and drawn from the
    // frame's records, then, once the cycle's last face is in, the
    // convolution of the capture into the probe's cube. The pass's timing
    // span opens on the first cull and closes on the last encoder.
    pub(in crate::metal) fn encode_prop_probe(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        params: &super::graph_exec::GraphFrameParams<'_>,
    ) -> RenderResult<()> {
        let (Some((frame, slice)), Some(capture)) =
            (self.probe.prop.frame, self.probe.prop.capture.as_ref())
        else {
            return Ok(());
        };
        let (Some(object_buffer), Some(draw_args_buffer)) =
            (params.object_buffer, params.draw_args_buffer)
        else {
            return Ok(());
        };
        let eye = frame.eye();
        let slot = self.prop_probe_cull_slot();
        let encoders = prop_probe_encoders(&slice);
        for (i, face) in slice.faces().enumerate() {
            self.encode_mirror_cull(
                cmd_buf,
                MirrorCull {
                    object_buffer,
                    draw_args_buffer,
                    frustum: &capture::face_frustum(eye, face, frame.capture_distance),
                    eye,
                    slot,
                    timer: PassTimer::span(PassId::PropProbe, 2 * i, encoders),
                    exclude: frame.records,
                    capture_tiles: self.state.placement.capture_tiles,
                },
            )?;
            let culled = self.cull.mirror_slots.get(slot);
            self.encode_main_into_face(
                cmd_buf,
                FaceTargets {
                    color_msaa: capture.msaa_color.as_deref(),
                    depth: &capture.depth,
                    resolve: &capture.capture,
                    resolve_slice: face,
                },
                MainPassCamera {
                    elapsed: narrow_f32(params.elapsed),
                    vp: capture::face_view_projection(eye, face),
                    view: capture::face_view_matrix(eye, face),
                    cam_pos: eye,
                },
                GpuFrameBuffers {
                    object_buffer: params.object_buffer,
                    material_params: params.material_params,
                    bindless_tex_args: params.bindless_tex_args,
                    deformed_skinned: params.deformed_skinned,
                    counts: self.draw_record_counts(),
                    regions: self.cull.mirror_regions(culled),
                },
                FacePass {
                    icb_override: culled.map(|s| s.icb.as_ref()),
                    scissor: None,
                    clusters: None,
                    timer: PassTimer::span(PassId::PropProbe, 2 * i + 1, encoders),
                    storage: FaceStorage::PROBE,
                    daylight: &self.state.view.daylight,
                    stand_ins: frame.stand_ins(),
                    sky_from_cube: true,
                },
            )?;
        }
        if !slice.convolve {
            return Ok(());
        }
        self.encode_prop_convolution(
            cmd_buf,
            capture,
            PassTimer::span(PassId::PropProbe, encoders - 1, encoders),
        )
    }

    // The whole convolution of the capture into the probe's cube in one
    // compute encoder, whose serial dispatches order the source pyramid ahead
    // of every GGX mip that reads it.
    fn encode_prop_convolution(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        capture: &PropCapture,
        timer: PassTimer,
    ) -> RenderResult<()> {
        use objc2_metal::MTLCommandBuffer as _;
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or_else(|| RenderError::Other("probe: prefilter pipelines missing".into()))?;
        let desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute_timer(&desc, timer);
        }
        let enc = super::scoped_encoder::ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("probe: failed to get prop probe compute encoder".into())
                })?,
            ns_string!("prop probe convolution"),
        );
        encode_pyramid(&enc, pipelines, &capture.gpu, &capture.plan);
        for mip in 1..capture.plan.mips() {
            encode_ggx_mip(
                &enc,
                pipelines,
                &self.scene.cube_sampler,
                &capture.gpu,
                &capture.plan,
                mip,
            );
        }
        Ok(())
    }
}
