//! Bloom, written once for every backend.
//!
//! A soft glow built over a chain of octaves, each half the size of the one
//! above it. The prefilter thresholds the scene into the top octave, each
//! downsample reduces one octave into the next, and each upsample expands one
//! back into the octave above it, additively, until the top octave holds the
//! accumulated glow the composite samples.
//!
//! The top octave is the render graph's `bloom_top` transient: the pool owns it
//! and the composite reads it, so the caller supplies it. The octaves below it
//! are one target with a mip level per octave, owned here.
//!
//! The first downsample samples the top octave the prefilter just wrote, and
//! the last upsample writes it again. A backend whose resource states follow
//! the graph has to move the top octave between those, which is why the chain
//! is also encodable in three pieces.

use crate::gfx::render_types::PostProcessParams;
use crate::render::error::RenderResult;
use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage, bloom_top_desc,
    full_mip_levels,
};

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming, resolve_extent,
};
use super::program::{BLOOM_PREFILTER_CONSTANTS, PostProgram};

/// Fewest octaves the chain holds, the top one included.
pub const MIN_OCTAVES: u32 = 4;

/// Most octaves the chain holds, the top one included. Enough for a wide soft
/// glow without spending render passes on sub-pixel octaves.
pub const MAX_OCTAVES: u32 = 6;

/// Octaves in the chain for an output of `extent`, the top one included: one
/// fewer than the octaves down to a single texel on the shorter axis, since the
/// top octave is already half the output, clamped to
/// [`MIN_OCTAVES`]`..=`[`MAX_OCTAVES`].
pub fn octave_count(extent: PostExtent) -> u32 {
    let min_dim = extent.width.min(extent.height).max(1);
    let log2 = u32::BITS - 1 - min_dim.leading_zeros();
    log2.saturating_sub(1).clamp(MIN_OCTAVES, MAX_OCTAVES)
}

/// The extent of the top octave, the graph's `bloom_top`, for an output of
/// `extent`.
pub fn top_extent(extent: PostExtent) -> PostExtent {
    resolve_extent(&bloom_top_desc(), extent)
}

/// The shape of the octaves below the top one for an output of `extent`: a
/// quarter of the output, one mip level per octave.
///
/// A target cannot hold more levels than its full chain down to one texel, so
/// an output too small for [`octave_count`] gets as many as fit; only a window
/// a few pixels across is that small.
pub fn chain_desc(extent: PostExtent) -> TextureDesc {
    let quarter = TextureSize::DrawableScaled(0.25);
    let mut desc = TextureDesc {
        width: quarter,
        height: quarter,
        depth: 1,
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    };
    let base = resolve_extent(&desc, extent);
    desc.mip_levels = (octave_count(extent) - 1).min(full_mip_levels(base.width, base.height));
    desc
}

/// The per-frame inputs the chain reads and writes beyond its own octaves.
pub struct BloomInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The scene the prefilter thresholds.
    pub scene: D::TextureRef<'t>,
    /// The top octave, as the prefilter's and the last upsample's target.
    pub top: D::Attachment<'t>,
    /// The same octave, as the first downsample's source.
    pub top_ref: D::TextureRef<'t>,
}

/// The three pipelines, built together. What shader hot reload rebuilds and
/// hands to [`BloomPass::swap_pipelines`].
pub struct BloomPipelines<Pipeline> {
    /// The scene thresholded into the top octave.
    pub prefilter: Pipeline,
    /// One octave reduced into the next.
    pub downsample: Pipeline,
    /// One octave added into the octave above it.
    pub upsample: Pipeline,
}

/// Build every pipeline on its own, without touching the octaves.
pub fn build_pipelines<D: PostPassDevice>(device: &D) -> RenderResult<BloomPipelines<D::Pipeline>> {
    let format = PixelFormat::Rgba16Float;
    Ok(BloomPipelines {
        prefilter: device.create_pipeline(
            PostProgram::BloomPrefilter,
            format,
            PostBlend::Replace,
        )?,
        downsample: device.create_pipeline(
            PostProgram::BloomDownsample,
            format,
            PostBlend::Replace,
        )?,
        upsample: device.create_pipeline(
            PostProgram::BloomUpsample,
            format,
            PostBlend::Additive,
        )?,
    })
}

// The graph label the octaves below the top carry into a backend's own debug
// naming.
const CHAIN_LABEL: &str = "bloom_chain";

/// The bloom pipelines and the octaves below the top one.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the temporal resolve: a backend's device value borrows, and
/// the pass is stored on its context.
pub struct BloomPass<Pipeline, Target> {
    pipelines: BloomPipelines<Pipeline>,
    chain: Target,
    // Levels `chain` holds: the octaves below the top. Always at least one.
    levels: u32,
}

impl<Pipeline, Target> BloomPass<Pipeline, Target> {
    /// Build every pipeline and the octaves for an output of `extent`.
    pub fn new<D>(device: &D, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let desc = chain_desc(extent);
        Ok(Self {
            pipelines: build_pipelines(device)?,
            chain: device.create_target(CHAIN_LABEL, &desc, extent)?,
            levels: desc.mip_levels,
        })
    }

    /// Octaves in the chain, the top one included.
    pub fn octaves(&self) -> u32 {
        self.levels + 1
    }

    /// Recreate the octaves for an output of `extent`. The caller has already
    /// idled the device.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let desc = chain_desc(extent);
        self.chain = device.create_target(CHAIN_LABEL, &desc, extent)?;
        self.levels = desc.mip_levels;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: BloomPipelines<Pipeline>) {
        self.pipelines = pipelines;
    }

    /// Encode the whole chain: on return `inputs.top` holds the glow.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: BloomInputs<'t, D>,
        post: &PostProcessParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        self.encode_prefilter(device, rec, inputs.scene, inputs.top, post)?;
        self.encode_chain(device, rec, inputs.top_ref)?;
        self.encode_last_upsample(device, rec, inputs.top)
    }

    /// Encode the prefilter: `scene` thresholded into the top octave.
    pub fn encode_prefilter<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        scene: D::TextureRef<'t>,
        top: D::Attachment<'t>,
        post: &PostProcessParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        device.encode(
            rec,
            &PostDraw {
                target: top,
                // The top octave is the graph's `bloom_top`, which this node
                // writes.
                state: PostTargetState::Graph,
                load: PostLoadOp::DontCare,
                timing: PostTiming::First(PassId::Bloom),
                pipeline: &self.pipelines.prefilter,
                binds: &[linear::<D>(scene)],
                constants: prefilter_constants(post),
                label: "bloom prefilter",
            },
        )
    }

    /// Encode every downsample from `top` and every upsample back to the octave
    /// below the top: on return that octave holds the glow of every octave
    /// under it.
    pub fn encode_chain<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        top: D::TextureRef<'t>,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let chain = &self.chain;
        // Chain level `l` is octave `l + 1`; the first one reduces the top.
        for level in 0..self.levels {
            let source = match level {
                0 => top,
                _ => device.target_level_ref(chain, level - 1)?,
            };
            device.encode(
                rec,
                &private_draw(
                    device.target_level_attachment(chain, level)?,
                    PostLoadOp::DontCare,
                    &self.pipelines.downsample,
                    &[linear::<D>(source)],
                    "bloom downsample",
                ),
            )?;
        }
        for level in (0..self.levels.saturating_sub(1)).rev() {
            device.encode(
                rec,
                &private_draw(
                    device.target_level_attachment(chain, level)?,
                    // The blend adds onto the octave's own downsample.
                    PostLoadOp::Load,
                    &self.pipelines.upsample,
                    &[linear::<D>(device.target_level_ref(chain, level + 1)?)],
                    "bloom upsample",
                ),
            )?;
        }
        Ok(())
    }

    /// Encode the last upsample: the octave below the top added into `top`.
    pub fn encode_last_upsample<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        top: D::Attachment<'t>,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        device.encode(
            rec,
            &PostDraw {
                target: top,
                state: PostTargetState::Graph,
                load: PostLoadOp::Load,
                timing: PostTiming::Last(PassId::Bloom),
                pipeline: &self.pipelines.upsample,
                binds: &[linear::<D>(device.target_level_ref(&self.chain, 0)?)],
                constants: &[],
                label: "bloom upsample",
            },
        )
    }
}

/// The bytes of `post` the prefilter declares.
pub fn prefilter_constants(post: &PostProcessParams) -> &[u8] {
    &bytemuck::bytes_of(post)[..BLOOM_PREFILTER_CONSTANTS]
}

// A draw into one of the pass's own octaves, which only this pass reads.
fn private_draw<'a, 't, D: PostPassDevice + ?Sized + 't>(
    target: D::Attachment<'t>,
    load: PostLoadOp,
    pipeline: &'a D::Pipeline,
    binds: &'a [PostBind<'t, D>],
    label: &'a str,
) -> PostDraw<'a, 't, D> {
    PostDraw {
        target,
        state: PostTargetState::Pass,
        load,
        timing: PostTiming::None,
        pipeline,
        binds,
        constants: &[],
        label,
    }
}

fn linear<'t, D: PostPassDevice + ?Sized + 't>(texture: D::TextureRef<'t>) -> PostBind<'t, D> {
    PostBind {
        texture,
        sampler: PostSampler::LinearClamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::{MockDevice, MockDraw, MockPipeline, MockTexture};
    use alloc::vec::Vec;

    const EXTENT: PostExtent = PostExtent {
        width: 1920,
        height: 1080,
    };

    const SCENE: MockTexture = MockTexture::External(1);
    const TOP: MockTexture = MockTexture::External(2);

    // The chain target is the only one the pass creates.
    const CHAIN: usize = 0;

    fn post() -> PostProcessParams {
        PostProcessParams {
            bloom_intensity: 0.5,
            bloom_threshold: 1.1,
            bloom_knee: 0.4,
            exposure: 0.7,
            vignette: 0.3,
            lut_strength: 0.85,
            hdr_output: 1.0,
            pq_output: 1.0,
            fxaa: 1.0,
        }
    }

    fn encode_frame(device: &MockDevice, pass: &BloomPass<MockPipeline, usize>) -> Vec<MockDraw> {
        device.draws.borrow_mut().clear();
        pass.encode(
            device,
            &(),
            BloomInputs {
                scene: SCENE,
                top: TOP,
                top_ref: TOP,
            },
            &post(),
        )
        .expect("encode");
        device.draws.borrow().clone()
    }

    fn sources(draw: &MockDraw) -> Vec<MockTexture> {
        draw.binds.iter().map(|b| b.0).collect()
    }

    fn extent(width: u32, height: u32) -> PostExtent {
        PostExtent { width, height }
    }

    #[test]
    fn the_octave_count_clamps_to_four_to_six() {
        // Common resolutions land on the widest glow.
        assert_eq!(octave_count(extent(1920, 1080)), 6);
        assert_eq!(octave_count(extent(1280, 720)), 6);
        // Smaller outputs earn fewer octaves before the floor.
        assert_eq!(octave_count(extent(64, 64)), 5);
        assert_eq!(octave_count(extent(16, 16)), 4);
        assert_eq!(octave_count(extent(1, 1)), 4);
        assert_eq!(octave_count(extent(0, 0)), 4);
    }

    #[test]
    fn the_octave_count_follows_the_shorter_axis() {
        assert_eq!(octave_count(extent(4096, 64)), 5);
        assert_eq!(octave_count(extent(127, 4096)), 5);
        assert_eq!(octave_count(extent(128, 4096)), 6);
    }

    #[test]
    fn each_octave_below_the_top_is_half_the_one_above() {
        // The top octave is the pool's half-output `bloom_top`; octave `n`
        // below it is the output shifted right by `n + 1`, floored at a texel.
        let desc = chain_desc(extent(1921, 1081));
        assert_eq!(desc.mip_levels, 5);
        let base = resolve_extent(&desc, extent(1921, 1081));
        assert_eq!(base, extent(1921 >> 2, 1081 >> 2));
        assert_eq!(desc.format, PixelFormat::Rgba16Float);
        assert!(desc.usage.contains(TextureUsage::RENDER_TARGET));
        assert!(desc.usage.contains(TextureUsage::SHADER_READ));
    }

    #[test]
    fn the_chain_starts_one_octave_below_the_graphs_top() {
        for (w, h) in [(1920, 1080), (1921, 1081), (3, 1), (1, 1)] {
            let e = extent(w, h);
            let top = top_extent(e);
            assert_eq!(top, extent((w >> 1).max(1), (h >> 1).max(1)));
            let base = resolve_extent(&chain_desc(e), e);
            assert_eq!(base, extent((w >> 2).max(1), (h >> 2).max(1)));
            assert_eq!(bloom_top_desc().format, chain_desc(e).format);
        }
    }

    #[test]
    fn a_tiny_output_holds_only_the_levels_its_chain_has() {
        // Four octaves are asked for, but a one-texel chain has one level.
        assert_eq!(chain_desc(extent(1, 1)).mip_levels, 1);
        assert_eq!(chain_desc(extent(8, 8)).mip_levels, 2);
        assert_eq!(chain_desc(extent(16, 16)).mip_levels, 3);
    }

    #[test]
    fn a_frame_prefilters_then_walks_down_and_back_up() {
        let device = MockDevice::new();
        let pass = BloomPass::new(&device, EXTENT).expect("pass");
        assert_eq!(pass.octaves(), 6);
        let draws = encode_frame(&device, &pass);
        let programs: Vec<PostProgram> = draws.iter().map(|d| d.program).collect();
        let mut expected = alloc::vec![PostProgram::BloomPrefilter];
        expected.extend([PostProgram::BloomDownsample; 5]);
        expected.extend([PostProgram::BloomUpsample; 5]);
        assert_eq!(programs, expected);

        assert_eq!(draws[0].target, TOP);
        assert_eq!(sources(&draws[0]), [SCENE]);
        // Downsample into chain level `l` from the level above it, the first
        // one from the top octave.
        assert_eq!(draws[1].target, MockTexture::Level(CHAIN, 0));
        assert_eq!(sources(&draws[1]), [TOP]);
        for level in 1..5u32 {
            let d = &draws[1 + level as usize];
            assert_eq!(d.target, MockTexture::Level(CHAIN, level));
            assert_eq!(sources(d), [MockTexture::Level(CHAIN, level - 1)]);
        }
        // Upsample back up, coarsest first, ending in the top octave.
        for (i, level) in (0..4u32).rev().enumerate() {
            let d = &draws[6 + i];
            assert_eq!(d.target, MockTexture::Level(CHAIN, level));
            assert_eq!(sources(d), [MockTexture::Level(CHAIN, level + 1)]);
        }
        let last = draws.last().expect("draws");
        assert_eq!(last.target, TOP);
        assert_eq!(sources(last), [MockTexture::Level(CHAIN, 0)]);
    }

    #[test]
    fn only_the_top_octave_is_the_graphs_and_only_upsamples_load() {
        let device = MockDevice::new();
        let pass = BloomPass::new(&device, EXTENT).expect("pass");
        for d in encode_frame(&device, &pass) {
            let graph = d.target == TOP;
            let state = if graph {
                PostTargetState::Graph
            } else {
                PostTargetState::Pass
            };
            assert_eq!(d.state, state, "{}", d.label);
            let load = if d.program == PostProgram::BloomUpsample {
                PostLoadOp::Load
            } else {
                PostLoadOp::DontCare
            };
            assert_eq!(d.load, load, "{}", d.label);
            assert!(d.binds.iter().all(|b| b.1 == PostSampler::LinearClamp));
        }
        let p = build_pipelines(&device).expect("pipelines");
        assert_eq!(p.prefilter.blend, PostBlend::Replace);
        assert_eq!(p.downsample.blend, PostBlend::Replace);
        assert_eq!(p.upsample.blend, PostBlend::Additive);
        for pipeline in [p.prefilter, p.downsample, p.upsample] {
            assert_eq!(pipeline.format, PixelFormat::Rgba16Float);
        }
    }

    #[test]
    fn only_the_prefilter_reads_the_tunables() {
        let device = MockDevice::new();
        let pass = BloomPass::new(&device, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let p = post();
        assert_eq!(
            draws[0].constants,
            &bytemuck::bytes_of(&p)[..BLOOM_PREFILTER_CONSTANTS]
        );
        assert_eq!(draws[0].constants.len(), 24);
        assert!(draws[1..].iter().all(|d| d.constants.is_empty()));
    }

    #[test]
    fn the_timing_span_opens_on_the_prefilter_and_closes_on_the_last_upsample() {
        let device = MockDevice::new();
        let pass = BloomPass::new(&device, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let timings: Vec<PostTiming> = draws.iter().map(|d| d.timing).collect();
        assert_eq!(timings[0], PostTiming::First(PassId::Bloom));
        assert_eq!(timings[timings.len() - 1], PostTiming::Last(PassId::Bloom));
        assert!(
            timings[1..timings.len() - 1]
                .iter()
                .all(|t| *t == PostTiming::None)
        );
    }

    #[test]
    fn the_pieces_encode_what_the_whole_does() {
        let whole_device = MockDevice::new();
        let whole = encode_frame(
            &whole_device,
            &BloomPass::new(&whole_device, EXTENT).expect("pass"),
        );
        let device = MockDevice::new();
        let pass = BloomPass::new(&device, EXTENT).expect("pass");
        pass.encode_prefilter(&device, &(), SCENE, TOP, &post())
            .expect("prefilter");
        pass.encode_chain(&device, &(), TOP).expect("chain");
        pass.encode_last_upsample(&device, &(), TOP)
            .expect("last upsample");
        let pieces = device.draws.borrow();
        assert_eq!(pieces.len(), whole.len());
        for (a, b) in pieces.iter().zip(&whole) {
            assert_eq!((a.program, a.target), (b.program, b.target));
            assert_eq!(a.binds, b.binds);
        }
    }

    #[test]
    fn a_tiny_output_still_encodes_a_whole_chain() {
        let device = MockDevice::new();
        let pass = BloomPass::new(&device, extent(1, 1)).expect("pass");
        assert_eq!(pass.octaves(), 2);
        let draws = encode_frame(&device, &pass);
        let programs: Vec<PostProgram> = draws.iter().map(|d| d.program).collect();
        assert_eq!(
            programs,
            [
                PostProgram::BloomPrefilter,
                PostProgram::BloomDownsample,
                PostProgram::BloomUpsample,
            ]
        );
    }

    #[test]
    fn a_resize_recreates_the_octaves_at_the_new_output() {
        let device = MockDevice::new();
        let mut pass = BloomPass::new(&device, EXTENT).expect("pass");
        pass.resize(&device, extent(64, 64)).expect("resize");
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[1].extent, extent(16, 16));
        assert_eq!(targets[1].levels, 4);
        assert_eq!(pass.octaves(), 5);
    }
}
