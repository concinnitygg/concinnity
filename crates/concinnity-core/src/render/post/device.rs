//! The backend seam a shared fullscreen post pass encodes through.
//!
//! Three operations, modeled on what a post pass on the thinnest of the three
//! backends already does: build a pipeline from a program identity plus an
//! output format, create a persistent target from a render-graph texture
//! description, and encode one fullscreen draw. Everything a pass needs beyond
//! those (its own state, its target lifecycle, what it binds where) is portable
//! and lives in the pass.
//!
//! Three associated types absorb the divergence without naming a backend type:
//! `Recorder` is the per-backend command recorder, `TextureRef` is whatever that
//! backend binds a sampled source by (a texture object, an image view, a
//! descriptor handle), and `Attachment` is whatever it writes a draw through.
//! Both reference types borrow from the value they name, so a pass can bind or
//! write a target it created here beside one another subsystem owns, which is
//! what every post pass actually does: its own buffers plus the scene and
//! G-buffer channels somebody else produced.

use alloc::format;
use alloc::string::String;

use crate::render::render_graph::{PassId, PixelFormat, TextureDesc, TransientTexture};

use super::program::{PostProgram, PostProgramBindings};

/// Blending on a post pass's single color attachment.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PostBlend {
    /// The fragment replaces the destination. Every pass that writes a fresh
    /// target.
    Replace,
    /// Additive accumulation onto content the pass loaded.
    Additive,
    /// Premultiplied "over": the fragment already folded coverage into color.
    PremultipliedOver,
}

/// What happens to a target's existing contents at the head of the pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PostLoadOp {
    /// The draw covers every pixel, so the previous contents are discarded.
    DontCare,
    /// The previous contents are preserved and blended into.
    Load,
}

/// Which sampler a bound source is read through.
///
/// The seam names the kind rather than passing a backend sampler object across
/// it, so a pass says what it needs and each host supplies its own state. A
/// kind is added when a pass asks for one rather than present and silently
/// resolving to another.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PostSampler {
    /// Bilinear filtering, clamped to the edge. What every screen-space source
    /// is read through.
    LinearClamp,
    /// Trilinear filtering across the whole mip chain, clamped to the edge.
    /// What a prefiltered environment cube is read through, where the mip
    /// level carries the surface roughness.
    LinearCube,
}

/// Who moves a draw's target into its render state and back out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PostTargetState {
    /// The render graph declares the target, so its executor has already put
    /// it in the render state and the next consumer's transition takes it back.
    Graph,
    /// The target is private to the pass. It rests readable between passes, so
    /// the draw moves it in and out itself.
    Pass,
}

/// Where a draw sits within its effect's GPU-timing span.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PostTiming {
    /// The draw records no timing sample.
    None,
    /// The effect's only draw: both its start and end samples land here.
    Whole(PassId),
    /// The first draw of a multi-draw effect: the start sample.
    First(PassId),
    /// The last draw of a multi-draw effect: the end sample.
    Last(PassId),
}

/// One sampled source of a post draw, at its slot in declaration order.
pub struct PostBind<'t, D: PostPassDevice + ?Sized + 't> {
    /// The texture this slot samples.
    pub texture: D::TextureRef<'t>,
    /// The sampler state it is read through.
    pub sampler: PostSampler,
}

// Derived by hand: `TextureRef` is `Copy` but a `#[derive(Clone)]` would demand
// `D: Clone` as well, which no backend device is.
impl<D: PostPassDevice + ?Sized> Clone for PostBind<'_, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: PostPassDevice + ?Sized> Copy for PostBind<'_, D> {}

/// Everything one fullscreen post draw needs: where it writes, what it runs,
/// what it binds, and where it sits in the GPU-timing span.
pub struct PostDraw<'a, 't, D: PostPassDevice + ?Sized + 't> {
    /// The color target the draw writes.
    pub target: D::Attachment<'t>,
    /// Who transitions that target around the draw.
    pub state: PostTargetState,
    /// What happens to that target's contents on load.
    pub load: PostLoadOp,
    /// Where the draw sits in its effect's GPU-timing span.
    pub timing: PostTiming,
    /// The pipeline to run.
    pub pipeline: &'a D::Pipeline,
    /// Sampled sources, in the program's declaration order. Its length must be
    /// the program's declared texture count.
    pub binds: &'a [PostBind<'t, D>],
    /// The constants blob, in the program's declared layout. Its length must be
    /// the program's declared constant size, and it is empty when the program
    /// declares none.
    pub constants: &'a [u8],
    /// Debug label for the encoder / marker region.
    pub label: &'a str,
}

impl<D: PostPassDevice + ?Sized> PostDraw<'_, '_, D> {
    /// Whether the draw hands over exactly the sources and constants `declared`
    /// says its program binds. A mismatch would bind a slot the shader does not
    /// read or leave one it does read unbound.
    pub fn check(&self, declared: PostProgramBindings) -> Result<(), String> {
        if self.binds.len() == declared.textures && self.constants.len() == declared.constants {
            return Ok(());
        }
        Err(format!(
            "{}: the draw binds {} texture(s) and {} constant byte(s) where the program \
             declares {} and {}",
            self.label,
            self.binds.len(),
            self.constants.len(),
            declared.textures,
            declared.constants,
        ))
    }
}

/// The pixel size a target is created at, after the graph's fractional sizes
/// are resolved.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PostExtent {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl PostExtent {
    /// The extent, with both axes floored to at least one pixel so a minimized
    /// or mid-resize window never asks a backend for a zero-sized texture.
    pub fn clamped(self) -> Self {
        Self {
            width: self.width.max(1),
            height: self.height.max(1),
        }
    }
}

/// A backend's implementation of the operations a fullscreen post pass needs.
/// Implemented once per backend; the passes above it are written once.
///
/// Every method takes `&self`, matching the read-only parallel-encode contract
/// the graph executors record under: a backend that needs interior state (a
/// cached render pass, a per-frame descriptor cursor) owns that state's
/// synchronization itself.
pub trait PostPassDevice {
    /// The per-backend command recorder a draw is encoded into.
    type Recorder: ?Sized;
    /// A built fullscreen pipeline.
    type Pipeline;
    /// A persistent target created through [`PostPassDevice::create_target`].
    type Target;
    /// How this backend names a sampled source: whatever a bind takes, borrowed
    /// from the value that owns it.
    type TextureRef<'a>: Copy;
    /// How this backend names a draw's color target: whatever a render pass
    /// writes through, borrowed from the value that owns it.
    type Attachment<'a>: Copy;

    /// Build a fullscreen-triangle pipeline running `program`'s fragment against
    /// a single color attachment of `format` with `blend`.
    ///
    /// A program that declares the reflection-probe set gets the backend's own
    /// probe bindings laid out after its declared ones.
    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend: PostBlend,
    ) -> Result<Self::Pipeline, String>;

    /// Create one persistent target from a render-graph texture description,
    /// with its fractional sizes resolved against `extent`. Persistent rather
    /// than pooled because a temporal pass's accumulation buffers must survive
    /// the frame that wrote them, which is exactly what the transient pool's
    /// aliaser is free to break.
    fn create_target(
        &self,
        label: &'static str,
        desc: &TextureDesc,
        extent: PostExtent,
    ) -> Result<Self::Target, String>;

    /// Bind `target` as a sampled source. A temporal pass reads the slot it
    /// wrote last frame, so a created target has to be nameable as an input.
    fn target_ref<'a>(&self, target: &'a Self::Target) -> Self::TextureRef<'a>;

    /// Name `target` as a draw's color attachment.
    fn target_attachment<'a>(&self, target: &'a Self::Target) -> Self::Attachment<'a>;

    /// Encode one fullscreen draw.
    ///
    /// When the pipeline's program declares the reflection-probe set, the
    /// device binds the one the world holds this frame: the probe records and
    /// the cube array together, which no pass chooses between.
    fn encode(&self, rec: &Self::Recorder, draw: &PostDraw<'_, '_, Self>) -> Result<(), String>;
}

/// Resolve a render-graph texture description's fractional sizes against a
/// drawable extent, the way every backend's transient pool already does for a
/// pooled resource.
pub fn resolve_extent(desc: &TextureDesc, extent: PostExtent) -> PostExtent {
    let extent = extent.clamped();
    PostExtent {
        width: desc.width.resolve(extent.width),
        height: desc.height.resolve(extent.height),
    }
}

/// The same description with its sizes resolved, in the shape each backend's
/// transient pool already knows how to turn into a native texture descriptor.
///
/// A post pass's targets are persistent rather than pooled -- a temporal pass
/// accumulates across frames, which is exactly what the aliaser is free to break
/// -- but they are the same kind of resource, so they should go through the same
/// translation. Handing back a [`TransientTexture`] means no backend grows a
/// second table of format and usage mappings that could drift from the pool's.
pub fn resolved_texture(
    label: &'static str,
    desc: &TextureDesc,
    extent: PostExtent,
) -> TransientTexture {
    let PostExtent { width, height } = resolve_extent(desc, extent);
    TransientTexture {
        label,
        width,
        height,
        depth: desc.depth.max(1),
        format: desc.format,
        sample_count: desc.sample_count.max(1),
        array_layers: desc.array_layers.max(1),
        mip_levels: desc.mip_levels.max(1),
        usage: desc.usage,
        clear: desc.clear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::{MockDevice, MockTexture};
    use crate::render::render_graph::{ClearValue, TextureSize, TextureUsage};

    fn desc(width: TextureSize, height: TextureSize) -> TextureDesc {
        TextureDesc {
            width,
            height,
            depth: 1,
            format: PixelFormat::Rgba16Float,
            sample_count: 1,
            array_layers: 1,
            mip_levels: 1,
            usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            clear: ClearValue::Color([0.0; 4]),
        }
    }

    #[test]
    fn a_drawable_sized_target_follows_the_extent() {
        let d = desc(TextureSize::Drawable, TextureSize::Drawable);
        assert_eq!(
            resolve_extent(
                &d,
                PostExtent {
                    width: 1920,
                    height: 1080
                }
            ),
            PostExtent {
                width: 1920,
                height: 1080
            }
        );
    }

    #[test]
    fn a_scaled_target_floors_to_one_pixel() {
        let d = desc(
            TextureSize::DrawableScaled(0.5),
            TextureSize::DrawableScaled(0.5),
        );
        assert_eq!(
            resolve_extent(
                &d,
                PostExtent {
                    width: 3,
                    height: 1
                }
            ),
            PostExtent {
                width: 1,
                height: 1
            }
        );
    }

    #[test]
    fn an_absolute_target_ignores_the_extent() {
        let d = desc(TextureSize::Absolute(2048), TextureSize::Absolute(512));
        assert_eq!(
            resolve_extent(
                &d,
                PostExtent {
                    width: 800,
                    height: 600
                }
            ),
            PostExtent {
                width: 2048,
                height: 512
            }
        );
    }

    #[test]
    fn a_zero_extent_never_reaches_a_backend() {
        // Minimized or mid-resize: the drawable is zero, and a texture of that
        // size is a creation failure on every backend.
        let d = desc(TextureSize::Drawable, TextureSize::Drawable);
        assert_eq!(
            resolve_extent(
                &d,
                PostExtent {
                    width: 0,
                    height: 0
                }
            ),
            PostExtent {
                width: 1,
                height: 1
            }
        );
    }

    fn draw_with<'a>(
        pipeline: &'a <MockDevice as PostPassDevice>::Pipeline,
        binds: &'a [PostBind<'a, MockDevice>],
        constants: &'a [u8],
    ) -> PostDraw<'a, 'a, MockDevice> {
        PostDraw {
            target: MockTexture::External(0),
            state: PostTargetState::Pass,
            load: PostLoadOp::DontCare,
            timing: PostTiming::None,
            pipeline,
            binds,
            constants,
            label: "probe",
        }
    }

    #[test]
    fn a_draw_matching_its_declaration_passes_the_check() {
        let device = MockDevice::new();
        let pipeline = device
            .create_pipeline(
                PostProgram::TaaResolve,
                PixelFormat::Rgba16Float,
                PostBlend::Replace,
            )
            .expect("mock pipeline");
        let bind = PostBind {
            texture: MockTexture::External(1),
            sampler: PostSampler::LinearClamp,
        };
        let binds = [bind; 3];
        let constants = [0u8; 4];
        let draw = draw_with(&pipeline, &binds, &constants);
        assert!(draw.check(PostProgram::TaaResolve.bindings()).is_ok());
    }

    #[test]
    fn a_short_bind_list_or_constants_blob_fails_the_check() {
        let device = MockDevice::new();
        let pipeline = device
            .create_pipeline(
                PostProgram::TaaResolve,
                PixelFormat::Rgba16Float,
                PostBlend::Replace,
            )
            .expect("mock pipeline");
        let bind = PostBind {
            texture: MockTexture::External(1),
            sampler: PostSampler::LinearClamp,
        };
        let declared = PostProgram::TaaResolve.bindings();
        let two = [bind; 2];
        let three = [bind; 3];
        let err = draw_with(&pipeline, &two, &[0u8; 4])
            .check(declared)
            .expect_err("one source short");
        assert!(err.contains("2 texture(s)"), "{err}");
        assert!(
            draw_with(&pipeline, &three, &[]).check(declared).is_err(),
            "missing constants"
        );
    }
}
