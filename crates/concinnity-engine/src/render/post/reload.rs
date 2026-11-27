//! The engine-owned post passes' pipelines as a shader rebuild makes them:
//! each pass is rebuilt from the current sources the same way on every
//! backend, so the builds are declared once here and a backend only says
//! which passes are live and where their pipelines go.

use super::bloom::{self, BloomPipelines};
use super::device::PostPassDevice;
use super::disocclusion::{self, DisocclusionPipelines};
use super::reflection_composite::{self, ReflectionCompositePipelines};
use super::ssao::{self, SsaoPipelines};
use super::ssgi::{self, SsgiPipelines};
use super::ssr;
use super::taa::{self, TaaPipelines};
use crate::render::backend::error::RenderResult;
use alloc::vec::Vec;

/// One engine-owned post pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostPass {
    /// The temporal resolve.
    Taa,
    /// The bloom chain.
    Bloom,
    /// Ambient occlusion.
    Ssao,
    /// The screen-space reflection resolve.
    Ssr,
    /// Screen-space indirect light.
    Ssgi,
    /// The disocclusion mask.
    Disocclusion,
    /// The reflection blur and composite.
    ReflectionComposite,
}

/// Every engine-owned post pass, in the order a rebuild builds them.
pub const POST_PASSES: [PostPass; 7] = [
    PostPass::Taa,
    PostPass::Bloom,
    PostPass::Ssao,
    PostPass::Ssr,
    PostPass::Ssgi,
    PostPass::Disocclusion,
    PostPass::ReflectionComposite,
];

/// One post pass's rebuilt pipelines.
pub enum PostPipelines<P> {
    /// [`PostPass::Taa`]'s.
    Taa(TaaPipelines<P>),
    /// [`PostPass::Bloom`]'s.
    Bloom(BloomPipelines<P>),
    /// [`PostPass::Ssao`]'s.
    Ssao(SsaoPipelines<P>),
    /// [`PostPass::Ssr`]'s.
    Ssr(P),
    /// [`PostPass::Ssgi`]'s.
    Ssgi(SsgiPipelines<P>),
    /// [`PostPass::Disocclusion`]'s.
    Disocclusion(DisocclusionPipelines<P>),
    /// [`PostPass::ReflectionComposite`]'s.
    ReflectionComposite(ReflectionCompositePipelines<P>),
}

#[cfg(test)]
impl<P> PostPipelines<P> {
    // The pass these pipelines belong to.
    fn pass(&self) -> PostPass {
        match self {
            Self::Taa(_) => PostPass::Taa,
            Self::Bloom(_) => PostPass::Bloom,
            Self::Ssao(_) => PostPass::Ssao,
            Self::Ssr(_) => PostPass::Ssr,
            Self::Ssgi(_) => PostPass::Ssgi,
            Self::Disocclusion(_) => PostPass::Disocclusion,
            Self::ReflectionComposite(_) => PostPass::ReflectionComposite,
        }
    }
}

// `pass`'s pipelines built through `device`.
fn build<D: PostPassDevice>(
    device: &D,
    pass: PostPass,
) -> RenderResult<PostPipelines<D::Pipeline>> {
    Ok(match pass {
        PostPass::Taa => PostPipelines::Taa(taa::build_pipelines(device)?),
        PostPass::Bloom => PostPipelines::Bloom(bloom::build_pipelines(device)?),
        PostPass::Ssao => PostPipelines::Ssao(ssao::build_pipelines(device)?),
        PostPass::Ssr => PostPipelines::Ssr(ssr::build_pipeline(device)?),
        PostPass::Ssgi => PostPipelines::Ssgi(ssgi::build_pipelines(device)?),
        PostPass::Disocclusion => {
            PostPipelines::Disocclusion(disocclusion::build_pipelines(device)?)
        }
        PostPass::ReflectionComposite => {
            PostPipelines::ReflectionComposite(reflection_composite::build_pipelines(device)?)
        }
    })
}

/// The pipelines of every pass `live` names, built through `device`.
///
/// # Errors
///
/// Returns the first pipeline that fails to build.
pub fn build_live<D: PostPassDevice>(
    device: &D,
    live: impl Fn(PostPass) -> bool,
) -> RenderResult<Vec<PostPipelines<D::Pipeline>>> {
    POST_PASSES
        .into_iter()
        .filter(|&pass| live(pass))
        .map(|pass| build(device, pass))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::MockDevice;

    #[test]
    fn each_pass_builds_its_own_pipelines() {
        let device = MockDevice::new();
        for pass in POST_PASSES {
            assert_eq!(build(&device, pass).expect("builds").pass(), pass);
        }
    }

    #[test]
    fn only_the_live_passes_are_built() {
        let live = |pass| matches!(pass, PostPass::Bloom | PostPass::Ssr);
        let built = build_live(&MockDevice::new(), live).expect("builds");
        let passes: Vec<PostPass> = built.iter().map(PostPipelines::pass).collect();
        assert_eq!(passes, [PostPass::Bloom, PostPass::Ssr]);
    }
}
