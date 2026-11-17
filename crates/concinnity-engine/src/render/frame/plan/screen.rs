//! What the main pass reads beside the scene, and how much of it it blends in.

use crate::gfx::render_types::ReflectionScopeGpu;
use crate::render::relevance::{Effect, Fades};
use crate::render::render_graph::{PassDecisions, PassId};

/// What the main pass reads beside the scene this frame, as the pass
/// selection left it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScreenReads {
    /// A reflection resolve composites over glossy surfaces, which hand it
    /// their sharp specular.
    pub reflections: bool,
    /// The screen-space indirect target is written: the ambient occlusion, or
    /// the bounce's resolve that folds it in.
    pub indirect: bool,
}

impl ScreenReads {
    /// What `passes` leave the main pass to read, where a built reflection
    /// composite `composites` the resolve that takes the reflection slot.
    pub fn of(passes: &PassDecisions, composites: bool) -> Self {
        let runs = |pass| passes.get(pass).runs;
        Self {
            reflections: composites && (runs(PassId::RtReflections) || runs(PassId::SsrResolve)),
            indirect: runs(PassId::SsaoBlur) || runs(PassId::SsgiResolve),
        }
    }
}

/// How much of each screen-space effect the main pass blends in, from 0 for
/// none to 1 in full: what it reads, scaled by how much of the effect shows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScreenBlend {
    /// The reflection resolve's composite over glossy surfaces, which they
    /// hand their share of their sharp specular, and where on screen.
    pub reflections: ReflectionScopeGpu,
    /// The screen-space indirect target's occlusion.
    pub occlusion: f32,
    /// The bounce that target carries beside the occlusion.
    pub bounce: f32,
}

impl ScreenBlend {
    /// The blend of what `reads` names under `fades`, the reflections over
    /// `scope`, where the indirect target carries a bounce when `bounce` is
    /// built.
    pub fn of(reads: ScreenReads, fades: &Fades, scope: ReflectionScopeGpu, bounce: bool) -> Self {
        let occlusion = if reads.indirect {
            fades.get(Effect::AmbientOcclusion)
        } else {
            0.0
        };
        Self {
            reflections: if reads.reflections {
                scope
            } else {
                ReflectionScopeGpu::NONE
            },
            occlusion,
            bounce: if bounce {
                occlusion * fades.get(Effect::IndirectLighting)
            } else {
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::view_modes::{ShowFlags, ViewMode};
    use crate::render::relevance::{Margins, RelevanceState, Relevant};
    use crate::render::render_graph::{FeatureGates, PassInputs};

    // The main pass reads the occlusion target and the reflection composite only
    // on frames whose passes write them.
    #[test]
    fn the_screen_reads_follow_the_passes_that_ran() {
        let inputs = PassInputs {
            gates: FeatureGates {
                ssao: true,
                ssr: true,
                ..FeatureGates::default()
            },
            mode: ViewMode::Lit,
            show: ShowFlags::all(),
            ..PassInputs::default()
        };
        let all = PassDecisions::select(&inputs);
        assert_eq!(
            ScreenReads::of(&all, true),
            ScreenReads {
                reflections: true,
                indirect: true,
            }
        );
        // A composite that was never built has nothing to read.
        assert!(!ScreenReads::of(&all, false).reflections);
        let none = PassDecisions::select(&PassInputs {
            relevant: Relevant::ALL
                .with(Effect::AmbientOcclusion, false)
                .with(Effect::Reflections, false),
            ..inputs
        });
        assert_eq!(ScreenReads::of(&none, true), ScreenReads::default());
    }

    // The main pass blends in what it reads by how much of each effect shows,
    // the bounce only as far as the occlusion it rides with, and the
    // reflections over the part of the screen they show on.
    #[test]
    fn the_screen_blend_scales_what_is_read_by_the_fades() {
        let both = ScreenReads {
            reflections: true,
            indirect: true,
        };
        let full = ReflectionScopeGpu::FULL;
        assert_eq!(
            ScreenBlend::of(both, &Fades::FULL, full, true),
            ScreenBlend {
                reflections: full,
                occlusion: 1.0,
                bounce: 1.0,
            }
        );
        assert_eq!(ScreenBlend::of(both, &Fades::FULL, full, false).bounce, 0.0);
        assert_eq!(
            ScreenBlend::of(ScreenReads::default(), &Fades::FULL, full, true),
            ScreenBlend::default()
        );

        let mut state = RelevanceState::default();
        let failing = Margins::UNBOUNDED
            .with(Effect::AmbientOcclusion, 0.0)
            .with(Effect::Reflections, 0.0);
        state.update(&failing, 0.0);
        state.update(&failing, 0.25);
        let scope = state.reflection_scope().gpu([64, 64]);
        let half = ScreenBlend::of(both, &state.fades(), scope, true);
        assert!((half.reflections.whole - 0.5).abs() < 1e-4, "{half:?}");
        assert!((half.occlusion - 0.5).abs() < 1e-4, "{half:?}");
        assert!((half.bounce - 0.5).abs() < 1e-4, "{half:?}");
    }
}
