//! Which upscaler a world's request resolves to, and the order the backends
//! fall back through when one fails to initialize.

use concinnity_core::components::UpscalerBackend;
use concinnity_core::render::backend_init::PostSettings;
use concinnity_core::render::dlss::DlssPreset;
use concinnity_core::render::error::RenderResult;

/// What temporal upscaling was asked for: the upscaler, and the render preset
/// DLSS runs if it is the one built. Kept by a backend so a resize rebuilds
/// the same, skipping the candidates that already failed to initialize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UpscaleRequest {
    pub(crate) backend: UpscalerBackend,
    pub(crate) dlss_preset: DlssPreset,
    refused: Availability,
}

impl UpscaleRequest {
    /// The request a backend's post settings make.
    pub(crate) fn from_post(post: &PostSettings) -> Self {
        Self {
            backend: post.upscale_backend,
            dlss_preset: post.dlss_preset,
            refused: Availability::NONE,
        }
    }

    /// Keep the preset a DLSS fallback settled on, so a rebuild (every resize)
    /// goes straight to it instead of repeating the one NGX refused. `running`
    /// is `None` when the built upscaler is not DLSS.
    pub(crate) fn adopt_running_preset(&mut self, running: Option<DlssPreset>) {
        if let Some(preset) = running {
            self.dlss_preset = preset;
        }
    }
}

/// The upscaler a backend built, or native-resolution rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedBackend {
    Fsr,
    Dlss,
    Xess,
    Native,
}

/// Which upscaler SDKs this build can try.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Availability {
    pub(crate) dlss: bool,
    pub(crate) xess: bool,
    pub(crate) fsr: bool,
}

impl Availability {
    const NONE: Self = Self {
        dlss: false,
        xess: false,
        fsr: false,
    };

    fn without(self, refused: Self) -> Self {
        Self {
            dlss: self.dlss && !refused.dlss,
            xess: self.xess && !refused.xess,
            fsr: self.fsr && !refused.fsr,
        }
    }

    fn mark(&mut self, candidate: ResolvedBackend) {
        match candidate {
            ResolvedBackend::Dlss => self.dlss = true,
            ResolvedBackend::Xess => self.xess = true,
            ResolvedBackend::Fsr => self.fsr = true,
            ResolvedBackend::Native => {}
        }
    }
}

// The candidates in the order they are tried: the explicitly requested one
// first when available, then the automatic priority (DLSS, XeSS, FSR), then
// Native, which is always last and always available.
fn backend_order(requested: UpscalerBackend, available: Availability) -> Vec<ResolvedBackend> {
    let mut order = Vec::with_capacity(4);
    match requested {
        UpscalerBackend::Dlss if available.dlss => order.push(ResolvedBackend::Dlss),
        UpscalerBackend::Xess if available.xess => order.push(ResolvedBackend::Xess),
        UpscalerBackend::Fsr3 if available.fsr => order.push(ResolvedBackend::Fsr),
        _ => {}
    }
    for (candidate, avail) in [
        (ResolvedBackend::Dlss, available.dlss),
        (ResolvedBackend::Xess, available.xess),
        (ResolvedBackend::Fsr, available.fsr),
    ] {
        if avail && !order.contains(&candidate) {
            order.push(candidate);
        }
    }
    order.push(ResolvedBackend::Native);
    order
}

/// The first candidate `requested` resolves to before any initialization is
/// tried.
#[cfg(any(backend_vk, test))]
pub(crate) fn preferred(requested: UpscalerBackend, available: Availability) -> ResolvedBackend {
    backend_order(requested, available)
        .first()
        .copied()
        .unwrap_or(ResolvedBackend::Native)
}

impl UpscaleRequest {
    /// Build the first candidate `build` initializes, in fallback order, and
    /// name it; `None` with `Native` when every vendor candidate failed. A
    /// candidate that fails is not tried again by later builds from this
    /// request. `build` is never called for `Native`, and an error from it ends
    /// the search.
    pub(crate) fn build_first_available<T>(
        &mut self,
        available: Availability,
        output: (u32, u32),
        mut build: impl FnMut(ResolvedBackend, DlssPreset) -> RenderResult<Option<T>>,
    ) -> RenderResult<(Option<T>, ResolvedBackend)> {
        let (width, height) = output;
        for candidate in backend_order(self.backend, available.without(self.refused)) {
            if candidate == ResolvedBackend::Native {
                break;
            }
            if let Some(built) = build(candidate, self.dlss_preset)? {
                tracing::info!(
                    "temporal upscaling: using {candidate:?} backend (output {width}x{height})"
                );
                return Ok((Some(built), candidate));
            }
            self.refused.mark(candidate);
            tracing::warn!("temporal upscaling: {candidate:?} unavailable, trying next backend");
        }
        tracing::info!("temporal upscaling: no backend available, rendering at native resolution");
        Ok((None, ResolvedBackend::Native))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::UpscalerBackend as B;

    fn avail(dlss: bool, xess: bool, fsr: bool) -> Availability {
        Availability { dlss, xess, fsr }
    }

    fn request(backend: UpscalerBackend) -> UpscaleRequest {
        UpscaleRequest {
            backend,
            dlss_preset: DlssPreset::Default,
            refused: Availability::NONE,
        }
    }

    #[test]
    fn auto_prefers_dlss_then_xess_then_fsr_then_native() {
        assert_eq!(
            preferred(B::Auto, avail(true, true, true)),
            ResolvedBackend::Dlss
        );
        assert_eq!(
            preferred(B::Auto, avail(false, true, true)),
            ResolvedBackend::Xess
        );
        assert_eq!(
            preferred(B::Auto, avail(false, false, true)),
            ResolvedBackend::Fsr
        );
        assert_eq!(
            preferred(B::Auto, avail(false, false, false)),
            ResolvedBackend::Native
        );
    }

    #[test]
    fn explicit_choice_used_when_available() {
        assert_eq!(
            preferred(B::Dlss, avail(true, true, true)),
            ResolvedBackend::Dlss
        );
        assert_eq!(
            preferred(B::Xess, avail(true, true, true)),
            ResolvedBackend::Xess
        );
        assert_eq!(
            preferred(B::Fsr3, avail(true, true, true)),
            ResolvedBackend::Fsr
        );
    }

    #[test]
    fn explicit_choice_falls_through_when_unavailable() {
        assert_eq!(
            preferred(B::Dlss, avail(false, true, true)),
            ResolvedBackend::Xess
        );
        assert_eq!(
            preferred(B::Xess, avail(false, false, true)),
            ResolvedBackend::Fsr
        );
        assert_eq!(
            preferred(B::Fsr3, avail(false, false, false)),
            ResolvedBackend::Native
        );
    }

    #[test]
    fn every_order_lists_each_candidate_once_and_ends_native() {
        let order = backend_order(B::Fsr3, avail(true, true, true));
        assert_eq!(
            order,
            [
                ResolvedBackend::Fsr,
                ResolvedBackend::Dlss,
                ResolvedBackend::Xess,
                ResolvedBackend::Native
            ]
        );
    }

    #[test]
    fn a_rebuild_keeps_the_preset_dlss_settled_on() {
        let mut request = request(B::Dlss);
        request.dlss_preset = DlssPreset::M;
        request.adopt_running_preset(None);
        assert_eq!(request.dlss_preset, DlssPreset::M);
        request.adopt_running_preset(Some(DlssPreset::Default));
        assert_eq!(request.dlss_preset, DlssPreset::Default);
        assert_eq!(request.backend, B::Dlss);
    }

    #[test]
    fn a_failed_candidate_falls_through_to_the_next() {
        let mut tried = Vec::new();
        let (built, resolved) = request(B::Dlss)
            .build_first_available(avail(true, true, true), (8, 8), |c, _| {
                tried.push(c);
                Ok((c == ResolvedBackend::Fsr).then_some(c))
            })
            .unwrap();
        assert_eq!(
            (built, resolved),
            (Some(ResolvedBackend::Fsr), ResolvedBackend::Fsr)
        );
        assert_eq!(
            tried,
            [
                ResolvedBackend::Dlss,
                ResolvedBackend::Xess,
                ResolvedBackend::Fsr
            ]
        );
    }

    #[test]
    fn nothing_initializing_resolves_to_native_without_building_it() {
        let mut tried = Vec::new();
        let (built, resolved) = request(B::Auto)
            .build_first_available(avail(false, true, true), (8, 8), |c, _| {
                tried.push(c);
                Ok(None::<()>)
            })
            .unwrap();
        assert_eq!((built, resolved), (None, ResolvedBackend::Native));
        assert_eq!(tried, [ResolvedBackend::Xess, ResolvedBackend::Fsr]);
    }

    #[test]
    fn a_rebuild_skips_the_candidates_that_failed() {
        let mut request = request(B::Dlss);
        let mut tried = Vec::new();
        let mut build = |c, _| {
            tried.push(c);
            Ok((c == ResolvedBackend::Xess).then_some(c))
        };
        let available = avail(true, true, true);
        let first = request.build_first_available(available, (8, 8), &mut build);
        let rebuilt = request.build_first_available(available, (8, 8), &mut build);
        assert_eq!(first.unwrap().1, ResolvedBackend::Xess);
        assert_eq!(rebuilt.unwrap().1, ResolvedBackend::Xess);
        assert_eq!(
            tried,
            [
                ResolvedBackend::Dlss,
                ResolvedBackend::Xess,
                ResolvedBackend::Xess
            ]
        );
    }

    #[test]
    fn a_rebuild_after_every_candidate_failed_goes_straight_to_native() {
        let mut request = request(B::Auto);
        let mut tried = 0;
        let available = avail(true, true, true);
        for _ in 0..2 {
            let (built, resolved) = request
                .build_first_available(available, (8, 8), |_, _| {
                    tried += 1;
                    Ok(None::<()>)
                })
                .unwrap();
            assert_eq!((built, resolved), (None, ResolvedBackend::Native));
        }
        assert_eq!(tried, 3);
    }

    #[test]
    fn the_build_runs_the_requested_preset() {
        let mut request = request(B::Dlss);
        request.dlss_preset = DlssPreset::L;
        let (built, _) = request
            .build_first_available(avail(true, false, false), (8, 8), |_, preset| {
                Ok(Some(preset))
            })
            .unwrap();
        assert_eq!(built, Some(DlssPreset::L));
    }
}
