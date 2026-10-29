//! Which upscaler a world's request resolves to, and the order the backends
//! fall back through when one fails to initialize.

use concinnity_engine::components::UpscalerBackend;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::output::dlss::DlssPreset;
use concinnity_engine::render::output::upscaler::{
    UpscalerAvailability, VendorUpscaler, requested_upscaler, upscaler_order,
};
use concinnity_engine::settings::ResolvedGraphics;

/// What temporal upscaling was asked for: the upscaler, and the render preset
/// DLSS runs if it is the one built. Kept by a backend so a resize rebuilds
/// the same, skipping the candidates that already failed to initialize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UpscaleRequest {
    pub(crate) backend: UpscalerBackend,
    pub(crate) dlss_preset: DlssPreset,
    refused: UpscalerAvailability,
}

impl UpscaleRequest {
    /// The request a backend's graphics settings make.
    pub(crate) fn from_graphics(graphics: &ResolvedGraphics) -> Self {
        Self {
            backend: graphics.anti_aliasing.upscaler,
            dlss_preset: graphics.dlss_preset,
            refused: UpscalerAvailability::NONE,
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

impl From<VendorUpscaler> for ResolvedBackend {
    fn from(upscaler: VendorUpscaler) -> Self {
        match upscaler {
            VendorUpscaler::Dlss => Self::Dlss,
            VendorUpscaler::Xess => Self::Xess,
            VendorUpscaler::Fsr => Self::Fsr,
        }
    }
}

// The candidates in the order they are tried (the shared vendor order), then
// Native, which is always last and always available.
fn backend_order(
    requested: UpscalerBackend,
    available: UpscalerAvailability,
) -> Vec<ResolvedBackend> {
    let mut order: Vec<ResolvedBackend> = upscaler_order(requested, available)
        .into_iter()
        .map(ResolvedBackend::from)
        .collect();
    order.push(ResolvedBackend::Native);
    order
}

/// The first candidate `requested` resolves to before any initialization is
/// tried.
#[cfg(any(backend_vk, test))]
pub(crate) fn preferred(
    requested: UpscalerBackend,
    available: UpscalerAvailability,
) -> ResolvedBackend {
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
        available: UpscalerAvailability,
        output: (u32, u32),
        mut build: impl FnMut(ResolvedBackend, DlssPreset) -> RenderResult<Option<T>>,
    ) -> RenderResult<(Option<T>, ResolvedBackend)> {
        let (width, height) = output;
        if let Some(missing) = self.take_unbundled_request(available) {
            tracing::warn!(
                "temporal upscaling: {:?} was requested but this build does not bundle it, \
                 falling back",
                ResolvedBackend::from(missing)
            );
        }
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
            if let Some(vendor) = vendor_of(candidate) {
                self.refused.mark(vendor);
            }
            tracing::warn!("temporal upscaling: {candidate:?} unavailable, trying next backend");
        }
        tracing::info!("temporal upscaling: no backend available, rendering at native resolution");
        Ok((None, ResolvedBackend::Native))
    }
}

impl UpscaleRequest {
    // The requested vendor upscaler when `available` lacks it, reported once:
    // it is marked refused so a rebuild does not report it again.
    fn take_unbundled_request(
        &mut self,
        available: UpscalerAvailability,
    ) -> Option<VendorUpscaler> {
        let requested = requested_upscaler(self.backend)?;
        if available.contains(requested) || self.refused.contains(requested) {
            return None;
        }
        self.refused.mark(requested);
        Some(requested)
    }
}

fn vendor_of(backend: ResolvedBackend) -> Option<VendorUpscaler> {
    match backend {
        ResolvedBackend::Dlss => Some(VendorUpscaler::Dlss),
        ResolvedBackend::Xess => Some(VendorUpscaler::Xess),
        ResolvedBackend::Fsr => Some(VendorUpscaler::Fsr),
        ResolvedBackend::Native => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UpscalerBackend as B;

    fn avail(dlss: bool, xess: bool, fsr: bool) -> UpscalerAvailability {
        UpscalerAvailability { dlss, xess, fsr }
    }

    fn request(backend: UpscalerBackend) -> UpscaleRequest {
        UpscaleRequest {
            backend,
            dlss_preset: DlssPreset::Default,
            refused: UpscalerAvailability::NONE,
        }
    }

    #[test]
    fn the_order_ends_native() {
        assert_eq!(
            backend_order(B::Fsr3, avail(false, false, true)),
            [ResolvedBackend::Fsr, ResolvedBackend::Native]
        );
        assert_eq!(
            preferred(B::Auto, avail(false, false, false)),
            ResolvedBackend::Native
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
    fn an_unbundled_request_is_reported_once() {
        let mut fsr = request(B::Fsr3);
        let available = avail(true, false, false);
        assert_eq!(
            fsr.take_unbundled_request(available),
            Some(VendorUpscaler::Fsr)
        );
        assert_eq!(fsr.take_unbundled_request(available), None);
        assert_eq!(
            request(B::Fsr3).take_unbundled_request(avail(false, false, true)),
            None
        );
        assert_eq!(
            request(B::Auto).take_unbundled_request(avail(false, false, false)),
            None
        );
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
