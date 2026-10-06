//! Which upscaler a world's request resolves to, and the order the backends
//! fall back through when one fails to initialize.

use concinnity_core::components::UpscalerBackend;
use concinnity_core::render::error::RenderResult;

/// The upscaler a backend built, or native-resolution rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedBackend {
    Fsr,
    Dlss,
    Xess,
    Native,
}

/// Which upscaler SDKs this build can try.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Availability {
    pub(crate) dlss: bool,
    pub(crate) xess: bool,
    pub(crate) fsr: bool,
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

/// Build the first candidate `build` initializes, in fallback order, and name
/// it; `None` with `Native` when every vendor candidate failed. `build` is
/// never called for `Native`. An error from `build` ends the search.
pub(crate) fn build_first_available<T>(
    requested: UpscalerBackend,
    available: Availability,
    output: (u32, u32),
    mut build: impl FnMut(ResolvedBackend) -> RenderResult<Option<T>>,
) -> RenderResult<(Option<T>, ResolvedBackend)> {
    let (width, height) = output;
    for candidate in backend_order(requested, available) {
        if candidate == ResolvedBackend::Native {
            break;
        }
        if let Some(built) = build(candidate)? {
            tracing::info!(
                "temporal upscaling: using {candidate:?} backend (output {width}x{height})"
            );
            return Ok((Some(built), candidate));
        }
        tracing::warn!("temporal upscaling: {candidate:?} unavailable, trying next backend");
    }
    tracing::info!("temporal upscaling: no backend available, rendering at native resolution");
    Ok((None, ResolvedBackend::Native))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::UpscalerBackend as B;

    fn avail(dlss: bool, xess: bool, fsr: bool) -> Availability {
        Availability { dlss, xess, fsr }
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
    fn a_failed_candidate_falls_through_to_the_next() {
        let mut tried = Vec::new();
        let (built, resolved) =
            build_first_available(B::Dlss, avail(true, true, true), (8, 8), |c| {
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
        let (built, resolved) =
            build_first_available(B::Auto, avail(false, true, true), (8, 8), |c| {
                tried.push(c);
                Ok(None::<()>)
            })
            .unwrap();
        assert_eq!((built, resolved), (None, ResolvedBackend::Native));
        assert_eq!(tried, [ResolvedBackend::Xess, ResolvedBackend::Fsr]);
    }
}
