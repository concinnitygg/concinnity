//! Which vendor temporal upscaler a request resolves to, and the order a
//! backend falls back through when one fails to initialize. Shared by the
//! settings resolver, which predicts the choice before any backend exists, and
//! by the backends that make it.

use alloc::vec::Vec;

use crate::components::UpscalerBackend;

/// A vendor temporal upscaler a build can try.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VendorUpscaler {
    /// NVIDIA DLSS.
    Dlss,
    /// Intel XeSS.
    Xess,
    /// AMD FidelityFX FSR.
    Fsr,
}

/// Which vendor upscalers a build can try.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpscalerAvailability {
    /// DLSS.
    pub dlss: bool,
    /// XeSS.
    pub xess: bool,
    /// FSR.
    pub fsr: bool,
}

impl UpscalerAvailability {
    /// None of them.
    pub const NONE: Self = Self {
        dlss: false,
        xess: false,
        fsr: false,
    };

    /// Every vendor upscaler.
    pub const ALL: Self = Self {
        dlss: true,
        xess: true,
        fsr: true,
    };

    /// These, minus the ones in `refused`.
    #[must_use]
    pub fn without(self, refused: Self) -> Self {
        Self {
            dlss: self.dlss && !refused.dlss,
            xess: self.xess && !refused.xess,
            fsr: self.fsr && !refused.fsr,
        }
    }

    /// Whether `upscaler` is in the set.
    #[must_use]
    pub fn contains(self, upscaler: VendorUpscaler) -> bool {
        match upscaler {
            VendorUpscaler::Dlss => self.dlss,
            VendorUpscaler::Xess => self.xess,
            VendorUpscaler::Fsr => self.fsr,
        }
    }

    /// Add `upscaler` to the set.
    pub fn mark(&mut self, upscaler: VendorUpscaler) {
        match upscaler {
            VendorUpscaler::Dlss => self.dlss = true,
            VendorUpscaler::Xess => self.xess = true,
            VendorUpscaler::Fsr => self.fsr = true,
        }
    }
}

/// The vendor upscalers in the order a request tries them: the requested one
/// first when available, then DLSS, XeSS, FSR. Native rendering follows the
/// last.
pub fn upscaler_order(
    requested: UpscalerBackend,
    available: UpscalerAvailability,
) -> Vec<VendorUpscaler> {
    let mut order = Vec::with_capacity(3);
    if let Some(first) = requested_upscaler(requested).filter(|&u| available.contains(u)) {
        order.push(first);
    }
    for candidate in [
        VendorUpscaler::Dlss,
        VendorUpscaler::Xess,
        VendorUpscaler::Fsr,
    ] {
        if available.contains(candidate) && !order.contains(&candidate) {
            order.push(candidate);
        }
    }
    order
}

/// The vendor upscaler `requested` names, or `None` for `Auto`.
#[must_use]
pub fn requested_upscaler(requested: UpscalerBackend) -> Option<VendorUpscaler> {
    match requested {
        UpscalerBackend::Auto => None,
        UpscalerBackend::Dlss => Some(VendorUpscaler::Dlss),
        UpscalerBackend::Xess => Some(VendorUpscaler::Xess),
        UpscalerBackend::Fsr3 => Some(VendorUpscaler::Fsr),
    }
}

/// The upscaler `requested` resolves to before any initialization is tried,
/// or `None` for native rendering.
pub fn preferred_upscaler(
    requested: UpscalerBackend,
    available: UpscalerAvailability,
) -> Option<VendorUpscaler> {
    upscaler_order(requested, available).first().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use UpscalerBackend as B;

    fn avail(dlss: bool, xess: bool, fsr: bool) -> UpscalerAvailability {
        UpscalerAvailability { dlss, xess, fsr }
    }

    #[test]
    fn auto_prefers_dlss_then_xess_then_fsr_then_native() {
        use VendorUpscaler::*;
        assert_eq!(
            preferred_upscaler(B::Auto, avail(true, true, true)),
            Some(Dlss)
        );
        assert_eq!(
            preferred_upscaler(B::Auto, avail(false, true, true)),
            Some(Xess)
        );
        assert_eq!(
            preferred_upscaler(B::Auto, avail(false, false, true)),
            Some(Fsr)
        );
        assert_eq!(
            preferred_upscaler(B::Auto, UpscalerAvailability::NONE),
            None
        );
    }

    #[test]
    fn an_explicit_choice_leads_when_available_and_falls_through_when_not() {
        use VendorUpscaler::*;
        assert_eq!(
            preferred_upscaler(B::Xess, avail(true, true, true)),
            Some(Xess)
        );
        assert_eq!(
            preferred_upscaler(B::Fsr3, avail(true, true, true)),
            Some(Fsr)
        );
        assert_eq!(
            preferred_upscaler(B::Dlss, avail(false, true, true)),
            Some(Xess)
        );
        assert_eq!(
            preferred_upscaler(B::Fsr3, UpscalerAvailability::NONE),
            None
        );
    }

    #[test]
    fn every_order_lists_each_candidate_once() {
        use VendorUpscaler::*;
        assert_eq!(
            upscaler_order(B::Fsr3, UpscalerAvailability::ALL),
            [Fsr, Dlss, Xess]
        );
    }

    #[test]
    fn auto_names_no_upscaler_and_the_rest_name_their_own() {
        use VendorUpscaler::*;
        assert_eq!(requested_upscaler(B::Auto), None);
        assert_eq!(requested_upscaler(B::Dlss), Some(Dlss));
        assert_eq!(requested_upscaler(B::Xess), Some(Xess));
        assert_eq!(requested_upscaler(B::Fsr3), Some(Fsr));
    }

    #[test]
    fn contains_reads_each_member() {
        let set = avail(false, true, false);
        assert!(!set.contains(VendorUpscaler::Dlss));
        assert!(set.contains(VendorUpscaler::Xess));
        assert!(!set.contains(VendorUpscaler::Fsr));
    }

    #[test]
    fn a_refused_upscaler_leaves_the_set() {
        let mut refused = UpscalerAvailability::NONE;
        refused.mark(VendorUpscaler::Dlss);
        assert_eq!(
            UpscalerAvailability::ALL.without(refused),
            avail(false, true, true)
        );
    }
}
