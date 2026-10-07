//! What the engine fixes about NVIDIA DLSS outside any graphics backend: the
//! render preset a user picks, and the NGX API version the integration is
//! written against, which the build checks the vendored SDK for.

/// `NVSDK_NGX_Version_API` of the NGX headers the engine's DLSS declarations
/// are transcribed from (1.5.0).
pub const NGX_API_VERSION: u32 = 0x15;

/// The DLSS render preset every quality mode runs, named by NVIDIA's letters.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DlssPreset {
    /// DLSS's own pick per quality mode, which follows its driver updates.
    #[default]
    Default,
    /// The transformer model NVIDIA recommends for DLAA, Quality and Balanced.
    K,
    /// The second-generation transformer NVIDIA recommends for Ultra
    /// Performance.
    L,
    /// The second-generation transformer NVIDIA recommends for Performance.
    M,
}

impl DlssPreset {
    /// Every preset, in settings-menu order.
    pub const ALL: [DlssPreset; 4] = [
        DlssPreset::Default,
        DlssPreset::K,
        DlssPreset::L,
        DlssPreset::M,
    ];

    /// The preset's name, as NVIDIA's tooling shows it.
    pub const fn label(self) -> &'static str {
        match self {
            DlssPreset::Default => "Default",
            DlssPreset::K => "K",
            DlssPreset::L => "L",
            DlssPreset::M => "M",
        }
    }

    /// Every preset's label, in [`DlssPreset::ALL`] order.
    pub const LABELS: [&'static str; 4] = {
        let mut labels = [""; 4];
        let mut i = 0;
        while i < labels.len() {
            labels[i] = DlssPreset::ALL[i].label();
            i += 1;
        }
        labels
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse_in_snake_case() {
        for (preset, name) in DlssPreset::ALL.into_iter().zip(["default", "k", "l", "m"]) {
            let json = serde_json::to_value(preset).unwrap();
            assert_eq!(json, name);
            assert_eq!(serde_json::from_value::<DlssPreset>(json).unwrap(), preset);
        }
    }

    #[test]
    fn the_labels_follow_the_menu_order() {
        assert_eq!(DlssPreset::LABELS, ["Default", "K", "L", "M"]);
        assert_eq!(DlssPreset::ALL[0], DlssPreset::default());
    }
}
