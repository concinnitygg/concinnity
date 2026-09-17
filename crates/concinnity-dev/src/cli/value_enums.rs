// The argv mirrors of engine types. The value-enum derives live here so no
// layer below the command line carries a clap dependency.
use concinnity_core::render::rt_geom::RtDynamicMode;
use concinnity_engine::gfx::quality_preset::QualityPreset;

use crate::export::BundleFormat;

// The argv face of the engine's `QualityPreset`.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum QualityPresetArg {
    Auto,
    Low,
    Medium,
    High,
    Ultra,
    Custom,
}

impl From<QualityPresetArg> for QualityPreset {
    fn from(p: QualityPresetArg) -> Self {
        match p {
            QualityPresetArg::Auto => QualityPreset::Auto,
            QualityPresetArg::Low => QualityPreset::Low,
            QualityPresetArg::Medium => QualityPreset::Medium,
            QualityPresetArg::High => QualityPreset::High,
            QualityPresetArg::Ultra => QualityPreset::Ultra,
            QualityPresetArg::Custom => QualityPreset::Custom,
        }
    }
}

// The argv face of the export `BundleFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum BundleFormatArg {
    Zip,
    Dir,
}

impl From<BundleFormatArg> for BundleFormat {
    fn from(f: BundleFormatArg) -> Self {
        match f {
            BundleFormatArg::Zip => BundleFormat::Zip,
            BundleFormatArg::Dir => BundleFormat::Dir,
        }
    }
}

// The argv face of the render layer's `RtDynamicMode`.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum RtDynamicArg {
    Off,
    Auto,
    Rebuild,
    Tlas,
}

impl From<RtDynamicArg> for RtDynamicMode {
    fn from(m: RtDynamicArg) -> Self {
        match m {
            RtDynamicArg::Off => RtDynamicMode::Off,
            RtDynamicArg::Auto => RtDynamicMode::Auto,
            RtDynamicArg::Rebuild => RtDynamicMode::Rebuild,
            RtDynamicArg::Tlas => RtDynamicMode::Tlas,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The value enums are the argv face of the engine types, so a variant added
    // to one and not the other has to fail here rather than at a launch.
    #[test]
    fn the_render_value_enums_map_onto_the_engine_types() {
        use clap::ValueEnum;

        let presets: Vec<QualityPreset> = QualityPresetArg::value_variants()
            .iter()
            .map(|&a| a.into())
            .collect();
        assert_eq!(presets, QualityPreset::ALL.to_vec());

        let modes: Vec<RtDynamicMode> = RtDynamicArg::value_variants()
            .iter()
            .map(|&a| a.into())
            .collect();
        assert_eq!(
            modes,
            vec![
                RtDynamicMode::Off,
                RtDynamicMode::Auto,
                RtDynamicMode::Rebuild,
                RtDynamicMode::Tlas,
            ]
        );
    }
}
