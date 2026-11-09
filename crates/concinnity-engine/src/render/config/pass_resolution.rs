// Internal resolution of a screen-space pass.

use crate::components::Vocabulary;

/// Internal resolution a screen-space pass renders at before its result is
/// upsampled to native. `half` (the default) shades a quarter of the pixels;
/// `full` keeps the pass at native resolution; `quarter` is the cheapest.
// Variants are ordered finest first, so the coarser of two is their `max`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[derive(Default, Vocabulary)]
pub enum PassResolution {
    /// Native resolution.
    #[vocab("full")]
    Full,
    /// Half resolution per axis.
    #[default]
    #[vocab("half")]
    Half,
    /// Quarter resolution per axis.
    #[vocab("quarter")]
    Quarter,
}

impl PassResolution {
    /// Per-axis render-resolution divisor the pass target is scaled by.
    pub fn scale_divisor(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
            Self::Quarter => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_and_quarter_resolutions_divide_the_target() {
        assert_eq!(PassResolution::Full.scale_divisor(), 1);
        assert_eq!(PassResolution::Half.scale_divisor(), 2);
        assert_eq!(PassResolution::Quarter.scale_divisor(), 4);
    }

    #[test]
    fn the_coarser_resolution_is_the_max() {
        assert!(PassResolution::ALL.windows(2).all(|w| w[0] < w[1]));
        assert!(
            PassResolution::ALL
                .windows(2)
                .all(|w| w[0].scale_divisor() < w[1].scale_divisor())
        );
        assert_eq!(
            PassResolution::Full.max(PassResolution::Quarter),
            PassResolution::Quarter
        );
    }

    #[test]
    fn names_parse_in_snake_case() {
        assert_eq!(
            serde_json::from_str::<PassResolution>(r#""quarter""#).unwrap(),
            PassResolution::Quarter
        );
        assert_eq!(
            serde_json::from_str::<PassResolution>(r#""full""#).unwrap(),
            PassResolution::Full
        );
    }
}
