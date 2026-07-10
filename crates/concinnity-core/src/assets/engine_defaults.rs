// src/assets/engine_defaults.rs

use crate::assets::EngineDefaults;
use crate::ecs::{AssetOrigin, Component};

impl Component for EngineDefaults {
    const NAME: &'static str = "EngineDefaults";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_args_enable_every_default() {
        let d: EngineDefaults = serde_json::from_str("{}").unwrap();
        assert!(d.hud && d.debug_hud && d.sky && d.story_pause_menu);
    }

    #[test]
    fn individual_flags_opt_out() {
        let d: EngineDefaults = serde_json::from_str(r#"{"sky":false}"#).unwrap();
        assert!(!d.sky);
        assert!(d.hud && d.debug_hud && d.story_pause_menu);

        let d: EngineDefaults = serde_json::from_str(r#"{"story_pause_menu":false}"#).unwrap();
        assert!(!d.story_pause_menu);
        assert!(d.hud && d.debug_hud && d.sky);
    }
}
