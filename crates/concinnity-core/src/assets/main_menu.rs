// src/assets/main_menu.rs

use crate::assets::MainMenu;
use crate::ecs::{AssetOrigin, Component};

impl Component for MainMenu {
    const NAME: &'static str = "MainMenu";
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
    use crate::assets::SettingsProfile;

    #[test]
    fn bare_args_default_to_return_settings_quit() {
        let m: MainMenu = serde_json::from_str("{}").unwrap();
        let labels: Vec<&str> = m.items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, vec!["Return", "Settings", "Quit"]);
        // Closed on load by default: the scene shows first, Escape opens.
        assert!(!m.initial);
        assert_eq!(m.toggle_key, "Escape");
        assert!(m.cursor);
    }

    #[test]
    fn explicit_items_replace_the_default() {
        let json = r#"{"items":[{"label":"Play","action":"scene:level_1"}]}"#;
        let m: MainMenu = serde_json::from_str(json).unwrap();
        assert_eq!(m.items.len(), 1);
        assert_eq!(m.items[0].label, "Play");
        assert_eq!(m.items[0].action, "scene:level_1");
    }

    #[test]
    fn settings_profile_defaults_to_full_and_parses_lowercase() {
        let m: MainMenu = serde_json::from_str("{}").unwrap();
        assert_eq!(m.settings_profile, SettingsProfile::Full);
        let m: MainMenu = serde_json::from_str(r#"{"settings_profile":"minimal"}"#).unwrap();
        assert_eq!(m.settings_profile, SettingsProfile::Minimal);
    }
}
