// Action text builders for generated HitRegions and KeyBindings. Generated
// entries are authored-form JSON emitted before names are interned, so targets
// stay names here and resolve when the args deserialize into a UiAction.

use concinnity_core::components::{SettingVerb, StoryCommand};
use concinnity_core::settings::SettingKey;

/// Stop the application.
pub(crate) fn quit() -> String {
    "quit".to_string()
}

/// Show the named screen, replacing the top of the stack.
pub(crate) fn screen_show(name: &str) -> String {
    format!("screen:show:{name}")
}

/// Toggle the named screen.
pub(crate) fn screen_toggle(name: &str) -> String {
    format!("screen:toggle:{name}")
}

/// Close the top screen.
pub(crate) fn screen_hide() -> String {
    "screen:hide".to_string()
}

/// Expand or collapse a settings-screen group.
pub(crate) fn group_toggle(gid: usize) -> String {
    format!("group:toggle:{gid}")
}

/// Drive the story system.
pub(crate) fn story(cmd: StoryCommand) -> String {
    match cmd.index() {
        Some(i) => format!("story:{}:{i}", cmd.verb()),
        None => format!("story:{}", cmd.verb()),
    }
}

/// Operate a settings row.
pub(crate) fn setting(key: SettingKey, verb: SettingVerb) -> String {
    format!("setting:{}:{}", key.as_str(), verb.as_str())
}

#[cfg(test)]
mod tests {
    use concinnity_core::components::{ScreenCommand, UiAction};
    use concinnity_core::ecs::asset_id::AssetId;
    use concinnity_core::input::keymap::Bindable;

    use super::*;

    fn parse(text: &str) -> UiAction {
        let resolve = |name: &str| (name == "pause").then_some(AssetId(9));
        UiAction::parse(text, resolve).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    #[test]
    fn target_builders_parse_to_their_named_asset() {
        assert_eq!(parse(&quit()), UiAction::Quit);
        assert_eq!(
            parse(&screen_show("pause")),
            UiAction::Screen(ScreenCommand::Show(AssetId(9)))
        );
        assert_eq!(
            parse(&screen_toggle("pause")),
            UiAction::Screen(ScreenCommand::Toggle(AssetId(9)))
        );
        assert_eq!(parse(&screen_hide()), UiAction::Screen(ScreenCommand::Hide));
        assert_eq!(parse(&group_toggle(3)), UiAction::GroupToggle(3));
    }

    #[test]
    fn story_builder_parses_every_verb() {
        for verb in StoryCommand::VERBS {
            let cmd = StoryCommand::from_verb(verb, Some(2)).unwrap();
            assert_eq!(parse(&story(cmd.clone())), UiAction::Story(cmd));
        }
    }

    #[test]
    fn setting_builder_parses_every_verb() {
        let key = SettingKey::KeyRebind(Bindable::Jump);
        for verb in SettingVerb::ALL {
            assert_eq!(parse(&setting(key, verb)), UiAction::Setting { key, verb });
        }
    }
}
