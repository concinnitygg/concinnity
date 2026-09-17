// The `setting:<key>:<verb>` action grammar the settings-menu HitRegions carry.
// One parse shared by the UI input pass (which verb a click means, which regions
// are panel content, and which belong to a disabled row), the focus pass (which
// regions are value rows), the settings-row capture that maps each key to its
// value label, and the init-time capability gating and value-label sync.

use concinnity_core::settings::SettingKey;

// The setting and verb of a `setting:<key>:<verb>` action, or `None` for any
// other action or a key that names no setting.
pub(crate) fn parse(action: &str) -> Option<(SettingKey, &str)> {
    let rest = action.strip_prefix("setting:")?;
    let (key, verb) = rest.rsplit_once(':')?;
    Some((SettingKey::parse(key)?, verb))
}

// The setting of a `setting:<key>:<verb>` action with any verb, or `None`.
pub(crate) fn key(action: &str) -> Option<SettingKey> {
    parse(action).map(|(key, _)| key)
}

// The setting of an action carrying `verb`, or `None`.
pub(crate) fn key_with_verb(action: &str, verb: &str) -> Option<SettingKey> {
    parse(action)
        .filter(|&(_, v)| v == verb)
        .map(|(key, _)| key)
}

// The setting of a cycle row's value-carrying region, or `None`. A stepper
// row emits a `:next` region (with a matching `:prev` sharing the same value
// label, so capturing `:next` alone maps the key once); a dropdown row emits a
// single `:open` region. Both carry the value label, so matching either maps
// each cycle key to its value label exactly once.
pub(crate) fn cycle_key(action: &str) -> Option<SettingKey> {
    parse(action)
        .filter(|&(_, verb)| verb == "next" || verb == "open")
        .map(|(key, _)| key)
}

#[cfg(test)]
mod tests {
    use concinnity_core::input::keymap::Bindable;

    use super::*;

    #[test]
    fn parse_splits_key_and_verb() {
        assert_eq!(
            parse("setting:render_scale:drag"),
            Some((SettingKey::RenderScale, "drag"))
        );
        assert_eq!(
            parse("setting:key_jump:rebind"),
            Some((SettingKey::KeyRebind(Bindable::Jump), "rebind"))
        );
    }

    // A non-setting action belongs to no verb, and a verb with no key in front
    // of it is rejected rather than parsed as an empty setting.
    #[test]
    fn parse_rejects_other_actions_and_empty_keys() {
        for action in ["screen:show:pause", "menu:quit", "", "drag", "setting:"] {
            assert_eq!(parse(action), None, "{action}");
        }
        assert_eq!(parse("setting:no_verb"), None);
        assert_eq!(parse("setting::drag"), None);
    }

    // A key that names no setting is as malformed as a missing one.
    #[test]
    fn parse_rejects_an_unknown_key() {
        assert_eq!(parse("setting:shadows:next"), None);
        assert_eq!(parse("setting:key_nonsense:rebind"), None);
    }

    #[test]
    fn key_with_verb_matches_only_its_verb() {
        assert_eq!(
            key_with_verb("setting:render_scale:drag", "drag"),
            Some(SettingKey::RenderScale)
        );
        assert_eq!(key_with_verb("setting:render_scale:next", "drag"), None);
        assert_eq!(key_with_verb("setting::rebind", "rebind"), None);
    }

    #[test]
    fn cycle_key_accepts_a_stepper_next_and_a_dropdown_open() {
        assert_eq!(
            cycle_key("setting:shadow_map_size:next"),
            Some(SettingKey::ShadowMapSize)
        );
        assert_eq!(
            cycle_key("setting:resolution:open"),
            Some(SettingKey::Resolution)
        );
        assert_eq!(cycle_key("setting:shadow_map_size:prev"), None);
        assert_eq!(cycle_key("screen:hide"), None);
        assert_eq!(cycle_key("setting::next"), None);
        assert_eq!(cycle_key("setting::open"), None);
    }
}
