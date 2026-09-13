// The `setting:<key>:<verb>` action grammar the settings-menu HitRegions carry.
// One parse shared by the UI input pass (which verb a click means), the focus
// pass (which regions are value rows), and the settings-row capture that maps
// each key to its value label.

// The key and verb of a `setting:<key>:<verb>` action, or `None` for any other
// action or an empty key.
pub(crate) fn parse(action: &str) -> Option<(&str, &str)> {
    let rest = action.strip_prefix("setting:")?;
    let (key, verb) = rest.rsplit_once(':')?;
    (!key.is_empty()).then_some((key, verb))
}

// The setting key of an action carrying `verb`, or `None`.
pub(crate) fn key_with_verb<'a>(action: &'a str, verb: &str) -> Option<&'a str> {
    parse(action)
        .filter(|&(_, v)| v == verb)
        .map(|(key, _)| key)
}

// The setting key of a cycle row's value-carrying region, or `None`. A stepper
// row emits a `:next` region (with a matching `:prev` sharing the same value
// label, so capturing `:next` alone maps the key once); a dropdown row emits a
// single `:open` region. Both carry the value label, so matching either maps
// each cycle key to its value label exactly once.
pub(crate) fn cycle_key(action: &str) -> Option<&str> {
    parse(action)
        .filter(|&(_, verb)| verb == "next" || verb == "open")
        .map(|(key, _)| key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_splits_key_and_verb() {
        assert_eq!(
            parse("setting:render_scale:drag"),
            Some(("render_scale", "drag"))
        );
        assert_eq!(parse("setting:jump:rebind"), Some(("jump", "rebind")));
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

    #[test]
    fn key_with_verb_matches_only_its_verb() {
        assert_eq!(
            key_with_verb("setting:render_scale:drag", "drag"),
            Some("render_scale")
        );
        assert_eq!(key_with_verb("setting:render_scale:next", "drag"), None);
        assert_eq!(key_with_verb("setting::rebind", "rebind"), None);
    }

    #[test]
    fn cycle_key_accepts_a_stepper_next_and_a_dropdown_open() {
        assert_eq!(cycle_key("setting:shadows:next"), Some("shadows"));
        assert_eq!(cycle_key("setting:resolution:open"), Some("resolution"));
        assert_eq!(cycle_key("setting:shadows:prev"), None);
        assert_eq!(cycle_key("screen:hide"), None);
        assert_eq!(cycle_key("setting::next"), None);
        assert_eq!(cycle_key("setting::open"), None);
    }
}
