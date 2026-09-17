//! The action vocabulary a HitRegion click or KeyBinding press fires.

use core::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::components::{ScreenCommand, StoryCommand};
use crate::ecs::asset_id::AssetId;
use crate::ecs::resolver::resolve_name;
use crate::settings::SettingKey;

/// What a [HitRegion](#hitregion) click or a [KeyBinding](#keybinding) press
/// does.
///
/// Authored as text:
/// - `"quit"`: stop the application
/// - `"scene:<name>"`: jump to the named [Scene](#scene)
/// - `"screen:show:<name>"`: show the named [Screen](#screen), replacing the top of the stack
/// - `"screen:push:<name>"`: open the named [Screen](#screen) on top of what is showing
/// - `"screen:toggle:<name>"`: close the named [Screen](#screen) if it is on top, open it otherwise
/// - `"screen:hide"`: close the top [Screen](#screen)
/// - `"story:<verb>"`: drive the story (`start`, `continue`, `advance`,
///   `choose:<i>`, `slot:<i>`, `auto`, `skip`, `log`, `save`, `load`, `pause`,
///   `settings`, `settings_back`)
///
/// A target may also be an already-resolved integer id. Generated menus also
/// emit `"group:toggle:<i>"` and `"setting:<key>:<verb>"` actions.
///
/// ```rust
/// # use concinnity_core::components::{ScreenCommand, UiAction};
/// # use concinnity_core::ecs::asset_id::AssetId;
/// let action = UiAction::parse("screen:toggle:7", |_| None).unwrap();
/// assert_eq!(action, UiAction::Screen(ScreenCommand::Toggle(AssetId(7))));
/// assert_eq!(action.to_string(), "screen:toggle:7");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiAction {
    /// Stop the application.
    Quit,
    /// Change to a scene, dismissing every open screen.
    Scene(AssetId),
    /// Apply a screen-stack transition. Never `Clear`, which only the engine sends.
    Screen(ScreenCommand),
    /// Expand or collapse a settings-screen group by index.
    GroupToggle(usize),
    /// Drive the story system.
    Story(StoryCommand),
    /// Operate a settings row.
    Setting {
        /// The setting the row edits.
        key: SettingKey,
        /// What the region does to it.
        verb: SettingVerb,
    },
}

/// What a settings-row region does to its setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingVerb {
    /// Step the value one option forward.
    Next,
    /// Step the value one option back.
    Prev,
    /// Drag a slider along its track.
    Drag,
    /// Capture the next key or button press as the binding.
    Rebind,
    /// Open a dropdown list of the options.
    Open,
}

impl SettingVerb {
    /// Every verb, in declaration order.
    pub const ALL: [SettingVerb; 5] = [
        SettingVerb::Next,
        SettingVerb::Prev,
        SettingVerb::Drag,
        SettingVerb::Rebind,
        SettingVerb::Open,
    ];

    /// The verb as it appears in a `setting:<key>:<verb>` action.
    pub const fn as_str(self) -> &'static str {
        match self {
            SettingVerb::Next => "next",
            SettingVerb::Prev => "prev",
            SettingVerb::Drag => "drag",
            SettingVerb::Rebind => "rebind",
            SettingVerb::Open => "open",
        }
    }

    /// The verb named by `text`, or `None`.
    pub fn parse(text: &str) -> Option<SettingVerb> {
        Self::ALL.into_iter().find(|v| v.as_str() == text)
    }
}

/// Why action text failed to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiActionError {
    /// The text names no action.
    Unknown,
    /// A scene or screen action has no target.
    MissingTarget,
    /// A target name could not be resolved to an asset id.
    UnresolvedTarget,
    /// `screen:clear`, which only the engine sends.
    EngineOnly,
    /// A story or group index is missing or not a non-negative integer.
    BadIndex,
    /// A `story:` verb that names no story command.
    UnknownStoryVerb,
    /// A `setting:` key that names no setting.
    UnknownSettingKey,
    /// A `setting:` verb that is missing or unknown.
    UnknownSettingVerb,
}

impl fmt::Display for UiActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            UiActionError::Unknown => "unknown action",
            UiActionError::MissingTarget => "missing target",
            UiActionError::UnresolvedTarget => "unresolved target name",
            UiActionError::EngineOnly => "screen:clear is sent only by the engine",
            UiActionError::BadIndex => "missing or invalid index",
            UiActionError::UnknownStoryVerb => "unknown story verb",
            UiActionError::UnknownSettingKey => "unknown setting key",
            UiActionError::UnknownSettingVerb => "missing or unknown setting verb",
        })
    }
}

impl UiAction {
    /// Parse action text. A scene or screen target is an integer id or a name
    /// passed to `resolve`.
    pub fn parse(
        text: &str,
        resolve: impl Fn(&str) -> Option<AssetId>,
    ) -> Result<UiAction, UiActionError> {
        let target = |name: &str| -> Result<AssetId, UiActionError> {
            if name.is_empty() {
                Err(UiActionError::MissingTarget)
            } else if let Ok(id) = name.parse::<u32>() {
                Ok(AssetId(id))
            } else {
                resolve(name).ok_or(UiActionError::UnresolvedTarget)
            }
        };
        let index = |i: &str| i.parse::<usize>().map_err(|_| UiActionError::BadIndex);

        if text == "quit" {
            return Ok(UiAction::Quit);
        }
        let (kind, rest) = text.split_once(':').ok_or(UiActionError::Unknown)?;
        match kind {
            "scene" => target(rest).map(UiAction::Scene),
            "screen" => {
                let (verb, name) = rest.split_once(':').unwrap_or((rest, ""));
                let cmd = match verb {
                    "hide" if name.is_empty() && !rest.ends_with(':') => ScreenCommand::Hide,
                    "show" => ScreenCommand::Show(target(name)?),
                    "push" => ScreenCommand::Push(target(name)?),
                    "toggle" => ScreenCommand::Toggle(target(name)?),
                    "clear" => return Err(UiActionError::EngineOnly),
                    _ => return Err(UiActionError::Unknown),
                };
                Ok(UiAction::Screen(cmd))
            }
            "group" => match rest.split_once(':') {
                Some(("toggle", i)) => index(i).map(UiAction::GroupToggle),
                _ => Err(UiActionError::Unknown),
            },
            "story" => {
                let (verb, i) = match rest.split_once(':') {
                    Some((verb, i)) => (verb, Some(index(i)?)),
                    None => (rest, None),
                };
                if !StoryCommand::VERBS.contains(&verb) {
                    return Err(UiActionError::UnknownStoryVerb);
                }
                StoryCommand::from_verb(verb, i)
                    .map(UiAction::Story)
                    .ok_or(UiActionError::BadIndex)
            }
            "setting" => {
                let (key, verb) = rest
                    .rsplit_once(':')
                    .ok_or(UiActionError::UnknownSettingVerb)?;
                let key = SettingKey::parse(key).ok_or(UiActionError::UnknownSettingKey)?;
                let verb = SettingVerb::parse(verb).ok_or(UiActionError::UnknownSettingVerb)?;
                Ok(UiAction::Setting { key, verb })
            }
            _ => Err(UiActionError::Unknown),
        }
    }
}

impl fmt::Display for UiAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UiAction::Quit => f.write_str("quit"),
            UiAction::Scene(id) => write!(f, "scene:{}", id.0),
            UiAction::Screen(ScreenCommand::Show(id)) => write!(f, "screen:show:{}", id.0),
            UiAction::Screen(ScreenCommand::Push(id)) => write!(f, "screen:push:{}", id.0),
            UiAction::Screen(ScreenCommand::Toggle(id)) => write!(f, "screen:toggle:{}", id.0),
            UiAction::Screen(ScreenCommand::Hide) => f.write_str("screen:hide"),
            UiAction::Screen(ScreenCommand::Clear) => f.write_str("screen:clear"),
            UiAction::GroupToggle(i) => write!(f, "group:toggle:{i}"),
            UiAction::Story(cmd) => match cmd.index() {
                Some(i) => write!(f, "story:{}:{i}", cmd.verb()),
                None => write!(f, "story:{}", cmd.verb()),
            },
            UiAction::Setting { key, verb } => {
                write!(f, "setting:{}:{}", key.as_str(), verb.as_str())
            }
        }
    }
}

// The blob form: a variant tag plus a payload, with the story verb and the
// setting key and verb carried as positions in their tables.
#[derive(Serialize, Deserialize)]
enum Encoded {
    Quit,
    Scene(u32),
    ScreenShow(u32),
    ScreenPush(u32),
    ScreenToggle(u32),
    ScreenHide,
    GroupToggle(u32),
    Story { verb: u8, index: u32 },
    Setting { key: u8, verb: u8 },
}

impl Encoded {
    fn encode(action: &UiAction) -> Option<Encoded> {
        Some(match action {
            UiAction::Quit => Encoded::Quit,
            UiAction::Scene(id) => Encoded::Scene(id.0),
            UiAction::Screen(ScreenCommand::Show(id)) => Encoded::ScreenShow(id.0),
            UiAction::Screen(ScreenCommand::Push(id)) => Encoded::ScreenPush(id.0),
            UiAction::Screen(ScreenCommand::Toggle(id)) => Encoded::ScreenToggle(id.0),
            UiAction::Screen(ScreenCommand::Hide) => Encoded::ScreenHide,
            UiAction::Screen(ScreenCommand::Clear) => return None,
            UiAction::GroupToggle(i) => Encoded::GroupToggle(u32::try_from(*i).ok()?),
            UiAction::Story(cmd) => Encoded::Story {
                verb: StoryCommand::VERBS.iter().position(|v| *v == cmd.verb())? as u8,
                index: u32::try_from(cmd.index().unwrap_or(0)).ok()?,
            },
            UiAction::Setting { key, verb } => Encoded::Setting {
                key: u8::try_from(SettingKey::ALL.iter().position(|k| k == key)?).ok()?,
                verb: SettingVerb::ALL.iter().position(|v| v == verb)? as u8,
            },
        })
    }

    fn decode(self) -> Option<UiAction> {
        Some(match self {
            Encoded::Quit => UiAction::Quit,
            Encoded::Scene(id) => UiAction::Scene(AssetId(id)),
            Encoded::ScreenShow(id) => UiAction::Screen(ScreenCommand::Show(AssetId(id))),
            Encoded::ScreenPush(id) => UiAction::Screen(ScreenCommand::Push(AssetId(id))),
            Encoded::ScreenToggle(id) => UiAction::Screen(ScreenCommand::Toggle(AssetId(id))),
            Encoded::ScreenHide => UiAction::Screen(ScreenCommand::Hide),
            Encoded::GroupToggle(i) => UiAction::GroupToggle(i as usize),
            Encoded::Story { verb, index } => UiAction::Story(StoryCommand::from_verb(
                StoryCommand::VERBS.get(verb as usize)?,
                Some(index as usize),
            )?),
            Encoded::Setting { key, verb } => UiAction::Setting {
                key: *SettingKey::ALL.get(key as usize)?,
                verb: *SettingVerb::ALL.get(verb as usize)?,
            },
        })
    }
}

impl Serialize for UiAction {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        if matches!(self, UiAction::Screen(ScreenCommand::Clear)) {
            return Err(S::Error::custom(UiActionError::EngineOnly));
        }
        if s.is_human_readable() {
            s.collect_str(self)
        } else {
            Encoded::encode(self)
                .ok_or_else(|| S::Error::custom("action payload out of range"))?
                .serialize(s)
        }
    }
}

struct TextVisitor;

impl Visitor<'_> for TextVisitor {
    type Value = UiAction;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an action string")
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<UiAction, E> {
        UiAction::parse(v, |name| resolve_name(name).map(AssetId))
            .map_err(|e| E::custom(format_args!("invalid action {v:?}: {e}")))
    }
}

impl<'de> Deserialize<'de> for UiAction {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            d.deserialize_str(TextVisitor)
        } else {
            Encoded::deserialize(d)?
                .decode()
                .ok_or_else(|| de::Error::custom("invalid encoded action"))
        }
    }
}

/// `serde` `with` helpers for an optional action field.
///
/// In text form an empty string or null reads as `None`, and `None` writes as
/// an empty string. Apply with `#[serde(default, with =
/// "concinnity_core::components::ui_action::optional")]`.
pub mod optional {
    use super::*;

    /// Serialize an optional action.
    pub fn serialize<S: Serializer>(action: &Option<UiAction>, s: S) -> Result<S::Ok, S::Error> {
        match action {
            Some(action) if s.is_human_readable() => action.serialize(s),
            None if s.is_human_readable() => s.serialize_str(""),
            action => action.serialize(s),
        }
    }

    /// Deserialize an optional action.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<UiAction>, D::Error> {
        if !d.is_human_readable() {
            return Option::<UiAction>::deserialize(d);
        }

        struct OptVisitor;

        impl Visitor<'_> for OptVisitor {
            type Value = Option<UiAction>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an action string or null")
            }

            fn visit_unit<E: de::Error>(self) -> Result<Option<UiAction>, E> {
                Ok(None)
            }
            fn visit_none<E: de::Error>(self) -> Result<Option<UiAction>, E> {
                Ok(None)
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<UiAction>, E> {
                if v.is_empty() {
                    Ok(None)
                } else {
                    TextVisitor.visit_str(v).map(Some)
                }
            }
        }

        d.deserialize_any(OptVisitor)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    fn stub(name: &str) -> Option<AssetId> {
        (name != "missing").then_some(AssetId(name.len() as u32))
    }

    fn every_action() -> Vec<UiAction> {
        let mut all = vec![
            UiAction::Quit,
            UiAction::Scene(AssetId(3)),
            UiAction::Screen(ScreenCommand::Show(AssetId(4))),
            UiAction::Screen(ScreenCommand::Push(AssetId(5))),
            UiAction::Screen(ScreenCommand::Toggle(AssetId(6))),
            UiAction::Screen(ScreenCommand::Hide),
            UiAction::GroupToggle(2),
        ];
        for verb in StoryCommand::VERBS {
            all.push(UiAction::Story(
                StoryCommand::from_verb(verb, Some(1)).unwrap(),
            ));
        }
        for key in SettingKey::ALL {
            for verb in SettingVerb::ALL {
                all.push(UiAction::Setting { key, verb });
            }
        }
        all
    }

    #[test]
    fn display_round_trips_through_parse_for_every_variant() {
        for action in every_action() {
            let text = action.to_string();
            assert_eq!(UiAction::parse(&text, stub), Ok(action), "{text}");
        }
    }

    #[test]
    fn every_action_round_trips_through_postcard() {
        let all = every_action();
        let bytes = postcard::to_allocvec(&all).unwrap();
        let back: Vec<UiAction> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, all);
    }

    #[test]
    fn a_target_name_resolves_through_the_installed_resolver() {
        crate::test_support::install_resolvers();
        let action: UiAction = serde_json::from_str("\"screen:push:pause\"").unwrap();
        assert_eq!(action, UiAction::Screen(ScreenCommand::Push(AssetId(5))));
        assert_eq!(serde_json::to_string(&action).unwrap(), "\"screen:push:5\"");
        assert_eq!(
            UiAction::parse("scene:missing", stub),
            Err(UiActionError::UnresolvedTarget)
        );
    }

    #[test]
    fn malformed_actions_are_rejected() {
        let cases = [
            ("setting::next", UiActionError::UnknownSettingKey),
            ("setting:vsync", UiActionError::UnknownSettingVerb),
            ("setting:vsync:spin", UiActionError::UnknownSettingVerb),
            ("story:choose", UiActionError::BadIndex),
            ("story:choose:x", UiActionError::BadIndex),
            ("story:dance", UiActionError::UnknownStoryVerb),
            ("screen:show:", UiActionError::MissingTarget),
            ("screen:hide:", UiActionError::Unknown),
            ("screen:clear", UiActionError::EngineOnly),
            ("group:toggle:-1", UiActionError::BadIndex),
            ("teleport", UiActionError::Unknown),
            ("", UiActionError::Unknown),
        ];
        for (text, err) in cases {
            assert_eq!(UiAction::parse(text, stub), Err(err), "{text}");
        }
    }

    #[test]
    fn clear_displays_but_never_serializes() {
        let clear = UiAction::Screen(ScreenCommand::Clear);
        assert_eq!(clear.to_string(), "screen:clear");
        assert!(serde_json::to_string(&clear).is_err());
        assert!(postcard::to_allocvec(&clear).is_err());
    }

    #[test]
    fn an_encoded_tag_past_the_last_variant_is_rejected() {
        assert!(postcard::from_bytes::<UiAction>(&[9]).is_err());
        assert!(postcard::from_bytes::<UiAction>(&[7, 13, 0]).is_err());
    }

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Holder {
        #[serde(default, with = "optional")]
        action: Option<UiAction>,
    }

    #[test]
    fn optional_text_reads_empty_null_and_missing_as_none() {
        for json in [r#"{"action":""}"#, r#"{"action":null}"#, "{}"] {
            let h: Holder = serde_json::from_str(json).unwrap();
            assert_eq!(h.action, None, "{json}");
        }
        let none = Holder { action: None };
        assert_eq!(serde_json::to_string(&none).unwrap(), r#"{"action":""}"#);
        let quit: Holder = serde_json::from_str(r#"{"action":"quit"}"#).unwrap();
        assert_eq!(quit.action, Some(UiAction::Quit));
        assert!(serde_json::from_str::<Holder>(r#"{"action":"teleport"}"#).is_err());
    }

    #[test]
    fn optional_round_trips_through_postcard() {
        for action in [None, Some(UiAction::GroupToggle(4))] {
            let h = Holder { action };
            let bytes = postcard::to_allocvec(&h).unwrap();
            assert_eq!(postcard::from_bytes::<Holder>(&bytes).unwrap(), h);
        }
    }
}
