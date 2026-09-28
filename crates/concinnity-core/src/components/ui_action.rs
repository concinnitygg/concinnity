//! The action vocabulary a HitRegion click or KeyBinding press fires.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};

use crate::components::{AuthoredAction, Scene, Screen, ScreenCommand, StoryCommand};
use crate::ecs::Ref;
use crate::ecs::asset_fields::{AssetFields, FieldTable};
use crate::ecs::asset_id::AssetId;
use crate::settings::SettingKey;

/// What a [HitRegion](#hitregion) click or a [KeyBinding](#keybinding) press
/// does, with its targets resolved to asset ids.
///
/// Authored as an [`AuthoredAction`], whose [Scene](#scene) and
/// [Screen](#screen) names resolve to ids as it deserializes.
///
/// ```rust
/// # use concinnity_core::components::{ScreenCommand, UiAction};
/// # use concinnity_core::ecs::asset_id::AssetId;
/// let action: UiAction = serde_json::from_value(serde_json::json!({"toggle": 7})).unwrap();
/// assert_eq!(action, UiAction::Screen(ScreenCommand::Toggle(AssetId(7))));
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

    /// The verb as a settings row's action writes it.
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

// The authored form with its targets resolved: how a `UiAction` reads and
// writes as text.
type Resolved = AuthoredAction<Ref<Scene>, Ref<Screen>>;

impl From<Resolved> for UiAction {
    fn from(action: Resolved) -> Self {
        match action {
            AuthoredAction::Quit => UiAction::Quit,
            AuthoredAction::Scene(scene) => UiAction::Scene(scene.id()),
            AuthoredAction::Show(screen) => UiAction::Screen(ScreenCommand::Show(screen.id())),
            AuthoredAction::Push(screen) => UiAction::Screen(ScreenCommand::Push(screen.id())),
            AuthoredAction::Toggle(screen) => UiAction::Screen(ScreenCommand::Toggle(screen.id())),
            AuthoredAction::Hide => UiAction::Screen(ScreenCommand::Hide),
            AuthoredAction::Story(cmd) => UiAction::Story(cmd),
            AuthoredAction::GroupToggle(i) => UiAction::GroupToggle(i),
            AuthoredAction::Setting { key, verb } => UiAction::Setting { key, verb },
        }
    }
}

impl UiAction {
    // The authored form, or `None` for `Screen(Clear)`, which only the engine
    // sends.
    fn authored(&self) -> Option<Resolved> {
        Some(match self {
            UiAction::Quit => AuthoredAction::Quit,
            UiAction::Scene(id) => AuthoredAction::Scene(Ref::new(*id)),
            UiAction::Screen(ScreenCommand::Show(id)) => AuthoredAction::Show(Ref::new(*id)),
            UiAction::Screen(ScreenCommand::Push(id)) => AuthoredAction::Push(Ref::new(*id)),
            UiAction::Screen(ScreenCommand::Toggle(id)) => AuthoredAction::Toggle(Ref::new(*id)),
            UiAction::Screen(ScreenCommand::Hide) => AuthoredAction::Hide,
            UiAction::Screen(ScreenCommand::Clear) => return None,
            UiAction::Story(cmd) => AuthoredAction::Story(cmd.clone()),
            UiAction::GroupToggle(i) => AuthoredAction::GroupToggle(*i),
            UiAction::Setting { key, verb } => AuthoredAction::Setting {
                key: *key,
                verb: *verb,
            },
        })
    }
}

impl AssetFields for UiAction {
    fn collect_fields(prefix: &str, out: &mut FieldTable) {
        Resolved::collect_fields(prefix, out);
    }
}

impl Serialize for SettingVerb {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SettingVerb {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let verb = alloc::string::String::deserialize(d)?;
        SettingVerb::parse(&verb)
            .ok_or_else(|| de::Error::custom(format_args!("unknown setting verb {verb:?}")))
    }
}

impl Serialize for UiAction {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let engine_only = || S::Error::custom("screen clear is sent only by the engine");
        if s.is_human_readable() {
            self.authored().ok_or_else(engine_only)?.serialize(s)
        } else {
            Encoded::encode(self).ok_or_else(engine_only)?.serialize(s)
        }
    }
}

impl<'de> Deserialize<'de> for UiAction {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            Resolved::deserialize(d).map(UiAction::from)
        } else {
            Encoded::deserialize(d)?
                .decode()
                .ok_or_else(|| de::Error::custom("invalid encoded action"))
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

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
    fn every_action_round_trips_through_json_and_postcard() {
        let all = every_action();
        for action in &all {
            let json = serde_json::to_value(action).unwrap();
            assert_eq!(
                &serde_json::from_value::<UiAction>(json.clone()).unwrap(),
                action,
                "{json}"
            );
        }
        let bytes = postcard::to_allocvec(&all).unwrap();
        let back: Vec<UiAction> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, all);
    }

    #[test]
    fn a_target_name_resolves_through_the_installed_resolver() {
        crate::test_support::install_resolvers();
        let action: UiAction =
            serde_json::from_value(serde_json::json!({"push": "pause"})).unwrap();
        assert_eq!(action, UiAction::Screen(ScreenCommand::Push(AssetId(5))));
        assert_eq!(
            serde_json::to_value(&action).unwrap(),
            serde_json::json!({"push": 5})
        );
    }

    #[test]
    fn clear_never_serializes() {
        let clear = UiAction::Screen(ScreenCommand::Clear);
        assert!(serde_json::to_string(&clear).is_err());
        assert!(postcard::to_allocvec(&clear).is_err());
    }

    #[test]
    fn an_encoded_tag_past_the_last_variant_is_rejected() {
        assert!(postcard::from_bytes::<UiAction>(&[9]).is_err());
        assert!(postcard::from_bytes::<UiAction>(&[7, 13, 0]).is_err());
    }

    #[test]
    fn an_optional_action_reads_null_and_missing_as_none() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Holder {
            #[serde(default)]
            action: Option<UiAction>,
        }
        for json in [r#"{"action":null}"#, "{}"] {
            assert_eq!(
                serde_json::from_str::<Holder>(json).unwrap().action,
                None,
                "{json}"
            );
        }
        for action in [None, Some(UiAction::GroupToggle(4))] {
            let h = Holder { action };
            let bytes = postcard::to_allocvec(&h).unwrap();
            assert_eq!(postcard::from_bytes::<Holder>(&bytes).unwrap(), h);
        }
    }
}
