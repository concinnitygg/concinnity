// Bindings: a named value an overlay element reads or writes, wherever the
// value lives, and the app-owned state and event stream they reach.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use concinnity_core::math::SaturatingFrom;
use core::fmt::Write as _;
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};

use crate::ecs::asset_id::AssetId;
use crate::settings::SettingKey;

/// A named value an overlay element reads or writes, written `"setting:vsync"`,
/// `"var:score"`, or `"state:grid"`.
///
/// - `setting:` names a user setting, stepped and set as its settings row does.
/// - `var:` names a variable the world's [Variables](#variables) declares, the
///   same one its behaviors read.
/// - `state:` names a value in the app's UI state: kept apart from the world's
///   variables, read and written by the app's own systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Binding {
    /// A user setting.
    Setting(SettingKey),
    /// A world variable, by name.
    Var(String),
    /// A UI state value, by name.
    State(String),
}

impl Binding {
    /// The binding a `kind:name` string names, or why it names none.
    ///
    /// # Errors
    ///
    /// Returns a message when the kind is unknown or the setting key names no
    /// setting.
    pub fn parse(text: &str) -> Result<Self, String> {
        let (kind, name) = text
            .split_once(':')
            .ok_or_else(|| format!("binding {text:?} is not `kind:name`"))?;
        if name.is_empty() {
            return Err(format!("binding {text:?} names nothing"));
        }
        match kind {
            "setting" => SettingKey::parse(name)
                .map(Self::Setting)
                .ok_or_else(|| format!("binding {text:?} names no setting")),
            "var" => Ok(Self::Var(name.to_string())),
            "state" => Ok(Self::State(name.to_string())),
            _ => Err(format!(
                "binding {text:?} is not a `setting:`, `var:`, or `state:` binding"
            )),
        }
    }

    /// The binding as it is written.
    pub fn text(&self) -> String {
        match self {
            Self::Setting(key) => format!("setting:{}", key.as_str()),
            Self::Var(name) => format!("var:{name}"),
            Self::State(name) => format!("state:{name}"),
        }
    }
}

#[cfg(feature = "schema")]
impl crate::ecs::schema::Described for Binding {
    const TYPE: crate::ecs::schema::FieldType = crate::ecs::schema::FieldType::Str;
}

// The blob form: the setting key as its position in the key table.
#[derive(Serialize, Deserialize)]
enum EncodedBinding {
    Setting(u8),
    Var(String),
    State(String),
}

impl Serialize for Binding {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            return s.serialize_str(&self.text());
        }
        match self {
            Self::Setting(key) => {
                let index = SettingKey::ALL
                    .iter()
                    .position(|k| k == key)
                    .and_then(|i| u8::try_from(i).ok())
                    .ok_or_else(|| serde::ser::Error::custom("an unlisted setting key"))?;
                EncodedBinding::Setting(index).serialize(s)
            }
            Self::Var(name) => EncodedBinding::Var(name.clone()).serialize(s),
            Self::State(name) => EncodedBinding::State(name.clone()).serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for Binding {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            let text = String::deserialize(d)?;
            return Self::parse(&text).map_err(de::Error::custom);
        }
        Ok(match EncodedBinding::deserialize(d)? {
            EncodedBinding::Setting(index) => Self::Setting(
                *SettingKey::ALL
                    .get(index as usize)
                    .ok_or_else(|| de::Error::custom("an encoded setting past the table"))?,
            ),
            EncodedBinding::Var(name) => Self::Var(name),
            EncodedBinding::State(name) => Self::State(name),
        })
    }
}

/// A value a binding holds or is set to: a boolean, a whole number,
/// a number, or text, written as the plain JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum UiValue {
    /// `true` or `false`.
    Bool(bool),
    /// A whole number.
    Int(i32),
    /// A number with a fractional part.
    Float(f32),
    /// Text.
    Text(String),
}

impl UiValue {
    /// The value as a label shows it.
    pub fn display(&self) -> String {
        match self {
            Self::Bool(b) => b.to_string(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{f}"),
            Self::Text(t) => t.clone(),
        }
    }

    /// Write the value as a label shows it into `out`, replacing what it held.
    #[expect(
        clippy::let_underscore_must_use,
        reason = "writing to a String cannot fail"
    )]
    pub fn write_display(&self, out: &mut String) {
        out.clear();
        let _ = match self {
            Self::Bool(b) => write!(out, "{b}"),
            Self::Int(i) => write!(out, "{i}"),
            Self::Float(f) => write!(out, "{f}"),
            Self::Text(t) => {
                out.push_str(t);
                Ok(())
            }
        };
    }

    /// The value as a number, `None` for text.
    pub fn number(&self) -> Option<f32> {
        match self {
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Self::Int(i) => Some(*i as f32),
            Self::Float(f) => Some(*f),
            Self::Text(_) => None,
        }
    }
}

#[derive(Serialize, Deserialize)]
enum EncodedValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    Text(String),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    Text(String),
}

impl Serialize for UiValue {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match (s.is_human_readable(), self) {
            (true, Self::Bool(b)) => s.serialize_bool(*b),
            (true, Self::Int(i)) => s.serialize_i32(*i),
            (true, Self::Float(f)) => s.serialize_f32(*f),
            (true, Self::Text(t)) => s.serialize_str(t),
            (false, Self::Bool(b)) => EncodedValue::Bool(*b).serialize(s),
            (false, Self::Int(i)) => EncodedValue::Int(*i).serialize(s),
            (false, Self::Float(f)) => EncodedValue::Float(*f).serialize(s),
            (false, Self::Text(t)) => EncodedValue::Text(t.clone()).serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for UiValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            return Ok(match JsonValue::deserialize(d)? {
                JsonValue::Bool(b) => Self::Bool(b),
                JsonValue::Int(i) => Self::Int(i),
                JsonValue::Float(f) => Self::Float(f),
                JsonValue::Text(t) => Self::Text(t),
            });
        }
        Ok(match EncodedValue::deserialize(d)? {
            EncodedValue::Bool(b) => Self::Bool(b),
            EncodedValue::Int(i) => Self::Int(i),
            EncodedValue::Float(f) => Self::Float(f),
            EncodedValue::Text(t) => Self::Text(t),
        })
    }
}

#[cfg(feature = "schema")]
impl crate::ecs::schema::Described for UiValue {
    const TYPE: crate::ecs::schema::FieldType = crate::ecs::schema::FieldType::Object;
}

/// One value the UI state starts with: its name and its starting value, whose
/// type the value fixes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, crate::ecs::AssetFields)]
pub struct UiStateValue {
    /// The name `state:` bindings use.
    pub name: String,
    /// The starting value, written as plain JSON: a boolean, a whole number, a
    /// number with a fractional part, or text.
    pub value: UiValue,
}

/// The values the app's UI state holds when the world loads.
///
/// Every name a `state:` binding uses is declared here, and its starting value
/// fixes its type, so the build catches a misspelled name or an action that
/// does not fit the value (flipping a number, adding to text). A world may
/// declare several, one per screen, say; a name is declared once across them.
///
/// ```rust
/// # use concinnity_engine::components::{UiStateValue, UiStateValues, UiValue};
/// UiStateValues {
///     values: vec![UiStateValue {
///         name: "show_grid".into(),
///         value: UiValue::Bool(true),
///     }],
/// };
/// ```
#[derive(
    Debug, Clone, Serialize, Deserialize, crate::ecs::AssetFields, crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct UiStateValues {
    /// The declared values.
    pub values: alloc::vec::Vec<UiStateValue>,
}

/// What an action does to the value a binding names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueOp {
    /// Set it to this value.
    Set(UiValue),
    /// Turn a boolean over: `false` to `true` and back.
    Flip,
    /// Add this amount to a number.
    Add(f32),
    /// Step a setting through its options, or a whole number, by this many.
    Step(i32),
}

impl ValueOp {
    /// The value `current` becomes, or `None` when the op does not apply to
    /// it (adding to text, flipping a number). A value not yet held reads as
    /// `false` for a flip and 0 for a number.
    pub fn apply(&self, current: Option<&UiValue>) -> Option<UiValue> {
        match (self, current) {
            (Self::Set(value), _) => Some(value.clone()),
            (Self::Flip, None) => Some(UiValue::Bool(true)),
            (Self::Flip, Some(UiValue::Bool(b))) => Some(UiValue::Bool(!b)),
            (Self::Add(by), None) => Some(UiValue::Float(*by)),
            (Self::Add(by), Some(UiValue::Float(f))) => Some(UiValue::Float(f + by)),
            (Self::Add(by), Some(UiValue::Int(i))) => Some(UiValue::Int(
                i.saturating_add(i32::saturating_from(concinnity_core::math::round(*by))),
            )),
            (Self::Step(by), None) => Some(UiValue::Int(*by)),
            (Self::Step(by), Some(UiValue::Int(i))) => Some(UiValue::Int(i.saturating_add(*by))),
            (Self::Step(by), Some(UiValue::Float(f))) => Some(UiValue::Float(f + *by as f32)),
            _ => None,
        }
    }
}

/// The app's UI state: named values the overlay reads and writes through
/// `state:` bindings, and the app's own systems read and write directly. Kept
/// apart from the world's variables.
#[derive(Debug, Clone, Default)]
pub struct UiState {
    values: BTreeMap<String, UiValue>,
    revision: u64,
}

impl UiState {
    /// The value held under `name`.
    pub fn get(&self, name: &str) -> Option<&UiValue> {
        self.values.get(name)
    }

    /// Hold `value` under `name`.
    pub fn set(&mut self, name: &str, value: UiValue) {
        if self.values.get(name) != Some(&value) {
            self.values.insert(name.to_string(), value);
            self.revision = self.revision.wrapping_add(1);
        }
    }

    /// Hold `value` under `name` unless something already is.
    pub fn seed(&mut self, name: &str, value: UiValue) {
        if !self.values.contains_key(name) {
            self.set(name, value);
        }
    }

    /// A count that changes whenever a value does, so a reader can skip
    /// rereading an unchanged state.
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

/// What an `emit` action sends: a name, written alone as `{"emit": "save"}`,
/// or with a value as `{"emit": {"name": "pick", "value": 3}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct UiEmit {
    /// The name the app's systems match the event by.
    pub name: String,
    /// What the event carries, if anything.
    pub value: Option<UiValue>,
}

impl From<&str> for UiEmit {
    fn from(name: &str) -> Self {
        Self {
            name: name.to_string(),
            value: None,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmitFields {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    value: Option<UiValue>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmitText {
    Name(String),
    Fields(EmitFields),
}

impl Serialize for UiEmit {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() && self.value.is_none() {
            return s.serialize_str(&self.name);
        }
        if s.is_human_readable() {
            return EmitFields {
                name: self.name.clone(),
                value: self.value.clone(),
            }
            .serialize(s);
        }
        (&self.name, &self.value).serialize(s)
    }
}

impl<'de> Deserialize<'de> for UiEmit {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            return Ok(match EmitText::deserialize(d)? {
                EmitText::Name(name) => Self { name, value: None },
                EmitText::Fields(EmitFields { name, value }) => Self { name, value },
            });
        }
        let (name, value) = <(String, Option<UiValue>)>::deserialize(d)?;
        Ok(Self { name, value })
    }
}

/// An `emit` action fired by a click or key press, for the app's own systems
/// to act on.
#[derive(Debug, Clone, PartialEq)]
pub struct UiEvent {
    /// The name the action emits.
    pub name: String,
    /// The [`HitRegion`](#hitregion) or [`KeyBinding`](#keybinding) that fired
    /// it, when it has an id.
    pub element: Option<AssetId>,
    /// What the action carries, if anything.
    pub value: Option<UiValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_read_and_write_as_kind_and_name() {
        for text in ["setting:vsync", "var:score", "state:grid"] {
            let binding: Binding = serde_json::from_value(serde_json::json!(text)).unwrap();
            assert_eq!(serde_json::to_value(&binding).unwrap(), text);
        }
        assert_eq!(
            Binding::parse("setting:vsync"),
            Ok(Binding::Setting(SettingKey::Vsync))
        );
        for bad in ["vsync", "setting:nope", "colour:red", "state:"] {
            assert!(Binding::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn bindings_and_values_round_trip_through_postcard() {
        let all = [
            (Binding::Setting(SettingKey::Fov), UiValue::Float(1.5)),
            (Binding::Var("score".into()), UiValue::Int(-3)),
            (Binding::State("grid".into()), UiValue::Text("on".into())),
            (Binding::State("open".into()), UiValue::Bool(true)),
        ];
        for (binding, value) in all {
            let bytes = postcard::to_allocvec(&(binding.clone(), value.clone())).unwrap();
            let back: (Binding, UiValue) = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back, (binding, value));
        }
    }

    #[test]
    fn values_read_as_plain_json() {
        let read = |json| serde_json::from_value::<UiValue>(json).unwrap();
        assert_eq!(read(serde_json::json!(true)), UiValue::Bool(true));
        assert_eq!(read(serde_json::json!(4)), UiValue::Int(4));
        assert_eq!(read(serde_json::json!(0.5)), UiValue::Float(0.5));
        assert_eq!(read(serde_json::json!("hi")), UiValue::Text("hi".into()));
    }

    #[test]
    fn ops_change_the_values_they_apply_to() {
        let int = UiValue::Int(2);
        assert_eq!(ValueOp::Flip.apply(None), Some(UiValue::Bool(true)));
        assert_eq!(
            ValueOp::Flip.apply(Some(&UiValue::Bool(true))),
            Some(UiValue::Bool(false))
        );
        assert_eq!(ValueOp::Flip.apply(Some(&int)), None);
        assert_eq!(
            ValueOp::Add(1.5).apply(Some(&UiValue::Float(1.0))),
            Some(UiValue::Float(2.5))
        );
        assert_eq!(ValueOp::Step(-3).apply(Some(&int)), Some(UiValue::Int(-1)));
        assert_eq!(
            ValueOp::Add(1.0).apply(Some(&UiValue::Text("x".into()))),
            None
        );
        assert_eq!(ValueOp::Set(int.clone()).apply(None), Some(int));
    }

    #[test]
    fn state_counts_only_real_changes() {
        let mut state = UiState::default();
        state.set("a", UiValue::Bool(true));
        let after_set = state.revision();
        state.set("a", UiValue::Bool(true));
        assert_eq!(state.revision(), after_set, "an unchanged set");
        state.seed("a", UiValue::Bool(false));
        assert_eq!(
            state.get("a"),
            Some(&UiValue::Bool(true)),
            "seed keeps a held value"
        );
        state.seed("b", UiValue::Int(1));
        assert_eq!(state.get("b"), Some(&UiValue::Int(1)));
    }
}
