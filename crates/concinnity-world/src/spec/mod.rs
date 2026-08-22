//! Typed asset specs: the struct form of a world.jsonl line.
//!
//! An `AssetSpec` is a `{name, type, args}` entry described as data rather than a
//! JSON string. `args` is an ordered list of `(key, ArgValue)` pairs, where
//! `ArgValue` is a small JSON-shaped value tree. Builders in `asset` assemble
//! these; `json` converts one into the engine's `serde_json::Value` so a consumer
//! never parses a JSON string. This is the substrate both the asset builders and
//! the world templates in `crate::template` are expressed in.

pub mod asset;

mod json;

pub use json::{arg_value_to_json, spec_args, spec_to_value};

/// A JSON-shaped value: exactly the shapes an asset's `args` object can hold. Kept
/// deliberately small so it maps one-to-one onto the engine's accept path
/// (`serde_json::from_value` with `#[serde(default)]`), where an omitted field
/// falls back to its default.
#[derive(Clone, Debug, PartialEq)]
pub enum ArgValue {
    /// A JSON null.
    Null,
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A floating-point number.
    Float(f64),
    /// A string.
    Str(String),
    /// An array of values.
    Array(Vec<ArgValue>),
    /// An object, in insertion order (nested args reuse the same ordered shape).
    Object(Vec<(String, ArgValue)>),
}

impl ArgValue {
    /// A numeric array from float components (colours, positions, sizes).
    pub fn floats(vals: &[f32]) -> ArgValue {
        ArgValue::Array(vals.iter().map(|&v| ArgValue::Float(v as f64)).collect())
    }
}

impl From<bool> for ArgValue {
    fn from(v: bool) -> Self {
        ArgValue::Bool(v)
    }
}
impl From<i64> for ArgValue {
    fn from(v: i64) -> Self {
        ArgValue::Int(v)
    }
}
impl From<u32> for ArgValue {
    fn from(v: u32) -> Self {
        ArgValue::Int(v as i64)
    }
}
impl From<usize> for ArgValue {
    fn from(v: usize) -> Self {
        ArgValue::Int(v as i64)
    }
}
impl From<f32> for ArgValue {
    fn from(v: f32) -> Self {
        ArgValue::Float(v as f64)
    }
}
impl From<f64> for ArgValue {
    fn from(v: f64) -> Self {
        ArgValue::Float(v)
    }
}
impl From<&str> for ArgValue {
    fn from(v: &str) -> Self {
        ArgValue::Str(String::from(v))
    }
}
impl From<String> for ArgValue {
    fn from(v: String) -> Self {
        ArgValue::Str(v)
    }
}
impl<const N: usize> From<[f32; N]> for ArgValue {
    fn from(v: [f32; N]) -> Self {
        ArgValue::floats(&v)
    }
}

/// A named asset entry: the `{name, type, args}` of one world.jsonl line, as data.
#[derive(Clone, Debug, PartialEq)]
pub struct AssetSpec {
    /// The readable asset name (the world-line key). Ignored when a spec is
    /// materialized straight into a live component, which carries no name field.
    pub name: String,
    /// The registered asset type string (`"Sprite"`, `"DirectionalLight"`, ...).
    pub asset_type: &'static str,
    /// The `args` object, in insertion order.
    pub fields: Vec<(String, ArgValue)>,
}

impl AssetSpec {
    /// An entry of `asset_type` named `name`, with no args set yet (each unset
    /// field takes the type's serde default when materialized).
    pub fn new(name: impl Into<String>, asset_type: &'static str) -> Self {
        AssetSpec {
            name: name.into(),
            asset_type,
            fields: Vec::new(),
        }
    }

    /// Set one arg, chainable. A later `set` of the same key appends a second
    /// entry; builders set each key once.
    pub fn set(mut self, key: impl Into<String>, value: impl Into<ArgValue>) -> Self {
        self.fields.push((key.into(), value.into()));
        self
    }

    /// The `args` object as a single `ArgValue`.
    pub fn args(&self) -> ArgValue {
        ArgValue::Object(self.fields.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_records_fields_in_order() {
        let spec = AssetSpec::new("lamp", "PointLight")
            .set("intensity", 8.0f32)
            .set("range", 20.0f32);
        assert_eq!(spec.name, "lamp");
        assert_eq!(spec.asset_type, "PointLight");
        assert_eq!(
            spec.fields
                .iter()
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>(),
            vec!["intensity", "range"]
        );
    }

    #[test]
    fn conversions_pick_the_expected_variant() {
        assert_eq!(ArgValue::from(true), ArgValue::Bool(true));
        assert_eq!(ArgValue::from(48u32), ArgValue::Int(48));
        assert_eq!(ArgValue::from(1.5f32), ArgValue::Float(1.5));
        assert_eq!(ArgValue::from("sky"), ArgValue::Str("sky".to_string()));
        assert_eq!(
            ArgValue::from([1.0f32, 0.0, 0.0]),
            ArgValue::Array(vec![
                ArgValue::Float(1.0),
                ArgValue::Float(0.0),
                ArgValue::Float(0.0),
            ])
        );
    }

    #[test]
    fn args_wraps_fields_as_an_object() {
        let spec = AssetSpec::new("s", "Sprite").set("visible", false);
        assert_eq!(
            spec.args(),
            ArgValue::Object(vec![("visible".to_string(), ArgValue::Bool(false))])
        );
    }
}
