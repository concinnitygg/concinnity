// A field's default as the JSON an author would write: produced by running the
// type's `Default` the way serde does for an omitted key, then serialized.

use concinnity_core::ecs::schema::{DefaultValue, FieldDefault, FieldSchema, TypeSchema};
use serde_json::Value;

/// A type's defaults, serialized once for all of its fields.
pub(super) struct Defaults {
    container: Option<Value>,
}

impl Defaults {
    pub(super) fn of(schema: &TypeSchema) -> Self {
        Self {
            container: schema
                .default
                .and_then(|default| default())
                .and_then(to_json),
        }
    }

    /// The whole type's default, when it has one that serializes.
    pub(super) fn container(&self) -> Option<&Value> {
        self.container.as_ref()
    }

    /// What `field` reads as when its key is omitted: `None` when the key is
    /// required or its default cannot be serialized. A key the container's
    /// serialized default leaves out (a `skip_serializing_if`) reads as `null`.
    pub(super) fn field(&self, field: &FieldSchema) -> Option<Value> {
        match field.default {
            FieldDefault::Required => None,
            FieldDefault::Null => Some(Value::Null),
            FieldDefault::Container => self
                .container
                .as_ref()
                .map(|container| container.get(field.key).cloned().unwrap_or(Value::Null)),
            FieldDefault::Value(default) => default().and_then(to_json),
        }
    }
}

// Serialize through JSON text rather than `serde_json::to_value`: the text form
// writes an `f32` at its own shortest precision (`0.1`), where the value form
// widens it to the `f64` beside it (`0.10000000149011612`).
fn to_json(value: DefaultValue) -> Option<Value> {
    serde_json::to_string(&*value)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

/// A default as the page states it: compact, with a space after each list
/// comma and object colon, the way it reads in prose.
pub(super) fn default_text(value: &Value) -> String {
    match value {
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(default_text).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(fields) => {
            let fields: Vec<String> = fields
                .iter()
                .map(|(key, value)| {
                    format!("{}: {}", Value::from(key.as_str()), default_text(value))
                })
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
        scalar => scalar.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::schema::{Body, FieldType};

    const FIELD: fn(FieldDefault) -> FieldSchema = |default| FieldSchema {
        key: "scale",
        doc: "",
        ty: || FieldType::Float,
        default,
    };

    const CONTAINER: TypeSchema = TypeSchema {
        name: "Scaled",
        doc: "",
        body: Body::Fields(&[]),
        default: Some(|| {
            Some(Box::new(std::collections::BTreeMap::from([(
                "scale", 0.1_f32,
            )])))
        }),
    };

    #[test]
    fn each_kind_of_default_resolves_as_serde_reads_it() {
        let defaults = Defaults::of(&CONTAINER);
        assert_eq!(defaults.field(&FIELD(FieldDefault::Required)), None);
        assert_eq!(
            defaults.field(&FIELD(FieldDefault::Null)),
            Some(Value::Null)
        );
        assert_eq!(
            defaults.field(&FIELD(FieldDefault::Value(|| Some(Box::new(3_u32))))),
            Some(Value::from(3))
        );
        assert_eq!(defaults.field(&FIELD(FieldDefault::Value(|| None))), None);
        // The container's default is read at the field's key, at the float's
        // own precision.
        let scale = defaults.field(&FIELD(FieldDefault::Container)).unwrap();
        assert_eq!(default_text(&scale), "0.1");
    }

    #[test]
    fn a_key_the_container_default_omits_reads_as_null() {
        const EMPTY: TypeSchema = TypeSchema {
            default: Some(|| Some(Box::new(std::collections::BTreeMap::<&str, u8>::new()))),
            ..CONTAINER
        };
        let defaults = Defaults::of(&EMPTY);
        assert_eq!(
            defaults.field(&FIELD(FieldDefault::Container)),
            Some(Value::Null)
        );
        // With no container default at all there is nothing to read.
        let none = Defaults::of(&TypeSchema {
            default: None,
            ..CONTAINER
        });
        assert_eq!(none.field(&FIELD(FieldDefault::Container)), None);
    }

    #[test]
    fn default_text_spaces_lists_and_objects() {
        let value = serde_json::json!({"size": [1.0, 2.5], "name": "a\"b", "on": true});
        assert_eq!(
            default_text(&value),
            r#"{"size": [1.0, 2.5], "name": "a\"b", "on": true}"#
        );
        assert_eq!(default_text(&Value::from(2048)), "2048");
        assert_eq!(default_text(&serde_json::json!([])), "[]");
    }
}
