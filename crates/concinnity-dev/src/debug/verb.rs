//! The vocabulary every debug verb is declared in: what it is called, whether
//! it reads or mutates the world, its typed parameters, and the handler that
//! answers it.
//!
//! A verb's parameters are the one description of its arguments. They render as
//! the JSON Schema a client sees, and the same list checks every call before
//! the handler runs: an undeclared key, a missing required parameter, or a value
//! of the wrong type is refused with the parameter's name, so a handler only
//! ever reads arguments of the declared shape. A parameter is `required` when a
//! call without it is refused, and a blank `Name` counts as missing; every
//! other parameter carries its default in its description.

use serde_json::{Map, Value, json};

use super::call::Call;

/// Whether a verb reads the world or mutates it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Access {
    /// Answers without changing the world, whether from the per-frame snapshot
    /// or from a query the engine thread runs on its behalf.
    ReadOnly,
    /// Changes the running world, or the engine's view of it.
    Mutating,
}

impl Access {
    pub(crate) fn is_read_only(self) -> bool {
        matches!(self, Access::ReadOnly)
    }
}

/// The JSON type of one verb parameter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    Text,
    /// A string naming something, refused as missing when blank.
    Name,
    TextOrNull,
    Number,
    NumberOrNull,
    /// A non-negative whole number.
    Count,
    Vec3,
    Vec4,
    NumberList,
    /// A string drawn from a closed set. The set reaches the client as a JSON
    /// Schema `enum`, so it is not restated in the description.
    Choice(&'static [&'static str]),
}

impl Kind {
    fn json_type(self) -> Value {
        let numbers = |len: usize| {
            json!({
                "type": "array",
                "items": { "type": "number" },
                "minItems": len,
                "maxItems": len,
            })
        };
        match self {
            Kind::Text => json!({ "type": "string" }),
            Kind::Name => json!({ "type": "string", "pattern": "\\S" }),
            Kind::TextOrNull => json!({ "type": ["string", "null"] }),
            Kind::Number => json!({ "type": "number" }),
            Kind::NumberOrNull => json!({ "type": ["number", "null"] }),
            Kind::Count => json!({ "type": "integer", "minimum": 0 }),
            Kind::Vec3 => numbers(3),
            Kind::Vec4 => numbers(4),
            Kind::NumberList => json!({ "type": "array", "items": { "type": "number" } }),
            Kind::Choice(values) => json!({ "type": "string", "enum": values }),
        }
    }

    fn accepts(self, value: &Value) -> bool {
        match self {
            Kind::Text | Kind::Name => value.is_string(),
            Kind::TextOrNull => value.is_string() || value.is_null(),
            Kind::Number => value.is_number(),
            Kind::NumberOrNull => value.is_number() || value.is_null(),
            Kind::Count => value.is_u64(),
            Kind::Vec3 => is_numbers(value, Some(3)),
            Kind::Vec4 => is_numbers(value, Some(4)),
            Kind::NumberList => is_numbers(value, None),
            Kind::Choice(values) => value.as_str().is_some_and(|v| values.contains(&v)),
        }
    }

    fn expected(self) -> String {
        match self {
            Kind::Text | Kind::Name => "a string".into(),
            Kind::TextOrNull => "a string or null".into(),
            Kind::Number => "a number".into(),
            Kind::NumberOrNull => "a number or null".into(),
            Kind::Count => "a non-negative integer".into(),
            Kind::Vec3 => "an array of 3 numbers".into(),
            Kind::Vec4 => "an array of 4 numbers".into(),
            Kind::NumberList => "an array of numbers".into(),
            Kind::Choice(values) => format!("one of {}", values.join(" | ")),
        }
    }
}

fn is_blank(value: &Value) -> bool {
    value.as_str().is_some_and(|text| text.trim().is_empty())
}

fn is_numbers(value: &Value, len: Option<usize>) -> bool {
    value.as_array().is_some_and(|items| {
        len.is_none_or(|len| items.len() == len) && items.iter().all(Value::is_number)
    })
}

/// One parameter of a verb.
pub(super) struct Param {
    pub(super) name: &'static str,
    pub(super) kind: Kind,
    /// True when a call that omits this parameter is refused.
    pub(super) required: bool,
    pub(super) description: &'static str,
}

pub(super) const fn required(name: &'static str, kind: Kind, description: &'static str) -> Param {
    Param {
        name,
        kind,
        required: true,
        description,
    }
}

pub(super) const fn optional(name: &'static str, kind: Kind, description: &'static str) -> Param {
    Param {
        name,
        kind,
        required: false,
        description,
    }
}

/// A verb's answer: the reply's fields as a JSON object, or why the call failed.
pub(super) type Reply = Result<Value, String>;

/// The reply every verb that only queues a change answers with.
pub(super) fn queued() -> Reply {
    Ok(json!({ "queued": true }))
}

/// One verb the debug server answers.
pub(crate) struct Verb {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) access: Access,
    pub(super) params: &'static [Param],
    pub(super) run: fn(&Call, Args) -> Reply,
}

impl Verb {
    /// The verb's parameters as a draft 2020-12 object schema.
    pub(crate) fn schema(&self) -> Value {
        let mut properties = Map::new();
        let mut required = Vec::new();
        for param in self.params {
            let mut property = param.kind.json_type();
            property["description"] = Value::String(param.description.to_string());
            properties.insert(param.name.to_string(), property);
            if param.required {
                required.push(Value::String(param.name.to_string()));
            }
        }
        let mut schema = json!({
            "type": "object",
            "properties": Value::Object(properties),
            "additionalProperties": false,
        });
        if !required.is_empty() {
            schema["required"] = Value::Array(required);
        }
        schema
    }

    /// Check `arguments` against the parameters and hand them over for the
    /// handler, or name the first one that does not fit.
    pub(super) fn arguments(&self, arguments: Map<String, Value>) -> Result<Args, String> {
        if let Some(key) = arguments
            .keys()
            .find(|key| !self.params.iter().any(|p| p.name == key.as_str()))
        {
            return Err(self.unknown_field(key));
        }
        for param in self.params {
            match arguments.get(param.name) {
                None if param.required => return Err(self.missing(param)),
                Some(value) if !param.kind.accepts(value) => {
                    return Err(self.mistyped(param, value));
                }
                Some(value) if param.kind == Kind::Name && is_blank(value) => {
                    return Err(self.missing(param));
                }
                _ => {}
            }
        }
        Ok(Args {
            verb: self.name,
            fields: arguments,
        })
    }

    fn missing(&self, param: &Param) -> String {
        format!("{}: missing '{}'", self.name, param.name)
    }

    fn unknown_field(&self, key: &str) -> String {
        if self.params.is_empty() {
            return format!("{}: unknown field `{key}`, there are no fields", self.name);
        }
        let expected: Vec<String> = self
            .params
            .iter()
            .map(|p| format!("`{}`", p.name))
            .collect();
        format!(
            "{}: unknown field `{key}`, expected one of {}",
            self.name,
            expected.join(", ")
        )
    }

    fn mistyped(&self, param: &Param, value: &Value) -> String {
        match (param.kind, value.as_str()) {
            (Kind::Choice(values), Some(text)) => format!(
                "{}: unknown {} '{text}' (use {})",
                self.name,
                param.name,
                values.join(" | ")
            ),
            (kind, _) => format!(
                "{}: '{}' must be {}",
                self.name,
                param.name,
                kind.expected()
            ),
        }
    }
}

/// One call's arguments, already checked against its verb's parameters.
pub(super) struct Args {
    verb: &'static str,
    fields: Map<String, Value>,
}

impl Args {
    /// The arguments as a request struct. An omitted optional parameter takes
    /// the struct's serde default.
    pub(super) fn parse<T: serde::de::DeserializeOwned>(self) -> Result<T, String> {
        serde_json::from_value(Value::Object(self.fields))
            .map_err(|e| format!("{}: invalid arguments: {e}", self.verb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nothing(_: &Call, _: Args) -> Reply {
        Ok(json!({}))
    }

    const SHAPES: Verb = Verb {
        name: "shapes",
        description: "Exercise every parameter kind.",
        access: Access::Mutating,
        params: &[
            required("label", Kind::Text, "A label."),
            optional("id", Kind::Name, "An identifier."),
            optional("tag", Kind::TextOrNull, "A tag."),
            optional("scale", Kind::Number, "A scale."),
            optional("limit", Kind::NumberOrNull, "A limit."),
            optional("frames", Kind::Count, "A count."),
            optional("at", Kind::Vec3, "A point."),
            optional("tint", Kind::Vec4, "A color."),
            optional("weights", Kind::NumberList, "Weights."),
            optional("op", Kind::Choice(&["next", "prev"]), "A direction."),
        ],
        run: nothing,
    };

    fn check(arguments: Value) -> Result<(), String> {
        let Value::Object(map) = arguments else {
            panic!("arguments are an object");
        };
        SHAPES.arguments(map).map(|_| ())
    }

    #[test]
    fn a_call_of_every_declared_kind_is_accepted() {
        check(json!({
            "label": "a",
            "tag": null,
            "scale": 1,
            "limit": null,
            "frames": 3,
            "at": [0, 1.5, 2],
            "tint": [1, 1, 1, 0.5],
            "weights": [],
            "op": "prev",
        }))
        .expect("every value fits its kind");
        check(json!({ "label": "", "tag": "t", "limit": 2.5, "id": " a " }))
            .expect("optional values may vary, and a padded name is not blank");
    }

    #[test]
    fn each_kind_refuses_a_value_of_another_shape() {
        let cases = [
            (json!({ "label": 1 }), "'label' must be a string"),
            (
                json!({ "label": "a", "tag": 2 }),
                "'tag' must be a string or null",
            ),
            (
                json!({ "label": "a", "scale": "big" }),
                "'scale' must be a number",
            ),
            (
                json!({ "label": "a", "scale": null }),
                "'scale' must be a number",
            ),
            (
                json!({ "label": "a", "limit": [] }),
                "'limit' must be a number or null",
            ),
            (
                json!({ "label": "a", "frames": -1 }),
                "'frames' must be a non-negative integer",
            ),
            (
                json!({ "label": "a", "frames": 1.5 }),
                "'frames' must be a non-negative integer",
            ),
            (
                json!({ "label": "a", "at": [0, 1] }),
                "'at' must be an array of 3 numbers",
            ),
            (
                json!({ "label": "a", "at": [0, 1, "2"] }),
                "'at' must be an array of 3 numbers",
            ),
            (
                json!({ "label": "a", "tint": [0, 1, 2] }),
                "'tint' must be an array of 4 numbers",
            ),
            (
                json!({ "label": "a", "weights": [true] }),
                "'weights' must be an array of numbers",
            ),
            (
                json!({ "label": "a", "op": 1 }),
                "'op' must be one of next | prev",
            ),
            (
                json!({ "label": "a", "op": "sideways" }),
                "unknown op 'sideways' (use next | prev)",
            ),
        ];
        for (arguments, needle) in cases {
            let error = check(arguments.clone()).expect_err(&arguments.to_string());
            assert!(error.starts_with("shapes: "), "{error}");
            assert!(error.contains(needle), "{arguments}: {error}");
        }
    }

    #[test]
    fn a_missing_required_parameter_is_named() {
        assert_eq!(check(json!({})), Err("shapes: missing 'label'".to_string()));
    }

    #[test]
    fn a_blank_name_is_missing_and_a_mistyped_one_is_not_a_string() {
        for blank in ["", "  ", "\t\n"] {
            let error = check(json!({ "label": "a", "id": blank }));
            assert_eq!(error, Err("shapes: missing 'id'".to_string()), "{blank:?}");
        }
        let error = check(json!({ "label": "a", "id": 3 })).unwrap_err();
        assert!(error.contains("'id' must be a string"), "{error}");
    }

    #[test]
    fn an_undeclared_key_is_refused_with_the_declared_ones() {
        let error = check(json!({ "label": "a", "positon": [0, 0, 0] })).unwrap_err();
        assert!(error.contains("unknown field `positon`"), "{error}");
        assert!(error.contains("`label`, `id`, `tag`"), "{error}");
    }

    #[test]
    fn the_schema_lists_every_parameter_and_only_the_required_ones_as_required() {
        let schema = SHAPES.schema();
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"], json!(["label"]));
        assert_eq!(schema["properties"]["at"]["minItems"], 3);
        assert_eq!(schema["properties"]["op"]["enum"], json!(["next", "prev"]));
        assert_eq!(schema["properties"]["frames"]["description"], "A count.");
        assert_eq!(schema["properties"]["id"]["pattern"], "\\S");
    }

    #[test]
    fn parsed_arguments_take_the_struct_defaults() {
        #[derive(serde::Deserialize)]
        struct Shapes {
            label: String,
            #[serde(default)]
            frames: u32,
        }
        let Value::Object(map) = json!({ "label": "a" }) else {
            unreachable!()
        };
        let parsed: Shapes = SHAPES.arguments(map).unwrap().parse().unwrap();
        assert_eq!((parsed.label.as_str(), parsed.frames), ("a", 0));
    }
}
