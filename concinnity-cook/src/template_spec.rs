// The std bridge for `concinnity-templates`.
//
// The templates crate is `#![no_std]` and describes assets as typed `AssetSpec`
// / `ArgValue` data. This turns that data into the engine's `serde_json::Value`
// so a std consumer never parses a JSON string: the cook pipeline builds UI
// assets from the shared `asset::` builders, and authoring consumers turn a spec
// into a world-line value. The cook crate is the natural home: it owns the
// build-time asset construction that needs this, and (unlike concinnity-core) it
// is excluded from the shipped runtime, so a game never links this authoring
// code. The app authoring layer re-exports these; the editor reaches them there.

use concinnity_templates::{ArgValue, AssetSpec};
use serde_json::{Map, Number, Value};

// One `ArgValue` as a `serde_json::Value`.
pub fn arg_value_to_json(v: &ArgValue) -> Value {
    match v {
        ArgValue::Null => Value::Null,
        ArgValue::Bool(b) => Value::Bool(*b),
        ArgValue::Int(i) => Value::Number(Number::from(*i)),
        ArgValue::Float(f) => Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        ArgValue::Str(s) => Value::String(s.clone()),
        ArgValue::Array(items) => Value::Array(items.iter().map(arg_value_to_json).collect()),
        ArgValue::Object(fields) => Value::Object(object_map(fields)),
    }
}

// A spec's `args` object as a `serde_json::Value` (what a component deserializes
// from).
pub fn spec_args(spec: &AssetSpec) -> Value {
    Value::Object(object_map(&spec.fields))
}

// A spec as a full world-line value: `{"name", "type", "args"}`, the shape
// `parse_world_jsonl` yields and the cook pipeline validates.
pub fn spec_to_value(spec: &AssetSpec) -> Value {
    let mut obj = Map::new();
    obj.insert("name".to_string(), Value::String(spec.name.clone()));
    obj.insert(
        "type".to_string(),
        Value::String(spec.asset_type.to_string()),
    );
    obj.insert("args".to_string(), spec_args(spec));
    Value::Object(obj)
}

fn object_map(fields: &[(String, ArgValue)]) -> Map<String, Value> {
    let mut obj = Map::new();
    for (k, v) in fields {
        obj.insert(k.clone(), arg_value_to_json(v));
    }
    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_value_maps_each_shape() {
        assert_eq!(arg_value_to_json(&ArgValue::Bool(true)), Value::Bool(true));
        assert_eq!(arg_value_to_json(&ArgValue::Int(48)), serde_json::json!(48));
        assert_eq!(
            arg_value_to_json(&ArgValue::floats(&[1.0, 0.0, 0.0])),
            serde_json::json!([1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn spec_args_wraps_fields_in_order() {
        let spec = AssetSpec::new("s", "Sprite")
            .set("x", 10.0f32)
            .set("y", 20.0f32);
        let v = spec_args(&spec);
        assert_eq!(v["x"], 10.0);
        assert_eq!(v["y"], 20.0);
    }

    #[test]
    fn spec_to_value_wraps_name_type_args() {
        let spec = AssetSpec::new("lamp", "PointLight").set("intensity", 8.0f32);
        let v = spec_to_value(&spec);
        assert_eq!(v["name"], "lamp");
        assert_eq!(v["type"], "PointLight");
        assert_eq!(v["args"]["intensity"], 8.0);
    }
}
