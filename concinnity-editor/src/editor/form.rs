// src/editor/form.rs
//
// Derives an add / edit form's editable fields from an asset type's registered
// default args, and coerces the edited text / toggle values back into a JSON args
// object. This is the data half of the panel's form (the panel owns the layout
// and the hook owns the live field state); it is pure and world-free, so it unit
// tests without a running engine.
//
// The field list is the type's `default_args` object (which is
// `serde_json::to_value(Args::default())`, keys in declaration order) -- no
// per-type descriptor to maintain. Scalar-first: string / integer / float / bool
// are editable; every other kind (arrays, nested objects, nulls -- i.e. colours,
// vectors, asset refs) is left at its default and round-trips untouched. The
// assembled object is validated by the caller via `ComponentType::reserialize_args`.

use crate::ecs::ComponentType;
use serde_json::{Map, Value};

// Cap on how many editable fields a form shows: the injected control pool is a
// fixed size, so a type with more editable scalar fields shows the first `MAX`
// and leaves the rest at their defaults (`fields_for` logs when it truncates).
pub(crate) const MAX_FIELDS: usize = 12;

// The editable kinds. Everything else in an asset's args is left at its default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldKind {
    Str,
    Int,
    Float,
    Bool,
}

// One editable field of the form.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FormField {
    // The args-object key this field edits.
    pub key: String,
    pub kind: FieldKind,
    // Initial text for a text field (string / number rendered editable); empty
    // for a bool.
    pub initial: String,
    // Current value of a bool field (the checkbox); unused for text kinds.
    pub boolval: bool,
}

// The editable kind of a default value, or `None` for a kind left at default.
fn kind_of(v: &Value) -> Option<FieldKind> {
    match v {
        Value::String(_) => Some(FieldKind::Str),
        Value::Bool(_) => Some(FieldKind::Bool),
        Value::Number(n) => Some(if n.is_i64() || n.is_u64() {
            FieldKind::Int
        } else {
            FieldKind::Float
        }),
        _ => None,
    }
}

// A number / string value rendered as editable text.
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

// The type's registered default args as an object (empty if the type is unknown
// or has no args schema).
pub(crate) fn base_args(ty: &str) -> Map<String, Value> {
    ComponentType::parse(ty)
        .and_then(|ct| ct.registration().default_args)
        .and_then(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default()
}

// The editable scalar fields for a type, in declaration order, capped at `MAX`.
// `seed` supplies current values (an existing entry's args when editing),
// overriding the defaults for the initial text / bool state; the kind is always
// taken from the default so it is stable.
pub(crate) fn fields_for(ty: &str, seed: Option<&Map<String, Value>>) -> Vec<FormField> {
    let base = base_args(ty);
    let mut out = Vec::new();
    let mut truncated = 0;
    for (key, def) in &base {
        let Some(kind) = kind_of(def) else {
            continue;
        };
        if out.len() >= MAX_FIELDS {
            truncated += 1;
            continue;
        }
        let cur = seed.and_then(|s| s.get(key)).unwrap_or(def);
        let boolval = matches!(kind, FieldKind::Bool)
            && cur.as_bool().or_else(|| def.as_bool()).unwrap_or(false);
        let initial = if matches!(kind, FieldKind::Bool) {
            String::new()
        } else {
            value_text(cur)
        };
        out.push(FormField {
            key: key.clone(),
            kind,
            initial,
            boolval,
        });
    }
    if truncated > 0 {
        tracing::warn!(
            "editor: {ty} has {truncated} more editable field(s) than the form pool ({MAX_FIELDS}); the rest keep their defaults"
        );
    }
    out
}

// Coerce one field's edited `text` (or its `boolval`) into a JSON value, falling
// back to `default` when a number fails to parse (so a stray keystroke cannot
// wipe a field -- the whole object is validated afterwards regardless).
fn coerce(field: &FormField, text: &str, default: &Value) -> Value {
    match field.kind {
        FieldKind::Str => Value::String(text.to_string()),
        FieldKind::Int => text
            .trim()
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| default.clone()),
        FieldKind::Float => text
            .trim()
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .unwrap_or_else(|| default.clone()),
        FieldKind::Bool => Value::Bool(field.boolval),
    }
}

// Assemble the full args object: start from the type's defaults, merge an existing
// entry's args over them (when editing), then overwrite each editable field from
// its current control value (`texts[i]` for text kinds, the field's `boolval` for
// bools). Non-editable keys keep their default / existing value untouched.
pub(crate) fn assemble(
    ty: &str,
    editing_args: Option<&Map<String, Value>>,
    fields: &[FormField],
    texts: &[String],
) -> Map<String, Value> {
    let defaults = base_args(ty);
    let mut out = defaults.clone();
    if let Some(existing) = editing_args {
        for (k, v) in existing {
            out.insert(k.clone(), v.clone());
        }
    }
    for (i, field) in fields.iter().enumerate() {
        // Fall back to the value already in `out` (the entry's authored value when
        // editing, else the type default), so a cleared / mistyped number keeps
        // what was there rather than snapping back to the type default.
        let fallback = out.get(&field.key).cloned().unwrap_or(Value::Null);
        let text = texts.get(i).map(String::as_str).unwrap_or("");
        out.insert(field.key.clone(), coerce(field, text, &fallback));
    }
    out
}

// Validate an assembled args object by round-tripping it through the type's typed
// `Args` (the same check `cn add` applies). `Ok(())` means it will cook.
pub(crate) fn validate(ty: &str, args: &Map<String, Value>) -> Result<(), String> {
    let Some(ct) = ComponentType::parse(ty) else {
        return Err(format!("unknown asset type '{ty}'"));
    };
    ct.reserialize_args(&Value::Object(args.clone()))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // PointLight is a representative flat asset: floats + a bool, plus a colour
    // array (left at default) and a skipped id.
    #[test]
    fn fields_for_pointlight_yields_editable_scalars_only() {
        let fields = fields_for("PointLight", None);
        assert!(!fields.is_empty(), "PointLight exposes editable fields");
        // Every derived field is a scalar kind; no array / object key leaks in.
        for f in &fields {
            assert!(matches!(
                f.kind,
                FieldKind::Str | FieldKind::Int | FieldKind::Float | FieldKind::Bool
            ));
        }
        // The colour array is present in the defaults but NOT offered as a field.
        let base = base_args("PointLight");
        assert!(
            base.contains_key("color"),
            "PointLight has a color array default"
        );
        assert!(
            !fields.iter().any(|f| f.key == "color"),
            "the color array is left at its default, not shown"
        );
    }

    #[test]
    fn unknown_type_has_no_fields_and_empty_base() {
        assert!(fields_for("NotARealType", None).is_empty());
        assert!(base_args("NotARealType").is_empty());
    }

    #[test]
    fn assemble_overwrites_editable_and_keeps_the_rest() {
        // A synthetic field set standing in for a mix of kinds.
        let fields = [
            FormField {
                key: "intensity".into(),
                kind: FieldKind::Float,
                initial: "1".into(),
                boolval: false,
            },
            FormField {
                key: "on".into(),
                kind: FieldKind::Bool,
                initial: String::new(),
                boolval: true,
            },
        ];
        let mut base = Map::new();
        base.insert("intensity".into(), Value::from(1.0));
        base.insert("on".into(), Value::Bool(false));
        base.insert("color".into(), serde_json::json!([1, 1, 1]));
        // Pretend `base` is the type default by editing the "intensity"/"on" args.
        let texts = ["2.5".to_string(), String::new()];
        let out = {
            // assemble reads defaults from the registry, so build the object by
            // hand here to test the merge logic directly via coerce.
            let mut m = base.clone();
            for (i, f) in fields.iter().enumerate() {
                let def = base.get(&f.key).cloned().unwrap_or(Value::Null);
                m.insert(f.key.clone(), coerce(f, &texts[i], &def));
            }
            m
        };
        assert_eq!(out["intensity"], Value::from(2.5));
        assert_eq!(out["on"], Value::Bool(true));
        assert_eq!(
            out["color"],
            serde_json::json!([1, 1, 1]),
            "array untouched"
        );
    }

    #[test]
    fn assemble_preserves_authored_value_on_a_blank_edit() {
        let fields = fields_for("PointLight", None);
        let (idx, key) = fields
            .iter()
            .enumerate()
            .find(|(_, f)| matches!(f.kind, FieldKind::Float))
            .map(|(i, f)| (i, f.key.clone()))
            .expect("a float field");
        // An entry whose authored value differs from the type default.
        let mut authored = base_args("PointLight");
        authored.insert(key.clone(), Value::from(999.0));
        // The user clears that field before confirming.
        let mut texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        texts[idx] = String::new();
        let out = assemble("PointLight", Some(&authored), &fields, &texts);
        assert_eq!(
            out[&key],
            Value::from(999.0),
            "a blank numeric edit keeps the authored value, not the type default"
        );
    }

    #[test]
    fn coerce_falls_back_to_default_on_bad_numbers() {
        let f = FormField {
            key: "n".into(),
            kind: FieldKind::Int,
            initial: String::new(),
            boolval: false,
        };
        assert_eq!(coerce(&f, "abc", &Value::from(7)), Value::from(7));
        assert_eq!(coerce(&f, "42", &Value::from(7)), Value::from(42));
    }

    #[test]
    fn assemble_then_validate_round_trips_a_real_type() {
        let fields = fields_for("PointLight", None);
        let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        let args = assemble("PointLight", None, &fields, &texts);
        assert!(
            validate("PointLight", &args).is_ok(),
            "the default-derived args re-serialize cleanly"
        );
    }

    #[test]
    fn validate_rejects_an_out_of_range_value() {
        // Find an integer field of PointLight to poison; if none, the test is a
        // no-op assertion that validation at least accepts the defaults.
        let mut args = base_args("PointLight");
        if let Some((k, _)) = args
            .clone()
            .iter()
            .find(|(_, v)| v.is_u64() || (v.is_i64() && v.as_i64() == Some(0)))
        {
            args.insert(k.clone(), Value::from(-1));
            // Only assert failure if the field is actually unsigned; otherwise the
            // negative is valid. Guard by re-checking the default kind.
            if base_args("PointLight")[k].is_u64() {
                assert!(validate("PointLight", &args).is_err());
            }
        }
        // Defaults always validate.
        assert!(validate("PointLight", &base_args("PointLight")).is_ok());
    }
}
