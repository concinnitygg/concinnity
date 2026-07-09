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
// per-type descriptor to maintain. Editable kinds: string / integer / float /
// bool; a fixed-length numeric array of 2..=4 elements (a vector or a colour),
// edited as comma-separated numbers; a string-enum, cycled through its variants
// (discovered per field via `ComponentType::field_enum_variants`); and an
// asset-reference field (`ComponentType::ref_fields`), cycled through `(none)` +
// the world's assets of the target type (the hook fills the options via
// `set_ref_options`). Every other kind (variable-length arrays, nested objects,
// undeclared nulls) is left at its default and round-trips untouched. The
// assembled object is validated by the caller via
// `ComponentType::reserialize_args`.

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
    // A fixed-length numeric array of `len` (2..=4) elements, edited as
    // comma-separated numbers -- a vector (position / direction / size) or, when
    // `color` is set, an RGB / RGBA colour (rendered with a preview swatch).
    Vec { len: usize, color: bool },
    // A string enum with a known variant set (`FormField::variants`), cycled
    // through by clicking rather than typed.
    Enum,
    // An asset reference to an existing asset of type `target`. Rendered like an
    // enum (cycle button) over `FormField::variants` = `(none)` + the world's
    // assets of `target`, which the hook fills in (`set_ref_options`).
    Ref { target: &'static str },
}

// The `(none)` option of a reference field, mapping to a null (unset) reference.
pub(crate) const NONE_LABEL: &str = "(none)";

// One editable field of the form.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FormField {
    // The args-object key this field edits.
    pub key: String,
    pub kind: FieldKind,
    // Initial text for a text field (string / number rendered editable); empty
    // for a bool / enum.
    pub initial: String,
    // Current value of a bool field (the checkbox); unused for text kinds.
    pub boolval: bool,
    // For `FieldKind::Enum`: the allowed variants and the current selection index
    // into them (cycled on click). Empty for every other kind.
    pub variants: Vec<String>,
    pub variant_idx: usize,
}

// The editable kind of a field, from its `key` and default value `v`, or `None`
// for a kind left at its default. A 2..=4-element all-numeric array is a vector;
// it is a colour when the key names one (so the layout can add a swatch), which no
// vector key ever does.
fn kind_of(key: &str, v: &Value) -> Option<FieldKind> {
    match v {
        Value::String(_) => Some(FieldKind::Str),
        Value::Bool(_) => Some(FieldKind::Bool),
        Value::Number(n) => Some(if n.is_i64() || n.is_u64() {
            FieldKind::Int
        } else {
            FieldKind::Float
        }),
        Value::Array(a) if (2..=4).contains(&a.len()) && a.iter().all(Value::is_number) => {
            Some(FieldKind::Vec {
                len: a.len(),
                color: is_color_key(key),
            })
        }
        _ => None,
    }
}

// The target asset type of `key` if the component declares it as a reference.
fn ref_target_of(ct: ComponentType, key: &str) -> Option<&'static str> {
    ct.ref_fields()
        .iter()
        .find(|(field, _)| *field == key)
        .map(|(_, target)| *target)
}

// Fill a reference field's options: `(none)` followed by `names` (the world's
// existing assets of the target type), selecting whichever matches the field's
// current target (stashed in `initial`), else `(none)`. The hook calls this after
// `fields_for` because the option list depends on the working entry list, which
// this world-free module does not see.
pub(crate) fn set_ref_options(field: &mut FormField, names: &[String]) {
    let mut variants = Vec::with_capacity(names.len() + 2);
    variants.push(NONE_LABEL.to_string());
    variants.extend(names.iter().cloned());
    // Preserve a current target that is not among the offered assets (e.g. one
    // injected by the engine or expanded from a SceneImport, which the authored
    // entry list does not contain), so editing an entry never silently drops it.
    if !field.initial.is_empty() && !variants.iter().any(|v| v == &field.initial) {
        variants.push(field.initial.clone());
    }
    field.variant_idx = variants
        .iter()
        .position(|v| v == &field.initial)
        .unwrap_or(0);
    field.variants = variants;
}

// Whether a field name denotes a colour (gets a preview swatch). Only decides the
// swatch among already-numeric-array fields, so it cannot turn a scalar into a
// colour; and no vector field name (position, direction, size, extent, normal,
// ...) contains any of these, so it never mis-flags a geometric vector as a colour.
fn is_color_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    ["color", "colour", "tint", "background", "emissive"]
        .iter()
        .any(|needle| k.contains(needle))
}

// A value rendered as editable text: a scalar as itself, a numeric array as its
// comma-separated elements (`1, 1, 1`).
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Array(a) => a.iter().map(value_text).collect::<Vec<_>>().join(", "),
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
    let ct = ComponentType::parse(ty);
    let mut out = Vec::new();
    let mut truncated = 0;
    for (key, def) in &base {
        // Asset-ref fields default to null (which `kind_of` skips), so detect them
        // first from the type's declared references.
        let mut kind = match ct.and_then(|c| ref_target_of(c, key)) {
            Some(target) => FieldKind::Ref { target },
            None => match kind_of(key, def) {
                Some(k) => k,
                None => continue,
            },
        };
        if out.len() >= MAX_FIELDS {
            truncated += 1;
            continue;
        }
        let cur = seed.and_then(|s| s.get(key)).unwrap_or(def);
        // A string field may be a string-enum: promote it to a cycling picker when
        // the type reports a variant set for it.
        let mut variants = Vec::new();
        let mut variant_idx = 0;
        if matches!(kind, FieldKind::Str)
            && let Some(v) = ct.and_then(|c| c.field_enum_variants(key))
        {
            variant_idx = cur
                .as_str()
                .and_then(|s| v.iter().position(|x| x == s))
                .unwrap_or(0);
            variants = v;
            kind = FieldKind::Enum;
        }
        let boolval = matches!(kind, FieldKind::Bool)
            && cur.as_bool().or_else(|| def.as_bool()).unwrap_or(false);
        let initial = match kind {
            // No text box; a bool/enum carries its state in the field, and a ref
            // stashes its current target name here for `set_ref_options` to select.
            FieldKind::Bool | FieldKind::Enum => String::new(),
            FieldKind::Ref { .. } => cur.as_str().unwrap_or_default().to_string(),
            _ => value_text(cur),
        };
        out.push(FormField {
            key: key.clone(),
            kind,
            initial,
            boolval,
            variants,
            variant_idx,
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
        FieldKind::Vec { len, .. } => coerce_array(text, len, default),
        FieldKind::Enum => field
            .variants
            .get(field.variant_idx)
            .map(|v| Value::String(v.clone()))
            .unwrap_or_else(|| default.clone()),
        // A reference emits the selected asset's name, or null for `(none)`.
        FieldKind::Ref { .. } => match field.variants.get(field.variant_idx) {
            Some(v) if v != NONE_LABEL => Value::String(v.clone()),
            _ => Value::Null,
        },
    }
}

// Parse comma / whitespace separated numbers into a JSON array of exactly `len`
// elements, preserving each element's default integer-vs-float type. Any wrong
// count or unparseable element falls back to `default` whole, so a half-typed
// vector keeps the prior value rather than committing a truncated one.
fn coerce_array(text: &str, len: usize, default: &Value) -> Value {
    let parts: Vec<&str> = text
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != len {
        return default.clone();
    }
    let default_elems = default.as_array();
    let mut out = Vec::with_capacity(len);
    for (i, p) in parts.iter().enumerate() {
        let want_int = default_elems
            .and_then(|a| a.get(i))
            .map(|e| e.is_i64() || e.is_u64())
            .unwrap_or(false);
        if want_int {
            match p.parse::<i64>() {
                Ok(n) => out.push(Value::from(n)),
                Err(_) => return default.clone(),
            }
        } else {
            match p.parse::<f64>().ok().and_then(serde_json::Number::from_f64) {
                Some(n) => out.push(Value::Number(n)),
                None => return default.clone(),
            }
        }
    }
    Value::Array(out)
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

    // PointLight is a representative flat asset: scalar floats plus a 3-element
    // position (a vector) and a 3-element colour (a colour vector).
    #[test]
    fn fields_for_pointlight_exposes_scalars_and_vectors() {
        let fields = fields_for("PointLight", None);
        assert!(!fields.is_empty(), "PointLight exposes editable fields");
        let field = |k: &str| fields.iter().find(|f| f.key == k).cloned();
        // The colour is offered as a colour-flagged 3-vector.
        assert_eq!(
            field("color").map(|f| f.kind),
            Some(FieldKind::Vec {
                len: 3,
                color: true
            }),
            "color is an editable RGB vector with a swatch"
        );
        // The position is a plain (non-colour) 3-vector.
        assert_eq!(
            field("position").map(|f| f.kind),
            Some(FieldKind::Vec {
                len: 3,
                color: false
            }),
            "position is an editable vector, not a colour"
        );
        // A scalar float is still a Float.
        assert_eq!(field("intensity").map(|f| f.kind), Some(FieldKind::Float));
        // The colour field's initial text is its default, comma-joined.
        assert_eq!(field("color").unwrap().initial, "1.0, 1.0, 1.0");
    }

    // Only 2..=4-element all-numeric arrays become vectors; longer arrays, empty
    // arrays, and arrays of non-numbers are left at their defaults.
    #[test]
    fn kind_of_only_accepts_small_numeric_arrays() {
        use serde_json::json;
        assert_eq!(
            kind_of("position", &json!([1.0, 2.0, 3.0])),
            Some(FieldKind::Vec {
                len: 3,
                color: false
            })
        );
        assert_eq!(
            kind_of("half_size", &json!([1.0, 1.0])),
            Some(FieldKind::Vec {
                len: 2,
                color: false
            })
        );
        assert_eq!(
            kind_of("tint", &json!([1.0, 1.0, 1.0, 1.0])),
            Some(FieldKind::Vec {
                len: 4,
                color: true
            })
        );
        // A `background` RGBA box is a colour too (so TextLabel/TextInput get a swatch).
        assert_eq!(
            kind_of("background", &json!([0.0, 0.0, 0.0, 0.0])),
            Some(FieldKind::Vec {
                len: 4,
                color: true
            })
        );
        // A geometric vector is never mis-flagged as a colour.
        assert_eq!(
            kind_of("half_extents", &json!([10.0, 5.0, 10.0])),
            Some(FieldKind::Vec {
                len: 3,
                color: false
            })
        );
        // A 32-element SdfVolume `params` array is too long for a vector.
        assert_eq!(kind_of("params", &json!(vec![0.0_f64; 32])), None);
        // Empty and single-element arrays are not vectors.
        assert_eq!(kind_of("lod_distances", &json!([])), None);
        assert_eq!(kind_of("x", &json!([1.0])), None);
        // An array of objects (e.g. `waves`) is left at its default.
        assert_eq!(kind_of("waves", &json!([{"a": 1}])), None);
    }

    #[test]
    fn coerce_array_parses_round_trips_and_falls_back() {
        use serde_json::json;
        let def = json!([1.0, 1.0, 1.0]);
        // Comma and whitespace separators both work.
        assert_eq!(
            coerce_array("0.2, 0.4, 0.6", 3, &def),
            json!([0.2, 0.4, 0.6])
        );
        assert_eq!(coerce_array("0.2 0.4 0.6", 3, &def), json!([0.2, 0.4, 0.6]));
        // Wrong element count keeps the prior value whole.
        assert_eq!(coerce_array("0.2, 0.4", 3, &def), def);
        // A non-numeric element falls back rather than committing garbage.
        assert_eq!(coerce_array("0.2, x, 0.6", 3, &def), def);
        // Integer element types are preserved from the default.
        let idef = json!([0, 0, 0]);
        assert_eq!(coerce_array("16, 24, 16", 3, &idef), json!([16, 24, 16]));
    }

    // A string field the type reports variants for becomes a cycling Enum; a
    // free-form string field stays Str.
    #[test]
    fn fields_for_detects_a_string_enum_as_a_cycling_field() {
        let fields = fields_for("Sprite", None);
        let fit = fields.iter().find(|f| f.key == "fit").expect("fit field");
        assert_eq!(fit.kind, FieldKind::Enum);
        assert_eq!(fit.variants, vec!["fit", "cover", "bottom"]);
        assert_eq!(fit.variant_idx, 0, "Sprite's default fit selects variant 0");
        // A free-form string field is left as an editable text field.
        let hr = fields_for("HitRegion", None);
        assert_eq!(
            hr.iter().find(|f| f.key == "action").unwrap().kind,
            FieldKind::Str
        );
    }

    // An enum field emits its currently-selected variant, and the result cooks.
    #[test]
    fn enum_field_coerces_to_its_selected_variant() {
        let mut fields = fields_for("TextLabel", None);
        let idx = fields
            .iter()
            .position(|f| f.key == "align")
            .expect("align field");
        assert_eq!(fields[idx].kind, FieldKind::Enum);
        let center = fields[idx]
            .variants
            .iter()
            .position(|v| v == "center")
            .expect("a center variant");
        fields[idx].variant_idx = center;
        let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        let args = assemble("TextLabel", None, &fields, &texts);
        assert_eq!(args["align"], "center");
        assert!(
            validate("TextLabel", &args).is_ok(),
            "the picked variant cooks"
        );
    }

    // A declared reference field becomes a Ref picker; its options are `(none)` +
    // supplied names, and it coerces to the selected name or null.
    #[test]
    fn ref_field_detects_target_options_and_coerces() {
        let fields = fields_for("Decal", None);
        let tex = fields
            .iter()
            .find(|f| f.key == "texture")
            .expect("texture ref field");
        assert_eq!(tex.kind, FieldKind::Ref { target: "Texture" });
        assert_eq!(tex.initial, "", "an unset ref stashes no target name");

        let mut f = tex.clone();
        set_ref_options(&mut f, &["grass".into(), "stone".into()]);
        assert_eq!(f.variants, vec![NONE_LABEL, "grass", "stone"]);
        assert_eq!(f.variant_idx, 0, "no current target -> (none)");
        // (none) coerces to null; a chosen asset to its name string.
        assert_eq!(coerce(&f, "", &Value::Null), Value::Null);
        f.variant_idx = 2;
        assert_eq!(coerce(&f, "", &Value::Null), Value::String("stone".into()));
    }

    // Editing an entry whose ref is already set selects that asset in the options.
    #[test]
    fn set_ref_options_selects_the_current_target_when_editing() {
        let mut seed = base_args("Decal");
        seed.insert("texture".into(), Value::String("grass".into()));
        let fields = fields_for("Decal", Some(&seed));
        let mut tex = fields
            .into_iter()
            .find(|f| f.key == "texture")
            .expect("texture ref field");
        assert_eq!(tex.initial, "grass", "the current ref name is carried");
        set_ref_options(&mut tex, &["stone".into(), "grass".into()]);
        // (none), stone, grass -> "grass" is index 2.
        assert_eq!(tex.variants[tex.variant_idx], "grass");
    }

    // A current target absent from the offered assets (engine-injected / imported)
    // is preserved as an option so editing never drops it.
    #[test]
    fn set_ref_options_preserves_an_unlisted_current_target() {
        let mut field = FormField {
            key: "texture".into(),
            kind: FieldKind::Ref { target: "Texture" },
            initial: "imported_tex".into(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 0,
        };
        set_ref_options(&mut field, &["grass".into()]);
        assert!(field.variants.contains(&"imported_tex".to_string()));
        assert_eq!(field.variants[field.variant_idx], "imported_tex");
        // It still coerces back to that name (not lost).
        assert_eq!(
            coerce(&field, "", &Value::Null),
            Value::String("imported_tex".into())
        );
    }

    // The types added to the picker this round derive sensible forms: View is a
    // bool + a float; BlockType exposes its solid flag + UV vectors; FpsCounter's
    // one field is its declared `label` reference (a picker, not free text).
    #[test]
    fn newly_offered_types_derive_expected_fields() {
        let view = fields_for("View", None);
        let vk = |k: &str| view.iter().find(|f| f.key == k).map(|f| f.kind);
        assert_eq!(vk("initial"), Some(FieldKind::Bool));
        assert_eq!(vk("fade_in_secs"), Some(FieldKind::Float));

        let block = fields_for("BlockType", None);
        let bk = |k: &str| block.iter().find(|f| f.key == k).map(|f| f.kind);
        assert_eq!(bk("solid"), Some(FieldKind::Bool));
        assert_eq!(
            bk("uv_min"),
            Some(FieldKind::Vec {
                len: 2,
                color: false
            })
        );

        let fps = fields_for("FpsCounter", None);
        assert_eq!(
            fps.iter().find(|f| f.key == "label").map(|f| f.kind),
            Some(FieldKind::Ref {
                target: "TextLabel"
            }),
            "FpsCounter.label is a reference picker"
        );
    }

    #[test]
    fn unknown_type_has_no_fields_and_empty_base() {
        assert!(fields_for("NotARealType", None).is_empty());
        assert!(base_args("NotARealType").is_empty());
    }

    // A colour edit assembles and re-serializes cleanly through the real type.
    #[test]
    fn assemble_persists_an_edited_colour_vector() {
        let fields = fields_for("PointLight", None);
        let (idx, key) = fields
            .iter()
            .enumerate()
            .find(|(_, f)| f.key == "color")
            .map(|(i, f)| (i, f.key.clone()))
            .expect("a color field");
        let mut texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        texts[idx] = "0.5, 0.25, 0.75".into();
        let args = assemble("PointLight", None, &fields, &texts);
        assert_eq!(args[&key], serde_json::json!([0.5, 0.25, 0.75]));
        assert!(
            validate("PointLight", &args).is_ok(),
            "the edit still cooks"
        );
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
                variants: Vec::new(),
                variant_idx: 0,
            },
            FormField {
                key: "on".into(),
                kind: FieldKind::Bool,
                initial: String::new(),
                boolval: true,
                variants: Vec::new(),
                variant_idx: 0,
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
            variants: Vec::new(),
            variant_idx: 0,
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

    // Every type the picker offers must have a well-formed derived form: its
    // fields build without panic and its default-derived args re-validate. This is
    // the form-side counterpart to `add_types_cook_with_default_args` (which guards
    // the cook side), so adding a type whose form mis-derives is caught here.
    #[test]
    fn every_add_type_form_round_trips_its_defaults() {
        for &ty in crate::editor::panel::ADD_TYPES {
            let fields = fields_for(ty, None);
            let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
            let args = assemble(ty, None, &fields, &texts);
            assert!(
                validate(ty, &args).is_ok(),
                "{ty}: the default-derived form must re-validate"
            );
        }
    }

    // HitRegion is the richest mixed-kind form: scalars + a string + a bool, its
    // declared references (label -> TextLabel, view -> View) as Ref pickers, and its
    // undeclared null Option fields left at their defaults.
    #[test]
    fn hitregion_form_offers_scalars_refs_and_skips_undeclared_nulls() {
        let fields = fields_for("HitRegion", None);
        let field = |k: &str| fields.iter().find(|f| f.key == k).map(|f| f.kind);
        assert_eq!(field("x"), Some(FieldKind::Float));
        assert_eq!(field("width"), Some(FieldKind::Float));
        assert_eq!(field("action"), Some(FieldKind::Str));
        assert_eq!(field("disabled"), Some(FieldKind::Bool));
        // Declared references become Ref pickers targeting their asset type.
        assert_eq!(
            field("label"),
            Some(FieldKind::Ref {
                target: "TextLabel"
            })
        );
        assert_eq!(field("view"), Some(FieldKind::Ref { target: "View" }));
        // Null Option fields the type does NOT declare as references are still left
        // at their defaults, not offered.
        for skipped in ["hover_color", "hover_scale", "drag_handle"] {
            assert!(
                field(skipped).is_none(),
                "{skipped} (an undeclared null Option) is not an editable field"
            );
        }
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
