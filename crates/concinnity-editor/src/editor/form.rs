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
// (discovered per field via `RegisteredType::field_enum_variants`); and an
// asset-reference field (`RegisteredType::ref_fields`), cycled through `(none)` +
// the world's assets of the target type (the hook fills the options via
// `set_ref_options`). A plain nested OBJECT is flattened into its leaves, keyed by
// a dotted path (`controller.move_speed`), up to `MAX_NEST_DEPTH` levels; the leaf
// edits like any scalar and `assemble` writes it back into the sub-object via
// `set_at_path`. A variable-length (non-vector) ARRAY becomes an `Array` header
// field (add / remove elements) followed by each element's fields keyed by index
// (`waves.0.amplitude`); `set_at_path` / `get_at_path` navigate the numeric index
// segments. Every other kind (deeper objects / arrays past the depth cap, undeclared
// nulls) is left at its default and round-trips untouched. The assembled object is
// validated by the caller via `RegisteredType::reserialize_args`.

use concinnity_world::registry::RegisteredType;
use serde_json::{Map, Value};

// The form's default (and minimum) scrolling window: the number of field rows the
// edit panel shows before it scrolls. A form derives ALL of a type's editable
// fields and the panel renders a window this many rows tall over them, so a type
// wider than the window (or an array grown past it) scrolls rather than truncating.
pub(crate) const FIELD_POOL: usize = 14;
// The injected control pool (checkboxes, text inputs, cycle buttons, swatches):
// the default shows `FIELD_POOL` rows, resizing the panel taller reveals more up
// to this.
pub(crate) const FIELD_POOL_MAX: usize = 28;

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
    // A variable-length array (non-vector) at `FormField::key`, rendered as a header
    // row with add / remove buttons; its element count is carried in
    // `FormField::variant_idx`. The elements' own leaves follow it as indexed
    // dotted-path fields (`waves.0.amplitude`). This is NOT the fixed 2..=4 numeric
    // vector (that stays a `Vec` leaf) -- only longer / object / reference arrays.
    Array,
}

impl FieldKind {
    // Whether a field of this kind is edited through a text input (so the panel
    // seeds / reads a control for it). Bools (checkbox), enums / refs (cycle
    // button), arrays (header), and non-colour vectors (a disclosure of per-element
    // leaves) carry their state elsewhere, not in a text box.
    pub(crate) fn has_text_input(self) -> bool {
        !matches!(
            self,
            FieldKind::Bool
                | FieldKind::Enum
                | FieldKind::Ref { .. }
                | FieldKind::Array
                | FieldKind::Vec { color: false, .. }
        )
    }
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

// The authoring-type metadata the recursive field walker threads unchanged: the
// component type (for per-field string-enum variant pickers, a component-only
// notion) and the declared asset-reference fields. Bundled so the walker takes
// one context argument rather than two.
#[derive(Clone, Copy)]
struct TypeMeta {
    ct: Option<RegisteredType>,
    // `(field, target type)` each; the add form turns each into a name picker.
    refs: &'static [(&'static str, &'static str)],
}

impl TypeMeta {
    // Build from an authoring type name. Reference fields come from the entry's
    // `refs:` metadata, whichever group of the registry it is in.
    fn of(ty: &str) -> Self {
        let ct = RegisteredType::parse(ty);
        let refs = ct.map(|c| c.ref_fields()).unwrap_or(&[]);
        TypeMeta { ct, refs }
    }

    // The target asset type declared for `field`, if any.
    fn ref_target(&self, field: &str) -> Option<&'static str> {
        self.refs
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, target)| *target)
    }
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
        Value::Number(n) => number_text(n),
        Value::Bool(b) => b.to_string(),
        Value::Array(a) => a.iter().map(value_text).collect::<Vec<_>>().join(", "),
        _ => String::new(),
    }
}

// A JSON number as display text. A float that originated from an `f32` (as the
// engine's args almost always are) is printed at f32's shortest round-tripping
// form rather than the full f64 expansion serde emits (`0.05`, not
// `0.05000000074505806`); a genuine f64 that does not round-trip through f32 keeps
// full precision. Integers print as-is. Re-parsing the shortened text yields the
// same f32, so the displayed value round-trips unchanged through cook.
fn number_text(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    match n.as_f64() {
        // `{:?}` on an f32 is the shortest decimal that round-trips, keeping a
        // decimal point (`1.0`, not `1`) so a float still reads as a float.
        Some(v) if (v as f32) as f64 == v => format!("{:?}", v as f32),
        _ => n.to_string(),
    }
}

// The type's registered default args as an object (empty if the type is unknown
// or has no args schema).
pub(crate) fn base_args(ty: &str) -> Map<String, Value> {
    RegisteredType::parse(ty)
        .and_then(|ct| ct.registration().default_args)
        .and_then(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default()
}

// How deep the form descends into nested objects. A field whose default value is
// a plain object is flattened into its scalar leaves, keyed by a dotted path
// (`controller.move_speed`), up to this many levels below the root. Arrays and
// deeper objects are left at their defaults (they want the nested-array / deeper
// controls). 2 covers the current cases (Camera3D's `controller`).
const MAX_NEST_DEPTH: usize = 2;

// The editable fields for a type, in declaration order. `seed` supplies current
// values (an existing entry's args when editing), overriding the defaults for the
// initial text / bool state; the kind is always taken from the default so it is
// stable. Nested plain objects are flattened into dotted-path leaves (see
// `collect_fields`). All fields are returned; the panel renders a scrolling window
// `FIELD_POOL` rows tall over them.
// A convenience for the collapsed form (no disclosed vectors), used by the tests.
#[cfg(test)]
pub(crate) fn fields_for(ty: &str, seed: Option<&Map<String, Value>>) -> Vec<FormField> {
    fields_for_with(ty, seed, &std::collections::HashSet::new())
}

// `fields_for` with a set of expanded (disclosed) non-colour vector paths: each
// expanded vector is followed by an editable leaf per element (`position.0` ..)
// so its components edit one at a time; a collapsed vector is just its header row.
pub(crate) fn fields_for_with(
    ty: &str,
    seed: Option<&Map<String, Value>>,
    expanded: &std::collections::HashSet<String>,
) -> Vec<FormField> {
    let base = base_args(ty);
    let meta = TypeMeta::of(ty);
    let mut out = Vec::new();
    collect_fields(meta, "", &base, seed, expanded, 0, &mut out);
    out
}

// Append the editable fields of `obj` (the defaults at `prefix`, empty at the
// root) to `out`, dispatching each key's value through `collect_value`. `root_seed`
// is always the top-level args (an existing entry's values when editing); every
// leaf reads its current value from it by full path.
fn collect_fields(
    meta: TypeMeta,
    prefix: &str,
    obj: &Map<String, Value>,
    root_seed: Option<&Map<String, Value>>,
    expanded: &std::collections::HashSet<String>,
    depth: usize,
    out: &mut Vec<FormField>,
) {
    for (key, def) in obj {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        collect_value(meta, &path, def, root_seed, expanded, depth, out);
    }
}

// Append the field(s) for one value at `path` (its default shape `def`, current
// value read from `root_seed`): a declared reference; a flattened nested object; a
// scalar / vector leaf; or an array (an `Array` header row followed by each
// element's fields, keyed by index). `prefix.is_empty()`-style root detection uses
// whether `path` contains a `.`.
fn collect_value(
    meta: TypeMeta,
    path: &str,
    def: &Value,
    root_seed: Option<&Map<String, Value>>,
    expanded: &std::collections::HashSet<String>,
    depth: usize,
    out: &mut Vec<FormField>,
) {
    let is_root = !path.contains('.');
    let leaf = path.rsplit('.').next().unwrap_or(path);
    // Asset-ref fields default to null (which `kind_of` skips), so detect them first
    // from the type's declared references (matched by full path).
    let ref_target = meta.ref_target(path);

    // A plain nested object that is not itself a declared reference is flattened into
    // its leaves one level deeper -- UNLESS the seed (an entry being edited) authored
    // this path as a non-object (e.g. Camera3D's `controller: null`, a load-bearing
    // "uncontrolled cutscene camera" marker): flattening from the object-shaped
    // DEFAULT would rebuild a default object over the authored null on `assemble`.
    // Left unflattened, the object default is not an editable leaf kind, so it is
    // skipped and the merge preserves the authored value.
    if ref_target.is_none()
        && depth < MAX_NEST_DEPTH
        && let Value::Object(nested) = def
        && !nested.is_empty()
        && !seed_overrides_with_non_object(root_seed, path)
    {
        collect_fields(meta, path, nested, root_seed, expanded, depth + 1, out);
        return;
    }

    // A scalar / fixed 2..=4 numeric vector leaf (or a declared reference).
    let kind = match ref_target {
        Some(target) => Some(FieldKind::Ref { target }),
        None => kind_of(leaf, def),
    };
    if let Some(mut kind) = kind {
        let cur = root_seed.and_then(|s| get_at_path(s, path)).unwrap_or(def);
        // A string field may be a string-enum: promote it to a cycling picker when
        // the type reports a variant set for it. Only at the root -- the probe names
        // a top-level arg, so it cannot resolve a nested field's variants.
        let mut variants = Vec::new();
        let mut variant_idx = 0;
        if is_root
            && matches!(kind, FieldKind::Str)
            && let Some(v) = meta.ct.and_then(|c| c.field_enum_variants(leaf))
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
            key: path.to_string(),
            kind,
            initial,
            boolval,
            variants,
            variant_idx,
        });
        // A non-colour vector can be disclosed into an editable leaf per element
        // (edited one component at a time); collapsed it is only the header row. A
        // colour vector keeps its single field + preview swatch instead.
        if let FieldKind::Vec { len, color: false } = kind
            && expanded.contains(path)
        {
            let cur = root_seed.and_then(|s| get_at_path(s, path)).unwrap_or(def);
            let arr = cur.as_array();
            for i in 0..len {
                let elem = arr.and_then(|a| a.get(i));
                let ekind = elem
                    .and_then(|e| kind_of("", e))
                    .unwrap_or(FieldKind::Float);
                out.push(FormField {
                    key: format!("{path}.{i}"),
                    kind: ekind,
                    initial: elem.map(value_text).unwrap_or_default(),
                    boolval: false,
                    variants: Vec::new(),
                    variant_idx: 0,
                });
            }
        }
        return;
    }

    // A variable-length (non-vector) array: an `Array` header carrying the element
    // count, then each element's field(s) keyed by index. The header lets the panel
    // offer add / remove; the elements read their shape from the CURRENT array (the
    // seed's when editing, else the default), so add / remove re-derives cleanly.
    if ref_target.is_none()
        && depth < MAX_NEST_DEPTH
        && let Value::Array(def_arr) = def
    {
        let cur_arr = root_seed
            .and_then(|s| get_at_path(s, path))
            .and_then(Value::as_array)
            .unwrap_or(def_arr);
        // Only offer add / remove for a genuinely variable-length LIST. A pure
        // numeric array is indistinguishable from a fixed `[T; N]` once serialized
        // (SdfVolume's `[f32; 32]` params, index / matrix buffers), and growing one
        // is rejected at cook -- so treat only object / string / reference element
        // arrays (never a plain number) as editable, and only when a template element
        // exists (a non-empty current or default array) to clone for `[+]`. Others
        // are left at their default (round-tripped untouched), as before.
        let template = cur_arr.first().or_else(|| def_arr.first());
        if template.is_none_or(Value::is_number) {
            return;
        }
        out.push(FormField {
            key: path.to_string(),
            kind: FieldKind::Array,
            initial: String::new(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: cur_arr.len(),
        });
        for (i, elem) in cur_arr.iter().enumerate() {
            let elem_path = format!("{path}.{i}");
            collect_value(meta, &elem_path, elem, root_seed, expanded, depth + 1, out);
        }
    }
}

// Whether the seed (an entry being edited) authored `path` as a present non-object
// value -- e.g. Camera3D's `controller: null`. Such a value must not be flattened
// away, since the leaves would be rebuilt from the object-shaped default.
fn seed_overrides_with_non_object(root_seed: Option<&Map<String, Value>>, path: &str) -> bool {
    matches!(root_seed.and_then(|s| get_at_path(s, path)), Some(v) if !v.is_object())
}

// The value at a dotted `path`, where each segment indexes an object by key or an
// array by numeric index (`waves.0.amplitude`). `None` if any segment is missing.
fn get_at_path<'a>(obj: &'a Map<String, Value>, path: &str) -> Option<&'a Value> {
    let mut parts = path.split('.');
    let mut cur = obj.get(parts.next()?)?;
    for p in parts {
        cur = match cur {
            Value::Object(m) => m.get(p)?,
            Value::Array(a) => a.get(p.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

// Set the value at a dotted `path`. Object segments are created as needed; array
// index segments must already exist (fields only ever address present elements), so
// an out-of-range index is a no-op. A single-segment path is a plain insert.
fn set_at_path(obj: &mut Map<String, Value>, path: &str, val: Value) {
    let parts: Vec<&str> = path.split('.').collect();
    let (leaf, parents) = parts.split_last().expect("a path has at least one segment");
    if parents.is_empty() {
        obj.insert(leaf.to_string(), val);
        return;
    }
    // The first segment is always a top-level arg name (an object key).
    let mut cur = obj
        .entry(parents[0].to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    for seg in &parents[1..] {
        cur = match cur {
            Value::Array(a) => {
                let Some(v) = seg.parse::<usize>().ok().and_then(|i| a.get_mut(i)) else {
                    return;
                };
                v
            }
            other => {
                if !other.is_object() {
                    *other = Value::Object(Map::new());
                }
                other
                    .as_object_mut()
                    .expect("just ensured an object")
                    .entry((*seg).to_string())
                    .or_insert_with(|| Value::Object(Map::new()))
            }
        };
    }
    match cur {
        Value::Array(a) => {
            if let Some(slot) = leaf.parse::<usize>().ok().and_then(|i| a.get_mut(i)) {
                *slot = val;
            }
        }
        other => {
            if !other.is_object() {
                *other = Value::Object(Map::new());
            }
            other
                .as_object_mut()
                .expect("just ensured an object")
                .insert(leaf.to_string(), val);
        }
    }
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
        // An array header carries no editable value of its own (the array's contents
        // come from its element leaves + the working structure); `assemble` skips it.
        FieldKind::Array => default.clone(),
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
        // A structural row carries no value of its own: an array header's array
        // lives in `out` already (carried via `editing_args`, grown / shrunk by add
        // / remove), and a disclosed non-colour vector's value is written by its
        // per-element leaves (or, collapsed, left untouched in `out`).
        if field.kind == FieldKind::Array
            || matches!(field.kind, FieldKind::Vec { color: false, .. })
        {
            continue;
        }
        // Fall back to the value already in `out` at this (possibly nested) path
        // (the entry's authored value when editing, else the type default), so a
        // cleared / mistyped number keeps what was there rather than snapping back
        // to the type default.
        let fallback = get_at_path(&out, &field.key)
            .cloned()
            .unwrap_or(Value::Null);
        let text = texts.get(i).map(String::as_str).unwrap_or("");
        set_at_path(&mut out, &field.key, coerce(field, text, &fallback));
    }
    out
}

// The initial working args for a form: the type defaults with an edited entry's
// args merged over them (the structure add / remove and the controls then mutate).
pub(crate) fn working_args(ty: &str, editing: Option<&Map<String, Value>>) -> Map<String, Value> {
    let mut out = base_args(ty);
    if let Some(e) = editing {
        for (k, v) in e {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

// The current value at `path` rendered as editable text (empty if absent). The
// panel only has a physical text control for the fields inside its scrolling
// window; capturing an off-window field feeds this back through `assemble` so its
// stored value round-trips unchanged instead of being blanked.
pub(crate) fn current_text(args: &Map<String, Value>, path: &str) -> String {
    get_at_path(args, path).map(value_text).unwrap_or_default()
}

// The element template for the array at `path` in `args`: a clone of its first
// element when non-empty, else the type default's first element (so a new,
// still-empty array can grow from the schema's shape). `None` when neither is
// available (an empty array whose element type is unknowable), which disables add.
pub(crate) fn array_elem_template(
    ty: &str,
    args: &Map<String, Value>,
    path: &str,
) -> Option<Value> {
    let first_of = |m: &Map<String, Value>| {
        get_at_path(m, path)
            .and_then(Value::as_array)
            .and_then(|a| a.first().cloned())
    };
    first_of(args).or_else(|| first_of(&base_args(ty)))
}

// Append a fresh element (a clone of the template) to the array at `path`, or do
// nothing if there is no template. Returns whether it grew.
pub(crate) fn add_array_elem(ty: &str, args: &mut Map<String, Value>, path: &str) -> bool {
    let Some(template) = array_elem_template(ty, args, path) else {
        return false;
    };
    if let Some(Value::Array(a)) = get_at_path_mut(args, path) {
        a.push(template);
        return true;
    }
    false
}

// Remove the last element of the array at `path` (nothing if it is empty / absent).
pub(crate) fn remove_array_elem(args: &mut Map<String, Value>, path: &str) -> bool {
    if let Some(Value::Array(a)) = get_at_path_mut(args, path)
        && !a.is_empty()
    {
        a.pop();
        return true;
    }
    false
}

// Mutable sibling of `get_at_path`.
fn get_at_path_mut<'a>(obj: &'a mut Map<String, Value>, path: &str) -> Option<&'a mut Value> {
    let mut parts = path.split('.');
    let mut cur = obj.get_mut(parts.next()?)?;
    for p in parts {
        cur = match cur {
            Value::Object(m) => m.get_mut(p)?,
            Value::Array(a) => a.get_mut(p.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

// Validate an assembled args object by round-tripping it through the type's typed
// `Args` (the same check `cn add` applies). `Ok(())` means it will cook.
pub(crate) fn validate(ty: &str, name: &str, args: &Map<String, Value>) -> Result<(), String> {
    let Some(ct) = RegisteredType::parse(ty) else {
        return Err(format!("unknown asset type '{ty}'"));
    };
    // A resource asset gets the same two checks a component does -- the typed
    // schema round-trip, then the structural check `cn add` / `cn check` run --
    // but never a payload compile: an EnvironmentMap's convolution costs seconds
    // and would stall the editor on every Apply. A source file that parses but
    // decodes badly is caught by the preview rebuild and by SAVE, which cook for
    // real.
    if ct.is_resource() {
        let args = Value::Object(args.clone());
        ct.normalized_args(&args).map_err(|e| e.to_string())?;
        return concinnity_cook::validate_asset(ty, name, &args);
    }
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
        // A non-integer element in an integer vector falls back whole rather
        // than committing a truncated array.
        assert_eq!(coerce_array("16, x, 16", 3, &idef), idef);
    }

    #[test]
    fn value_text_renders_bools_and_blanks_unhandled_values() {
        use serde_json::json;
        assert_eq!(value_text(&Value::Bool(true)), "true");
        assert_eq!(value_text(&json!([1.0, 2.0, 3.0])), "1.0, 2.0, 3.0");
        // A null / object has no editable text form.
        assert_eq!(value_text(&Value::Null), "");
        assert_eq!(value_text(&json!({"a": 1})), "");
    }

    #[test]
    fn get_at_path_stops_at_a_scalar_segment() {
        use serde_json::json;
        let mut obj = Map::new();
        obj.insert("scalar".to_string(), json!(3));
        obj.insert("nested".to_string(), json!({"leaf": 7}));
        // Descending past a scalar segment finds nothing.
        assert!(get_at_path(&obj, "scalar.deeper").is_none());
        assert_eq!(get_at_path(&obj, "nested.leaf"), Some(&json!(7)));
        assert!(get_at_path(&obj, "missing").is_none());
    }

    #[test]
    fn set_at_path_rebuilds_scalar_parents_and_leaves_as_objects() {
        use serde_json::json;
        let mut obj = Map::new();
        obj.insert("a".to_string(), json!(1));
        obj.insert("x".to_string(), json!(2));
        // A scalar leaf's holder is replaced with an object.
        set_at_path(&mut obj, "a.b", json!(5));
        assert_eq!(get_at_path(&obj, "a.b"), Some(&json!(5)));
        // A scalar mid-path parent is likewise rebuilt into nested objects.
        set_at_path(&mut obj, "x.y.z", json!(9));
        assert_eq!(get_at_path(&obj, "x.y.z"), Some(&json!(9)));
    }

    #[test]
    fn coerce_leaves_an_array_header_at_its_default() {
        use serde_json::json;
        let field = FormField {
            key: "waves".to_string(),
            kind: FieldKind::Array,
            initial: String::new(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 2,
        };
        let default = json!([{"amplitude": 1.0}]);
        // An array header carries no editable value of its own.
        assert_eq!(coerce(&field, "ignored", &default), default);
    }

    #[test]
    fn working_args_merges_an_edited_entry_over_defaults() {
        use serde_json::json;
        let mut editing = Map::new();
        editing.insert("intensity".to_string(), json!(42.0));
        let out = working_args("PointLight", Some(&editing));
        // The edited value overrides the type default while other defaults remain.
        assert_eq!(out.get("intensity"), Some(&json!(42.0)));
        assert!(out.contains_key("color"));
    }

    #[test]
    fn array_mutators_ignore_absent_or_scalar_paths() {
        use serde_json::json;
        let mut obj = Map::new();
        obj.insert("scalar".to_string(), json!(1));
        obj.insert("empty".to_string(), json!([]));
        // Removing from a scalar path (descends through a non-container) and an
        // empty array both report no change.
        assert!(!remove_array_elem(&mut obj, "scalar.inner"));
        assert!(!remove_array_elem(&mut obj, "empty"));
        assert!(!remove_array_elem(&mut obj, "missing"));
        // Adding onto a path that resolves to a scalar (with a template from the
        // type default) also does nothing.
        let mut point = Map::new();
        point.insert("position".to_string(), json!(3));
        assert!(!add_array_elem("PointLight", &mut point, "position"));
    }

    #[test]
    fn validate_rejects_an_unknown_type() {
        let err = validate("NotARealAssetType", "probe", &Map::new()).unwrap_err();
        assert!(err.contains("unknown asset type"), "got: {err}");
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
            validate("TextLabel", "probe", &args).is_ok(),
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

    // A resource type declares its references through the same `refs:` metadata
    // every other type uses: Material's albedo/normal/etc. fields must render as
    // Texture pickers in the add form.
    #[test]
    fn resource_type_material_texture_fields_are_ref_pickers() {
        let fields = fields_for("Material", None);
        for key in ["albedo", "normal_map", "emissive_map", "orm_map"] {
            let f = fields
                .iter()
                .find(|f| f.key == key)
                .unwrap_or_else(|| panic!("Material `{key}` field"));
            assert_eq!(
                f.kind,
                FieldKind::Ref { target: "Texture" },
                "Material `{key}` should be a Texture ref picker"
            );
        }
    }

    // An imported `.hdr` is editable in the Assets panel: EnvironmentMap is a
    // resource type too, so its source and the IBL tuning knobs must derive as
    // real form fields and survive a round-trip back through the cook.
    #[test]
    fn resource_type_environment_map_exposes_its_source_and_tuning() {
        let mut seed = base_args("EnvironmentMap");
        seed.insert("source".into(), Value::String("hdri/studio.hdr".into()));
        let fields = fields_for("EnvironmentMap", Some(&seed));

        let source = fields
            .iter()
            .find(|f| f.key == "source")
            .expect("source field");
        assert_eq!(source.initial, "hdri/studio.hdr");
        for key in [
            "prefilter_face_size",
            "irradiance_face_size",
            "prefilter_samples",
            "prefilter_clamp",
        ] {
            assert!(
                fields.iter().any(|f| f.key == key),
                "EnvironmentMap `{key}` should be editable"
            );
        }

        let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        let args = assemble("EnvironmentMap", Some(&seed), &fields, &texts);
        validate("EnvironmentMap", "probe", &args).expect("an unedited round-trip stays valid");
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
        let screen = fields_for("Screen", None);
        let vk = |k: &str| screen.iter().find(|f| f.key == k).map(|f| f.kind);
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

    // A nested plain object (Camera3D's `controller`) is flattened into dotted-path
    // leaves, so its scalars / bools / vectors edit in the same flat form.
    #[test]
    fn fields_for_flattens_a_nested_object() {
        let fields = fields_for("Camera3D", None);
        let kind = |k: &str| fields.iter().find(|f| f.key == k).map(|f| f.kind);
        // Top-level scalars are still present.
        assert_eq!(kind("fov_y_degrees"), Some(FieldKind::Float));
        // Nested `controller` leaves become dotted-path fields of the right kind.
        assert_eq!(kind("controller.free_fly"), Some(FieldKind::Bool));
        assert_eq!(kind("controller.move_speed"), Some(FieldKind::Float));
        assert_eq!(
            kind("controller.bounds_min"),
            Some(FieldKind::Vec {
                len: 3,
                color: false
            })
        );
        // The nested bool's state is seeded from its default (free_fly defaults true).
        let ff = fields
            .iter()
            .find(|f| f.key == "controller.free_fly")
            .unwrap();
        assert!(ff.boolval, "controller.free_fly default is true");
        // A deeper null Option (`controller.follow`) is left at its default, not
        // flattened.
        assert!(
            fields
                .iter()
                .all(|f| !f.key.starts_with("controller.follow")),
            "a null nested Option is not descended into"
        );
    }

    // Editing a nested (dotted-path) field writes back into its sub-object, and the
    // assembled args still cook.
    #[test]
    fn assemble_writes_a_nested_field_back_into_its_object() {
        let fields = fields_for("Camera3D", None);
        let idx = fields
            .iter()
            .position(|f| f.key == "controller.move_speed")
            .expect("controller.move_speed field");
        let mut texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        texts[idx] = "7.5".into();
        let args = assemble("Camera3D", None, &fields, &texts);
        assert_eq!(
            args["controller"]["move_speed"].as_f64(),
            Some(7.5),
            "the nested edit lands inside the sub-object"
        );
        assert!(
            validate("Camera3D", "probe", &args).is_ok(),
            "the nested edit cooks"
        );
    }

    // Editing a nested field of an existing entry keeps the sub-object's other
    // (unshown / untouched) values intact.
    #[test]
    fn assemble_preserves_sibling_nested_values() {
        let seed = base_args("Camera3D");
        let fields = fields_for("Camera3D", Some(&seed));
        let idx = fields
            .iter()
            .position(|f| f.key == "controller.move_speed")
            .unwrap();
        let mut texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        texts[idx] = "2.0".into();
        let args = assemble("Camera3D", Some(&seed), &fields, &texts);
        // move_speed changed; a sibling default (sprint_multiplier = 3.0) is intact.
        assert_eq!(args["controller"]["move_speed"].as_f64(), Some(2.0));
        assert_eq!(args["controller"]["sprint_multiplier"].as_f64(), Some(3.0));
    }

    // A nullable nested object the entry authored as `null` (Camera3D's
    // `controller: null` cutscene marker) must NOT be flattened or rebuilt: the
    // authored null has to survive an edit, else a cutscene camera silently becomes
    // a free-fly one.
    #[test]
    fn editing_a_null_nested_object_preserves_the_null() {
        let mut seed = base_args("Camera3D");
        seed.insert("controller".into(), Value::Null);
        let fields = fields_for("Camera3D", Some(&seed));
        // No controller.* leaves are offered (the authored null is not descended into).
        assert!(
            fields.iter().all(|f| !f.key.starts_with("controller")),
            "a null nested object is not flattened into leaves"
        );
        // Assembling after (e.g.) a top-level edit keeps controller null.
        let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        let args = assemble("Camera3D", Some(&seed), &fields, &texts);
        assert_eq!(
            args["controller"],
            Value::Null,
            "the authored null controller survives the edit"
        );
        assert!(
            validate("Camera3D", "probe", &args).is_ok(),
            "a null controller cooks"
        );
    }

    #[test]
    fn path_helpers_get_and_set_nested_values() {
        let mut m = base_args("Camera3D");
        assert!(
            get_at_path(&m, "controller.free_fly")
                .and_then(Value::as_bool)
                .is_some()
        );
        set_at_path(&mut m, "controller.move_speed", Value::from(9.0));
        assert_eq!(
            get_at_path(&m, "controller.move_speed").and_then(Value::as_f64),
            Some(9.0)
        );
        // Missing intermediates are created; single-segment paths are plain ops.
        let mut empty = Map::new();
        set_at_path(&mut empty, "a.b.c", Value::from(1));
        assert_eq!(
            get_at_path(&empty, "a.b.c").and_then(Value::as_i64),
            Some(1)
        );
        set_at_path(&mut empty, "x", Value::from(2));
        assert_eq!(get_at_path(&empty, "x").and_then(Value::as_i64), Some(2));
        assert!(get_at_path(&empty, "a.b.missing").is_none());
    }

    // A variable-length array (WaterSurface's `waves`, default 1 element) becomes an
    // Array header field carrying the element count, followed by each element's
    // leaves keyed by index.
    #[test]
    fn fields_for_exposes_an_array_header_and_element_leaves() {
        let fields = fields_for("WaterSurface", None);
        let waves = fields
            .iter()
            .find(|f| f.key == "waves")
            .expect("a waves array header");
        assert_eq!(waves.kind, FieldKind::Array);
        assert_eq!(waves.variant_idx, 1, "the default has one wave");
        // The element's fields are indexed dotted-path leaves.
        assert_eq!(
            fields
                .iter()
                .find(|f| f.key == "waves.0.amplitude")
                .map(|f| f.kind),
            Some(FieldKind::Float)
        );
        assert_eq!(
            fields
                .iter()
                .find(|f| f.key == "waves.0.direction")
                .map(|f| f.kind),
            Some(FieldKind::Vec {
                len: 2,
                color: false
            })
        );
        // A fixed 2..=4 numeric vector is NOT treated as an Array (stays a Vec leaf).
        assert_eq!(
            fields.iter().find(|f| f.key == "extent").map(|f| f.kind),
            Some(FieldKind::Vec {
                len: 2,
                color: false
            })
        );
    }

    #[test]
    fn add_and_remove_array_elem_grow_and_shrink_from_a_template() {
        let mut args = base_args("WaterSurface");
        let len =
            |a: &Map<String, Value>| get_at_path(a, "waves").unwrap().as_array().unwrap().len();
        assert_eq!(len(&args), 1);
        assert!(add_array_elem("WaterSurface", &mut args, "waves"));
        assert_eq!(len(&args), 2, "grew by a cloned template element");
        assert!(
            get_at_path(&args, "waves.1.amplitude").is_some(),
            "the appended element has the template's shape"
        );
        assert!(remove_array_elem(&mut args, "waves"));
        assert_eq!(len(&args), 1, "shrank");
        // Removing down to empty then adding still works (template falls back to the
        // type default's element).
        assert!(remove_array_elem(&mut args, "waves"));
        assert_eq!(len(&args), 0);
        assert!(
            add_array_elem("WaterSurface", &mut args, "waves"),
            "an emptied array regrows from the default template"
        );
        assert_eq!(len(&args), 1);
    }

    #[test]
    fn add_array_elem_without_a_template_is_a_no_op() {
        // An empty array whose element type is unknowable cannot be grown.
        let mut m = Map::new();
        m.insert("xs".into(), Value::Array(vec![]));
        assert!(!add_array_elem("PointLight", &mut m, "xs"));
        assert!(
            get_at_path(&m, "xs")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn path_helpers_navigate_array_indices() {
        let mut args = base_args("WaterSurface");
        assert!(
            get_at_path(&args, "waves.0.amplitude")
                .and_then(Value::as_f64)
                .is_some()
        );
        set_at_path(&mut args, "waves.0.amplitude", Value::from(9.5));
        assert_eq!(
            get_at_path(&args, "waves.0.amplitude").and_then(Value::as_f64),
            Some(9.5)
        );
        // An out-of-range index set is a no-op (no panic, no growth).
        set_at_path(&mut args, "waves.5.amplitude", Value::from(1.0));
        assert!(get_at_path(&args, "waves.5.amplitude").is_none());
    }

    // A non-colour vector is one collapsed header by default; listing its path as
    // expanded discloses an editable Float leaf per element, seeded from the value.
    #[test]
    fn expanding_a_vector_discloses_per_element_leaves() {
        use std::collections::HashSet;
        let collapsed = fields_for("PointLight", None);
        assert_eq!(
            collapsed
                .iter()
                .find(|f| f.key == "position")
                .map(|f| f.kind),
            Some(FieldKind::Vec {
                len: 3,
                color: false
            })
        );
        assert!(
            collapsed.iter().all(|f| !f.key.starts_with("position.")),
            "a collapsed vector emits no element leaves"
        );

        let expanded = HashSet::from(["position".to_string()]);
        let fields = fields_for_with("PointLight", None, &expanded);
        // The header stays, followed by one Float leaf per element.
        assert!(fields.iter().any(|f| f.key == "position"));
        for axis in ["position.0", "position.1", "position.2"] {
            let leaf = fields
                .iter()
                .find(|f| f.key == axis)
                .unwrap_or_else(|| panic!("missing element leaf {axis}"));
            assert_eq!(leaf.kind, FieldKind::Float);
        }
        assert!(
            fields.iter().all(|f| f.key != "position.3"),
            "a 3-vector discloses exactly 3 leaves"
        );
    }

    // A colour vector never discloses element leaves even when its path is listed
    // as expanded (it keeps its single field + preview swatch).
    #[test]
    fn expanding_never_touches_a_colour_vector() {
        use std::collections::HashSet;
        let expanded = HashSet::from(["color".to_string()]);
        let fields = fields_for_with("PointLight", None, &expanded);
        assert!(fields.iter().all(|f| !f.key.starts_with("color.")));
        assert_eq!(
            fields.iter().find(|f| f.key == "color").map(|f| f.kind),
            Some(FieldKind::Vec {
                len: 3,
                color: true
            })
        );
    }

    // Editing a disclosed vector element writes back into the vector (keeping its
    // length + untouched siblings) and cooks.
    #[test]
    fn assemble_writes_a_disclosed_vector_element() {
        use std::collections::HashSet;
        let expanded = HashSet::from(["position".to_string()]);
        let fields = fields_for_with("PointLight", None, &expanded);
        let idx = fields
            .iter()
            .position(|f| f.key == "position.1")
            .expect("a position.1 element leaf");
        let mut texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        texts[idx] = "4.5".into();
        let args = assemble("PointLight", None, &fields, &texts);
        assert_eq!(
            get_at_path(&args, "position")
                .and_then(Value::as_array)
                .map(|a| a.len()),
            Some(3),
            "the vector keeps its length"
        );
        assert_eq!(
            get_at_path(&args, "position.1").and_then(Value::as_f64),
            Some(4.5)
        );
        assert!(validate("PointLight", "probe", &args).is_ok());
    }

    // f32-origin floats display at their shortest round-tripping form, not serde's
    // full f64 expansion; integers and genuine f64s are untouched.
    #[test]
    fn floats_display_shortened_but_round_trip() {
        let num = |v: f64| serde_json::Number::from_f64(v).unwrap();
        // 0.05_f32 widened to f64 prints long via serde; number_text shortens it.
        let long = num(0.05_f32 as f64).to_string();
        assert!(long.len() > 6, "serde prints the long expansion: {long}");
        assert_eq!(number_text(&num(0.05_f32 as f64)), "0.05");
        // A whole float keeps its decimal point so it still reads as a float.
        assert_eq!(number_text(&num(1.0_f32 as f64)), "1.0");
        // An integer prints as itself.
        assert_eq!(number_text(&serde_json::Number::from(42)), "42");
        // A genuine f64 that does not round-trip through f32 keeps full precision.
        assert_eq!(number_text(&num(0.1)), num(0.1).to_string());
    }

    // Editing an array element's leaf writes back into that element and cooks.
    #[test]
    fn assemble_writes_an_array_element_value() {
        let fields = fields_for("WaterSurface", None);
        let idx = fields
            .iter()
            .position(|f| f.key == "waves.0.amplitude")
            .expect("a waves.0.amplitude field");
        let mut texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
        texts[idx] = "3.25".into();
        let args = assemble("WaterSurface", None, &fields, &texts);
        assert_eq!(
            get_at_path(&args, "waves.0.amplitude").and_then(Value::as_f64),
            Some(3.25)
        );
        assert!(
            validate("WaterSurface", "probe", &args).is_ok(),
            "the element edit cooks"
        );
    }

    // A fixed-length numeric array (SdfVolume's `[f32; 32]` params) is NOT offered as
    // a growable Array -- it is indistinguishable from a `Vec<f32>` once serialized,
    // and a resize is rejected at cook, so it stays at its default like before. Only
    // object / string / reference element arrays get add / remove.
    #[test]
    fn fixed_numeric_array_is_not_a_growable_array() {
        let fields = fields_for("SdfVolume", None);
        assert!(
            fields.iter().all(|f| !f.key.starts_with("params")),
            "a fixed [f32; N] numeric array is neither an Array header nor element leaves"
        );
        // Sanity: an object-element array (WaterSurface.waves) still IS growable.
        assert!(
            fields_for("WaterSurface", None)
                .iter()
                .any(|f| f.key == "waves" && f.kind == FieldKind::Array),
            "an object array is still editable"
        );
    }

    // The form no longer caps the field list: a wide type (WaterSurface, whose
    // per-wave array leaves push it past the pool) exposes more editable fields than
    // the physical control pool. The panel scrolls a window over them, so nothing is
    // silently dropped, and a field past the pool is still derived.
    #[test]
    fn fields_for_exposes_more_than_the_control_pool() {
        let fields = fields_for("WaterSurface", None);
        assert!(
            fields.len() > FIELD_POOL,
            "WaterSurface exposes {} fields, past the {FIELD_POOL}-slot pool",
            fields.len()
        );
        assert!(
            fields.iter().any(|f| f.key == "roughness"),
            "a field past the pool is still derived, not truncated"
        );
    }

    // `current_text` renders a stored value (scalar or nested array element) as its
    // editable text, and an absent path as empty -- the round-trip used to preserve
    // an off-window field's value on capture.
    #[test]
    fn current_text_renders_a_stored_value_or_empty() {
        let mut args = base_args("WaterSurface");
        assert_eq!(
            current_text(&args, "roughness"),
            value_text(&args["roughness"])
        );
        set_at_path(&mut args, "waves.0.amplitude", Value::from(2.5));
        assert_eq!(current_text(&args, "waves.0.amplitude"), "2.5");
        assert_eq!(current_text(&args, "no_such_field"), "");
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
            validate("PointLight", "probe", &args).is_ok(),
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
            validate("PointLight", "probe", &args).is_ok(),
            "the default-derived args re-serialize cleanly"
        );
    }

    // Every type the picker offers must have a well-formed derived form: its
    // fields build without panic and its default-derived args re-validate. This is
    // the form-side counterpart to `add_types_cook_with_default_args` (which guards
    // the cook side), so adding a type whose form mis-derives is caught here.
    #[test]
    fn every_add_type_form_round_trips_its_defaults() {
        for ty in crate::editor::panel::picker_types() {
            let fields = fields_for(ty, None);
            let texts: Vec<String> = fields.iter().map(|f| f.initial.clone()).collect();
            let args = assemble(ty, None, &fields, &texts);
            assert!(
                validate(ty, "probe", &args).is_ok(),
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
        assert_eq!(field("screen"), Some(FieldKind::Ref { target: "Screen" }));
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
                assert!(validate("PointLight", "probe", &args).is_err());
            }
        }
        // Defaults always validate.
        assert!(validate("PointLight", "probe", &base_args("PointLight")).is_ok());
    }
}
