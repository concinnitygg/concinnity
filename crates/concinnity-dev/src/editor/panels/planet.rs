//! The data half of the Planet panel: every arg of the world's `Planet` laid out
//! as themed sections of rows, and the edits its controls make. The scalar
//! fields resolve through `form::fields_for_with` like the Lighting panel's, so
//! their kinds and current values match the edit form; the nested pieces the
//! form does not reach (the sea and each ground rule) are built here over the
//! same `FormField`. Every edit lands on a working copy of the args the hook
//! validates through `form::validate` before committing. Pure and world-free.

use serde_json::{Map, Value, json};

use super::form::{self, FieldKind, FormField};

/// The asset type the panel edits.
pub(crate) const TY: &str = "Planet";

/// The most covers a planet lays.
pub(crate) const MAX_COVERS: usize = concinnity_engine::planet::MAX_GROUND_COVERS;

/// The most grass layers the panel adds: as many as the grass pass draws.
pub(crate) const MAX_LAYERS: usize = concinnity_engine::render::uniforms::grass::MAX_GRASS_LAYERS;

// Rows under each list entry: its heading, its own reference and its rule.
const ITEM_ROWS: usize = 8;

/// The most rows a planet shows: six headings, the shape's eight, the coasts'
/// four, the sea's two, the ground's material and covers heading, every cover,
/// the grass heading, every layer and the gravity.
pub(crate) const MAX_ROWS: usize =
    6 + 8 + 4 + 2 + 2 + MAX_COVERS * ITEM_ROWS + 1 + MAX_LAYERS * ITEM_ROWS + 1;

// The vectors shown as one row per component.
const EXPANDED: &[&str] = &["center"];

// The target type of the sea's water reference.
const WATER_TARGETS: &[&str] = &["WaterSurface"];

/// One themed group of rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Section {
    Shape,
    Coasts,
    Sea,
    Ground,
    Grass,
    Gravity,
}

impl Section {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::Shape => "Shape",
            Self::Coasts => "Continents and coasts",
            Self::Sea => "Sea",
            Self::Ground => "Ground",
            Self::Grass => "Grass",
            Self::Gravity => "Gravity",
        }
    }
}

/// A list of the planet's: its covers or its grass layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum List {
    Covers,
    Layers,
}

impl List {
    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Covers => "covers",
            Self::Layers => "layers",
        }
    }

    pub(crate) fn max(self) -> usize {
        match self {
            Self::Covers => MAX_COVERS,
            Self::Layers => MAX_LAYERS,
        }
    }

    fn item(self) -> &'static str {
        match self {
            Self::Covers => "cover",
            Self::Layers => "layer",
        }
    }
}

/// A button on a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Button {
    /// Draw a new seed.
    Reroll,
    /// Drop the sea.
    RemoveSea,
    /// Append an entry to a list.
    Add(List),
    /// Move entry `i` of a list one place earlier.
    Raise(List, usize),
    /// Drop entry `i` of a list.
    Remove(List, usize),
}

impl Button {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Reroll => "reroll",
            Self::RemoveSea | Self::Remove(..) => "remove",
            Self::Add(_) => "add",
            Self::Raise(..) => "up",
        }
    }
}

/// What a row shows.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RowKind {
    Header(Section),
    /// The world has no planet.
    AddPlanet,
    /// The planet has no sea.
    AddSea,
    /// One control. `nullable`: an empty number box leaves the value unset,
    /// and a reference may name nothing.
    Field {
        field: FormField,
        nullable: bool,
    },
    /// A list's heading with its length.
    List {
        list: List,
        count: usize,
    },
    /// The heading of entry `index` of a list.
    Item {
        list: List,
        index: usize,
    },
}

/// One panel row: what it shows, its caption, and its buttons.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Row {
    pub kind: RowKind,
    pub caption: String,
    /// From the row's right end leftward.
    pub buttons: [Option<Button>; 2],
}

impl Row {
    fn new(kind: RowKind, caption: impl Into<String>) -> Self {
        Self {
            kind,
            caption: caption.into(),
            buttons: [None, None],
        }
    }

    fn with(mut self, buttons: [Option<Button>; 2]) -> Self {
        self.buttons = buttons;
        self
    }

    /// The control's field, on a field row.
    pub(crate) fn field(&self) -> Option<&FormField> {
        match &self.kind {
            RowKind::Field { field, .. } => Some(field),
            _ => None,
        }
    }
}

/// The planet's args with every default filled in: the type's defaults under
/// the authored args, and each cover's and layer's rule complete, so every
/// row addresses a value that exists. The `$id` is kept.
pub(crate) fn working(args: &Map<String, Value>) -> Map<String, Value> {
    let mut out = form::base_args(TY);
    for (k, v) in args {
        out.insert(k.clone(), v.clone());
    }
    let rule = rule_defaults();
    for list in [List::Covers, List::Layers] {
        if let Some(Value::Array(entries)) = out.get_mut(list.key()) {
            for entry in entries.iter_mut().filter_map(Value::as_object_mut) {
                let slot = entry
                    .entry("rule")
                    .or_insert_with(|| Value::Object(Map::new()));
                if !slot.is_object() {
                    *slot = Value::Object(Map::new());
                }
                if let Some(r) = slot.as_object_mut() {
                    for (k, v) in &rule {
                        r.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                }
            }
        }
    }
    if let Some(Value::Object(sea)) = out.get_mut("sea") {
        sea.entry("level").or_insert_with(|| json!(0.0));
    }
    out
}

fn rule_defaults() -> Map<String, Value> {
    match serde_json::to_value(concinnity_engine::components::GroundRule::default()) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

/// The panel's rows over the planet's `args` (as [`working`] fills them), or
/// the add row for a world without a planet.
pub(crate) fn rows(args: Option<&Map<String, Value>>) -> Vec<Row> {
    let Some(args) = args else {
        return vec![Row::new(RowKind::AddPlanet, "+ Add Planet")];
    };
    let expanded = EXPANDED.iter().map(ToString::to_string).collect();
    let derived = form::fields_for_with(TY, Some(args), &expanded);
    let mut out = Vec::new();
    let derived_row = |path: &str, caption: &str, nullable: bool| {
        derived
            .iter()
            .find(|f| f.key == path)
            .map(|f| field_row(f.clone(), caption, nullable))
    };

    out.push(Row::new(
        RowKind::Header(Section::Shape),
        Section::Shape.title(),
    ));
    for (path, caption) in [
        ("center.0", "center x"),
        ("center.1", "center y"),
        ("center.2", "center z"),
        ("radius", "radius"),
        ("amplitude", "amplitude"),
        ("feature_size", "feature size"),
        ("octaves", "octaves"),
    ] {
        out.extend(derived_row(path, caption, false));
    }
    out.extend(derived_row("seed", "seed", false).map(|r| r.with([Some(Button::Reroll), None])));

    out.push(Row::new(
        RowKind::Header(Section::Coasts),
        Section::Coasts.title(),
    ));
    for (path, caption) in [
        ("land_share", "land share"),
        ("continent_size", "continent size"),
        ("ocean_depth", "ocean depth"),
        ("beach_width", "beach width"),
    ] {
        out.extend(derived_row(path, caption, false));
    }

    let sea = args.get("sea").filter(|s| s.is_object());
    let header = Row::new(RowKind::Header(Section::Sea), Section::Sea.title());
    if sea.is_some() {
        out.push(header.with([Some(Button::RemoveSea), None]));
        out.push(field_row(number_field(args, "sea.level"), "level", false));
        out.push(field_row(
            ref_field(args, "sea.water", WATER_TARGETS),
            "water",
            false,
        ));
    } else {
        out.push(header);
        out.push(Row::new(RowKind::AddSea, "+ Add sea"));
    }

    out.push(Row::new(
        RowKind::Header(Section::Ground),
        Section::Ground.title(),
    ));
    out.extend(derived_row("material", "material", true));
    list_rows(args, List::Covers, &mut out, |i, out| {
        out.extend(derived_row(
            &format!("covers.{i}.material"),
            "material",
            true,
        ));
    });

    out.push(Row::new(
        RowKind::Header(Section::Grass),
        Section::Grass.title(),
    ));
    list_rows(args, List::Layers, &mut out, |i, out| {
        out.extend(derived_row(&format!("layers.{i}.grass"), "grass", false));
    });

    out.push(Row::new(
        RowKind::Header(Section::Gravity),
        Section::Gravity.title(),
    ));
    out.extend(derived_row("gravity", "gravity", false));
    out
}

// A list's heading, then each entry's heading, its `own` rows and its rule.
fn list_rows(
    args: &Map<String, Value>,
    list: List,
    out: &mut Vec<Row>,
    own: impl Fn(usize, &mut Vec<Row>),
) {
    let count = args
        .get(list.key())
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let add = (count < list.max()).then_some(Button::Add(list));
    out.push(
        Row::new(
            RowKind::List { list, count },
            format!("{} ({count}/{})", list.key(), list.max()),
        )
        .with([add, None]),
    );
    for i in 0..count {
        let raise = (i > 0).then_some(Button::Raise(list, i));
        out.push(
            Row::new(
                RowKind::Item { list, index: i },
                format!("{} {}", list.item(), i + 1),
            )
            .with([Some(Button::Remove(list, i)), raise]),
        );
        own(i, out);
        rule_rows(args, &format!("{}.{i}.rule", list.key()), out);
    }
}

// The six rows of the ground rule at `prefix`: its two bands, each end
// unset when its box is empty, and their edges.
fn rule_rows(args: &Map<String, Value>, prefix: &str, out: &mut Vec<Row>) {
    for (leaf, caption, nullable) in [
        ("altitude.0", "lowest altitude", true),
        ("altitude.1", "highest altitude", true),
        ("altitude_edge", "altitude edge", false),
        ("slope.0", "gentlest slope", true),
        ("slope.1", "steepest slope", true),
        ("slope_edge", "slope edge", false),
    ] {
        let field = number_field(args, &format!("{prefix}.{leaf}"));
        out.push(field_row(field, caption, nullable));
    }
}

fn field_row(field: FormField, caption: &str, nullable: bool) -> Row {
    Row::new(RowKind::Field { field, nullable }, caption)
}

fn blank_field(key: &str, kind: FieldKind, initial: String) -> FormField {
    FormField {
        key: key.to_string(),
        kind,
        initial,
        boolval: false,
        variants: Vec::new(),
        variant_idx: 0,
        choices: Vec::new(),
        sparse: false,
        unset: false,
    }
}

// A float box over the value at `path`, empty where it is unset.
fn number_field(args: &Map<String, Value>, path: &str) -> FormField {
    let initial = match value_at(args, path) {
        Some(Value::Number(n)) => form::number_text(n),
        _ => String::new(),
    };
    blank_field(path, FieldKind::Float, initial)
}

// A reference picker over the name at `path`.
fn ref_field(args: &Map<String, Value>, path: &str, targets: &'static [&'static str]) -> FormField {
    let initial = value_at(args, path)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    blank_field(path, FieldKind::Ref { targets }, initial)
}

/// Fill a reference field's options from `names`, the world's assets of its
/// target types. A reference that may not name nothing offers no `(none)`.
pub(crate) fn set_ref_options(field: &mut FormField, nullable: bool, names: &[String]) {
    form::set_ref_options(field, names);
    if !nullable && field.variants.len() > 1 {
        field.variants.remove(0);
        field.variant_idx = field.variant_idx.saturating_sub(1);
    }
}

/// What a reference picker writes on a click: the name after its current
/// option, null for `(none)`; `None` with no options to move through.
pub(crate) fn next_ref(field: &FormField) -> Option<Value> {
    if field.variants.is_empty() {
        return None;
    }
    let next = &field.variants[(field.variant_idx + 1) % field.variants.len()];
    Some(if next == form::NONE_LABEL {
        Value::Null
    } else {
        Value::String(next.clone())
    })
}

/// `args` with every number box's text written in: `texts[i]` is row `i`'s
/// box. An empty box on a nullable row unsets its value; a box that does not
/// parse keeps the value it had.
pub(crate) fn assemble(
    args: &Map<String, Value>,
    rows: &[Row],
    texts: &[String],
) -> Map<String, Value> {
    let boxes: Vec<(&FormField, bool, &str)> = rows
        .iter()
        .zip(texts)
        .filter_map(|(row, text)| match &row.kind {
            RowKind::Field { field, nullable } if field.kind.has_text_input() => {
                Some((field, *nullable, text.as_str()))
            }
            _ => None,
        })
        .collect();
    let fields: Vec<FormField> = boxes.iter().map(|(f, ..)| (*f).clone()).collect();
    let typed: Vec<String> = boxes.iter().map(|(.., t)| (*t).to_string()).collect();
    let mut out = form::assemble(TY, Some(args), &fields, &typed);
    for (field, nullable, text) in boxes {
        if nullable && text.trim().is_empty() {
            set_value(&mut out, &field.key, Value::Null);
        }
    }
    out
}

/// The value at a dotted `path` of `args`, list indices included.
pub(crate) fn value_at<'a>(args: &'a Map<String, Value>, path: &str) -> Option<&'a Value> {
    let mut parts = path.split('.');
    let mut cur = args.get(parts.next()?)?;
    for p in parts {
        cur = match cur {
            Value::Object(m) => m.get(p)?,
            Value::Array(a) => a.get(p.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

/// Write `value` at a dotted `path` that already exists in `args`; `false`
/// when it does not.
pub(crate) fn set_value(args: &mut Map<String, Value>, path: &str, value: Value) -> bool {
    let mut parts = path.split('.');
    let Some(mut cur) = parts.next().and_then(|first| args.get_mut(first)) else {
        return false;
    };
    for p in parts {
        let next = match cur {
            Value::Object(m) => m.get_mut(p),
            Value::Array(a) => p.parse::<usize>().ok().and_then(|i| a.get_mut(i)),
            _ => None,
        };
        let Some(next) = next else {
            return false;
        };
        cur = next;
    }
    *cur = value;
    true
}

/// The authored args after an edit took the [`working`] args `before` to
/// `after`: each top-level arg the edit changed written whole, one it dropped
/// removed, and every other left as authored, so the defaults the working
/// args fill in are never written back.
pub(crate) fn authored_edit(
    authored: &Map<String, Value>,
    before: &Map<String, Value>,
    after: &Map<String, Value>,
) -> Map<String, Value> {
    let mut out = authored.clone();
    for (k, v) in after {
        if before.get(k) != Some(v) {
            out.insert(k.clone(), v.clone());
        }
    }
    for k in before.keys().filter(|k| !after.contains_key(*k)) {
        out.remove(k);
    }
    out
}

/// The planet with a sea at its radius whose water looks like `water`.
pub(crate) fn with_sea(args: &Map<String, Value>, water: &str) -> Map<String, Value> {
    let mut out = args.clone();
    out.insert("sea".to_string(), json!({"level": 0.0, "water": water}));
    out
}

/// A seed different from `seed`, the same for the same seed.
pub(crate) fn next_seed(seed: u32) -> u32 {
    let mut x = seed.wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

/// The planet after `button`, or `None` when it changes nothing. A new grass
/// layer grows `grass`; without one no layer is added.
pub(crate) fn press(
    args: &Map<String, Value>,
    button: Button,
    grass: Option<&str>,
) -> Option<Map<String, Value>> {
    let mut out = args.clone();
    match button {
        Button::Reroll => {
            let seed = out.get("seed").and_then(Value::as_u64).unwrap_or(0);
            let seed = next_seed(u32::try_from(seed).unwrap_or(0));
            out.insert("seed".to_string(), json!(seed));
        }
        Button::RemoveSea => {
            out.remove("sea")?;
        }
        Button::Add(list) => {
            let entry = match list {
                List::Covers => json!({"material": null, "rule": rule_defaults()}),
                List::Layers => json!({"grass": grass?, "rule": rule_defaults()}),
            };
            let entries = list_mut(&mut out, list)?;
            if entries.len() >= list.max() {
                return None;
            }
            entries.push(entry);
        }
        Button::Raise(list, i) => {
            let entries = list_mut(&mut out, list)?;
            if i == 0 || i >= entries.len() {
                return None;
            }
            entries.swap(i - 1, i);
        }
        Button::Remove(list, i) => {
            let entries = list_mut(&mut out, list)?;
            if i >= entries.len() {
                return None;
            }
            entries.remove(i);
        }
    }
    Some(out)
}

// The list `list` of `args`, made an empty one where it is missing.
fn list_mut(args: &mut Map<String, Value>, list: List) -> Option<&mut Vec<Value>> {
    let slot = args
        .entry(list.key().to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if !slot.is_array() {
        *slot = Value::Array(Vec::new());
    }
    slot.as_array_mut()
}

#[cfg(test)]
mod tests;
