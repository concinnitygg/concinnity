// Assembles the asset reference: one documented entry per authorable asset,
// plus the reference types their fields reach.
//
// Everything comes from the schema the derives emit (`concinnity_core::ecs::schema`),
// reached through the authoring registry: each registered type names its args
// schema, and a field's type names the nested struct or enum it holds.
//
// For each asset (and each nested value type) the entry contains:
//
// - `summary`: first paragraph of the type's rustdoc.
// - `full_doc`: the rustdoc (hand-written table lines stripped) followed by a
//   `## Parameters` bullet list of the type's fields. Each bullet states the
//   field's JSON type in prose (so no Rust type name, enum, struct, or
//   otherwise, ever reaches the user), folds in the field's own rustdoc, and
//   appends the default unless the prose already covers it.
//
// Which types get a page is discovered, not listed: every registered type
// whose origin is anything other than RuntimeOnly is an authorable asset and
// gets a page. Nested objects a field embeds (a Prop's collider, the element
// type of an array) and documented string enums a field uses (AaMode, ...)
// each get their own page too and are linked from the fields that use them,
// the way a JSON schema separates `$defs` from the objects that reference
// them.
//
// A documented page links cross-references as relative markdown:
// `[AaMode](AaMode.md)`, so the docs cross-link correctly when browsed as plain
// markdown. Hand-written `](#anchor)` links in the rustdoc are rewritten to the
// same relative form. A docs viewer rewrites the `.md` suffix to its own
// routes at render time.

use std::collections::{BTreeMap, HashMap};

use concinnity_core::ecs::schema::{self, Body, FieldSchema, TypeSchema, ValueSchema};

use super::defaults::{Defaults, default_text};
use super::prose::{
    collapse_doc, entry_examples_as_lines, first_paragraph, strip_rust_blocks, strip_table_lines,
};
use super::render::{
    EnumValue, FieldEntry, FieldType, render_parameters, render_values, rewrite_doc_links, slug,
};

/// One documented type: an authorable asset, or a reference type (a nested
/// value type or documented enum) an asset embeds.
pub(super) struct AssetDoc {
    /// The type's registry name.
    pub(super) type_name: String,
    /// First paragraph of the type's rustdoc.
    pub(super) summary: String,
    /// The type's full rustdoc body.
    pub(super) full_doc: String,
    /// True for a nested value type or enum rather than an asset.
    pub(super) is_reference_type: bool,
}

/// An authorable asset: its registry name and the schema a world line is read
/// through.
pub(super) struct AssetEntry {
    pub(super) name: &'static str,
    pub(super) schema: &'static TypeSchema,
}

/// Every authorable asset in the authoring registry. A RuntimeOnly component
/// is engine-internal, never declared in a world, so it gets no page.
pub(super) fn registry_assets() -> Vec<AssetEntry> {
    use concinnity_cook::authoring::registry::{AssetOrigin, RegisteredType};

    RegisteredType::all()
        .iter()
        .filter(|ty| ty.registration().origin != AssetOrigin::RuntimeOnly)
        .map(|ty| AssetEntry {
            name: ty.as_str(),
            schema: ty.schema(),
        })
        .collect()
}

/// Every documented type, assets first, each group sorted by name.
pub(super) fn build(assets: &[AssetEntry]) -> Vec<AssetDoc> {
    let mut ctx = Ctx {
        asset_by_schema: assets.iter().map(|a| (a.schema.name, a.name)).collect(),
        value_types: BTreeMap::new(),
        enums: BTreeMap::new(),
    };

    let mut asset_docs: Vec<Entry> = assets
        .iter()
        .map(|a| {
            let (summary, full_doc) = render_struct(a.schema, &mut ctx);
            Entry {
                name: a.name.to_string(),
                summary,
                full_doc,
            }
        })
        .collect();

    // Reference types: value-type structs to a fixpoint (one may embed another
    // or reference an enum), then the documented enums those passes reached.
    let mut ref_types: Vec<Entry> = Vec::new();
    let mut done: Vec<&str> = Vec::new();
    loop {
        let pending: Vec<&'static TypeSchema> = ctx
            .value_types
            .values()
            .filter(|s| !done.contains(&s.name))
            .copied()
            .collect();
        if pending.is_empty() {
            break;
        }
        for s in pending {
            done.push(s.name);
            let (summary, full_doc) = render_struct(s, &mut ctx);
            ref_types.push(Entry {
                name: s.name.to_string(),
                summary: or_fallback(summary, "Nested object embedded by other assets."),
                full_doc,
            });
        }
    }
    for s in ctx.enums.values() {
        let (summary, full_doc) = render_enum(s);
        ref_types.push(Entry {
            name: s.name.to_string(),
            summary: or_fallback(summary, "A set of named string values."),
            full_doc,
        });
    }

    // Rewrite hand-written `](#anchor)` cross-references in every doc body to
    // the relative `.md` form, resolving anchors through the set of all
    // documented names.
    let name_for_slug: HashMap<String, String> = asset_docs
        .iter()
        .chain(ref_types.iter())
        .map(|e| (slug(&e.name), e.name.clone()))
        .collect();
    for e in asset_docs.iter_mut().chain(ref_types.iter_mut()) {
        e.summary = rewrite_doc_links(&e.summary, &name_for_slug);
        e.full_doc = rewrite_doc_links(&e.full_doc, &name_for_slug);
    }

    asset_docs.sort_by(|a, b| a.name.cmp(&b.name));
    ref_types.sort_by(|a, b| a.name.cmp(&b.name));
    let docs = |entries: Vec<Entry>, is_reference_type| {
        entries.into_iter().map(move |e| AssetDoc {
            type_name: e.name,
            summary: e.summary,
            full_doc: e.full_doc,
            is_reference_type,
        })
    };
    docs(asset_docs, false)
        .chain(docs(ref_types, true))
        .collect()
}

// The types a render pass has reached so far.
struct Ctx {
    // An asset's args schema name -> the asset's registry name, so a field
    // embedding another asset links to that asset's own page.
    asset_by_schema: HashMap<&'static str, &'static str>,
    // Value-type structs reached from a field, each owed a page.
    value_types: BTreeMap<&'static str, &'static TypeSchema>,
    // Documented enums reached from a field, each owed a page.
    enums: BTreeMap<&'static str, &'static TypeSchema>,
}

// One rendered type: its name, one-line summary, and full doc body (the
// description followed by the generated Parameters/Values section).
struct Entry {
    name: String,
    summary: String,
    full_doc: String,
}

fn or_fallback(summary: String, fallback: &str) -> String {
    if summary.is_empty() {
        fallback.to_string()
    } else {
        summary
    }
}

// A struct's page body: its rustdoc, then its fields as parameter bullets.
fn render_struct(s: &'static TypeSchema, ctx: &mut Ctx) -> (String, String) {
    let doc = entry_examples_as_lines(&strip_rust_blocks(s.doc));
    let mut fields = Vec::new();
    collect_fields(s, ctx, &mut fields);
    let full_doc = combine(&strip_table_lines(&doc), &render_parameters(&fields));
    (first_paragraph(&doc), full_doc)
}

// Render a documented enum's page body: its enum-level rustdoc followed by a
// `## Values` list, one bullet per authored name with its own doc.
fn render_enum(s: &TypeSchema) -> (String, String) {
    let values: Vec<EnumValue> = enum_values(s)
        .iter()
        .map(|v| EnumValue {
            value: v.name.to_string(),
            doc: collapse_doc(v.doc),
        })
        .collect();
    let cleaned = strip_table_lines(&strip_rust_blocks(s.doc));
    (
        first_paragraph(s.doc),
        combine(&cleaned, &render_values(&values)),
    )
}

// Join a cleaned description with a generated section, dropping whichever is
// empty.
fn combine(description: &str, section: &str) -> String {
    match (description.is_empty(), section.is_empty()) {
        (_, true) => description.to_string(),
        (true, false) => section.to_string(),
        (false, false) => format!("{}\n\n{}", description, section.trim_end()),
    }
}

// The parameter bullets for `s`'s fields. A flattened field contributes its
// own type's fields in its place, with that type's defaults, since serde reads
// them from the same object.
fn collect_fields(s: &'static TypeSchema, ctx: &mut Ctx, out: &mut Vec<FieldEntry>) {
    let Body::Fields(fields) = s.body else {
        return;
    };
    let defaults = Defaults::of(s);
    for field in fields {
        if field.is_flattened() {
            if let schema::FieldType::Nested(inner) = (field.ty)() {
                collect_fields(inner, ctx, out);
            }
            continue;
        }
        out.push(field_entry(field, &defaults, ctx));
    }
}

fn field_entry(field: &FieldSchema, defaults: &Defaults, ctx: &mut Ctx) -> FieldEntry {
    let ty = (field.ty)();
    FieldEntry {
        key: field.key.to_string(),
        optional: matches!(ty, schema::FieldType::Optional(_)),
        default: defaults
            .field(field)
            .filter(|value| states_something(unwrap_optional(ty), value))
            .map(|value| default_text(&value)),
        ty: resolve_type(ty, ctx),
        doc: collapse_doc(field.doc),
    }
}

// Whether a default says more than the page already does. A reference naming
// no asset says nothing, and neither does a nested object that is exactly its
// own type's default, which that type's page states field by field.
fn states_something(ty: schema::FieldType, value: &serde_json::Value) -> bool {
    match ty {
        schema::FieldType::Reference(_) => value.as_str() != Some(""),
        schema::FieldType::Nested(inner) => Defaults::of(inner).container() != Some(value),
        _ => true,
    }
}

fn unwrap_optional(ty: schema::FieldType) -> schema::FieldType {
    match ty {
        schema::FieldType::Optional(inner) => unwrap_optional(inner()),
        other => other,
    }
}

// The phrase-level type of a field, recording each nested struct and
// documented enum it reaches so that type gets its own page.
fn resolve_type(ty: schema::FieldType, ctx: &mut Ctx) -> FieldType {
    match ty {
        schema::FieldType::Bool => FieldType::Bool,
        schema::FieldType::Float => FieldType::Float,
        schema::FieldType::Integer => FieldType::Integer,
        schema::FieldType::Str | schema::FieldType::Reference(_) => FieldType::Str,
        schema::FieldType::Object => FieldType::Object,
        schema::FieldType::Optional(inner) => resolve_type(inner(), ctx),
        schema::FieldType::Array { elem, len } => FieldType::Array {
            elem: Box::new(resolve_type(elem(), ctx)),
            len,
        },
        schema::FieldType::Nested(s) => match ctx.asset_by_schema.get(s.name) {
            Some(asset) => FieldType::Named(asset.to_string()),
            None => {
                ctx.value_types.entry(s.name).or_insert(s);
                FieldType::Named(s.name.to_string())
            }
        },
        schema::FieldType::Enum(s) => {
            if enum_is_documented(s) {
                ctx.enums.entry(s.name).or_insert(s);
                FieldType::NamedEnum(s.name.to_string())
            } else {
                FieldType::Enum(enum_values(s).iter().map(|v| v.name.to_string()).collect())
            }
        }
    }
}

fn enum_values(s: &TypeSchema) -> &'static [ValueSchema] {
    match s.body {
        Body::Values(values) => values,
        Body::Fields(_) => &[],
    }
}

// A documented enum gets its own page (its values carry their docs there); an
// undocumented one is rendered inline as a closed set of string values.
fn enum_is_documented(s: &TypeSchema) -> bool {
    !s.doc.trim().is_empty() || enum_values(s).iter().any(|v| !v.doc.trim().is_empty())
}
