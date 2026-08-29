// Assembles the asset reference: one documented entry per authorable asset,
// plus the reference types their fields reach.
//
// The asset schema is read from two source trees: the assets a world can hold
// or the cook compiles, in concinnity-core, and the build-only ones the cook
// expands away, in concinnity-world. Both are read here and joined into one
// index.
//
// For each asset (and each nested value type) the entry contains:
//
// - `summary`: first paragraph of the struct-level rustdoc.
// - `full_doc`: struct-level rustdoc (hand-written table lines stripped)
//   followed by a `## Parameters` bullet list generated from the asset's
//   `args` fields. Each bullet states the field's JSON type in prose (so no
//   Rust type name, enum, struct, or otherwise, ever reaches the user), folds
//   in the field's own rustdoc, and appends the default unless the prose
//   already covers it.
//
// Which types get a page is discovered, not listed: every Component whose
// `ORIGIN` is anything other than RuntimeOnly is an authorable asset and gets a
// page. Nested objects a field embeds (a Prop's collider, the element type of
// an array) and documented string enums a field uses (ShaderKind, AaMode, ...)
// each get their own page too and are linked from the fields that use them, the
// way a JSON schema separates `$defs` from the objects that reference them.
//
// A documented page links cross-references as relative markdown:
// `[ShaderKind](ShaderKind.md)`, so the docs cross-link correctly when browsed
// as plain markdown. Hand-written `](#anchor)` links in the source rustdoc are
// rewritten to the same relative form. A docs viewer rewrites the `.md` suffix
// to its own routes at render time.

use super::render::{
    EnumValue, FieldEntry, FieldType, render_parameters, render_values, rewrite_doc_links, slug,
};
use super::schema::{self, DocField, DocFieldType, DocShape, DocType, DocValue};

use std::collections::{BTreeSet, HashMap};
use std::io;
use std::path::Path;

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

// The schema source trees, relative to a checkout of the engine. The stored
// and resource vocabulary sits with its runtime half in core; the build-only
// assets, which never reach a running world, sit with their registry group in
// concinnity-world.
const BUILD_ONLY_SCHEMA: &str = "crates/concinnity-world/src/schema";
const RUNTIME_SCHEMA: &str = "crates/concinnity-core/src/components";

/// Every documented type, assets first, each group sorted by name.
///
/// `engine_root` is a checkout of the engine whose asset sources the prose is
/// read from.
pub(super) fn build(engine_root: &Path) -> io::Result<Vec<AssetDoc>> {
    let authored_src = engine_root.join(BUILD_ONLY_SCHEMA);
    let runtime_src = engine_root.join(RUNTIME_SCHEMA);
    for dir in [&authored_src, &runtime_src] {
        if !dir.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "{} not found: `cn docs` reads the asset prose out of the engine's \
                     sources, so it runs from a checkout of the engine",
                    dir.display()
                ),
            ));
        }
    }

    let mut types = schema::extract(&[authored_src], &[])?;
    types.extend(schema::extract(&[runtime_src], &[])?);

    let reference = assemble(&types);
    let mut out = Vec::with_capacity(reference.assets.len() + reference.ref_types.len());
    for e in reference.assets {
        out.push(AssetDoc {
            type_name: e.name,
            summary: e.summary,
            full_doc: e.full_doc,
            is_reference_type: false,
        });
    }
    for e in reference.ref_types {
        out.push(AssetDoc {
            type_name: e.name,
            summary: e.summary,
            full_doc: e.full_doc,
            is_reference_type: true,
        });
    }
    Ok(out)
}

// Cross-tree indices over the extracted schema. An enum, a value-type struct,
// and the asset that references them can each come from a different source
// tree, so every lookup goes through these.
struct Ctx<'a> {
    // Named-field struct name -> its extracted shape.
    structs: HashMap<&'a str, &'a DocType>,
    // String-valued enum name -> its extracted shape.
    enums: HashMap<&'a str, &'a DocType>,
    // Documented Component struct ident -> its declarable NAME (for linking to
    // an asset's page). Only authorable components are listed, so links never
    // point at a page that was not generated.
    comp_by_struct: HashMap<String, String>,
}

impl<'a> Ctx<'a> {
    fn new(types: &'a [DocType], comp_by_struct: HashMap<String, String>) -> Self {
        let mut structs = HashMap::new();
        let mut enums = HashMap::new();
        for ty in types {
            match ty.shape {
                DocShape::Fields(_) => structs.insert(ty.name.as_str(), ty),
                DocShape::Values(_) => enums.insert(ty.name.as_str(), ty),
            };
        }
        Ctx {
            structs,
            enums,
            comp_by_struct,
        }
    }

    // The type's rustdoc, verbatim; empty for a name that is not a struct.
    fn struct_doc(&self, name: &str) -> &'a str {
        self.structs.get(name).map_or("", |ty| ty.doc.as_str())
    }

    // The struct's serialized fields; empty for a name that is not a struct.
    fn fields(&self, name: &str) -> &'a [DocField] {
        match self.structs.get(name).map(|ty| &ty.shape) {
            Some(DocShape::Fields(fields)) => fields,
            _ => &[],
        }
    }

    fn enum_values(&self, name: &str) -> Option<(&'a str, &'a [DocValue])> {
        match self.enums.get(name).map(|ty| (&ty.doc, &ty.shape)) {
            Some((doc, DocShape::Values(values))) => Some((doc.as_str(), values.as_slice())),
            _ => None,
        }
    }
}

// A documented enum gets its own page (its values carry their docs there); an
// undocumented one is rendered inline as a closed set of string values.
fn enum_is_documented(doc: &str, values: &[DocValue]) -> bool {
    !doc.trim().is_empty() || values.iter().any(|v| !v.doc.trim().is_empty())
}

struct ComponentMeta {
    name: String,
    struct_ident: String,
    args_struct: String,
    // "External" | "RuntimeOnly" | "BuildOnly" (RuntimeOnly when unspecified).
    origin: String,
}

// Value-type structs and documented enums a render pass discovered as reachable
// from a field. Each gets its own page.
#[derive(Default)]
struct Refs {
    value_types: BTreeSet<String>,
    enums: BTreeSet<String>,
}

// One rendered type: its name, one-line summary, and full doc body (the
// description followed by the generated Parameters/Values section).
struct Entry {
    name: String,
    summary: String,
    full_doc: String,
}

// The whole reference, split into the authorable assets and the reference types
// their fields reach. Both are sorted by name.
struct Reference {
    assets: Vec<Entry>,
    ref_types: Vec<Entry>,
}

// Join the extracted schema into one index and render every documented type
// from it.
fn assemble(types: &[DocType]) -> Reference {
    // Manual components keep a literal `impl Component`; the rest come from the
    // concinnity-world registries. A type appears in exactly one of the two.
    let all_components = collect_registry_components();

    // Authorable assets only: a RuntimeOnly component is engine-internal, never
    // declared in a world, so it gets no page.
    let documented: Vec<&ComponentMeta> = all_components
        .iter()
        .filter(|c| c.origin != "RuntimeOnly")
        .collect();
    let ctx = Ctx::new(
        types,
        documented
            .iter()
            .map(|c| (c.struct_ident.clone(), c.name.clone()))
            .collect(),
    );

    // Render every asset, collecting the value types and documented enums its
    // fields reach.
    let mut refs = Refs::default();
    let mut assets: Vec<Entry> = Vec::new();
    for c in &documented {
        let (summary, full_doc) =
            render_doc_entry(&c.struct_ident, &c.args_struct, &ctx, &mut refs);
        assets.push(Entry {
            name: c.name.clone(),
            summary,
            full_doc,
        });
    }

    // Reference types: value-type structs to a fixpoint (one may embed another
    // or reference an enum), then the documented enums those passes reached.
    let mut ref_types: Vec<Entry> = Vec::new();
    let mut done_vt: BTreeSet<String> = BTreeSet::new();
    loop {
        let pending: Vec<String> = refs
            .value_types
            .iter()
            .filter(|n| !done_vt.contains(*n) && ctx.structs.contains_key(n.as_str()))
            .cloned()
            .collect();
        if pending.is_empty() {
            break;
        }
        for name in pending {
            done_vt.insert(name.clone());
            let (summary, full_doc) = render_doc_entry(&name, &name, &ctx, &mut refs);
            let summary = if summary.is_empty() {
                "Nested object embedded by other assets.".to_string()
            } else {
                summary
            };
            ref_types.push(Entry {
                name,
                summary,
                full_doc,
            });
        }
    }
    for name in &refs.enums {
        let (summary, full_doc) = render_enum_doc(name, &ctx);
        let summary = if summary.is_empty() {
            "A set of named string values.".to_string()
        } else {
            summary
        };
        ref_types.push(Entry {
            name: name.clone(),
            summary,
            full_doc,
        });
    }

    // Rewrite hand-written `](#anchor)` cross-references in every doc body to
    // the relative `.md` form, resolving anchors through the set of all
    // documented names.
    let mut name_for_slug: HashMap<String, String> = HashMap::new();
    for e in assets.iter().chain(ref_types.iter()) {
        name_for_slug.insert(slug(&e.name), e.name.clone());
    }
    for e in assets.iter_mut().chain(ref_types.iter_mut()) {
        e.summary = rewrite_doc_links(&e.summary, &name_for_slug);
        e.full_doc = rewrite_doc_links(&e.full_doc, &name_for_slug);
    }

    assets.sort_by(|a, b| a.name.cmp(&b.name));
    ref_types.sort_by(|a, b| a.name.cmp(&b.name));

    Reference { assets, ref_types }
}

// Every documented asset type, read through the concinnity-world authoring
// registry (the single source of the origin / args-schema metadata, since the
// runtime `Component` trait carries none). `RegisteredType::all` covers every
// declarable type, components and resources alike; `args_struct_name` names the
// authored args schema (the type itself for pass-through assets, the `args:`
// override for the divergent ones, whose fields the parameter table renders).
fn collect_registry_components() -> Vec<ComponentMeta> {
    use concinnity_world::registry::RegisteredType;

    RegisteredType::all()
        .iter()
        .map(|ty| {
            let name = ty.as_str().to_string();
            ComponentMeta {
                name: name.clone(),
                struct_ident: name,
                args_struct: ty.args_struct_name().to_string(),
                origin: format!("{:?}", ty.registration().origin),
            }
        })
        .collect()
}

// Doc entry rendering

// Render one entry: the description comes from `doc_ident`'s rustdoc, the
// parameter bullets from `args_ident`'s fields. For `type Args = Self` assets
// the two are the same struct; for value types both are the value type itself.
fn render_doc_entry(
    doc_ident: &str,
    args_ident: &str,
    ctx: &Ctx,
    refs: &mut Refs,
) -> (String, String) {
    let doc = strip_rust_blocks(ctx.struct_doc(doc_ident));
    let cleaned = strip_table_lines(&doc);
    let fields = build_fields(args_ident, ctx, refs);
    let params = render_parameters(&fields);
    let full_doc = combine(&cleaned, &params);
    (first_paragraph(&doc), full_doc)
}

// Render a documented enum's page body: its enum-level rustdoc followed by a
// `## Values` list, one bullet per serialized value with its own doc.
fn render_enum_doc(name: &str, ctx: &Ctx) -> (String, String) {
    let Some((doc, values)) = ctx.enum_values(name) else {
        return (String::new(), String::new());
    };
    let cleaned = strip_table_lines(&strip_rust_blocks(doc));
    let values: Vec<EnumValue> = values
        .iter()
        .map(|v| EnumValue {
            value: v.value.to_string(),
            doc: v.doc.to_string(),
        })
        .collect();
    let vals = render_values(&values);
    (first_paragraph(doc), combine(&cleaned, &vals))
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

fn build_fields(args_ident: &str, ctx: &Ctx, refs: &mut Refs) -> Vec<FieldEntry> {
    ctx.fields(args_ident)
        .iter()
        .map(|f| FieldEntry {
            key: f.key.to_string(),
            ty: resolve_type(&f.ty, ctx, refs),
            optional: f.optional,
            default: f.default.clone(),
            doc: f.doc.to_string(),
        })
        .collect()
}

// Resolve an extracted field type against the joined index. Records any nested
// non-asset struct, or any documented enum, it links to in `refs` so it gets
// its own page.
fn resolve_type(ty: &DocFieldType, ctx: &Ctx, refs: &mut Refs) -> FieldType {
    match ty {
        DocFieldType::Bool => FieldType::Bool,
        DocFieldType::Float => FieldType::Float,
        DocFieldType::Integer => FieldType::Integer,
        DocFieldType::Str => FieldType::Str,
        DocFieldType::Object => FieldType::Object,
        DocFieldType::Array { elem, len } => FieldType::Array {
            elem: Box::new(resolve_type(elem, ctx, refs)),
            len: *len,
        },
        DocFieldType::Name(id) => resolve_name(id, ctx, refs),
    }
}

// A name the extractor left undecided: an enum, another asset, a value type
// with its own page, or nothing the reference documents.
fn resolve_name(id: &str, ctx: &Ctx, refs: &mut Refs) -> FieldType {
    if let Some((doc, values)) = ctx.enum_values(id) {
        if enum_is_documented(doc, values) {
            refs.enums.insert(id.to_string());
            FieldType::NamedEnum(id.to_string())
        } else {
            FieldType::Enum(values.iter().map(|v| v.value.to_string()).collect())
        }
    } else if let Some(name) = ctx.comp_by_struct.get(id) {
        // A field embedding another asset's struct links to that asset's own
        // page.
        FieldType::Named(name.clone())
    } else if ctx.structs.contains_key(id) {
        refs.value_types.insert(id.to_string());
        FieldType::Named(id.to_string())
    } else {
        FieldType::Object
    }
}

// Doc-comment helpers

fn first_paragraph(doc: &str) -> String {
    let para = doc.split("\n\n").next().unwrap_or("");
    para.split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// Drop ```rust blocks. They are for the Rust caller reading `concinnity::components`;
// this reference is for the world.jsonl author, who is never shown a Rust name.
fn strip_rust_blocks(doc: &str) -> String {
    let mut out = String::new();
    let mut in_rust = false;
    for line in doc.lines() {
        let trimmed = line.trim();
        if in_rust {
            if trimmed == "```" {
                in_rust = false;
            }
            continue;
        }
        if trimmed.starts_with("```rust") || trimmed == "```no_run" || trimmed == "```ignore" {
            in_rust = true;
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

// Remove markdown table lines (starting with '|') from a doc string.
// Collapses the resulting double-blank lines left behind.
fn strip_table_lines(doc: &str) -> String {
    let mut out = String::new();
    let mut prev_blank = false;
    for line in doc.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('|') {
            continue;
        }
        let is_blank = trimmed.is_empty();
        if is_blank && prev_blank {
            continue;
        }
        out.push_str(line);
        out.push('\n');
        prev_blank = is_blank;
    }
    out.trim_end().to_string()
}
