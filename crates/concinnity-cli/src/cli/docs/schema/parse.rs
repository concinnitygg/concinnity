// Source walking and item collection: every named-field struct and every
// string-valued enum declared at the top level of a `.rs` file under the roots.

use super::attrs::{apply_case, collapse_doc, extract_doc, has_serde_skip, serde_kv};
use super::defaults::{self, UNKNOWN};
use super::model::{DocField, DocFieldType, DocShape, DocType, DocValue};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

pub(super) fn types(roots: &[PathBuf], exclude: &[PathBuf]) -> io::Result<Vec<DocType>> {
    let mut out = Vec::new();
    for root in roots {
        let mut paths = Vec::new();
        collect(root, &mut paths)?;
        paths.sort();
        for path in paths {
            if exclude.iter().any(|e| e == &path) {
                continue;
            }
            file_types(&path, &mut out)?;
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", dir.display())))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn file_types(path: &Path, out: &mut Vec<DocType>) -> io::Result<()> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let file = syn::parse_file(&src).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {e}", path.display()),
        )
    })?;
    let defaults = defaults::per_struct(&file);
    for item in &file.items {
        match item {
            syn::Item::Struct(s) => {
                let syn::Fields::Named(named) = &s.fields else {
                    continue;
                };
                let name = s.ident.to_string();
                let fields = struct_fields(named, defaults.get(&name));
                out.push(DocType {
                    doc: extract_doc(&s.attrs),
                    name,
                    shape: DocShape::Fields(fields),
                });
            }
            syn::Item::Enum(e) => {
                let Some(values) = string_values(e) else {
                    continue;
                };
                out.push(DocType {
                    name: e.ident.to_string(),
                    doc: extract_doc(&e.attrs),
                    shape: DocShape::Values(values),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn struct_fields(
    named: &syn::FieldsNamed,
    defaults: Option<&HashMap<String, String>>,
) -> Vec<DocField> {
    let mut out = Vec::new();
    for field in &named.named {
        if has_serde_skip(&field.attrs) {
            continue;
        }
        let Some(ident) = field.ident.as_ref().map(syn::Ident::to_string) else {
            continue;
        };
        let (ty, optional) = match option_inner(&field.ty) {
            Some(inner) => (map_type(inner), true),
            None => (map_type(&field.ty), false),
        };
        out.push(DocField {
            key: serde_kv(&field.attrs, "rename").unwrap_or_else(|| ident.clone()),
            doc: collapse_doc(&extract_doc(&field.attrs)),
            ty,
            optional,
            default: defaults
                .and_then(|d| d.get(&ident))
                .filter(|d| d.as_str() != UNKNOWN)
                .cloned(),
        });
    }
    out
}

// The serialized values of an enum that serializes to a plain string. A
// data-carrying enum becomes a JSON object, so it is not one of these.
fn string_values(e: &syn::ItemEnum) -> Option<Vec<DocValue>> {
    if e.variants
        .iter()
        .any(|v| !matches!(v.fields, syn::Fields::Unit))
    {
        return None;
    }
    let rule = serde_kv(&e.attrs, "rename_all");
    Some(
        e.variants
            .iter()
            .map(|v| DocValue {
                value: serde_kv(&v.attrs, "rename")
                    .unwrap_or_else(|| apply_case(&v.ident.to_string(), rule.as_deref())),
                doc: collapse_doc(&extract_doc(&v.attrs)),
            })
            .collect(),
    )
}

// Translate a Rust field type to its JSON shape. An ident this crate cannot
// decide on its own stays a `Name`: whether it is an enum, another asset, or a
// nested object depends on types the sources under one root cannot all see.
fn map_type(ty: &syn::Type) -> DocFieldType {
    match ty {
        syn::Type::Path(tp) => {
            let Some(seg) = tp.path.segments.last() else {
                return DocFieldType::Object;
            };
            let id = seg.ident.to_string();
            match id.as_str() {
                "f32" | "f64" => DocFieldType::Float,
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize" => {
                    DocFieldType::Integer
                }
                "bool" => DocFieldType::Bool,
                // `AssetId` and the per-kind resource handles are authored as a
                // by-name reference string (an already-resolved integer is the
                // compiled form), so they document as a string like any other
                // cross-reference field.
                "String" | "AssetId" => DocFieldType::Str,
                "TextureHandle"
                | "MeshHandle"
                | "MaterialHandle"
                | "FontHandle"
                | "AudioClipHandle"
                | "CubemapTextureHandle"
                | "EnvironmentMapHandle"
                | "ColorLutHandle"
                | "SkinnedMeshHandle" => DocFieldType::Str,
                // serde_json::Value and maps are open-ended JSON objects.
                "Value" | "HashMap" | "BTreeMap" => DocFieldType::Object,
                "Option" | "Box" => first_generic(seg)
                    .map(map_type)
                    .unwrap_or(DocFieldType::Object),
                "Vec" => DocFieldType::Array {
                    elem: Box::new(
                        first_generic(seg)
                            .map(map_type)
                            .unwrap_or(DocFieldType::Object),
                    ),
                    len: None,
                },
                _ => DocFieldType::Name(id),
            }
        }
        syn::Type::Array(arr) => DocFieldType::Array {
            elem: Box::new(map_type(&arr.elem)),
            len: array_len(&arr.len),
        },
        syn::Type::Reference(r) => map_type(&r.elem),
        _ => DocFieldType::Object,
    }
}

fn first_generic(seg: &syn::PathSegment) -> Option<&syn::Type> {
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    args.args.iter().find_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

fn option_inner(ty: &syn::Type) -> Option<&syn::Type> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
        && seg.ident == "Option"
    {
        return first_generic(seg);
    }
    None
}

fn array_len(expr: &syn::Expr) -> Option<usize> {
    if let syn::Expr::Lit(l) = expr
        && let syn::Lit::Int(i) = &l.lit
    {
        return i.base10_parse::<usize>().ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ty_of(src: &str) -> DocFieldType {
        let file: syn::File = syn::parse_str(&format!("struct A {{ f: {src} }}")).expect("parse");
        let syn::Item::Struct(s) = &file.items[0] else {
            panic!("expected a struct");
        };
        map_type(&s.fields.iter().next().expect("one field").ty)
    }

    fn array(elem: DocFieldType, len: Option<usize>) -> DocFieldType {
        DocFieldType::Array {
            elem: Box::new(elem),
            len,
        }
    }

    #[test]
    fn scalars_map_to_their_json_shapes() {
        assert_eq!(ty_of("bool"), DocFieldType::Bool);
        assert_eq!(ty_of("f32"), DocFieldType::Float);
        assert_eq!(ty_of("u16"), DocFieldType::Integer);
        assert_eq!(ty_of("String"), DocFieldType::Str);
        assert_eq!(ty_of("serde_json::Value"), DocFieldType::Object);
    }

    #[test]
    fn references_by_name_map_to_strings() {
        assert_eq!(ty_of("AssetId"), DocFieldType::Str);
        assert_eq!(ty_of("TextureHandle"), DocFieldType::Str);
    }

    #[test]
    fn containers_unwrap_to_arrays_and_elements() {
        assert_eq!(ty_of("Vec<f32>"), array(DocFieldType::Float, None));
        assert_eq!(ty_of("[f32; 4]"), array(DocFieldType::Float, Some(4)));
        assert_eq!(
            ty_of("Vec<[f32; 2]>"),
            array(array(DocFieldType::Float, Some(2)), None)
        );
        assert_eq!(ty_of("Box<bool>"), DocFieldType::Bool);
    }

    #[test]
    fn an_unrecognised_ident_stays_unresolved() {
        assert_eq!(
            ty_of("PropCollider"),
            DocFieldType::Name("PropCollider".to_string())
        );
        assert_eq!(
            ty_of("Vec<WaterWave>"),
            array(DocFieldType::Name("WaterWave".to_string()), None)
        );
    }

    fn parse_types(src: &str) -> Vec<DocType> {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.rs"), src).expect("write");
        std::fs::write(dir.path().join("notes.md"), "not rust").expect("write");
        types(&[dir.path().to_path_buf()], &[]).expect("extract")
    }

    #[test]
    fn a_struct_carries_its_doc_fields_and_defaults() {
        let out = parse_types(
            r#"
            /// A prop.
            ///
            /// More prose.
            pub struct Prop {
                /// The mesh
                /// to draw.
                pub mesh: String,
                #[serde(rename = "type")]
                pub kind: u32,
                #[serde(skip)]
                pub cached: u32,
                pub collider: Option<PropCollider>,
            }
            impl Default for Prop {
                fn default() -> Self {
                    Self { mesh: "cube".to_string(), kind: 3, collider: None }
                }
            }
            "#,
        );
        assert_eq!(out.len(), 1);
        let t = &out[0];
        assert_eq!(t.name, "Prop");
        assert_eq!(t.doc, "A prop.\n\nMore prose.");
        let DocShape::Fields(fields) = &t.shape else {
            panic!("expected fields");
        };
        assert_eq!(fields.len(), 3, "the skipped field is dropped");
        assert_eq!(fields[0].key, "mesh");
        assert_eq!(fields[0].doc, "The mesh to draw.");
        assert_eq!(fields[0].default.as_deref(), Some("\"cube\""));
        assert_eq!(fields[1].key, "type", "the serde rename is the key");
        assert!(!fields[1].optional);
        assert!(fields[2].optional);
        assert_eq!(fields[2].default.as_deref(), Some("null"));
    }

    #[test]
    fn a_string_enum_carries_its_renamed_values() {
        let out = parse_types(
            r#"
            /// How to shade.
            #[serde(rename_all = "snake_case")]
            pub enum ShaderKind {
                /// Per vertex.
                VertexInstanced,
                #[serde(rename = "frag")]
                Fragment,
            }
            "#,
        );
        let DocShape::Values(values) = &out[0].shape else {
            panic!("expected values");
        };
        assert_eq!(values[0].value, "vertex_instanced");
        assert_eq!(values[0].doc, "Per vertex.");
        assert_eq!(values[1].value, "frag");
    }

    #[test]
    fn data_carrying_enums_and_tuple_structs_are_skipped() {
        let out = parse_types(
            "pub enum Shape { Circle(f32) }
             pub struct Wrapper(u32);
             pub struct Unit;",
        );
        assert!(out.is_empty());
    }

    #[test]
    fn excluded_files_contribute_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let skipped = dir.path().join("skip.rs");
        std::fs::write(&skipped, "pub struct Gone { pub a: u32 }").expect("write");
        std::fs::write(dir.path().join("keep.rs"), "pub struct Kept { pub a: u32 }")
            .expect("write");
        let out = types(&[dir.path().to_path_buf()], &[skipped]).expect("extract");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "Kept");
    }

    #[test]
    fn types_come_back_sorted_by_name() {
        let out = parse_types("pub struct Zeta { pub a: u32 } pub struct Alpha { pub a: u32 }");
        let names: Vec<&str> = out.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["Alpha", "Zeta"]);
    }
}
