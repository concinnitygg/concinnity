use alloc::string::String;
use alloc::vec::Vec;

use super::*;
use crate::components::{SpriteFit, TextLabel, Vocabulary};
use crate::ecs::{AssetFields, Ref, TextureHandle};

/// A leaf with defaults of its own.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, AssetFields)]
#[serde(default)]
struct Item {
    /// Shown on the item.
    label: String,
    weight: f32,
}

impl Default for Item {
    fn default() -> Self {
        Self {
            label: String::from("item"),
            weight: 2.5,
        }
    }
}

fn seven() -> u32 {
    7
}

/// The top.
///
/// Second paragraph.
#[derive(AssetFields)]
#[expect(dead_code, reason = "only the schema is read")]
struct Top {
    #[serde(skip)]
    cache: u32,
    /// Picked by type.
    #[serde(rename = "type")]
    kind: SpriteFit,
    items: Vec<Item>,
    maybe: Option<[f32; 3]>,
    #[serde(default = "seven")]
    count: u32,
    #[serde(default)]
    flag: bool,
    #[serde(deserialize_with = "crate::ecs::de_opt_ref")]
    label: Option<Ref<TextLabel>>,
    texture: TextureHandle,
    table: alloc::collections::BTreeMap<String, f32>,
    #[serde(flatten)]
    inner: Item,
}

/// A vocabulary with serde's own spelling.
#[derive(
    Debug, Clone, Copy, serde::Serialize, serde::Deserialize, crate::components::Vocabulary,
)]
#[serde(rename_all = "snake_case")]
enum Pace {
    /// Constant speed.
    #[vocab("linear")]
    Linear,
    #[vocab("ease_in")]
    EaseIn,
}

// A type that derives the schema but not `Serialize`: its default states
// nothing.
#[derive(Default, AssetFields)]
#[serde(default)]
#[expect(dead_code, reason = "only the schema is read")]
struct Unserialized {
    n: u32,
}

fn fields(schema: &TypeSchema) -> &'static [FieldSchema] {
    match schema.body {
        Body::Fields(fields) => fields,
        Body::Values(_) => panic!("{} is a struct", schema.name),
    }
}

fn field(schema: &TypeSchema, key: &str) -> FieldSchema {
    *fields(schema)
        .iter()
        .find(|f| f.key == key)
        .unwrap_or_else(|| panic!("{} has no {key}", schema.name))
}

fn json(value: Option<DefaultValue>) -> serde_json::Value {
    let value = value.expect("the default serializes");
    serde_json::to_value(&*value).expect("the default serializes to JSON")
}

#[test]
fn a_struct_records_its_name_doc_and_authored_keys() {
    let schema = Top::SCHEMA;
    assert_eq!(schema.name, "Top");
    assert_eq!(schema.doc, "The top.\n\nSecond paragraph.");
    let keys: Vec<&str> = fields(schema).iter().map(|f| f.key).collect();
    assert_eq!(
        keys,
        [
            "type", "items", "maybe", "count", "flag", "label", "texture", "table", ""
        ]
    );
    assert_eq!(field(schema, "type").doc, "Picked by type.");
    assert!(field(schema, "").is_flattened());
}

#[test]
fn field_types_are_described_in_json_terms() {
    let schema = Top::SCHEMA;
    let ty = |key| (field(schema, key).ty)();

    let FieldType::Enum(pace) = ty("type") else {
        panic!("a vocabulary field is an enum");
    };
    assert_eq!(pace.name, "SpriteFit");

    let FieldType::Array { elem, len: None } = ty("items") else {
        panic!("a Vec is an unsized array");
    };
    let FieldType::Nested(item) = elem() else {
        panic!("a derived element type is nested");
    };
    assert_eq!(item.name, "Item");

    let FieldType::Optional(inner) = ty("maybe") else {
        panic!("an Option is optional");
    };
    assert!(matches!(inner(), FieldType::Array { len: Some(3), .. }));

    let FieldType::Optional(inner) = ty("label") else {
        panic!("an Option is optional");
    };
    assert!(matches!(inner(), FieldType::Reference(["TextLabel"])));
    assert!(matches!(ty("texture"), FieldType::Reference(["Texture"])));
    assert!(matches!(ty("count"), FieldType::Integer));
    assert!(matches!(ty("flag"), FieldType::Bool));
    // A type with no description is an object of no described shape.
    assert!(matches!(ty("table"), FieldType::Object));
}

// serde's precedence: the field's own default, then the container's, then
// `None` for an `Option`; anything else must be present.
#[test]
fn an_omitted_key_reads_as_serde_would_read_it() {
    let schema = Top::SCHEMA;
    assert!(schema.default.is_none(), "Top has no container default");
    let FieldDefault::Value(count) = field(schema, "count").default else {
        panic!("`default = \"seven\"` is the field's own");
    };
    assert_eq!(json(count()), serde_json::json!(7));
    let FieldDefault::Value(flag) = field(schema, "flag").default else {
        panic!("a bare `default` is the type's own");
    };
    assert_eq!(json(flag()), serde_json::json!(false));
    assert!(matches!(field(schema, "maybe").default, FieldDefault::Null));
    assert!(matches!(
        field(schema, "items").default,
        FieldDefault::Required
    ));
    // A `deserialize_with` reader keeps serde from treating a missing Option as
    // `None`.
    assert!(matches!(
        field(schema, "label").default,
        FieldDefault::Required
    ));

    for f in fields(Item::SCHEMA) {
        assert!(matches!(f.default, FieldDefault::Container));
    }
}

// An item type reached only through a `Vec` carries its own default, which
// the parent's default (an empty list) does not contain.
#[test]
fn a_nested_item_reaches_its_own_default() {
    let FieldType::Array { elem, .. } = (field(Top::SCHEMA, "items").ty)() else {
        panic!("items is an array");
    };
    let FieldType::Nested(item) = elem() else {
        panic!("its element is nested");
    };
    let default = item.default.expect("Item has a container default");
    assert_eq!(
        json(default()),
        serde_json::json!({ "label": "item", "weight": 2.5 })
    );
}

#[test]
fn a_default_that_cannot_serialize_states_nothing() {
    let default = Unserialized::SCHEMA.default.expect("a container default");
    assert!(default().is_none());
}

#[test]
fn a_vocabulary_records_each_name_with_its_doc() {
    let Body::Values(values) = Pace::SCHEMA.body else {
        panic!("an enum's schema holds values");
    };
    let named: Vec<(&str, &str)> = values.iter().map(|v| (v.name, v.doc)).collect();
    assert_eq!(named, [("linear", "Constant speed."), ("ease_in", "")]);
    assert_eq!(Pace::SCHEMA.doc, "A vocabulary with serde's own spelling.");
    assert!(matches!(<Pace as Described>::TYPE, FieldType::Enum(_)));
    // The name list, `as_str` and the trait all say what serde writes.
    assert_eq!(<Pace as Vocabulary>::VARIANTS, ["linear", "ease_in"]);
    for pace in Pace::ALL {
        assert_eq!(
            serde_json::to_value(pace).expect("serializes"),
            serde_json::json!(pace.as_str())
        );
    }
}

#[test]
fn scalars_and_named_references_are_described() {
    assert!(matches!(<f64 as Described>::TYPE, FieldType::Float));
    assert!(matches!(<i16 as Described>::TYPE, FieldType::Integer));
    assert!(matches!(<char as Described>::TYPE, FieldType::Str));
    assert!(matches!(
        <crate::ecs::asset_id::AssetId as Described>::TYPE,
        FieldType::Reference([])
    ));
}
