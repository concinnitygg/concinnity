// src/shader_layout/reflect.rs
//
// slangc's `-reflection-json` for one program, reduced to the byte layout of
// every struct it declares. The JSON nests a struct wherever it is used -- as a
// constant buffer's element, a structured buffer's record, or the type of
// another struct's field -- so the reader walks the whole tree and keys what it
// finds by struct name.
//
// A struct's fields carry `{"kind": "uniform", "offset", "size"}`; a constant
// buffer additionally states the block size its element occupies, which a
// structured-buffer record has no equivalent for. Vertex inputs carry
// `varyingInput` bindings with attribute indices instead of byte offsets, so
// they never appear here.

use std::collections::BTreeMap;

use serde_json::Value;

// One member of a shader struct, as slangc lays it out for the target.
pub(super) struct ShaderField {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

// One shader struct's layout for one target.
pub(super) struct ShaderStruct {
    pub fields: Vec<ShaderField>,
    // Block size slangc reports when the struct is a constant buffer's element.
    // Absent when the struct is only ever a structured-buffer record, which the
    // reflection states no total for.
    pub block_size: Option<usize>,
}

impl ShaderStruct {
    // Byte past the last declared member. What the struct occupies before any
    // target-specific rounding of the block it sits in.
    pub fn extent(&self) -> usize {
        self.fields
            .iter()
            .map(|f| f.offset + f.size)
            .max()
            .unwrap_or(0)
    }
}

// Every struct the reflection names, keyed by its shader-side name.
pub(super) fn structs(json: &str) -> Result<BTreeMap<String, ShaderStruct>, String> {
    let root: Value = serde_json::from_str(json).map_err(|e| format!("reflection json: {e}"))?;
    let mut fields = BTreeMap::new();
    let mut blocks = BTreeMap::new();
    collect(&root, &mut fields, &mut blocks);
    Ok(fields
        .into_iter()
        .map(|(name, fields)| {
            let block_size = blocks.get(&name).copied();
            (name, ShaderStruct { fields, block_size })
        })
        .collect())
}

// A struct is declared where it is used, and the constant buffer that states
// its block size wraps it, so the two are gathered separately and joined after
// the walk rather than in visit order.
fn collect(
    node: &Value,
    fields: &mut BTreeMap<String, Vec<ShaderField>>,
    blocks: &mut BTreeMap<String, usize>,
) {
    match node {
        Value::Object(map) => {
            if let Some(name) = struct_name(map) {
                let members = members(map);
                if !members.is_empty() {
                    fields.entry(name).or_insert(members);
                }
            }
            if let Some((name, size)) = block(map) {
                blocks.entry(name).or_insert(size);
            }
            for value in map.values() {
                collect(value, fields, blocks);
            }
        }
        Value::Array(items) => {
            for value in items {
                collect(value, fields, blocks);
            }
        }
        _ => {}
    }
}

// The struct name of a `{"kind": "struct", "name": ..., "fields": [...]}` node.
fn struct_name(map: &serde_json::Map<String, Value>) -> Option<String> {
    (map.get("kind")?.as_str()? == "struct")
        .then(|| map.get("name")?.as_str().map(str::to_string))
        .flatten()
}

// The uniform-bound members of a struct node, in declaration order.
fn members(map: &serde_json::Map<String, Value>) -> Vec<ShaderField> {
    let Some(fields) = map.get("fields").and_then(Value::as_array) else {
        return Vec::new();
    };
    fields
        .iter()
        .filter_map(|field| {
            let binding = field.get("binding")?;
            (binding.get("kind")?.as_str()? == "uniform").then_some(())?;
            Some(ShaderField {
                name: field.get("name")?.as_str()?.to_string(),
                offset: usize::try_from(binding.get("offset")?.as_u64()?).ok()?,
                size: usize::try_from(binding.get("size")?.as_u64()?).ok()?,
            })
        })
        .collect()
}

// A `constantBuffer` node states the block size of its struct element. The size
// is the target's, not the language's: Metal and SPIR-V round a 276-byte block
// up to 288 where the DirectX leg reports the unrounded extent.
fn block(map: &serde_json::Map<String, Value>) -> Option<(String, usize)> {
    (map.get("kind")?.as_str()? == "constantBuffer").then_some(())?;
    let element = map.get("elementVarLayout")?;
    let ty = element.get("type")?;
    (ty.get("kind")?.as_str()? == "struct").then_some(())?;
    let name = ty.get("name")?.as_str()?.to_string();
    let size = usize::try_from(element.get("binding")?.get("size")?.as_u64()?).ok()?;
    Some((name, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "parameters": [{
            "name": "cb",
            "type": {
                "kind": "constantBuffer",
                "elementVarLayout": {
                    "type": {
                        "kind": "struct",
                        "name": "Params",
                        "fields": [
                            {"name": "a", "type": {"kind": "scalar"},
                             "binding": {"kind": "uniform", "offset": 0, "size": 12}},
                            {"name": "b", "type": {"kind": "scalar"},
                             "binding": {"kind": "uniform", "offset": 12, "size": 4}}
                        ]
                    },
                    "binding": {"kind": "uniform", "offset": 0, "size": 32}
                }
            }
        }, {
            "name": "vin",
            "type": {
                "kind": "struct",
                "name": "VertexIn",
                "fields": [
                    {"name": "pos", "binding": {"kind": "varyingInput", "index": 0}}
                ]
            }
        }]
    }"#;

    #[test]
    fn a_constant_buffer_yields_its_fields_and_block_size() {
        let found = structs(SAMPLE).expect("parse");
        let params = found.get("Params").expect("Params reflected");
        assert_eq!(params.block_size, Some(32));
        assert_eq!(params.extent(), 16);
        let names: Vec<_> = params.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(params.fields[1].offset, 12);
        assert_eq!(params.fields[1].size, 4);
    }

    // Vertex inputs bind by attribute index, not byte offset, so they carry no
    // layout to compare a payload struct against.
    #[test]
    fn a_varying_input_struct_is_not_reflected_as_a_layout() {
        let found = structs(SAMPLE).expect("parse");
        assert!(!found.contains_key("VertexIn"));
    }

    #[test]
    fn malformed_json_reports_rather_than_panics() {
        assert!(structs("{ not json").is_err());
    }
}
