// world.jsonl I/O: parsing, serialization, and file-patch utilities. This is
// authoring input handling -- the compile pipeline and the editor read and
// rewrite world.jsonl through here. The shipped runtime plays compiled blobs
// and never touches world.jsonl, so this lives in the build crate, not core.

use super::identity::{ID_KEY, entry_handles, is_label_of};
use crate::authoring::registry::RegisteredType;

/// An asset entry after $include resolution and type parsing.
#[derive(Clone, Debug)]
pub struct WorldJsonlAsset {
    /// The asset's handle: the `$id` it declares, or the `<Type>#<ordinal>`
    /// label a loaded world gives an anonymous entry.
    pub id: String,
    /// The asset's registered type.
    pub asset_type: RegisteredType,
    /// The asset's authored args, without the `$id`.
    pub args: serde_json::Value,
}

impl WorldJsonlAsset {
    /// Build a typed entry from a loaded JSON asset object, moving its `$id`
    /// out of the args. Fails, naming the asset, when it carries no `$id` or
    /// `type` is not an exact registered name.
    pub fn from_value(v: &serde_json::Value) -> Result<Self, String> {
        let type_str = v.get("type").and_then(|t| t.as_str());
        let mut args = v
            .get("args")
            .cloned()
            .filter(|a| !a.is_null())
            .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
        let Some(id) = take_entry_id_from_args(&mut args) else {
            return Err(format!(
                "asset of type '{}': missing `{ID_KEY}`",
                type_str.unwrap_or("")
            ));
        };
        let Some(type_str) = type_str else {
            return Err(format!("'{id}': missing `type` field"));
        };
        let asset_type = RegisteredType::parse(type_str)
            .ok_or_else(|| format!("'{id}': unknown type '{type_str}'"))?;
        Ok(WorldJsonlAsset {
            id,
            asset_type,
            args,
        })
    }

    /// The asset as a world line: its args with the `$id` declared, or none
    /// for an anonymous asset, whose label is not something a line can declare.
    pub fn to_entry(&self) -> serde_json::Value {
        let args = if self.is_anonymous() {
            self.args.clone()
        } else {
            super::identity::args_with_id(self.args.clone(), &self.id)
        };
        serde_json::json!({"type": self.asset_type.as_str(), "args": args})
    }

    /// Whether the asset declares no `$id` of its own, so its handle is the
    /// label it was loaded under and nothing can reference it.
    pub fn is_anonymous(&self) -> bool {
        is_label_of(&self.id, self.asset_type.as_str())
    }
}

fn take_entry_id_from_args(args: &mut serde_json::Value) -> Option<String> {
    match args.as_object_mut()?.remove(ID_KEY)? {
        serde_json::Value::String(id) => Some(id),
        _ => None,
    }
}

/// Parse a world.jsonl string into a flat list of raw asset objects.
///
/// Each non-blank, non-comment line must be a valid JSON object. The order
/// of entries is preserved. Returns an error on the first malformed line.
pub fn parse_world_jsonl(content: &str) -> Result<Vec<serde_json::Value>, serde_json::Error> {
    let mut assets = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)?;
        assets.push(value);
    }
    Ok(assets)
}

/// Serialize a list of asset objects back to world.jsonl format.
///
/// Each entry is written as a compact single-line JSON object followed by a
/// newline. The result is a valid world.jsonl file.
pub fn write_world_jsonl(assets: &[serde_json::Value]) -> serde_json::Result<String> {
    let mut out = String::new();
    for asset in assets {
        out.push_str(&serde_json::to_string(asset)?);
        out.push('\n');
    }
    Ok(out)
}

/// Read src_path, apply a fallible mutation to the asset list, and write
/// the result to dst_path. src and dst may be the same path or different.
pub fn patch_world_jsonl_to<F>(src_path: &str, dst_path: &str, f: F) -> std::io::Result<()>
where
    F: FnOnce(&mut Vec<serde_json::Value>) -> std::io::Result<()>,
{
    let content = std::fs::read_to_string(src_path).map_err(|e| {
        tracing::error!("Could not read {}: {}", src_path, e);
        e
    })?;

    let mut assets = parse_world_jsonl(&content).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse {}: {}", src_path, e),
        )
    })?;

    f(&mut assets)?;

    let out = write_world_jsonl(&assets).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(dst_path, out)
}

/// Read world.jsonl at json_path, mutate the asset list in-place, write back.
pub fn patch_world_jsonl<F>(json_path: &str, f: F) -> std::io::Result<()>
where
    F: FnOnce(&mut Vec<serde_json::Value>),
{
    patch_world_jsonl_to(json_path, json_path, |assets| {
        f(assets);
        Ok(())
    })
}

/// Read every entry's handle from world.jsonl without a full parse, for error
/// messages: its `$id`, else its `<Type>#<ordinal>` label.
pub fn known_names(json_path: &str) -> std::io::Result<Vec<String>> {
    let content = std::fs::read_to_string(json_path)?;
    let assets = parse_world_jsonl(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(entry_handles(&assets).into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_value_parses_an_exact_type() {
        let asset = WorldJsonlAsset::from_value(
            &serde_json::json!({"type": "Prop", "args": {"$id": "p", "mesh": "m"}}),
        )
        .unwrap();
        assert_eq!(asset.id, "p");
        assert_eq!(asset.asset_type, RegisteredType::Prop);
        assert_eq!(asset.args["mesh"], "m");
    }

    #[test]
    fn from_value_defaults_missing_args_to_an_empty_object() {
        let asset = WorldJsonlAsset::from_value(
            &serde_json::json!({"type": "Window", "args": {"$id": "w"}}),
        )
        .unwrap();
        assert_eq!(asset.args, serde_json::json!({}));
    }

    // A loaded world gives every entry a `$id`, so one without is not loaded.
    #[test]
    fn from_value_rejects_a_missing_id() {
        let err = WorldJsonlAsset::from_value(&serde_json::json!({"type": "Prop"})).unwrap_err();
        assert!(err.contains("missing `$id`"), "{err}");
    }

    #[test]
    fn from_value_moves_the_id_out_of_the_args() {
        let asset = WorldJsonlAsset::from_value(
            &serde_json::json!({"type": "Prop", "args": {"$id": "p", "mesh": "m"}}),
        )
        .unwrap();
        assert_eq!(asset.args, serde_json::json!({"mesh": "m"}));
        assert!(!asset.is_anonymous());
    }

    #[test]
    fn a_label_of_its_own_type_marks_the_asset_anonymous() {
        let labeled = |ty: &str, id: &str| {
            WorldJsonlAsset::from_value(&serde_json::json!({"type": ty, "args": {"$id": id}}))
                .unwrap()
        };
        assert!(labeled("Prop", "Prop#3").is_anonymous());
        assert!(!labeled("Screen", "MainMenu#0").is_anonymous());
        assert!(!labeled("Prop", "Prop#3_body").is_anonymous());
    }

    #[test]
    fn to_entry_declares_the_id_unless_anonymous() {
        let asset = |id: &str| {
            WorldJsonlAsset::from_value(
                &serde_json::json!({"type": "Prop", "args": {"$id": id, "mesh": "m"}}),
            )
            .unwrap()
        };
        assert_eq!(
            asset("crate").to_entry(),
            serde_json::json!({"type": "Prop", "args": {"$id": "crate", "mesh": "m"}})
        );
        assert_eq!(
            asset("Prop#2").to_entry(),
            serde_json::json!({"type": "Prop", "args": {"mesh": "m"}})
        );
    }

    #[test]
    fn from_value_rejects_a_missing_type() {
        let err =
            WorldJsonlAsset::from_value(&serde_json::json!({"args": {"$id": "p"}})).unwrap_err();
        assert!(
            err.contains("'p'") && err.contains("missing `type`"),
            "{err}"
        );
    }

    #[test]
    fn from_value_rejects_an_unknown_type() {
        let err = WorldJsonlAsset::from_value(
            &serde_json::json!({"type": "Gizmo", "args": {"$id": "p"}}),
        )
        .unwrap_err();
        assert!(err.contains("'p'") && err.contains("'Gizmo'"), "{err}");
    }

    #[test]
    fn from_value_rejects_inexact_spellings() {
        for ty in ["prop", "PROP", "color_lut", "Color_Lut", "colorlut"] {
            let err =
                WorldJsonlAsset::from_value(&serde_json::json!({"type": ty, "args": {"$id": "a"}}))
                    .unwrap_err();
            assert!(err.contains("'a'") && err.contains(ty), "{ty}: {err}");
        }
    }

    #[test]
    fn parse_world_jsonl_empty_string_returns_empty() {
        let assets = parse_world_jsonl("").unwrap();
        assert!(assets.is_empty());
    }

    #[test]
    fn parse_world_jsonl_skips_blank_and_comment_lines() {
        let content = "\n  \n// this is a comment\n";
        let assets = parse_world_jsonl(content).unwrap();
        assert!(assets.is_empty());
    }

    #[test]
    fn parse_world_jsonl_returns_entries_in_order() {
        let content = r#"{"type":"Logger","args":{"$id":"a"}}
{"type":"Window","args":{"$id":"b"}}
"#;
        let assets = parse_world_jsonl(content).unwrap();
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["args"]["$id"], "a");
        assert_eq!(assets[1]["args"]["$id"], "b");
    }

    #[test]
    fn parse_world_jsonl_errors_on_invalid_json() {
        let result = parse_world_jsonl("{not valid json}");
        assert!(result.is_err());
    }

    #[test]
    fn write_world_jsonl_one_line_per_entry() {
        let assets = vec![
            serde_json::json!({"type": "Logger", "args": {"$id": "a"}}),
            serde_json::json!({"type": "Window", "args": {"$id": "b"}}),
        ];
        let out = write_world_jsonl(&assets).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"a\""));
        assert!(lines[1].contains("\"b\""));
    }

    #[test]
    fn write_world_jsonl_round_trips_through_parse() {
        let assets = vec![serde_json::json!({"type": "Logger", "args": {"$id": "x"}})];
        let out = write_world_jsonl(&assets).unwrap();
        let parsed = parse_world_jsonl(&out).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["args"]["$id"], "x");
    }

    #[test]
    fn patch_world_jsonl_to_applies_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("world.jsonl");
        let dst = dir.path().join("out.jsonl");
        std::fs::write(&src, "{\"type\":\"Logger\",\"args\":{\"$id\":\"a\"}}\n").unwrap();

        patch_world_jsonl_to(src.to_str().unwrap(), dst.to_str().unwrap(), |assets| {
            assets.push(serde_json::json!({"type":"Window","args":{"$id":"b"}}));
            Ok(())
        })
        .unwrap();

        let content = std::fs::read_to_string(&dst).unwrap();
        let parsed = parse_world_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1]["args"]["$id"], "b");
    }

    #[test]
    fn patch_world_jsonl_to_propagates_mutation_error() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("world.jsonl");
        std::fs::write(&src, "{\"type\":\"Logger\",\"args\":{\"$id\":\"a\"}}\n").unwrap();

        let result =
            patch_world_jsonl_to(src.to_str().unwrap(), src.to_str().unwrap(), |_assets| {
                Err(std::io::Error::other("boom"))
            });
        assert!(result.is_err());
    }

    #[test]
    fn known_names_lists_ids_and_labels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(
            &path,
            "{\"type\":\"Logger\",\"args\":{\"$id\":\"a\"}}\n{\"type\":\"Window\"}\n",
        )
        .unwrap();
        let names = known_names(path.to_str().unwrap()).unwrap();
        assert_eq!(names, vec!["a", "Window#0"]);
    }

    #[test]
    fn known_names_empty_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, "").unwrap();
        let names = known_names(path.to_str().unwrap()).unwrap();
        assert!(names.is_empty());
    }
}
