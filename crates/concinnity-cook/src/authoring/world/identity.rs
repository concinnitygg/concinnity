// An entry's identity: the optional `$id` it declares in its args, or, for an
// anonymous entry, the `<Type>#<ordinal>` label it is addressed by wherever a
// handle has to be printed or typed.

use serde_json::Value;

/// The args key an entry declares its identity under. The `$` keeps it apart
/// from every schema field, so no asset type can ever collide with it.
pub const ID_KEY: &str = "$id";

/// The `$id` an entry declares, or `None` for an anonymous entry.
pub fn entry_id(entry: &Value) -> Option<&str> {
    entry.get("args")?.get(ID_KEY)?.as_str()
}

/// Declare `id` as the entry's identity, creating its args object when the
/// entry has none. An entry whose args are not an object is left unchanged.
pub fn set_entry_id(entry: &mut Value, id: &str) {
    let Some(obj) = entry.as_object_mut() else {
        return;
    };
    let args = obj
        .entry("args")
        .or_insert_with(|| Value::Object(Default::default()));
    if args.is_null() {
        *args = Value::Object(Default::default());
    }
    if let Some(args) = args.as_object_mut() {
        args.insert(ID_KEY.to_string(), Value::String(id.to_string()));
    }
}

/// `args` with `id` declared as its `$id`: the args of an entry that has not
/// been assembled yet. Args that are not an object come back unchanged.
pub fn args_with_id(mut args: Value, id: &str) -> Value {
    if args.is_null() {
        args = Value::Object(Default::default());
    }
    if let Some(obj) = args.as_object_mut() {
        obj.insert(ID_KEY.to_string(), Value::String(id.to_string()));
    }
    args
}

/// Replace the entry's args with `args`, keeping the `$id` it declares: an
/// edit to what an asset is never changes which asset it is.
pub fn replace_args(entry: &mut Value, args: Value) {
    let id = entry_id(entry).map(str::to_string);
    let Some(obj) = entry.as_object_mut() else {
        return;
    };
    let args = match id {
        Some(id) => args_with_id(args, &id),
        None => args,
    };
    obj.insert("args".to_string(), args);
}

/// `args` without its `$id`: what a schema or an args editor reads.
pub fn args_without_id(mut args: Value) -> Value {
    if let Some(obj) = args.as_object_mut() {
        obj.remove(ID_KEY);
    }
    args
}

/// Remove the entry's `$id`, leaving it anonymous. Returns the id it declared.
pub fn take_entry_id(entry: &mut Value) -> Option<String> {
    let args = entry.get_mut("args")?.as_object_mut()?;
    match args.remove(ID_KEY)? {
        Value::String(id) => Some(id),
        _ => None,
    }
}

/// The handle an anonymous `ty` entry is addressed by: the type and the
/// entry's position among the anonymous entries of that type, e.g. `Prop#3`.
pub fn anonymous_label(ty: &str, ordinal: usize) -> String {
    format!("{ty}#{ordinal}")
}

/// Whether `handle` is the label of an anonymous `ty` entry rather than an
/// identity something declared.
pub fn is_label_of(handle: &str, ty: &str) -> bool {
    handle
        .strip_prefix(ty)
        .and_then(|rest| rest.strip_prefix('#'))
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Every entry's handle, in order: its `$id`, else its anonymous label. An
/// entry with no type string (an `$include`, a malformed line) has none.
pub fn entry_handles(entries: &[Value]) -> Vec<Option<String>> {
    let mut ordinals: std::collections::HashMap<&str, usize> = Default::default();
    entries
        .iter()
        .map(|entry| {
            if let Some(id) = entry_id(entry) {
                return Some(id.to_string());
            }
            let ty = entry.get("type")?.as_str()?;
            let ordinal = ordinals.entry(ty).or_insert(0);
            let label = anonymous_label(ty, *ordinal);
            *ordinal += 1;
            Some(label)
        })
        .collect()
}

/// The handle of the entry at `index`, as [`entry_handles`] would give it.
pub fn entry_handle(entries: &[Value], index: usize) -> Option<String> {
    let entry = entries.get(index)?;
    if let Some(id) = entry_id(entry) {
        return Some(id.to_string());
    }
    let ty = entry.get("type")?.as_str()?;
    let ordinal = entries[..index]
        .iter()
        .filter(|e| entry_id(e).is_none() && e.get("type").and_then(Value::as_str) == Some(ty))
        .count();
    Some(anonymous_label(ty, ordinal))
}

/// The position of the entry `handle` addresses: the entry declaring it as
/// its `$id`, or the anonymous entry it labels.
pub fn find_entry(entries: &[Value], handle: &str) -> Option<usize> {
    if let Some(i) = entries.iter().position(|e| entry_id(e) == Some(handle)) {
        return Some(i);
    }
    let (ty, _) = handle.split_once('#')?;
    if !is_label_of(handle, ty) {
        return None;
    }
    entry_handles(entries)
        .iter()
        .position(|h| h.as_deref() == Some(handle))
}

// The structural rules on a declared `$id`: a non-empty string not shaped like
// a label, which is reserved for anonymous entries so a handle names one entry.
// A `#` elsewhere is allowed: it is how the assets an anonymous entry expands to
// are named (`MainMenu#0_title`), and a line patching one declares that name.
pub(crate) fn check_declared_id(value: &Value) -> Result<&str, String> {
    let Some(id) = value.as_str() else {
        return Err(format!("`{ID_KEY}` must be a string"));
    };
    if id.is_empty() {
        return Err(format!("`{ID_KEY}` must not be empty"));
    }
    if id
        .split_once('#')
        .is_some_and(|(ty, _)| is_label_of(id, ty))
    {
        return Err(format!(
            "`{ID_KEY}` '{id}' is shaped like `<Type>#<ordinal>`, which is reserved for the \
             handle of an anonymous asset"
        ));
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn entry_id_reads_the_declared_id_from_args() {
        assert_eq!(
            entry_id(&json!({"type": "Prop", "args": {"$id": "crate"}})),
            Some("crate")
        );
        assert_eq!(entry_id(&json!({"type": "Prop", "args": {}})), None);
        assert_eq!(entry_id(&json!({"type": "Prop"})), None);
        assert_eq!(entry_id(&json!({"type": "Prop", "args": {"$id": 3}})), None);
    }

    #[test]
    fn set_entry_id_creates_missing_or_null_args() {
        let mut v = json!({"type": "Prop"});
        set_entry_id(&mut v, "a");
        assert_eq!(v["args"]["$id"], "a");
        let mut v = json!({"type": "Prop", "args": null});
        set_entry_id(&mut v, "b");
        assert_eq!(v["args"]["$id"], "b");
        let mut v = json!({"type": "Prop", "args": {"mesh": "m"}});
        set_entry_id(&mut v, "c");
        assert_eq!(v["args"], json!({"mesh": "m", "$id": "c"}));
    }

    #[test]
    fn args_with_id_declares_the_id_in_an_object() {
        assert_eq!(
            args_with_id(json!({"mesh": "m"}), "a"),
            json!({"mesh": "m", "$id": "a"})
        );
        assert_eq!(args_with_id(Value::Null, "a"), json!({"$id": "a"}));
        assert_eq!(args_with_id(json!([]), "a"), json!([]));
    }

    #[test]
    fn replace_args_keeps_the_declared_id() {
        let mut v = json!({"type": "Prop", "args": {"$id": "a", "mesh": "m"}});
        replace_args(&mut v, json!({"mesh": "n"}));
        assert_eq!(v["args"], json!({"mesh": "n", "$id": "a"}));
        let mut anon = json!({"type": "Prop", "args": {"mesh": "m"}});
        replace_args(&mut anon, json!({"mesh": "n"}));
        assert_eq!(anon["args"], json!({"mesh": "n"}));
        assert_eq!(
            args_without_id(json!({"$id": "a", "mesh": "m"})),
            json!({"mesh": "m"})
        );
    }

    #[test]
    fn take_entry_id_leaves_the_entry_anonymous() {
        let mut v = json!({"type": "Prop", "args": {"$id": "a", "mesh": "m"}});
        assert_eq!(take_entry_id(&mut v).as_deref(), Some("a"));
        assert_eq!(v["args"], json!({"mesh": "m"}));
        assert_eq!(take_entry_id(&mut v), None);
    }

    #[test]
    fn labels_count_the_anonymous_entries_of_each_type() {
        let entries = vec![
            json!({"type": "Prop", "args": {}}),
            json!({"type": "Prop", "args": {"$id": "named"}}),
            json!({"type": "PointLight"}),
            json!({"type": "Prop", "args": {}}),
            json!({"$include": "x.json"}),
        ];
        let handles = entry_handles(&entries);
        assert_eq!(
            handles,
            [
                Some("Prop#0".to_string()),
                Some("named".to_string()),
                Some("PointLight#0".to_string()),
                Some("Prop#1".to_string()),
                None,
            ]
        );
        for (i, h) in handles.iter().enumerate() {
            assert_eq!(&entry_handle(&entries, i), h, "entry {i}");
            if let Some(h) = h {
                assert_eq!(find_entry(&entries, h), Some(i), "{h}");
            }
        }
        assert_eq!(find_entry(&entries, "Prop#2"), None);
        assert_eq!(find_entry(&entries, "ghost"), None);
    }

    #[test]
    fn is_label_of_matches_only_the_exact_label_shape() {
        assert!(is_label_of("Prop#0", "Prop"));
        assert!(is_label_of("Prop#12", "Prop"));
        assert!(!is_label_of("Prop#", "Prop"));
        assert!(!is_label_of("Prop#1_body", "Prop"));
        assert!(!is_label_of("Prop#1", "PointLight"));
        assert!(!is_label_of("MainMenu#0", "Screen"));
        assert!(!is_label_of("Prop1", "Prop"));
    }

    #[test]
    fn check_declared_id_rejects_non_strings_empties_and_labels() {
        assert_eq!(check_declared_id(&json!("crate")), Ok("crate"));
        assert!(check_declared_id(&json!(4)).unwrap_err().contains("string"));
        assert!(check_declared_id(&json!("")).unwrap_err().contains("empty"));
        assert!(
            check_declared_id(&json!("Prop#1"))
                .unwrap_err()
                .contains("reserved")
        );
        // A generated name under a label is not itself a label.
        assert_eq!(
            check_declared_id(&json!("MainMenu#0_title")),
            Ok("MainMenu#0_title")
        );
    }
}
