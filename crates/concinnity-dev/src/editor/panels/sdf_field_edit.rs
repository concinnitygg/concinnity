//! The data half of deleting a distance field from the Shaders panel: which of
//! its volumes the editor removes, what the dialog says, whether the file can
//! go too, and the toast after.

use super::sdf_field_list::FieldDecl;
use super::shader_edit::file_name;
use super::shader_list::ShaderDecl;
use super::shader_source::same_file;
use crate::editor::widget_check::Check;

// What deleting a field does: the volumes removed, and who still reads its
// file after, an included volume the editor cannot remove or a Shader
// declaring the same file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldDelete {
    pub(crate) removed: Vec<String>,
    pub(crate) readers: Vec<String>,
}

// The delete of `field`, where `read_only` says whether a volume is included
// from another file and `shaders` are the world's Shaders.
pub(crate) fn plan(
    field: &FieldDecl,
    read_only: impl Fn(&str) -> bool,
    shaders: &[ShaderDecl],
) -> FieldDelete {
    let (kept, removed): (Vec<&str>, Vec<&str>) =
        field.names().into_iter().partition(|name| read_only(name));
    let shader_readers = shaders
        .iter()
        .filter(|s| s.files.iter().any(|f| same_file(&f.path, &field.path)))
        .map(|s| s.name.as_str());
    FieldDelete {
        removed: removed.into_iter().map(str::to_string).collect(),
        readers: kept
            .into_iter()
            .chain(shader_readers)
            .map(str::to_string)
            .collect(),
    }
}

fn quoted(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    match quoted.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

// What the dialog asks.
pub(crate) fn delete_message(field: &FieldDecl, delete: &FieldDelete) -> String {
    let mut out = format!("Delete the SDF field {}?", field.declared);
    match delete.removed.len() {
        0 => {}
        1 => out.push_str(&format!(
            " Its volume {} is deleted.",
            quoted(&delete.removed)
        )),
        _ => out.push_str(&format!(
            " Its volumes {} are deleted.",
            quoted(&delete.removed)
        )),
    }
    out
}

// The dialog's "Also delete its files" box, off. It is disabled while
// anything the delete leaves still reads the file, and its note says who.
pub(crate) fn files_check(field: &FieldDecl, delete: &FieldDelete) -> Check {
    let enabled = delete.readers.is_empty();
    Check {
        caption: "Also delete its files".to_string(),
        on: false,
        enabled,
        note: (!enabled).then(|| {
            format!(
                "{} is still read by {}",
                file_name(&field.declared),
                quoted(&delete.readers)
            )
        }),
    }
}

// The toast after a delete: the volumes, the file when it went, and why it
// could not.
pub(crate) fn deleted_message(
    field: &FieldDecl,
    removed: &[String],
    file: Option<Result<(), String>>,
) -> String {
    let mut out = format!(
        "Deleted the SDF field {} ({})",
        field.declared,
        quoted(removed)
    );
    match file {
        Some(Ok(())) => out.push_str(&format!(
            " and {}; undo restores the volumes but not the file",
            file_name(&field.declared)
        )),
        Some(Err(e)) => out.push_str(&format!("; could not delete {}: {e}", field.path)),
        None => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::panels::sdf_field_list::FieldVolume;
    use crate::editor::panels::shader_list::declared;
    use serde_json::json;

    fn field(names: &[&str]) -> FieldDecl {
        FieldDecl {
            declared: "shaders/blob.hlsl".to_string(),
            path: "/cn-none/assets/shaders/blob.hlsl".to_string(),
            volumes: names
                .iter()
                .map(|n| FieldVolume {
                    name: n.to_string(),
                    volumetric: false,
                })
                .collect(),
        }
    }

    // Every volume the editor can remove goes; one included from another file
    // stays, and with it the file, so the box is disabled saying who reads it.
    #[test]
    fn an_included_volume_stays_and_keeps_the_file() {
        let blob = field(&["a", "b", "inc"]);
        let delete = plan(&blob, |n| n == "inc", &[]);
        assert_eq!(delete.removed, ["a", "b"]);
        assert_eq!(delete.readers, ["inc"]);
        let check = files_check(&blob, &delete);
        assert!(!check.enabled && !check.on);
        assert_eq!(
            check.note.as_deref(),
            Some("blob.hlsl is still read by 'inc'")
        );
        assert_eq!(
            delete_message(&blob, &delete),
            "Delete the SDF field shaders/blob.hlsl? Its volumes 'a' and 'b' are deleted."
        );
    }

    // With nothing left reading the file, the box is open, and off.
    #[test]
    fn a_file_only_its_volumes_read_can_go_with_them() {
        let blob = field(&["a"]);
        let delete = plan(&blob, |_| false, &[]);
        let check = files_check(&blob, &delete);
        assert!(check.enabled && !check.on && check.note.is_none());
        assert_eq!(
            delete_message(&blob, &delete),
            "Delete the SDF field shaders/blob.hlsl? Its volume 'a' is deleted."
        );
    }

    #[test]
    fn a_shader_declaring_the_file_keeps_it() {
        let blob = field(&["a"]);
        let shaders = declared(
            &[json!({"type": "Shader", "args": {"$id": "lit", "fragment": "x"}})],
            |_| blob.path.clone(),
        );
        let delete = plan(&blob, |_| false, &shaders);
        assert_eq!(delete.readers, ["lit"]);
        assert!(!files_check(&blob, &delete).enabled);
    }

    #[test]
    fn the_toast_says_the_file_is_not_undone() {
        let blob = field(&["a", "b"]);
        let removed = ["a".to_string(), "b".to_string()];
        assert_eq!(
            deleted_message(&blob, &removed, None),
            "Deleted the SDF field shaders/blob.hlsl ('a' and 'b')"
        );
        let gone = deleted_message(&blob, &removed, Some(Ok(())));
        assert!(gone.ends_with("not the file"), "{gone}");
        let failed = deleted_message(&blob, &removed, Some(Err("denied".to_string())));
        assert!(failed.contains("could not delete"), "{failed}");
    }
}
