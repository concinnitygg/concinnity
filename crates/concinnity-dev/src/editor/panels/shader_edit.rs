//! The data half of the Shaders panel's edits: whether a typed name can name
//! a Shader, the file name a Shader's name becomes, what a delete says it will
//! do, and which of a Shader's files no other Shader reads.

use concinnity_cook::authoring::world::is_label_of;

use super::shader_list::ShaderDecl;
use super::shader_source::{SourceKey, same_file, stage_name};
use crate::editor::widget_check::Check;
use concinnity_core::components::ShaderStage;

// An edit that takes a file away from the Shader source panel, carried through
// the unsaved-changes question when that file is open there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShaderEdit {
    // Stop declaring the Shader's vertex file (the file stays on disk).
    RemoveVertex(String),
    // Remove the Shader, and with `files` the files only it reads.
    Delete { name: String, files: bool },
}

impl ShaderEdit {
    // Whether the edit takes away the file `open` shows.
    pub(crate) fn closes(&self, open: &SourceKey) -> bool {
        match self {
            ShaderEdit::RemoveVertex(name) => {
                open.shader == *name && open.stage == ShaderStage::Vertex
            }
            ShaderEdit::Delete { name, .. } => open.shader == *name,
        }
    }
}

// The name typed for a Shader, trimmed, or why it cannot be one: a
// Material names its Shader, so the name cannot be blank, and a
// `<Type>#<n>` label is reserved for an unnamed entry.
pub(crate) fn check_name(typed: &str) -> Result<String, String> {
    let name = typed.trim();
    if name.is_empty() {
        return Err("Enter a name for the Shader.".to_string());
    }
    if name
        .split_once('#')
        .is_some_and(|(ty, _)| is_label_of(name, ty))
    {
        return Err(format!(
            "'{name}' is reserved for an unnamed asset; choose another name."
        ));
    }
    Ok(name.to_string())
}

// The file name a Shader named `name` gives its files: every character a path
// could misread becomes `_`.
pub(crate) fn file_stem(name: &str) -> String {
    let stem: String = name
        .chars()
        .map(
            |c| match c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                true => c,
                false => '_',
            },
        )
        .collect();
    match stem.is_empty() {
        true => "shader".to_string(),
        false => stem,
    }
}

// The name of the vertex file added to a Shader named `name`.
pub(crate) fn vertex_stem(name: &str) -> String {
    format!("{}_vertex", file_stem(name))
}

// What deleting Shader `i` would do to its files.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Files {
    // On-disk paths of the files no other Shader reads.
    pub(crate) own: Vec<String>,
    // Each file another Shader also reads, as declared, with that Shader's
    // name.
    pub(crate) shared: Vec<(String, String)>,
}

pub(crate) fn files_of(shaders: &[ShaderDecl], i: usize) -> Files {
    let Some(shader) = shaders.get(i) else {
        return Files::default();
    };
    let mut out = Files::default();
    for file in &shader.files {
        let reader = shaders
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .find(|(_, s)| s.files.iter().any(|f| same_file(&f.path, &file.path)));
        match reader {
            Some((_, other)) => out.shared.push((file.declared.clone(), other.name.clone())),
            None if !out.own.iter().any(|p| same_file(p, &file.path)) => {
                out.own.push(file.path.clone())
            }
            None => {}
        }
    }
    out
}

// What the delete dialog says about Shader `i`: the Materials that fall back,
// and which Shader becomes the world default when this one was it.
pub(crate) fn delete_message(shaders: &[ShaderDecl], i: usize) -> String {
    let Some(shader) = shaders.get(i) else {
        return String::new();
    };
    let mut out = format!("Delete Shader '{}'?", shader.name);
    let remaining = shaders.len() > 1;
    let fallback = match remaining {
        true => "the default Shader",
        false => "the engine's own shading",
    };
    match shader.materials.len() {
        0 => {}
        1 => out.push_str(&format!(" 1 Material falls back to {fallback}.")),
        n => out.push_str(&format!(" {n} Materials fall back to {fallback}.")),
    }
    if shader.default
        && let Some(next) = shaders.iter().enumerate().find(|&(j, _)| j != i)
    {
        out.push_str(&format!(" '{}' becomes the world default.", next.1.name));
    }
    out
}

// The dialog's "Also delete its files" box, off. It is disabled when every
// file is also another Shader's, and its note names what a delete keeps.
pub(crate) fn files_check(files: &Files) -> Check {
    let shared: Vec<String> = files
        .shared
        .iter()
        .map(|(declared, other)| format!("{} is also read by '{other}'", file_name(declared)))
        .collect();
    let enabled = !files.own.is_empty();
    let note = match (shared.is_empty(), enabled) {
        (true, _) => None,
        (false, true) => Some(format!("Keeps what others read: {}", shared.join("; "))),
        (false, false) => Some(shared.join("; ")),
    };
    Check {
        caption: "Also delete its files".to_string(),
        on: false,
        enabled,
        note,
    }
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

// The toast after a delete: the Shader, the files deleted, and those that
// could not be.
pub(crate) fn deleted_message(name: &str, deleted: &[String], failed: &[String]) -> String {
    let mut out = format!("Deleted Shader '{name}'");
    if !deleted.is_empty() {
        let names: Vec<&str> = deleted.iter().map(|p| file_name(p)).collect();
        out.push_str(&format!(
            " and {}; undo restores the Shader but not its files",
            names.join(", ")
        ));
    }
    if !failed.is_empty() {
        out.push_str(&format!("; could not delete {}", failed.join("; ")));
    }
    out
}

// The toast after removing a Shader's vertex file.
pub(crate) fn removed_vertex_message(name: &str, declared: &str) -> String {
    format!(
        "Removed the {} file from '{name}'; {declared} stays on disk",
        stage_name(ShaderStage::Vertex)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::panels::shader_list::declared;
    use serde_json::json;

    fn shader(name: &str, fragment: &str) -> serde_json::Value {
        json!({"type": "Shader", "args": {"$id": name, "fragment": fragment}})
    }

    fn material(name: &str, shader: &str) -> serde_json::Value {
        json!({"type": "Material", "args": {"$id": name, "shader": shader}})
    }

    fn decls(entries: &[serde_json::Value]) -> Vec<ShaderDecl> {
        declared(entries, str::to_string)
    }

    #[test]
    fn a_name_is_trimmed_and_cannot_be_blank_or_a_label() {
        assert_eq!(check_name("  water "), Ok("water".to_string()));
        assert!(check_name("   ").is_err());
        assert!(check_name("Shader#2").is_err());
        assert!(
            check_name("water#2").is_err(),
            "any label shape is reserved"
        );
        assert_eq!(check_name("water#a"), Ok("water#a".to_string()));
    }

    #[test]
    fn a_file_stem_keeps_only_path_safe_characters() {
        assert_eq!(file_stem("water"), "water");
        assert_eq!(file_stem("../sea/deep"), "___sea_deep");
        assert_eq!(file_stem("Shader#0"), "Shader_0");
        assert_eq!(file_stem(""), "shader");
        assert_eq!(vertex_stem("reeds"), "reeds_vertex");
    }

    // The message counts the Materials that fall back, and names the next
    // Shader when the default goes.
    #[test]
    fn the_delete_message_says_what_falls_back() {
        let entries = [
            shader("lit", "/cn-none/lit.hlsl"),
            shader("reeds", "/cn-none/reeds.hlsl"),
            material("a", "reeds"),
            material("b", "reeds"),
            material("c", "lit"),
        ];
        let shaders = decls(&entries);
        let reeds = delete_message(&shaders, 1);
        assert!(
            reeds.contains("2 Materials fall back to the default Shader"),
            "{reeds}"
        );
        assert!(!reeds.contains("world default"), "{reeds}");
        let lit = delete_message(&shaders, 0);
        assert!(
            lit.contains("1 Material falls back to the default Shader"),
            "{lit}"
        );
        assert!(lit.contains("'reeds' becomes the world default"), "{lit}");

        let only = decls(&[shader("lit", "/cn-none/lit.hlsl")]);
        let message = delete_message(&only, 0);
        assert_eq!(message, "Delete Shader 'lit'?");
        let alone = decls(&[shader("lit", "/cn-none/lit.hlsl"), material("m", "lit")]);
        assert!(delete_message(&alone, 0).contains("falls back to the engine's own shading"));
    }

    // A file another Shader reads is never the delete's; with nothing left of
    // its own the box is disabled, saying why.
    #[test]
    fn the_file_box_is_disabled_for_a_shared_file() {
        let entries = [
            shader("lit", "/cn-none/common.hlsl"),
            shader("reeds", "/cn-none/common.hlsl"),
        ];
        let shaders = decls(&entries);
        let files = files_of(&shaders, 1);
        assert!(files.own.is_empty());
        assert_eq!(
            files.shared,
            [("/cn-none/common.hlsl".to_string(), "lit".to_string())]
        );
        let check = files_check(&files);
        assert!(!check.enabled && !check.on);
        assert_eq!(
            check.note.as_deref(),
            Some("common.hlsl is also read by 'lit'")
        );

        let mut entries = entries.to_vec();
        entries[1]["args"]["vertex"] = json!("/cn-none/sway.hlsl");
        let files = files_of(&decls(&entries), 1);
        assert_eq!(files.own, ["/cn-none/sway.hlsl"]);
        let check = files_check(&files);
        assert!(check.enabled && !check.on, "off by default");
        assert!(check.note.unwrap().starts_with("Keeps"));

        let solo = decls(&[shader("lit", "/cn-none/lit.hlsl")]);
        let check = files_check(&files_of(&solo, 0));
        assert!(check.enabled && check.note.is_none());
    }

    #[test]
    fn an_edit_closes_only_the_files_it_takes_away() {
        let open = |stage| SourceKey {
            shader: "reeds".to_string(),
            stage,
        };
        let remove = ShaderEdit::RemoveVertex("reeds".to_string());
        assert!(remove.closes(&open(ShaderStage::Vertex)));
        assert!(!remove.closes(&open(ShaderStage::Fragment)));
        let delete = ShaderEdit::Delete {
            name: "reeds".to_string(),
            files: false,
        };
        assert!(delete.closes(&open(ShaderStage::Fragment)));
        let other = ShaderEdit::Delete {
            name: "lit".to_string(),
            files: false,
        };
        assert!(!other.closes(&open(ShaderStage::Fragment)));
    }

    #[test]
    fn the_delete_toast_says_file_deletion_is_not_undone() {
        assert_eq!(deleted_message("lit", &[], &[]), "Deleted Shader 'lit'");
        let with_files = deleted_message("lit", &["/p/shaders/lit.hlsl".to_string()], &[]);
        assert!(with_files.contains("lit.hlsl"), "{with_files}");
        assert!(with_files.contains("not its files"), "{with_files}");
    }
}
