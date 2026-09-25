//! The data half of the Shader source panel: which of a Shader's files it
//! edits and where that file is on disk, the starter file "+ New Shader"
//! writes, and the rules for leaving a file with unsaved edits and for a file
//! that changed on disk while open. What a reload outcome shows is
//! `shader_diagnostics`.

use concinnity_core::components::ShaderStage;
use std::path::{Path, PathBuf};

use crate::editor::modal;

// One file of one Shader: the Shader's name and which of its files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceKey {
    pub(crate) shader: String,
    pub(crate) stage: ShaderStage,
}

pub(crate) fn stage_name(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
    }
}

// The on-disk path of a declared Shader file. This is the resolution the
// renderer's hot-reload catalog applies (`ShaderFile::resolved_path`), and a
// recompile's diagnostics name a Shader's files by that path, so the panel
// matches them against what this returns.
pub(crate) fn resolve_path(declared: &str, assets_dir: Option<&Path>) -> String {
    concinnity_host::store::source::resolve_source_path(declared, assets_dir)
}

// Whether two spellings name one file: the same text, or the same file once
// both are resolved on disk.
pub(crate) fn same_file(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (Path::new(a).canonicalize(), Path::new(b).canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

// The fragment file a "+ New Shader" writes: the engine's own lighting,
// returned as-is, ready to adjust. Pinned compilable by test.
pub(crate) const STARTER_SHADER: &str = "\
// The surface's color. shade_surface is the engine's own lighting.
float4 shade(VertexOut v, GpuObjectData od)
{
    return shade_surface(v, od);
}
";

// Where a new Shader named `name` keeps its fragment file: `shaders/<name>.hlsl`
// under `dir`, numbered past any file already there.
pub(crate) fn starter_path(dir: &Path, name: &str, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let shaders = dir.join("shaders");
    let first = shaders.join(format!("{name}.hlsl"));
    if !exists(&first) {
        return first;
    }
    (1..)
        .map(|i| shaders.join(format!("{name}_{i}.hlsl")))
        .find(|p| !exists(p))
        .expect("an unbounded search finds a free name")
}

// How a world line declares `path`: relative to `root` when it is under it,
// with forward slashes so the line reads the same on every host.
pub(crate) fn declared_form(path: &Path, root: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(relative) => relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

// Where the panel goes next when it leaves the open file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Leave {
    Open(SourceKey),
    Close,
}

// Whether leaving the open file `open` for `then` has to ask first: only with
// unsaved edits, and never to reopen the file already open.
pub(crate) fn must_ask(dirty: bool, open: &SourceKey, then: &Leave) -> bool {
    dirty
        && match then {
            Leave::Open(key) => key != open,
            Leave::Close => true,
        }
}

// The question, and its buttons left to right: Cancel stays, Discard leaves
// without writing, Save writes and then leaves.
pub(crate) fn leave_prompt(file: &str) -> String {
    format!("{file} has unsaved changes.")
}

pub(crate) fn leave_buttons(then: Leave) -> Vec<modal::Button> {
    vec![
        modal::Button {
            label: "Cancel".to_string(),
            danger: false,
            action: modal::Action::Dismiss,
        },
        modal::Button {
            label: "Discard".to_string(),
            danger: true,
            action: modal::Action::LeaveShaderSource {
                save: false,
                then: then.clone(),
            },
        },
        modal::Button {
            label: "Save".to_string(),
            danger: false,
            action: modal::Action::LeaveShaderSource { save: true, then },
        },
    ]
}

// What a check of the open file on disk calls for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiskChange {
    // The file holds what the buffer was loaded from (or saved as).
    Unchanged,
    // It changed and the buffer has no edits: read it back in.
    Reload,
    // It changed under unsaved edits: say so once, and keep the buffer.
    Notice,
}

// `known` is the text the buffer was loaded from or last saved as, `disk` what
// the file holds now, and `noticed` the disk text a notice was already shown
// for.
pub(crate) fn disk_change(
    known: &str,
    disk: &str,
    dirty: bool,
    noticed: Option<&str>,
) -> DiskChange {
    if disk == known {
        DiskChange::Unchanged
    } else if !dirty {
        DiskChange::Reload
    } else if noticed == Some(disk) {
        DiskChange::Unchanged
    } else {
        DiskChange::Notice
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(shader: &str, stage: ShaderStage) -> SourceKey {
        SourceKey {
            shader: shader.to_string(),
            stage,
        }
    }

    // Leaving asks only over unsaved edits, and reopening the open file is not
    // leaving it.
    #[test]
    fn leaving_asks_only_over_unsaved_edits() {
        let open = key("water", ShaderStage::Fragment);
        let other = Leave::Open(key("water", ShaderStage::Vertex));
        assert!(must_ask(true, &open, &other));
        assert!(must_ask(true, &open, &Leave::Close));
        assert!(!must_ask(false, &open, &other));
        assert!(!must_ask(false, &open, &Leave::Close));
        assert!(!must_ask(true, &open, &Leave::Open(open.clone())));
    }

    // Cancel stays put; Discard and Save both carry where to go next, and only
    // Save writes first.
    #[test]
    fn the_leave_buttons_cancel_discard_or_save() {
        let then = Leave::Open(key("lit", ShaderStage::Fragment));
        let buttons = leave_buttons(then.clone());
        let labels: Vec<&str> = buttons.iter().map(|b| b.label.as_str()).collect();
        assert_eq!(labels, ["Cancel", "Discard", "Save"]);
        assert_eq!(buttons[0].action, modal::Action::Dismiss);
        assert_eq!(
            buttons[1].action,
            modal::Action::LeaveShaderSource {
                save: false,
                then: then.clone()
            }
        );
        assert!(buttons[1].danger);
        assert_eq!(
            buttons[2].action,
            modal::Action::LeaveShaderSource { save: true, then }
        );
    }

    // A clean buffer follows the file; a dirty one is never overwritten, and a
    // change is announced once.
    #[test]
    fn a_change_on_disk_reloads_a_clean_buffer_and_notices_a_dirty_one() {
        assert_eq!(disk_change("a", "a", true, None), DiskChange::Unchanged);
        assert_eq!(disk_change("a", "b", false, None), DiskChange::Reload);
        assert_eq!(disk_change("a", "b", true, None), DiskChange::Notice);
        assert_eq!(
            disk_change("a", "b", true, Some("b")),
            DiskChange::Unchanged
        );
        assert_eq!(disk_change("a", "c", true, Some("b")), DiskChange::Notice);
        assert_eq!(disk_change("a", "b", false, Some("b")), DiskChange::Reload);
    }

    #[test]
    fn the_starter_path_numbers_past_existing_files() {
        let dir = Path::new("/cn-none/assets");
        let taken = [
            dir.join("shaders/editor_shader.hlsl"),
            dir.join("shaders/editor_shader_1.hlsl"),
        ];
        let exists = |p: &Path| taken.iter().any(|t| t == p);
        assert_eq!(
            starter_path(dir, "editor_shader", exists),
            dir.join("shaders/editor_shader_2.hlsl")
        );
        assert_eq!(
            starter_path(dir, "fresh", exists),
            dir.join("shaders/fresh.hlsl")
        );
    }

    #[test]
    fn a_declared_path_is_relative_to_the_root_when_under_it() {
        let root = Path::new("/cn-none/project");
        assert_eq!(
            declared_form(&root.join("assets").join("shaders").join("a.hlsl"), root),
            "assets/shaders/a.hlsl"
        );
        assert_eq!(
            declared_form(Path::new("/cn-none/elsewhere/a.hlsl"), root),
            "/cn-none/elsewhere/a.hlsl"
        );
    }

    // A path with a directory resolves as written; a bare name is looked up
    // under the assets directory, as the renderer's catalog resolves it.
    #[test]
    fn paths_resolve_as_the_renderer_resolves_them() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("shaders");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("lit.hlsl"), "").unwrap();
        assert_eq!(
            resolve_path("lit.hlsl", Some(dir.path())),
            nested.join("lit.hlsl").to_string_lossy()
        );
        assert_eq!(resolve_path("a/b.hlsl", Some(dir.path())), "a/b.hlsl");
        assert!(same_file(
            &resolve_path("lit.hlsl", Some(dir.path())),
            &nested.join(".").join("lit.hlsl").to_string_lossy()
        ));
        assert!(!same_file("/cn-none/a.hlsl", "/cn-none/b.hlsl"));
    }

    // The starter must compile as a real Shader; a template change breaks this
    // test instead of shipping a starter that fails its first save.
    #[test]
    fn the_starter_shader_compiles() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let sources = concinnity_core::render::shader_programs::surface::Sources {
            vertex: None,
            fragment: concinnity_core::render::shader_programs::surface::SourceFile {
                path: "shaders/starter.hlsl",
                text: STARTER_SHADER,
            },
        };
        let compiled = concinnity_cook::compile::shader::compile_world_shader(
            "starter",
            &sources,
            crate::cook_platform(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(compiled.warnings.is_empty(), "{:?}", compiled.warnings);
    }
}
