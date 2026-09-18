// In-memory world authoring + build orchestration.
//
// The add / rm / check / build operations on a world JSONL, plus the templates
// bridge, shared by this crate's `command` layer and the editor.

mod add;
mod build;
mod check;
mod rm;
mod template_spec;

pub(crate) use add::add_to_path;
pub(crate) use build::{
    build_world_and_shadows, build_world_file, build_world_from_path, build_world_from_str,
    build_world_str_to_disk,
};
pub(crate) use check::{check_at_path, report_validation_errors};
pub(crate) use rm::rm_at_path;
pub(crate) use template_spec::{spec_args, world_template_entries};
// The path-to-entries resolution and the extensions it handles, shared with
// the editor's Import panel (its Add path and its file picker's filters) and
// the editor console's /add (the full target resolution).
pub(crate) use add::{
    IMPORT_EXTENSION_GROUPS, apply_id_override, entry_from_path, is_path_like, resolve_add_target,
    try_retarget_environment_map,
};
