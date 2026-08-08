// In-memory world authoring + build orchestration.
//
// The add / rm / check / build operations on a world JSONL, plus the templates
// bridge, shared by the `concinnity` CLI (concinnity-cli).

mod add;
mod build;
mod check;
pub(crate) mod name_table;
pub(crate) mod reload_sources;
mod rm;
mod template_spec;

pub use add::add_to_path;
// The path-to-entries resolution and the extensions it handles, shared with
// the editor's Import panel (its Add path and its file picker's filters) and
// the editor console's /add (the full target resolution).
pub(crate) use add::{
    IMPORT_EXTENSION_GROUPS, apply_name_override, entry_from_path, is_path_like,
    resolve_add_target, try_retarget_environment_map,
};
pub(crate) use build::build_world_str_to_disk_with_progress;
pub use build::{
    build_world_from_path, build_world_from_str, build_world_to_disk, world_from_loaded,
};
pub use check::{check_at_path, check_from_str};
pub use rm::rm_at_path;
pub use template_spec::{arg_value_to_json, spec_args, spec_to_value, world_template_entries};
