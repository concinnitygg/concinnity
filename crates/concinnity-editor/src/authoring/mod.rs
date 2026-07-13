// In-memory world authoring + build orchestration.
//
// The add / rm / check / build operations on a world JSONL, plus the templates
// bridge, shared by the `concinnity` CLI (concinnity-cli) and the C-ABI
// embedding surface (concinnity-ffi). This is the compile pipeline driven as a
// library: a shipped runtime links none of it.

mod add;
mod build;
mod check;
mod rm;
mod template_spec;

pub use add::add_to_path;
pub use build::{
    build_world_from_path, build_world_from_str, build_world_to_disk, world_from_loaded,
};
pub use check::{check_at_path, check_from_str};
pub use rm::rm_at_path;
pub use template_spec::{arg_value_to_json, spec_args, spec_to_value, world_template_entries};
