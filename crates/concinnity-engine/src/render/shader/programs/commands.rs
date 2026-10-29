//! The programs that write indirect draw commands as data, which Vulkan and
//! DirectX compile and Metal does not: its indirect command buffers are
//! encoded by the hand-written `cull_encode.metal`.

use super::ShaderProgram;

/// `cull_scatter` from `cull_compact.hlsl` under `CULL_SCATTER`: every kept
/// record's command into its stream's list.
pub static CULL_SCATTER: ShaderProgram = ShaderProgram {
    file: "cull_compact.hlsl",
    entry: "cull_scatter",
    label: "cull_scatter.hlsl",
    gates: &["CULL_SCATTER"],
    msaa: false,
};

/// Every program in this module.
pub static ALL: &[&ShaderProgram] = &[&CULL_SCATTER];

#[cfg(test)]
mod tests {
    #[test]
    fn every_declared_program_is_in_the_table() {
        super::super::declared::assert_table_is_complete(include_str!("commands.rs"));
    }
}
