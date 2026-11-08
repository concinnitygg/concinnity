use alloc::collections::BTreeSet;
use alloc::string::ToString;

use super::register::{DxBinding, srv};
use super::slot::buffer;
use super::*;

// A parameter a row names twice binds two ways, and only one of them can match
// the declaration the build reads.
#[test]
fn every_row_names_each_parameter_once() {
    for abi in DXIL_ENTRY_ABI {
        let mut seen = BTreeSet::new();
        for binding in abi.registers {
            let name = binding.name;
            assert!(
                seen.insert(name),
                "{}: `{name}` listed twice",
                abi.program.entry
            );
        }
    }
    for abi in METAL_ENTRY_ABI {
        let mut seen = BTreeSet::new();
        for binding in abi.slots() {
            let name = binding.name;
            assert!(
                seen.insert(name),
                "{}: `{name}` listed twice",
                abi.program.entry
            );
        }
    }
}

// The spellings the build compares against the compiled declarations.
#[test]
fn bindings_spell_as_the_shaders_declare_them() {
    let pool: DxBinding = srv("tex_pool", 0).in_space(1);
    assert_eq!(srv("objects", 3).to_string(), "t3");
    assert_eq!(pool.to_string(), "t0, space1");
    assert_eq!(buffer("objects", 9).to_string(), "buffer(9)");
}

// The scan's list lengths and the scatter's commands ride one root parameter.
#[test]
fn the_scan_and_scatter_share_their_last_register() {
    use super::cull::dx::compact::{COMMANDS, STREAM_COUNTS};
    assert_eq!(
        STREAM_COUNTS.binding().to_string(),
        COMMANDS.binding().to_string()
    );
}

// The root signatures bind these pairs as one two-register range, so each
// second register must follow its first.
#[test]
fn ranged_main_registers_are_adjacent() {
    use super::main::dx::*;
    assert_eq!(PREFILTER_CUBE.index, IRRADIANCE_CUBE.index + 1);
    assert_eq!(LTC_MAGNITUDE.index, LTC_MATRIX.index + 1);
    assert_eq!(CUBE_SAMPLER.index, LINEAR_SAMPLER.index + 1);
    assert_eq!(TEX_POOL.index, 0, "the bindless range starts its space");
}
