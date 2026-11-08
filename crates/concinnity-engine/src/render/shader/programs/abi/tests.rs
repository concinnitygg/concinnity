use alloc::collections::BTreeSet;
use alloc::string::ToString;

use super::register::{DxBinding, Register, Srv};
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
    let pool: DxBinding = Register::<Srv>::new("tex_pool", 0).in_space(1).binding();
    assert_eq!(
        Register::<Srv>::new("objects", 3).binding().to_string(),
        "t3"
    );
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

// The volume's root signature binds its textures and its samplers as one
// range each, so each must follow the one before.
#[test]
fn ranged_raymarch_registers_are_adjacent() {
    use super::raymarch::dx::*;
    assert_eq!(IRRADIANCE_CUBE.index, SHADOW_MAP.index + 1);
    assert_eq!(PREFILTER_CUBE.index, SHADOW_MAP.index + 2);
    assert_eq!(SCENE_COLOR.index, SHADOW_MAP.index + 3);
    assert_eq!(CUBE_SAMPLER.index, SHADOW_SAMPLER.index + 1);
    assert_eq!(SCENE_SAMPLER.index, SHADOW_SAMPLER.index + 2);
}

// The transparent root signatures bind these runs as one range each.
#[test]
fn ranged_transparent_registers_are_adjacent() {
    use super::transparent::dx::*;
    assert_eq!(IRRADIANCE_CUBE.index, SHADOW_MAP.index + 1);
    assert_eq!(OCEAN_SLOPES.index, OCEAN_DISPLACEMENT.index + 1);
    assert_eq!(WAKE_FIELD.index, OCEAN_DISPLACEMENT.index + 2);
    assert_eq!(GLASS_REFLECTION_BACK.index, GLASS_REFLECTION.index + 1);
}
