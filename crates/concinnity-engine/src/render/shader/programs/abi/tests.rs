use alloc::collections::BTreeSet;

use super::*;

// A parameter a row names twice binds two ways, and only one of them can match
// the declaration the build reads.
#[test]
fn every_row_names_each_parameter_once() {
    for abi in DXIL_ENTRY_ABI {
        let mut seen = BTreeSet::new();
        for (param, _) in abi.registers {
            assert!(
                seen.insert(*param),
                "{}: `{param}` listed twice",
                abi.program.entry
            );
        }
    }
    for abi in METAL_ENTRY_ABI {
        let mut seen = BTreeSet::new();
        for (param, _) in abi.slots() {
            assert!(
                seen.insert(*param),
                "{}: `{param}` listed twice",
                abi.program.entry
            );
        }
    }
}

// The spellings the build compares against the compiled declarations: a DirectX
// register class and index with an optional space, and a Metal attribute.
#[test]
fn every_binding_is_spelled_as_the_build_reads_it() {
    for abi in DXIL_ENTRY_ABI {
        for (param, register) in abi.registers {
            let (slot, space) = register.split_once(", space").unwrap_or((register, "0"));
            let (class, index) = slot.split_at(1);
            assert!(
                matches!(class, "b" | "t" | "u" | "s")
                    && index.parse::<u32>().is_ok()
                    && space.parse::<u32>().is_ok(),
                "{}: `{param}` has register `{register}`",
                abi.program.entry
            );
        }
    }
    for abi in METAL_ENTRY_ABI {
        for (param, attribute) in abi.slots() {
            let index = ["buffer(", "texture(", "sampler("]
                .iter()
                .find_map(|kind| attribute.strip_prefix(kind))
                .and_then(|rest| rest.strip_suffix(')'));
            assert!(
                index.is_some_and(|i| i.parse::<u32>().is_ok()),
                "{}: `{param}` has attribute `{attribute}`",
                abi.program.entry
            );
        }
    }
}
