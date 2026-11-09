use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use concinnity_core::platform::Platform;

use super::*;
use crate::render::shader::consts;
use crate::render::shader::programs::{Table, dx, metal as metal_programs, vk};

const TABLES: [(Platform, &Table); 3] = [
    (Platform::DirectX, &dx::TABLE),
    (Platform::Vulkan, &vk::TABLE),
    (Platform::Metal, &metal_programs::TABLE),
];

// The `[numthreads(...)]` sizes `source` declares on entry point `entry`, one
// per declaration, each axis read as a literal or a named constant.
fn numthreads(source: &str, entry: &str) -> Vec<[u32; 3]> {
    const MARKER: &str = "[numthreads(";
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(start) = source[at..].find(MARKER) {
        let open = at + start + MARKER.len();
        let close = open + source[open..].find(')').expect("numthreads closes");
        at = close;
        if entry_after(source, close) != Some(entry) {
            continue;
        }
        let axes: Vec<u32> = source[open..close]
            .split(',')
            .map(|axis| {
                let axis = axis.trim();
                axis.parse().unwrap_or_else(|_| {
                    u32::try_from(consts::uint(source, axis)).expect("a group axis fits a u32")
                })
            })
            .collect();
        found.push(<[u32; 3]>::try_from(axes).expect("numthreads names three axes"));
    }
    found
}

// The function named after the attribute ending at `at`: the identifier
// before the first parameter list, past any further bracketed attributes.
fn entry_after(source: &str, at: usize) -> Option<&str> {
    let rest = &source[source[at..].find(']')? + at + 1..];
    let mut depth = 0usize;
    let mut name: Option<&str> = None;
    let mut ident = None;
    for (i, c) in rest.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            _ if depth > 0 => {}
            '(' => return name.or_else(|| ident.map(|s: usize| &rest[s..i])),
            '{' | '}' | ';' => return None,
            c if c.is_ascii_alphanumeric() || c == '_' => {
                ident.get_or_insert(i);
                name = None;
            }
            _ => {
                if let Some(s) = ident.take() {
                    name = Some(&rest[s..i]);
                }
            }
        }
    }
    None
}

#[test]
fn every_group_matches_the_numthreads_its_entry_declares() {
    for (platform, _) in TABLES {
        for row in KERNEL_GROUPS {
            let program = row.program;
            let source = program.at(false).assemble(platform);
            let declared = numthreads(&source, program.entry);
            assert!(
                !declared.is_empty(),
                "{platform:?} {}: `{}` declares no [numthreads]",
                program.label,
                program.entry
            );
            for size in declared {
                assert_eq!(
                    size, row.group.size,
                    "{platform:?} {}: `{}` declares [numthreads({size:?})], not the {:?} its \
                     host dispatches at",
                    program.label, program.entry, row.group.size
                );
            }
        }
    }
}

// A compute program left off the table would have its host size its dispatch
// from a number nothing checks.
#[test]
fn every_compute_program_has_one_group() {
    let listed: BTreeSet<&str> = KERNEL_GROUPS.iter().map(|r| r.program.label).collect();
    assert_eq!(
        listed.len(),
        KERNEL_GROUPS.len(),
        "a program is listed twice"
    );
    for (platform, table) in TABLES {
        for program in table.programs() {
            let source = program.at(false).assemble(platform);
            if !numthreads(&source, program.entry).is_empty() {
                assert!(
                    listed.contains(program.label),
                    "{platform:?}: compute program {} has no row in KERNEL_GROUPS",
                    program.label
                );
            }
        }
    }
}

#[test]
fn numthreads_reads_literals_and_named_constants() {
    let source = "\
static const uint WIDE = 16u;
[shader(\"compute\")]
[numthreads(WIDE, 4, 1)]
void a_kernel(uint3 id : SV_DispatchThreadID) {}
[numthreads(8, 8, 1)]
[shader(\"compute\")]
void b_kernel(uint3 id : SV_DispatchThreadID) {}
";
    assert_eq!(numthreads(source, "a_kernel"), [[16, 4, 1]]);
    assert_eq!(numthreads(source, "b_kernel"), [[8, 8, 1]]);
    assert!(numthreads(source, "c_kernel").is_empty());
}

#[test]
fn groups_round_each_axis_up() {
    let group = ThreadGroup::new(8, 8, 1);
    assert_eq!(group.groups([16, 17, 1]), [2, 3, 1]);
    assert_eq!(group.groups_2d([1, 64]), [1, 8, 1]);
    assert_eq!(ThreadGroup::new(64, 1, 1).groups_1d(65), [2, 1, 1]);
    assert_eq!(group.groups_1d(0), [0, 1, 1]);
}
